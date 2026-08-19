import { TRIBE_DESCRIPTOR_SCHEMA_VERSION } from "./descriptors.js";

export const TRIBE_PLAIN_LANGUAGE_SCHEMA_VERSION = "tribe-creator-plain-language/1";

const PHASES = Object.freeze([
  Object.freeze({ id: "early", label: "opening" }),
  Object.freeze({ id: "middle", label: "middle" }),
  Object.freeze({ id: "late", label: "ending" }),
]);

function fail(message) {
  throw new Error(`Invalid TRIBE plain-language report: ${message}`);
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function finite(value, label) {
  if (!Number.isFinite(value)) fail(`${label} must be finite.`);
  return value;
}

function almostEqual(left, right) {
  return Math.abs(left - right) <= Number.EPSILON * Math.max(16, Math.abs(left), Math.abs(right));
}

function deepFreeze(value) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  for (const child of Object.values(value)) deepFreeze(child);
  return Object.freeze(value);
}

function validateMoment(moment, label, frames, field) {
  if (moment === null && field === "changeRms" && frames.length === 1) return null;
  if (!isRecord(moment) || !Number.isInteger(moment.index) || !frames[moment.index]) {
    fail(`${label} must reference an existing prediction interval.`);
  }
  const value = finite(moment.value, `${label}.value`);
  if (value < 0 || !almostEqual(value, frames[moment.index][field])) {
    fail(`${label} must match its referenced interval.`);
  }
  return moment;
}

function validateReport(report) {
  if (!isRecord(report) || report.schemaVersion !== TRIBE_DESCRIPTOR_SCHEMA_VERSION) {
    fail(`schemaVersion must be ${TRIBE_DESCRIPTOR_SCHEMA_VERSION}.`);
  }
  if (!isRecord(report.source)
      || report.source.descriptorType !== "descriptive cortical response"
      || report.source.predictionType !== "average-subject cortical BOLD response"
      || report.source.isViralityScore !== false
      || report.source.behavioralPrediction !== false
      || report.source.model?.id !== "facebook/tribev2") {
    fail("the source must be a non-behavioral TRIBE v2 cortical descriptor result.");
  }
  if (!Array.isArray(report.frames) || report.frames.length < 1 || !isRecord(report.clip)) {
    fail("prediction intervals and clip descriptors are required.");
  }

  let previousEnd = -Infinity;
  const frames = report.frames.map((frame, index) => {
    if (!isRecord(frame) || frame.index !== index) fail(`frames[${index}] is out of tensor order.`);
    const time = finite(frame.time, `frames[${index}].time`);
    const duration = finite(frame.duration, `frames[${index}].duration`);
    const endTime = finite(frame.endTime, `frames[${index}].endTime`);
    const rms = finite(frame.rms, `frames[${index}].rms`);
    if (duration <= 0 || rms < 0 || time < previousEnd || !almostEqual(endTime, time + duration)) {
      fail(`frames[${index}] has invalid timing or magnitude.`);
    }
    let changeRms = null;
    if (index === 0) {
      if (frame.changeRms !== null || frame.continuity !== null) fail("the first interval must not invent an adjacent comparison.");
    } else {
      changeRms = finite(frame.changeRms, `frames[${index}].changeRms`);
      if (changeRms < 0) fail(`frames[${index}].changeRms must not be negative.`);
      if (frame.continuity !== null) {
        finite(frame.continuity, `frames[${index}].continuity`);
        if (frame.continuity < -1 || frame.continuity > 1) fail(`frames[${index}].continuity is outside its valid range.`);
      }
    }
    previousEnd = endTime;
    return { index, time, duration, endTime, rms, changeRms };
  });

  const continuity = report.clip.continuity === null ? null : finite(report.clip.continuity, "clip.continuity");
  if (continuity !== null && (continuity < -1 || continuity > 1)) fail("clip.continuity is outside its valid range.");
  const spatialDistribution = finite(report.clip.spatialDistribution, "clip.spatialDistribution");
  if (spatialDistribution < 0 || spatialDistribution > 100) fail("clip.spatialDistribution is outside its valid range.");
  const peakMagnitude = validateMoment(report.clip.peakMagnitude, "clip.peakMagnitude", frames, "rms");
  const largestChange = validateMoment(report.clip.largestChange, "clip.largestChange", frames, "changeRms");
  const expectedPeak = frames.reduce((best, frame) => frame.rms > best.rms ? frame : best);
  if (peakMagnitude.index !== expectedPeak.index) fail("clip.peakMagnitude must identify the earliest maximum response interval.");
  if (frames.length > 1) {
    const expectedChange = frames.slice(1).reduce((best, frame) => frame.changeRms > best.changeRms ? frame : best);
    if (largestChange.index !== expectedChange.index) fail("clip.largestChange must identify the earliest maximum pattern change.");
  }

  if (!Array.isArray(report.phases) || report.phases.length !== PHASES.length) {
    fail("opening, middle, and ending phase descriptors are required.");
  }
  let phaseEnd = null;
  const phases = report.phases.map((phase, index) => {
    const expected = PHASES[index];
    if (!isRecord(phase) || phase.id !== expected.id) fail("phase descriptors must preserve opening-to-ending order.");
    const startTime = finite(phase.startTime, `phases[${index}].startTime`);
    const endTime = finite(phase.endTime, `phases[${index}].endTime`);
    const responseMagnitude = finite(phase.responseMagnitude, `phases[${index}].responseMagnitude`);
    if (endTime <= startTime || responseMagnitude < 0 || (phaseEnd !== null && !almostEqual(startTime, phaseEnd))) {
      fail(`phases[${index}] has invalid timing or magnitude.`);
    }
    phaseEnd = endTime;
    return { ...expected, startTime, endTime, responseMagnitude };
  });
  if (!almostEqual(phases[0].startTime, frames[0].time)
      || !almostEqual(phases.at(-1).endTime, frames.at(-1).endTime)) {
    fail("phase descriptors must cover the prediction intervals.");
  }

  return {
    frames,
    phases,
    continuity,
    spatialDistribution,
    largestChange,
    isVisionOnly: report.source.inferenceMode === "vision-only",
  };
}

function phaseForTime(phases, time) {
  return phases.find((phase, index) => time >= phase.startTime
    && (index === phases.length - 1 ? time <= phase.endTime : time < phase.endTime)) || phases.at(-1);
}

function describeTiming(phases) {
  const ranked = [...phases].sort((left, right) => right.responseMagnitude - left.responseMagnitude
    || PHASES.findIndex(({ id }) => id === left.id) - PHASES.findIndex(({ id }) => id === right.id));
  const strongest = ranked[0];
  const weakest = ranked.at(-1);
  const scale = Math.max(strongest.responseMagnitude, Number.EPSILON);
  const isEven = (strongest.responseMagnitude - weakest.responseMagnitude) / scale <= 0.08;
  if (isEven) {
    return {
      kind: "even",
      strongestPhase: null,
      title: "The response is spread across the whole clip",
      body: "The model’s predicted cortical response is fairly even through the opening, middle, and ending. No single section separates clearly from the others in this clip-only comparison.",
    };
  }
  return {
    kind: strongest.id,
    strongestPhase: strongest.id,
    title: `The ${strongest.label} carries the clearest response`,
    body: `The model’s predicted cortical response is most pronounced in the ${strongest.label}. The remaining sections are less prominent when compared only with this same clip.`,
  };
}

function describeDynamics(continuity, largestChange, phases) {
  if (continuity === null) {
    return {
      kind: "unavailable",
      title: "The pattern-to-pattern read is limited",
      body: "The adjacent predicted cortical patterns do not support a reliable stability comparison, so the report does not label the clip as steady or changeable.",
    };
  }
  const changePhase = largestChange ? phaseForTime(phases, largestChange.time) : null;
  const location = changePhase ? ` The clearest shift appears in the ${changePhase.label}.` : "";
  if (continuity >= 0.75) {
    return {
      kind: "steady",
      title: "The predicted pattern stays fairly consistent",
      body: `The cortical pattern changes gently from one part of the clip to the next.${location}`,
    };
  }
  if (continuity >= 0.25) {
    return {
      kind: "developing",
      title: "The predicted pattern develops as the clip moves",
      body: `The cortical pattern keeps some continuity while still changing across the clip.${location}`,
    };
  }
  return {
    kind: "changing",
    title: "The predicted pattern changes noticeably",
    body: `The cortical pattern differs meaningfully between parts of the clip rather than staying uniform.${location}`,
  };
}

function describeSpread(spatialDistribution) {
  if (spatialDistribution >= 60) {
    return {
      kind: "broad",
      title: "The response is broadly distributed",
      body: "The predicted signal is spread across much of the usable cortical surface instead of being concentrated in a small set of surface locations.",
    };
  }
  if (spatialDistribution <= 30) {
    return {
      kind: "concentrated",
      title: "The response is more concentrated",
      body: "The predicted signal is carried more heavily by a smaller share of the usable cortical surface rather than being distributed broadly.",
    };
  }
  return {
    kind: "mixed",
    title: "The response has a mixed spatial spread",
    body: "The predicted signal is neither strongly concentrated nor broadly uniform across the usable cortical surface.",
  };
}

function describeExperiment(timing, largestChange, phases) {
  const changePhase = largestChange ? phaseForTime(phases, largestChange.time) : null;
  if (changePhase && changePhase.id !== "early") {
    return {
      kind: "preview-change",
      title: "Test an earlier preview of the clearest shift",
      body: `Make one alternate cut that previews the moment linked to the clearest cortical-pattern change from the ${changePhase.label} in the opening. Keep the current cut as the control and decide between them using observed platform results.`,
    };
  }
  if (changePhase?.id === "early") {
    return {
      kind: "hold-opening",
      title: "Test a cleaner version of the opening shift",
      body: "Make one alternate cut that gives the opening moment linked to the clearest cortical-pattern change more visual space. Keep the current cut as the control and decide between them using observed platform results.",
    };
  }
  if (timing.strongestPhase && timing.strongestPhase !== "early") {
    const phase = phases.find(({ id }) => id === timing.strongestPhase);
    return {
      kind: "preview-response",
      title: "Test an opening preview from the strongest section",
      body: `Make one alternate cut that previews a representative moment from the ${phase.label} in the opening. Keep the current cut as the control and decide between them using observed platform results.`,
    };
  }
  return {
    kind: "opening-variation",
    title: "Test one focused opening variation",
    body: "Keep the current clip as the control and make one alternate cut with a simpler opening visual. Use observed platform results to decide whether the edit helps your actual audience.",
  };
}

/**
 * Converts a verified, non-behavioral TRIBE cortical descriptor report into
 * creator-readable qualitative language. Numeric values are never emitted.
 */
export function createTribePlainLanguageReport(report) {
  const validated = validateReport(report);
  const timing = describeTiming(validated.phases);
  const dynamics = describeDynamics(validated.continuity, validated.largestChange, validated.phases);
  const spread = describeSpread(validated.spatialDistribution);
  const experiment = describeExperiment(timing, validated.largestChange, validated.phases);
  const modality = validated.isVisionOnly
    ? {
        kind: "vision-only",
        title: "This is a video-only model read",
        body: "No separate audio or text input was modeled. Visible on-screen text may still be present as pixels in the video, but it was not interpreted as language.",
      }
    : {
        kind: "declared-inputs",
        title: "This read follows the inputs declared by the result",
        body: "Interpret the summary only within the input coverage recorded by the verified model result.",
      };

  return deepFreeze({
    schemaVersion: TRIBE_PLAIN_LANGUAGE_SCHEMA_VERSION,
    title: "What the model says about this video",
    introduction: "Here is the verified cortical prediction translated into plain creator language.",
    timing,
    dynamics,
    spread,
    experiment,
    modality,
    boundary: "This is a description of a model-predicted cortical signal. It is not a measurement of viewer response and does not forecast publishing performance.",
  });
}
