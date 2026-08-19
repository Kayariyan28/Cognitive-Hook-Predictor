const MAX_FRAME_SAMPLES = 48;
const METADATA_TIMEOUT_MS = 12_000;
const SEEK_TIMEOUT_MS = 5_000;
const AUDIO_READ_TIMEOUT_MS = 20_000;
const AUDIO_DECODE_TIMEOUT_MS = 20_000;

const SCORE_KEYS = [
  "viralPotential",
  "engagementScore",
  "hookScore",
  "holdRate",
  "visualSignal",
  "editDynamics",
  "audioSignal",
  "visualStability",
  "speechLikelihood",
  "coverage",
];

const VIRALITY_FORMULA_VERSION = "local-content-outlook-v1";
const VIRALITY_COMPONENT_WEIGHTS = Object.freeze({
  opening: 0.24,
  sustain: 0.24,
  pacing: 0.14,
  ending: 0.1,
  visualClarity: 0.12,
  visualStability: 0.08,
  audioSupport: 0.08,
});

function clamp(value, minimum = 0, maximum = 1) {
  const number = Number(value);
  if (!Number.isFinite(number)) return minimum;
  return Math.min(maximum, Math.max(minimum, number));
}

function clamp01(value) {
  return clamp(value, 0, 1);
}

function average(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function standardDeviation(values) {
  if (values.length < 2) return 0;
  const center = average(values);
  return Math.sqrt(average(values.map((value) => (value - center) ** 2)));
}

function percentile(values, position) {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const index = clamp01(position) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const fraction = index - lower;
  return sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
}

function scoreWithinBand(value, low, high, lowerShoulder, upperShoulder = lowerShoulder) {
  if (value >= low && value <= high) return 1;
  if (value < low) return clamp01((value - (low - lowerShoulder)) / lowerShoulder);
  return clamp01(1 - (value - high) / upperShoulder);
}

function roundScore(value) {
  return Math.round(clamp(value, 0, 100));
}

function roundMetric(value, digits = 4) {
  if (!Number.isFinite(value)) return 0;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function normalizeFrame(frame, index) {
  return {
    time: Math.max(0, finiteNumber(frame?.time, index)),
    luminance: clamp01(frame?.luminance ?? frame?.brightness),
    contrast: clamp01(frame?.contrast),
    colorfulness: clamp01(frame?.colorfulness ?? frame?.color),
    edgeDensity: clamp01(frame?.edgeDensity ?? frame?.edges),
    motion: clamp01(frame?.motion),
    cut: clamp01(frame?.cut ?? frame?.cutSignal),
  };
}

function normalizeFrames(signals) {
  const frames = Array.isArray(signals?.frames)
    ? signals.frames
    : Array.isArray(signals?.visual?.frames)
      ? signals.visual.frames
      : [];

  return frames
    .map(normalizeFrame)
    .sort((left, right) => left.time - right.time);
}

function normalizeAudio(audio) {
  const available = audio?.available === true;
  const windows = Array.isArray(audio?.windows)
    ? audio.windows
        .map((window, index) => ({
          time: Math.max(0, finiteNumber(window?.time, index)),
          rms: clamp01(window?.rms),
          onset: clamp01(window?.onset),
        }))
        .sort((left, right) => left.time - right.time)
    : [];

  return {
    available,
    rms: clamp01(audio?.rms),
    dynamics: clamp01(audio?.dynamics),
    silenceRatio: clamp01(audio?.silenceRatio),
    onsetRate: Math.max(0, finiteNumber(audio?.onsetRate)),
    speechLikelihood: clamp01(audio?.speechLikelihood),
    windows,
    reason: typeof audio?.reason === "string" ? audio.reason : "",
  };
}

function rmsInterest(rms) {
  if (rms <= 0) return 0;
  const decibels = 20 * Math.log10(rms);
  const usefulLevel = clamp01((decibels + 48) / 34);
  const clippingPenalty = decibels > -3 ? clamp01(1 - (decibels + 3) / 3) : 1;
  return usefulLevel * clippingPenalty;
}

function motionInterest(motion) {
  const activity = clamp01(motion / 0.16);
  const control = motion <= 0.4 ? 1 : clamp01(1 - (motion - 0.4) / 0.45);
  return activity * control;
}

function frameVisualSignal(frame) {
  const brightness = scoreWithinBand(frame.luminance, 0.2, 0.82, 0.2, 0.18);
  const contrast = clamp01(frame.contrast / 0.28);
  const color = clamp01(frame.colorfulness / 0.45);
  const edges = scoreWithinBand(frame.edgeDensity, 0.055, 0.34, 0.055, 0.34);
  const motion = motionInterest(frame.motion);

  return clamp01(
    brightness * 0.14
      + contrast * 0.22
      + color * 0.16
      + edges * 0.17
      + motion * 0.23
      + frame.cut * 0.08,
  );
}

function nearestAudioWindow(windows, time) {
  if (!windows.length) return null;
  let nearest = windows[0];
  let distance = Math.abs(nearest.time - time);
  for (let index = 1; index < windows.length; index += 1) {
    const nextDistance = Math.abs(windows[index].time - time);
    if (nextDistance >= distance) break;
    nearest = windows[index];
    distance = nextDistance;
  }
  return nearest;
}

function buildTimeline(frames, audio) {
  return frames.map((frame) => {
    const visual = frameVisualSignal(frame);
    let combined = visual;

    if (audio.available) {
      const audioWindow = nearestAudioWindow(audio.windows, frame.time);
      const localAudio = audioWindow
        ? rmsInterest(audioWindow.rms) * 0.76 + audioWindow.onset * 0.24
        : rmsInterest(audio.rms) * 0.8 + clamp01(audio.onsetRate / 4) * 0.2;
      combined = visual * 0.84 + localAudio * 0.16;
    }

    return {
      time: roundMetric(frame.time, 3),
      score: roundScore(combined * 100),
    };
  });
}

function derivePeakTime(timeline) {
  if (!timeline.length) return 0;
  return timeline.reduce(
    (peak, point) => (point.score > peak.score ? point : peak),
    timeline[0],
  ).time;
}

function makeStrengths(scores, audioAvailable) {
  const candidates = [
    [scores.hookScore, "The opening has a strong measured content-feature signal."],
    [scores.holdRate, "The sampled content-feature signal stays comparatively strong beyond the opening."],
    [scores.visualSignal, "Contrast, color, and composition create a strong visual signal."],
    [scores.editDynamics, "Motion and edit cadence create visible progression across the clip."],
    [scores.visualStability, "The picture remains visually stable and readable."],
  ];

  if (audioAvailable) {
    candidates.push(
      [scores.audioSignal, "Audio energy and dynamics reinforce the visual pacing."],
      [scores.speechLikelihood, "The audio has a speech-like modulation signal."],
    );
  }

  const strong = candidates
    .filter(([score]) => score >= 64)
    .sort((left, right) => right[0] - left[0])
    .slice(0, 3)
    .map(([, message]) => message);

  if (strong.length) return strong;

  const relative = [...candidates].sort((left, right) => right[0] - left[0])[0];
  return relative
    ? [`The most developed measured signal is: ${relative[1].charAt(0).toLowerCase()}${relative[1].slice(1)}`]
    : ["More usable frame samples are needed to identify a strength."];
}

function makeImprovements(scores, frames, audio) {
  const meanMotion = average(frames.map((frame) => frame.motion));
  const meanCut = average(frames.map((frame) => frame.cut));
  const candidates = [
    [100 - scores.hookScore, "Strengthen the first few seconds with a clearer subject, change, or payoff cue."],
    [100 - scores.visualSignal, "Increase subject separation with more intentional contrast, color, or framing."],
    [100 - scores.editDynamics, "Use purposeful movement or edit beats to create clearer visual progression."],
    [100 - scores.visualStability, "Reduce abrupt exposure changes, visual clutter, or uncontrolled camera motion."],
  ];

  if (meanMotion < 0.025) {
    candidates.push([92, "The sampled frames change very little; add meaningful movement or visual progression."]);
  } else if (meanMotion > 0.42) {
    candidates.push([92, "Motion is unusually intense; add steadier moments so the subject stays legible."]);
  }

  if (meanCut < 0.018 && frames.length > 3) {
    candidates.push([70, "Consider a deliberate pattern interrupt where the message changes direction."]);
  } else if (meanCut > 0.34) {
    candidates.push([86, "Frequent scene changes may compete with comprehension; let key shots breathe."]);
  }

  if (!audio.available) {
    candidates.push([96, "No decodable audio track was found; add or verify audio if sound is part of the concept."]);
  } else {
    candidates.push(
      [100 - scores.audioSignal, "Refine audio level, dynamics, and rhythmic accents so sound supports the edit."],
    );
    if (audio.silenceRatio > 0.45) {
      candidates.push([88, "Long low-energy audio stretches weaken the sound signal; tighten them if they are not intentional."]);
    }
  }

  const improvements = candidates
    .filter(([priority]) => priority >= 38)
    .sort((left, right) => right[0] - left[0])
    .slice(0, 3)
    .map(([, message]) => message);

  return improvements.length
    ? improvements
    : ["The measured signals are well balanced; test alternate openings or edit rhythms to compare audience response."];
}

function makeViralityComponent(label, score, baseWeight, sampleCount, sufficient, reason) {
  return {
    label,
    score: sufficient ? roundScore(score) : null,
    baseWeight,
    weight: 0,
    available: sampleCount > 0,
    sufficient,
    sampleCount,
    reason,
  };
}

function deriveViralityOutlook({
  raw,
  sustainedAttention,
  pacing,
  endingStrength,
  openingSampleCount,
  postOpeningSampleCount,
  endingSampleCount,
  frameSampleCount,
  audio,
  evidenceCoverage,
}) {
  const components = {
    opening: makeViralityComponent(
      "Opening signal",
      raw.hookScore,
      VIRALITY_COMPONENT_WEIGHTS.opening,
      openingSampleCount,
      openingSampleCount >= 1,
      openingSampleCount >= 1
        ? "Visual and available audio feature signal in the opening window."
        : "Needs at least one sampled opening frame.",
    ),
    sustain: makeViralityComponent(
      "Post-opening continuity",
      sustainedAttention,
      VIRALITY_COMPONENT_WEIGHTS.sustain,
      postOpeningSampleCount,
      postOpeningSampleCount >= 2,
      postOpeningSampleCount >= 2
        ? "Consistency of sampled content-feature signal after the opening window."
        : "Needs at least two samples after the opening window; no sustain value is inferred.",
    ),
    pacing: makeViralityComponent(
      "Pacing signal",
      pacing,
      VIRALITY_COMPONENT_WEIGHTS.pacing,
      frameSampleCount,
      frameSampleCount >= 2,
      frameSampleCount >= 2
        ? "Observed motion and cut cadence across sampled frames."
        : "Needs at least two frames to measure change over time.",
    ),
    ending: makeViralityComponent(
      "Ending signal",
      endingStrength,
      VIRALITY_COMPONENT_WEIGHTS.ending,
      endingSampleCount,
      frameSampleCount >= 3,
      frameSampleCount >= 3
        ? "Content-feature signal in the final sampled segment relative to the opening."
        : "Needs at least three frames to distinguish an ending segment.",
    ),
    visualClarity: makeViralityComponent(
      "Visual clarity",
      raw.visualSignal,
      VIRALITY_COMPONENT_WEIGHTS.visualClarity,
      frameSampleCount,
      frameSampleCount >= 1,
      frameSampleCount >= 1
        ? "Sampled luminance, contrast, color, edge density, and visual variation."
        : "Needs at least one sampled frame.",
    ),
    visualStability: makeViralityComponent(
      "Visual stability",
      raw.visualStability,
      VIRALITY_COMPONENT_WEIGHTS.visualStability,
      frameSampleCount,
      frameSampleCount >= 2,
      frameSampleCount >= 2
        ? "Sampled exposure, contrast, motion, and cut control over time."
        : "Needs at least two frames to estimate stability over time.",
    ),
    audioSupport: makeViralityComponent(
      "Audio support",
      raw.audioSignal,
      VIRALITY_COMPONENT_WEIGHTS.audioSupport,
      audio.available ? Math.max(1, audio.windows.length) : 0,
      audio.available,
      audio.available
        ? "Decoded audio level, dynamics, silence, onset cadence, and speech-like modulation."
        : `Audio is excluded from the score${audio.reason ? `: ${audio.reason}` : "."}`,
    ),
  };

  const usableComponents = Object.values(components).filter(
    (component) => component.sufficient && component.score !== null,
  );
  const usableBaseWeight = usableComponents.reduce(
    (sum, component) => sum + component.baseWeight,
    0,
  );

  let weightedScore = 0;
  if (usableBaseWeight > 0) {
    for (const component of usableComponents) {
      component.weight = roundMetric(component.baseWeight / usableBaseWeight, 6);
      weightedScore += component.score * (component.baseWeight / usableBaseWeight);
    }
  }

  return {
    score: roundScore(weightedScore),
    label: "Pre-publish virality outlook",
    scoreKind: "deterministic-content-feature-index",
    interpretation:
      "Higher values mean stronger alignment with this declared clip-feature formula. They do not mean a higher measured probability of going viral.",
    evidenceCoverage: roundScore(evidenceCoverage),
    evidenceKind: "input-coverage-only",
    calibration: "none",
    behavioralOutcome: false,
    tribeContribution: false,
    formulaVersion: VIRALITY_FORMULA_VERSION,
    components,
    methodology:
      "Deterministic, hand-weighted index from locally sampled opening signal, post-opening continuity, pacing, ending signal, visual clarity, visual stability, and available audio support. Missing or insufficient components receive no score and their weights are redistributed across sufficient evidence. It is not trained or calibrated against views, shares, retention, or watch time; it is not a probability or behavioral outcome; and TRIBE v2 does not contribute to this score.",
  };
}

/**
 * Convert normalized extraction signals into deterministic, explainable content scores.
 * Frame features use the 0..1 range. `onsetRate` is measured in onsets per second.
 * The returned values are content-signal proxies, not probabilities or view forecasts.
 */
export function deriveScores(signals = {}) {
  const frames = normalizeFrames(signals);
  const audio = normalizeAudio(signals.audio);
  const inferredDuration = frames.length ? frames[frames.length - 1].time : 0;
  const duration = Math.max(
    0,
    finiteNumber(signals.duration ?? signals.metadata?.duration, inferredDuration),
  );
  const timeline = buildTimeline(frames, audio);

  const luminances = frames.map((frame) => frame.luminance);
  const contrasts = frames.map((frame) => frame.contrast);
  const colors = frames.map((frame) => frame.colorfulness);
  const edges = frames.map((frame) => frame.edgeDensity);
  const motions = frames.map((frame) => frame.motion);
  const cuts = frames.map((frame) => frame.cut);

  const brightnessQuality = average(
    luminances.map((value) => scoreWithinBand(value, 0.2, 0.82, 0.2, 0.18)),
  );
  const contrastQuality = clamp01(average(contrasts) / 0.28);
  const colorQuality = clamp01(average(colors) / 0.45);
  const edgeQuality = average(
    edges.map((value) => scoreWithinBand(value, 0.055, 0.34, 0.055, 0.34)),
  );
  const visualVariation = clamp01(
    standardDeviation(luminances) / 0.14
      + standardDeviation(contrasts) / 0.18
      + average(motions) / 0.32,
  );
  const visualSignal = frames.length
    ? (brightnessQuality * 0.15
      + contrastQuality * 0.25
      + colorQuality * 0.2
      + edgeQuality * 0.2
      + visualVariation * 0.2) * 100
    : 0;

  const motionQuality = average(motions.map(motionInterest));
  const cutCadence = scoreWithinBand(average(cuts), 0.025, 0.3, 0.025, 0.4);
  const meanTimeline = average(timeline.map((point) => point.score)) / 100;
  const editDynamics = frames.length
    ? (meanTimeline * 0.42 + motionQuality * 0.38 + cutCadence * 0.2) * 100
    : 0;

  const luminanceStability = clamp01(1 - standardDeviation(luminances) / 0.3);
  const contrastStability = clamp01(1 - standardDeviation(contrasts) / 0.32);
  const motionControl = clamp01(1 - Math.max(0, average(motions) - 0.24) / 0.42);
  const cutControl = clamp01(1 - Math.max(0, average(cuts) - 0.22) / 0.55);
  const visualStability = frames.length
    ? (luminanceStability * 0.34
      + contrastStability * 0.25
      + motionControl * 0.25
      + cutControl * 0.16) * 100
    : 0;

  const hookWindowEnd = duration > 0
    ? Math.min(3, Math.max(0.75, duration * 0.25))
    : 3;
  const sampledOpening = timeline.filter((point) => point.time <= hookWindowEnd);
  const opening = sampledOpening.length ? sampledOpening : timeline;
  const hookMean = average(opening.map((point) => point.score));
  const hookPeak = opening.length ? Math.max(...opening.map((point) => point.score)) : 0;
  const firstSignal = opening[0]?.score ?? 0;
  const hookScore = hookMean * 0.55 + hookPeak * 0.3 + firstSignal * 0.15;

  const postOpening = timeline.filter((point) => point.time > hookWindowEnd);
  const sustained = postOpening.length ? postOpening : timeline;
  const sustainedMean = average(sustained.map((point) => point.score));
  const thirdSize = Math.max(1, Math.ceil(timeline.length / 3));
  const earlyMean = average(timeline.slice(0, thirdSize).map((point) => point.score));
  const lateMean = average(timeline.slice(-thirdSize).map((point) => point.score));
  const nonDrop = clamp01(0.75 + (lateMean - earlyMean) / 80);
  const holdRate = frames.length
    ? hookScore * 0.23
      + sustainedMean * 0.32
      + editDynamics * 0.2
      + visualStability * 0.13
      + nonDrop * 100 * 0.12
    : 0;

  // Engagement is a clip-signal read, not a retention model trained on audience
  // outcomes. Keep each component independently inspectable so callers never
  // have to present the aggregate as measured or calibrated watch time.
  const pacing = frames.length
    ? (motionQuality * 0.58 + cutCadence * 0.42) * 100
    : 0;
  const sustainedAttention = frames.length
    ? sustainedMean * 0.68 + nonDrop * 100 * 0.2 + editDynamics * 0.12
    : 0;
  const endingStrength = frames.length
    ? lateMean * 0.76 + nonDrop * 100 * 0.24
    : 0;

  const audioLevel = rmsInterest(audio.rms);
  const audioDynamics = scoreWithinBand(audio.dynamics, 0.12, 0.72, 0.12, 0.28);
  const onsetCadence = scoreWithinBand(audio.onsetRate, 0.35, 4.8, 0.35, 3.2);
  const audioSignal = audio.available
    ? (audioLevel * 0.27
      + audioDynamics * 0.23
      + (1 - audio.silenceRatio) * 0.2
      + onsetCadence * 0.2
      + audio.speechLikelihood * 0.1) * 100
    : 50;
  const speechLikelihood = audio.available ? audio.speechLikelihood * 100 : 0;

  const raw = {
    hookScore,
    holdRate,
    visualSignal,
    editDynamics,
    audioSignal,
    visualStability,
    speechLikelihood,
  };

  const viralPotential = audio.available
    ? raw.hookScore * 0.22
      + raw.holdRate * 0.25
      + raw.visualSignal * 0.18
      + raw.editDynamics * 0.13
      + raw.visualStability * 0.08
      + raw.audioSignal * 0.1
      + raw.speechLikelihood * 0.04
    : raw.hookScore * 0.24
      + raw.holdRate * 0.29
      + raw.visualSignal * 0.21
      + raw.editDynamics * 0.16
      + raw.visualStability * 0.1;

  const engagementScore = audio.available
    ? raw.hookScore * 0.24
      + sustainedAttention * 0.25
      + pacing * 0.17
      + endingStrength * 0.12
      + raw.visualStability * 0.1
      + raw.audioSignal * 0.12
    : raw.hookScore * 0.27
      + sustainedAttention * 0.28
      + pacing * 0.2
      + endingStrength * 0.14
      + raw.visualStability * 0.11;

  const expectedFrames = Math.min(
    24,
    Math.max(6, Math.ceil((duration || frames.length) * 1.5)),
  );
  const sampleSufficiency = clamp01(frames.length / expectedFrames);
  const temporalCoverage = frames.length >= 2 && duration > 0
    ? clamp01((frames[frames.length - 1].time - frames[0].time) / duration)
    : frames.length ? 0.35 : 0;
  const metadataConfidence = duration > 0 && frames.length ? 1 : frames.length ? 0.45 : 0;
  const coverage = (
    sampleSufficiency * 0.55
      + temporalCoverage * 0.2
      + metadataConfidence * 0.15
      + (audio.available ? 1 : 0) * 0.1
  ) * 100;

  const scores = {
    viralPotential: roundScore(viralPotential),
    engagementScore: roundScore(engagementScore),
    hookScore: roundScore(raw.hookScore),
    holdRate: roundScore(raw.holdRate),
    visualSignal: roundScore(raw.visualSignal),
    editDynamics: roundScore(raw.editDynamics),
    audioSignal: roundScore(raw.audioSignal),
    visualStability: roundScore(raw.visualStability),
    speechLikelihood: roundScore(raw.speechLikelihood),
    coverage: roundScore(coverage),
  };

  // This assertion is intentionally internal: it protects UI consumers if formulas change.
  for (const key of SCORE_KEYS) scores[key] = roundScore(scores[key]);

  const engagementComponents = {
    earlyAttention: roundScore(raw.hookScore),
    sustainedAttention: roundScore(sustainedAttention),
    pacing: roundScore(pacing),
    endingStrength: roundScore(endingStrength),
    visualStability: roundScore(raw.visualStability),
    audioDynamics: audio.available ? roundScore(raw.audioSignal) : null,
  };

  const engagementMethodology = audio.available
    ? "Deterministic, hand-weighted content-feature proxy from sampled early attention, sustained timeline, pacing, ending, visual stability, and audio dynamics. Confidence reflects input coverage only. It is not measured retention, predicted watch-time seconds, model certainty, or a calibrated probability."
    : "Deterministic, hand-weighted content-feature proxy from sampled early attention, sustained timeline, pacing, ending, and visual stability; audio was unavailable and its weight was redistributed. Confidence reflects input coverage only. It is not measured retention, predicted watch-time seconds, model certainty, or a calibrated probability.";

  const virality = deriveViralityOutlook({
    raw,
    sustainedAttention,
    pacing,
    endingStrength,
    openingSampleCount: sampledOpening.length,
    postOpeningSampleCount: postOpening.length,
    endingSampleCount: Math.min(thirdSize, timeline.length),
    frameSampleCount: frames.length,
    audio,
    evidenceCoverage: scores.coverage,
  });

  return {
    scores,
    virality,
    engagement: {
      score: scores.engagementScore,
      confidence: scores.coverage,
      coverage: scores.coverage,
      confidenceKind: "input-coverage-only",
      components: engagementComponents,
      methodology: engagementMethodology,
    },
    timeline,
    peakTime: derivePeakTime(timeline),
    strengths: makeStrengths(scores, audio.available),
    improvements: makeImprovements(scores, frames, audio),
    methodology:
      "Deterministic content-feature proxy based on sampled image and audio signals; it is not a probability of virality or a forecast of views.",
  };
}

function reportProgress(onProgress, stage, progress, message) {
  if (typeof onProgress !== "function") return;
  try {
    onProgress({ stage, progress: clamp(progress, 0, 100), message });
  } catch {
    // Reporting is advisory; a rendering callback must not abort file analysis.
  }
}

function withTimeout(promise, milliseconds, message) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(message)), milliseconds);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

function waitForMetadata(video) {
  if (video.readyState >= 1 && Number.isFinite(video.duration)) return Promise.resolve();

  return withTimeout(
    new Promise((resolve, reject) => {
      const cleanup = () => {
        video.removeEventListener("loadedmetadata", handleMetadata);
        video.removeEventListener("error", handleError);
      };
      const handleMetadata = () => {
        cleanup();
        resolve();
      };
      const handleError = () => {
        cleanup();
        reject(new Error("The browser could not read this video's metadata."));
      };
      video.addEventListener("loadedmetadata", handleMetadata, { once: true });
      video.addEventListener("error", handleError, { once: true });
    }),
    METADATA_TIMEOUT_MS,
    "Timed out while reading video metadata.",
  );
}

function seekVideo(video, targetTime) {
  const safeTarget = clamp(targetTime, 0, Math.max(0, video.duration - 0.001));

  return withTimeout(
    new Promise((resolve, reject) => {
      let settled = false;
      const cleanup = () => {
        video.removeEventListener("seeked", handleReady);
        video.removeEventListener("loadeddata", handleReady);
        video.removeEventListener("canplay", handleReady);
        video.removeEventListener("error", handleError);
      };
      const finish = () => {
        if (settled) return;
        settled = true;
        cleanup();
        resolve();
      };
      const handleReady = () => {
        if (video.readyState >= 2 && Math.abs(video.currentTime - safeTarget) <= 0.12) finish();
      };
      const handleError = () => {
        if (settled) return;
        settled = true;
        cleanup();
        reject(new Error(`Could not decode the frame near ${safeTarget.toFixed(2)}s.`));
      };

      video.addEventListener("seeked", handleReady);
      video.addEventListener("loadeddata", handleReady);
      video.addEventListener("canplay", handleReady);
      video.addEventListener("error", handleError, { once: true });

      try {
        video.currentTime = safeTarget;
        if (video.readyState >= 2 && Math.abs(video.currentTime - safeTarget) <= 0.012) {
          queueMicrotask(finish);
        }
      } catch (error) {
        cleanup();
        reject(error);
      }
    }),
    SEEK_TIMEOUT_MS,
    `Timed out while seeking to ${safeTarget.toFixed(2)}s.`,
  );
}

function chooseCanvasSize(width, height) {
  const safeWidth = Math.max(1, finiteNumber(width, 1));
  const safeHeight = Math.max(1, finiteNumber(height, 1));
  const longestSide = 192;
  const scale = Math.min(1, longestSide / Math.max(safeWidth, safeHeight));
  return {
    width: Math.max(1, Math.round(safeWidth * scale)),
    height: Math.max(1, Math.round(safeHeight * scale)),
  };
}

function extractFrameFeatures(imageData, previousFrame, time) {
  const { data, width, height } = imageData;
  const pixelCount = width * height;
  const grayscale = new Float32Array(pixelCount);
  const histogram = new Float32Array(16);

  let luminanceSum = 0;
  let luminanceSquaredSum = 0;
  let redGreenSum = 0;
  let redGreenSquaredSum = 0;
  let yellowBlueSum = 0;
  let yellowBlueSquaredSum = 0;

  for (let pixel = 0, offset = 0; pixel < pixelCount; pixel += 1, offset += 4) {
    const red = data[offset];
    const green = data[offset + 1];
    const blue = data[offset + 2];
    const luminance = (red * 0.2126 + green * 0.7152 + blue * 0.0722) / 255;
    const redGreen = red - green;
    const yellowBlue = (red + green) / 2 - blue;

    grayscale[pixel] = luminance;
    histogram[Math.min(15, Math.floor(luminance * 16))] += 1 / pixelCount;
    luminanceSum += luminance;
    luminanceSquaredSum += luminance ** 2;
    redGreenSum += redGreen;
    redGreenSquaredSum += redGreen ** 2;
    yellowBlueSum += yellowBlue;
    yellowBlueSquaredSum += yellowBlue ** 2;
  }

  const luminance = luminanceSum / pixelCount;
  const contrast = Math.sqrt(Math.max(0, luminanceSquaredSum / pixelCount - luminance ** 2));
  const meanRedGreen = redGreenSum / pixelCount;
  const meanYellowBlue = yellowBlueSum / pixelCount;
  const stdRedGreen = Math.sqrt(
    Math.max(0, redGreenSquaredSum / pixelCount - meanRedGreen ** 2),
  );
  const stdYellowBlue = Math.sqrt(
    Math.max(0, yellowBlueSquaredSum / pixelCount - meanYellowBlue ** 2),
  );
  const colorfulness = clamp01(
    (Math.sqrt(stdRedGreen ** 2 + stdYellowBlue ** 2)
      + 0.3 * Math.sqrt(meanRedGreen ** 2 + meanYellowBlue ** 2)) / 180,
  );

  let edgePixels = 0;
  let edgeComparisons = 0;
  for (let y = 1; y < height; y += 1) {
    for (let x = 1; x < width; x += 1) {
      const index = y * width + x;
      const energy = (
        Math.abs(grayscale[index] - grayscale[index - 1])
          + Math.abs(grayscale[index] - grayscale[index - width])
      ) / 2;
      if (energy > 0.105) edgePixels += 1;
      edgeComparisons += 1;
    }
  }
  const edgeDensity = edgeComparisons ? edgePixels / edgeComparisons : 0;

  let motion = 0;
  let histogramDistance = 0;
  if (previousFrame?.grayscale?.length === grayscale.length) {
    for (let index = 0; index < grayscale.length; index += 1) {
      motion += Math.abs(grayscale[index] - previousFrame.grayscale[index]);
    }
    motion /= grayscale.length;
    for (let index = 0; index < histogram.length; index += 1) {
      histogramDistance += Math.abs(histogram[index] - previousFrame.histogram[index]);
    }
    histogramDistance /= 2;
  }

  const cut = previousFrame
    ? clamp01(
      clamp01((motion - 0.055) / 0.22) * 0.72
        + clamp01((histogramDistance - 0.08) / 0.42) * 0.28,
    )
    : 0;

  return {
    publicSignals: {
      time: roundMetric(time, 3),
      luminance: roundMetric(luminance),
      contrast: roundMetric(contrast),
      colorfulness: roundMetric(colorfulness),
      edgeDensity: roundMetric(edgeDensity),
      motion: roundMetric(motion),
      cut: roundMetric(cut),
    },
    grayscale,
    histogram,
  };
}

function frameSampleTimes(duration, count) {
  if (count <= 1) return [0];
  const end = Math.max(0, duration - Math.min(0.05, duration * 0.01));
  return Array.from({ length: count }, (_, index) => (end * index) / (count - 1));
}

async function sampleVideoFrames(video, onProgress, warnings) {
  const requestedCount = Math.min(
    MAX_FRAME_SAMPLES,
    Math.max(8, Math.ceil(video.duration * 2)),
  );
  const times = frameSampleTimes(video.duration, requestedCount);
  const canvas = document.createElement("canvas");
  const dimensions = chooseCanvasSize(video.videoWidth, video.videoHeight);
  canvas.width = dimensions.width;
  canvas.height = dimensions.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("This browser does not provide a readable 2D canvas.");

  const frames = [];
  let previousFrame = null;

  for (let index = 0; index < times.length; index += 1) {
    const time = times[index];
    try {
      await seekVideo(video, time);
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
      const extracted = extractFrameFeatures(imageData, previousFrame, time);
      frames.push(extracted.publicSignals);
      previousFrame = extracted;
    } catch (error) {
      warnings.push(error instanceof Error ? error.message : `Skipped the frame near ${time.toFixed(2)}s.`);
    }

    reportProgress(
      onProgress,
      "frames",
      10 + ((index + 1) / times.length) * 65,
      `Sampling frame ${index + 1} of ${times.length}`,
    );
  }

  if (!frames.length) throw new Error("No video frames could be decoded for analysis.");
  return { frames, requestedCount };
}

function makeNoAudio(reason) {
  return {
    available: false,
    reason,
    rms: 0,
    rmsDb: -100,
    dynamics: 0,
    silenceRatio: 1,
    onsetRate: 0,
    speechLikelihood: 0,
    windows: [],
  };
}

async function decodeAudioFile(file, fallbackDuration) {
  const AudioContextConstructor = globalThis.AudioContext ?? globalThis.webkitAudioContext;
  if (typeof AudioContextConstructor !== "function") {
    return makeNoAudio("Web Audio decoding is unavailable in this browser.");
  }

  let context;
  try {
    const arrayBuffer = await withTimeout(
      file.arrayBuffer(),
      AUDIO_READ_TIMEOUT_MS,
      "Timed out while reading the audio track.",
    );
    context = new AudioContextConstructor();
    const audioBuffer = await withTimeout(
      context.decodeAudioData(arrayBuffer),
      AUDIO_DECODE_TIMEOUT_MS,
      "Timed out while decoding the audio track.",
    );

    if (!audioBuffer?.numberOfChannels || !audioBuffer.length) {
      return makeNoAudio("The file has no decodable audio samples.");
    }

    const duration = finiteNumber(audioBuffer.duration, fallbackDuration);
    const windowCount = Math.min(240, Math.max(16, Math.ceil(duration / 0.25)));
    const channelData = Array.from(
      { length: audioBuffer.numberOfChannels },
      (_, channel) => audioBuffer.getChannelData(channel),
    );
    const windows = [];
    let totalSquares = 0;
    let totalSamples = 0;
    let globalPeak = 0;

    for (let windowIndex = 0; windowIndex < windowCount; windowIndex += 1) {
      const start = Math.floor((windowIndex / windowCount) * audioBuffer.length);
      const end = Math.max(start + 1, Math.floor(((windowIndex + 1) / windowCount) * audioBuffer.length));
      const stride = Math.max(1, Math.floor((end - start) / 2048));
      let squares = 0;
      let samples = 0;
      let peak = 0;
      let zeroCrossings = 0;
      let previousSample = 0;

      for (let sampleIndex = start; sampleIndex < end; sampleIndex += stride) {
        let sample = 0;
        for (const channel of channelData) sample += channel[sampleIndex] ?? 0;
        sample /= channelData.length;
        squares += sample ** 2;
        peak = Math.max(peak, Math.abs(sample));
        if (samples > 0 && (sample >= 0) !== (previousSample >= 0)) zeroCrossings += 1;
        previousSample = sample;
        samples += 1;
      }

      const rms = samples ? Math.sqrt(squares / samples) : 0;
      windows.push({
        time: roundMetric(((windowIndex + 0.5) / windowCount) * duration, 3),
        rms,
        zcr: samples > 1 ? zeroCrossings / (samples - 1) : 0,
        onset: 0,
      });
      totalSquares += squares;
      totalSamples += samples;
      globalPeak = Math.max(globalPeak, peak);
    }

    const windowRms = windows.map((window) => window.rms);
    const windowDecibels = windowRms.map((rms) => 20 * Math.log10(Math.max(rms, 1e-5)));
    const dynamicRange = percentile(windowDecibels, 0.9) - percentile(windowDecibels, 0.1);
    const dynamics = clamp01(dynamicRange / 24);
    const silenceThreshold = Math.max(0.002, percentile(windowRms, 0.9) * 0.045);
    const silenceRatio = windows.filter((window) => window.rms <= silenceThreshold).length / windows.length;

    let onsetCount = 0;
    for (let index = 1; index < windows.length; index += 1) {
      const previous = windows[index - 1].rms;
      const deltaRatio = (windows[index].rms - previous) / Math.max(previous, 0.006);
      if (windows[index].rms > 0.008 && deltaRatio > 0.48) {
        windows[index].onset = 1;
        onsetCount += 1;
      }
    }

    const activeRatio = 1 - silenceRatio;
    const meanZcr = average(windows.filter((window) => window.rms > silenceThreshold).map((window) => window.zcr));
    const speechLikeZcr = scoreWithinBand(meanZcr, 0.025, 0.22, 0.025, 0.2);
    const speechLikeDynamics = scoreWithinBand(dynamics, 0.12, 0.7, 0.12, 0.3);
    const speechLikelihood = clamp01(
      speechLikeZcr * 0.38 + speechLikeDynamics * 0.28 + activeRatio * 0.34,
    );
    const rms = totalSamples ? Math.sqrt(totalSquares / totalSamples) : 0;

    return {
      available: true,
      reason: "",
      duration: roundMetric(duration, 3),
      sampleRate: audioBuffer.sampleRate,
      channels: audioBuffer.numberOfChannels,
      rms: roundMetric(rms, 6),
      rmsDb: roundMetric(20 * Math.log10(Math.max(rms, 1e-5)), 2),
      peak: roundMetric(globalPeak, 6),
      crestFactor: roundMetric(globalPeak / Math.max(rms, 1e-6), 3),
      dynamics: roundMetric(dynamics),
      silenceRatio: roundMetric(silenceRatio),
      onsetRate: roundMetric(onsetCount / Math.max(duration, 0.001), 3),
      speechLikelihood: roundMetric(speechLikelihood),
      windows: windows.map(({ time, rms: windowRmsValue, onset }) => ({
        time,
        rms: roundMetric(windowRmsValue, 6),
        onset,
      })),
    };
  } catch (error) {
    return makeNoAudio(error instanceof Error ? error.message : "The audio track could not be decoded.");
  } finally {
    if (context && typeof context.close === "function") {
      try {
        await context.close();
      } catch {
        // Closing is best-effort and does not affect the already-computed signals.
      }
    }
  }
}

/**
 * Analyze a browser File/Blob without uploading it. Progress is reported as
 * `{ stage, progress, message }`, where progress is a percentage from 0 to 100.
 */
export async function analyzeVideoFile(file, onProgress) {
  if (!file || typeof file.arrayBuffer !== "function") {
    throw new TypeError("analyzeVideoFile expects a video File or Blob.");
  }
  if (typeof document === "undefined" || typeof globalThis.URL?.createObjectURL !== "function") {
    throw new Error("Video analysis requires a browser with video, canvas, and object URL support.");
  }

  const video = document.createElement("video");
  video.preload = "auto";
  video.muted = true;
  video.playsInline = true;
  const objectUrl = globalThis.URL.createObjectURL(file);
  const warnings = [];

  try {
    reportProgress(onProgress, "metadata", 2, "Reading video metadata");
    video.src = objectUrl;
    video.load();
    await waitForMetadata(video);

    const duration = finiteNumber(video.duration);
    if (duration <= 0) throw new Error("The video duration is missing or invalid.");
    if (!video.videoWidth || !video.videoHeight) {
      throw new Error("The video does not expose a decodable picture track.");
    }

    reportProgress(onProgress, "frames", 10, "Preparing evenly distributed frame samples");
    const sampled = await sampleVideoFrames(video, onProgress, warnings);

    reportProgress(onProgress, "audio", 78, "Checking the audio track");
    const audio = await decodeAudioFile(file, duration);
    if (!audio.available && audio.reason) warnings.push(audio.reason);

    reportProgress(onProgress, "scores", 94, "Deriving explainable content scores");
    const signals = { duration, frames: sampled.frames, audio };
    const derived = deriveScores(signals);
    const metadata = {
      name: typeof file.name === "string" ? file.name : "video",
      fileName: typeof file.name === "string" ? file.name : "video",
      type: typeof file.type === "string" ? file.type : "",
      size: Math.max(0, finiteNumber(file.size)),
      duration: roundMetric(duration, 3),
      width: video.videoWidth,
      height: video.videoHeight,
      aspectRatio: roundMetric(video.videoWidth / video.videoHeight, 4),
      requestedFrameSamples: sampled.requestedCount,
      frameSampleCount: sampled.frames.length,
      audioAvailable: audio.available,
      audioDuration: audio.available ? audio.duration : 0,
    };

    reportProgress(onProgress, "complete", 100, "Analysis complete");
    return {
      metadata,
      ...derived,
      signals,
      warnings: [...new Set(warnings)],
    };
  } finally {
    video.pause();
    video.removeAttribute("src");
    video.load();
    globalThis.URL.revokeObjectURL(objectUrl);
  }
}
