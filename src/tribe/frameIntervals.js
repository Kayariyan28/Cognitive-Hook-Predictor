import { TRIBE_DESCRIPTOR_SCHEMA_VERSION } from "./descriptors.js";

const SORT_MODES = Object.freeze({
  magnitude: Object.freeze({ field: "responseMagnitude", sourceField: "responseMagnitude" }),
  change: Object.freeze({ field: "changeRms", sourceField: "changeRms" }),
});

function fail(message) {
  throw new Error(`Invalid TRIBE frame interval report: ${message}`);
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function finite(value, label) {
  if (!Number.isFinite(value)) fail(`${label} must be finite.`);
  return value;
}

function nullableFinite(value, label) {
  if (value === null) return null;
  return finite(value, label);
}

function nearlyEqual(left, right) {
  return Math.abs(left - right) <= 1e-12 * Math.max(1, Math.abs(left), Math.abs(right));
}

function freezeInterval(interval) {
  if (interval.topParcel) Object.freeze(interval.topParcel);
  Object.freeze(interval.labels);
  Object.freeze(interval.badges);
  return Object.freeze(interval);
}

function validateTopParcel(parcel, frameIndex) {
  if (!isRecord(parcel) || typeof parcel.key !== "string" || !parcel.key
      || !["left", "right"].includes(parcel.hemisphere)
      || !Number.isInteger(parcel.labelIndex)
      || typeof parcel.name !== "string"
      || !Number.isInteger(parcel.vertexCount) || parcel.vertexCount < 1) {
    fail(`frames[${frameIndex}].topParcel is invalid.`);
  }
  finite(parcel.mean, `frames[${frameIndex}].topParcel.mean`);
  const rms = finite(parcel.rms, `frames[${frameIndex}].topParcel.rms`);
  if (rms < 0) fail(`frames[${frameIndex}].topParcel.rms must not be negative.`);
  return { ...parcel };
}

function validateReport(report) {
  if (!isRecord(report) || report.schemaVersion !== TRIBE_DESCRIPTOR_SCHEMA_VERSION) {
    fail(`schemaVersion must be ${TRIBE_DESCRIPTOR_SCHEMA_VERSION}.`);
  }
  if (!Array.isArray(report.frames) || report.frames.length < 1 || !isRecord(report.clip)) {
    fail("frames and clip descriptors are required.");
  }

  let previousStart = -Infinity;
  let previousEnd = -Infinity;
  const frames = report.frames.map((frame, position) => {
    if (!isRecord(frame) || frame.index !== position) fail(`frames[${position}].index must preserve authoritative tensor order.`);
    const startTime = finite(frame.time, `frames[${position}].time`);
    const endTime = finite(frame.endTime, `frames[${position}].endTime`);
    const duration = finite(frame.duration, `frames[${position}].duration`);
    if (duration <= 0) fail(`frames[${position}].duration must be positive.`);
    if (startTime <= previousStart) fail("frame start times must be strictly increasing.");
    if (startTime < previousEnd - Number.EPSILON) fail("frame intervals must not overlap.");
    if (!nearlyEqual(endTime, startTime + duration)) fail(`frames[${position}].endTime must equal time plus duration.`);

    const responseMagnitude = finite(frame.rms, `frames[${position}].rms`);
    if (responseMagnitude < 0) fail(`frames[${position}].rms must not be negative.`);
    const changeRms = nullableFinite(frame.changeRms, `frames[${position}].changeRms`);
    const changeRate = nullableFinite(frame.changeRate, `frames[${position}].changeRate`);
    const continuity = nullableFinite(frame.continuity, `frames[${position}].continuity`);
    if (changeRms !== null && changeRms < 0) fail(`frames[${position}].changeRms must not be negative.`);
    if (changeRate !== null && changeRate < 0) fail(`frames[${position}].changeRate must not be negative.`);
    if (continuity !== null && (continuity < -1 || continuity > 1)) fail(`frames[${position}].continuity must be between -1 and 1.`);
    if (position === 0 && (changeRms !== null || changeRate !== null || continuity !== null)) {
      fail("the first frame must not invent an adjacent-frame descriptor.");
    }
    if (position > 0 && (changeRms === null || changeRate === null)) {
      fail(`frames[${position}] must include its adjacent change and change rate.`);
    }

    previousStart = startTime;
    previousEnd = endTime;
    return {
      index: frame.index,
      startTime,
      endTime,
      duration,
      responseMagnitude,
      changeRms,
      changeRate,
      continuity,
      topParcel: validateTopParcel(frame.topParcel, position),
    };
  });

  const expectedPeak = frames.reduce((best, frame) => frame.responseMagnitude > best.responseMagnitude ? frame : best);
  const declaredPeak = report.clip.peakMagnitude;
  if (!isRecord(declaredPeak) || declaredPeak.index !== expectedPeak.index
      || !Number.isFinite(declaredPeak.value) || !nearlyEqual(declaredPeak.value, expectedPeak.responseMagnitude)) {
    fail("clip.peakMagnitude does not match the earliest maximum frame RMS.");
  }

  const changeFrames = frames.filter((frame) => frame.changeRms !== null);
  const expectedChange = changeFrames.length
    ? changeFrames.reduce((best, frame) => frame.changeRms > best.changeRms ? frame : best)
    : null;
  const declaredChange = report.clip.largestChange;
  if (expectedChange === null) {
    if (declaredChange !== null) fail("clip.largestChange must be null when no transition exists.");
  } else if (!isRecord(declaredChange) || declaredChange.index !== expectedChange.index
      || !Number.isFinite(declaredChange.value) || !nearlyEqual(declaredChange.value, expectedChange.changeRms)) {
    fail("clip.largestChange does not match the earliest maximum adjacent change.");
  }

  return {
    frames,
    peakIndex: expectedPeak.index,
    largestChangeIndex: expectedChange?.index ?? null,
  };
}

function percentileRank(value, availableValues) {
  if (value === null) return null;
  if (availableValues.length === 1) return null;
  let below = 0;
  let equal = 0;
  for (const candidate of availableValues) {
    if (candidate < value) below += 1;
    else if (candidate === value) equal += 1;
  }
  return 100 * (below + (equal - 1) / 2) / (availableValues.length - 1);
}

function competitionRank(value, availableValues) {
  if (value === null) return null;
  return 1 + availableValues.filter((candidate) => candidate > value).length;
}

function withinClipBand(percentile) {
  if (percentile === null) return null;
  if (percentile >= 75) return "higher-within-clip";
  if (percentile < 25) return "lower-within-clip";
  return "middle-within-clip";
}

/**
 * Ranks every authoritative TRIBE prediction interval using only raw
 * descriptive tensor statistics. No behavioral outcome is inferred.
 */
export function rankTribeFrameIntervals(report, sortMode = "magnitude") {
  const mode = SORT_MODES[sortMode];
  if (!mode) fail("sortMode must be magnitude or change.");
  const { frames, peakIndex, largestChangeIndex } = validateReport(report);
  const availableValues = frames
    .map((frame) => frame[mode.sourceField])
    .filter((value) => value !== null);

  const intervals = frames.map((frame) => {
    const selectedValue = frame[mode.sourceField];
    const peakMagnitude = frame.index === peakIndex;
    const largestChange = frame.index === largestChangeIndex;
    const labels = [];
    if (peakMagnitude) labels.push("Peak response magnitude");
    if (largestChange) labels.push("Largest response change");
    const withinClipPercentile = percentileRank(selectedValue, availableValues);
    return {
      ...frame,
      rank: competitionRank(selectedValue, availableValues),
      withinClipPercentile,
      withinClipBand: withinClipBand(withinClipPercentile),
      selectedMetric: mode.field,
      selectedValue,
      selectedMetricAvailable: selectedValue !== null,
      labels,
      badges: { peakMagnitude, largestChange },
    };
  });

  intervals.sort((left, right) => {
    if (left.selectedValue === null) return right.selectedValue === null ? left.index - right.index : 1;
    if (right.selectedValue === null) return -1;
    return right.selectedValue - left.selectedValue || left.index - right.index;
  });
  return Object.freeze(intervals.map(freezeInterval));
}
