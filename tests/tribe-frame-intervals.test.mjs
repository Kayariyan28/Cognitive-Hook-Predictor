import assert from "node:assert/strict";
import test from "node:test";
import { TRIBE_DESCRIPTOR_SCHEMA_VERSION } from "../src/tribe/descriptors.js";
import { rankTribeFrameIntervals } from "../src/tribe/frameIntervals.js";

function topParcel(index) {
  return {
    key: `${index % 2 ? "right" : "left"}:${index + 1}`,
    hemisphere: index % 2 ? "right" : "left",
    labelIndex: index + 1,
    name: `Parcel_${index}`,
    vertexCount: 20 + index,
    mean: index / 10,
    rms: index + 1,
  };
}

function descriptorReport() {
  const frames = [
    { index: 0, time: 0, endTime: 0.5, duration: 0.5, rms: 2, changeRms: null, changeRate: null, continuity: null, topParcel: topParcel(0) },
    { index: 1, time: 0.5, endTime: 1.5, duration: 1, rms: 4, changeRms: 3, changeRate: 3, continuity: 0.8, topParcel: topParcel(1) },
    { index: 2, time: 2, endTime: 2.25, duration: 0.25, rms: 4, changeRms: 1, changeRate: 2 / 3, continuity: -0.2, topParcel: topParcel(2) },
    { index: 3, time: 3, endTime: 5, duration: 2, rms: 1, changeRms: 3, changeRate: 3, continuity: null, topParcel: topParcel(3) },
  ];
  return {
    schemaVersion: TRIBE_DESCRIPTOR_SCHEMA_VERSION,
    displayScale: { domain: [-1, 0, 1] },
    frames,
    clip: {
      peakMagnitude: { index: 1, time: 0.5, duration: 1, value: 4 },
      largestChange: { index: 1, time: 0.5, duration: 1, value: 3 },
    },
  };
}

test("ranks response-magnitude intervals with tied competition ranks and exact authoritative times", () => {
  const report = descriptorReport();
  const intervals = rankTribeFrameIntervals(report);

  assert.deepEqual(intervals.map(({ index }) => index), [1, 2, 0, 3]);
  assert.deepEqual(intervals.map(({ rank }) => rank), [1, 1, 3, 4]);
  assert.deepEqual(intervals.map(({ selectedMetric }) => selectedMetric), [
    "responseMagnitude", "responseMagnitude", "responseMagnitude", "responseMagnitude",
  ]);
  assert.deepEqual(intervals.map(({ selectedValue }) => selectedValue), [4, 4, 2, 1]);
  assert.deepEqual(intervals.map(({ withinClipPercentile }) => withinClipPercentile), [
    100 * 2.5 / 3,
    100 * 2.5 / 3,
    100 / 3,
    0,
  ]);
  assert.deepEqual(intervals.map(({ withinClipBand }) => withinClipBand), [
    "higher-within-clip",
    "higher-within-clip",
    "middle-within-clip",
    "lower-within-clip",
  ]);
  assert.deepEqual(
    intervals.map(({ index, startTime, endTime, duration }) => [index, startTime, endTime, duration]),
    [[1, 0.5, 1.5, 1], [2, 2, 2.25, 0.25], [0, 0, 0.5, 0.5], [3, 3, 5, 2]],
  );
  assert.deepEqual(intervals[0].labels, ["Peak response magnitude", "Largest response change"]);
  assert.deepEqual(intervals[0].badges, { peakMagnitude: true, largestChange: true });
  assert.deepEqual(intervals[1].labels, []);
  assert.deepEqual(intervals[1].badges, { peakMagnitude: false, largestChange: false });
});

test("change mode includes the first interval as explicitly unavailable and sorts it last", () => {
  const intervals = rankTribeFrameIntervals(descriptorReport(), "change");

  assert.deepEqual(intervals.map(({ index }) => index), [1, 3, 2, 0]);
  assert.deepEqual(intervals.map(({ rank }) => rank), [1, 1, 3, null]);
  assert.deepEqual(intervals.map(({ selectedValue }) => selectedValue), [3, 3, 1, null]);
  assert.deepEqual(intervals.map(({ withinClipPercentile }) => withinClipPercentile), [75, 75, 0, null]);
  assert.equal(intervals[3].selectedMetric, "changeRms");
  assert.equal(intervals[3].selectedMetricAvailable, false);
  assert.equal(intervals[3].withinClipBand, null);
  assert.equal(intervals[3].changeRms, null);
  assert.equal(intervals[3].changeRate, null);
  assert.equal(intervals[3].continuity, null);
  assert.deepEqual(intervals[1].labels, []);
  assert.equal(intervals[1].badges.largestChange, false);
});

test("uses tie-aware midrank percentiles without splitting identical values", () => {
  const report = descriptorReport();
  report.frames = report.frames.slice(0, 2);
  report.frames[0] = { ...report.frames[0], rms: 4 };
  report.clip = {
    peakMagnitude: { index: 0, time: 0, duration: 0.5, value: 4 },
    largestChange: { index: 1, time: 0.5, duration: 1, value: 3 },
  };
  const tied = rankTribeFrameIntervals(report);
  assert.deepEqual(tied.map(({ index, rank, withinClipPercentile }) => [index, rank, withinClipPercentile]), [
    [0, 1, 50],
    [1, 1, 50],
  ]);

  const singleton = descriptorReport();
  singleton.frames = singleton.frames.slice(0, 1);
  singleton.clip = {
    peakMagnitude: { index: 0, time: 0, duration: 0.5, value: 2 },
    largestChange: null,
  };
  const [only] = rankTribeFrameIntervals(singleton);
  assert.equal(only.rank, 1);
  assert.equal(only.withinClipPercentile, null);
  assert.equal(only.withinClipBand, null);
});

test("does not mutate the report or depend on display scaling and freezes its output", () => {
  const report = descriptorReport();
  const snapshot = structuredClone(report);
  const before = rankTribeFrameIntervals(report);
  assert.deepEqual(report, snapshot);

  report.displayScale.domain = [-10_000, 100, 20_000];
  report.frames.forEach((frame) => { frame.displayScaledRms = 99_999; });
  const after = rankTribeFrameIntervals(report);
  assert.deepEqual(after, before);
  assert.equal(Object.isFrozen(after), true);
  assert.equal(Object.isFrozen(after[0]), true);
  assert.equal(Object.isFrozen(after[0].topParcel), true);
  assert.equal(Object.isFrozen(after[0].labels), true);
  assert.equal(Object.isFrozen(after[0].badges), true);
});

test("rejects unknown schemas, modes, inconsistent extrema, invalid values, and altered intervals", () => {
  const wrongSchema = descriptorReport();
  wrongSchema.schemaVersion = "unknown/1";
  assert.throws(() => rankTribeFrameIntervals(wrongSchema), /schemaVersion/);
  assert.throws(() => rankTribeFrameIntervals(descriptorReport(), "scaled"), /sortMode/);

  const wrongPeak = descriptorReport();
  wrongPeak.clip.peakMagnitude.index = 2;
  assert.throws(() => rankTribeFrameIntervals(wrongPeak), /peakMagnitude/);

  const notFinite = descriptorReport();
  notFinite.frames[2].rms = Number.NaN;
  assert.throws(() => rankTribeFrameIntervals(notFinite), /must be finite/);

  const alteredEnd = descriptorReport();
  alteredEnd.frames[1].endTime = 1.75;
  assert.throws(() => rankTribeFrameIntervals(alteredEnd), /time plus duration/);
});
