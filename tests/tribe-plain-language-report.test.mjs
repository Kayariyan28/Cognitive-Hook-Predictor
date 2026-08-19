import assert from "node:assert/strict";
import test from "node:test";
import { TRIBE_DESCRIPTOR_SCHEMA_VERSION } from "../src/tribe/descriptors.js";
import {
  createTribePlainLanguageReport,
  TRIBE_PLAIN_LANGUAGE_SCHEMA_VERSION,
} from "../src/tribe/plainLanguageReport.js";

function descriptorReport(overrides = {}) {
  const frames = [
    { index: 0, time: 0, duration: 3, endTime: 3, rms: 1, changeRms: null, continuity: null },
    { index: 1, time: 3, duration: 3, endTime: 6, rms: 2, changeRms: 0.5, continuity: 0.2 },
    { index: 2, time: 6, duration: 3, endTime: 9, rms: 3, changeRms: 1.5, continuity: 0.1 },
  ];
  return {
    schemaVersion: TRIBE_DESCRIPTOR_SCHEMA_VERSION,
    source: {
      descriptorType: "descriptive cortical response",
      predictionType: "average-subject cortical BOLD response",
      isViralityScore: false,
      behavioralPrediction: false,
      inferenceMode: "vision-only",
      model: { id: "facebook/tribev2" },
    },
    frames,
    clip: {
      peakMagnitude: { index: 2, value: 3 },
      largestChange: { index: 2, time: 6, value: 1.5 },
      continuity: 0.15,
      spatialDistribution: 22,
    },
    phases: [
      { id: "early", startTime: 0, endTime: 3, responseMagnitude: 1 },
      { id: "middle", startTime: 3, endTime: 6, responseMagnitude: 2 },
      { id: "late", startTime: 6, endTime: 9, responseMagnitude: 3 },
    ],
    ...overrides,
  };
}

function visibleCopy(report) {
  return [
    report.title,
    report.introduction,
    report.timing.title,
    report.timing.body,
    report.dynamics.title,
    report.dynamics.body,
    report.spread.title,
    report.spread.body,
    report.experiment.title,
    report.experiment.body,
    report.modality.title,
    report.modality.body,
    report.boundary,
  ].join(" ");
}

test("translates verified descriptors into deterministic creator language without numeric output", () => {
  const first = createTribePlainLanguageReport(descriptorReport());
  const second = createTribePlainLanguageReport(descriptorReport());

  assert.deepEqual(first, second);
  assert.equal(first.schemaVersion, TRIBE_PLAIN_LANGUAGE_SCHEMA_VERSION);
  assert.equal(first.timing.kind, "late");
  assert.equal(first.dynamics.kind, "changing");
  assert.equal(first.spread.kind, "concentrated");
  assert.equal(first.experiment.kind, "preview-change");
  assert.equal(first.modality.kind, "vision-only");
  assert.match(first.experiment.body, /observed platform results/);
  assert.match(first.modality.body, /audio or text input/);
  assert.doesNotMatch(visibleCopy(first), /[\d%]/);
  assert.equal(Object.isFrozen(first), true);
  assert.equal(Object.isFrozen(first.timing), true);
});

test("describes even timing, steady dynamics, broad spread, and an opening change without inventing functions", () => {
  const input = descriptorReport();
  input.phases = input.phases.map((phase) => ({ ...phase, responseMagnitude: 2 }));
  input.frames[1] = { ...input.frames[1], changeRms: 2 };
  input.frames[2] = { ...input.frames[2], changeRms: 1 };
  input.clip = {
    ...input.clip,
    largestChange: { index: 1, time: 3, value: 2 },
    continuity: 0.9,
    spatialDistribution: 80,
  };
  input.phases = [
    { ...input.phases[0], endTime: 4 },
    { ...input.phases[1], startTime: 4, endTime: 7 },
    { ...input.phases[2], startTime: 7 },
  ];

  const result = createTribePlainLanguageReport(input);
  assert.equal(result.timing.kind, "even");
  assert.equal(result.dynamics.kind, "steady");
  assert.equal(result.spread.kind, "broad");
  assert.equal(result.experiment.kind, "hold-opening");
  for (const forbidden of ["visual cortex", "auditory cortex", "emotion", "memory", "importance"] ) {
    assert.equal(visibleCopy(result).toLowerCase().includes(forbidden), false);
  }
});

test("makes modality coverage explicit for non-video-only results", () => {
  const input = descriptorReport();
  input.source = { ...input.source, inferenceMode: "audio-video-text" };
  const result = createTribePlainLanguageReport(input);
  assert.equal(result.modality.kind, "declared-inputs");
  assert.match(result.modality.body, /input coverage recorded by the verified model result/);
});

test("rejects behavioral sources, invalid phases, malformed moments, and non-finite descriptors", () => {
  const behavioral = descriptorReport();
  behavioral.source = { ...behavioral.source, behavioralPrediction: true };
  assert.throws(() => createTribePlainLanguageReport(behavioral), /non-behavioral/);

  const phases = descriptorReport();
  phases.phases = [phases.phases[1], phases.phases[0], phases.phases[2]];
  assert.throws(() => createTribePlainLanguageReport(phases), /opening-to-ending order/);

  const moment = descriptorReport();
  moment.clip = { ...moment.clip, peakMagnitude: { index: 1, value: 999 } };
  assert.throws(() => createTribePlainLanguageReport(moment), /referenced interval/);

  const consistentButWrongMoment = descriptorReport();
  consistentButWrongMoment.clip = { ...consistentButWrongMoment.clip, peakMagnitude: { index: 1, value: 2 } };
  assert.throws(() => createTribePlainLanguageReport(consistentButWrongMoment), /earliest maximum response/);

  const nonFinite = descriptorReport();
  nonFinite.clip = { ...nonFinite.clip, spatialDistribution: Number.NaN };
  assert.throws(() => createTribePlainLanguageReport(nonFinite), /must be finite/);
});
