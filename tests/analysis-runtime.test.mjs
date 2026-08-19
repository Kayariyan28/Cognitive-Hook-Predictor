import assert from "node:assert/strict";
import test from "node:test";
import { deriveScores } from "../src/analysis.js";

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

function makeFrames(count, factory) {
  return Array.from({ length: count }, (_, index) => ({
    time: index,
    ...factory(index),
  }));
}

const engagingSignals = {
  duration: 23,
  frames: makeFrames(24, (index) => ({
    luminance: 0.42 + (index % 5) * 0.045,
    contrast: 0.23 + (index % 3) * 0.025,
    colorfulness: 0.38 + (index % 4) * 0.035,
    edgeDensity: 0.16 + (index % 3) * 0.025,
    motion: index === 0 ? 0 : 0.11 + (index % 4) * 0.025,
    cut: index > 0 && index % 5 === 0 ? 0.82 : 0.035,
  })),
  audio: {
    available: true,
    rms: 0.11,
    dynamics: 0.42,
    silenceRatio: 0.08,
    onsetRate: 1.7,
    speechLikelihood: 0.76,
    windows: makeFrames(24, (index) => ({
      rms: 0.075 + (index % 4) * 0.018,
      onset: index % 4 === 0 ? 1 : 0,
    })),
  },
};

const flatSignals = {
  duration: 23,
  frames: makeFrames(24, () => ({
    luminance: 0.5,
    contrast: 0.018,
    colorfulness: 0.015,
    edgeDensity: 0,
    motion: 0,
    cut: 0,
  })),
  audio: {
    available: true,
    rms: 0.0001,
    dynamics: 0,
    silenceRatio: 1,
    onsetRate: 0,
    speechLikelihood: 0,
    windows: [],
  },
};

test("deriveScores is deterministic and returns the documented bounded score contract", () => {
  const first = deriveScores(engagingSignals);
  const second = deriveScores(structuredClone(engagingSignals));

  assert.deepEqual(first, second);
  assert.deepEqual(Object.keys(first.scores), SCORE_KEYS);
  for (const score of Object.values(first.scores)) {
    assert.equal(Number.isInteger(score), true);
    assert.ok(score >= 0 && score <= 100);
  }
  assert.equal(first.timeline.length, engagingSignals.frames.length);
  assert.match(first.methodology, /not a probability/i);
  assert.equal(first.engagement.score, first.scores.engagementScore);
  assert.equal(first.engagement.confidence, first.scores.coverage);
  assert.equal(first.engagement.coverage, first.scores.coverage);
  assert.equal(first.engagement.confidenceKind, "input-coverage-only");
  assert.match(first.engagement.methodology, /hand-weighted content-feature proxy/i);
  assert.match(first.engagement.methodology, /confidence reflects input coverage only/i);
  assert.match(first.engagement.methodology, /not measured retention/i);
  assert.match(first.engagement.methodology, /not .+predicted watch-time seconds/i);
  assert.match(first.engagement.methodology, /model certainty/i);
  for (const component of Object.values(first.engagement.components)) {
    if (component === null) continue;
    assert.equal(Number.isInteger(component), true);
    assert.ok(component >= 0 && component <= 100);
  }
  assert.equal(first.virality.label, "Pre-publish virality outlook");
  assert.equal(first.virality.scoreKind, "deterministic-content-feature-index");
  assert.equal(first.virality.calibration, "none");
  assert.equal(first.virality.behavioralOutcome, false);
  assert.equal(first.virality.tribeContribution, false);
  assert.equal(first.virality.formulaVersion, "local-content-outlook-v1");
  assert.equal(first.virality.evidenceCoverage, first.scores.coverage);
  assert.equal(Number.isInteger(first.virality.score), true);
  assert.ok(first.virality.score >= 0 && first.virality.score <= 100);
  assert.ok(first.strengths.length > 0);
  assert.ok(first.improvements.length > 0);
});

test("virality outlook exposes a transparent weighted formula", () => {
  const { virality } = deriveScores(engagingSignals);
  const expectedBaseWeights = {
    opening: 0.24,
    sustain: 0.24,
    pacing: 0.14,
    ending: 0.1,
    visualClarity: 0.12,
    visualStability: 0.08,
    audioSupport: 0.08,
  };

  assert.deepEqual(
    Object.fromEntries(
      Object.entries(virality.components).map(([key, component]) => [key, component.baseWeight]),
    ),
    expectedBaseWeights,
  );
  assert.ok(Object.values(virality.components).every((component) => component.sufficient));
  assert.ok(
    Math.abs(Object.values(virality.components).reduce((sum, component) => sum + component.weight, 0) - 1)
      < 0.000001,
  );
  const recomputed = Math.round(
    Object.values(virality.components).reduce(
      (sum, component) => sum + component.score * component.weight,
      0,
    ),
  );
  assert.equal(virality.score, recomputed);
});

test("active, varied signals score above a flat silent sample without claiming a probability", () => {
  const engaging = deriveScores(engagingSignals);
  const flat = deriveScores(flatSignals);

  assert.ok(engaging.scores.viralPotential > flat.scores.viralPotential);
  assert.ok(engaging.scores.engagementScore > flat.scores.engagementScore);
  assert.ok(engaging.scores.hookScore > flat.scores.hookScore);
  assert.ok(engaging.scores.visualSignal > flat.scores.visualSignal);
  assert.ok(engaging.scores.audioSignal > flat.scores.audioSignal);
});

test("missing audio is graceful, neutral in the audio score, and lowers coverage", () => {
  const withoutAudio = deriveScores({
    ...engagingSignals,
    audio: { available: false, reason: "No audio track" },
  });
  const withAudio = deriveScores(engagingSignals);

  assert.equal(withoutAudio.scores.audioSignal, 50);
  assert.equal(withoutAudio.scores.speechLikelihood, 0);
  assert.equal(withoutAudio.engagement.components.audioDynamics, null);
  assert.equal(withoutAudio.virality.components.audioSupport.score, null);
  assert.equal(withoutAudio.virality.components.audioSupport.weight, 0);
  assert.equal(withoutAudio.virality.components.audioSupport.available, false);
  assert.equal(withoutAudio.virality.components.audioSupport.sufficient, false);
  assert.ok(
    Math.abs(
      Object.values(withoutAudio.virality.components)
        .reduce((sum, component) => sum + component.weight, 0) - 1,
    ) < 0.00001,
  );
  assert.match(withoutAudio.engagement.methodology, /audio was unavailable/i);
  assert.match(withoutAudio.virality.methodology, /weights are redistributed/i);
  assert.ok(withoutAudio.scores.coverage < withAudio.scores.coverage);
  assert.ok(Math.abs(withoutAudio.scores.engagementScore - withAudio.scores.engagementScore) < 25);
  assert.ok(withoutAudio.improvements.some((item) => /audio track/i.test(item)));
  for (const score of Object.values(withoutAudio.scores)) assert.ok(Number.isFinite(score));
});

test("virality sustain is withheld when post-opening sampling is insufficient", () => {
  const result = deriveScores({
    duration: 2,
    frames: makeFrames(2, () => ({
      luminance: 0.5,
      contrast: 0.2,
      colorfulness: 0.3,
      edgeDensity: 0.12,
      motion: 0.08,
      cut: 0.02,
    })),
    audio: { available: false },
  });

  assert.equal(result.virality.components.sustain.sampleCount, 1);
  assert.equal(result.virality.components.sustain.score, null);
  assert.equal(result.virality.components.sustain.weight, 0);
  assert.equal(result.virality.components.sustain.sufficient, false);
  assert.match(result.virality.components.sustain.reason, /at least two samples/i);
  assert.ok(Number.isInteger(result.virality.score));
});

test("virality outlook ignores TRIBE-shaped extra input and states its evidence boundary", () => {
  const withoutTribe = deriveScores(engagingSignals);
  const withTribe = deriveScores({
    ...engagingSignals,
    tribe: {
      vertexCount: 20_484,
      values: [999, -999],
      model: "TRIBE v2",
    },
  });

  assert.deepEqual(withTribe.virality, withoutTribe.virality);
  assert.equal(withTribe.virality.tribeContribution, false);
  assert.match(withTribe.virality.interpretation, /do not mean.+probability/i);
  assert.match(withTribe.virality.methodology, /not trained or calibrated/i);
  assert.match(withTribe.virality.methodology, /views, shares, retention, or watch time/i);
  assert.match(withTribe.virality.methodology, /TRIBE v2 does not contribute/i);
  for (const message of [...withTribe.strengths, ...withTribe.improvements]) {
    assert.doesNotMatch(message, /attention|comparatively engaging|refresh attention/i);
  }
});

test("a stronger measured opening raises early attention and the engagement score", () => {
  const weakOpening = deriveScores({
    ...engagingSignals,
    frames: engagingSignals.frames.map((frame, index) => index <= 3
      ? {
          ...frame,
          contrast: 0.012,
          colorfulness: 0.01,
          edgeDensity: 0,
          motion: 0,
          cut: 0,
        }
      : frame),
  });
  const strongOpening = deriveScores(engagingSignals);

  assert.ok(
    strongOpening.engagement.components.earlyAttention
      > weakOpening.engagement.components.earlyAttention,
  );
  assert.ok(strongOpening.scores.engagementScore > weakOpening.scores.engagementScore);
});

test("purposeful motion, edit cadence, and a sustained close improve engagement components", () => {
  const staticClip = deriveScores({
    duration: 11,
    frames: makeFrames(12, () => ({
      luminance: 0.5,
      contrast: 0.2,
      colorfulness: 0.28,
      edgeDensity: 0.14,
      motion: 0,
      cut: 0,
    })),
    audio: { available: false },
  });
  const pacedClip = deriveScores({
    duration: 11,
    frames: makeFrames(12, (index) => ({
      luminance: 0.46 + (index % 3) * 0.025,
      contrast: 0.24 + (index % 2) * 0.025,
      colorfulness: 0.36 + (index % 3) * 0.025,
      edgeDensity: 0.16 + (index % 2) * 0.02,
      motion: index === 0 ? 0 : 0.13,
      cut: index > 0 && index % 4 === 0 ? 0.7 : 0.03,
    })),
    audio: { available: false },
  });

  assert.ok(pacedClip.engagement.components.pacing > staticClip.engagement.components.pacing);
  assert.ok(
    pacedClip.engagement.components.sustainedAttention
      > staticClip.engagement.components.sustainedAttention,
  );
  assert.ok(
    pacedClip.engagement.components.endingStrength
      > staticClip.engagement.components.endingStrength,
  );
  assert.ok(pacedClip.scores.engagementScore > staticClip.scores.engagementScore);
});

test("timeline is sorted and peakTime selects the earliest highest point", () => {
  const result = deriveScores({
    duration: 4,
    frames: [
      { time: 4, luminance: 0.5, contrast: 0.02, colorfulness: 0.02, edgeDensity: 0, motion: 0, cut: 0 },
      { time: 0, luminance: 0.5, contrast: 0.3, colorfulness: 0.45, edgeDensity: 0.2, motion: 0.16, cut: 0.6 },
      { time: 2, luminance: 0.5, contrast: 0.3, colorfulness: 0.45, edgeDensity: 0.2, motion: 0.16, cut: 0.6 },
    ],
    audio: { available: false },
  });

  assert.deepEqual(result.timeline.map((point) => point.time), [0, 2, 4]);
  assert.equal(result.timeline[0].score, result.timeline[1].score);
  assert.equal(result.peakTime, 0);
});

test("invalid and extreme pure inputs are sanitized instead of producing NaN", () => {
  const result = deriveScores({
    duration: Number.POSITIVE_INFINITY,
    frames: [
      {
        time: -10,
        luminance: Number.NaN,
        contrast: 9,
        colorfulness: -5,
        edgeDensity: Number.POSITIVE_INFINITY,
        motion: 3,
        cut: -2,
      },
    ],
    audio: {
      available: true,
      rms: Number.NaN,
      dynamics: 4,
      silenceRatio: -2,
      onsetRate: Number.POSITIVE_INFINITY,
      speechLikelihood: 8,
    },
  });

  for (const score of Object.values(result.scores)) {
    assert.equal(Number.isFinite(score), true);
    assert.ok(score >= 0 && score <= 100);
  }
  assert.deepEqual(result.timeline.map((point) => point.time), [0]);
  assert.equal(result.peakTime, 0);
});
