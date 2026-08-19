import assert from "node:assert/strict";
import test from "node:test";

import { compareTribeVariantReports } from "../src/tribe/variantComparison.js";

function report(overrides = {}) {
  const base = {
    schemaVersion: "tribe-cortical-descriptors/1",
    source: {
      descriptorType: "descriptive cortical response",
      resultId: "1".repeat(32),
      predictionType: "average-subject cortical BOLD response",
      isViralityScore: false,
      behavioralPrediction: false,
      model: { id: "facebook/tribev2", revision: "rev", weightsSha256: "weights", codeRevision: "code" },
      extractor: { id: "facebook/vjepa2", revision: "video-rev", weightsSha256: "video-weights", device: "mps", precision: "autocast-fp16" },
      inferenceMode: "vision-only",
      surface: { space: "fsaverage5", mappingId: "mapping", vertexCount: 20_484 },
      atlas: { mappingSha256: "atlas" },
    },
    units: { response: "target z units" },
    clip: {
      responseMagnitude: 2,
      largestChange: { value: 3 },
      continuity: 0.5,
      spatialDistribution: 40,
    },
  };
  return {
    ...base,
    ...overrides,
    source: { ...base.source, ...overrides.source },
    units: { ...base.units, ...overrides.units },
    clip: { ...base.clip, ...overrides.clip },
  };
}

test("returns deterministic ordered descriptor differences without a winner", () => {
  const comparison = compareTribeVariantReports(report(), report({
    source: { resultId: "2".repeat(32) },
    clip: { responseMagnitude: 4, largestChange: { value: 1 }, continuity: 0.5, spatialDistribution: 50 },
  }));
  assert.deepEqual(comparison.rows.map((item) => item.id), [
    "response-magnitude", "pattern-change", "pattern-continuity", "spatial-distribution",
  ]);
  assert.deepEqual(comparison.rows.map((item) => item.direction), [
    "variant-higher", "primary-higher", "tie", "variant-higher",
  ]);
  assert.equal(comparison.rows[2].delta, 0);
  assert.equal(comparison.behavioralPrediction, false);
  assert.doesNotMatch(JSON.stringify(comparison.rows), /winner|uplift/i);
  assert.match(comparison.boundary, /randomized platform A\/B test/i);
  assert.equal(Object.isFrozen(comparison.rows[0]), true);
});

test("treats values within the documented relative comparison tolerance as tied", () => {
  const comparison = compareTribeVariantReports(report(), report({ clip: { responseMagnitude: 2 + 1e-7 } }));
  assert.equal(comparison.rows[0].direction, "tie");
  assert.equal(comparison.rows[0].delta, 0);
});

for (const [name, source, expected] of [
  ["revision", { model: { id: "facebook/tribev2", revision: "other", weightsSha256: "weights", codeRevision: "code" } }, /model revisions differ/],
  ["mode", { inferenceMode: "full" }, /inference modes differ/],
  ["surface", { surface: { space: "other", mappingId: "mapping", vertexCount: 20_484 } }, /surface spaces differ/],
  ["precision", { extractor: { id: "facebook/vjepa2", revision: "video-rev", weightsSha256: "video-weights", device: "mps", precision: "fp32" } }, /precision policies differ/],
]) {
  test(`rejects incompatible ${name}`, () => {
    assert.throws(() => compareTribeVariantReports(report(), report({ source })), expected);
  });
}
