import assert from "node:assert/strict";
import test from "node:test";
import { TRIBE_VERTEX_COUNT } from "../src/tribe/contract.js";
import {
  decodePredictionBuffer,
  frameIndexAtMediaTime,
  predictionFrameAtMediaTime,
  splitHemisphereFrame,
  writeNeutralColors,
  writeScientificColors,
} from "../src/tribe/frame.js";

const manifest = {
  frames: {
    byteLength: TRIBE_VERTEX_COUNT * 2 * 4,
    timesSeconds: [1, 3],
    durationsSeconds: [1, 1],
    displayScale: { domain: [-1, 0, 1] },
  },
};

test("decodes time-major float32 frames and selects only authoritative timestamps", () => {
  const source = new Float32Array(TRIBE_VERTEX_COUNT * 2);
  source[0] = 0.25;
  source[TRIBE_VERTEX_COUNT] = 0.75;
  const values = decodePredictionBuffer(source.buffer, manifest);

  assert.equal(frameIndexAtMediaTime(0.99, manifest.frames.timesSeconds), null);
  assert.equal(frameIndexAtMediaTime(1, manifest.frames.timesSeconds, manifest.frames.durationsSeconds), 0);
  assert.equal(frameIndexAtMediaTime(2.9, manifest.frames.timesSeconds, manifest.frames.durationsSeconds), null);
  assert.equal(frameIndexAtMediaTime(3, manifest.frames.timesSeconds, manifest.frames.durationsSeconds), 1);
  assert.equal(predictionFrameAtMediaTime(values, manifest, 3).frame[0], 0.75);
});

test("splits hemispheres at exactly 10,242 and writes one color per original vertex", () => {
  const frame = new Float32Array(TRIBE_VERTEX_COUNT);
  frame[0] = -1;
  frame[10_242] = 1;
  const split = splitHemisphereFrame(frame);
  assert.equal(split.left.length, 10_242);
  assert.equal(split.right.length, 10_242);
  assert.equal(split.left[0], -1);
  assert.equal(split.right[0], 1);

  const colors = writeNeutralColors(new Float32Array(TRIBE_VERTEX_COUNT * 3));
  writeScientificColors(frame, [-1, 0, 1], colors);
  assert.notDeepEqual([...colors.slice(0, 3)], [...colors.slice(3, 6)]);
  assert.notDeepEqual([...colors.slice(10_242 * 3, 10_242 * 3 + 3)], [...colors.slice(3, 6)]);
});

test("rejects non-finite prediction values", () => {
  const source = new Float32Array(TRIBE_VERTEX_COUNT * 2);
  source[99] = Number.NaN;
  assert.throws(() => decodePredictionBuffer(source.buffer, manifest), /non-finite/);
});
