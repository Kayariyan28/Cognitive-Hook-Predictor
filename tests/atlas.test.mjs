import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  applyParcelFocus,
  DESTRIEUX_MAPPING_SHA256,
  summarizeFrameByParcel,
} from "../src/tribe/atlas.js";
import { TRIBE_VERTEX_COUNT } from "../src/tribe/contract.js";

test("bundled Destrieux atlas preserves the exact fsaverage5 vertex mapping and digest", async () => {
  const [bytes, metadata] = await Promise.all([
    readFile(new URL("../public/assets/brain/destrieux-fsaverage5.u8", import.meta.url)),
    readFile(new URL("../public/assets/brain/destrieux-fsaverage5.json", import.meta.url), "utf8").then(JSON.parse),
  ]);
  assert.equal(bytes.byteLength, TRIBE_VERTEX_COUNT);
  assert.equal(metadata.vertexCount, TRIBE_VERTEX_COUNT);
  assert.equal(metadata.verticesPerHemisphere, 10_242);
  assert.equal(metadata.labels.length, 76);
  assert.equal(createHash("sha256").update(bytes).digest("hex"), DESTRIEUX_MAPPING_SHA256);
  assert.equal(metadata.mappingSha256, DESTRIEUX_MAPPING_SHA256);
});

test("parcel summaries and focus use atlas vertex membership without hand-positioned regions", async () => {
  const [bytes, metadata] = await Promise.all([
    readFile(new URL("../public/assets/brain/destrieux-fsaverage5.u8", import.meta.url)),
    readFile(new URL("../public/assets/brain/destrieux-fsaverage5.json", import.meta.url), "utf8").then(JSON.parse),
  ]);
  const atlas = { parcelIds: new Uint8Array(bytes), metadata };
  const chosenVertex = atlas.parcelIds.findIndex((labelIndex, vertex) => vertex >= 10_242 && labelIndex > 0 && metadata.labels[labelIndex] !== "Medial_wall");
  assert.ok(chosenVertex >= 10_242);
  const chosenLabel = atlas.parcelIds[chosenVertex];
  const frame = new Float32Array(TRIBE_VERTEX_COUNT);
  for (let vertex = 10_242; vertex < TRIBE_VERTEX_COUNT; vertex += 1) {
    if (atlas.parcelIds[vertex] === chosenLabel) frame[vertex] = 2;
  }
  const [top] = summarizeFrameByParcel(frame, atlas, 1);
  assert.equal(top.hemisphere, "right");
  assert.equal(top.labelIndex, chosenLabel);
  assert.equal(top.mean, 2);

  const colors = new Float32Array(TRIBE_VERTEX_COUNT * 3).fill(1);
  applyParcelFocus(colors, atlas, top);
  assert.deepEqual([...colors.slice(chosenVertex * 3, chosenVertex * 3 + 3)], [1, 1, 1]);
  assert.ok(colors[0] < 0.25 && colors[0] > 0.23);
});
