import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { TRIBE_MESH_MAPPING_ID, TRIBE_VERTEX_COUNT } from "../src/tribe/contract.js";
import {
  FSAVERAGE5_MESH_METADATA_URL,
  FSAVERAGE5_MESH_SHA256,
  loadFsaverage5Mesh,
  parseFsaverage5Mesh,
  sha256Hex,
  validateFsaverage5MeshMetadata,
} from "../src/tribe/meshLoader.js";

const metadataPath = new URL("../public/assets/brain/fsaverage5-half.json", import.meta.url);
const binaryPath = new URL("../public/assets/brain/fsaverage5-half.bin", import.meta.url);

async function loadFixture() {
  const [metadataText, binary] = await Promise.all([
    readFile(metadataPath, "utf8"),
    readFile(binaryPath),
  ]);
  const buffer = binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength);
  return { metadata: JSON.parse(metadataText), buffer };
}

test("runtime metadata and SFB1 binary lock the exact TRIBE fsaverage5 mapping", async () => {
  const { metadata, buffer } = await loadFixture();
  const validated = validateFsaverage5MeshMetadata(metadata);
  const mesh = parseFsaverage5Mesh(buffer);

  assert.equal(FSAVERAGE5_MESH_METADATA_URL, "/assets/brain/fsaverage5-half.json");
  assert.equal(validated.mappingId, TRIBE_MESH_MAPPING_ID);
  assert.equal(validated.binarySha256, FSAVERAGE5_MESH_SHA256);
  assert.equal(await sha256Hex(buffer), FSAVERAGE5_MESH_SHA256);
  assert.equal(buffer.byteLength, validated.byteLength);
  assert.equal(mesh.vertexCount, TRIBE_VERTEX_COUNT);
  assert.equal(mesh.leftVertexCount, 10_242);
  assert.equal(mesh.rightVertexCount, 10_242);
  assert.equal(mesh.triangleCount, 40_960);
  assert.equal(mesh.positions.length, TRIBE_VERTEX_COUNT * 3);
  assert.equal(mesh.indices.length, 40_960 * 3);
  assert.equal(mesh.indices.every((index) => index < TRIBE_VERTEX_COUNT), true);
  assert.equal(mesh.positions.every(Number.isFinite), true);

  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let index = 0; index < mesh.positions.length; index += 3) {
    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], mesh.positions[index + axis]);
      max[axis] = Math.max(max[axis], mesh.positions[index + axis]);
    }
  }
  assert.deepEqual(min, validated.bounds.min);
  assert.deepEqual(max, validated.bounds.max);
});

test("loader fetches physical metadata first and verifies the binary before parsing", async () => {
  const { metadata, buffer } = await loadFixture();
  const requested = [];
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async (url) => {
    requested.push(String(url));
    if (String(url) === FSAVERAGE5_MESH_METADATA_URL) {
      return {
        ok: true,
        status: 200,
        url: "http://localhost/assets/brain/fsaverage5-half.json",
        json: async () => metadata,
      };
    }
    if (String(url) === "http://localhost/assets/brain/fsaverage5-half.bin") {
      return { ok: true, status: 200, arrayBuffer: async () => buffer.slice(0) };
    }
    return { ok: false, status: 404 };
  };

  try {
    const mesh = await loadFsaverage5Mesh();
    assert.deepEqual(requested, [
      "/assets/brain/fsaverage5-half.json",
      "http://localhost/assets/brain/fsaverage5-half.bin",
    ]);
    assert.equal(mesh.vertexCount, TRIBE_VERTEX_COUNT);
    assert.equal(mesh.metadata.mappingId, TRIBE_MESH_MAPPING_ID);
  } finally {
    globalThis.fetch = previousFetch;
  }
});

test("metadata and binary corruption fail closed", async () => {
  const { metadata, buffer } = await loadFixture();
  assert.throws(
    () => validateFsaverage5MeshMetadata({ ...metadata, binarySha256: "0".repeat(64) }),
    /audited cortical mapping/,
  );
  assert.throws(
    () => validateFsaverage5MeshMetadata({ ...metadata, ordering: ["right", "left"] }),
    /vertex ordering/,
  );

  const badMagic = buffer.slice(0);
  new Uint8Array(badMagic, 0, 4).set([0, 0, 0, 0]);
  assert.throws(() => parseFsaverage5Mesh(badMagic), /Unsupported fsaverage5 mesh format/);

  const badIndex = buffer.slice(0);
  const indexOffset = new DataView(badIndex).getUint32(20, true);
  new DataView(badIndex).setUint32(indexOffset, TRIBE_VERTEX_COUNT, true);
  assert.throws(() => parseFsaverage5Mesh(badIndex), /out-of-range index/);
});
