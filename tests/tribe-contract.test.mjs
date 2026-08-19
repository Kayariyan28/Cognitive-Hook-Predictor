import assert from "node:assert/strict";
import test from "node:test";
import {
  TRIBE_MESH_MAPPING_ID,
  TRIBE_CODE_REVISION,
  TRIBE_MODEL_REVISION,
  TRIBE_MODEL_WEIGHTS_SHA256,
  TRIBE_VIDEO_MODEL_ID,
  TRIBE_VIDEO_MODEL_REVISION,
  TRIBE_VIDEO_MODEL_WEIGHTS_SHA256,
  TRIBE_VERTEX_COUNT,
  validateTribeManifest,
} from "../src/tribe/contract.js";

function manifest(overrides = {}) {
  const frameCount = 2;
  return {
    schemaVersion: "tribe-fsaverage5/1",
    resultId: "1".repeat(32),
    predictionType: "average-subject cortical BOLD response",
    isViralityScore: false,
    inferenceMode: "vision-only",
    modalitiesUsed: ["video"],
    missingModalities: ["audio", "text"],
    model: { id: "facebook/tribev2", revision: TRIBE_MODEL_REVISION, weightsSha256: TRIBE_MODEL_WEIGHTS_SHA256, codeRevision: TRIBE_CODE_REVISION, license: "CC-BY-NC-4.0" },
    extractors: { video: { id: TRIBE_VIDEO_MODEL_ID, revision: TRIBE_VIDEO_MODEL_REVISION, weightsSha256: TRIBE_VIDEO_MODEL_WEIGHTS_SHA256 } },
    surface: {
      space: "fsaverage5",
      mappingId: TRIBE_MESH_MAPPING_ID,
      vertexCount: TRIBE_VERTEX_COUNT,
      hemispheres: [
        { id: "left", offset: 0, count: 10_242 },
        { id: "right", offset: 10_242, count: 10_242 },
      ],
    },
    frames: {
      url: "/frames.f32",
      dtype: "float32",
      byteOrder: "little-endian",
      layout: "time-major",
      shape: [frameCount, TRIBE_VERTEX_COUNT],
      byteLength: frameCount * TRIBE_VERTEX_COUNT * 4,
      sha256: "a".repeat(64),
      timesSeconds: [0, 1],
      durationsSeconds: [1, 1],
      hemodynamicOffsetSeconds: 5,
      displayScale: { type: "robust-diverging-linear", scope: "all-frames-all-vertices", domain: [-1.2, 0, 1.4], clamp: true, units: "predicted BOLD" },
    },
    ...overrides,
  };
}

function thumbnails(status = "available") {
  const common = {
    status,
    source: "uploaded-video",
    captureTimeBasis: "tribe-segment-start",
    selection: "first decoded source frame at-or-after requested time",
    format: "image/jpeg",
    maxWidthPixels: 320,
  };
  if (status === "unavailable") return { ...common, reason: "source-frame-extraction-failed", items: [] };
  if (status === "partial") return {
    ...common,
    items: [{ index: 0, requestedTimeSeconds: 0, url: "thumbnails/0000.jpg", byteLength: 100, sha256: "1".repeat(64) }],
    unavailableItems: [{ index: 1, requestedTimeSeconds: 1, reason: "outside-source-video" }],
  };
  return {
    ...common,
    items: [0, 1].map((requestedTimeSeconds, index) => ({
      index,
      requestedTimeSeconds,
      url: `thumbnails/${String(index).padStart(4, "0")}.jpg`,
      byteLength: 100 + index,
      sha256: String(index + 1).repeat(64),
    })),
  };
}

test("accepts the exact fsaverage5 vertex contract and resolves the frame URL", () => {
  const validated = validateTribeManifest(manifest(), "https://gpu.example/api/tribe/v1/predict");
  assert.equal(validated.frames.url, "https://gpu.example/frames.f32");
  assert.deepEqual(validated.frames.shape, [2, TRIBE_VERTEX_COUNT]);
  assert.deepEqual(validated.modalitiesUsed, ["video"]);
  assert.deepEqual(validated.missingModalities, ["audio", "text"]);
  assert.equal(Object.isFrozen(validated.modalitiesUsed), true);
  assert.equal(Object.isFrozen(validated.missingModalities), true);
});

test("validates available source-video thumbnails and resolves their URLs", () => {
  const validated = validateTribeManifest(manifest({ thumbnails: thumbnails() }), "https://gpu.example/results/id/manifest.json");
  assert.equal(validated.thumbnails.status, "available");
  assert.equal(validated.thumbnails.items[0].url, "https://gpu.example/results/id/thumbnails/0000.jpg");
  assert.equal(validated.thumbnails.items[1].requestedTimeSeconds, validated.frames.timesSeconds[1]);
  assert.equal(Object.isFrozen(validated.thumbnails.items), true);
});

test("accepts explicit thumbnail unavailability without inventing image items", () => {
  const validated = validateTribeManifest(manifest({ thumbnails: thumbnails("unavailable") }));
  assert.equal(validated.thumbnails.status, "unavailable");
  assert.equal(validated.thumbnails.reason, "source-frame-extraction-failed");
  assert.deepEqual(validated.thumbnails.items, []);
});

test("accepts a sparse partial thumbnail partition with exact missing-frame evidence", () => {
  const validated = validateTribeManifest(manifest({ thumbnails: thumbnails("partial") }), "https://gpu.example/results/id/manifest.json");
  assert.equal(validated.thumbnails.status, "partial");
  assert.equal(validated.thumbnails.items[0].url, "https://gpu.example/results/id/thumbnails/0000.jpg");
  assert.deepEqual(validated.thumbnails.unavailableItems, [{ index: 1, requestedTimeSeconds: 1, reason: "outside-source-video" }]);
  assert.equal(Object.isFrozen(validated.thumbnails.unavailableItems), true);
});

test("rejects invalid thumbnail timing, integrity, URL, and unavailable states", () => {
  const wrongTime = thumbnails();
  wrongTime.items[1].requestedTimeSeconds = 1.5;
  assert.throws(() => validateTribeManifest(manifest({ thumbnails: wrongTime })), /thumbnail requested time/);
  const wrongBytes = thumbnails();
  wrongBytes.items[0].byteLength = 0;
  assert.throws(() => validateTribeManifest(manifest({ thumbnails: wrongBytes })), /byteLength/);
  const wrongSha = thumbnails();
  wrongSha.items[0].sha256 = "not-a-digest";
  assert.throws(() => validateTribeManifest(manifest({ thumbnails: wrongSha })), /sha256/);
  const unsafeUrl = thumbnails();
  unsafeUrl.items[0].url = "data:image/jpeg;base64,/9j/2Q==";
  assert.throws(() => validateTribeManifest(manifest({ thumbnails: unsafeUrl })), /HTTP or HTTPS/);
  const ambiguous = thumbnails("unavailable");
  ambiguous.items = thumbnails().items;
  assert.throws(() => validateTribeManifest(manifest({ thumbnails: ambiguous })), /must not contain image items/);

  const overlap = thumbnails("partial");
  overlap.unavailableItems[0].index = 0;
  overlap.unavailableItems[0].requestedTimeSeconds = 0;
  assert.throws(() => validateTribeManifest(manifest({ thumbnails: overlap })), /must not overlap or repeat/);

  const missingCoverage = thumbnails("partial");
  missingCoverage.unavailableItems = [];
  assert.throws(() => validateTribeManifest(manifest({ thumbnails: missingCoverage })), /must disclose unavailableItems/);

  const wrongUnavailableTime = thumbnails("partial");
  wrongUnavailableTime.unavailableItems[0].requestedTimeSeconds = 0.9;
  assert.throws(() => validateTribeManifest(manifest({ thumbnails: wrongUnavailableTime })), /requested time must match/);

  const wrongUnavailableReason = thumbnails("partial");
  wrongUnavailableReason.unavailableItems[0].reason = "decode-failed";
  assert.throws(() => validateTribeManifest(manifest({ thumbnails: wrongUnavailableReason })), /outside-source-video/);
});

test("accepts the full trimodal contract only with its exact modality declaration", () => {
  const validated = validateTribeManifest(manifest({
    inferenceMode: "full",
    modalitiesUsed: ["video", "audio", "text"],
    missingModalities: [],
  }));
  assert.equal(validated.inferenceMode, "full");
  assert.deepEqual(validated.modalitiesUsed, ["video", "audio", "text"]);
});

test("accepts audited execution provenance while preserving legacy stored result readability", () => {
  const accelerated = validateTribeManifest(manifest({
    extractors: {
      video: {
        id: TRIBE_VIDEO_MODEL_ID,
        revision: TRIBE_VIDEO_MODEL_REVISION,
        weightsSha256: TRIBE_VIDEO_MODEL_WEIGHTS_SHA256,
        device: "mps",
        precision: "autocast-fp16",
      },
    },
  }));
  assert.equal(accelerated.extractors.video.device, "mps");
  assert.equal(accelerated.extractors.video.precision, "autocast-fp16");

  assert.throws(() => validateTribeManifest(manifest({
    extractors: { video: { ...manifest().extractors.video, device: "cuda", precision: "autocast-fp16" } },
  })), /Invalid TRIBE v2 prediction/);
  assert.throws(() => validateTribeManifest(manifest({
    extractors: { video: { ...manifest().extractors.video, device: "mps" } },
  })), /Invalid TRIBE v2 prediction/);
});

test("rejects wrong space, vertex count, ordering, mapping, dtype, shape, and timestamps", () => {
  const cases = [
    () => ({ ...manifest(), surface: { ...manifest().surface, space: "fsaverage6" } }),
    () => ({ ...manifest(), surface: { ...manifest().surface, vertexCount: 20_000 } }),
    () => ({ ...manifest(), surface: { ...manifest().surface, mappingId: "unverified" } }),
    () => ({ ...manifest(), surface: { ...manifest().surface, hemispheres: [...manifest().surface.hemispheres].reverse() } }),
    () => ({ ...manifest(), frames: { ...manifest().frames, dtype: "float64" } }),
    () => ({ ...manifest(), frames: { ...manifest().frames, shape: [2, 20_000] } }),
    () => ({ ...manifest(), frames: { ...manifest().frames, timesSeconds: [1, 1] } }),
    () => ({ ...manifest(), frames: { ...manifest().frames, durationsSeconds: [1] } }),
    () => ({ ...manifest(), frames: { ...manifest().frames, durationsSeconds: [2, 1] } }),
    () => ({ ...manifest(), frames: { ...manifest().frames, hemodynamicOffsetSeconds: 6 } }),
    () => ({ ...manifest(), resultId: "result-1" }),
    () => ({ ...manifest(), predictionType: "viral probability" }),
    () => ({ ...manifest(), isViralityScore: true }),
    () => ({ ...manifest(), model: { ...manifest().model, revision: "a".repeat(40) } }),
    () => ({ ...manifest(), model: { ...manifest().model, weightsSha256: "a".repeat(64) } }),
    () => ({ ...manifest(), model: { ...manifest().model, codeRevision: "a".repeat(40) } }),
    () => ({ ...manifest(), model: { ...manifest().model, license: "MIT" } }),
    () => ({ ...manifest(), extractors: { video: { id: TRIBE_VIDEO_MODEL_ID, revision: "a".repeat(40) } } }),
    () => ({ ...manifest(), inferenceMode: "proxy" }),
    () => { const value = manifest(); delete value.inferenceMode; return value; },
    () => ({ ...manifest(), modalitiesUsed: ["video", "audio", "text"] }),
    () => ({ ...manifest(), modalitiesUsed: ["video", "video"] }),
    () => ({ ...manifest(), missingModalities: [] }),
    () => ({ ...manifest(), missingModalities: ["text", "audio"] }),
  ];
  for (const makeInvalid of cases) assert.throws(() => validateTribeManifest(makeInvalid()), /Invalid TRIBE v2 prediction/);
});
