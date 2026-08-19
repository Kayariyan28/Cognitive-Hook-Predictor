import assert from "node:assert/strict";
import test from "node:test";
import {
  VIDEO_THUMBNAIL_CONCURRENCY,
  fetchVerifiedVideoThumbnails,
  planVideoIntervalThumbnails,
  thumbnailTimeAvailability,
} from "../src/tribe/videoThumbnails.js";

test("plans source thumbnails at exact TRIBE interval starts without shifting timing", () => {
  const intervals = [
    { index: 4, startTime: 10, endTime: 11 },
    { index: 1, startTime: 5.25, endTime: 6.25 },
  ];
  const plan = planVideoIntervalThumbnails(intervals);
  assert.deepEqual(plan, [{ index: 1, startTime: 5.25 }, { index: 4, startTime: 10 }]);
  assert.equal(Object.isFrozen(plan), true);
  assert.equal(Object.isFrozen(plan[0]), true);
  assert.equal(VIDEO_THUMBNAIL_CONCURRENCY, 1);
});

test("marks times outside the decodable source duration unavailable instead of clamping", () => {
  assert.deepEqual(thumbnailTimeAvailability(0, 11), { available: true, reason: null });
  assert.deepEqual(thumbnailTimeAvailability(10.999, 11), { available: true, reason: null });
  assert.deepEqual(thumbnailTimeAvailability(11, 11), { available: false, reason: "outside-source-duration" });
  assert.deepEqual(thumbnailTimeAvailability(20, 11), { available: false, reason: "outside-source-duration" });
});

test("rejects ambiguous, duplicated, or nonfinite thumbnail requests", () => {
  assert.throws(() => planVideoIntervalThumbnails([{ index: 0, startTime: 1 }, { index: 0, startTime: 2 }]), /duplicated/);
  assert.throws(() => planVideoIntervalThumbnails([{ index: 0.5, startTime: 1 }]), /non-negative integer/);
  assert.throws(() => planVideoIntervalThumbnails([{ index: 0, startTime: Number.NaN }]), /finite/);
  assert.throws(() => thumbnailTimeAvailability(-1, 5), /non-negative/);
});

test("accepts verified JPEG bytes and rejects byte-length and SHA-256 mismatches", async () => {
  const bytes = new Uint8Array([0xff, 0xd8, 0x01, 0x02, 0xff, 0xd9]);
  const digestBuffer = await crypto.subtle.digest("SHA-256", bytes);
  const sha256 = Array.from(new Uint8Array(digestBuffer), (byte) => byte.toString(16).padStart(2, "0")).join("");
  const base = {
    status: "available",
    format: "image/jpeg",
    items: [{ index: 0, requestedTimeSeconds: 1, url: "https://example.test/0.jpg", byteLength: bytes.byteLength, sha256 }],
  };
  const received = [];
  const fetchRef = async () => new Response(bytes, { status: 200, headers: { "content-type": "image/jpeg" } });
  const result = await fetchVerifiedVideoThumbnails({ thumbnails: base, fetchRef, cryptoRef: crypto, onThumbnail: (item) => received.push(item) });
  assert.equal(result.extracted, 1);
  assert.equal(received[0].verified, true);
  assert.equal(received[0].blob.type, "image/jpeg");

  await assert.rejects(() => fetchVerifiedVideoThumbnails({
    thumbnails: { ...base, items: [{ ...base.items[0], byteLength: bytes.byteLength + 1 }] }, fetchRef, cryptoRef: crypto,
  }), /byte-length check/);
  await assert.rejects(() => fetchVerifiedVideoThumbnails({
    thumbnails: { ...base, items: [{ ...base.items[0], sha256: "0".repeat(64) }] }, fetchRef, cryptoRef: crypto,
  }), /SHA-256 check/);
});

test("fetches sparse verified partial thumbnails and reports exact missing intervals", async () => {
  const bytes = new Uint8Array([0xff, 0xd8, 0x01, 0xff, 0xd9]);
  const digestBuffer = await crypto.subtle.digest("SHA-256", bytes);
  const sha256 = Array.from(new Uint8Array(digestBuffer), (byte) => byte.toString(16).padStart(2, "0")).join("");
  const thumbnails = {
    status: "partial",
    format: "image/jpeg",
    items: [{ index: 0, requestedTimeSeconds: 0, url: "https://example.test/0.jpg", byteLength: bytes.byteLength, sha256 }],
    unavailableItems: [{ index: 1, requestedTimeSeconds: 1, reason: "outside-source-video" }],
  };
  const received = [];
  const unavailable = [];
  const result = await fetchVerifiedVideoThumbnails({
    thumbnails,
    fetchRef: async () => new Response(bytes, { status: 200 }),
    cryptoRef: crypto,
    onThumbnail: (item) => received.push(item),
    onUnavailable: (item) => unavailable.push(item),
  });
  assert.equal(result.extracted, 1);
  assert.equal(result.unavailable, 1);
  assert.equal(received[0].index, 0);
  assert.deepEqual(unavailable, [{ index: 1, requestedTimeSeconds: 1, reason: "outside-source-video", startTime: 1 }]);
});
