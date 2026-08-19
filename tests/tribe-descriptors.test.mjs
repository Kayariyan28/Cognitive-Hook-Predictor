import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { DESTRIEUX_MAPPING_SHA256 } from "../src/tribe/atlas.js";
import { TRIBE_VERTEX_COUNT } from "../src/tribe/contract.js";
import {
  deriveTribeDescriptors,
  TRIBE_DESCRIPTOR_SCHEMA_VERSION,
} from "../src/tribe/descriptors.js";

const metadata = JSON.parse(await readFile(
  new URL("../public/assets/brain/destrieux-fsaverage5.json", import.meta.url),
  "utf8",
));
const usableLabels = metadata.labels
  .map((name, labelIndex) => ({ name, labelIndex }))
  .filter(({ name }) => name !== "Unknown" && name !== "Medial_wall")
  .map(({ labelIndex }) => labelIndex);

function syntheticAtlas() {
  const parcelIds = new Uint8Array(TRIBE_VERTEX_COUNT);
  const perHemisphere = TRIBE_VERTEX_COUNT / 2;
  for (let hemisphere = 0; hemisphere < 2; hemisphere += 1) {
    const offset = hemisphere * perHemisphere;
    parcelIds[offset] = 0;
    parcelIds[offset + 1] = metadata.labels.indexOf("Medial_wall");
    for (let localVertex = 2; localVertex < perHemisphere; localVertex += 1) {
      parcelIds[offset + localVertex] = usableLabels[(localVertex - 2) % usableLabels.length];
    }
  }
  return { metadata: { ...metadata }, parcelIds };
}

function prediction(frameValues = [1, -1, 3]) {
  const atlas = syntheticAtlas();
  const values = new Float32Array(frameValues.length * TRIBE_VERTEX_COUNT);
  for (let frameIndex = 0; frameIndex < frameValues.length; frameIndex += 1) {
    const offset = frameIndex * TRIBE_VERTEX_COUNT;
    values.fill(frameValues[frameIndex], offset, offset + TRIBE_VERTEX_COUNT);
    // Excluded atlas vertices must not affect any descriptor.
    values[offset] = 1_000;
    values[offset + 1] = -1_000;
    values[offset + TRIBE_VERTEX_COUNT / 2] = 1_000;
    values[offset + TRIBE_VERTEX_COUNT / 2 + 1] = -1_000;
  }
  return {
    atlas,
    prediction: {
      values,
      manifest: {
        resultId: "1".repeat(32),
        predictionType: "average-subject cortical BOLD response",
        isViralityScore: false,
        inferenceMode: "vision-only",
        modalitiesUsed: ["video"],
        missingModalities: ["audio", "text"],
        model: {
          id: "facebook/tribev2",
          revision: "model-revision",
          weightsSha256: "2".repeat(64),
          codeRevision: "code-revision",
          license: "CC-BY-NC-4.0",
        },
        extractors: { video: { id: "facebook/vjepa2-vitg-fpc64-256", revision: "extractor-revision" } },
        surface: { space: "fsaverage5", mappingId: "verified-mapping", vertexCount: TRIBE_VERTEX_COUNT },
        frames: {
          shape: [frameValues.length, TRIBE_VERTEX_COUNT],
          timesSeconds: [0, 2, 5].slice(0, frameValues.length),
          durationsSeconds: [2, 3, 1].slice(0, frameValues.length),
          sha256: "3".repeat(64),
          dtype: "float32",
          layout: "time-major",
          hemodynamicOffsetSeconds: 5,
          displayScale: { units: "synthetic predicted BOLD units" },
        },
      },
    },
  };
}

function close(actual, expected, epsilon = 1e-12) {
  assert.ok(Math.abs(actual - expected) <= epsilon, `${actual} was not within ${epsilon} of ${expected}`);
}

test("derives duration-weighted whole-cortex, transition, phase, and provenance descriptors", () => {
  const fixture = prediction();
  const report = deriveTribeDescriptors(fixture.prediction, fixture.atlas);

  assert.equal(report.schemaVersion, TRIBE_DESCRIPTOR_SCHEMA_VERSION);
  assert.equal(report.source.resultId, "1".repeat(32));
  assert.equal(report.source.isViralityScore, false);
  assert.equal(report.source.behavioralPrediction, false);
  assert.equal(report.source.atlas.mappingSha256, DESTRIEUX_MAPPING_SHA256);
  assert.equal(report.units.response, "synthetic predicted BOLD units");
  assert.equal(report.frames.length, 3);
  assert.equal(report.clip.usableVertexCount, TRIBE_VERTEX_COUNT - 4);

  assert.deepEqual(
    report.frames.map(({ time, duration, endTime }) => [time, duration, endTime]),
    [[0, 2, 2], [2, 3, 5], [5, 1, 6]],
  );
  assert.deepEqual(
    report.frames.map(({ mean, rms, spatialStd, positiveFraction, negativeFraction }) => (
      [mean, rms, spatialStd, positiveFraction, negativeFraction]
    )),
    [[1, 1, 0, 1, 0], [-1, 1, 0, 0, 1], [3, 3, 0, 1, 0]],
  );
  assert.equal(report.frames[0].changeRms, null);
  assert.equal(report.frames[0].changeRate, null);
  assert.equal(report.frames[0].continuity, null);
  assert.equal(report.frames[1].changeRms, 2);
  assert.equal(report.frames[1].changeRate, 1);
  assert.equal(report.frames[1].continuity, -1);
  assert.equal(report.frames[2].changeRms, 4);
  close(report.frames[2].changeRate, 4 / 3);
  assert.equal(report.frames[2].continuity, -1);
  for (const frame of report.frames) close(frame.spatialDistribution, 100);

  assert.equal(report.clip.coveredDuration, 6);
  close(report.clip.responseMagnitude, Math.sqrt(14 / 6));
  close(report.clip.signedMean, 1 / 3);
  assert.equal(report.clip.continuity, -1);
  close(report.clip.spatialDistribution, 100);
  assert.equal(report.clip.peakMagnitude.index, 2);
  assert.equal(report.clip.peakMagnitude.time, 5);
  assert.equal(report.clip.peakMagnitude.value, 3);
  assert.equal(report.clip.largestChange.index, 2);
  assert.equal(report.clip.largestChange.value, 4);
  assert.equal(report.clip.persistence.durationSeconds, 1);
  close(report.clip.persistence.fraction, 1 / 6);
  close(report.clip.persistence.percent, 100 / 6);

  assert.deepEqual(report.phases.map(({ id, startTime, endTime, coveredDuration }) => (
    [id, startTime, endTime, coveredDuration]
  )), [
    ["early", 0, 2, 2],
    ["middle", 2, 4, 2],
    ["late", 4, 6, 2],
  ]);
  assert.equal(report.phases[0].responseMagnitude, 1);
  assert.equal(report.phases[0].signedMean, 1);
  assert.equal(report.phases[1].responseMagnitude, 1);
  assert.equal(report.phases[1].signedMean, -1);
  close(report.phases[2].responseMagnitude, Math.sqrt(5));
  assert.equal(report.phases[2].signedMean, 1);
  assert.equal(Object.isFrozen(report), true);
  assert.equal(Object.isFrozen(report.frames[0].topParcel), true);
});

test("produces 148 deterministic hemispheric parcel time summaries", () => {
  const fixture = prediction();
  const report = deriveTribeDescriptors(fixture.prediction, fixture.atlas);

  assert.equal(report.parcels.length, 148);
  assert.equal(new Set(report.parcels.map(({ key }) => key)).size, 148);
  assert.equal(report.parcels[0].hemisphere, "left");
  assert.equal(report.parcels[0].labelIndex, usableLabels[0]);
  for (const parcel of report.parcels) {
    close(parcel.signedMean, 1 / 3);
    close(parcel.rms, Math.sqrt(14 / 6));
    assert.deepEqual(parcel.maxPositive, { index: 2, time: 5, duration: 1, value: 3 });
    assert.deepEqual(parcel.maxNegative, { index: 1, time: 2, duration: 3, value: -1 });
    assert.deepEqual(parcel.maxAbsolute, { index: 2, time: 5, duration: 1, value: 3 });
    assert.deepEqual(parcel.peakMagnitude, { index: 2, time: 5, duration: 1, value: 3 });
  }
  assert.equal(report.frames[0].topParcel.key, `left:${usableLabels[0]}`);
  assert.equal(report.frames[0].topParcel.rms, 1);
});

test("uses deterministic earliest ties and leaves zero-vector cosine undefined", () => {
  const fixture = prediction([0, 0]);
  fixture.prediction.manifest.frames.timesSeconds = [0, 2];
  fixture.prediction.manifest.frames.durationsSeconds = [2, 2];
  for (let index = 0; index < fixture.prediction.values.length; index += 1) fixture.prediction.values[index] = 0;
  const report = deriveTribeDescriptors(fixture.prediction, fixture.atlas);

  assert.equal(report.clip.peakMagnitude.index, 0);
  assert.equal(report.clip.largestChange.index, 1);
  assert.equal(report.frames[1].continuity, null);
  assert.equal(report.clip.continuity, null);
  assert.equal(report.clip.persistence.durationSeconds, 0);
  assert.equal(report.clip.persistence.fraction, 0);
  assert.equal(report.clip.spatialDistribution, 0);
});

test("rejects malformed tensor shapes, non-finite values, timings, and unverified atlases", () => {
  const wrongLength = prediction();
  wrongLength.prediction.values = wrongLength.prediction.values.slice(1);
  assert.throws(() => deriveTribeDescriptors(wrongLength.prediction, wrongLength.atlas), /tensor length/);

  const notFinite = prediction();
  notFinite.prediction.values[123] = Number.NaN;
  assert.throws(() => deriveTribeDescriptors(notFinite.prediction, notFinite.atlas), /not finite/);

  const overlapping = prediction();
  overlapping.prediction.manifest.frames.timesSeconds = [0, 1, 5];
  assert.throws(() => deriveTribeDescriptors(overlapping.prediction, overlapping.atlas), /must not overlap/);

  const wrongAtlas = prediction();
  wrongAtlas.atlas.metadata.mappingSha256 = "0".repeat(64);
  assert.throws(() => deriveTribeDescriptors(wrongAtlas.prediction, wrongAtlas.atlas), /verified Destrieux/);
});
