import { DESTRIEUX_MAPPING_SHA256, DESTRIEUX_SCHEMA_VERSION, atlasParcelKey } from "./atlas.js";
import { TRIBE_HEMISPHERE_VERTEX_COUNT, TRIBE_VERTEX_COUNT } from "./contract.js";

export const TRIBE_DESCRIPTOR_SCHEMA_VERSION = "tribe-cortical-descriptors/1";

const EXCLUDED_LABELS = new Set(["Unknown", "Medial_wall"]);
const EXPECTED_PARCEL_COUNT = 148;
const PERSISTENCE_THRESHOLD_FRACTION = 0.75;

function fail(message) {
  throw new Error(`Invalid TRIBE descriptor input: ${message}`);
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function finite(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number)) fail(`${label} must be finite.`);
  return number;
}

function finiteOrNull(value) {
  return Number.isFinite(value) ? value : null;
}

function deepFreeze(value) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  for (const child of Object.values(value)) deepFreeze(child);
  return Object.freeze(value);
}

function validatePrediction(prediction) {
  if (!isRecord(prediction) || !(prediction.values instanceof Float32Array) || !isRecord(prediction.manifest)) {
    fail("a verified prediction with a Float32Array tensor and manifest is required.");
  }
  const { manifest, values } = prediction;
  const frames = manifest.frames;
  if (manifest.predictionType !== "average-subject cortical BOLD response" || manifest.isViralityScore !== false) {
    fail("the manifest must identify a non-behavioral TRIBE cortical BOLD prediction.");
  }
  if (manifest.model?.id !== "facebook/tribev2"
      || manifest.surface?.space !== "fsaverage5"
      || manifest.surface?.vertexCount !== TRIBE_VERTEX_COUNT) {
    fail("the manifest does not identify the verified TRIBE v2 fsaverage5 surface contract.");
  }
  if (!isRecord(frames) || !Array.isArray(frames.shape) || frames.shape.length !== 2) {
    fail("manifest.frames.shape must be [time, vertex].");
  }
  const frameCount = Number(frames.shape[0]);
  const vertexCount = Number(frames.shape[1]);
  if (!Number.isInteger(frameCount) || frameCount < 1 || vertexCount !== TRIBE_VERTEX_COUNT) {
    fail(`the tensor must contain at least one full ${TRIBE_VERTEX_COUNT}-vertex frame.`);
  }
  if (values.length !== frameCount * TRIBE_VERTEX_COUNT) fail("tensor length does not match manifest.frames.shape.");
  if (!Array.isArray(frames.timesSeconds) || !Array.isArray(frames.durationsSeconds)
      || frames.timesSeconds.length !== frameCount || frames.durationsSeconds.length !== frameCount) {
    fail("frame times and durations must match the tensor frame count.");
  }

  const times = new Float64Array(frameCount);
  const durations = new Float64Array(frameCount);
  let previousTime = -Infinity;
  let previousEnd = -Infinity;
  for (let index = 0; index < frameCount; index += 1) {
    const time = finite(frames.timesSeconds[index], `frames.timesSeconds[${index}]`);
    const duration = finite(frames.durationsSeconds[index], `frames.durationsSeconds[${index}]`);
    if (duration <= 0) fail(`frames.durationsSeconds[${index}] must be positive.`);
    if (time <= previousTime) fail("frame times must be strictly increasing.");
    if (time < previousEnd - Number.EPSILON) fail("frame intervals must not overlap.");
    times[index] = time;
    durations[index] = duration;
    previousTime = time;
    previousEnd = time + duration;
  }
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) fail(`tensor value ${index} is not finite.`);
  }
  return { manifest, values, frameCount, times, durations };
}

function validateAtlas(atlas) {
  if (!isRecord(atlas) || !(atlas.parcelIds instanceof Uint8Array) || !isRecord(atlas.metadata)) {
    fail("a verified Destrieux fsaverage5 atlas is required.");
  }
  const { metadata, parcelIds } = atlas;
  if (metadata.schemaVersion !== DESTRIEUX_SCHEMA_VERSION
      || metadata.space !== "fsaverage5"
      || metadata.vertexCount !== TRIBE_VERTEX_COUNT
      || metadata.verticesPerHemisphere !== TRIBE_HEMISPHERE_VERTEX_COUNT
      || metadata.mappingSha256 !== DESTRIEUX_MAPPING_SHA256
      || parcelIds.length !== TRIBE_VERTEX_COUNT) {
    fail("the atlas does not match the verified Destrieux fsaverage5 mapping.");
  }
  if (!Array.isArray(metadata.labels) || metadata.labels.length !== 76 || metadata.labels[0] !== "Unknown") {
    fail("the atlas label table is invalid.");
  }
  const medialWallCount = metadata.labels.filter((label) => label === "Medial_wall").length;
  if (medialWallCount !== 1) fail("the atlas must contain exactly one Medial_wall label.");

  const parcels = [];
  const slotByHemisphereAndLabel = new Int16Array(metadata.labels.length * 2).fill(-1);
  for (let hemisphereIndex = 0; hemisphereIndex < 2; hemisphereIndex += 1) {
    const hemisphere = hemisphereIndex === 0 ? "left" : "right";
    for (let labelIndex = 0; labelIndex < metadata.labels.length; labelIndex += 1) {
      const name = metadata.labels[labelIndex];
      if (EXCLUDED_LABELS.has(name)) continue;
      const slot = parcels.length;
      slotByHemisphereAndLabel[hemisphereIndex * metadata.labels.length + labelIndex] = slot;
      parcels.push({
        key: atlasParcelKey(hemisphere, labelIndex),
        hemisphere,
        labelIndex,
        name,
        vertexCount: 0,
      });
    }
  }
  if (parcels.length !== EXPECTED_PARCEL_COUNT) fail(`the atlas must define exactly ${EXPECTED_PARCEL_COUNT} usable hemispheric parcels.`);

  const vertexSlots = new Int16Array(TRIBE_VERTEX_COUNT).fill(-1);
  let usableVertexCount = 0;
  for (let vertex = 0; vertex < TRIBE_VERTEX_COUNT; vertex += 1) {
    const labelIndex = parcelIds[vertex];
    if (labelIndex >= metadata.labels.length) fail(`atlas vertex ${vertex} has an invalid label.`);
    const hemisphereIndex = vertex < TRIBE_HEMISPHERE_VERTEX_COUNT ? 0 : 1;
    const slot = slotByHemisphereAndLabel[hemisphereIndex * metadata.labels.length + labelIndex];
    if (slot < 0) continue;
    vertexSlots[vertex] = slot;
    parcels[slot].vertexCount += 1;
    usableVertexCount += 1;
  }
  if (!usableVertexCount) fail("the atlas has no usable cortical vertices.");
  const emptyParcel = parcels.find((parcel) => parcel.vertexCount === 0);
  if (emptyParcel) fail(`atlas parcel ${emptyParcel.key} contains no vertices.`);
  return { metadata, parcels, vertexSlots, usableVertexCount };
}

function basicMoment(frame, value) {
  return {
    index: frame.index,
    time: frame.time,
    duration: frame.duration,
    value,
  };
}

function momentWithParcel(frame, value) {
  return { ...basicMoment(frame, value), topParcel: frame.topParcel };
}

function chooseLarger(current, candidate, magnitude = (value) => value) {
  if (!current || magnitude(candidate.value) > magnitude(current.value)) return candidate;
  return current;
}

function frameOverlap(startA, endA, startB, endB) {
  return Math.max(0, Math.min(endA, endB) - Math.max(startA, startB));
}

/**
 * Derives descriptive cortical-response statistics from a contract-validated
 * TRIBE v2 tensor. These values do not estimate attention, engagement,
 * watch-time, or virality.
 */
export function deriveTribeDescriptors(prediction, atlas) {
  const { manifest, values, frameCount, times, durations } = validatePrediction(prediction);
  const atlasData = validateAtlas(atlas);
  const { parcels: parcelDefinitions, vertexSlots, usableVertexCount } = atlasData;
  const parcelCount = parcelDefinitions.length;

  const parcelWeightedSums = new Float64Array(parcelCount);
  const parcelWeightedSumSquares = new Float64Array(parcelCount);
  const parcelDurationWeights = new Float64Array(parcelCount);
  const parcelMaxPositive = new Array(parcelCount).fill(null);
  const parcelMaxNegative = new Array(parcelCount).fill(null);
  const parcelMaxAbsolute = new Array(parcelCount).fill(null);
  const parcelPeakMagnitude = new Array(parcelCount).fill(null);
  const frames = [];

  let durationSum = 0;
  let weightedSum = 0;
  let weightedSumSquares = 0;
  let weightedSpatialDistribution = 0;
  let weightedContinuity = 0;
  let continuityWeight = 0;
  let peakMagnitude = null;
  let largestChange = null;

  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    const offset = frameIndex * TRIBE_VERTEX_COUNT;
    const parcelSums = new Float64Array(parcelCount);
    const parcelSumSquares = new Float64Array(parcelCount);
    let sum = 0;
    let sumSquares = 0;
    let sumFourthPowers = 0;
    let positiveCount = 0;
    let negativeCount = 0;
    let changeSumSquares = 0;
    let previousSumSquares = 0;
    let dot = 0;

    for (let vertex = 0; vertex < TRIBE_VERTEX_COUNT; vertex += 1) {
      const slot = vertexSlots[vertex];
      if (slot < 0) continue;
      const value = values[offset + vertex];
      const square = value * value;
      sum += value;
      sumSquares += square;
      sumFourthPowers += square * square;
      if (value > 0) positiveCount += 1;
      else if (value < 0) negativeCount += 1;
      parcelSums[slot] += value;
      parcelSumSquares[slot] += square;

      if (frameIndex > 0) {
        const previousValue = values[offset - TRIBE_VERTEX_COUNT + vertex];
        const difference = value - previousValue;
        changeSumSquares += difference * difference;
        previousSumSquares += previousValue * previousValue;
        dot += value * previousValue;
      }
    }

    const duration = durations[frameIndex];
    const time = times[frameIndex];
    const mean = sum / usableVertexCount;
    const meanSquare = sumSquares / usableVertexCount;
    const rms = Math.sqrt(meanSquare);
    const spatialStd = Math.sqrt(Math.max(0, meanSquare - mean * mean));
    const spatialDistribution = sumSquares > 0 && sumFourthPowers > 0
      ? Math.min(100, 100 * (sumSquares * sumSquares) / (usableVertexCount * sumFourthPowers))
      : 0;
    const changeRms = frameIndex > 0 ? Math.sqrt(changeSumSquares / usableVertexCount) : null;
    const deltaTime = frameIndex > 0 ? time - times[frameIndex - 1] : null;
    const changeRate = changeRms === null ? null : changeRms / deltaTime;
    let continuity = null;
    if (frameIndex > 0) {
      const denominator = Math.sqrt(previousSumSquares * sumSquares);
      if (denominator > 0) continuity = Math.max(-1, Math.min(1, dot / denominator));
    }

    let topParcel = null;
    for (let slot = 0; slot < parcelCount; slot += 1) {
      const definition = parcelDefinitions[slot];
      const parcelMean = parcelSums[slot] / definition.vertexCount;
      const parcelRms = Math.sqrt(parcelSumSquares[slot] / definition.vertexCount);
      const candidate = {
        key: definition.key,
        hemisphere: definition.hemisphere,
        labelIndex: definition.labelIndex,
        name: definition.name,
        vertexCount: definition.vertexCount,
        mean: parcelMean,
        rms: parcelRms,
      };
      if (!topParcel || candidate.rms > topParcel.rms) topParcel = candidate;

      const frameForMoment = { index: frameIndex, time, duration };
      if (parcelMean > 0) parcelMaxPositive[slot] = chooseLarger(parcelMaxPositive[slot], basicMoment(frameForMoment, parcelMean));
      if (parcelMean < 0) parcelMaxNegative[slot] = chooseLarger(parcelMaxNegative[slot], basicMoment(frameForMoment, parcelMean), (value) => -value);
      parcelMaxAbsolute[slot] = chooseLarger(parcelMaxAbsolute[slot], basicMoment(frameForMoment, parcelMean), Math.abs);
      parcelPeakMagnitude[slot] = chooseLarger(parcelPeakMagnitude[slot], basicMoment(frameForMoment, parcelRms));
      parcelWeightedSums[slot] += duration * parcelSums[slot];
      parcelWeightedSumSquares[slot] += duration * parcelSumSquares[slot];
      parcelDurationWeights[slot] += duration;
    }

    const frame = {
      index: frameIndex,
      time,
      duration,
      endTime: time + duration,
      mean,
      rms,
      spatialStd,
      positiveFraction: positiveCount / usableVertexCount,
      negativeFraction: negativeCount / usableVertexCount,
      spatialDistribution,
      changeRms,
      changeRate,
      continuity,
      topParcel,
    };
    frames.push(frame);

    durationSum += duration;
    weightedSum += duration * sum;
    weightedSumSquares += duration * sumSquares;
    weightedSpatialDistribution += duration * spatialDistribution;
    if (continuity !== null) {
      weightedContinuity += duration * continuity;
      continuityWeight += duration;
    }
    peakMagnitude = chooseLarger(peakMagnitude, momentWithParcel(frame, rms));
    if (changeRms !== null) {
      const candidate = {
        ...momentWithParcel(frame, changeRms),
        changeRate,
        continuity,
      };
      largestChange = chooseLarger(largestChange, candidate);
    }
  }

  const persistenceThreshold = peakMagnitude.value * PERSISTENCE_THRESHOLD_FRACTION;
  const persistentDuration = peakMagnitude.value > 0
    ? frames.reduce((total, frame) => total + (frame.rms >= persistenceThreshold ? frame.duration : 0), 0)
    : 0;
  const startTime = frames[0].time;
  const endTime = frames.at(-1).endTime;
  const phaseSpan = (endTime - startTime) / 3;
  const phases = ["early", "middle", "late"].map((id, phaseIndex) => {
    const phaseStart = startTime + phaseSpan * phaseIndex;
    const phaseEnd = phaseIndex === 2 ? endTime : startTime + phaseSpan * (phaseIndex + 1);
    let coveredDuration = 0;
    let phaseWeightedMean = 0;
    let phaseWeightedMeanSquares = 0;
    for (const frame of frames) {
      const overlap = frameOverlap(frame.time, frame.endTime, phaseStart, phaseEnd);
      if (!overlap) continue;
      coveredDuration += overlap;
      phaseWeightedMean += overlap * frame.mean;
      phaseWeightedMeanSquares += overlap * frame.rms * frame.rms;
    }
    return {
      id,
      startTime: phaseStart,
      endTime: phaseEnd,
      coveredDuration,
      responseMagnitude: coveredDuration > 0 ? Math.sqrt(phaseWeightedMeanSquares / coveredDuration) : null,
      signedMean: coveredDuration > 0 ? phaseWeightedMean / coveredDuration : null,
    };
  });

  const parcels = parcelDefinitions.map((definition, slot) => ({
    ...definition,
    signedMean: parcelWeightedSums[slot] / (parcelDurationWeights[slot] * definition.vertexCount),
    rms: Math.sqrt(parcelWeightedSumSquares[slot] / (parcelDurationWeights[slot] * definition.vertexCount)),
    maxPositive: parcelMaxPositive[slot],
    maxNegative: parcelMaxNegative[slot],
    maxAbsolute: parcelMaxAbsolute[slot],
    peakMagnitude: parcelPeakMagnitude[slot],
  })).sort((left, right) => right.rms - left.rms
    || (left.hemisphere === right.hemisphere ? 0 : left.hemisphere === "left" ? -1 : 1)
    || left.labelIndex - right.labelIndex);

  const responseUnits = typeof manifest.frames?.displayScale?.units === "string" && manifest.frames.displayScale.units
    ? manifest.frames.displayScale.units
    : "TRIBE-v2 predicted BOLD (training-target z-score units)";
  const result = {
    schemaVersion: TRIBE_DESCRIPTOR_SCHEMA_VERSION,
    source: {
      descriptorType: "descriptive cortical response",
      resultId: manifest.resultId ?? null,
      predictionType: manifest.predictionType ?? "average-subject cortical BOLD response",
      isViralityScore: false,
      behavioralPrediction: false,
      model: {
        id: manifest.model?.id ?? null,
        revision: manifest.model?.revision ?? null,
        weightsSha256: manifest.model?.weightsSha256 ?? null,
        codeRevision: manifest.model?.codeRevision ?? null,
        license: manifest.model?.license ?? null,
      },
      extractor: { ...manifest.extractors?.video },
      inferenceMode: manifest.inferenceMode ?? null,
      modalitiesUsed: Array.isArray(manifest.modalitiesUsed) ? [...manifest.modalitiesUsed] : [],
      missingModalities: Array.isArray(manifest.missingModalities) ? [...manifest.missingModalities] : [],
      surface: {
        space: manifest.surface?.space ?? "fsaverage5",
        mappingId: manifest.surface?.mappingId ?? null,
        vertexCount: TRIBE_VERTEX_COUNT,
      },
      tensor: {
        sha256: manifest.frames?.sha256 ?? null,
        frameCount,
        dtype: manifest.frames?.dtype ?? "float32",
        layout: manifest.frames?.layout ?? "time-major",
        hemodynamicOffsetSeconds: finiteOrNull(manifest.frames?.hemodynamicOffsetSeconds),
      },
      atlas: {
        name: atlasData.metadata.atlas ?? "Destrieux et al. 2010 cortical parcellation",
        schemaVersion: atlasData.metadata.schemaVersion,
        mappingSha256: atlasData.metadata.mappingSha256,
      },
      excludedAtlasLabels: [...EXCLUDED_LABELS],
    },
    units: {
      response: responseUnits,
      change: `RMS difference in ${responseUnits}`,
      changeRate: `RMS difference in ${responseUnits} per second`,
      time: "seconds",
      fraction: "unit interval (0 to 1)",
      spatialDistribution: "normalized energy participation ratio (0 to 100)",
    },
    frames,
    clip: {
      startTime,
      endTime,
      coveredDuration: durationSum,
      usableVertexCount,
      peakMagnitude,
      largestChange,
      responseMagnitude: Math.sqrt(weightedSumSquares / (durationSum * usableVertexCount)),
      signedMean: weightedSum / (durationSum * usableVertexCount),
      continuity: continuityWeight > 0 ? weightedContinuity / continuityWeight : null,
      persistence: {
        thresholdFraction: PERSISTENCE_THRESHOLD_FRACTION,
        thresholdValue: persistenceThreshold,
        durationSeconds: persistentDuration,
        fraction: persistentDuration / durationSum,
        percent: 100 * persistentDuration / durationSum,
      },
      spatialDistribution: weightedSpatialDistribution / durationSum,
    },
    phases,
    parcels,
  };
  return deepFreeze(result);
}
