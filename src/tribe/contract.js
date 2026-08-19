export const TRIBE_VERTEX_COUNT = 20_484;
export const TRIBE_HEMISPHERE_VERTEX_COUNT = 10_242;
export const TRIBE_MODEL_REVISION = "f894e783020944dcd96e5568550afe2aa9743f9f";
export const TRIBE_MODEL_WEIGHTS_SHA256 = "9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321";
export const TRIBE_CODE_REVISION = "af58661791a351a448a489042a28f6c37e1c14b7";
export const TRIBE_VIDEO_MODEL_ID = "facebook/vjepa2-vitg-fpc64-256";
export const TRIBE_VIDEO_MODEL_REVISION = "875c192b7b704b87d1e1d99345769632dd5f739a";
export const TRIBE_VIDEO_MODEL_WEIGHTS_SHA256 = "f205e77aa2ade168db6b09d4bc420d156141f64ab964278a9c181a2bdf2a232b";
const TRIBE_VIDEO_DEVICES = new Set(["cpu", "cuda", "mps"]);
const TRIBE_VIDEO_PRECISIONS = new Set(["fp32", "autocast-fp16"]);
const THUMBNAIL_MAX_BYTES = 2 * 1024 * 1024;
export const TRIBE_MESH_MAPPING_ID =
  "fsaverage5-half:52eda3c221c9439b320f97663d441390c56a14e69c2cddb7a5073f6d550466cb";

function fail(message) {
  throw new Error(`Invalid TRIBE v2 prediction: ${message}`);
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function finite(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number)) fail(`${label} must be finite.`);
  return number;
}

function integer(value, label) {
  const number = finite(value, label);
  if (!Number.isInteger(number)) fail(`${label} must be an integer.`);
  return number;
}

function validateTimes(times, frameCount) {
  if (!Array.isArray(times) || times.length !== frameCount) {
    fail(`frames.timesSeconds must contain exactly ${frameCount} values.`);
  }
  let previous = -Infinity;
  return times.map((value, index) => {
    const time = finite(value, `frames.timesSeconds[${index}]`);
    if (time <= previous) fail("frames.timesSeconds must be strictly increasing.");
    previous = time;
    return time;
  });
}

function validateDurations(durations, times) {
  if (!Array.isArray(durations) || durations.length !== times.length) {
    fail(`frames.durationsSeconds must contain exactly ${times.length} values.`);
  }
  return durations.map((value, index) => {
    const duration = finite(value, `frames.durationsSeconds[${index}]`);
    if (duration <= 0) fail(`frames.durationsSeconds[${index}] must be positive.`);
    if (index < times.length - 1 && times[index] + duration > times[index + 1] + Number.EPSILON) {
      fail("prediction frame intervals must not overlap.");
    }
    return duration;
  });
}

function validateHemispheres(hemispheres) {
  if (!Array.isArray(hemispheres) || hemispheres.length !== 2) {
    fail("surface.hemispheres must contain left and right entries.");
  }
  const [left, right] = hemispheres;
  const expected = [
    [left, "left", 0],
    [right, "right", TRIBE_HEMISPHERE_VERTEX_COUNT],
  ];
  for (const [hemisphere, id, offset] of expected) {
    if (!isRecord(hemisphere)) fail(`surface.hemispheres.${id} is missing.`);
    if (hemisphere.id !== id) fail(`expected ${id} hemisphere first.`);
    if (integer(hemisphere.offset, `${id}.offset`) !== offset) fail(`${id} vertex offset does not match fsaverage5 ordering.`);
    if (integer(hemisphere.count, `${id}.count`) !== TRIBE_HEMISPHERE_VERTEX_COUNT) {
      fail(`${id} hemisphere must contain ${TRIBE_HEMISPHERE_VERTEX_COUNT} vertices.`);
    }
  }
}

function validateInferenceMode(input) {
  const expected = input.inferenceMode === "vision-only"
    ? { used: ["video"], missing: ["audio", "text"] }
    : input.inferenceMode === "full"
      ? { used: ["video", "audio", "text"], missing: [] }
      : null;
  if (!expected) fail("inferenceMode must be full or vision-only.");

  for (const [field, values] of [["modalitiesUsed", expected.used], ["missingModalities", expected.missing]]) {
    const actual = input[field];
    if (!Array.isArray(actual) || actual.length !== values.length || actual.some((value, index) => value !== values[index])) {
      fail(`${field} does not match inferenceMode ${input.inferenceMode}.`);
    }
  }
  return {
    inferenceMode: input.inferenceMode,
    modalitiesUsed: Object.freeze([...expected.used]),
    missingModalities: Object.freeze([...expected.missing]),
  };
}

function resolveFramesUrl(value, responseUrl) {
  if (typeof value !== "string" || !value) fail("frames.url is required.");
  try {
    return new URL(value, responseUrl || globalThis.location?.href || "http://localhost/").href;
  } catch {
    fail("frames.url is not a valid URL.");
  }
}

function resolveHttpUrl(value, responseUrl, label) {
  if (typeof value !== "string" || !value) fail(`${label} is required.`);
  try {
    const resolved = new URL(value, responseUrl || globalThis.location?.href || "http://localhost/");
    if (!new Set(["http:", "https:"]).has(resolved.protocol)) fail(`${label} must use HTTP or HTTPS.`);
    return resolved.href;
  } catch (error) {
    if (error?.message?.startsWith("Invalid TRIBE")) throw error;
    fail(`${label} is not a valid URL.`);
  }
}

function validateThumbnails(thumbnails, timesSeconds, responseUrl) {
  if (thumbnails === undefined) return undefined;
  if (!isRecord(thumbnails)) fail("thumbnails must be an object when provided.");
  if (!new Set(["available", "partial", "unavailable"]).has(thumbnails.status)) fail("thumbnails.status must be available, partial, or unavailable.");
  if (thumbnails.source !== "uploaded-video") fail("thumbnails.source must be uploaded-video.");
  if (thumbnails.captureTimeBasis !== "tribe-segment-start") fail("thumbnails.captureTimeBasis must use TRIBE segment starts.");
  if (thumbnails.selection !== "first decoded source frame at-or-after requested time") fail("thumbnails.selection is unsupported.");
  if (thumbnails.format !== "image/jpeg") fail("thumbnails.format must be image/jpeg.");
  if (integer(thumbnails.maxWidthPixels, "thumbnails.maxWidthPixels") !== 320) fail("thumbnails.maxWidthPixels must be 320.");
  if (thumbnails.status === "unavailable") {
    if (thumbnails.reason !== "source-frame-extraction-failed") fail("unavailable thumbnails must disclose the supported failure reason.");
    if (!Array.isArray(thumbnails.items) || thumbnails.items.length !== 0) fail("unavailable thumbnails must not contain image items.");
    return Object.freeze({ ...thumbnails, items: Object.freeze([]) });
  }
  if (!Array.isArray(thumbnails.items)) fail("thumbnails.items must be an array.");
  if (thumbnails.status === "available" && thumbnails.items.length !== timesSeconds.length) {
    fail(`available thumbnails.items must contain exactly ${timesSeconds.length} entries.`);
  }
  if (thumbnails.status === "partial" && (thumbnails.items.length < 1 || thumbnails.items.length >= timesSeconds.length)) {
    fail("partial thumbnails must contain some, but not all, source frames.");
  }
  const coveredIndices = new Set();
  const items = thumbnails.items.map((item, position) => {
    if (!isRecord(item)) fail(`thumbnails.items[${position}] must be an object.`);
    const index = integer(item.index, `thumbnails.items[${position}].index`);
    if (index < 0 || index >= timesSeconds.length) fail("thumbnail index is outside the TRIBE frame range.");
    if (coveredIndices.has(index)) fail("thumbnail frame indices must not be duplicated.");
    coveredIndices.add(index);
    const requestedTimeSeconds = finite(item.requestedTimeSeconds, `thumbnails.items[${position}].requestedTimeSeconds`);
    if (Math.abs(requestedTimeSeconds - timesSeconds[index]) > 1e-9) fail("thumbnail requested time must match its TRIBE interval start.");
    const byteLength = integer(item.byteLength, `thumbnails.items[${position}].byteLength`);
    if (byteLength <= 4 || byteLength > THUMBNAIL_MAX_BYTES) fail("thumbnail byteLength is outside the supported range.");
    if (typeof item.sha256 !== "string" || !/^[a-f0-9]{64}$/i.test(item.sha256)) fail("thumbnail sha256 must be a SHA-256 digest.");
    return Object.freeze({
      index,
      requestedTimeSeconds,
      url: resolveHttpUrl(item.url, responseUrl, `thumbnails.items[${position}].url`),
      byteLength,
      sha256: item.sha256.toLowerCase(),
    });
  });
  if (thumbnails.status === "available") {
    if (thumbnails.unavailableItems !== undefined && (!Array.isArray(thumbnails.unavailableItems) || thumbnails.unavailableItems.length)) {
      fail("available thumbnails must not contain unavailableItems.");
    }
    if (coveredIndices.size !== timesSeconds.length) fail("available thumbnails must cover every TRIBE frame exactly once.");
    return Object.freeze({ ...thumbnails, items: Object.freeze(items), unavailableItems: Object.freeze([]) });
  }

  if (!Array.isArray(thumbnails.unavailableItems) || !thumbnails.unavailableItems.length) {
    fail("partial thumbnails must disclose unavailableItems.");
  }
  const unavailableItems = thumbnails.unavailableItems.map((item, position) => {
    if (!isRecord(item)) fail(`thumbnails.unavailableItems[${position}] must be an object.`);
    const index = integer(item.index, `thumbnails.unavailableItems[${position}].index`);
    if (index < 0 || index >= timesSeconds.length) fail("unavailable thumbnail index is outside the TRIBE frame range.");
    if (coveredIndices.has(index)) fail("available and unavailable thumbnail indices must not overlap or repeat.");
    coveredIndices.add(index);
    const requestedTimeSeconds = finite(item.requestedTimeSeconds, `thumbnails.unavailableItems[${position}].requestedTimeSeconds`);
    if (Math.abs(requestedTimeSeconds - timesSeconds[index]) > 1e-9) fail("unavailable thumbnail requested time must match its TRIBE interval start.");
    if (item.reason !== "outside-source-video") fail("partial thumbnail unavailability must be outside-source-video.");
    return Object.freeze({ index, requestedTimeSeconds, reason: item.reason });
  });
  if (coveredIndices.size !== timesSeconds.length) fail("partial thumbnail partition must cover every TRIBE frame exactly once.");
  return Object.freeze({ ...thumbnails, items: Object.freeze(items), unavailableItems: Object.freeze(unavailableItems) });
}

export function validateTribeManifest(input, responseUrl) {
  if (!isRecord(input)) fail("manifest must be an object.");
  if (input.schemaVersion !== "tribe-fsaverage5/1") fail("schemaVersion must be tribe-fsaverage5/1.");
  if (typeof input.resultId !== "string" || !/^[a-f0-9]{32}$/i.test(input.resultId)) fail("resultId must be a 32-character UUID hex value.");
  if (input.predictionType !== "average-subject cortical BOLD response") fail("predictionType must describe the released TRIBE cortical target.");
  if (input.isViralityScore !== false) fail("isViralityScore must be false.");
  if (!isRecord(input.model) || input.model.id !== "facebook/tribev2") fail("model.id must be facebook/tribev2.");
  if (input.model.revision !== TRIBE_MODEL_REVISION) fail("model.revision does not match the pinned official checkpoint snapshot.");
  if (input.model.weightsSha256 !== TRIBE_MODEL_WEIGHTS_SHA256) fail("model.weightsSha256 does not match the audited TRIBE checkpoint.");
  if (input.model.codeRevision !== TRIBE_CODE_REVISION) fail("model.codeRevision does not match the audited official TRIBE source revision.");
  if (input.model.license !== "CC-BY-NC-4.0") fail("model.license must disclose CC-BY-NC-4.0.");
  if (!isRecord(input.extractors) || !isRecord(input.extractors.video)) fail("extractors.video provenance is required.");
  if (input.extractors.video.id !== TRIBE_VIDEO_MODEL_ID || input.extractors.video.revision !== TRIBE_VIDEO_MODEL_REVISION || input.extractors.video.weightsSha256 !== TRIBE_VIDEO_MODEL_WEIGHTS_SHA256) {
    fail("extractors.video does not match the pinned official V-JEPA2 snapshot.");
  }
  const videoDevice = input.extractors.video.device;
  const videoPrecision = input.extractors.video.precision;
  const hasExecutionPolicy = videoDevice !== undefined || videoPrecision !== undefined;
  if (hasExecutionPolicy) {
    if (!TRIBE_VIDEO_DEVICES.has(videoDevice) || !TRIBE_VIDEO_PRECISIONS.has(videoPrecision)) {
      fail("extractors.video execution policy is incomplete or unsupported.");
    }
    if (videoPrecision === "autocast-fp16" && videoDevice !== "mps") {
      fail("extractors.video autocast-fp16 requires the MPS device.");
    }
  }
  const inference = validateInferenceMode(input);

  const surface = input.surface;
  if (!isRecord(surface) || surface.space !== "fsaverage5") fail("surface.space must be fsaverage5.");
  if (surface.mappingId !== TRIBE_MESH_MAPPING_ID) fail("surface.mappingId does not match the bundled vertex ordering.");
  if (integer(surface.vertexCount, "surface.vertexCount") !== TRIBE_VERTEX_COUNT) {
    fail(`surface.vertexCount must be ${TRIBE_VERTEX_COUNT}.`);
  }
  validateHemispheres(surface.hemispheres);

  const frames = input.frames;
  if (!isRecord(frames)) fail("frames is required.");
  if (frames.dtype !== "float32" || frames.byteOrder !== "little-endian" || frames.layout !== "time-major") {
    fail("frames must use little-endian, time-major float32 values.");
  }
  if (!Array.isArray(frames.shape) || frames.shape.length !== 2) fail("frames.shape must be [time, vertex].");
  const frameCount = integer(frames.shape[0], "frames.shape[0]");
  const vertexCount = integer(frames.shape[1], "frames.shape[1]");
  if (frameCount < 1) fail("at least one prediction frame is required.");
  if (vertexCount !== TRIBE_VERTEX_COUNT) fail(`each prediction frame must contain ${TRIBE_VERTEX_COUNT} vertices.`);
  const expectedByteLength = frameCount * vertexCount * Float32Array.BYTES_PER_ELEMENT;
  if (integer(frames.byteLength, "frames.byteLength") !== expectedByteLength) fail("frames.byteLength does not match frames.shape.");
  if (typeof frames.sha256 !== "string" || !/^[a-f0-9]{64}$/i.test(frames.sha256)) fail("frames.sha256 must be a SHA-256 digest.");
  const timesSeconds = validateTimes(frames.timesSeconds, frameCount);
  const durationsSeconds = validateDurations(frames.durationsSeconds, timesSeconds);
  const hemodynamicOffsetSeconds = finite(frames.hemodynamicOffsetSeconds, "frames.hemodynamicOffsetSeconds");
  if (hemodynamicOffsetSeconds !== 5) fail("frames.hemodynamicOffsetSeconds must match the published 5-second model configuration.");

  const displayScale = frames.displayScale;
  if (!isRecord(displayScale) || displayScale.type !== "robust-diverging-linear" || displayScale.clamp !== true) {
    fail("frames.displayScale must be a fixed, clamped robust-diverging-linear scale.");
  }
  if (displayScale.scope !== "all-frames-all-vertices") {
    fail("frames.displayScale must be computed once across the complete prediction tensor.");
  }
  if (!Array.isArray(displayScale.domain) || displayScale.domain.length !== 3) fail("frames.displayScale.domain must contain [minimum, midpoint, maximum].");
  const domain = displayScale.domain.map((value, index) => finite(value, `frames.displayScale.domain[${index}]`));
  if (!(domain[0] < domain[1] && domain[1] < domain[2])) fail("frames.displayScale.domain must be strictly increasing.");

  const thumbnails = validateThumbnails(input.thumbnails, timesSeconds, responseUrl);

  return Object.freeze({
    ...input,
    ...inference,
    extractors: Object.freeze({ video: Object.freeze({ ...input.extractors.video }) }),
    surface: Object.freeze({ ...surface, hemispheres: surface.hemispheres.map((item) => Object.freeze({ ...item })) }),
    frames: Object.freeze({
      ...frames,
      url: resolveFramesUrl(frames.url, responseUrl),
      shape: Object.freeze([frameCount, vertexCount]),
      timesSeconds: Object.freeze(timesSeconds),
      durationsSeconds: Object.freeze(durationsSeconds),
      hemodynamicOffsetSeconds,
      displayScale: Object.freeze({ ...displayScale, domain: Object.freeze(domain) }),
    }),
    ...(thumbnails ? { thumbnails } : {}),
  });
}
