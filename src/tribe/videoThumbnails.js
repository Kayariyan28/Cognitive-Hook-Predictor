export const VIDEO_THUMBNAIL_CONCURRENCY = 1;

function finiteNumber(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number)) throw new TypeError(`${label} must be finite.`);
  return number;
}

export function planVideoIntervalThumbnails(intervals) {
  if (!Array.isArray(intervals)) throw new TypeError("intervals must be an array.");
  const seen = new Set();
  const requests = intervals.map((interval, position) => {
    const index = finiteNumber(interval?.index, `intervals[${position}].index`);
    const startTime = finiteNumber(interval?.startTime, `intervals[${position}].startTime`);
    if (!Number.isInteger(index) || index < 0) throw new RangeError(`intervals[${position}].index must be a non-negative integer.`);
    if (startTime < 0) throw new RangeError(`intervals[${position}].startTime must be non-negative.`);
    if (seen.has(index)) throw new RangeError(`interval index ${index} is duplicated.`);
    seen.add(index);
    return Object.freeze({ index, startTime });
  });
  requests.sort((left, right) => left.startTime - right.startTime || left.index - right.index);
  return Object.freeze(requests);
}

export function thumbnailTimeAvailability(startTime, sourceDuration) {
  const time = finiteNumber(startTime, "startTime");
  const duration = finiteNumber(sourceDuration, "sourceDuration");
  if (time < 0) throw new RangeError("startTime must be non-negative.");
  if (duration <= 0 || time >= duration) {
    return Object.freeze({ available: false, reason: "outside-source-duration" });
  }
  return Object.freeze({ available: true, reason: null });
}

function abortError() {
  return new DOMException("Thumbnail extraction was aborted.", "AbortError");
}

function waitForMediaEvent(target, successEvent, failureEvents, signal) {
  if (signal?.aborted) return Promise.reject(signal.reason || abortError());
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      target.removeEventListener(successEvent, succeed);
      failureEvents.forEach((eventName) => target.removeEventListener(eventName, fail));
      signal?.removeEventListener("abort", abort);
    };
    const succeed = () => { cleanup(); resolve(); };
    const fail = () => { cleanup(); reject(new Error("The source video could not be decoded for thumbnails.")); };
    const abort = () => { cleanup(); reject(signal.reason || abortError()); };
    target.addEventListener(successEvent, succeed, { once: true });
    failureEvents.forEach((eventName) => target.addEventListener(eventName, fail, { once: true }));
    signal?.addEventListener("abort", abort, { once: true });
  });
}

function canvasBlob(canvas, mimeType, quality, signal) {
  if (signal?.aborted) return Promise.reject(signal.reason || abortError());
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (signal?.aborted) reject(signal.reason || abortError());
      else if (blob) resolve(blob);
      else reject(new Error("The browser could not encode the source-video thumbnail."));
    }, mimeType, quality);
  });
}

/**
 * Extracts source-video frames serially with one decoder. The requested times are
 * the exact published TRIBE interval starts; they are never shifted or clamped.
 */
export async function extractVideoIntervalThumbnails({
  sourceUrl,
  intervals,
  signal,
  onThumbnail,
  onUnavailable,
  maxWidth = 180,
  maxHeight = 112,
  mimeType = "image/webp",
  quality = 0.82,
  documentRef = globalThis.document,
}) {
  if (!sourceUrl) throw new TypeError("sourceUrl is required.");
  if (!documentRef?.createElement) throw new Error("Video thumbnail extraction requires a browser document.");
  const requests = planVideoIntervalThumbnails(intervals);
  if (!requests.length) return Object.freeze({ duration: 0, extracted: 0, unavailable: 0 });
  if (signal?.aborted) throw signal.reason || abortError();

  const video = documentRef.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.src = sourceUrl;

  try {
    if (video.readyState < 1) {
      const metadataReady = waitForMediaEvent(video, "loadedmetadata", ["error", "abort"], signal);
      video.load();
      await metadataReady;
    }
    const duration = finiteNumber(video.duration, "source video duration");
    const naturalWidth = finiteNumber(video.videoWidth, "source video width");
    const naturalHeight = finiteNumber(video.videoHeight, "source video height");
    if (duration <= 0 || naturalWidth <= 0 || naturalHeight <= 0) throw new Error("The source video has invalid media dimensions or duration.");

    const scale = Math.min(1, maxWidth / naturalWidth, maxHeight / naturalHeight);
    const width = Math.max(1, Math.round(naturalWidth * scale));
    const height = Math.max(1, Math.round(naturalHeight * scale));
    let extracted = 0;
    let unavailable = 0;

    for (const request of requests) {
      if (signal?.aborted) throw signal.reason || abortError();
      const availability = thumbnailTimeAvailability(request.startTime, duration);
      if (!availability.available) {
        unavailable += 1;
        onUnavailable?.({ ...request, reason: availability.reason, duration });
        continue;
      }

      if (Math.abs(video.currentTime - request.startTime) > 0.001) {
        const seeked = waitForMediaEvent(video, "seeked", ["error", "abort"], signal);
        video.currentTime = request.startTime;
        await seeked;
      } else if (video.readyState < 2) {
        await waitForMediaEvent(video, "loadeddata", ["error", "abort"], signal);
      }
      if (signal?.aborted) throw signal.reason || abortError();

      const canvas = documentRef.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext("2d", { alpha: false });
      if (!context) throw new Error("The browser could not create a thumbnail canvas context.");
      context.drawImage(video, 0, 0, width, height);
      const blob = await canvasBlob(canvas, mimeType, quality, signal);
      extracted += 1;
      onThumbnail?.({
        ...request,
        actualTime: video.currentTime,
        width,
        height,
        mimeType: blob.type || mimeType,
        blob,
      });
    }

    return Object.freeze({ duration, extracted, unavailable });
  } finally {
    video.removeAttribute("src");
    video.load();
  }
}

function jpegBytesAreComplete(bytes) {
  return bytes.byteLength > 4
    && bytes[0] === 0xff
    && bytes[1] === 0xd8
    && bytes[bytes.byteLength - 2] === 0xff
    && bytes[bytes.byteLength - 1] === 0xd9;
}

export async function fetchVerifiedVideoThumbnails({
  thumbnails,
  signal,
  onThumbnail,
  onUnavailable,
  fetchRef = globalThis.fetch,
  cryptoRef = globalThis.crypto,
}) {
  if (!thumbnails?.items?.length) throw new TypeError("Verified thumbnail metadata is required.");
  if (typeof fetchRef !== "function" || !cryptoRef?.subtle) throw new Error("This browser cannot verify server thumbnail bytes.");
  for (const item of thumbnails.unavailableItems || []) onUnavailable?.({ ...item, startTime: item.requestedTimeSeconds });
  let extracted = 0;
  for (const item of thumbnails.items) {
    if (signal?.aborted) throw signal.reason || abortError();
    const response = await fetchRef(item.url, { signal, cache: "no-store" });
    if (!response.ok) throw new Error(`Source thumbnail ${item.index + 1} could not be retrieved (${response.status}).`);
    const buffer = await response.arrayBuffer();
    if (buffer.byteLength !== item.byteLength) throw new Error(`Source thumbnail ${item.index + 1} failed its byte-length check.`);
    const bytes = new Uint8Array(buffer);
    if (!jpegBytesAreComplete(bytes)) throw new Error(`Source thumbnail ${item.index + 1} is not a complete JPEG.`);
    const digestBuffer = await cryptoRef.subtle.digest("SHA-256", buffer);
    const digest = Array.from(new Uint8Array(digestBuffer), (byte) => byte.toString(16).padStart(2, "0")).join("");
    if (digest !== item.sha256.toLowerCase()) throw new Error(`Source thumbnail ${item.index + 1} failed its SHA-256 check.`);
    const blob = new Blob([buffer], { type: thumbnails.format });
    extracted += 1;
    onThumbnail?.({
      index: item.index,
      startTime: item.requestedTimeSeconds,
      requestedTime: item.requestedTimeSeconds,
      actualTime: null,
      mimeType: thumbnails.format,
      byteLength: item.byteLength,
      sha256: item.sha256,
      blob,
      verified: true,
    });
  }
  return Object.freeze({ extracted, unavailable: thumbnails.unavailableItems?.length || 0, verified: true });
}
