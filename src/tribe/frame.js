import { TRIBE_VERTEX_COUNT } from "./contract.js";

const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
export const NEUTRAL_VERTEX_COLOR = Object.freeze([0.17, 0.18, 0.18]);

export function decodePredictionBuffer(buffer, manifest) {
  if (!(buffer instanceof ArrayBuffer)) throw new Error("TRIBE prediction payload must be an ArrayBuffer.");
  if (buffer.byteLength !== manifest.frames.byteLength) {
    throw new Error(`TRIBE prediction payload is ${buffer.byteLength} bytes; expected ${manifest.frames.byteLength}.`);
  }

  let values;
  if (LITTLE_ENDIAN) {
    values = new Float32Array(buffer);
  } else {
    const view = new DataView(buffer);
    values = new Float32Array(buffer.byteLength / 4);
    for (let index = 0; index < values.length; index += 1) values[index] = view.getFloat32(index * 4, true);
  }
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) throw new Error(`TRIBE prediction contains a non-finite value at index ${index}.`);
  }
  return values;
}

export function frameIndexAtMediaTime(mediaTime, timesSeconds, durationsSeconds) {
  const time = Number(mediaTime);
  if (!Number.isFinite(time) || !timesSeconds.length || time < timesSeconds[0]) return null;
  let low = 0;
  let high = timesSeconds.length - 1;
  while (low <= high) {
    const middle = (low + high) >> 1;
    if (timesSeconds[middle] <= time) low = middle + 1;
    else high = middle - 1;
  }
  const index = Math.max(0, high);
  if (Array.isArray(durationsSeconds) && durationsSeconds.length === timesSeconds.length) {
    const duration = durationsSeconds[index];
    if (!Number.isFinite(duration) || duration <= 0 || time >= timesSeconds[index] + duration) return null;
  }
  return index;
}

export function predictionFrameAtMediaTime(values, manifest, mediaTime) {
  const index = frameIndexAtMediaTime(mediaTime, manifest.frames.timesSeconds, manifest.frames.durationsSeconds);
  if (index === null) return { index: null, frame: null };
  const start = index * TRIBE_VERTEX_COUNT;
  return { index, frame: values.subarray(start, start + TRIBE_VERTEX_COUNT) };
}

export function splitHemisphereFrame(frame) {
  if (!(frame instanceof Float32Array) || frame.length !== TRIBE_VERTEX_COUNT) {
    throw new Error(`A TRIBE frame must contain ${TRIBE_VERTEX_COUNT} float32 values.`);
  }
  return {
    left: frame.subarray(0, TRIBE_VERTEX_COUNT / 2),
    right: frame.subarray(TRIBE_VERTEX_COUNT / 2),
  };
}

function lerp(a, b, amount) {
  return a + (b - a) * amount;
}

function mixColor(from, to, amount, target, offset) {
  target[offset] = lerp(from[0], to[0], amount);
  target[offset + 1] = lerp(from[1], to[1], amount);
  target[offset + 2] = lerp(from[2], to[2], amount);
}

export function writeScientificColors(frame, domain, target) {
  if (!(target instanceof Float32Array) || target.length !== TRIBE_VERTEX_COUNT * 3) {
    throw new Error("The cortical color buffer has the wrong size.");
  }
  const [minimum, midpoint, maximum] = domain;
  const negative = [0.43, 0.25, 0.98];
  const neutral = NEUTRAL_VERTEX_COLOR;
  const positive = [1, 0.93, 0.25];
  for (let index = 0; index < frame.length; index += 1) {
    const value = Math.max(minimum, Math.min(maximum, frame[index]));
    const offset = index * 3;
    if (value <= midpoint) {
      const amount = (value - minimum) / Math.max(Number.EPSILON, midpoint - minimum);
      mixColor(negative, neutral, amount, target, offset);
    } else {
      const amount = (value - midpoint) / Math.max(Number.EPSILON, maximum - midpoint);
      mixColor(neutral, positive, amount, target, offset);
    }
  }
  return target;
}

export function writeNeutralColors(target) {
  for (let index = 0; index < target.length; index += 3) {
    target[index] = NEUTRAL_VERTEX_COLOR[0];
    target[index + 1] = NEUTRAL_VERTEX_COLOR[1];
    target[index + 2] = NEUTRAL_VERTEX_COLOR[2];
  }
  return target;
}
