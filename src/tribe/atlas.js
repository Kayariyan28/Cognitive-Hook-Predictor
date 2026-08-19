import { TRIBE_VERTEX_COUNT } from "./contract.js";
import { sha256Hex } from "./meshLoader.js";

export const DESTRIEUX_MAPPING_SHA256 = "4eb0866387418d2916a3d5d5441843de2edb8ff4e80d8a5b20a2c192d00b32cc";
export const DESTRIEUX_SCHEMA_VERSION = "destrieux-fsaverage5/1";
const VERTICES_PER_HEMISPHERE = TRIBE_VERTEX_COUNT / 2;
let atlasPromise;

function validateMetadata(metadata) {
  if (!metadata || metadata.schemaVersion !== DESTRIEUX_SCHEMA_VERSION) throw new Error("The cortical atlas schema is not supported.");
  if (metadata.space !== "fsaverage5" || metadata.vertexCount !== TRIBE_VERTEX_COUNT) throw new Error("The cortical atlas does not match fsaverage5.");
  if (metadata.verticesPerHemisphere !== VERTICES_PER_HEMISPHERE) throw new Error("The cortical atlas hemisphere split is invalid.");
  if (metadata.dtype !== "uint8" || metadata.layout !== "left-then-right") throw new Error("The cortical atlas binary layout is invalid.");
  if (metadata.mappingSha256 !== DESTRIEUX_MAPPING_SHA256) throw new Error("The cortical atlas mapping identifier is not recognized.");
  if (!Array.isArray(metadata.labels) || metadata.labels.length !== metadata.labelCount || metadata.labels.length > 255) {
    throw new Error("The cortical atlas label table is invalid.");
  }
  return Object.freeze({ ...metadata, labels: Object.freeze([...metadata.labels]) });
}

async function loadAtlas(signal) {
  const [metadataResponse, mappingResponse] = await Promise.all([
    fetch("/assets/brain/destrieux-fsaverage5.json", { signal, cache: "force-cache" }),
    fetch("/assets/brain/destrieux-fsaverage5.u8", { signal, cache: "force-cache" }),
  ]);
  if (!metadataResponse.ok || !mappingResponse.ok) throw new Error("The verified cortical atlas could not be loaded.");
  const metadata = validateMetadata(await metadataResponse.json());
  const buffer = await mappingResponse.arrayBuffer();
  if (buffer.byteLength !== TRIBE_VERTEX_COUNT) throw new Error("The cortical atlas has the wrong vertex count.");
  if ((await sha256Hex(buffer)) !== metadata.mappingSha256) throw new Error("The cortical atlas failed its SHA-256 integrity check.");
  const parcelIds = new Uint8Array(buffer);
  for (let index = 0; index < parcelIds.length; index += 1) {
    if (parcelIds[index] >= metadata.labelCount) throw new Error(`The cortical atlas has an invalid label at vertex ${index}.`);
  }
  return Object.freeze({ metadata, parcelIds });
}

export function loadDestrieuxAtlas(signal) {
  if (signal) return loadAtlas(signal);
  atlasPromise ||= loadAtlas();
  return atlasPromise;
}

export function atlasParcelKey(hemisphere, labelIndex) {
  return `${hemisphere}:${labelIndex}`;
}

export function displayAtlasLabel(value) {
  return String(value || "Unknown").replaceAll("_", " ").replaceAll("-", " · ");
}

export function summarizeFrameByParcel(frame, atlas, limit = 6) {
  if (!(frame instanceof Float32Array) || frame.length !== TRIBE_VERTEX_COUNT) throw new Error("A full TRIBE frame is required for parcel summaries.");
  if (!atlas?.parcelIds || atlas.parcelIds.length !== TRIBE_VERTEX_COUNT) throw new Error("A verified fsaverage5 atlas is required for parcel summaries.");
  const labelCount = atlas.metadata.labels.length;
  const sums = new Float64Array(labelCount * 2);
  const counts = new Uint32Array(labelCount * 2);
  for (let vertex = 0; vertex < frame.length; vertex += 1) {
    const hemisphereIndex = vertex < VERTICES_PER_HEMISPHERE ? 0 : 1;
    const labelIndex = atlas.parcelIds[vertex];
    const slot = hemisphereIndex * labelCount + labelIndex;
    sums[slot] += frame[vertex];
    counts[slot] += 1;
  }

  const parcels = [];
  for (let hemisphereIndex = 0; hemisphereIndex < 2; hemisphereIndex += 1) {
    for (let labelIndex = 1; labelIndex < labelCount; labelIndex += 1) {
      const name = atlas.metadata.labels[labelIndex];
      if (name === "Medial_wall") continue;
      const slot = hemisphereIndex * labelCount + labelIndex;
      if (!counts[slot]) continue;
      const hemisphere = hemisphereIndex === 0 ? "left" : "right";
      parcels.push(Object.freeze({
        key: atlasParcelKey(hemisphere, labelIndex),
        hemisphere,
        labelIndex,
        name,
        vertexCount: counts[slot],
        mean: sums[slot] / counts[slot],
      }));
    }
  }
  parcels.sort((a, b) => b.mean - a.mean || b.vertexCount - a.vertexCount || a.key.localeCompare(b.key));
  return Object.freeze(parcels.slice(0, Math.max(1, Math.floor(limit))));
}

export function applyParcelFocus(colors, atlas, focus, dimAmount = 0.24) {
  if (!focus || !atlas?.parcelIds) return colors;
  const offset = focus.hemisphere === "right" ? VERTICES_PER_HEMISPHERE : 0;
  const end = offset + VERTICES_PER_HEMISPHERE;
  for (let vertex = 0; vertex < TRIBE_VERTEX_COUNT; vertex += 1) {
    const selected = vertex >= offset && vertex < end && atlas.parcelIds[vertex] === focus.labelIndex;
    if (selected) continue;
    const colorOffset = vertex * 3;
    colors[colorOffset] *= dimAmount;
    colors[colorOffset + 1] *= dimAmount;
    colors[colorOffset + 2] *= dimAmount;
  }
  return colors;
}
