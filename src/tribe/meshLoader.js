import {
  TRIBE_HEMISPHERE_VERTEX_COUNT,
  TRIBE_MESH_MAPPING_ID,
  TRIBE_VERTEX_COUNT,
} from "./contract.js";

export const FSAVERAGE5_MESH_METADATA_URL = "/assets/brain/fsaverage5-half.json";
export const FSAVERAGE5_MESH_SHA256 = TRIBE_MESH_MAPPING_ID.split(":")[1];

const FORMAT = "SFB1";
const SCHEMA_VERSION = "signalframe-fsaverage5-mesh/1";
const TRIANGLE_COUNT = 40_960;
const HEADER_BYTES = 32;
const POSITIONS_OFFSET = HEADER_BYTES;
const POSITION_VALUE_COUNT = TRIBE_VERTEX_COUNT * 3;
const INDICES_OFFSET = POSITIONS_OFFSET + POSITION_VALUE_COUNT * Float32Array.BYTES_PER_ELEMENT;
const INDEX_VALUE_COUNT = TRIANGLE_COUNT * 3;
const BINARY_BYTE_LENGTH = INDICES_OFFSET + INDEX_VALUE_COUNT * Uint32Array.BYTES_PER_ELEMENT;
const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

function bytesToHex(bytes) {
  return [...bytes].map((value) => value.toString(16).padStart(2, "0")).join("");
}

function digestSource(value) {
  if (value instanceof ArrayBuffer) return value;
  if (ArrayBuffer.isView(value)) {
    return value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
  }
  throw new TypeError("SHA-256 input must be an ArrayBuffer or typed-array view.");
}

export async function sha256Hex(value) {
  if (!globalThis.crypto?.subtle) {
    throw new Error("This browser cannot verify cortical asset integrity.");
  }
  const digest = await globalThis.crypto.subtle.digest("SHA-256", digestSource(value));
  return bytesToHex(new Uint8Array(digest));
}

function exactNumberArray(value, expected, label) {
  if (!Array.isArray(value) || value.length !== expected.length) {
    throw new Error(`The fsaverage5 metadata ${label} is invalid.`);
  }
  for (let index = 0; index < expected.length; index += 1) {
    if (!Number.isFinite(value[index]) || value[index] !== expected[index]) {
      throw new Error(`The fsaverage5 metadata ${label} is invalid.`);
    }
  }
  return Object.freeze([...value]);
}

export function validateFsaverage5MeshMetadata(input) {
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    throw new Error("The fsaverage5 mesh metadata is missing.");
  }
  if (input.schemaVersion !== SCHEMA_VERSION || input.format !== FORMAT) {
    throw new Error("Unsupported fsaverage5 mesh metadata format.");
  }
  if (
    input.mappingId !== TRIBE_MESH_MAPPING_ID
    || input.binarySha256 !== FSAVERAGE5_MESH_SHA256
    || input.binary !== "fsaverage5-half.bin"
  ) {
    throw new Error("The fsaverage5 metadata does not match the audited cortical mapping.");
  }
  if (
    input.vertexCount !== TRIBE_VERTEX_COUNT
    || input.leftVertexCount !== TRIBE_HEMISPHERE_VERTEX_COUNT
    || input.rightVertexCount !== TRIBE_HEMISPHERE_VERTEX_COUNT
    || input.triangleCount !== TRIANGLE_COUNT
  ) {
    throw new Error("The fsaverage5 metadata vertex or triangle counts are invalid.");
  }
  if (!Array.isArray(input.ordering) || input.ordering.length !== 2 || input.ordering[0] !== "left" || input.ordering[1] !== "right") {
    throw new Error("The fsaverage5 metadata vertex ordering does not match TRIBE v2.");
  }
  if (
    input.byteOrder !== "little-endian"
    || input.headerBytes !== HEADER_BYTES
    || input.positionsOffset !== POSITIONS_OFFSET
    || input.indicesOffset !== INDICES_OFFSET
    || input.byteLength !== BINARY_BYTE_LENGTH
  ) {
    throw new Error("The fsaverage5 metadata binary layout is invalid.");
  }

  const bounds = input.bounds;
  if (!bounds || typeof bounds !== "object") {
    throw new Error("The fsaverage5 metadata bounds are missing.");
  }
  const min = exactNumberArray(
    bounds.min,
    [-75.177978515625, -106.34244537353516, -60.84112548828125],
    "minimum bounds",
  );
  const max = exactNumberArray(
    bounds.max,
    [75.1108169555664, 88.21171569824219, 75.89260864257812],
    "maximum bounds",
  );

  return Object.freeze({
    ...input,
    ordering: Object.freeze([...input.ordering]),
    bounds: Object.freeze({ min, max }),
  });
}

function readFloat32Array(buffer, byteOffset, length) {
  if (LITTLE_ENDIAN) return new Float32Array(buffer, byteOffset, length).slice();
  const view = new DataView(buffer);
  const output = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    output[index] = view.getFloat32(byteOffset + index * Float32Array.BYTES_PER_ELEMENT, true);
  }
  return output;
}

function readUint32Array(buffer, byteOffset, length) {
  if (LITTLE_ENDIAN) return new Uint32Array(buffer, byteOffset, length).slice();
  const view = new DataView(buffer);
  const output = new Uint32Array(length);
  for (let index = 0; index < length; index += 1) {
    output[index] = view.getUint32(byteOffset + index * Uint32Array.BYTES_PER_ELEMENT, true);
  }
  return output;
}

export function parseFsaverage5Mesh(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < HEADER_BYTES) {
    throw new Error("The fsaverage5 mesh asset is incomplete.");
  }

  const view = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  const vertexCount = view.getUint32(4, true);
  const triangleCount = view.getUint32(8, true);
  const leftVertexCount = view.getUint32(12, true);
  const positionsOffset = view.getUint32(16, true);
  const indicesOffset = view.getUint32(20, true);
  const version = view.getUint32(24, true);
  const reserved = view.getUint32(28, true);

  if (magic !== FORMAT || version !== 1 || reserved !== 0) {
    throw new Error("Unsupported fsaverage5 mesh format.");
  }
  if (vertexCount !== TRIBE_VERTEX_COUNT || leftVertexCount !== TRIBE_HEMISPHERE_VERTEX_COUNT) {
    throw new Error("The fsaverage5 mesh vertex ordering does not match TRIBE v2.");
  }
  if (triangleCount !== TRIANGLE_COUNT) {
    throw new Error("The fsaverage5 mesh triangle count is invalid.");
  }
  if (positionsOffset !== POSITIONS_OFFSET || indicesOffset !== INDICES_OFFSET) {
    throw new Error("The fsaverage5 mesh offsets are invalid.");
  }
  if (buffer.byteLength !== BINARY_BYTE_LENGTH) {
    throw new Error("The fsaverage5 mesh byte length is invalid.");
  }

  const positions = readFloat32Array(buffer, positionsOffset, POSITION_VALUE_COUNT);
  const indices = readUint32Array(buffer, indicesOffset, INDEX_VALUE_COUNT);
  for (let index = 0; index < positions.length; index += 1) {
    if (!Number.isFinite(positions[index])) {
      throw new Error(`The fsaverage5 mesh has a non-finite position at ${index}.`);
    }
  }
  for (let index = 0; index < indices.length; index += 1) {
    if (indices[index] >= vertexCount) {
      throw new Error(`The fsaverage5 mesh has an out-of-range index at ${index}.`);
    }
  }

  return Object.freeze({
    positions,
    indices,
    vertexCount,
    triangleCount,
    leftVertexCount,
    rightVertexCount: vertexCount - leftVertexCount,
  });
}

function responseBaseUrl(response, requestedUrl) {
  if (response.url) return response.url;
  const location = globalThis.location?.href || "http://localhost/";
  return new URL(requestedUrl, location).href;
}

export async function loadFsaverage5Mesh(metadataUrl = FSAVERAGE5_MESH_METADATA_URL, signal) {
  const metadataResponse = await fetch(metadataUrl, { signal, cache: "force-cache" });
  if (!metadataResponse.ok) {
    throw new Error(`Could not load the fsaverage5 metadata (${metadataResponse.status}).`);
  }
  const metadata = validateFsaverage5MeshMetadata(await metadataResponse.json());
  const binaryUrl = new URL(metadata.binary, responseBaseUrl(metadataResponse, metadataUrl)).href;
  const binaryResponse = await fetch(binaryUrl, { signal, cache: "force-cache" });
  if (!binaryResponse.ok) {
    throw new Error(`Could not load the fsaverage5 surface (${binaryResponse.status}).`);
  }
  const buffer = await binaryResponse.arrayBuffer();
  if ((await sha256Hex(buffer)) !== metadata.binarySha256) {
    throw new Error("The fsaverage5 surface failed its SHA-256 integrity check.");
  }
  return Object.freeze({ ...parseFsaverage5Mesh(buffer), metadata });
}
