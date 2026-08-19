import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(scriptDirectory, "..");
const sourceRoot = resolve(projectRoot, "../../work/destrieux");
const outputRoot = resolve(projectRoot, "public/assets/brain");
const vertexCountPerHemisphere = 10_242;
const unknown = 255;

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function parseAnnotation(buffer, hemisphere) {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  let offset = 0;
  const readInt = () => {
    if (offset + 4 > view.byteLength) throw new Error(`${hemisphere} annotation ended unexpectedly.`);
    const value = view.getInt32(offset, false);
    offset += 4;
    return value;
  };
  const readString = () => {
    const length = readInt();
    if (length < 0 || offset + length > view.byteLength) throw new Error(`${hemisphere} annotation contains an invalid string.`);
    const value = new TextDecoder().decode(buffer.subarray(offset, offset + length)).replace(/\0+$/, "");
    offset += length;
    return value;
  };

  const vertexCount = readInt();
  if (vertexCount !== vertexCountPerHemisphere) {
    throw new Error(`${hemisphere} atlas has ${vertexCount} vertices; expected ${vertexCountPerHemisphere}.`);
  }

  const annotations = new Uint32Array(vertexCount);
  const seen = new Uint8Array(vertexCount);
  for (let row = 0; row < vertexCount; row += 1) {
    const vertex = readInt();
    const annotation = readInt() >>> 0;
    if (vertex < 0 || vertex >= vertexCount || seen[vertex]) throw new Error(`${hemisphere} atlas has an invalid vertex index.`);
    annotations[vertex] = annotation;
    seen[vertex] = 1;
  }

  if (readInt() !== 1) throw new Error(`${hemisphere} atlas has no embedded color table.`);
  const versionMarker = readInt();
  if (versionMarker !== -2) throw new Error(`${hemisphere} atlas uses unsupported color-table version ${versionMarker}.`);
  const labelCapacity = readInt();
  readString(); // Original FreeSurfer color-table path; provenance only.
  const entryCount = readInt();
  if (labelCapacity <= 0 || labelCapacity > unknown || entryCount <= 0 || entryCount > labelCapacity) {
    throw new Error(`${hemisphere} atlas has an invalid color table.`);
  }

  const labels = Array(labelCapacity).fill(null);
  const annotationToIndex = new Map();
  for (let row = 0; row < entryCount; row += 1) {
    const labelIndex = readInt();
    const name = readString();
    const red = readInt();
    const green = readInt();
    const blue = readInt();
    const alpha = readInt();
    if (labelIndex < 0 || labelIndex >= labelCapacity || labels[labelIndex] !== null) {
      throw new Error(`${hemisphere} atlas has an invalid label index.`);
    }
    for (const channel of [red, green, blue, alpha]) {
      if (channel < 0 || channel > 255) throw new Error(`${hemisphere} atlas has an invalid color channel.`);
    }
    labels[labelIndex] = name;
    annotationToIndex.set((red + (green << 8) + (blue << 16) + (alpha << 24)) >>> 0, labelIndex);
  }

  const parcelIds = new Uint8Array(vertexCount);
  parcelIds.fill(unknown);
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const labelIndex = annotationToIndex.get(annotations[vertex]);
    if (labelIndex === undefined) throw new Error(`${hemisphere} vertex ${vertex} references an unknown atlas color.`);
    parcelIds[vertex] = labelIndex;
  }
  if (offset !== view.byteLength) throw new Error(`${hemisphere} annotation has ${view.byteLength - offset} unexpected trailing bytes.`);
  if (labels.some((label) => typeof label !== "string" || !label)) throw new Error(`${hemisphere} atlas has a missing label.`);
  return { parcelIds, labels };
}

const sources = {
  left: {
    path: resolve(sourceRoot, "lh.aparc.a2009s.annot"),
    url: "https://www.nitrc.org/frs/download.php/9343/lh.aparc.a2009s.annot",
  },
  right: {
    path: resolve(sourceRoot, "rh.aparc.a2009s.annot"),
    url: "https://www.nitrc.org/frs/download.php/9342/rh.aparc.a2009s.annot",
  },
};

const leftBytes = await readFile(sources.left.path);
const rightBytes = await readFile(sources.right.path);
const left = parseAnnotation(leftBytes, "left");
const right = parseAnnotation(rightBytes, "right");
if (JSON.stringify(left.labels) !== JSON.stringify(right.labels)) {
  throw new Error("Left and right Destrieux label tables do not match.");
}

const combined = new Uint8Array(vertexCountPerHemisphere * 2);
combined.set(left.parcelIds, 0);
combined.set(right.parcelIds, vertexCountPerHemisphere);
const mappingSha256 = sha256(combined);
const metadata = {
  schemaVersion: "destrieux-fsaverage5/1",
  atlas: "Destrieux et al. 2010 cortical parcellation",
  space: "fsaverage5",
  vertexCount: combined.length,
  verticesPerHemisphere: vertexCountPerHemisphere,
  hemispheres: [
    { id: "left", offset: 0, count: vertexCountPerHemisphere },
    { id: "right", offset: vertexCountPerHemisphere, count: vertexCountPerHemisphere },
  ],
  labelCount: left.labels.length,
  labels: left.labels,
  dtype: "uint8",
  layout: "left-then-right",
  mappingSha256,
  sources: {
    left: { url: sources.left.url, sha256: sha256(leftBytes) },
    right: { url: sources.right.url, sha256: sha256(rightBytes) },
    provider: "Nilearn fetch_atlas_surf_destrieux / NITRC FreeSurfer fsaverage5",
  },
};

await mkdir(outputRoot, { recursive: true });
await writeFile(resolve(outputRoot, "destrieux-fsaverage5.u8"), combined);
await writeFile(resolve(outputRoot, "destrieux-fsaverage5.json"), `${JSON.stringify(metadata, null, 2)}\n`);
console.log(JSON.stringify({ outputRoot, vertexCount: combined.length, labelCount: left.labels.length, mappingSha256 }, null, 2));
