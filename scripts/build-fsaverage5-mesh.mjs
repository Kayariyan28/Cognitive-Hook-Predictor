import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { gunzipSync, inflateSync } from "node:zlib";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const projectDirectory = path.resolve(scriptDirectory, "..");
const sourceDirectory = path.resolve(projectDirectory, "../../work/fsaverage5-gifti");
const outputDirectory = path.resolve(projectDirectory, "public/assets/brain");

const sourceFiles = {
  pialLeft: "pial_left.gii.gz",
  pialRight: "pial_right.gii.gz",
  inflatedLeft: "infl_left.gii.gz",
  inflatedRight: "infl_right.gii.gz",
};

function attributesFrom(fragment) {
  return Object.fromEntries(
    [...fragment.matchAll(/([A-Za-z][A-Za-z0-9]*)="([^"]*)"/g)].map((match) => [match[1], match[2]]),
  );
}

function decodeDataArray(fragment) {
  const attributes = attributesFrom(fragment);
  if (attributes.Encoding !== "GZipBase64Binary" || attributes.Endian !== "LittleEndian") {
    throw new Error(`Unsupported GIFTI encoding: ${attributes.Encoding}/${attributes.Endian}`);
  }

  const dataMatch = fragment.match(/<Data>([\s\S]*?)<\/Data>/);
  if (!dataMatch) throw new Error("GIFTI DataArray is missing its Data payload.");
  const inflated = inflateSync(Buffer.from(dataMatch[1].replace(/\s+/g, ""), "base64"));

  let values;
  if (attributes.DataType === "NIFTI_TYPE_FLOAT32") {
    values = new Float32Array(inflated.buffer, inflated.byteOffset, inflated.byteLength / Float32Array.BYTES_PER_ELEMENT).slice();
  } else if (attributes.DataType === "NIFTI_TYPE_INT32") {
    values = new Int32Array(inflated.buffer, inflated.byteOffset, inflated.byteLength / Int32Array.BYTES_PER_ELEMENT).slice();
  } else {
    throw new Error(`Unsupported GIFTI data type: ${attributes.DataType}`);
  }

  return {
    intent: attributes.Intent,
    dimensions: Array.from({ length: Number(attributes.Dimensionality) }, (_, index) => Number(attributes[`Dim${index}`])),
    values,
  };
}

async function readGifti(fileName) {
  const filePath = path.join(sourceDirectory, fileName);
  const compressed = await readFile(filePath);
  const xml = gunzipSync(compressed).toString("utf8");
  const arrays = [...xml.matchAll(/<DataArray\b([\s\S]*?)<\/DataArray>/g)].map((match) => decodeDataArray(match[1]));
  const points = arrays.find((array) => array.intent === "NIFTI_INTENT_POINTSET");
  const triangles = arrays.find((array) => array.intent === "NIFTI_INTENT_TRIANGLE");
  if (!points || !triangles) throw new Error(`${fileName} does not contain both point and triangle arrays.`);
  return { points, triangles, sha256: createHash("sha256").update(compressed).digest("hex") };
}

function blendSurfaces(pial, inflated) {
  if (pial.length !== inflated.length) throw new Error("Pial and inflated surfaces have different vertex counts.");
  return Float32Array.from(pial, (value, index) => (value + inflated[index]) * 0.5);
}

function shiftHemispheres(left, right) {
  let leftMaxX = -Infinity;
  let rightMinX = Infinity;
  for (let index = 0; index < left.length; index += 3) leftMaxX = Math.max(leftMaxX, left[index]);
  for (let index = 0; index < right.length; index += 3) rightMinX = Math.min(rightMinX, right[index]);
  for (let index = 0; index < left.length; index += 3) left[index] -= leftMaxX;
  for (let index = 0; index < right.length; index += 3) right[index] -= rightMinX;
}

function boundsOf(positions) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let index = 0; index < positions.length; index += 3) {
    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], positions[index + axis]);
      max[axis] = Math.max(max[axis], positions[index + axis]);
    }
  }
  return { min, max };
}

const [pialLeft, pialRight, inflatedLeft, inflatedRight] = await Promise.all([
  readGifti(sourceFiles.pialLeft),
  readGifti(sourceFiles.pialRight),
  readGifti(sourceFiles.inflatedLeft),
  readGifti(sourceFiles.inflatedRight),
]);

const leftPositions = blendSurfaces(pialLeft.points.values, inflatedLeft.points.values);
const rightPositions = blendSurfaces(pialRight.points.values, inflatedRight.points.values);
shiftHemispheres(leftPositions, rightPositions);

const leftVertexCount = leftPositions.length / 3;
const rightVertexCount = rightPositions.length / 3;
if (leftVertexCount !== 10242 || rightVertexCount !== 10242) {
  throw new Error(`Expected 10,242 vertices per hemisphere; got ${leftVertexCount} and ${rightVertexCount}.`);
}

const positions = new Float32Array(leftPositions.length + rightPositions.length);
positions.set(leftPositions, 0);
positions.set(rightPositions, leftPositions.length);

const leftIndices = pialLeft.triangles.values;
const rightIndices = pialRight.triangles.values;
const indices = new Uint32Array(leftIndices.length + rightIndices.length);
indices.set(leftIndices, 0);
for (let index = 0; index < rightIndices.length; index += 1) {
  indices[leftIndices.length + index] = rightIndices[index] + leftVertexCount;
}

const headerBytes = 32;
const positionsOffset = headerBytes;
const indicesOffset = positionsOffset + positions.byteLength;
const binary = Buffer.alloc(indicesOffset + indices.byteLength);
binary.write("SFB1", 0, 4, "ascii");
binary.writeUInt32LE(leftVertexCount + rightVertexCount, 4);
binary.writeUInt32LE(indices.length / 3, 8);
binary.writeUInt32LE(leftVertexCount, 12);
binary.writeUInt32LE(positionsOffset, 16);
binary.writeUInt32LE(indicesOffset, 20);
binary.writeUInt32LE(1, 24);
binary.writeUInt32LE(0, 28);
Buffer.from(positions.buffer).copy(binary, positionsOffset);
Buffer.from(indices.buffer).copy(binary, indicesOffset);
const binarySha256 = createHash("sha256").update(binary).digest("hex");

await writeFile(path.join(outputDirectory, "fsaverage5-half.bin"), binary);
await writeFile(
  path.join(outputDirectory, "fsaverage5-half.json"),
  `${JSON.stringify({
    schemaVersion: "signalframe-fsaverage5-mesh/1",
    format: "SFB1",
    mappingId: `fsaverage5-half:${binarySha256}`,
    binary: "fsaverage5-half.bin",
    binarySha256,
    byteOrder: "little-endian",
    byteLength: binary.byteLength,
    headerBytes,
    positionsOffset,
    indicesOffset,
    vertexCount: leftVertexCount + rightVertexCount,
    leftVertexCount,
    rightVertexCount,
    triangleCount: indices.length / 3,
    ordering: ["left", "right"],
    geometry: "0.5 * Nilearn fsaverage5 pial + 0.5 * inflated, matching TRIBE v2 BasePlotBrain(inflate='half')",
    bounds: boundsOf(positions),
    source: "https://github.com/nilearn/nilearn/tree/main/nilearn/datasets/data/fsaverage5",
    sourceSha256: Object.fromEntries(Object.entries(sourceFiles).map(([key, file]) => [file, { pialLeft, pialRight, inflatedLeft, inflatedRight }[key].sha256])),
  }, null, 2)}\n`,
);

console.log(`Wrote ${leftVertexCount + rightVertexCount} vertices and ${indices.length / 3} triangles.`);
