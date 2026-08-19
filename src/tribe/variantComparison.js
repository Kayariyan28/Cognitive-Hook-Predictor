import { TRIBE_DESCRIPTOR_SCHEMA_VERSION } from "./descriptors.js";

const COMPARISON_TOLERANCE = 1e-6;

function fail(message) {
  throw new Error(`Incompatible TRIBE A/B comparison: ${message}`);
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function finite(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number)) fail(`${label} must be finite.`);
  return number;
}

function validateReport(report, label) {
  if (!isRecord(report) || report.schemaVersion !== TRIBE_DESCRIPTOR_SCHEMA_VERSION) {
    fail(`${label} is not a verified cortical descriptor report.`);
  }
  if (!isRecord(report.source) || !isRecord(report.source.model) || !isRecord(report.source.surface)) {
    fail(`${label} provenance is incomplete.`);
  }
  if (report.source.descriptorType !== "descriptive cortical response"
      || report.source.predictionType !== "average-subject cortical BOLD response"
      || report.source.isViralityScore !== false
      || report.source.behavioralPrediction !== false) {
    fail(`${label} is not a non-behavioral TRIBE cortical response report.`);
  }
  if (!isRecord(report.clip)) fail(`${label}.clip is missing.`);
  return report;
}

function requireMatch(primary, variant, path, message) {
  const read = (record) => path.split(".").reduce((value, key) => value?.[key], record);
  if (read(primary) !== read(variant)) fail(message);
}

function direction(primary, variant) {
  const delta = variant - primary;
  const tolerance = COMPARISON_TOLERANCE * Math.max(1, Math.abs(primary), Math.abs(variant));
  if (Math.abs(delta) <= tolerance) return { delta: 0, direction: "tie", directionLabel: "No numerical difference" };
  return delta > 0
    ? { delta, direction: "variant-higher", directionLabel: "Variant B is higher" }
    : { delta, direction: "primary-higher", directionLabel: "Original is higher" };
}

function row(id, label, description, primary, variant) {
  const original = finite(primary, `original ${id}`);
  const alternate = finite(variant, `variant ${id}`);
  return Object.freeze({ id, label, description, primary: original, variant: alternate, ...direction(original, alternate) });
}

function deepFreeze(value) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  for (const child of Object.values(value)) deepFreeze(child);
  return Object.freeze(value);
}

/**
 * Compares like-for-like descriptive TRIBE cortical outputs. A higher value is
 * only a numerical direction; it is never an engagement or performance win.
 */
export function compareTribeVariantReports(primaryInput, variantInput) {
  const primary = validateReport(primaryInput, "original");
  const variant = validateReport(variantInput, "variant");

  for (const [path, message] of [
    ["source.model.id", "model IDs differ."],
    ["source.model.revision", "model revisions differ."],
    ["source.model.weightsSha256", "model weight snapshots differ."],
    ["source.model.codeRevision", "model code revisions differ."],
    ["source.extractor.id", "video extractor IDs differ."],
    ["source.extractor.revision", "video extractor revisions differ."],
    ["source.extractor.weightsSha256", "video extractor snapshots differ."],
    ["source.extractor.device", "video extractor devices differ."],
    ["source.extractor.precision", "video extractor precision policies differ."],
    ["source.inferenceMode", "inference modes differ; compare video-only with video-only or full with full."],
    ["source.surface.space", "surface spaces differ."],
    ["source.surface.mappingId", "surface vertex mappings differ."],
    ["source.surface.vertexCount", "surface vertex counts differ."],
    ["source.atlas.mappingSha256", "cortical atlas mappings differ."],
    ["units.response", "response units differ."],
  ]) requireMatch(primary, variant, path, message);

  const primaryChange = primary.clip.largestChange?.value;
  const variantChange = variant.clip.largestChange?.value;
  if (!Number.isFinite(primaryChange) || !Number.isFinite(variantChange)) {
    fail("both cuts need at least two prediction frames for a pattern-change comparison.");
  }

  const rows = [
    row("response-magnitude", "Response magnitude", "Duration-weighted usable-cortex RMS", primary.clip.responseMagnitude, variant.clip.responseMagnitude),
    row("pattern-change", "Largest pattern change", "Largest adjacent-frame usable-cortex RMS difference", primaryChange, variantChange),
    row("pattern-continuity", "Pattern continuity", "Mean adjacent-frame cosine similarity", primary.clip.continuity, variant.clip.continuity),
    row("spatial-distribution", "Spatial distribution", "Duration-weighted cortical surface spread index", primary.clip.spatialDistribution, variant.clip.spatialDistribution),
  ];

  return deepFreeze({
    schemaVersion: "tribe-ab-descriptors/1",
    comparisonType: "descriptive cortical response difference",
    behavioralPrediction: false,
    primaryResultId: primary.source.resultId,
    variantResultId: variant.source.resultId,
    compatibility: {
      modelRevision: primary.source.model.revision,
      inferenceMode: primary.source.inferenceMode,
      surfaceSpace: primary.source.surface.space,
      surfaceMappingId: primary.source.surface.mappingId,
      atlasMappingSha256: primary.source.atlas.mappingSha256,
    },
    rows,
    boundary: "Higher or lower predicted BOLD descriptors do not identify a winning cut and do not predict attention, engagement, retention, watch time, or virality. Publish a randomized platform A/B test to measure those outcomes.",
  });
}
