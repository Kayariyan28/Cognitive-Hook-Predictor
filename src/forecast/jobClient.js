import { forecastBackendConfiguration } from "./client.js";

export const FORECAST_JOB_SCHEMA_VERSION = "creator-forecast-job/1";
export const FORECAST_RESULT_SCHEMA_VERSION = "creator-forecast-result/1";
export const FORECAST_EVIDENCE_SCHEMA_VERSION = "creator-forecast-evidence/1";

const ACTIVE_JOB_STATES = new Set(["queued", "probing", "running"]);
const ALL_JOB_STATES = new Set([...ACTIVE_JOB_STATES, "complete", "failed"]);
const CORE_OPTIONAL_PROVIDER_KEYS = Object.freeze([
  "vjepa21",
  "videoLlama21Av",
  "audioModel",
  "account",
  "trends",
  "competitors",
]);
const LOCAL_OPTIONAL_PROVIDER_KEYS = Object.freeze(["semanticModel", "measuredAudio"]);
const OPTIONAL_PROVIDER_KEYS = Object.freeze([...CORE_OPTIONAL_PROVIDER_KEYS, ...LOCAL_OPTIONAL_PROVIDER_KEYS]);
const BEHAVIORAL_HEAD_KEYS = Object.freeze([
  "hookStrength",
  "retention5Second",
  "completionProbability",
  "rewatchPotential",
  "sharePotential",
  "predictedEngagement",
  "viralityPotential",
]);
const RESULT_TO_PRESENTATION_KEYS = Object.freeze({
  hookStrength: "hookStrength",
  retention5Second: "fiveSecondRetention",
  completionProbability: "completionProbability",
  rewatchPotential: "rewatchPotential",
  sharePotential: "sharePotential",
  predictedEngagement: "predictedEngagement",
  viralityPotential: "viralityPotential",
});
const HEX_32 = /^[a-f0-9]{32}$/;
const SHA_256 = /^[a-f0-9]{64}$/;
const COMMIT = /^[a-f0-9]{40}$/;
const STAGE = /^[a-z][a-z0-9-]{0,63}$/;
const FEATURE_NAME = /^[a-z][a-z0-9]*(?:[._][a-z0-9]+)*$/;
const FEATURE_PREFIXES = Object.freeze({
  vjepa21: "vjepa2_1.",
  videoLlama21Av: "videollama2.",
  audioModel: "audio.",
  semanticModel: "nanollava.",
  measuredAudio: "measured_audio.",
});
const EVIDENCE_KINDS = Object.freeze({
  vjepa21: new Set(["learned-visual-representation"]),
  videoLlama21Av: new Set(["learned-audio-visual-semantics"]),
  audioModel: new Set(["learned-speech-music-representation", "learned-audioset-label-evidence"]),
  semanticModel: new Set(["learned-keyframe-semantics"]),
  measuredAudio: new Set(["measured-audio-descriptors"]),
});
const FORBIDDEN_FEATURE_FRAGMENTS = ["tribe", "bold", "brain", "cortex", "cortical", "parcel", "virality", "retention", "completion", "rewatch"];

export class ForecastJobClientError extends Error {
  constructor(message, {
    code = "forecast_request_failed",
    status = null,
    retrySafe = false,
    job = null,
    idempotencyKey = null,
  } = {}) {
    super(message);
    this.name = "ForecastJobClientError";
    this.code = code;
    this.status = status;
    this.retrySafe = retrySafe;
    this.job = job;
    this.idempotencyKey = idempotencyKey;
  }
}

function deepFreeze(value, seen = new WeakSet()) {
  if (!value || typeof value !== "object" || seen.has(value)) return value;
  seen.add(value);
  for (const child of Object.values(value)) deepFreeze(child, seen);
  return Object.freeze(value);
}

function record(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value;
}

function exactKeys(value, expected, label) {
  record(value, label);
  const actual = Object.keys(value).sort();
  const required = [...expected].sort();
  if (actual.length !== required.length || actual.some((key, index) => key !== required[index])) {
    throw new Error(`${label} returned an unexpected shape.`);
  }
}

function keysWithOptional(value, required, optional, label) {
  record(value, label);
  const actual = new Set(Object.keys(value));
  if (required.some((key) => !actual.has(key)) || [...actual].some((key) => !required.includes(key) && !optional.includes(key))) {
    throw new Error(`${label} returned an unexpected shape.`);
  }
}

function finite(value, label, { minimum = -Infinity, maximum = Infinity } = {}) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum || value > maximum) {
    throw new Error(`${label} must be a finite number from ${minimum} to ${maximum}.`);
  }
  return value;
}

function integer(value, label, { minimum = 0 } = {}) {
  if (!Number.isInteger(value) || value < minimum) throw new Error(`${label} must be an integer.`);
  return value;
}

function text(value, label, { maximum = 8192 } = {}) {
  if (typeof value !== "string" || !value.trim() || value !== value.trim() || value.length > maximum) {
    throw new Error(`${label} must be non-empty trimmed text.`);
  }
  return value;
}

function timestamp(value, label, { nullable = false } = {}) {
  if (nullable && value === null) return null;
  text(value, label, { maximum: 64 });
  if (!/(?:Z|[+-]\d\d:\d\d)$/.test(value) || !Number.isFinite(Date.parse(value))) {
    throw new Error(`${label} must be a timestamp with a UTC offset.`);
  }
  return value;
}

function validateInput(value, { durationRequired = false } = {}) {
  exactKeys(value, ["sha256", "sizeBytes", "contentType", "durationSeconds"], "forecast input");
  if (typeof value.sha256 !== "string" || !SHA_256.test(value.sha256)) throw new Error("Forecast input SHA-256 is invalid.");
  integer(value.sizeBytes, "forecast input sizeBytes", { minimum: 1 });
  if (value.contentType !== null && (typeof value.contentType !== "string" || value.contentType.length > 256)) {
    throw new Error("Forecast input contentType is invalid.");
  }
  if (value.durationSeconds === null) {
    if (durationRequired) throw new Error("Completed forecast input omitted its authoritative duration.");
  } else {
    finite(value.durationSeconds, "forecast input durationSeconds", { minimum: 10, maximum: 60 });
  }
  return value;
}

function validateWorkerResult(value, branch, inputSha256) {
  exactKeys(value, [
    "schemaVersion", "branch", "inputSha256", "startedAt", "completedAt", "evidenceKind",
    "features", "observations", "warnings", "provenance", "behavioralOutcome",
  ], `${branch} worker result`);
  const measuredAudio = branch === "measuredAudio";
  const expectedSchema = measuredAudio ? "creator-forecast-measured-audio/1" : "creator-forecast-worker-output/1";
  if (value.schemaVersion !== expectedSchema || value.branch !== branch || value.inputSha256 !== inputSha256 || value.behavioralOutcome !== false) {
    throw new Error(`${branch} worker result identity is invalid.`);
  }
  timestamp(value.startedAt, `${branch} worker startedAt`);
  timestamp(value.completedAt, `${branch} worker completedAt`);
  text(value.evidenceKind, `${branch} evidenceKind`, { maximum: 128 });
  if (!EVIDENCE_KINDS[branch]?.has(value.evidenceKind)) throw new Error(`${branch} evidenceKind is unsupported.`);
  const features = record(value.features, `${branch} features`);
  const invalidFeature = Object.entries(features).some(([name, feature]) => !FEATURE_NAME.test(name)
    || !name.startsWith(FEATURE_PREFIXES[branch])
    || FORBIDDEN_FEATURE_FRAGMENTS.some((fragment) => name.toLowerCase().includes(fragment))
    || typeof feature !== "number" || !Number.isFinite(feature));
  if (Object.keys(features).length > 4096 || invalidFeature) {
    throw new Error(`${branch} features are invalid.`);
  }
  if (!Array.isArray(value.observations) || value.observations.length > 256) throw new Error(`${branch} observations are invalid.`);
  for (const [index, observation] of value.observations.entries()) {
    keysWithOptional(observation, ["kind", "startTime", "endTime", "text"], ["labels"], `${branch} observation ${index + 1}`);
    text(observation.kind, `${branch} observation kind`, { maximum: 128 });
    text(observation.text, `${branch} observation text`, { maximum: 4096 });
    finite(observation.startTime, `${branch} observation startTime`, { minimum: 0 });
    finite(observation.endTime, `${branch} observation endTime`, { minimum: 0 });
    if (observation.endTime <= observation.startTime) throw new Error(`${branch} observation interval is invalid.`);
    if (observation.labels !== undefined && (!Array.isArray(observation.labels) || observation.labels.length > 64 || observation.labels.some((label) => typeof label !== "string" || !label.trim()))) {
      throw new Error(`${branch} observation labels are invalid.`);
    }
  }
  if (!Array.isArray(value.warnings) || value.warnings.length > 64 || value.warnings.some((warning) => typeof warning !== "string" || !warning.trim())) {
    throw new Error(`${branch} warnings are invalid.`);
  }
  const provenance = value.provenance;
  if (measuredAudio) {
    exactKeys(provenance, ["adapterId", "preprocessingId", "sampleRateHz", "usesLearnedModel"], `${branch} provenance`);
    text(provenance.adapterId, `${branch} provenance adapterId`, { maximum: 256 });
    text(provenance.preprocessingId, `${branch} provenance preprocessingId`, { maximum: 256 });
    integer(provenance.sampleRateHz, `${branch} provenance sampleRateHz`, { minimum: 1 });
    if (provenance.usesLearnedModel !== false) throw new Error(`${branch} provenance made an unsupported learned-model claim.`);
  } else {
    exactKeys(provenance, [
      "branch", "modelId", "modelRevision", "modelWeightsSha256", "adapterId",
      "codeRevision", "preprocessingId", "preprocessingSha256",
    ], `${branch} provenance`);
    if (provenance.branch !== branch || !COMMIT.test(provenance.modelRevision) || !COMMIT.test(provenance.codeRevision)
        || !SHA_256.test(provenance.modelWeightsSha256) || !SHA_256.test(provenance.preprocessingSha256)) {
      throw new Error(`${branch} provenance is invalid.`);
    }
    for (const key of ["modelId", "adapterId", "preprocessingId"]) text(provenance[key], `${branch} provenance ${key}`, { maximum: 256 });
  }
  return value;
}

function validateProvider(value, branch, inputSha256) {
  const provider = record(value, `${branch} provider`);
  if (provider.status === "unavailable") {
    keysWithOptional(provider, ["status", "forecastContribution", "reason"], ["configured", "provider"], `${branch} provider`);
    if (provider.forecastContribution !== false || (provider.configured !== undefined && typeof provider.configured !== "boolean")) {
      throw new Error(`${branch} unavailable provider crossed the forecast boundary.`);
    }
    if (provider.provider !== undefined) record(provider.provider, `${branch} provider metadata`);
    text(provider.reason, `${branch} unavailable reason`);
    return provider;
  }
  if (provider.status !== "available" || !["vjepa21", "videoLlama21Av", "audioModel", "semanticModel", "measuredAudio"].includes(branch)) {
    throw new Error(`${branch} provider reported an unsupported availability state.`);
  }
  keysWithOptional(provider, ["status", "configured", "forecastContribution", "behavioralOutcome", "evidenceKind", "result"], ["provider"], `${branch} provider`);
  if (provider.configured !== true || provider.forecastContribution !== false || provider.behavioralOutcome !== false) {
    throw new Error(`${branch} provider crossed the descriptive-evidence boundary.`);
  }
  if (provider.provider !== undefined) record(provider.provider, `${branch} provider metadata`);
  const workerResult = validateWorkerResult(provider.result, branch, inputSha256);
  if (provider.evidenceKind !== workerResult.evidenceKind) throw new Error(`${branch} evidence kind changed between envelopes.`);
  return provider;
}

function validateEvidence(value, input) {
  exactKeys(value, ["schemaVersion", "videoMetadata", "creatorContext", "optionalProviders", "tribe"], "forecast evidence");
  if (value.schemaVersion !== FORECAST_EVIDENCE_SCHEMA_VERSION) throw new Error("Forecast evidence schema is unsupported.");
  exactKeys(value.videoMetadata, ["status", "source", "sha256", "sizeBytes", "durationSeconds", "contentType"], "video metadata evidence");
  if (value.videoMetadata.status !== "available" || value.videoMetadata.source !== "server-ffprobe"
      || value.videoMetadata.sha256 !== input.sha256 || value.videoMetadata.sizeBytes !== input.sizeBytes
      || value.videoMetadata.durationSeconds !== input.durationSeconds || value.videoMetadata.contentType !== input.contentType) {
    throw new Error("Video metadata evidence does not match the authoritative result input.");
  }
  exactKeys(value.creatorContext, ["status", "value", "forecastContribution"], "creator context evidence");
  if (!["provided", "not-provided"].includes(value.creatorContext.status) || value.creatorContext.forecastContribution !== false
      || (value.creatorContext.status === "provided" && (!value.creatorContext.value || typeof value.creatorContext.value !== "object" || Array.isArray(value.creatorContext.value)))
      || (value.creatorContext.status === "not-provided" && value.creatorContext.value !== null)) {
    throw new Error("Creator context evidence is invalid.");
  }
  const providerKeys = Object.keys(record(value.optionalProviders, "optional provider ledger")).sort();
  const coreKeys = [...CORE_OPTIONAL_PROVIDER_KEYS].sort();
  const extendedKeys = [...OPTIONAL_PROVIDER_KEYS].sort();
  const knownShape = [coreKeys, extendedKeys].some((shape) => providerKeys.length === shape.length && providerKeys.every((key, index) => key === shape[index]));
  if (!knownShape) throw new Error("Optional provider ledger returned an unexpected shape.");
  for (const branch of providerKeys) validateProvider(value.optionalProviders[branch], branch, input.sha256);
  exactKeys(value.tribe, ["status", "forecastContribution", "reason"], "TRIBE evidence boundary");
  if (value.tribe.status !== "not-invoked" || value.tribe.forecastContribution !== false) throw new Error("Forecast evidence attempted to invoke or use TRIBE.");
  text(value.tribe.reason, "TRIBE evidence boundary reason");
  return value;
}

function validateBehavioralHeads(value) {
  exactKeys(value, BEHAVIORAL_HEAD_KEYS, "behavioral-head ledger");
  for (const key of BEHAVIORAL_HEAD_KEYS) {
    const head = record(value[key], `${key} behavioral head`);
    if (head.status === "unavailable") {
      exactKeys(head, ["status", "value", "validation", "reason"], `${key} behavioral head`);
      if (head.value !== null || head.validation !== "not-calibrated") throw new Error(`${key} unavailable head is malformed.`);
      text(head.reason, `${key} withheld reason`);
    } else if (head.status === "available") {
      exactKeys(head, [
        "status", "value", "validation", "outputKind", "targetDefinition", "denominatorDefinition",
        "platform", "population", "horizon", "model",
      ], `${key} behavioral head`);
      finite(head.value, `${key} calibrated value`, { minimum: 0, maximum: 1 });
      if (head.validation !== "production-calibrated" || head.outputKind !== "calibrated-probability") throw new Error(`${key} available head is not production-calibrated.`);
      for (const field of ["targetDefinition", "denominatorDefinition", "platform", "population"]) text(head[field], `${key} ${field}`);
      exactKeys(head.horizon, ["kind", "value"], `${key} horizon`);
      text(head.horizon.kind, `${key} horizon kind`, { maximum: 64 });
      if (head.horizon.value !== null) finite(head.horizon.value, `${key} horizon value`, { minimum: Number.MIN_VALUE });
      const model = record(head.model, `${key} model provenance`);
      exactKeys(model, [
        "id", "revision", "artifactSha256", "license", "outputKind", "targetDefinition",
        "denominatorDefinition", "platform", "population", "horizon", "trainingDataCutoff", "calibration",
      ], `${key} model provenance`);
      if (!COMMIT.test(model.revision) || !SHA_256.test(model.artifactSha256) || model.outputKind !== "calibrated-probability"
          || model.targetDefinition !== head.targetDefinition || model.denominatorDefinition !== head.denominatorDefinition
          || model.platform !== head.platform || model.population !== head.population) {
        throw new Error(`${key} model provenance does not match its output definition.`);
      }
      text(model.id, `${key} model id`, { maximum: 256 });
      text(model.license, `${key} model license`, { maximum: 256 });
      timestamp(model.trainingDataCutoff, `${key} trainingDataCutoff`);
      if (JSON.stringify(model.horizon) !== JSON.stringify(head.horizon)) throw new Error(`${key} model horizon does not match its output horizon.`);
      exactKeys(model.calibration, [
        "revision", "artifactSha256", "method", "evaluationDataset", "evaluatedAt",
        "sampleCount", "brierScore", "expectedCalibrationError",
      ], `${key} calibration evidence`);
      if (!COMMIT.test(model.calibration.revision) || !SHA_256.test(model.calibration.artifactSha256)) throw new Error(`${key} calibration evidence is not immutable.`);
      text(model.calibration.method, `${key} calibration method`, { maximum: 128 });
      text(model.calibration.evaluationDataset, `${key} calibration dataset`, { maximum: 512 });
      timestamp(model.calibration.evaluatedAt, `${key} calibration evaluatedAt`);
      integer(model.calibration.sampleCount, `${key} calibration sampleCount`, { minimum: 1 });
      finite(model.calibration.brierScore, `${key} calibration brierScore`, { minimum: 0, maximum: 1 });
      finite(model.calibration.expectedCalibrationError, `${key} calibration expectedCalibrationError`, { minimum: 0, maximum: 1 });
    } else {
      throw new Error(`${key} behavioral head has an unsupported status.`);
    }
  }
  return value;
}

export function isActiveForecastJob(job) {
  return ACTIVE_JOB_STATES.has(job?.state);
}

export function validateForecastJob(payload) {
  exactKeys(payload, [
    "schemaVersion", "jobId", "state", "stage", "createdAt", "updatedAt", "heartbeatAt",
    "startedAt", "completedAt", "elapsedSeconds", "input", "contextAccepted", "resultId", "error", "links",
  ], "forecast job");
  if (payload.schemaVersion !== FORECAST_JOB_SCHEMA_VERSION || typeof payload.jobId !== "string" || !HEX_32.test(payload.jobId)
      || !ALL_JOB_STATES.has(payload.state) || typeof payload.stage !== "string" || !STAGE.test(payload.stage)) {
    throw new Error("Forecast job identity or state is invalid.");
  }
  timestamp(payload.createdAt, "forecast job createdAt");
  timestamp(payload.updatedAt, "forecast job updatedAt");
  timestamp(payload.heartbeatAt, "forecast job heartbeatAt");
  timestamp(payload.startedAt, "forecast job startedAt", { nullable: true });
  timestamp(payload.completedAt, "forecast job completedAt", { nullable: true });
  finite(payload.elapsedSeconds, "forecast job elapsedSeconds", { minimum: 0 });
  validateInput(payload.input, { durationRequired: payload.state === "complete" });
  if (typeof payload.contextAccepted !== "boolean") throw new Error("Forecast job contextAccepted is invalid.");
  exactKeys(payload.links, ["self", "result"], "forecast job links");
  if (payload.links.self !== `/api/forecast/v1/jobs/${payload.jobId}`) throw new Error("Forecast job self link is invalid.");
  if (payload.state === "complete") {
    if (payload.resultId !== payload.jobId || payload.links.result !== `/api/forecast/v1/results/${payload.jobId}` || payload.error !== null || payload.completedAt === null) {
      throw new Error("Completed forecast job is internally inconsistent.");
    }
  } else if (payload.state === "failed") {
    exactKeys(payload.error, ["code", "message"], "forecast job error");
    text(payload.error.code, "forecast job error code", { maximum: 128 });
    text(payload.error.message, "forecast job error message");
    if (payload.resultId !== null || payload.links.result !== null || payload.completedAt === null) throw new Error("Failed forecast job is internally inconsistent.");
  } else if (payload.resultId !== null || payload.links.result !== null || payload.error !== null || payload.completedAt !== null) {
    throw new Error("Active forecast job exposed terminal fields.");
  }
  return deepFreeze(payload);
}

export function validateForecastResult(payload) {
  exactKeys(payload, ["schemaVersion", "resultId", "jobId", "createdAt", "input", "evidence", "behavioralHeads", "boundaries"], "forecast result");
  if (payload.schemaVersion !== FORECAST_RESULT_SCHEMA_VERSION || typeof payload.resultId !== "string" || !HEX_32.test(payload.resultId) || payload.jobId !== payload.resultId) {
    throw new Error("Forecast result identity is invalid.");
  }
  timestamp(payload.createdAt, "forecast result createdAt");
  validateInput(payload.input, { durationRequired: true });
  validateEvidence(payload.evidence, payload.input);
  validateBehavioralHeads(payload.behavioralHeads);
  if (!Array.isArray(payload.boundaries) || !payload.boundaries.length || payload.boundaries.some((boundary) => typeof boundary !== "string" || !boundary.trim())) {
    throw new Error("Forecast result boundaries are invalid.");
  }
  return deepFreeze(payload);
}

function configuration(baseUrl) {
  const configured = typeof baseUrl === "string" && baseUrl.trim()
    ? { configured: true, baseUrl: baseUrl.trim().replace(/\/+$/, "") }
    : forecastBackendConfiguration();
  if (!configured.configured) {
    throw new ForecastJobClientError("No backend base URL is configured for forecast jobs.", { code: "forecast_not_configured", retrySafe: false });
  }
  return configured;
}

function normalizedIdempotencyKey(value) {
  if (typeof value !== "string" || !HEX_32.test(value.trim().toLowerCase())) {
    throw new ForecastJobClientError("A valid forecast submission idempotency key is required.", {
      code: "invalid_idempotency_key",
      retrySafe: false,
    });
  }
  return value.trim().toLowerCase();
}

export function createForecastIdempotencyKey() {
  const cryptoApi = globalThis.crypto;
  if (!cryptoApi || typeof cryptoApi.getRandomValues !== "function") {
    throw new ForecastJobClientError("This browser cannot create a secure forecast submission identity.", {
      code: "idempotency_unavailable",
      retrySafe: false,
    });
  }
  const bytes = cryptoApi.getRandomValues(new Uint8Array(16));
  // RFC 4122 version/variant bits make the value recognizable as a UUID while
  // the API uses its compact lowercase representation as the durable job ID.
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  return [...bytes].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function jsonResponse(response, validator, { retrySafe = false } = {}) {
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new ForecastJobClientError(`Forecast service returned non-JSON data (${response.status}).`, { code: "invalid_response", status: response.status, retrySafe: false });
  }
  if (!response.ok) {
    const detail = payload?.detail;
    const message = typeof detail?.message === "string" ? detail.message
      : typeof detail === "string" ? detail
        : typeof payload?.message === "string" ? payload.message
          : `Forecast request failed (${response.status}).`;
    throw new ForecastJobClientError(message, {
      code: typeof detail?.code === "string" ? detail.code : "forecast_request_failed",
      status: response.status,
      retrySafe,
    });
  }
  try {
    return validator(payload);
  } catch (error) {
    throw new ForecastJobClientError(error instanceof Error ? error.message : "Forecast response failed contract validation.", { code: "invalid_contract", status: response.status, retrySafe: false });
  }
}

export function buildForecastCreatorContext(inputs, { intendedPostAt = null } = {}) {
  const source = inputs && typeof inputs === "object" ? inputs : {};
  const optional = (value) => typeof value === "string" && value.trim() ? value.trim() : null;
  return Object.freeze({
    schemaVersion: "creator-forecast-context/1",
    platform: optional(source.platform) || "unspecified",
    caption: optional(source.caption),
    topic: optional(source.topic),
    genre: optional(source.genre),
    locale: optional(source.locale),
    language: optional(source.language),
    intendedPostAt: optional(intendedPostAt),
    utcOffset: optional(source.utcOffset),
    accountBand: optional(source.accountBand) || "unspecified",
    accountHistorySummary: optional(source.accountHistorySummary),
  });
}

export async function submitForecastJob(file, creatorContext, { signal, baseUrl, idempotencyKey } = {}) {
  if (!(file instanceof File)) throw new ForecastJobClientError("Choose a video File before starting a forecast job.", { code: "video_required", retrySafe: true });
  const config = configuration(baseUrl);
  const submissionKey = normalizedIdempotencyKey(idempotencyKey ?? createForecastIdempotencyKey());
  const body = new FormData();
  body.append("video", file, file.name);
  body.append("context", JSON.stringify(creatorContext ?? {}));
  let response;
  try {
    response = await fetch(`${config.baseUrl}/api/forecast/v1/jobs`, {
      method: "POST",
      body,
      signal,
      cache: "no-store",
      headers: { "Idempotency-Key": submissionKey },
    });
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    throw new ForecastJobClientError("The forecast upload connection ended before admission was confirmed. Retrying with the same submission identity cannot create a duplicate job.", {
      code: "submission_outcome_unknown",
      retrySafe: true,
      idempotencyKey: submissionKey,
    });
  }
  try {
    const job = await jsonResponse(response, validateForecastJob, { retrySafe: true });
    if (job.jobId !== submissionKey) {
      throw new ForecastJobClientError("Forecast admission returned a job that does not match the submission identity.", {
        code: "idempotency_identity_mismatch",
        status: response.status,
        retrySafe: false,
        idempotencyKey: submissionKey,
      });
    }
    return job;
  } catch (error) {
    if (error instanceof ForecastJobClientError) {
      error.idempotencyKey = submissionKey;
      // Reusing this key is idempotent even when an intermediary returned an
      // ambiguous HTTP error after the backend admitted the upload.
      if (error.code !== "invalid_contract" && error.code !== "idempotency_identity_mismatch") {
        error.retrySafe = true;
      }
    }
    throw error;
  }
}

export async function getForecastJob(jobId, { signal, baseUrl } = {}) {
  if (typeof jobId !== "string" || !HEX_32.test(jobId)) throw new ForecastJobClientError("A valid forecast job ID is required.", { code: "invalid_job_id" });
  const config = configuration(baseUrl);
  let response;
  try {
    response = await fetch(`${config.baseUrl}/api/forecast/v1/jobs/${jobId}`, { signal, cache: "no-store" });
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    throw new ForecastJobClientError("The forecast job status connection was interrupted.", {
      code: "job_status_unavailable",
      retrySafe: true,
    });
  }
  return jsonResponse(response, validateForecastJob, { retrySafe: true });
}

export async function getForecastResult(resultId, { signal, baseUrl } = {}) {
  if (typeof resultId !== "string" || !HEX_32.test(resultId)) throw new ForecastJobClientError("A valid forecast result ID is required.", { code: "invalid_result_id" });
  const config = configuration(baseUrl);
  let response;
  try {
    response = await fetch(`${config.baseUrl}/api/forecast/v1/results/${resultId}`, { signal, cache: "no-store" });
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    throw new ForecastJobClientError("The completed forecast result connection was interrupted.", {
      code: "result_temporarily_unavailable",
      retrySafe: true,
    });
  }
  return jsonResponse(response, validateForecastResult, { retrySafe: true });
}

function wait(milliseconds, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("The operation was aborted.", "AbortError"));
      return;
    }
    const timer = setTimeout(resolve, milliseconds);
    signal?.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(new DOMException("The operation was aborted.", "AbortError"));
    }, { once: true });
  });
}

export async function monitorForecastJob(initialJob, { signal, baseUrl, onJob, pollIntervalMs = 900 } = {}) {
  let job = validateForecastJob(initialJob);
  onJob?.(job);
  while (isActiveForecastJob(job)) {
    await wait(Math.max(250, pollIntervalMs), signal);
    try {
      job = await getForecastJob(job.jobId, { signal, baseUrl });
    } catch (error) {
      if (error?.name === "AbortError") throw error;
      if (error instanceof ForecastJobClientError) error.job = job;
      throw error;
    }
    onJob?.(job);
  }
  if (job.state === "failed") return Object.freeze({ job, result: null });
  let result;
  try {
    result = await getForecastResult(job.resultId, { signal, baseUrl });
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    if (error instanceof ForecastJobClientError) error.job = job;
    throw error;
  }
  if (result.jobId !== job.jobId) throw new ForecastJobClientError("Forecast result does not belong to the completed job.", { code: "result_identity_mismatch", retrySafe: false, job });
  if (result.input.sha256 !== job.input.sha256
      || result.input.sizeBytes !== job.input.sizeBytes
      || result.input.contentType !== job.input.contentType
      || result.input.durationSeconds !== job.input.durationSeconds) {
    throw new ForecastJobClientError("Forecast result input does not match the completed job.", { code: "result_input_mismatch", retrySafe: false, job });
  }
  return Object.freeze({ job, result });
}

export function forecastFromJobResult(result) {
  const validated = validateForecastResult(result);
  const metrics = {};
  for (const [headKey, metricKey] of Object.entries(RESULT_TO_PRESENTATION_KEYS)) {
    const head = validated.behavioralHeads[headKey];
    metrics[metricKey] = head.status === "available"
      ? {
          status: "available",
          value: head.value * 100,
          scoreKind: "behavioral-probability",
          behavioralOutcome: true,
          validation: { status: "production-calibrated", reason: null },
          reason: null,
        }
      : {
          status: "unavailable",
          value: null,
          scoreKind: "behavioral-probability",
          behavioralOutcome: true,
          validation: { status: "not-calibrated", reason: head.reason },
          reason: head.reason,
        };
  }
  return deepFreeze({
    schemaVersion: "creator-performance-forecast-job-view/1",
    sourceResultId: validated.resultId,
    metrics,
    confidence: {
      status: "unavailable",
      value: null,
      validation: { status: "not-calibrated", reason: "This job did not return a separately validated confidence-category head." },
      reason: "This job did not return a separately validated confidence-category head.",
    },
  });
}

export { BEHAVIORAL_HEAD_KEYS, OPTIONAL_PROVIDER_KEYS };
