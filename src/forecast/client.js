import { tribeBackendConfiguration } from "../tribe/client.js";

const FORECAST_CAPABILITY_SCHEMA = "creator-forecast-capabilities/1";
const REQUIRED_BRANCHES = Object.freeze([
  "browserLocalMetadata",
  "browserLocalVisual",
  "browserLocalAudio",
  "vjepa21",
  "videoLlama21Av",
  "audioModel",
  "semanticModel",
  "measuredAudio",
  "asr",
  "account",
  "trends",
  "competitors",
  "tribeCortical",
]);
const REQUIRED_HEADS = Object.freeze([
  "hookStrength",
  "retention5Second",
  "completionProbability",
  "rewatchPotential",
  "sharePotential",
  "predictedEngagement",
  "viralityPotential",
]);
const BROWSER_BRANCHES = Object.freeze([
  "browserLocalMetadata",
  "browserLocalVisual",
  "browserLocalAudio",
]);
const OPTIONAL_SERVER_BRANCHES = Object.freeze([
  "vjepa21",
  "videoLlama21Av",
  "audioModel",
  "semanticModel",
  "measuredAudio",
  "asr",
  "account",
  "trends",
  "competitors",
]);

function deepFreeze(value, seen = new WeakSet()) {
  if (!value || typeof value !== "object" || seen.has(value)) return value;
  seen.add(value);
  for (const child of Object.values(value)) deepFreeze(child, seen);
  return Object.isFrozen(value) ? value : Object.freeze(value);
}

function configurationError(message) {
  const error = new Error(message);
  error.code = "FORECAST_NOT_CONFIGURED";
  return error;
}

function exactKeys(record, expected, label) {
  if (!record || typeof record !== "object" || Array.isArray(record)) {
    throw new Error(`The forecast capability registry omitted ${label}.`);
  }
  const actual = Object.keys(record).sort();
  const required = [...expected].sort();
  if (actual.length !== required.length || actual.some((key, index) => key !== required[index])) {
    throw new Error(`The forecast capability registry returned an unexpected ${label} shape.`);
  }
}

export function forecastBackendConfiguration() {
  const tribeConfiguration = tribeBackendConfiguration();
  return Object.freeze({
    configured: tribeConfiguration.configured,
    baseUrl: tribeConfiguration.baseUrl,
    statusEndpoint: tribeConfiguration.baseUrl
      ? `${tribeConfiguration.baseUrl}/api/forecast/v1/status`
      : "",
  });
}

export function validateForecastCapabilityReport(payload) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("The forecast capability registry returned an invalid document.");
  }
  if (payload.schemaVersion !== FORECAST_CAPABILITY_SCHEMA
      || payload.service !== "creator-forecast"
      || payload.state !== "evidence-job-service") {
    throw new Error("The forecast capability registry returned an unsupported contract.");
  }
  if (typeof payload.scoreGenerationAvailable !== "boolean" || payload.uploadsAccepted !== true) {
    throw new Error("The forecast capability registry made an unsupported scoring or upload claim.");
  }
  const jobExecution = payload.jobExecution;
  exactKeys(jobExecution, [
    "schemaVersion", "state", "submissionRoute", "uploadsAccepted",
    "authoritativeMediaProbe", "durableIdempotency", "configuredBranchCount",
    "readyBranchCount", "executableCalibrationHeadCount",
  ], "job-execution contract");
  if (jobExecution.schemaVersion !== "creator-forecast-job-runtime/1"
      || jobExecution.state !== "ready"
      || jobExecution.submissionRoute !== "/api/forecast/v1/jobs"
      || jobExecution.uploadsAccepted !== true
      || jobExecution.authoritativeMediaProbe !== true
      || jobExecution.durableIdempotency !== "client-uuid-job-identity"
      || !Number.isInteger(jobExecution.configuredBranchCount)
      || jobExecution.configuredBranchCount < 0
      || !Number.isInteger(jobExecution.readyBranchCount)
      || jobExecution.readyBranchCount < 0
      || !Number.isInteger(jobExecution.executableCalibrationHeadCount)
      || jobExecution.executableCalibrationHeadCount < 0) {
    throw new Error("The forecast capability registry returned an invalid executable job route.");
  }
  if (payload.supportedInput?.kind !== "short-video"
      || payload.supportedInput?.durationSeconds?.minimum !== 10
      || payload.supportedInput?.durationSeconds?.maximum !== 60
      || payload.supportedInput?.durationSeconds?.inclusive !== true
      || typeof payload.supportedInput?.durationValidation !== "string") {
    throw new Error("The forecast capability registry returned an unsupported input-duration contract.");
  }
  if (payload.availableNowProfile?.id !== "available-now-evidence/1"
      || payload.availableNowProfile?.behavioralForecastAvailable !== payload.scoreGenerationAvailable
      || !Array.isArray(payload.availableNowProfile?.implementedDescriptiveProviders)) {
    throw new Error("The forecast capability registry returned an invalid available-now profile.");
  }
  if (payload.registry?.schemaVersion !== "creator-forecast-registry/1"
      || !Array.isArray(payload.boundaries)
      || !payload.boundaries.length
      || payload.boundaries.some((boundary) => typeof boundary !== "string" || !boundary.trim())) {
    throw new Error("The forecast capability registry omitted registry provenance or evidence boundaries.");
  }
  exactKeys(payload.branches, REQUIRED_BRANCHES, "branch ledger");
  for (const key of REQUIRED_BRANCHES) {
    const branch = payload.branches[key];
    if (!branch || typeof branch !== "object" || typeof branch.state !== "string"
        || typeof branch.forecastContribution !== "boolean") {
      throw new Error(`The forecast capability registry omitted a valid ${key} branch.`);
    }
  }
  for (const key of BROWSER_BRANCHES) {
    const branch = payload.branches[key];
    if (typeof branch.configured !== "boolean"
        || branch.serverExecutionAvailable !== false
        || branch.forecastContribution !== false
        || !["implemented-client-runtime", "source-unverified"].includes(branch.state)) {
      throw new Error(`The forecast capability registry returned an invalid ${key} browser branch.`);
    }
  }
  for (const key of OPTIONAL_SERVER_BRANCHES) {
    const branch = payload.branches[key];
    const stateValid = [
      "not-configured",
      "registered-artifact-runtime-unavailable",
      "configured-runtime-unavailable",
      "configured-readiness-deferred",
      "ready",
    ].includes(branch.state);
    const configuredConsistent = branch.configured === (branch.state !== "not-configured");
    const executionConsistent = branch.serverExecutionAvailable === (branch.state === "ready");
    const readinessConsistent = branch.currentlyReady === undefined
      || branch.currentlyReady === null
      || typeof branch.currentlyReady === "boolean";
    if (!stateValid || !configuredConsistent || !executionConsistent || !readinessConsistent
        || branch.forecastContribution !== false) {
      throw new Error(`The forecast capability registry made a malformed ${key} runtime claim.`);
    }
  }
  const tribe = payload.branches.tribeCortical;
  if (tribe.forecastContribution !== false
      || tribe.role !== "cortical-only"
      || typeof tribe.configured !== "boolean"
      || typeof tribe.currentlyReady !== "boolean") {
    throw new Error("The forecast capability registry did not keep TRIBE cortical evidence separate.");
  }
  exactKeys(payload.calibrationHeads, REQUIRED_HEADS, "calibration-head ledger");
  let executableHeadCount = 0;
  for (const key of REQUIRED_HEADS) {
    const head = payload.calibrationHeads[key];
    const stateValid = head?.state === "not-configured"
      || head?.state === "registered-calibrator-runtime-unavailable"
      || head?.state === "executable-production-calibrated";
    const configuredConsistent = head?.configured === (head?.state !== "not-configured");
    const scoringConsistent = head?.scoringAvailable === (head?.state === "executable-production-calibrated");
    if (!stateValid || !configuredConsistent || !scoringConsistent) {
      throw new Error(`The forecast capability registry made a malformed scoring claim for ${key}.`);
    }
    if (head.scoringAvailable) executableHeadCount += 1;
  }
  if (payload.scoreGenerationAvailable !== (executableHeadCount > 0)
      || jobExecution.executableCalibrationHeadCount !== executableHeadCount) {
    throw new Error("The forecast capability registry returned inconsistent executable-head totals.");
  }
  return deepFreeze(payload);
}

export async function getForecastCapabilityStatus({ signal } = {}) {
  const configuration = forecastBackendConfiguration();
  if (!configuration.configured) {
    throw configurationError("No backend base URL is configured for the forecast capability registry.");
  }
  const response = await fetch(configuration.statusEndpoint, { signal, cache: "no-store" });
  if (!response.ok) throw new Error(`Forecast capability registry failed (${response.status}).`);
  return validateForecastCapabilityReport(await response.json());
}
