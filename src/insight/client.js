/**
 * HTTP client for `/api/insight/v1`.
 *
 * A fail-closed `unavailable` document is a normal response, not an error: it is
 * returned to the caller so the panel can name the reason. Only transport,
 * caller, and contract failures raise.
 */

import { tribeBackendConfiguration } from "../tribe/client.js";
import { validateInsightArtifact, validateInsightBundle } from "./contract.js";

export class InsightClientError extends Error {
  constructor(message, { code = "insight_request_failed", status = null } = {}) {
    super(message);
    this.name = "InsightClientError";
    this.code = code;
    this.status = status;
  }
}

export function insightBackendConfiguration(baseUrl) {
  const tribeConfiguration = tribeBackendConfiguration();
  return Object.freeze({
    configured: Boolean(baseUrl) || tribeConfiguration.configured,
    baseUrl: baseUrl ?? tribeConfiguration.baseUrl,
  });
}

function configuration(baseUrl) {
  const config = insightBackendConfiguration(baseUrl);
  if (!config.configured || !config.baseUrl) {
    throw new InsightClientError(
      "The backend base URL is not configured, so the insight lane is unavailable.",
      { code: "insight_not_configured" },
    );
  }
  return config;
}

async function readJson(response) {
  try {
    return await response.json();
  } catch {
    throw new InsightClientError("The insight service returned a response that is not JSON.", {
      code: "invalid_contract",
      status: response.status,
    });
  }
}

function callerError(payload, status) {
  const detail = payload?.detail;
  const message = typeof detail?.message === "string"
    ? detail.message
    : `Insight request failed (${status}).`;
  return new InsightClientError(message, {
    code: typeof detail?.code === "string" ? detail.code : "insight_request_failed",
    status,
  });
}

/** Fetch provider readiness. This never generates anything. */
export async function fetchInsightStatus({ signal, baseUrl } = {}) {
  const config = configuration(baseUrl);
  const response = await fetch(`${config.baseUrl}/api/insight/v1/status`, {
    signal,
    cache: "no-store",
    headers: { Accept: "application/json" },
  });
  const payload = await readJson(response);
  if (!response.ok) throw callerError(payload, response.status);
  if (payload?.service !== "creator-insight") {
    throw new InsightClientError("The insight status response failed contract validation.", {
      code: "invalid_contract",
      status: response.status,
    });
  }
  return payload;
}

/**
 * Ask for one insight artifact. Returns either a validated artifact
 * (`{status: "available", artifact}`) or the service's unavailable document
 * (`{status: "unavailable", document}`).
 */
export async function generateInsight(
  { forecastResultId, tribeResultId = null, tribeDescriptors = null, hookOnly = false },
  { signal, baseUrl } = {},
) {
  const config = configuration(baseUrl);
  if (typeof forecastResultId !== "string" || !forecastResultId) {
    throw new InsightClientError("An insight request needs a completed forecast result.", {
      code: "forecast_result_required",
    });
  }
  const body = { forecastResultId, hookOnly: Boolean(hookOnly) };
  if (tribeResultId) body.tribeResultId = tribeResultId;
  if (tribeDescriptors) body.tribeDescriptors = tribeDescriptors;

  const response = await fetch(`${config.baseUrl}/api/insight/v1/generate`, {
    method: "POST",
    signal,
    cache: "no-store",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await readJson(response);
  if (!response.ok) throw callerError(payload, response.status);
  if (payload?.unavailable === true) {
    return Object.freeze({ status: "unavailable", document: payload });
  }
  try {
    return Object.freeze({
      status: "available",
      artifact: validateInsightArtifact(payload),
      cached: response.headers?.get?.("X-Insight-Cache") === "hit",
    });
  } catch (error) {
    throw new InsightClientError(
      error instanceof Error ? error.message : "The insight artifact failed contract validation.",
      { code: "invalid_contract", status: response.status },
    );
  }
}

/**
 * Fetch the deterministic hook readout: a timeline and a checklist computed
 * entirely from measurements. This route never calls a model, so it works when
 * no provider is installed.
 */
export async function fetchHookReadout(
  { forecastResultId, tribeResultId = null, tribeDescriptors = null },
  { signal, baseUrl } = {},
) {
  const config = configuration(baseUrl);
  const body = { forecastResultId };
  if (tribeResultId) body.tribeResultId = tribeResultId;
  if (tribeDescriptors) body.tribeDescriptors = tribeDescriptors;
  const response = await fetch(`${config.baseUrl}/api/insight/v1/hook-readout`, {
    method: "POST",
    signal,
    cache: "no-store",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await readJson(response);
  if (!response.ok) throw callerError(payload, response.status);
  if (payload?.unavailable === true) {
    return Object.freeze({ status: "unavailable", document: payload });
  }
  return Object.freeze({ status: "available", readout: payload });
}

/** Fetch the exact evidence a published artifact cites, for the chips. */
export async function fetchInsightEvidence(insightId, { signal, baseUrl } = {}) {
  const config = configuration(baseUrl);
  const response = await fetch(
    `${config.baseUrl}/api/insight/v1/results/${encodeURIComponent(insightId)}/evidence`,
    { signal, cache: "no-store", headers: { Accept: "application/json" } },
  );
  const payload = await readJson(response);
  if (!response.ok) throw callerError(payload, response.status);
  try {
    return validateInsightBundle(payload);
  } catch (error) {
    throw new InsightClientError(
      error instanceof Error ? error.message : "The evidence bundle failed contract validation.",
      { code: "invalid_contract", status: response.status },
    );
  }
}

/** Track one Hook Doctor experiment so the loop it opens can be closed. */
export async function createExperiment({ insightId, experimentId }, { signal, baseUrl } = {}) {
  const config = configuration(baseUrl);
  const response = await fetch(`${config.baseUrl}/api/insight/v1/experiments`, {
    method: "POST",
    signal,
    cache: "no-store",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ insightId, experimentId }),
  });
  const payload = await readJson(response);
  if (!response.ok) throw callerError(payload, response.status);
  return payload;
}

/** Record that the creator has made the edit, before any variant exists. */
export async function markExperimentEdited(experimentId, { signal, baseUrl } = {}) {
  const config = configuration(baseUrl);
  const response = await fetch(
    `${config.baseUrl}/api/insight/v1/experiments/${encodeURIComponent(experimentId)}/edited`,
    { method: "POST", signal, cache: "no-store", headers: { Accept: "application/json" } },
  );
  const payload = await readJson(response);
  if (!response.ok) throw callerError(payload, response.status);
  return payload;
}

/** Attach a second analysed clip and measure the declared signal shifts. */
export async function attachExperimentVariant(experimentId, forecastResultId, { signal, baseUrl } = {}) {
  const config = configuration(baseUrl);
  const response = await fetch(
    `${config.baseUrl}/api/insight/v1/experiments/${encodeURIComponent(experimentId)}/variant`,
    {
      method: "POST",
      signal,
      cache: "no-store",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ forecastResultId }),
    },
  );
  const payload = await readJson(response);
  if (!response.ok) throw callerError(payload, response.status);
  return payload;
}

/** List tracked experiments, newest first. */
export async function listExperiments({ signal, baseUrl } = {}) {
  const config = configuration(baseUrl);
  const response = await fetch(`${config.baseUrl}/api/insight/v1/experiments`, {
    signal,
    cache: "no-store",
    headers: { Accept: "application/json" },
  });
  const payload = await readJson(response);
  if (!response.ok) throw callerError(payload, response.status);
  const experiments = Array.isArray(payload?.experiments) ? payload.experiments : [];
  return Object.freeze(experiments);
}
