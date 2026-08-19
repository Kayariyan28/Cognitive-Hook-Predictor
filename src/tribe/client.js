import { TRIBE_MODEL_REVISION, TRIBE_MODEL_WEIGHTS_SHA256, TRIBE_VIDEO_MODEL_ID, TRIBE_VIDEO_MODEL_REVISION, TRIBE_VIDEO_MODEL_WEIGHTS_SHA256, validateTribeManifest } from "./contract.js";
import { decodePredictionBuffer } from "./frame.js";
import { sha256Hex } from "./meshLoader.js";

function configuredBaseUrl() {
  const value = String(import.meta.env.VITE_TRIBE_API_URL || "").trim();
  return value ? value.replace(/\/+$/, "") : "";
}

export function tribeBackendConfiguration() {
  const baseUrl = configuredBaseUrl();
  return {
    configured: Boolean(baseUrl),
    baseUrl,
    endpoint: baseUrl ? `${baseUrl}/api/tribe/v1/predict` : "",
    statusEndpoint: baseUrl ? `${baseUrl}/api/tribe/v1/status` : "",
  };
}

export async function getTribeWorkerStatus({ signal } = {}) {
  const configuration = tribeBackendConfiguration();
  if (!configuration.configured) throw new Error("No TRIBE v2 endpoint is configured for this build.");
  const response = await fetch(configuration.statusEndpoint, { signal, cache: "no-store" });
  if (!response.ok) throw new Error(`TRIBE v2 worker status failed (${response.status}).`);
  const payload = await response.json();
  if (!payload?.configured || payload?.model?.id !== "facebook/tribev2" || payload?.model?.revision !== TRIBE_MODEL_REVISION || payload?.model?.weightsSha256 !== TRIBE_MODEL_WEIGHTS_SHA256) {
    throw new Error("The TRIBE v2 worker is reachable but has no immutable official model revision configured.");
  }
  if (!["full", "vision-only"].includes(payload?.model?.inferenceMode)) {
    throw new Error("The TRIBE v2 worker reported an unsupported inference mode.");
  }
  const expectedModalities = payload.model.inferenceMode === "vision-only" ? ["video"] : ["video", "audio", "text"];
  if (!Array.isArray(payload.model.modalitiesUsed) || payload.model.modalitiesUsed.some((value, index) => value !== expectedModalities[index]) || payload.model.modalitiesUsed.length !== expectedModalities.length) {
    throw new Error("The TRIBE v2 worker reported inconsistent model modalities.");
  }
  if (payload.model.extractors?.video?.id !== TRIBE_VIDEO_MODEL_ID || payload.model.extractors?.video?.revision !== TRIBE_VIDEO_MODEL_REVISION || payload.model.extractors?.video?.weightsSha256 !== TRIBE_VIDEO_MODEL_WEIGHTS_SHA256) {
    throw new Error("The TRIBE v2 worker is not pinned to the audited V-JEPA2 snapshot.");
  }
  const { device, precision } = payload.model.extractors.video;
  if (!["cpu", "cuda", "mps"].includes(device) || !["fp32", "autocast-fp16"].includes(precision) || (precision === "autocast-fp16" && device !== "mps")) {
    throw new Error("The TRIBE v2 worker reported an unsupported video-extractor execution policy.");
  }
  return Object.freeze(payload);
}

async function readPredictionResponse(response, { signal, onProgress } = {}) {
  if (!response.ok) {
    let detail = "";
    try {
      const payload = await response.json();
      detail = payload?.detail?.message || payload?.detail || payload?.error || "";
    } catch {
      detail = await response.text();
    }
    throw new Error(typeof detail === "string" && detail ? detail : `TRIBE v2 request failed (${response.status}).`);
  }

  onProgress?.({ stage: "validating", message: "Validating pinned revisions, vertex mapping, and tensor integrity" });
  const manifest = validateTribeManifest(await response.json(), response.url);
  onProgress?.({ stage: "downloading", message: "Loading the verified 20,484-vertex cortical tensor" });
  const framesResponse = await fetch(manifest.frames.url, { signal, cache: "no-store" });
  if (!framesResponse.ok) throw new Error(`Could not retrieve TRIBE prediction frames (${framesResponse.status}).`);
  const buffer = await framesResponse.arrayBuffer();
  if ((await sha256Hex(buffer)) !== manifest.frames.sha256.toLowerCase()) {
    throw new Error("TRIBE prediction frames failed their SHA-256 integrity check.");
  }
  const values = decodePredictionBuffer(buffer, manifest);
  onProgress?.({ stage: "ready", message: "Contract-validated, hash-checked TRIBE v2 tensor ready" });
  return Object.freeze({ manifest, values });
}

export async function loadTribeResult(resultId, { signal, onProgress } = {}) {
  const configuration = tribeBackendConfiguration();
  if (!configuration.configured) throw new Error("No TRIBE v2 endpoint is configured for this build.");
  if (typeof resultId !== "string" || !/^[a-f0-9]{32}$/i.test(resultId)) throw new Error("A valid TRIBE result ID is required.");
  onProgress?.({ stage: "restoring", message: "Loading the stored TRIBE prediction tensor" });
  const response = await fetch(`${configuration.baseUrl}/api/tribe/v1/results/${resultId}/manifest.json`, { signal, cache: "no-store" });
  return readPredictionResponse(response, { signal, onProgress });
}

export async function predictWithTribe(file, { signal, onProgress } = {}) {
  const configuration = tribeBackendConfiguration();
  if (!configuration.configured) throw new Error("No TRIBE v2 GPU endpoint is configured for this build.");
  if (!(file instanceof File)) throw new Error("A video File is required for TRIBE v2 inference.");

  onProgress?.({ stage: "requesting", message: "Submitting the video and waiting for an authoritative worker stage" });
  const body = new FormData();
  body.append("video", file, file.name);
  const response = await fetch(configuration.endpoint, { method: "POST", body, signal });
  return readPredictionResponse(response, { signal, onProgress });
}
