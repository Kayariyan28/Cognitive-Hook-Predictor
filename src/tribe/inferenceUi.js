const RESULT_ID_PATTERN = /^[a-f0-9]{32}$/i;

export const TRIBE_STAGE_LABELS = Object.freeze({
  idle: "Ready for a video",
  restoring: "Loading saved result",
  requesting: "Submitting video",
  not_loaded: "Waiting for model load",
  loading: "Loading pinned model",
  predicting: "Running model inference",
  ready: "Waiting for result",
  validating: "Validating result contract",
  downloading: "Loading cortical tensor",
  complete: "Prediction ready",
  error: "Prediction unavailable",
  unconfigured: "Worker unavailable",
});

const WORKER_STAGE_MESSAGES = Object.freeze({
  not_loaded: "The worker is reachable; its pinned model has not started loading yet.",
  loading: "The worker is loading the pinned official model and extractor weights.",
  predicting: "The official TRIBE pipeline is predicting cortical response.",
  ready: "The worker is ready; this request is still waiting for its manifest and tensor.",
  unconfigured: "The worker reports that the official model is not configured.",
});

export function formatInferenceElapsed(seconds) {
  const totalSeconds = Math.max(0, Math.floor(Number.isFinite(Number(seconds)) ? Number(seconds) : 0));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const remainingSeconds = totalSeconds % 60;
  if (hours > 0) return `${hours}:${String(minutes).padStart(2, "0")}:${String(remainingSeconds).padStart(2, "0")}`;
  return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
}

export function labelForTribeStage(stage) {
  return TRIBE_STAGE_LABELS[stage] || "Worker request in progress";
}

export function describeWorkerStage(state) {
  const stage = Object.hasOwn(WORKER_STAGE_MESSAGES, state) ? state : "requesting";
  return Object.freeze({
    stage,
    message: WORKER_STAGE_MESSAGES[stage] || "The request is active; waiting for an authoritative worker stage.",
  });
}

export function urlWithTribeResult(href, resultId) {
  if (!RESULT_ID_PATTERN.test(String(resultId || ""))) throw new Error("A valid TRIBE result ID is required for URL persistence.");
  const url = new URL(href);
  url.searchParams.set("tribeResult", resultId);
  url.searchParams.delete("tribeFrame");
  return url.toString();
}

export function urlWithTribeFrame(href, frameIndex) {
  const url = new URL(href);
  if (!RESULT_ID_PATTERN.test(String(url.searchParams.get("tribeResult") || ""))) {
    throw new Error("A persisted TRIBE result is required before selecting a frame.");
  }
  const index = Number(frameIndex);
  if (!Number.isInteger(index) || index < 0) throw new Error("A zero-based TRIBE frame index is required.");
  url.searchParams.set("tribeFrame", String(index + 1));
  return url.toString();
}

export function urlWithoutTribeResult(href) {
  const url = new URL(href);
  url.searchParams.delete("tribeResult");
  url.searchParams.delete("tribeFrame");
  return url.toString();
}
