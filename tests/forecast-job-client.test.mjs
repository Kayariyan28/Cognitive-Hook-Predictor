import assert from "node:assert/strict";
import test from "node:test";
import {
  buildForecastCreatorContext,
  forecastFromJobResult,
  monitorForecastJob,
  submitForecastJob,
  validateForecastJob,
  validateForecastResult,
} from "../src/forecast/jobClient.js";

const ID = "a".repeat(32);
const SHA = "b".repeat(64);
const NOW = "2026-08-19T00:00:00Z";
const HEADS = [
  "hookStrength",
  "retention5Second",
  "completionProbability",
  "rewatchPotential",
  "sharePotential",
  "predictedEngagement",
  "viralityPotential",
];
const PROVIDERS = ["vjepa21", "videoLlama21Av", "audioModel", "account", "trends", "competitors"];

function input(durationSeconds = 24) {
  return { sha256: SHA, sizeBytes: 1234, contentType: "video/mp4", durationSeconds };
}

function job(state = "complete") {
  const terminal = state === "complete" || state === "failed";
  return {
    schemaVersion: "creator-forecast-job/1",
    jobId: ID,
    state,
    stage: state === "complete" ? "complete" : state === "failed" ? "failed" : state,
    createdAt: NOW,
    updatedAt: NOW,
    heartbeatAt: NOW,
    startedAt: state === "queued" ? null : NOW,
    completedAt: terminal ? NOW : null,
    elapsedSeconds: 0,
    input: input(state === "queued" ? null : 24),
    contextAccepted: true,
    resultId: state === "complete" ? ID : null,
    error: state === "failed" ? { code: "worker_failed", message: "Worker failed safely." } : null,
    links: {
      self: `/api/forecast/v1/jobs/${ID}`,
      result: state === "complete" ? `/api/forecast/v1/results/${ID}` : null,
    },
  };
}

function result() {
  return {
    schemaVersion: "creator-forecast-result/1",
    resultId: ID,
    jobId: ID,
    createdAt: NOW,
    input: input(),
    evidence: {
      schemaVersion: "creator-forecast-evidence/1",
      videoMetadata: { status: "available", source: "server-ffprobe", ...input() },
      creatorContext: {
        status: "provided",
        value: { schemaVersion: "creator-forecast-context/1", platform: "youtube-shorts" },
        forecastContribution: false,
      },
      optionalProviders: Object.fromEntries(PROVIDERS.map((key) => [key, {
        status: "unavailable",
        configured: false,
        forecastContribution: false,
        reason: `${key} is not configured.`,
      }])),
      tribe: {
        status: "not-invoked",
        forecastContribution: false,
        reason: "TRIBE remains separate.",
      },
    },
    behavioralHeads: Object.fromEntries(HEADS.map((key) => [key, {
      status: "unavailable",
      value: null,
      validation: "not-calibrated",
      reason: `${key} is not calibrated.`,
    }])),
    boundaries: ["No score is emitted without production calibration."],
  };
}

function availableHead(value, overrides = {}) {
  const horizon = { kind: "hours", value: 24 };
  const targetDefinition = "eligible viewer watches at least five seconds";
  const denominatorDefinition = "eligible play starts";
  return {
    status: "available",
    value,
    validation: "production-calibrated",
    outputKind: "calibrated-probability",
    targetDefinition,
    denominatorDefinition,
    platform: "youtube-shorts",
    population: "comparable organic short-form uploads",
    horizon,
    model: {
      id: "example/calibrated-head",
      revision: "c".repeat(40),
      artifactSha256: "d".repeat(64),
      license: "Proprietary-test-fixture",
      outputKind: "calibrated-probability",
      targetDefinition,
      denominatorDefinition,
      platform: "youtube-shorts",
      population: "comparable organic short-form uploads",
      horizon,
      trainingDataCutoff: "2026-08-01T00:00:00Z",
      calibration: {
        revision: "e".repeat(40),
        artifactSha256: "f".repeat(64),
        method: "platt",
        evaluationDataset: "locked chronological holdout",
        evaluatedAt: NOW,
        sampleCount: 1000,
        brierScore: 0.12,
        expectedCalibrationError: 0.03,
      },
    },
    ...overrides,
  };
}

test("job and result validators enforce terminal identities and deep-freeze evidence", () => {
  const queued = validateForecastJob(job("queued"));
  assert.equal(queued.state, "queued");
  assert.equal(Object.isFrozen(queued.input), true);
  const complete = validateForecastJob(job());
  assert.equal(complete.links.result, `/api/forecast/v1/results/${ID}`);

  const validated = validateForecastResult(result());
  assert.equal(validated.evidence.tribe.forecastContribution, false);
  assert.equal(Object.isFrozen(validated.evidence.optionalProviders.vjepa21), true);

  const wrongLink = structuredClone(job());
  wrongLink.links.result = "https://attacker.example/result";
  assert.throws(() => validateForecastJob(wrongLink), /internally inconsistent/i);
});

test("behavioral values render only after exact 0..1 production calibration", () => {
  const zero = result();
  zero.behavioralHeads.hookStrength = availableHead(0);
  const view = forecastFromJobResult(zero);
  assert.equal(view.metrics.hookStrength.status, "available");
  assert.equal(view.metrics.hookStrength.value, 0);
  assert.equal(view.metrics.fiveSecondRetention.status, "unavailable");

  const uncalibrated = result();
  uncalibrated.behavioralHeads.hookStrength = availableHead(0.8, { validation: "not-calibrated" });
  assert.throws(() => validateForecastResult(uncalibrated), /not production-calibrated/i);

  const outOfBounds = result();
  outOfBounds.behavioralHeads.hookStrength = availableHead(1.01);
  assert.throws(() => validateForecastResult(outOfBounds), /finite number/i);
});

test("TRIBE and descriptive providers cannot claim forecast contribution", () => {
  const tribeLeak = result();
  tribeLeak.evidence.tribe.forecastContribution = true;
  assert.throws(() => validateForecastResult(tribeLeak), /TRIBE/i);

  const providerLeak = result();
  providerLeak.evidence.optionalProviders.vjepa21.forecastContribution = true;
  assert.throws(() => validateForecastResult(providerLeak), /forecast boundary/i);
});

test("the known local semantic and measured-audio evidence lanes validate without impersonating learned heads", () => {
  const extended = result();
  extended.evidence.optionalProviders.semanticModel = {
    status: "unavailable",
    configured: false,
    forecastContribution: false,
    reason: "The pinned NanoLLaVA snapshot is absent.",
    provider: {
      role: "nanollava-keyframe-semantic-fallback",
      usesAudio: false,
      provenance: { modelId: "mlx-community/nanoLLaVA-1.5-8bit" },
    },
  };
  extended.evidence.optionalProviders.measuredAudio = {
    status: "available",
    configured: true,
    forecastContribution: false,
    behavioralOutcome: false,
    evidenceKind: "measured-audio-descriptors",
    provider: { role: "server-measured-audio-descriptors", usesLearnedModel: false },
    result: {
      schemaVersion: "creator-forecast-measured-audio/1",
      branch: "measuredAudio",
      inputSha256: SHA,
      startedAt: NOW,
      completedAt: NOW,
      evidenceKind: "measured-audio-descriptors",
      features: { "measured_audio.present": 1 },
      observations: [{ kind: "measured-energy", startTime: 1, endTime: 2, text: "A relative waveform-energy peak was measured." }],
      warnings: [],
      provenance: {
        adapterId: "server-measured-audio-descriptors",
        preprocessingId: "ffmpeg-f32le-mono-16khz-stft1024-hop512/1",
        sampleRateHz: 16000,
        usesLearnedModel: false,
      },
      behavioralOutcome: false,
    },
  };
  const validated = validateForecastResult(extended);
  assert.equal(validated.evidence.optionalProviders.measuredAudio.behavioralOutcome, false);
  assert.equal(validated.evidence.optionalProviders.semanticModel.status, "unavailable");
});

test("creator context uses an exact serializable allowlist", () => {
  const context = buildForecastCreatorContext({
    platform: " youtube-shorts ",
    topic: " travel ",
    caption: "",
    accountBand: "emerging",
    ignored: "not sent",
  }, { intendedPostAt: "2026-08-20T10:00:00+05:30" });
  assert.deepEqual(Object.keys(context), [
    "schemaVersion", "platform", "caption", "topic", "genre", "locale", "language",
    "intendedPostAt", "utcOffset", "accountBand", "accountHistorySummary",
  ]);
  assert.equal(context.topic, "travel");
  assert.equal(context.caption, null);
  assert.equal(Object.hasOwn(context, "ignored"), false);
});

test("submission sends multipart once and monitoring fetches the completed result", async () => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, options = {}) => {
    calls.push({ url: String(url), options });
    if (String(url).endsWith("/jobs")) {
      return new Response(JSON.stringify(job("queued")), { status: 202, headers: { "content-type": "application/json" } });
    }
    if (String(url).includes("/results/")) {
      return new Response(JSON.stringify(result()), { status: 200, headers: { "content-type": "application/json" } });
    }
    throw new Error("unexpected request");
  };
  try {
    const selected = new File(["video"], "clip.mp4", { type: "video/mp4" });
    const submitted = await submitForecastJob(selected, { platform: "youtube-shorts" }, {
      baseUrl: "http://127.0.0.1:8000",
      idempotencyKey: ID,
    });
    assert.equal(submitted.jobId, ID);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].options.method, "POST");
    assert.equal(calls[0].options.body instanceof FormData, true);
    assert.equal(calls[0].options.headers["Idempotency-Key"], ID);

    const outcome = await monitorForecastJob(job(), { baseUrl: "http://127.0.0.1:8000" });
    assert.equal(outcome.result.resultId, ID);
    assert.equal(calls.length, 2);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("an ambiguous submission remains retry-safe only under the same durable identity", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => { throw new TypeError("connection reset"); };
  try {
    const selected = new File(["video"], "clip.mp4", { type: "video/mp4" });
    let caught;
    try {
      await submitForecastJob(selected, {}, {
        baseUrl: "http://127.0.0.1:8000",
        idempotencyKey: ID,
      });
    } catch (error) {
      caught = error;
    }
    assert.equal(caught?.code, "submission_outcome_unknown");
    assert.equal(caught?.retrySafe, true);
    assert.equal(caught?.idempotencyKey, ID);
    assert.match(caught?.message, /same submission identity/i);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("a completed job survives a transient result fetch failure and can be resumed", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => { throw new TypeError("connection reset"); };
  try {
    let caught;
    try {
      await monitorForecastJob(job(), { baseUrl: "http://127.0.0.1:8000" });
    } catch (error) {
      caught = error;
    }
    assert.equal(caught?.retrySafe, true);
    assert.equal(caught?.code, "result_temporarily_unavailable");
    assert.equal(caught?.job?.state, "complete");
    assert.equal(caught?.job?.resultId, ID);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("result input identity is cross-checked against the completed job", async () => {
  const originalFetch = globalThis.fetch;
  const mismatched = result();
  mismatched.input.sha256 = "c".repeat(64);
  mismatched.evidence.videoMetadata.sha256 = mismatched.input.sha256;
  globalThis.fetch = async () => new Response(JSON.stringify(mismatched), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
  try {
    let caught;
    try {
      await monitorForecastJob(job(), { baseUrl: "http://127.0.0.1:8000" });
    } catch (error) {
      caught = error;
    }
    assert.equal(caught?.code, "result_input_mismatch");
    assert.equal(caught?.retrySafe, false);
    assert.equal(caught?.job?.jobId, ID);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
