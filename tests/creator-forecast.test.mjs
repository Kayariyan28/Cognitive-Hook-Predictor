import assert from "node:assert/strict";
import test from "node:test";
import {
  BEHAVIORAL_FORECAST_METRIC_IDS,
  CALIBRATION_ARTIFACT_SCHEMA_VERSION,
  FORECAST_FEATURE_CONTRACT_SCHEMA_VERSION,
  FORECAST_METRIC_DEFINITIONS,
  TREND_SNAPSHOT_SCHEMA_VERSION,
  empiricalPercentile,
  validateCalibrationArtifact,
  validateForecastFeatureContract,
  validateTrendSnapshot,
} from "../src/forecast/contract.js";
import { deriveCreatorForecast } from "../src/forecast/creatorForecast.js";

const NOW = "2026-08-19T06:00:00Z";
const MODEL = Object.freeze({
  id: "example/content-head",
  revision: "a".repeat(40),
  weightsSha256: "b".repeat(64),
});
const FEATURES = Object.freeze({
  schemaVersion: FORECAST_FEATURE_CONTRACT_SCHEMA_VERSION,
  id: "duration-only/v1",
  sha256: "c".repeat(64),
  names: Object.freeze(["metadata.durationSeconds"]),
});

const ANALYSIS = Object.freeze({
  scores: Object.freeze({
    viralPotential: 0,
    engagementScore: 62,
    hookScore: 71,
    holdRate: 54,
    visualSignal: 83,
    editDynamics: 49,
    audioSignal: 58,
    visualStability: 76,
    speechLikelihood: 44,
    coverage: 91,
  }),
  virality: Object.freeze({ score: 0 }),
  engagement: Object.freeze({
    score: 62,
    coverage: 91,
    components: Object.freeze({ audioDynamics: 58, pacing: 49, endingStrength: 47 }),
  }),
});

const CAPABILITY_BRANCH_KEYS = Object.freeze([
  "browserLocalMetadata",
  "browserLocalVisual",
  "browserLocalAudio",
  "vjepa21",
  "videoLlama21Av",
  "audioModel",
  "semanticModel",
  "measuredAudio",
  "asr",
  "ocr",
  "account",
  "trends",
  "competitors",
  "tribeCortical",
]);
const CAPABILITY_HEAD_KEYS = Object.freeze([
  "hookStrength",
  "retention5Second",
  "completionProbability",
  "rewatchPotential",
  "sharePotential",
  "predictedEngagement",
  "viralityPotential",
]);

function currentCapabilityReport() {
  return {
    schemaVersion: "creator-forecast-capabilities/1",
    service: "creator-forecast",
    state: "evidence-job-service",
    scoreGenerationAvailable: false,
    uploadsAccepted: true,
    jobExecution: {
      schemaVersion: "creator-forecast-job-runtime/1",
      state: "ready",
      submissionRoute: "/api/forecast/v1/jobs",
      uploadsAccepted: true,
      authoritativeMediaProbe: true,
      durableIdempotency: "client-uuid-job-identity",
      configuredBranchCount: 0,
      readyBranchCount: 0,
      executableCalibrationHeadCount: 0,
    },
    supportedInput: {
      kind: "short-video",
      durationSeconds: { minimum: 10, maximum: 60, inclusive: true },
      durationValidation: "A future upload route must use an authoritative server-side media probe.",
    },
    availableNowProfile: {
      id: "available-now-evidence/1",
      behavioralForecastAvailable: false,
      implementedDescriptiveProviders: [
        "browserLocalMetadata",
        "browserLocalVisual",
        "browserLocalAudio",
      ],
    },
    registry: { schemaVersion: "creator-forecast-registry/1" },
    branches: Object.fromEntries(CAPABILITY_BRANCH_KEYS.map((key) => [key, {
      state: key.startsWith("browserLocal") ? "implemented-client-runtime" : "not-configured",
      configured: key.startsWith("browserLocal"),
      serverExecutionAvailable: false,
      forecastContribution: false,
      role: key === "tribeCortical" ? "cortical-only" : "descriptive-or-optional",
      ...(key === "tribeCortical" ? { currentlyReady: false } : {}),
      reason: `${key} does not expose forecast execution.`,
    }])),
    calibrationHeads: Object.fromEntries(CAPABILITY_HEAD_KEYS.map((key) => [key, {
      scoringAvailable: false,
      state: "not-configured",
      configured: false,
      reason: `${key} has no executable scoring head.`,
    }])),
    boundaries: ["No outcome scores are generated."],
  };
}

function artifactFor(metricId, overrides = {}) {
  const metric = FORECAST_METRIC_DEFINITIONS[metricId];
  return {
    schemaVersion: CALIBRATION_ARTIFACT_SCHEMA_VERSION,
    artifactId: (overrides.artifactCharacter || "a").repeat(64),
    status: "production",
    metricId,
    target: {
      id: metric.targetId,
      definitionSha256: metric.definitionSha256,
      scoreKind: metric.scoreKind,
    },
    model: MODEL,
    featureContract: FEATURES,
    domain: {
      platform: "youtube-shorts",
      locale: "en-IN",
      durationSeconds: { minimum: 10, maximum: 60 },
    },
    validity: {
      notBefore: "2026-08-01T00:00:00Z",
      expiresAt: "2026-09-01T00:00:00Z",
    },
    evaluation: {
      lockedTestPassed: true,
      calibrationPassed: true,
      sliceCoveragePassed: true,
      evaluatedExamples: 4_000,
      expectedCalibrationError: 0.04,
      confidence: overrides.confidence || "medium",
    },
    ...overrides.fields,
  };
}

function baseInput(overrides = {}) {
  return {
    durationSeconds: 30,
    evaluatedAt: NOW,
    analysis: ANALYSIS,
    model: MODEL,
    featureContract: FEATURES,
    domain: { platform: "youtube-shorts", locale: "en-IN" },
    backendCapabilities: currentCapabilityReport(),
    ...overrides,
  };
}

function validTrendSnapshot(overrides = {}) {
  return {
    schemaVersion: TREND_SNAPSHOT_SCHEMA_VERSION,
    snapshotId: "d".repeat(64),
    platform: "youtube-shorts",
    locale: "en-IN",
    retrievedAt: "2026-08-19T05:00:00Z",
    expiresAt: "2026-08-19T08:00:00Z",
    sources: [
      {
        url: "https://trends.example.test/topic",
        retrievedAt: "2026-08-19T04:59:00Z",
        contentSha256: "9".repeat(64),
      },
    ],
    providers: {
      trends: {
        id: "example/trends",
        revision: "1".repeat(40),
        implementationSha256: "2".repeat(64),
      },
      competitors: {
        id: "example/competitors",
        revision: "3".repeat(40),
        implementationSha256: "4".repeat(64),
      },
    },
    metrics: {
      trendAlignment: {
        rawValue: 0,
        reference: {
          id: "same-day-topic-reference/v1",
          sampleCount: 20,
          values: Array.from({ length: 20 }, (_, index) => index + 1),
        },
      },
      contentSaturation: {
        rawValue: 25,
        reference: {
          id: "same-day-supply-reference/v1",
          sampleCount: 20,
          values: Array.from({ length: 20 }, (_, index) => index + 1),
        },
      },
    },
    ...overrides,
  };
}

test("default forecast withholds every behavioral outcome while retaining descriptive browser indices", () => {
  const report = deriveCreatorForecast(baseInput());

  assert.equal(report.schemaVersion, "creator-performance-forecast/1");
  assert.equal(report.executionProfile, "available-now");
  assert.deepEqual(Object.keys(report.metrics), [
    "hookStrength",
    "fiveSecondRetention",
    "completionProbability",
    "rewatchPotential",
    "sharePotential",
    "trendAlignment",
    "contentSaturation",
    "predictedEngagement",
    "viralityPotential",
  ]);
  for (const metricId of BEHAVIORAL_FORECAST_METRIC_IDS) {
    const metric = report.metrics[metricId];
    assert.equal(metric.status, "withheld");
    assert.equal(metric.value, null);
    assert.equal(metric.validation.status, "not-calibrated");
    assert.equal(metric.reason, "backend-scoring-unavailable");
  }
  assert.equal(report.confidence.status, "withheld");
  assert.equal(report.confidence.value, null);
  assert.equal(report.evidence.contentSignals.kind, "descriptive-content-feature-index");
  assert.equal(report.evidence.contentSignals.calibrated, false);
  assert.equal(report.evidence.contentSignals.values.localViralityOutlookIndex, 0);
  assert.equal(report.evidence.contentSignals.values.localEngagementIndex, 62);
  assert.deepEqual(
    Object.fromEntries([
      "openingSignal",
      "postOpeningContinuity",
      "pacing",
      "ending",
      "visualClarity",
      "visualStability",
      "audioSupport",
    ].map((key) => [key, report.evidence.contentSignals.values[key]])),
    {
      openingSignal: 71,
      postOpeningContinuity: 54,
      pacing: 49,
      ending: 47,
      visualClarity: 83,
      visualStability: 76,
      audioSupport: 58,
    },
  );
  assert.equal(report.metrics.viralityPotential.value, null);
  assert.equal(report.metrics.predictedEngagement.value, null);
});

test("missing audio stays null while a genuinely measured zero is preserved", () => {
  const missingAudioAnalysis = {
    ...ANALYSIS,
    scores: { ...ANALYSIS.scores, audioSignal: 0, speechLikelihood: 0 },
    engagement: {
      ...ANALYSIS.engagement,
      components: { ...ANALYSIS.engagement.components, audioDynamics: null },
    },
  };
  const missing = deriveCreatorForecast(baseInput({
    analysis: missingAudioAnalysis,
    execution: { browserAudio: { available: true } },
  }));
  assert.equal(missing.evidence.contentSignals.values.audioSupport, null);
  assert.equal(missing.evidence.contentSignals.values.audioSignalIndex, null);
  assert.equal(missing.evidence.contentSignals.values.speechLikelihoodIndex, null);
  assert.equal(missing.capabilities.branches.browserAudio.status, "unavailable");

  const measuredZero = deriveCreatorForecast(baseInput({
    analysis: {
      ...missingAudioAnalysis,
      engagement: {
        ...missingAudioAnalysis.engagement,
        components: { ...missingAudioAnalysis.engagement.components, audioDynamics: 0 },
      },
    },
  }));
  assert.equal(measuredZero.evidence.contentSignals.values.audioSupport, 0);
  assert.equal(measuredZero.evidence.contentSignals.values.audioSignalIndex, 0);
  assert.equal(measuredZero.evidence.contentSignals.values.speechLikelihoodIndex, 0);
  assert.equal(measuredZero.capabilities.branches.browserAudio.status, "available");
});

test("a caller-supplied artifact and real zero cannot forge an executable backend head", () => {
  const artifact = artifactFor("fiveSecondRetention");
  const report = deriveCreatorForecast(baseInput({
    calibrationArtifacts: { fiveSecondRetention: artifact },
    predictions: {
      fiveSecondRetention: {
        estimate: 0,
        interval: { lower: 0, upper: 0.08, coverage: 0.9 },
        calibrationArtifactId: artifact.artifactId,
      },
    },
  }));

  assert.equal(report.metrics.fiveSecondRetention.status, "withheld");
  assert.equal(report.metrics.fiveSecondRetention.value, null);
  assert.equal(report.metrics.fiveSecondRetention.reason, "backend-scoring-unavailable");
  assert.equal(report.metrics.fiveSecondRetention.validation.status, "not-calibrated");
  assert.equal(report.metrics.fiveSecondRetention.calibrationArtifactId, null);
  assert.equal(report.evidence.contentSignals.values.localViralityOutlookIndex, 0);
});

test("hook artifact metric ID and target identity remain separate and exact", () => {
  const artifact = artifactFor("hookStrength");
  const validated = validateCalibrationArtifact(artifact, {
    metricId: "hookStrength",
    model: MODEL,
    featureContract: FEATURES,
    durationSeconds: 30,
    domain: { platform: "youtube-shorts", locale: "en-IN" },
    evaluatedAt: NOW,
  });

  assert.equal(validated.metricId, "hookStrength");
  assert.equal(validated.target.id, "hook-peer-percentile-from-retention-5s/v1");
  assert.equal(validated.target.definitionSha256, FORECAST_METRIC_DEFINITIONS.hookStrength.definitionSha256);
  assert.throws(
    () => validateCalibrationArtifact(
      { ...artifact, target: { ...artifact.target, id: "watch-ge-5s-per-eligible-play-start/v1" } },
      {
        metricId: "hookStrength",
        model: MODEL,
        featureContract: FEATURES,
        durationSeconds: 30,
        domain: { platform: "youtube-shorts", locale: "en-IN" },
        evaluatedAt: NOW,
      },
    ),
    /target\.id must be hook-peer-percentile/i,
  );
});

test("production calibration fails closed on target, model, feature, domain, and freshness mismatches", () => {
  const valid = artifactFor("fiveSecondRetention");
  const variants = [
    { artifact: { ...valid, status: "experimental" }, detail: /status must be production/i },
    { artifact: { ...valid, target: { ...valid.target, definitionSha256: "f".repeat(64) } }, detail: /target definition hash/i },
    { artifact: { ...valid, model: { ...MODEL, weightsSha256: "e".repeat(64) } }, detail: /model identity/i },
    { artifact: { ...valid, featureContract: { ...FEATURES, sha256: "e".repeat(64) } }, detail: /feature contract/i },
    { artifact: { ...valid, domain: { ...valid.domain, locale: "en-US" } }, detail: /platform\/locale/i },
    { artifact: { ...valid, validity: { ...valid.validity, expiresAt: NOW } }, detail: /not valid at the evaluation time/i },
  ];

  for (const variant of variants) {
    assert.throws(
      () => validateCalibrationArtifact(variant.artifact, {
        metricId: "fiveSecondRetention",
        model: MODEL,
        featureContract: FEATURES,
        durationSeconds: 30,
        domain: { platform: "youtube-shorts", locale: "en-IN" },
        evaluatedAt: NOW,
      }),
      variant.detail,
    );
  }
});

test("the explicit feature allowlist rejects TRIBE, BOLD, cortex, cortical, and parcel inputs", () => {
  for (const name of ["tribe.prediction", "bold.mean", "cortex.activation", "cortical.response", "parcel.peak"]) {
    assert.throws(
      () => validateForecastFeatureContract({
        ...FEATURES,
        names: [name],
      }),
      /forbidden TRIBE\/BOLD\/cortical namespace/i,
    );
  }
  assert.throws(
    () => validateForecastFeatureContract({ ...FEATURES, names: ["unknown.magicScore"] }),
    /not in the explicit forecast allowlist/i,
  );
});

test("TRIBE presence, absence, and numerical perturbation leave metric serialization byte-identical", () => {
  const withoutTribe = deriveCreatorForecast(baseInput());
  const withTribe = deriveCreatorForecast(baseInput({
    tribePrediction: { values: [999, -999], parcelPeak: 1e12 },
  }));
  const perturbedTribe = deriveCreatorForecast(baseInput({
    tribePrediction: { values: [-1e30, 1e30], parcelPeak: -Infinity },
  }));

  const serialized = JSON.stringify(withoutTribe.metrics);
  assert.equal(JSON.stringify(withTribe.metrics), serialized);
  assert.equal(JSON.stringify(perturbedTribe.metrics), serialized);
  assert.equal(withoutTribe.evidence.tribe.forecastContribution, false);
  assert.equal(withTribe.evidence.tribe.forecastContribution, false);
  assert.equal(withTribe.capabilities.branches.tribe.forecastContribution, false);
});

test("mutating the capability-only report cannot forge scoring or context execution", () => {
  const forgedScore = currentCapabilityReport();
  forgedScore.scoreGenerationAvailable = true;
  forgedScore.calibrationHeads.retention5Second.scoringAvailable = true;
  const scoreReport = deriveCreatorForecast(baseInput({ backendCapabilities: forgedScore }));
  assert.equal(scoreReport.metrics.fiveSecondRetention.status, "withheld");
  assert.equal(scoreReport.metrics.fiveSecondRetention.reason, "backend-capability-report-invalid");
  assert.equal(scoreReport.metrics.fiveSecondRetention.value, null);

  const forgedProvider = currentCapabilityReport();
  forgedProvider.branches.trends.serverExecutionAvailable = true;
  forgedProvider.branches.competitors.serverExecutionAvailable = true;
  const contextReport = deriveCreatorForecast(baseInput({
    backendCapabilities: forgedProvider,
    trendSnapshot: validTrendSnapshot(),
  }));
  assert.equal(contextReport.metrics.trendAlignment.status, "withheld");
  assert.equal(contextReport.metrics.trendAlignment.reason, "backend-capability-report-invalid");
  assert.equal(contextReport.metrics.contentSaturation.value, null);
});

test("truthful executable-head readiness still cannot unlock a browser-supplied score", () => {
  const capabilities = currentCapabilityReport();
  capabilities.scoreGenerationAvailable = true;
  capabilities.availableNowProfile.behavioralForecastAvailable = true;
  capabilities.jobExecution.executableCalibrationHeadCount = 1;
  capabilities.calibrationHeads.retention5Second = {
    ...capabilities.calibrationHeads.retention5Second,
    state: "executable-production-calibrated",
    configured: true,
    scoringAvailable: true,
  };
  const artifact = artifactFor("fiveSecondRetention");
  const report = deriveCreatorForecast(baseInput({
    backendCapabilities: capabilities,
    calibrationArtifacts: { fiveSecondRetention: artifact },
    predictions: {
      fiveSecondRetention: {
        estimate: 0.99,
        calibrationArtifactId: artifact.artifactId,
      },
    },
  }));
  assert.equal(report.evidence.backendCapabilities.scoreGenerationAvailable, true);
  assert.equal(report.metrics.fiveSecondRetention.status, "withheld");
  assert.equal(report.metrics.fiveSecondRetention.value, null);
  assert.equal(report.metrics.fiveSecondRetention.reason, "completed-job-result-required");
});

test("duration support is inclusive at 10 and 60 seconds and rejects 9.99 and 60.01", () => {
  for (const durationSeconds of [10, 60]) {
    const report = deriveCreatorForecast(baseInput({ durationSeconds }));
    assert.equal(report.domain.supportedDuration, true);
    assert.equal(report.metrics.hookStrength.reason, "backend-scoring-unavailable");
  }
  for (const durationSeconds of [9.99, 60.01]) {
    const report = deriveCreatorForecast(baseInput({ durationSeconds }));
    assert.equal(report.domain.supportedDuration, false);
    for (const metric of Object.values(report.metrics)) {
      assert.equal(metric.status, "withheld");
      assert.equal(metric.value, null);
      assert.equal(metric.reason, "unsupported-duration");
    }
  }
});

test("a strict trend snapshot preserves zero, but the capability-only backend cannot publish its indices", () => {
  const validated = validateTrendSnapshot(validTrendSnapshot(), {
    platform: "youtube-shorts",
    locale: "en-IN",
    evaluatedAt: NOW,
  });
  assert.equal(
    empiricalPercentile(
      validated.metrics.trendAlignment.rawValue,
      validated.metrics.trendAlignment.reference.values,
    ),
    0,
  );
  assert.equal(
    empiricalPercentile(
      validated.metrics.contentSaturation.rawValue,
      validated.metrics.contentSaturation.reference.values,
    ),
    1,
  );
  assert.equal(validated.sources[0].contentSha256, "9".repeat(64));

  const report = deriveCreatorForecast(baseInput({ trendSnapshot: validTrendSnapshot() }));
  assert.equal(report.metrics.trendAlignment.status, "withheld");
  assert.equal(report.metrics.trendAlignment.value, null);
  assert.equal(report.metrics.trendAlignment.reason, "backend-context-provider-unavailable");
  assert.equal(report.metrics.contentSaturation.status, "withheld");
  assert.equal(report.metrics.contentSaturation.value, null);
  assert.equal(report.evidence.trendSnapshot.reason, "backend-context-provider-unavailable");
  assert.equal(report.capabilities.branches.trendContext.status, "unavailable");
});

test("invalid, future-dated, stale, wrong-domain, and weak-reference trend snapshots fail strict validation", () => {
  const cases = [
    validTrendSnapshot({ retrievedAt: "2026-08-19T07:00:00Z", expiresAt: "2026-08-19T08:00:00Z" }),
    validTrendSnapshot({ expiresAt: NOW }),
    validTrendSnapshot({ locale: "en-US" }),
    validTrendSnapshot({
      sources: [{
        url: "http://trends.example.test/topic",
        retrievedAt: "2026-08-19T04:59:00Z",
        contentSha256: "9".repeat(64),
      }],
    }),
    validTrendSnapshot({
      metrics: {
        ...validTrendSnapshot().metrics,
        trendAlignment: {
          rawValue: 0.5,
          reference: { id: "tiny", sampleCount: 2, values: [0, 1] },
        },
      },
    }),
    validTrendSnapshot({ retrievedAt: "2026-08-19T05:00:00" }),
  ];

  for (const trendSnapshot of cases) {
    assert.throws(() => validateTrendSnapshot(trendSnapshot, {
      platform: "youtube-shorts",
      locale: "en-IN",
      evaluatedAt: NOW,
    }));
  }
});

test("empirical context percentiles use a tie-aware midrank", () => {
  assert.equal(empiricalPercentile(1, [0, 1, 1, 2]), 0.5);
  assert.equal(empiricalPercentile(0, [1, 2, 3]), 0);
});

test("available-now branches use descriptive fallbacks without impersonating missing model kinds", () => {
  const report = deriveCreatorForecast(baseInput({
    creatorMetadata: { caption: "A factual creator-supplied caption" },
  }));

  assert.equal(report.capabilities.branches.browserVisual.status, "available");
  assert.equal(report.capabilities.branches.browserAudio.status, "available");
  assert.equal(report.capabilities.branches.creatorMetadata.status, "available");
  assert.equal(report.capabilities.branches.vjepa21.status, "unavailable");
  assert.equal(report.capabilities.branches.videoSemantics.status, "unavailable");
  assert.equal(report.capabilities.branches.dedicatedAudioModel.status, "unavailable");
  for (const substitution of report.capabilities.substitutions) {
    assert.equal(substitution.behavioralEquivalent, false);
  }
  assert.deepEqual(
    report.capabilities.substitutions.map((item) => item.using),
    ["browserVisual", "creatorMetadata", "browserAudio"],
  );
});

test("creator context preserves declared facts without parsing free-text numbers or local time as an instant", () => {
  const localTime = deriveCreatorForecast(baseInput({
    creatorContext: {
      platform: "YouTube-Shorts",
      locale: "en-IN",
      language: "English",
      intendedPostAt: "2026-08-20T12:30",
      accountBand: "Growing",
      accountHistory: {
        followerCount: "1200",
        priorPostCount: 15,
        medianCompletionRate: 0.62,
        asOf: "2026-08-18T22:00:00",
      },
    },
  }));

  assert.equal(localTime.creatorContext.platform, "youtube-shorts");
  assert.equal(localTime.creatorContext.intendedPostAt.declared, "2026-08-20T12:30");
  assert.equal(localTime.creatorContext.intendedPostAt.instant, null);
  assert.equal(localTime.creatorContext.intendedPostAt.usableForTemporalFeatures, false);
  assert.equal(localTime.creatorContext.intendedPostAt.reason, "timezone-required");
  assert.equal(Object.hasOwn(localTime.creatorContext.accountHistory.facts, "followerCount"), false);
  assert.equal(localTime.creatorContext.accountHistory.facts.priorPostCount, 15);
  assert.equal(localTime.creatorContext.accountHistory.status, "unavailable");

  const zoned = deriveCreatorForecast(baseInput({
    creatorContext: {
      intendedPostAt: "2026-08-20T12:30:00+05:30",
      accountHistory: { priorPostCount: 15, asOf: "2026-08-18T22:00:00+05:30" },
    },
  }));
  assert.equal(zoned.creatorContext.intendedPostAt.usableForTemporalFeatures, true);
  assert.equal(zoned.creatorContext.intendedPostAt.instant, "2026-08-20T07:00:00.000Z");
  assert.equal(zoned.creatorContext.accountHistory.status, "available");
  assert.equal(JSON.stringify(zoned.metrics), JSON.stringify(localTime.metrics));
});

test("capability-only backend gating takes precedence over caller-declared runtime features", () => {
  const captionFeatures = {
    ...FEATURES,
    id: "caption/v1",
    sha256: "e".repeat(64),
    names: ["metadata.caption"],
  };
  const artifact = {
    ...artifactFor("fiveSecondRetention"),
    featureContract: captionFeatures,
  };
  const report = deriveCreatorForecast(baseInput({
    featureContract: captionFeatures,
    calibrationArtifacts: { fiveSecondRetention: artifact },
    predictions: {
      fiveSecondRetention: { estimate: 0.5, calibrationArtifactId: artifact.artifactId },
    },
  }));

  assert.equal(report.metrics.fiveSecondRetention.status, "withheld");
  assert.equal(report.metrics.fiveSecondRetention.reason, "backend-scoring-unavailable");
  assert.equal(report.metrics.fiveSecondRetention.value, null);
});

test("caller-declared complete artifacts cannot create overall confidence in capability-only v1", () => {
  const calibrationArtifacts = {};
  const predictions = {};
  const digestCharacters = ["1", "2", "3", "4", "5", "6", "7"];
  BEHAVIORAL_FORECAST_METRIC_IDS.forEach((metricId, index) => {
    const confidence = index === 3 ? "low" : "high";
    const artifact = artifactFor(metricId, {
      artifactCharacter: digestCharacters[index],
      confidence,
    });
    calibrationArtifacts[metricId] = artifact;
    predictions[metricId] = {
      estimate: 0.5,
      interval: { lower: 0.4, upper: 0.6, coverage: 0.9 },
      calibrationArtifactId: artifact.artifactId,
    };
  });
  const report = deriveCreatorForecast(baseInput({ calibrationArtifacts, predictions }));

  assert.equal(report.confidence.status, "withheld");
  assert.equal(report.confidence.value, null);
  assert.equal(report.confidence.validation.status, "not-calibrated");
  assert.ok(BEHAVIORAL_FORECAST_METRIC_IDS.every(
    (metricId) => report.metrics[metricId].status === "withheld"
      && report.metrics[metricId].reason === "backend-scoring-unavailable",
  ));
});
