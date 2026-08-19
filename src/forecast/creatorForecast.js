import {
  BEHAVIORAL_FORECAST_METRIC_IDS,
  CONTEXT_FORECAST_METRIC_IDS,
  CREATOR_FORECAST_SCHEMA_VERSION,
  FORECAST_DURATION_SECONDS,
  FORECAST_METRIC_ALIASES,
  FORECAST_METRIC_DEFINITIONS,
  empiricalPercentile,
  validateCalibrationArtifact,
  validateTrendSnapshot,
} from "./contract.js";
import { validateForecastCapabilityReport } from "./client.js";

const CONFIDENCE_ORDER = Object.freeze({ low: 0, medium: 1, high: 2 });
const SHA256 = /^(?:sha256:)?([a-f0-9]{64})$/i;
const CAPABILITY_HEAD_KEYS = Object.freeze({
  hookStrength: "hookStrength",
  fiveSecondRetention: "retention5Second",
  completionProbability: "completionProbability",
  rewatchPotential: "rewatchPotential",
  sharePotential: "sharePotential",
  predictedEngagement: "predictedEngagement",
  viralityPotential: "viralityPotential",
});

function deepFreeze(value) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  for (const child of Object.values(value)) deepFreeze(child);
  return Object.freeze(value);
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function finiteOrNull(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function boundedIndexOrNull(value) {
  const number = finiteOrNull(value);
  return number !== null && number >= 0 && number <= 100 ? number : null;
}

function percent(value) {
  return Math.round(value * 1_000) / 10;
}

function normalizedDigest(value) {
  if (typeof value !== "string") return null;
  return value.match(SHA256)?.[1]?.toLowerCase() || null;
}

function normalizeMetricId(metricId) {
  return FORECAST_METRIC_ALIASES[metricId] || metricId;
}

function withheldMetric(metricId, reason, reasonDetail = null) {
  const definition = FORECAST_METRIC_DEFINITIONS[normalizeMetricId(metricId)];
  return deepFreeze({
    id: definition.id,
    label: definition.label,
    status: "withheld",
    value: null,
    interval: null,
    scoreKind: definition.scoreKind,
    behavioralOutcome: definition.behavioralOutcome,
    validation: {
      status: definition.behavioralOutcome ? "not-calibrated" : "not-validated",
      reason,
    },
    reason,
    reasonDetail,
    calibrationArtifactId: null,
    confidence: null,
  });
}

function normalizedPrediction(input, artifactId) {
  if (!isRecord(input)) throw new Error("A calibrated prediction object is required.");
  if (normalizedDigest(input.calibrationArtifactId) !== artifactId) {
    throw new Error("Prediction calibrationArtifactId does not match the validated artifact.");
  }
  const estimate = finiteOrNull(input.estimate);
  if (estimate === null || estimate < 0 || estimate > 1) {
    throw new Error("Prediction estimate must be between zero and one.");
  }
  if (input.interval === undefined || input.interval === null) {
    return { estimate, interval: null };
  }
  if (!isRecord(input.interval)) throw new Error("Prediction interval must be an object.");
  const lower = finiteOrNull(input.interval.lower);
  const upper = finiteOrNull(input.interval.upper);
  const coverage = finiteOrNull(input.interval.coverage);
  if (lower === null || upper === null || coverage === null
      || lower < 0 || upper > 1 || lower > estimate || estimate > upper
      || coverage <= 0 || coverage >= 1) {
    throw new Error("Prediction interval must contain the estimate and declare coverage between zero and one.");
  }
  return { estimate, interval: { lower, upper, coverage } };
}

function availableBehavioralMetric(definition, artifact, prediction) {
  return deepFreeze({
    id: definition.id,
    label: definition.label,
    status: "available",
    value: percent(prediction.estimate),
    interval: prediction.interval
      ? {
          lower: percent(prediction.interval.lower),
          upper: percent(prediction.interval.upper),
          coverage: prediction.interval.coverage,
        }
      : null,
    scoreKind: definition.scoreKind,
    behavioralOutcome: true,
    validation: {
      status: "production-calibrated",
      reason: null,
    },
    reason: null,
    reasonDetail: null,
    calibrationArtifactId: artifact.artifactId,
    confidence: artifact.evaluation.confidence,
  });
}

function deriveBehavioralMetric(
  metricId,
  input,
  commonContext,
  durationSupported,
  runtimeFeatureAvailable,
  backendCapability,
) {
  const definition = FORECAST_METRIC_DEFINITIONS[metricId];
  if (!durationSupported) {
    return { metric: withheldMetric(metricId, "unsupported-duration"), featureNames: [] };
  }
  if (!backendCapability.report) {
    return {
      metric: withheldMetric(
        metricId,
        backendCapability.reason,
        backendCapability.reasonDetail,
      ),
      featureNames: [],
    };
  }
  const backendHead = backendCapability.report.calibrationHeads[CAPABILITY_HEAD_KEYS[metricId]];
  if (backendCapability.report.scoreGenerationAvailable !== true
      || backendHead?.scoringAvailable !== true) {
    return {
      metric: withheldMetric(metricId, "backend-scoring-unavailable", backendHead?.reason || null),
      featureNames: [],
    };
  }
  // Capability is not a score payload. Even when the server truthfully reports
  // an executable calibrated head, browser-supplied artifacts/predictions must
  // never impersonate that head. Only forecastFromJobResult may surface it.
  return {
    metric: withheldMetric(
      metricId,
      "completed-job-result-required",
      "Run the evidence job and validate its result contract to receive this executable head.",
    ),
    featureNames: [],
  };
}

function extractContentSignals(analysis) {
  const scores = isRecord(analysis?.scores) ? analysis.scores : {};
  const measuredAudioSignal = boundedIndexOrNull(analysis?.engagement?.components?.audioDynamics);
  const values = {
    openingSignal: boundedIndexOrNull(scores.hookScore),
    postOpeningContinuity: boundedIndexOrNull(scores.holdRate),
    pacing: boundedIndexOrNull(analysis?.engagement?.components?.pacing ?? scores.editDynamics),
    ending: boundedIndexOrNull(analysis?.engagement?.components?.endingStrength),
    visualClarity: boundedIndexOrNull(scores.visualSignal),
    visualStability: boundedIndexOrNull(scores.visualStability),
    audioSupport: measuredAudioSignal,
    localViralityOutlookIndex: boundedIndexOrNull(analysis?.virality?.score ?? scores.viralPotential),
    localEngagementIndex: boundedIndexOrNull(analysis?.engagement?.score ?? scores.engagementScore),
    openingSignalIndex: boundedIndexOrNull(scores.hookScore),
    continuitySignalIndex: boundedIndexOrNull(scores.holdRate),
    visualSignalIndex: boundedIndexOrNull(scores.visualSignal),
    editDynamicsIndex: boundedIndexOrNull(scores.editDynamics),
    endingSignalIndex: boundedIndexOrNull(analysis?.engagement?.components?.endingStrength),
    // The aggregate analysis score uses zero when audio could not be decoded.
    // Keep missing evidence distinct from a legitimately measured zero here.
    audioSignalIndex: measuredAudioSignal,
    visualStabilityIndex: boundedIndexOrNull(scores.visualStability),
    speechLikelihoodIndex: measuredAudioSignal === null
      ? null
      : boundedIndexOrNull(scores.speechLikelihood),
  };
  const available = Object.values(values).some((value) => value !== null);
  return deepFreeze({
    status: available ? "available" : "unavailable",
    kind: "descriptive-content-feature-index",
    calibrated: false,
    behavioralOutcome: false,
    inputCoverage: boundedIndexOrNull(analysis?.engagement?.coverage ?? scores.coverage),
    values,
    boundary: "These browser-derived clip indices describe sampled visual and audio features. They are not retention, engagement, share, completion, rewatch, or virality probabilities and never backfill withheld forecast metrics.",
  });
}

function contextMetric(definition, snapshot) {
  const measure = snapshot.metrics[definition.id];
  const percentile = empiricalPercentile(measure.rawValue, measure.reference.values);
  return deepFreeze({
    id: definition.id,
    label: definition.label,
    status: "available",
    value: percent(percentile),
    interval: null,
    scoreKind: definition.scoreKind,
    behavioralOutcome: false,
    validation: {
      status: "validated-context-snapshot",
      reason: null,
    },
    reason: null,
    reasonDetail: null,
    calibrationArtifactId: null,
    confidence: null,
    evidence: {
      snapshotId: snapshot.snapshotId,
      retrievedAt: snapshot.retrievedAt,
      expiresAt: snapshot.expiresAt,
      sourceCount: snapshot.sources.length,
      referenceId: measure.reference.id,
      referenceSampleCount: measure.reference.sampleCount,
      polarity: definition.id === "contentSaturation" ? "higher-means-more-crowded" : "higher-means-more-aligned",
    },
  });
}

function deriveContextMetrics(input, domain, evaluatedAt, durationSupported, backendCapability) {
  if (!durationSupported) {
    return {
      snapshot: null,
      error: null,
      metrics: Object.fromEntries(CONTEXT_FORECAST_METRIC_IDS.map(
        (metricId) => [metricId, withheldMetric(metricId, "unsupported-duration")],
      )),
    };
  }
  if (!backendCapability.report) {
    return {
      snapshot: null,
      error: backendCapability.reasonDetail,
      metrics: Object.fromEntries(CONTEXT_FORECAST_METRIC_IDS.map(
        (metricId) => [metricId, withheldMetric(
          metricId,
          backendCapability.reason,
          backendCapability.reasonDetail,
        )],
      )),
    };
  }
  // The v1 backend report guarantees both providers are non-executable. A
  // caller-supplied snapshot cannot forge provider availability.
  if (backendCapability.report.branches.trends?.serverExecutionAvailable !== true
      || backendCapability.report.branches.competitors?.serverExecutionAvailable !== true) {
    return {
      snapshot: null,
      error: null,
      metrics: Object.fromEntries(CONTEXT_FORECAST_METRIC_IDS.map(
        (metricId) => [metricId, withheldMetric(metricId, "backend-context-provider-unavailable")],
      )),
    };
  }
  if (!input.trendSnapshot) {
    return {
      snapshot: null,
      error: null,
      metrics: Object.fromEntries(CONTEXT_FORECAST_METRIC_IDS.map(
        (metricId) => [metricId, withheldMetric(metricId, "trend-snapshot-missing")],
      )),
    };
  }
  let snapshot;
  try {
    snapshot = validateTrendSnapshot(input.trendSnapshot, {
      platform: domain.platform,
      locale: domain.locale,
      evaluatedAt,
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return {
      snapshot: null,
      error: detail,
      metrics: Object.fromEntries(CONTEXT_FORECAST_METRIC_IDS.map(
        (metricId) => [metricId, withheldMetric(metricId, "trend-snapshot-invalid", detail)],
      )),
    };
  }
  return {
    snapshot,
    error: null,
    metrics: Object.fromEntries(CONTEXT_FORECAST_METRIC_IDS.map(
      (metricId) => [metricId, contextMetric(FORECAST_METRIC_DEFINITIONS[metricId], snapshot)],
    )),
  };
}

function hasMetadata(metadata) {
  if (!isRecord(metadata)) return false;
  return [metadata.caption, metadata.topic, metadata.genre]
    .some((value) => typeof value === "string" && value.trim());
}

function optionalText(value) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function explicitTimezone(value) {
  return typeof value === "string" && /(?:Z|[+-]\d\d:\d\d)$/.test(value.trim());
}

function normalizedIntendedPostAt(value) {
  const declared = optionalText(value);
  if (!declared) {
    return deepFreeze({
      declared: null,
      instant: null,
      timezoneExplicit: false,
      usableForTemporalFeatures: false,
      reason: "not-provided",
    });
  }
  if (!explicitTimezone(declared)) {
    return deepFreeze({
      declared,
      instant: null,
      timezoneExplicit: false,
      usableForTemporalFeatures: false,
      reason: "timezone-required",
    });
  }
  const milliseconds = Date.parse(declared);
  if (!Number.isFinite(milliseconds)) {
    return deepFreeze({
      declared,
      instant: null,
      timezoneExplicit: true,
      usableForTemporalFeatures: false,
      reason: "invalid-timestamp",
    });
  }
  return deepFreeze({
    declared,
    instant: new Date(milliseconds).toISOString(),
    timezoneExplicit: true,
    usableForTemporalFeatures: true,
    reason: null,
  });
}

const ACCOUNT_HISTORY_FACTS = Object.freeze({
  followerCount: { integer: true, minimum: 0, maximum: Infinity },
  priorPostCount: { integer: true, minimum: 0, maximum: Infinity },
  lookbackDays: { integer: true, minimum: 1, maximum: Infinity },
  medianOrganicViews: { integer: false, minimum: 0, maximum: Infinity },
  medianFirstPassWatchFraction: { integer: false, minimum: 0, maximum: 1 },
  medianCompletionRate: { integer: false, minimum: 0, maximum: 1 },
  medianShareRate: { integer: false, minimum: 0, maximum: 1 },
});

function normalizeAccountHistory(input) {
  if (!isRecord(input)) {
    return deepFreeze({ status: "unavailable", asOf: null, facts: {}, reason: "not-provided" });
  }
  const facts = {};
  for (const [name, rule] of Object.entries(ACCOUNT_HISTORY_FACTS)) {
    const value = input[name];
    // Numeric strings are intentionally not parsed into model features.
    if (typeof value !== "number" || !Number.isFinite(value)
        || value < rule.minimum || value > rule.maximum
        || rule.integer && !Number.isInteger(value)) continue;
    facts[name] = value;
  }
  const asOfDeclared = optionalText(input.asOf);
  const asOfMilliseconds = explicitTimezone(asOfDeclared) ? Date.parse(asOfDeclared) : NaN;
  const asOf = Number.isFinite(asOfMilliseconds) ? new Date(asOfMilliseconds).toISOString() : null;
  const available = Object.keys(facts).length > 0 && asOf !== null;
  return deepFreeze({
    status: available ? "available" : "unavailable",
    asOf,
    facts,
    reason: available
      ? null
      : Object.keys(facts).length < 1
        ? "no-valid-structured-facts"
        : "explicit-as-of-timezone-required",
  });
}

function normalizeCreatorContext(input, domain) {
  const context = isRecord(input.creatorContext) ? input.creatorContext : {};
  return deepFreeze({
    platform: optionalText(context.platform)?.toLowerCase() || domain.platform,
    locale: optionalText(context.locale) || domain.locale,
    language: optionalText(context.language),
    intendedPostAt: normalizedIntendedPostAt(context.intendedPostAt),
    accountBand: optionalText(context.accountBand),
    accountHistory: normalizeAccountHistory(context.accountHistory),
    boundary: "Creator context is used only as supplied structured evidence. Free text is never parsed into numeric account features, and a local posting time without an explicit timezone is not used as a temporal cutoff.",
  });
}

function runtimeFeatureAvailability({ input, contentSignals, contextResult, creatorContext, durationSupported }) {
  const metadata = isRecord(input.creatorMetadata) ? input.creatorMetadata : {};
  const visualAvailable = [
    contentSignals.values.visualSignalIndex,
    contentSignals.values.editDynamicsIndex,
    contentSignals.values.visualStabilityIndex,
  ].some((value) => value !== null);
  const browserAudioAvailable = contentSignals.values.audioSupport !== null;
  return (featureName) => {
    if (featureName.startsWith("browser.visual.")) return visualAvailable;
    if (featureName.startsWith("browser.audio.")) return browserAudioAvailable;
    if (featureName === "metadata.durationSeconds") return durationSupported;
    if (featureName === "metadata.caption") return optionalText(metadata.caption) !== null;
    if (featureName === "metadata.topic") return optionalText(metadata.topic) !== null;
    if (featureName === "metadata.genre") return optionalText(metadata.genre) !== null;
    if (featureName.startsWith("vjepa2_1.")) return false;
    if (featureName.startsWith("videollama2.")) return false;
    if (featureName.startsWith("audio.")) return false;
    if (featureName === "account.history") return creatorContext.accountHistory.status === "available";
    if (featureName === "account.platform") return creatorContext.platform !== null;
    if (featureName === "account.postingTime") return creatorContext.intendedPostAt.usableForTemporalFeatures;
    if (featureName.startsWith("trends.")) return contextResult.snapshot !== null;
    return false;
  };
}

function validateCapabilityOnlyReport(input) {
  return validateForecastCapabilityReport(input);
}

function deriveBackendCapability(input) {
  if (!input) {
    return deepFreeze({
      report: null,
      reason: "backend-capability-report-missing",
      reasonDetail: null,
    });
  }
  try {
    return deepFreeze({
      report: validateCapabilityOnlyReport(input),
      reason: null,
      reasonDetail: null,
    });
  } catch (error) {
    return deepFreeze({
      report: null,
      reason: "backend-capability-report-invalid",
      reasonDetail: error instanceof Error ? error.message : String(error),
    });
  }
}

function usesPrefix(featureNames, prefixes) {
  return featureNames.some((feature) => prefixes.some((prefix) => feature.startsWith(prefix)));
}

function branch(id, status, evidenceKind, forecastContribution, reason, substitution = null) {
  return deepFreeze({ id, status, evidenceKind, forecastContribution, reason, substitution });
}

function buildCapabilities(input, contentSignals, contextResult, creatorContext, featureNames, backendCapability) {
  const browserVisualAvailable = [
    contentSignals.values.visualSignalIndex,
    contentSignals.values.editDynamicsIndex,
    contentSignals.values.visualStabilityIndex,
  ].some((value) => value !== null);
  const browserAudioAvailable = contentSignals.values.audioSupport !== null;
  const metadataAvailable = hasMetadata(input.creatorMetadata);
  const backendBranches = backendCapability.report?.branches || {};
  const serverAvailable = (key) => backendBranches[key]?.configured === true
    && backendBranches[key]?.serverExecutionAvailable === true;
  const vjepaAvailable = serverAvailable("vjepa21");
  const semanticsAvailable = serverAvailable("videoLlama21Av");
  const audioModelAvailable = serverAvailable("audioModel");
  const accountAvailable = serverAvailable("account")
    && creatorContext.accountHistory.status === "available";
  const trendAvailable = contextResult.snapshot !== null;
  const tribePresent = input.tribePrediction !== null && input.tribePrediction !== undefined;

  const substitutions = [];
  if (!vjepaAvailable && browserVisualAvailable) substitutions.push({
    unavailableBranch: "vjepa21",
    using: "browserVisual",
    scope: "descriptive-content-signals-only",
    behavioralEquivalent: false,
  });
  if (!semanticsAvailable && metadataAvailable) substitutions.push({
    unavailableBranch: "videoSemantics",
    using: "creatorMetadata",
    scope: "creator-supplied-context-only",
    behavioralEquivalent: false,
  });
  if (!audioModelAvailable && browserAudioAvailable) substitutions.push({
    unavailableBranch: "dedicatedAudioModel",
    using: "browserAudio",
    scope: "descriptive-content-signals-only",
    behavioralEquivalent: false,
  });

  const substitutionFor = (id) => substitutions.find((item) => item.unavailableBranch === id) || null;
  return deepFreeze({
    branches: {
      browserVisual: branch(
        "browserVisual",
        browserVisualAvailable ? "available" : "unavailable",
        "descriptive-browser-evidence",
        usesPrefix(featureNames, ["browser.visual."]),
        browserVisualAvailable ? "Browser frame measurements are usable now." : "No browser visual analysis is available.",
      ),
      browserAudio: branch(
        "browserAudio",
        browserAudioAvailable ? "available" : "unavailable",
        "descriptive-browser-evidence",
        usesPrefix(featureNames, ["browser.audio."]),
        browserAudioAvailable ? "Browser audio measurements are usable now." : "No decoded browser audio evidence is available.",
      ),
      creatorMetadata: branch(
        "creatorMetadata",
        "available",
        "creator-supplied-evidence",
        usesPrefix(featureNames, ["metadata."]),
        metadataAvailable ? "Creator-supplied caption, topic, or genre is present." : "This branch is available; no optional caption, topic, or genre was supplied.",
      ),
      vjepa21: branch(
        "vjepa21",
        vjepaAvailable ? "available" : "unavailable",
        "video-foundation-model-feature",
        vjepaAvailable && usesPrefix(featureNames, ["vjepa2_1."]),
        vjepaAvailable ? "A configured V-JEPA 2.1 branch was declared." : "V-JEPA 2.1 is not configured; browser measurements do not impersonate its embeddings.",
        substitutionFor("vjepa21"),
      ),
      videoSemantics: branch(
        "videoSemantics",
        semanticsAvailable ? "available" : "unavailable",
        "video-language-model-feature",
        semanticsAvailable && usesPrefix(featureNames, ["videollama2."]),
        semanticsAvailable ? "A configured semantic video branch was declared." : "Video semantics are not configured; creator metadata is not a model-generated semantic prediction.",
        substitutionFor("videoSemantics"),
      ),
      dedicatedAudioModel: branch(
        "dedicatedAudioModel",
        audioModelAvailable ? "available" : "unavailable",
        "audio-foundation-model-feature",
        audioModelAvailable && usesPrefix(featureNames, ["audio."]),
        audioModelAvailable ? "A configured dedicated audio branch was declared." : "A dedicated audio model is not configured; browser audio remains descriptive only.",
        substitutionFor("dedicatedAudioModel"),
      ),
      trendContext: branch(
        "trendContext",
        trendAvailable ? "available" : "unavailable",
        "time-sensitive-context-index",
        trendAvailable || usesPrefix(featureNames, ["trends."]),
        trendAvailable ? "A fresh, source-backed trend snapshot passed the strict contract." : "No valid fresh trend snapshot is configured; no substitute is used.",
      ),
      accountContext: branch(
        "accountContext",
        accountAvailable ? "available" : "unavailable",
        "creator-account-context",
        accountAvailable && usesPrefix(featureNames, ["account."]),
        accountAvailable ? "A configured pre-post account context branch was declared." : "Account history is not configured; content signals do not substitute for distribution history.",
      ),
      tribe: branch(
        "tribe",
        tribePresent ? "available" : "unavailable",
        "separate-cortical-bold-evidence",
        false,
        tribePresent
          ? "TRIBE evidence is present but scientifically separate and excluded from every performance metric."
          : "No TRIBE evidence is present; this never blocks the performance evidence layer.",
      ),
    },
    substitutions,
  });
}

function overallConfidence(metrics) {
  const required = BEHAVIORAL_FORECAST_METRIC_IDS.map((metricId) => metrics[metricId]);
  const unavailable = required.filter((metric) => metric.status !== "available").map((metric) => metric.id);
  if (unavailable.length) {
    return deepFreeze({
      status: "withheld",
      value: null,
      scoreKind: "calibration-confidence-category",
      validation: { status: "not-calibrated", reason: "required-behavioral-metrics-unavailable" },
      reason: "required-behavioral-metrics-unavailable",
      unavailableMetrics: unavailable,
      boundary: "Input coverage is reported separately and is never treated as model confidence.",
    });
  }
  const value = required
    .map((metric) => metric.confidence)
    .reduce((lowest, next) => (CONFIDENCE_ORDER[next] < CONFIDENCE_ORDER[lowest] ? next : lowest), "high");
  return deepFreeze({
    status: "available",
    value,
    scoreKind: "calibration-confidence-category",
    validation: { status: "production-calibrated", reason: null },
    reason: null,
    unavailableMetrics: [],
    boundary: "This is the lowest artifact confidence tier across all behavioral forecasts, not input coverage or a guarantee.",
  });
}

/**
 * Build a fail-closed, serializable creator forecast from already available
 * evidence. This function performs no network access and runs no model.
 */
export function deriveCreatorForecast(input = {}) {
  const evaluatedAt = typeof input.evaluatedAt === "string" && input.evaluatedAt
    ? input.evaluatedAt
    : new Date().toISOString();
  const durationSeconds = finiteOrNull(input.durationSeconds);
  const durationSupported = durationSeconds !== null
    && durationSeconds >= FORECAST_DURATION_SECONDS.minimum
    && durationSeconds <= FORECAST_DURATION_SECONDS.maximum;
  const declaredContext = isRecord(input.creatorContext) ? input.creatorContext : {};
  const domain = {
    durationSeconds,
    supportedDuration: durationSupported,
    platform: optionalText(input.domain?.platform)?.toLowerCase()
      || optionalText(declaredContext.platform)?.toLowerCase()
      || null,
    locale: optionalText(input.domain?.locale) || optionalText(declaredContext.locale) || null,
  };
  const commonContext = {
    model: input.model,
    featureContract: input.featureContract,
    durationSeconds,
    domain,
    evaluatedAt,
  };

  const backendCapability = deriveBackendCapability(input.backendCapabilities);
  const creatorContext = normalizeCreatorContext(input, domain);
  const contentSignals = extractContentSignals(input.analysis);
  const contextResult = deriveContextMetrics(
    input,
    domain,
    evaluatedAt,
    durationSupported,
    backendCapability,
  );
  const runtimeFeatureAvailable = runtimeFeatureAvailability({
    input,
    contentSignals,
    contextResult,
    creatorContext,
    durationSupported,
  });
  const metrics = {};
  const usedFeatureNames = [];
  for (const metricId of BEHAVIORAL_FORECAST_METRIC_IDS) {
    const result = deriveBehavioralMetric(
      metricId,
      input,
      commonContext,
      durationSupported,
      runtimeFeatureAvailable,
      backendCapability,
    );
    metrics[metricId] = result.metric;
    usedFeatureNames.push(...result.featureNames);
  }
  for (const metricId of CONTEXT_FORECAST_METRIC_IDS) metrics[metricId] = contextResult.metrics[metricId];

  // Stable public ordering follows the creator-facing report, not computation order.
  const orderedMetrics = {
    hookStrength: metrics.hookStrength,
    fiveSecondRetention: metrics.fiveSecondRetention,
    completionProbability: metrics.completionProbability,
    rewatchPotential: metrics.rewatchPotential,
    sharePotential: metrics.sharePotential,
    trendAlignment: metrics.trendAlignment,
    contentSaturation: metrics.contentSaturation,
    predictedEngagement: metrics.predictedEngagement,
    viralityPotential: metrics.viralityPotential,
  };
  const capabilities = buildCapabilities(
    input,
    contentSignals,
    contextResult,
    creatorContext,
    [...new Set(usedFeatureNames)],
    backendCapability,
  );
  const tribePresent = input.tribePrediction !== null && input.tribePrediction !== undefined;

  return deepFreeze({
    schemaVersion: CREATOR_FORECAST_SCHEMA_VERSION,
    executionProfile: "available-now",
    generatedAt: evaluatedAt,
    domain,
    creatorContext,
    capabilities,
    evidence: {
      contentSignals,
      backendCapabilities: backendCapability.report
        ? {
            status: "available",
            schemaVersion: backendCapability.report.schemaVersion,
            service: backendCapability.report.service,
            state: backendCapability.report.state,
            scoreGenerationAvailable: backendCapability.report.scoreGenerationAvailable,
            uploadsAccepted: backendCapability.report.uploadsAccepted,
          }
        : {
            status: "unavailable",
            schemaVersion: null,
            reason: backendCapability.reason,
            reasonDetail: backendCapability.reasonDetail,
          },
      trendSnapshot: contextResult.snapshot
        ? {
            status: "available",
            snapshotId: contextResult.snapshot.snapshotId,
            platform: contextResult.snapshot.platform,
            locale: contextResult.snapshot.locale,
            retrievedAt: contextResult.snapshot.retrievedAt,
            expiresAt: contextResult.snapshot.expiresAt,
            sources: contextResult.snapshot.sources,
          }
        : {
            status: "unavailable",
            snapshotId: null,
            reason: contextResult.metrics.trendAlignment.reason,
            reasonDetail: contextResult.error,
          },
      tribe: {
        present: tribePresent,
        forecastContribution: false,
        boundary: "TRIBE predicts average-subject cortical BOLD. It is not an attention, retention, engagement, share, completion, rewatch, or virality feature.",
      },
    },
    metrics: orderedMetrics,
    confidence: overallConfidence(orderedMetrics),
    boundary: "Capability readiness cannot unlock performance or context scores in the browser. Only a strict completed-job result may carry an executable production-calibrated value. Descriptive browser evidence and TRIBE never backfill a missing outcome model.",
  });
}
