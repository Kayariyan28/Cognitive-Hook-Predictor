export const CREATOR_FORECAST_SCHEMA_VERSION = "creator-performance-forecast/1";
export const CALIBRATION_ARTIFACT_SCHEMA_VERSION = "creator-performance-calibration/1";
export const TREND_SNAPSHOT_SCHEMA_VERSION = "creator-trend-snapshot/1";
export const FORECAST_FEATURE_CONTRACT_SCHEMA_VERSION = "creator-forecast-features/1";

export const FORECAST_DURATION_SECONDS = Object.freeze({ minimum: 10, maximum: 60 });
export const MINIMUM_TREND_REFERENCE_SIZE = 20;

const TARGETS = Object.freeze({
  hookStrength: Object.freeze({
    targetId: "hook-peer-percentile-from-retention-5s/v1",
    definitionSha256: "dd815a8dea89c26d94a2ac89ac96a8fbff85c07a017b8bf81e0e1062b0c4317e",
    scoreKind: "calibrated-peer-percentile",
  }),
  fiveSecondRetention: Object.freeze({
    targetId: "watch-ge-5s-per-eligible-play-start/v1",
    definitionSha256: "285ee695090d5c579c698052c8f05186983aa534a8bf14ff5182f95a64ebe62f",
    scoreKind: "calibrated-probability",
  }),
  completionProbability: Object.freeze({
    targetId: "first-pass-completion-per-eligible-play-start/v1",
    definitionSha256: "9a71faa80dbb63f35be1be4630392968562df8ff20a8d25aa3ac379ce31946f9",
    scoreKind: "calibrated-probability",
  }),
  rewatchPotential: Object.freeze({
    targetId: "explicit-replay-per-eligible-play-start/v1",
    definitionSha256: "dffc24b375d05221059fb86eb20f9692e8c89cfc6fcdf571141e0a3956ac5f1e",
    scoreKind: "calibrated-probability",
  }),
  sharePotential: Object.freeze({
    targetId: "unique-share-per-qualified-viewer/v1",
    definitionSha256: "bfb7a448be835f28de87ec248383886834e93ce0d9f99a8ecb6d6e108ae65f67",
    scoreKind: "calibrated-probability",
  }),
  predictedEngagement: Object.freeze({
    targetId: "expected-first-pass-watch-fraction/v1",
    definitionSha256: "82891bd9a424d49a49995f4c13fff53122d1a12b01d553ce3812b48a84ca8c38",
    scoreKind: "calibrated-expected-rate",
  }),
  viralityPotential: Object.freeze({
    targetId: "organic-d7-peer-threshold-probability/v1",
    definitionSha256: "df233a871d324462a4569b412410cabf14039d5f1c8e3a34ce2484cb84f8ada1",
    scoreKind: "calibrated-probability",
  }),
});

export const FORECAST_METRIC_DEFINITIONS = deepFreeze({
  hookStrength: {
    id: "hookStrength",
    label: "Hook strength",
    behavioralOutcome: true,
    ...TARGETS.hookStrength,
  },
  fiveSecondRetention: {
    id: "fiveSecondRetention",
    label: "5-sec retention",
    behavioralOutcome: true,
    ...TARGETS.fiveSecondRetention,
  },
  completionProbability: {
    id: "completionProbability",
    label: "Completion probability",
    behavioralOutcome: true,
    ...TARGETS.completionProbability,
  },
  rewatchPotential: {
    id: "rewatchPotential",
    label: "Rewatch potential",
    behavioralOutcome: true,
    ...TARGETS.rewatchPotential,
  },
  sharePotential: {
    id: "sharePotential",
    label: "Share potential",
    behavioralOutcome: true,
    ...TARGETS.sharePotential,
  },
  trendAlignment: {
    id: "trendAlignment",
    label: "Trend alignment",
    behavioralOutcome: false,
    scoreKind: "context-cohort-percentile",
  },
  contentSaturation: {
    id: "contentSaturation",
    label: "Content saturation",
    behavioralOutcome: false,
    scoreKind: "context-cohort-percentile",
  },
  predictedEngagement: {
    id: "predictedEngagement",
    label: "Expected first-pass watch",
    behavioralOutcome: true,
    ...TARGETS.predictedEngagement,
  },
  viralityPotential: {
    id: "viralityPotential",
    label: "Virality potential",
    behavioralOutcome: true,
    ...TARGETS.viralityPotential,
  },
});

export const BEHAVIORAL_FORECAST_METRIC_IDS = Object.freeze(Object.keys(TARGETS));
export const CONTEXT_FORECAST_METRIC_IDS = Object.freeze(["trendAlignment", "contentSaturation"]);
export const FORECAST_METRIC_ALIASES = Object.freeze({ retention5s: "fiveSecondRetention" });

export const ALLOWED_FORECAST_FEATURES = Object.freeze([
  "browser.visual.luminance",
  "browser.visual.contrast",
  "browser.visual.colorfulness",
  "browser.visual.edgeDensity",
  "browser.visual.motion",
  "browser.visual.cut",
  "browser.audio.rms",
  "browser.audio.dynamics",
  "browser.audio.silenceRatio",
  "browser.audio.onsetRate",
  "browser.audio.speechLikelihood",
  "metadata.durationSeconds",
  "metadata.caption",
  "metadata.topic",
  "metadata.genre",
  "vjepa2_1.temporalEmbedding",
  "vjepa2_1.motionEmbedding",
  "videollama2.semanticEmbedding",
  "videollama2.caption",
  "videollama2.topic",
  "audio.speechEmbedding",
  "audio.musicEmbedding",
  "audio.prosody",
  "account.history",
  "account.platform",
  "account.postingTime",
  "trends.alignment",
  "trends.saturation",
]);

const ALLOWED_FEATURE_SET = new Set(ALLOWED_FORECAST_FEATURES);
const FORBIDDEN_FEATURE_NAMESPACE = /(^|[._/-])(tribe|bold|cortex|cortical|parcel)([._/-]|$)/i;
const CONFIDENCE_LABELS = new Set(["low", "medium", "high"]);
const SHA256 = /^(?:sha256:)?[a-f0-9]{64}$/i;
const IMMUTABLE_REVISION = /^[a-f0-9]{40}$/i;

function deepFreeze(value) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  for (const child of Object.values(value)) deepFreeze(child);
  return Object.freeze(value);
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function fail(message) {
  throw new Error(`Invalid creator forecast contract: ${message}`);
}

function requiredString(value, label) {
  if (typeof value !== "string" || !value.trim()) fail(`${label} must be a non-empty string.`);
  return value.trim();
}

function digest(value, label) {
  const text = requiredString(value, label).toLowerCase();
  if (!SHA256.test(text)) fail(`${label} must be a SHA-256 digest.`);
  return text.replace(/^sha256:/, "");
}

function finite(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value)) fail(`${label} must be a finite number.`);
  return value;
}

function bounded(value, label, minimum = 0, maximum = 1) {
  const number = finite(value, label);
  if (number < minimum || number > maximum) fail(`${label} must be between ${minimum} and ${maximum}.`);
  return number;
}

function positiveInteger(value, label) {
  const number = finite(value, label);
  if (!Number.isInteger(number) || number < 1) fail(`${label} must be a positive integer.`);
  return number;
}

function instant(value, label) {
  const text = requiredString(value, label);
  if (!/(?:Z|[+-]\d\d:\d\d)$/.test(text)) fail(`${label} must include an explicit timezone.`);
  const milliseconds = Date.parse(text);
  if (!Number.isFinite(milliseconds)) fail(`${label} must be a valid ISO timestamp.`);
  return { text, milliseconds };
}

function normalizedIdentity(identity, label) {
  if (!isRecord(identity)) fail(`${label} is required.`);
  const revision = requiredString(identity.revision, `${label}.revision`).toLowerCase();
  if (!IMMUTABLE_REVISION.test(revision)) fail(`${label}.revision must be a full 40-character commit SHA.`);
  return Object.freeze({
    id: requiredString(identity.id, `${label}.id`),
    revision,
    weightsSha256: digest(identity.weightsSha256, `${label}.weightsSha256`),
  });
}

function sameIdentity(left, right) {
  return left.id === right.id
    && left.revision === right.revision
    && left.weightsSha256 === right.weightsSha256;
}

export function validateForecastFeatureNames(names) {
  if (!Array.isArray(names) || names.length < 1) fail("feature names must be a non-empty array.");
  const seen = new Set();
  const normalized = names.map((value, index) => {
    const name = requiredString(value, `features.names[${index}]`);
    if (FORBIDDEN_FEATURE_NAMESPACE.test(name)) {
      fail(`feature ${name} belongs to a forbidden TRIBE/BOLD/cortical namespace.`);
    }
    if (!ALLOWED_FEATURE_SET.has(name)) fail(`feature ${name} is not in the explicit forecast allowlist.`);
    if (seen.has(name)) fail(`feature ${name} is duplicated.`);
    seen.add(name);
    return name;
  });
  return Object.freeze(normalized);
}

export function validateForecastFeatureContract(input) {
  if (!isRecord(input)) fail("featureContract is required.");
  if (input.schemaVersion !== FORECAST_FEATURE_CONTRACT_SCHEMA_VERSION) {
    fail(`featureContract.schemaVersion must be ${FORECAST_FEATURE_CONTRACT_SCHEMA_VERSION}.`);
  }
  return Object.freeze({
    schemaVersion: FORECAST_FEATURE_CONTRACT_SCHEMA_VERSION,
    id: requiredString(input.id, "featureContract.id"),
    sha256: digest(input.sha256, "featureContract.sha256"),
    names: validateForecastFeatureNames(input.names),
  });
}

function validateDomain(input, durationSeconds) {
  if (!isRecord(input)) fail("domain is required.");
  const platform = requiredString(input.platform, "domain.platform").toLowerCase();
  const locale = requiredString(input.locale, "domain.locale");
  if (!isRecord(input.durationSeconds)) fail("domain.durationSeconds is required.");
  const minimum = finite(input.durationSeconds.minimum, "domain.durationSeconds.minimum");
  const maximum = finite(input.durationSeconds.maximum, "domain.durationSeconds.maximum");
  if (minimum < FORECAST_DURATION_SECONDS.minimum || maximum > FORECAST_DURATION_SECONDS.maximum || minimum > maximum) {
    fail(`artifact duration support must stay inside ${FORECAST_DURATION_SECONDS.minimum}-${FORECAST_DURATION_SECONDS.maximum} seconds.`);
  }
  if (durationSeconds < minimum || durationSeconds > maximum) fail("clip duration is outside the artifact domain.");
  return Object.freeze({ platform, locale, durationSeconds: Object.freeze({ minimum, maximum }) });
}

function validateEvaluation(input) {
  if (!isRecord(input)) fail("artifact.evaluation is required.");
  for (const gate of ["lockedTestPassed", "calibrationPassed", "sliceCoveragePassed"]) {
    if (input[gate] !== true) fail(`artifact.evaluation.${gate} must be true.`);
  }
  return Object.freeze({
    lockedTestPassed: true,
    calibrationPassed: true,
    sliceCoveragePassed: true,
    evaluatedExamples: positiveInteger(input.evaluatedExamples, "artifact.evaluation.evaluatedExamples"),
    expectedCalibrationError: bounded(input.expectedCalibrationError, "artifact.evaluation.expectedCalibrationError"),
    confidence: CONFIDENCE_LABELS.has(input.confidence)
      ? input.confidence
      : fail("artifact.evaluation.confidence must be low, medium, or high."),
  });
}

/**
 * Validate one immutable, production-released calibration artifact against the
 * exact inference context. This verifies the declared contract and provenance;
 * it does not independently reproduce the artifact's training study.
 */
export function validateCalibrationArtifact(input, context) {
  if (!isRecord(input)) fail("calibration artifact must be an object.");
  if (!isRecord(context)) fail("calibration validation context is required.");
  const metric = FORECAST_METRIC_DEFINITIONS[context.metricId];
  if (!metric?.behavioralOutcome) fail("metricId must identify a supported behavioral metric.");
  if (input.schemaVersion !== CALIBRATION_ARTIFACT_SCHEMA_VERSION) {
    fail(`artifact.schemaVersion must be ${CALIBRATION_ARTIFACT_SCHEMA_VERSION}.`);
  }
  if (input.status !== "production") fail("artifact.status must be production.");
  if (input.metricId !== metric.id) fail(`artifact.metricId must be ${metric.id}.`);
  const artifactId = digest(input.artifactId, "artifact.artifactId");

  if (!isRecord(input.target)) fail("artifact.target is required.");
  if (input.target.id !== metric.targetId) fail(`artifact.target.id must be ${metric.targetId}.`);
  if (digest(input.target.definitionSha256, "artifact.target.definitionSha256") !== metric.definitionSha256) {
    fail("artifact target definition hash does not match the metric contract.");
  }
  if (input.target.scoreKind !== metric.scoreKind) fail("artifact target scoreKind does not match the metric contract.");

  const requestModel = normalizedIdentity(context.model, "context.model");
  const artifactModel = normalizedIdentity(input.model, "artifact.model");
  if (!sameIdentity(requestModel, artifactModel)) fail("artifact model identity does not exactly match inference.");

  const requestFeatures = validateForecastFeatureContract(context.featureContract);
  const artifactFeatures = validateForecastFeatureContract(input.featureContract);
  if (artifactFeatures.id !== requestFeatures.id
      || artifactFeatures.sha256 !== requestFeatures.sha256
      || artifactFeatures.names.length !== requestFeatures.names.length
      || artifactFeatures.names.some((name, index) => name !== requestFeatures.names[index])) {
    fail("artifact feature contract does not exactly match inference.");
  }

  const durationSeconds = finite(context.durationSeconds, "context.durationSeconds");
  if (durationSeconds < FORECAST_DURATION_SECONDS.minimum || durationSeconds > FORECAST_DURATION_SECONDS.maximum) {
    fail(`clip duration must be ${FORECAST_DURATION_SECONDS.minimum}-${FORECAST_DURATION_SECONDS.maximum} seconds.`);
  }
  const artifactDomain = validateDomain(input.domain, durationSeconds);
  const requestedPlatform = requiredString(context.domain?.platform, "context.domain.platform").toLowerCase();
  const requestedLocale = requiredString(context.domain?.locale, "context.domain.locale");
  if (artifactDomain.platform !== requestedPlatform || artifactDomain.locale !== requestedLocale) {
    fail("artifact platform/locale domain does not exactly match inference.");
  }

  if (!isRecord(input.validity)) fail("artifact.validity is required.");
  const evaluatedAt = instant(context.evaluatedAt, "context.evaluatedAt");
  const notBefore = instant(input.validity.notBefore, "artifact.validity.notBefore");
  const expiresAt = instant(input.validity.expiresAt, "artifact.validity.expiresAt");
  if (notBefore.milliseconds >= expiresAt.milliseconds) fail("artifact validity interval is empty.");
  if (evaluatedAt.milliseconds < notBefore.milliseconds || evaluatedAt.milliseconds >= expiresAt.milliseconds) {
    fail("artifact is not valid at the evaluation time.");
  }

  return deepFreeze({
    schemaVersion: CALIBRATION_ARTIFACT_SCHEMA_VERSION,
    artifactId,
    status: "production",
    metricId: metric.id,
    target: {
      id: metric.targetId,
      definitionSha256: metric.definitionSha256,
      scoreKind: metric.scoreKind,
    },
    model: artifactModel,
    featureContract: artifactFeatures,
    domain: artifactDomain,
    validity: { notBefore: notBefore.text, expiresAt: expiresAt.text },
    evaluation: validateEvaluation(input.evaluation),
  });
}

function validateSource(source, index, retrievedAtMilliseconds) {
  if (!isRecord(source)) fail(`trendSnapshot.sources[${index}] must be an object.`);
  let url;
  try {
    url = new URL(requiredString(source.url, `trendSnapshot.sources[${index}].url`));
  } catch {
    fail(`trendSnapshot.sources[${index}].url must be an absolute URL.`);
  }
  if (url.protocol !== "https:") fail("trend source URLs must use HTTPS.");
  const sourceTime = instant(source.retrievedAt, `trendSnapshot.sources[${index}].retrievedAt`);
  if (sourceTime.milliseconds > retrievedAtMilliseconds) fail("trend source cannot be retrieved after its snapshot.");
  return Object.freeze({
    url: url.href,
    retrievedAt: sourceTime.text,
    contentSha256: digest(source.contentSha256, `trendSnapshot.sources[${index}].contentSha256`),
  });
}

function validateContextProvider(input, label) {
  if (!isRecord(input)) fail(`trendSnapshot.providers.${label} is required.`);
  const revision = requiredString(input.revision, `trendSnapshot.providers.${label}.revision`).toLowerCase();
  if (!IMMUTABLE_REVISION.test(revision)) {
    fail(`trendSnapshot.providers.${label}.revision must be a full 40-character commit SHA.`);
  }
  return Object.freeze({
    id: requiredString(input.id, `trendSnapshot.providers.${label}.id`),
    revision,
    implementationSha256: digest(
      input.implementationSha256,
      `trendSnapshot.providers.${label}.implementationSha256`,
    ),
  });
}

function validateTrendMeasure(input, metricId) {
  if (!isRecord(input)) fail(`trendSnapshot.metrics.${metricId} is required.`);
  const rawValue = finite(input.rawValue, `trendSnapshot.metrics.${metricId}.rawValue`);
  if (rawValue < 0) fail(`trendSnapshot.metrics.${metricId}.rawValue cannot be negative.`);
  if (!isRecord(input.reference)) fail(`trendSnapshot.metrics.${metricId}.reference is required.`);
  const referenceId = requiredString(input.reference.id, `trendSnapshot.metrics.${metricId}.reference.id`);
  if (!Array.isArray(input.reference.values)
      || input.reference.values.length < MINIMUM_TREND_REFERENCE_SIZE) {
    fail(`trendSnapshot.metrics.${metricId}.reference.values needs at least ${MINIMUM_TREND_REFERENCE_SIZE} observations.`);
  }
  const values = input.reference.values.map((value, index) => {
    const number = finite(value, `trendSnapshot.metrics.${metricId}.reference.values[${index}]`);
    if (number < 0) fail("trend reference values cannot be negative.");
    return number;
  });
  if (input.reference.sampleCount !== values.length) fail("trend reference sampleCount must match its values.");
  return deepFreeze({
    rawValue,
    reference: { id: referenceId, sampleCount: values.length, values },
  });
}

export function validateTrendSnapshot(input, context) {
  if (!isRecord(input)) fail("trendSnapshot must be an object.");
  if (!isRecord(context)) fail("trend snapshot validation context is required.");
  if (input.schemaVersion !== TREND_SNAPSHOT_SCHEMA_VERSION) {
    fail(`trendSnapshot.schemaVersion must be ${TREND_SNAPSHOT_SCHEMA_VERSION}.`);
  }
  const snapshotId = digest(input.snapshotId, "trendSnapshot.snapshotId");
  const platform = requiredString(input.platform, "trendSnapshot.platform").toLowerCase();
  const locale = requiredString(input.locale, "trendSnapshot.locale");
  const requestedPlatform = requiredString(context.platform, "context.platform").toLowerCase();
  const requestedLocale = requiredString(context.locale, "context.locale");
  if (platform !== requestedPlatform || locale !== requestedLocale) {
    fail("trend snapshot platform/locale does not match the requested context.");
  }
  const evaluatedAt = instant(context.evaluatedAt, "context.evaluatedAt");
  const retrievedAt = instant(input.retrievedAt, "trendSnapshot.retrievedAt");
  const expiresAt = instant(input.expiresAt, "trendSnapshot.expiresAt");
  if (retrievedAt.milliseconds >= expiresAt.milliseconds) fail("trend snapshot freshness interval is empty.");
  if (retrievedAt.milliseconds > evaluatedAt.milliseconds || evaluatedAt.milliseconds >= expiresAt.milliseconds) {
    fail("trend snapshot is future-dated or stale.");
  }
  if (!Array.isArray(input.sources) || input.sources.length < 1) fail("trend snapshot needs at least one source URL.");
  const sources = input.sources.map((source, index) => validateSource(source, index, retrievedAt.milliseconds));
  if (new Set(sources.map((source) => source.url)).size !== sources.length) fail("trend source URLs must be unique.");

  return deepFreeze({
    schemaVersion: TREND_SNAPSHOT_SCHEMA_VERSION,
    snapshotId,
    platform,
    locale,
    retrievedAt: retrievedAt.text,
    expiresAt: expiresAt.text,
    providers: {
      trends: validateContextProvider(input.providers?.trends, "trends"),
      competitors: validateContextProvider(input.providers?.competitors, "competitors"),
    },
    sources,
    metrics: {
      trendAlignment: validateTrendMeasure(input.metrics?.trendAlignment, "trendAlignment"),
      contentSaturation: validateTrendMeasure(input.metrics?.contentSaturation, "contentSaturation"),
    },
  });
}

export function empiricalPercentile(rawValue, referenceValues) {
  const raw = finite(rawValue, "rawValue");
  if (!Array.isArray(referenceValues) || referenceValues.length < 1) fail("referenceValues must not be empty.");
  let below = 0;
  let equal = 0;
  for (let index = 0; index < referenceValues.length; index += 1) {
    const reference = finite(referenceValues[index], `referenceValues[${index}]`);
    if (reference < raw) below += 1;
    else if (reference === raw) equal += 1;
  }
  return (below + equal * 0.5) / referenceValues.length;
}
