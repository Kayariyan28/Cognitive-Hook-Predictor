const PRODUCTION_CALIBRATED = "production-calibrated";
const VALIDATED_CONTEXT_SNAPSHOT = "validated-context-snapshot";
const CONTEXT_PERCENTILE = "context-cohort-percentile";
const CONFIDENCE_LEVELS = new Set(["low", "medium", "high"]);

function normalizedToken(value) {
  return typeof value === "string"
    ? value.trim().toLowerCase().replace(/[\s_]+/g, "-")
    : "";
}

function firstString(...values) {
  return values.find((value) => typeof value === "string" && value.trim())?.trim() || "";
}

function metricStatus(metric) {
  return normalizedToken(
    typeof metric?.status === "string"
      ? metric.status
      : metric?.status?.code ?? metric?.availability?.status,
  );
}

function metricValidationStatus(metric) {
  return normalizedToken(
    metric?.validation?.status
      ?? metric?.validation?.level
      ?? metric?.calibration?.status
      ?? metric?.calibration
      ?? metric?.validationStatus,
  );
}

export function metricCanShowNumber(metric) {
  const hasValue = metric?.value !== null && metric?.value !== undefined && metric?.value !== "";
  const contextSnapshotApproved = metric?.behavioralOutcome === false
    && normalizedToken(metric?.scoreKind) === CONTEXT_PERCENTILE
    && metricValidationStatus(metric) === VALIDATED_CONTEXT_SNAPSHOT;
  return metricStatus(metric) === "available"
    && (metricValidationStatus(metric) === PRODUCTION_CALIBRATED || contextSnapshotApproved)
    && hasValue
    && Number.isFinite(Number(metric?.value));
}

function metricCanShowCategoricalValue(metric, spec) {
  return spec?.valueKind === "categorical"
    && metricStatus(metric) === "available"
    && metricValidationStatus(metric) === PRODUCTION_CALIBRATED
    && CONFIDENCE_LEVELS.has(normalizedToken(metric?.value));
}

function formatNumber(value, digits = 0) {
  const roundedDigits = Number.isInteger(digits) ? Math.min(2, Math.max(0, digits)) : 0;
  return Number(value).toFixed(roundedDigits);
}

export function presentForecastMetric(metric, spec = {}) {
  const canShowNumber = metricCanShowNumber(metric);
  const canShowCategorical = metricCanShowCategoricalValue(metric, spec);
  const canShowValue = canShowNumber || canShowCategorical;
  const status = metricStatus(metric) || "unavailable";
  const validationStatus = metricValidationStatus(metric) || "not-calibrated";
  const unit = firstString(metric?.unit, spec.unit);
  let reason = firstString(
    metric?.reason,
    metric?.status?.reason,
    metric?.availability?.reason,
    metric?.validation?.reason,
    metric?.calibration?.reason,
  );

  if (!reason && status === "available" && !canShowValue) {
    reason = "Withheld because this value does not have the required production calibration or validated context snapshot.";
  } else if (!reason && status !== "available") {
    reason = "No production-calibrated outcome head is connected.";
  } else if (!reason && !Number.isFinite(Number(metric?.value))) {
    reason = "The calibrated head did not return a numeric result.";
  }

  return {
    canShowValue,
    canShowNumber,
    canShowCategorical,
    status,
    validationStatus,
    valueText: canShowNumber
      ? formatNumber(metric.value, metric?.digits ?? spec.digits)
      : canShowCategorical
        ? `${normalizedToken(metric.value).charAt(0).toUpperCase()}${normalizedToken(metric.value).slice(1)}`
        : "—",
    unitText: canShowNumber ? unit : "",
    evidenceLabel: canShowNumber && validationStatus === VALIDATED_CONTEXT_SNAPSHOT
      ? "Source-backed context percentile"
      : canShowValue
        ? "Production-calibrated result"
        : "",
    reason,
  };
}

function asMetricMap(forecast) {
  const candidates = [forecast?.metrics, forecast?.forecast?.metrics, forecast?.scores];
  for (const candidate of candidates) {
    if (Array.isArray(candidate)) {
      return Object.fromEntries(candidate
        .filter((metric) => metric && typeof metric === "object")
        .map((metric) => [metric.key ?? metric.id ?? metric.name, metric]));
    }
    if (candidate && typeof candidate === "object") return candidate;
  }
  return {};
}

export function resolveForecastMetric(forecast, spec) {
  const metricMap = asMetricMap(forecast);
  if (spec.key === "confidence" && forecast?.confidence !== null && forecast?.confidence !== undefined) {
    const confidence = forecast.confidence;
    if (confidence && typeof confidence === "object") {
      return {
        ...confidence,
        value: confidence.value ?? confidence.level,
      };
    }
    return {
      value: confidence,
      status: "unavailable",
      reason: "A bare confidence label was rejected because availability and calibration provenance are missing.",
    };
  }
  const aliases = [spec.key, ...(spec.aliases || [])];
  for (const alias of aliases) {
    if (Object.hasOwn(metricMap, alias)) {
      const raw = metricMap[alias];
      return raw && typeof raw === "object"
        ? raw
        : {
            value: raw,
            status: "unavailable",
            reason: "A bare value was rejected because availability and calibration provenance are missing.",
          };
    }
  }
  return {
    status: "unavailable",
    reason: firstString(
      forecast?.reason,
      forecast?.status?.reason,
      "No production-calibrated outcome head is connected.",
    ),
  };
}

export const FORECAST_METRIC_SPECS = Object.freeze([
  { key: "hookStrength", aliases: ["hook_strength", "hook"], label: "Hook Strength", lane: "intrinsic", unit: "/100" },
  { key: "fiveSecondRetention", aliases: ["five_second_retention", "retention5s", "fiveSecRetention"], label: "5-sec Retention", lane: "intrinsic", unit: "%" },
  { key: "completionProbability", aliases: ["completion_probability", "completion"], label: "Completion Probability", lane: "intrinsic", unit: "%" },
  { key: "rewatchPotential", aliases: ["rewatch_potential", "rewatch"], label: "Rewatch Potential", lane: "intrinsic", unit: "/100" },
  { key: "sharePotential", aliases: ["share_potential", "share"], label: "Share Potential", lane: "intrinsic", unit: "/100" },
  { key: "trendAlignment", aliases: ["trend_alignment", "trend"], label: "Trend Alignment", lane: "context", unit: "/100" },
  { key: "contentSaturation", aliases: ["content_saturation", "saturation"], label: "Content Saturation", lane: "context", unit: "/100" },
  { key: "expectedFirstPassWatch", aliases: ["predictedEngagement", "expected_first_pass_watch", "expectedFirstPassWatchPercent", "firstPassWatch"], label: "Expected first-pass watch %", lane: "intrinsic", unit: "%" },
  { key: "viralityPotential", aliases: ["virality_potential", "virality"], label: "Virality Potential", lane: "outcome", unit: "/100" },
  { key: "confidence", aliases: ["forecast_confidence", "modelConfidence"], label: "Confidence", lane: "outcome", valueKind: "categorical", unit: "" },
]);
