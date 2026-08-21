/**
 * Presentation rules for the insight panel.
 *
 * Unavailable is a first-class state here, not a loading state that eventually
 * turns into content. Every reasonCode maps to a specific sentence a creator can
 * act on, and nothing in this module invents a value or a score.
 */

import {
  EFFORT_ORDER,
  PHASE_ORDER,
  TRIBE_STANDING_CAPTION,
  resolveCitation,
} from "./contract.js";

export const HOOK_REPORT_SECTION_ORDER = Object.freeze([
  "whatTheHookContains",
  "observations",
  "hypotheses",
  "experiments",
]);

export const HOOK_REPORT_SECTION_LABELS = Object.freeze({
  whatTheHookContains: "What the hook contains",
  observations: "Observations",
  hypotheses: "Hypotheses",
  experiments: "Experiments",
});

const UNAVAILABLE_MESSAGES = Object.freeze({
  bundle_unavailable:
    "The evidence for this clip is incomplete, so there is nothing to describe yet. Finish the evidence job first.",
  provider_unavailable:
    "No insight model is available. Check the provider, the pinned model revision, and whether the local snapshot is present.",
  provider_error: "The insight model ran and returned nothing usable. Nothing was published.",
  output_not_json: "The model's response was not valid JSON. It was rejected whole rather than repaired.",
  output_too_large: "The model's response exceeded the configured size limit and was rejected.",
  schema_invalid: "The model's response did not match the insight contract and was rejected.",
  unknown_field: "The model's response contained a field the contract does not define and was rejected.",
  server_owned_field: "The model tried to set a field only the server may set, so nothing was published.",
  missing_citation: "The model wrote a claim with no evidence behind it, so nothing was published.",
  citation_malformed: "The model wrote a citation that does not parse, so nothing was published.",
  citation_unresolvable:
    "The model cited evidence this clip does not have, so nothing was published.",
  numeric_not_in_evidence:
    "The model wrote a number the evidence does not contain, so nothing was published.",
  claim_boundary_violation:
    "The model made an outcome or mental-state claim this project does not support, so nothing was published.",
});

const GENERIC_UNAVAILABLE =
  "The insight lane could not produce anything for this clip, so nothing was published.";

/** Turn a fail-closed document into an explicit, specific creator-facing state. */
export function presentUnavailable(document) {
  const reasonCode = typeof document?.reasonCode === "string" ? document.reasonCode : null;
  const detail = document?.detail;
  const violation = detail && typeof detail === "object" && !Array.isArray(detail) ? detail : null;
  return Object.freeze({
    status: "unavailable",
    reasonCode,
    message: (reasonCode && UNAVAILABLE_MESSAGES[reasonCode]) || GENERIC_UNAVAILABLE,
    detailText: violation
      ? `“${violation.sentence ?? ""}” — the term “${violation.term ?? ""}” is not allowed in this lane.`
      : typeof detail === "string" ? detail : null,
    rejectionId: typeof document?.rejectionId === "string" ? document.rejectionId : null,
  });
}

/** Describe why generation is offline before the creator presses anything. */
export function presentProviderState(status) {
  const provider = status?.provider;
  if (!provider || typeof provider !== "object") {
    return Object.freeze({
      available: false,
      message: "The insight service did not report a provider.",
      providerName: null,
      modelId: null,
      modelRevision: null,
    });
  }
  return Object.freeze({
    available: provider.available === true,
    message: typeof provider.reason === "string" ? provider.reason : "",
    providerName: provider.provider ?? null,
    modelId: provider.model?.id ?? null,
    modelRevision: provider.model?.revision ?? null,
    promptTemplateId: status?.promptTemplate?.id ?? null,
  });
}

/** Experiments, cheapest edit first; ties keep the model's order. */
export function experimentsByEffort(experiments = []) {
  return [...experiments]
    .map((experiment, index) => ({ experiment, index }))
    .sort((left, right) => {
      const effortDelta =
        EFFORT_ORDER.indexOf(left.experiment.effort) - EFFORT_ORDER.indexOf(right.experiment.effort);
      return effortDelta !== 0 ? effortDelta : left.index - right.index;
    })
    .map((entry) => entry.experiment);
}

/** Phase notes in clip order, whatever order the model wrote them in. */
export function phaseCommentaryInClipOrder(notes = []) {
  return [...notes].sort(
    (left, right) => PHASE_ORDER.indexOf(left.phase) - PHASE_ORDER.indexOf(right.phase),
  );
}

function formatValue(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return String(value);
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return `${value.length} item${value.length === 1 ? "" : "s"}`;
  return Object.keys(value).join(", ");
}

/**
 * Build what a citation chip reveals when the creator clicks it: the evidence
 * value if it resolves, and an explicit reason if it does not.
 */
export function presentCitation(bundle, citationText) {
  const resolved = resolveCitation(bundle, citationText);
  const lane = resolved.citation?.lane ?? null;
  const base = {
    label: lane ? `${lane}${resolved.citation.pointer}` : String(citationText),
    lane,
    pointer: resolved.citation?.pointer ?? null,
    seekSeconds: Number.isFinite(resolved.window?.[0]) ? resolved.window[0] : null,
    window: resolved.window ?? null,
  };
  if (resolved.status === "resolved") {
    return Object.freeze({
      ...base,
      status: "resolved",
      valueText: formatValue(resolved.value),
      value: resolved.value,
      caption: lane === "tribe" ? TRIBE_STANDING_CAPTION : null,
    });
  }
  const reason = resolved.status === "absent"
    ? resolved.reason
    : resolved.status === "malformed"
      ? "This citation does not parse under the citation grammar."
      : "This citation does not resolve in the evidence this artifact was generated from.";
  return Object.freeze({ ...base, status: resolved.status, valueText: null, value: undefined, reason, caption: null });
}

/**
 * Describe the deep link that hands one experiment to the existing A/B compare
 * flow. It carries identity only — the comparison itself stays where it lives.
 */
export function buildAbCompareLink(experiment, { forecastResultId = null, insightId = null } = {}) {
  if (!experiment || typeof experiment !== "object") return null;
  return Object.freeze({
    action: "open-ab-compare",
    insightId,
    baselineResultId: forecastResultId,
    experimentId: experiment.id,
    hypothesisId: experiment.hypothesisId,
    edit: experiment.edit,
    effort: experiment.effort,
    expectedSignalShift: Object.freeze(
      (experiment.expectedSignalShift ?? []).map((shift) =>
        Object.freeze({ metricPath: shift.metricPath, direction: shift.direction }),
      ),
    ),
  });
}

export { TRIBE_STANDING_CAPTION };

export const EXPERIMENT_STATUS_LABELS = Object.freeze({
  proposed: "Proposed",
  edited: "Edit made",
  compared: "Compared",
});

const MATCH_LABELS = Object.freeze({
  matched: "Signal moved as expected",
  opposite: "Signal moved the other way",
  unmatched: "Signal did not move",
  unmeasured: "Not measured in both results",
});

/**
 * Present one tracked experiment. Expected direction is an untested heuristic;
 * measured direction is a change in a measured signal. Neither is an outcome.
 */
export function presentExperiment(record) {
  if (!record || typeof record !== "object") return null;
  const [baselineResultId = null, variantResultId = null] = Array.isArray(record.linkedResultIds)
    ? record.linkedResultIds
    : [];
  const deltas = Array.isArray(record.measuredDeltas) ? record.measuredDeltas : [];
  return Object.freeze({
    id: record.id ?? null,
    status: record.status ?? "proposed",
    statusLabel: EXPERIMENT_STATUS_LABELS[record.status] ?? "Proposed",
    edit: record.edit ?? "",
    effort: record.effort ?? null,
    sourceInsightId: record.sourceInsightId ?? null,
    baselineResultId,
    variantResultId,
    behavioralOutcome: false,
    signals: Object.freeze(
      (Array.isArray(record.expectedSignalShift) ? record.expectedSignalShift : []).map((shift) => {
        const measured = deltas.find((entry) => entry?.metricPath === shift.metricPath) ?? null;
        const match = measured?.match ?? "unmeasured";
        return Object.freeze({
          metricPath: shift.metricPath,
          expectedDirection: shift.direction,
          observedDirection: measured?.observedDirection ?? null,
          delta: measured?.delta ?? null,
          baselineValue: measured?.baselineValue ?? null,
          variantValue: measured?.variantValue ?? null,
          match,
          matchLabel: MATCH_LABELS[match] ?? MATCH_LABELS.unmeasured,
          reason: measured?.reason ?? null,
        });
      }),
    ),
  });
}

export { MATCH_LABELS };

export const MARKER_LABELS = Object.freeze({
  "audio-peak": "Sound peak",
  "spoken-segment": "Speech",
  "on-screen-text": "On-screen text",
  keyframe: "Keyframe",
  "visual-window": "Visual window",
  "cortical-interval": "Cortical interval",
});

export const CHECK_STATUS_LABELS = Object.freeze({
  clear: "Clear",
  flagged: "Worth a look",
  unmeasured: "Not measured",
});

/**
 * Lay markers out as percentages of the hook window so a timeline can be drawn
 * with no measurement of its own. A marker outside the window is dropped rather
 * than clamped, because a clamped marker would sit at a time it never occurred.
 */
export function layoutHookTimeline(readout) {
  const bounds = Array.isArray(readout?.windowSeconds) ? readout.windowSeconds : [0, 3];
  const [start, end] = [Number(bounds[0]), Number(bounds[1])];
  const span = end - start;
  if (!Number.isFinite(span) || span <= 0) return Object.freeze([]);
  const markers = Array.isArray(readout?.timeline) ? readout.timeline : [];
  return Object.freeze(
    markers
      .filter((marker) => Number.isFinite(marker?.startSec) && marker.startSec < end)
      .map((marker) => {
        const from = Math.max(start, marker.startSec);
        const to = Number.isFinite(marker.endSec) ? Math.min(end, marker.endSec) : from;
        return Object.freeze({
          ...marker,
          typeLabel: MARKER_LABELS[marker.kind] ?? marker.kind,
          leftPercent: ((from - start) / span) * 100,
          widthPercent: Math.max(0, ((to - from) / span) * 100),
        });
      }),
  );
}

/** Group timeline markers into one lane per evidence kind, in a stable order. */
export function timelineLanes(readout) {
  const laid = layoutHookTimeline(readout);
  const order = Object.keys(MARKER_LABELS);
  return Object.freeze(
    order
      .map((kind) => Object.freeze({
        kind,
        label: MARKER_LABELS[kind],
        markers: Object.freeze(laid.filter((marker) => marker.kind === kind)),
      }))
      .filter((lane) => lane.markers.length > 0),
  );
}

/** Flagged checks first: they are the ones a creator can act on. */
export function presentChecklist(readout) {
  const checks = Array.isArray(readout?.checklist) ? readout.checklist : [];
  const rank = { flagged: 0, clear: 1, unmeasured: 2 };
  return Object.freeze(
    [...checks]
      .sort((left, right) => (rank[left.status] ?? 3) - (rank[right.status] ?? 3))
      .map((check) => Object.freeze({
        ...check,
        statusLabel: CHECK_STATUS_LABELS[check.status] ?? check.status,
        isConvention: check.thresholdKind === "declared-convention",
      })),
  );
}

/**
 * Present a variant comparison. A metric that moved names which cut sits
 * highest and lowest; a metric that did not move names neither, because there
 * is nothing to point at.
 */
export function presentVariantComparison(comparison) {
  if (!comparison || typeof comparison !== "object") return null;
  const variants = Array.isArray(comparison.variants) ? comparison.variants : [];
  const labels = new Map(variants.map((variant) => [variant.resultId, variant.label]));
  const metrics = Array.isArray(comparison.metrics) ? comparison.metrics : [];
  return Object.freeze({
    variants: Object.freeze([...variants]),
    behavioralOutcome: false,
    limits: comparison.limits ?? "",
    skipped: Object.freeze(
      (Array.isArray(comparison.skippedMetrics) ? comparison.skippedMetrics : []).map(
        (entry) => Object.freeze({ ...entry }),
      ),
    ),
    metrics: Object.freeze(
      metrics.map((metric) =>
        Object.freeze({
          metricPath: metric.metricPath,
          spread: metric.spread,
          differs: metric.differs === true,
          values: Object.freeze(
            (Array.isArray(metric.values) ? metric.values : []).map((entry) =>
              Object.freeze({
                ...entry,
                isLowest: metric.differs === true && entry.resultId === metric.lowestResultId,
                isHighest: metric.differs === true && entry.resultId === metric.highestResultId,
              }),
            ),
          ),
          lowestLabel: metric.differs ? labels.get(metric.lowestResultId) ?? null : null,
          highestLabel: metric.differs ? labels.get(metric.highestResultId) ?? null : null,
        }),
      ),
    ),
  });
}
