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
