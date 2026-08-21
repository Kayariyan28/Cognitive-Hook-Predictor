/**
 * Client-side contract for the descriptive insight lane.
 *
 * The backend is authoritative: it already validated this artifact against the
 * evidence bundle before publishing it. These checks exist so the UI refuses to
 * render a document that does not match the contract it was written for, rather
 * than rendering half of one.
 */

export const INSIGHT_SCHEMA_VERSION = "insight/1";
export const INSIGHT_BUNDLE_SCHEMA_VERSION = "insight-evidence-bundle/1";
export const HYPOTHESIS_LABEL = "untested heuristic";

export const INSIGHT_LANES = Object.freeze([
  "measured",
  "nanollava",
  "ast",
  "vjepa",
  "asr",
  "ocr",
  "context",
  "tribe",
]);

export const EFFORT_ORDER = Object.freeze(["low", "medium", "high"]);
export const PHASE_ORDER = Object.freeze(["early", "middle", "late"]);
export const DIRECTIONS = Object.freeze(["increase", "decrease", "unchanged"]);

const CITATION_PATTERN =
  /^([a-z][a-z0-9]*):(\/[^@]*)(?:@window\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\))?$/;

export const TRIBE_STANDING_CAPTION =
  "TRIBE values are predicted average-subject cortical BOLD, not audience behavior.";

function deepFreeze(value, seen = new WeakSet()) {
  if (!value || typeof value !== "object" || seen.has(value)) return value;
  seen.add(value);
  for (const child of Object.values(value)) deepFreeze(child, seen);
  return Object.isFrozen(value) ? value : Object.freeze(value);
}

function contractError(message) {
  const error = new Error(message);
  error.code = "invalid_contract";
  return error;
}

function requireObject(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw contractError(`The insight artifact omitted ${label}.`);
  }
  return value;
}

function requireArray(value, label) {
  if (!Array.isArray(value)) throw contractError(`The insight artifact omitted ${label}.`);
  return value;
}

function requireText(value, label) {
  if (typeof value !== "string" || !value.trim()) {
    throw contractError(`The insight artifact omitted ${label}.`);
  }
  return value.trim();
}

/** Parse one citation, or return null when it does not match the grammar. */
export function parseCitation(text) {
  if (typeof text !== "string") return null;
  const match = CITATION_PATTERN.exec(text);
  if (!match) return null;
  const [, lane, pointer, start, end] = match;
  if (!INSIGHT_LANES.includes(lane)) return null;
  const window = start === undefined
    ? null
    : [Number.parseFloat(start), Number.parseFloat(end)];
  if (window && !(window[0] < window[1])) return null;
  return Object.freeze({ text, lane, pointer, window });
}

function unescapeToken(token) {
  return token.replace(/~1/g, "/").replace(/~0/g, "~");
}

/** Resolve an RFC 6901 pointer, returning `undefined` when it does not resolve. */
export function resolveJsonPointer(document, pointer) {
  if (pointer === "") return document;
  if (typeof pointer !== "string" || !pointer.startsWith("/")) return undefined;
  let current = document;
  for (const token of pointer.split("/").slice(1)) {
    const key = unescapeToken(token);
    if (Array.isArray(current)) {
      if (!/^\d+$/.test(key)) return undefined;
      const index = Number.parseInt(key, 10);
      if (index >= current.length) return undefined;
      current = current[index];
    } else if (current && typeof current === "object") {
      if (!Object.prototype.hasOwnProperty.call(current, key)) return undefined;
      current = current[key];
    } else {
      return undefined;
    }
  }
  return current;
}

/**
 * Resolve one citation against the evidence bundle the artifact was generated
 * from. An absent lane resolves to nothing: the chip then says so instead of
 * showing a value that was never measured.
 */
export function resolveCitation(bundle, citationText) {
  const citation = parseCitation(citationText);
  if (!citation) {
    return Object.freeze({ status: "malformed", citation: null, value: undefined, window: null });
  }
  const lane = bundle?.lanes?.[citation.lane];
  if (!lane || lane.status !== "present") {
    return Object.freeze({
      status: "absent",
      citation,
      value: undefined,
      window: citation.window,
      reason: typeof lane?.reason === "string" ? lane.reason : `The ${citation.lane} lane carries no evidence.`,
    });
  }
  const value = resolveJsonPointer(lane, citation.pointer);
  if (value === undefined) {
    return Object.freeze({ status: "unresolved", citation, value: undefined, window: citation.window });
  }
  const seekWindow = citation.window
    ?? (value && typeof value === "object" && Number.isFinite(value.startSec) && Number.isFinite(value.endSec)
      ? [value.startSec, value.endSec]
      : null);
  return Object.freeze({ status: "resolved", citation, value, window: seekWindow });
}

function validateItem(item, label) {
  const record = requireObject(item, label);
  const citations = requireArray(record.citations, `${label} citations`);
  if (!citations.length) throw contractError(`${label} carries no citation.`);
  for (const citation of citations) {
    if (!parseCitation(citation)) throw contractError(`${label} carries a malformed citation.`);
  }
  return Object.freeze({ text: requireText(record.text, `${label} text`), citations: Object.freeze([...citations]) });
}

function validateHypothesis(item, index) {
  const label = `hypothesis ${index + 1}`;
  const record = requireObject(item, label);
  if (record.label !== HYPOTHESIS_LABEL) {
    throw contractError(`${label} is not labelled as an ${HYPOTHESIS_LABEL}.`);
  }
  return Object.freeze({
    ...validateItem(record, label),
    id: requireText(record.id, `${label} id`),
    label: HYPOTHESIS_LABEL,
  });
}

function validateExperiment(item, index, hypothesisIds) {
  const label = `experiment ${index + 1}`;
  const record = requireObject(item, label);
  if (!EFFORT_ORDER.includes(record.effort)) throw contractError(`${label} has an unknown effort.`);
  const hypothesisId = requireText(record.hypothesisId, `${label} hypothesisId`);
  if (!hypothesisIds.includes(hypothesisId)) {
    throw contractError(`${label} references a hypothesis this artifact does not contain.`);
  }
  const shifts = requireArray(record.expectedSignalShift, `${label} expectedSignalShift`);
  if (!shifts.length) throw contractError(`${label} declares no expected signal shift.`);
  const citations = requireArray(record.citations, `${label} citations`);
  if (!citations.length) throw contractError(`${label} carries no citation.`);
  return Object.freeze({
    id: requireText(record.id, `${label} id`),
    hypothesisId,
    edit: requireText(record.edit, `${label} edit`),
    effort: record.effort,
    citations: Object.freeze([...citations]),
    expectedSignalShift: Object.freeze(
      shifts.map((shift, shiftIndex) => {
        const entry = requireObject(shift, `${label} signal shift ${shiftIndex + 1}`);
        if (!DIRECTIONS.includes(entry.direction)) {
          throw contractError(`${label} declares an unknown signal direction.`);
        }
        if (!parseCitation(entry.metricPath)) {
          throw contractError(`${label} declares a malformed metric path.`);
        }
        return Object.freeze({ metricPath: entry.metricPath, direction: entry.direction });
      }),
    ),
  });
}

/** Validate one published `insight/1` artifact, or throw a contract error. */
export function validateInsightArtifact(payload) {
  const record = requireObject(payload, "artifact");
  if (record.schemaVersion !== INSIGHT_SCHEMA_VERSION) {
    throw contractError(`The insight artifact is not ${INSIGHT_SCHEMA_VERSION}.`);
  }
  if (record.behavioralOutcome !== false) {
    throw contractError("The insight artifact must declare behavioralOutcome false.");
  }
  const hookReport = requireObject(record.hookReport, "hookReport");
  const windowSeconds = requireArray(hookReport.windowSeconds, "hookReport windowSeconds");
  if (windowSeconds.length !== 2 || !windowSeconds.every((value) => Number.isFinite(value))) {
    throw contractError("The insight artifact declares an invalid hook window.");
  }
  const hypotheses = requireArray(hookReport.hypotheses, "hookReport hypotheses").map(validateHypothesis);
  const hypothesisIds = hypotheses.map((item) => item.id);
  const provenance = requireObject(record.provenance, "provenance");

  return deepFreeze({
    schemaVersion: record.schemaVersion,
    insightId: requireText(record.insightId, "insightId"),
    generatedAt: requireText(record.generatedAt, "generatedAt"),
    source: requireObject(record.source, "source"),
    behavioralOutcome: false,
    limits: requireText(record.limits, "limits"),
    hookReport: {
      windowSeconds: [Number(windowSeconds[0]), Number(windowSeconds[1])],
      whatTheHookContains: requireArray(hookReport.whatTheHookContains, "hookReport whatTheHookContains")
        .map((item, index) => validateItem(item, `hook content ${index + 1}`)),
      observations: requireArray(hookReport.observations, "hookReport observations")
        .map((item, index) => validateItem(item, `observation ${index + 1}`)),
      hypotheses,
      experiments: requireArray(hookReport.experiments, "hookReport experiments")
        .map((item, index) => validateExperiment(item, index, hypothesisIds)),
    },
    phaseCommentary: requireArray(record.phaseCommentary, "phaseCommentary").map((item, index) => {
      const entry = requireObject(item, `phase note ${index + 1}`);
      if (!PHASE_ORDER.includes(entry.phase)) {
        throw contractError(`Phase note ${index + 1} names an unknown phase.`);
      }
      return { ...validateItem(entry, `phase note ${index + 1}`), phase: entry.phase };
    }),
    tribeNotes: requireArray(record.tribeNotes, "tribeNotes").map((item, index) => {
      const note = validateItem(item, `TRIBE note ${index + 1}`);
      if (!note.citations.some((citation) => parseCitation(citation)?.lane === "tribe")) {
        throw contractError(`TRIBE note ${index + 1} does not cite the TRIBE lane.`);
      }
      return note;
    }),
    provenance: {
      provider: provenance.provider ?? null,
      modelId: provenance.modelId ?? null,
      modelRevision: provenance.modelRevision ?? null,
      promptTemplateId: provenance.promptTemplateId ?? null,
      promptHash: provenance.promptHash ?? null,
      temperature: provenance.temperature ?? null,
      inputEvidenceHash: provenance.inputEvidenceHash ?? null,
      outputHash: provenance.outputHash ?? null,
      hookOnly: provenance.hookOnly === true,
      elapsedSeconds: provenance.elapsedSeconds ?? null,
      behavioralOutcome: false,
    },
  });
}

/** Validate the evidence bundle a chip resolves against. */
export function validateInsightBundle(payload) {
  const record = requireObject(payload, "evidence bundle");
  if (record.schemaVersion !== INSIGHT_BUNDLE_SCHEMA_VERSION) {
    throw contractError(`The evidence bundle is not ${INSIGHT_BUNDLE_SCHEMA_VERSION}.`);
  }
  const lanes = requireObject(record.lanes, "evidence bundle lanes");
  for (const lane of INSIGHT_LANES) {
    const entry = requireObject(lanes[lane], `the ${lane} lane`);
    if (entry.status !== "present" && entry.status !== "absent") {
      throw contractError(`The ${lane} lane declares an unknown status.`);
    }
  }
  return deepFreeze(record);
}
