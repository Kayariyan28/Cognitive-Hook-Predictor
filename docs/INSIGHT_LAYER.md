# The insight lane

The insight lane turns evidence SignalFrame already measured into short,
cited, descriptive notes a creator can act on. It is the only part of the
project that produces natural language, and it is the part with the least
authority: it may restate evidence and propose experiments, and it may do
nothing else.

Its contract lives in `.claude/skills/signalframe-insight/`, and the code lives
in `backend/insight/` and `src/insight/`.

## What it is

- A **descriptive lane**. Every artifact declares `behavioralOutcome: false`
  and carries a server-owned `limits` string. Language-model output can never
  satisfy the behavioral-head approval contract in
  [FORECAST_ARCHITECTURE.md](../FORECAST_ARCHITECTURE.md).
- A **cited lane**. Every emitted item names the evidence it came from. A
  citation that does not resolve rejects the whole artifact.
- A **local lane by default**. The default provider is a pinned local MLX
  model. Sending derived evidence to a remote provider requires the operator to
  set `INSIGHT_CLOUD_ENABLED=true` and supply a key, and even then the payload
  is JSON evidence only — never video, audio, frames, or a tensor.

## What it is not

It is not a forecast, a score, a ranking, or a second opinion about how a clip
will perform. It produces no number that was not already measured. A
"hypothesis" in a Hook Doctor report is an untested heuristic with that exact
label, and an "experiment" is an edit the creator could make and then
re-measure — not a prediction about an audience.

## The evidence bundle

`backend/insight/bundle.py` assembles one JSON document from a completed
forecast result and, optionally, a `tribe-cortical-descriptors/1` creator-report
document. It has eight lanes:

| Lane | Carries |
| --- | --- |
| `measured` | Server media metadata and measured PCM/STFT audio descriptors |
| `nanollava` | Keyframe semantic observations |
| `ast` | AudioSet label windows |
| `vjepa` | V-JEPA 2.1 window observations and summary descriptors |
| `asr` | Transcript segments, when the transcript branch ran |
| `ocr` | On-screen text blocks, when the on-screen-text branch ran |
| `context` | Creator-declared publishing context |
| `tribe` | Cortical interval, phase, and top-8 parcel summaries |

A lane with no evidence is `{"status": "absent", "reason": "..."}` — an
explicit marker, never a default value. The TRIBE tensor never enters the
bundle, and per-dimension V-JEPA embeddings are excluded because they are not
evidence anyone can cite meaningfully.

`hook_evidence_card(bundle, window=(0.0, 3.0))` slices the bundle to the hook
window, keeping items that straddle the boundary and dropping items that merely
touch it.

## Citations

```
citation = lane ":" json-pointer [ "@window(start,end)" ]
```

`measured:/audio/descriptors/rms` resolves the `measured` lane, then the RFC
6901 pointer. The optional window asserts that the cited item covers that
interval. A citation into an absent lane is unresolvable, and unresolvable is a
rejection.

## Three enforcement layers

The claim vocabulary lives in exactly one module,
`backend/insight/claim_terms.py`. Three layers read it, and they are ordered by
how much they are trusted:

1. **The prompt** (`hook-doctor.v1`) embeds the term lists and a distilled
   summary of [SCIENTIFIC_LIMITS.md](SCIENTIFIC_LIMITS.md), so a compliant
   model rarely produces a violation. This layer is a convenience. It is not
   trusted.
2. **The deterministic validator** (`backend/insight/validation.py`) decides
   what may be published. It enforces the closed schema, resolves every
   citation, applies the numeric-copy rule, and runs the sentence-scoped claim
   lint. It is all-or-nothing: a validated artifact, or a rejection carrying a
   `reasonCode`. It never repairs malformed model JSON, and it is never
   loosened to make output pass — when the model violates a boundary, the
   prompt is what changes.
3. **CI fixtures and the optional judge** run every red-team fixture in
   `backend/tests/insight_fixtures/` on every test run, including the valid
   twins that must keep passing. An env-gated LLM judge exists as a tripwire
   for phrasing the term lists do not catch; it never runs in the request path.

## The numeric-copy rule

Numbers in insight text must be copies of the evidence the item cites. A
numeral is accepted when it equals a reachable value exactly, equals that value
rounded to two or more significant figures, matches after a percent
conversion, or equals a bound of the requested window. Anything else is
`numeric_not_in_evidence`.

## The service

| Route | Purpose |
| --- | --- |
| `GET /api/insight/v1/status` | Provider readiness, pinned model identity and revision, prompt template version, config summary. The API key appears only as a boolean. |
| `POST /api/insight/v1/generate` | Body `{forecastResultId, tribeResultId?, tribeDescriptors?, hookOnly?}`. Returns an artifact or an explicit unavailable document. |
| `GET /api/insight/v1/results/{id}` | One published artifact. |
| `GET /api/insight/v1/rejections/{id}` | One persisted rejection record, including the offending sentence. |

Responses are `private, no-store`. A malformed request is an HTTP 400 with the
repo's `{code, message}` detail. A lane-level failure is an HTTP 200 carrying
`{"unavailable": true, "reasonCode": ..., "detail": ...}`, matching how every
other SignalFrame branch reports unavailability inside an otherwise successful
response.

Artifacts are staged, fsynced, and atomically renamed before they are visible.
The cache key is `sha256(inputEvidenceHash, promptHash, provider,
modelRevision, temperature)`, so an exact repeat returns the same artifact and
any change to evidence, prompt, model, or provider forces a new generation.
**Rejections are persisted but never cached as successes**: the same request
runs the model again rather than serving a refusal from disk.

## Reading a rejection

`reasonCode` tells you which layer refused, and `detail` tells you why.

| reasonCode | What to look at |
| --- | --- |
| `bundle_unavailable` | The upstream result, or a missing TRIBE descriptor document |
| `provider_unavailable` | `/status` — provider disabled, key missing, model revision unpinned or unverified |
| `provider_error` | The provider ran and failed; check backend logs |
| `output_not_json`, `output_too_large` | The model's raw output |
| `schema_invalid`, `unknown_field`, `server_owned_field` | The artifact shape |
| `missing_citation`, `citation_malformed`, `citation_unresolvable` | The citations, and whether the lane is present |
| `numeric_not_in_evidence` | The numeral named in the detail |
| `claim_boundary_violation` | The `term`, `sentence`, and `itemPath` in the detail |

For a claim-boundary rejection, the fix is the prompt template or the bundle —
never the validator.
