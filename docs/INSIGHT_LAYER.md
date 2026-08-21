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
  set `INSIGHT_CLOUD_ENABLED=true` and supply a key.

### What leaves the machine when the remote provider is enabled

Exactly one thing: the evidence bundle, as JSON, plus the static prompt
template. Concretely that is the measured audio and video descriptors, the
keyframe and AudioSet observations, the V-JEPA summary descriptors, the
transcript and on-screen text when those branches ran, the TRIBE interval and
parcel summaries, and **the publishing context the creator typed in** —
caption, topic, locale, and account notes are evidence the Hook Doctor reads.

What never leaves: the video, its audio, its frames, the cortical tensor, any
file path, and the source video's content hash, which is excluded from the
bundle precisely because it is a fingerprint of creator media rather than
citable evidence. Operators who consider their declared context sensitive
should leave `INSIGHT_CLOUD_ENABLED=false`, which is the default.

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

### How the three layers relate

They are not three copies of one rule. They sit in a strict order of trust, and
each one exists because the one before it can fail.

| Layer | Where | What it can do | Why it is not enough |
| --- | --- | --- | --- |
| Prompt | `backend/insight/prompts/hook_doctor_v1.py` | Make a compliant model produce compliant text most of the time | A model can ignore any instruction; nothing verifies that it did not |
| Validator | `backend/insight/validation.py` | Decide, deterministically, whether an artifact may be published at all | It can only catch what its vocabulary and rules describe |
| CI fixtures and judge | `backend/tests/insight_fixtures/`, `backend/insight/judge.py` | Prove the validator still catches what it caught yesterday, and notice phrasing the vocabulary misses | The judge is itself a model, and it never gates a request |

The consequences of that ordering are the rules to work by:

- **Only layer 2 publishes or refuses.** Layers 1 and 3 never gate a request.
  `judge.py` is not imported by the service, the router, the validator, or the
  bundle assembler, and a test asserts that.
- **When the model violates a boundary, layer 1 changes.** Loosening the
  validator to make output pass is a defect, not a fix.
- **When the validator misses a violation, layer 3 found it — extend layer 2.**
  Add the term or rule to `claim_terms.py`, add the red-team fixture and its
  valid twin, and let the fixture keep the rule alive.
- **The vocabulary lives in one module.** `claim_terms.py` is the source;
  `src/insight/claim-terms.json` is generated from it, and a test fails if the
  two drift or if a second hard-coded list appears anywhere in the repository.
- **The frontend is checked too.** `tests/insight-claim-terms.test.mjs` reads
  that exported JSON and fails when an insight component hard-codes a forbidden
  outcome or mental-state claim in its own strings — a rule the model-facing
  layers cannot enforce, because those strings were written by us.

The judge runs only when `INSIGHT_JUDGE_ENABLED` is set and a provider is
configured; it is skipped by default and in CI. An unreadable verdict from it
is treated as a failure, never as a clean pass.

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
| `GET /api/insight/v1/results/{id}/evidence` | The exact evidence bundle an artifact cites, so a citation chip can reveal a real value. |
| `POST /api/insight/v1/experiments` | Track one Hook Doctor experiment. Body `{insightId, experimentId}`. |
| `POST /api/insight/v1/experiments/{id}/edited` | Record that the creator made the edit. |
| `POST /api/insight/v1/experiments/{id}/variant` | Attach a second analysed clip and measure the declared signal shifts. |
| `GET /api/insight/v1/experiments` | Tracked experiments, newest first. |
| `GET /api/insight/v1/experiments/{id}` | One tracked experiment. |

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

## Experiments

The insight lane proposes edits; the experiment tracker is what closes that
loop. An experiment moves through `proposed` → `edited` → `compared`, carries
the hypothesis it came from and the signal shift it was proposed to produce,
and links the baseline result and — once attached — the variant result.

When a variant is attached, each declared `metricPath` is resolved in both
clips' evidence bundles and the change is classified with the same tolerance
rule `src/tribe/variantComparison.js` already uses for A/B comparison; a test
reads that constant out of the JavaScript so a second rule cannot drift into
existence. Each metric is then reported as:

| match | Meaning |
| --- | --- |
| `matched` | The measured signal moved in the direction the hypothesis expected |
| `opposite` | It moved the other way |
| `unmatched` | It did not move beyond the comparison tolerance |
| `unmeasured` | The metric path does not resolve to a number in one of the two results |

`unmeasured` is a first-class state: a metric that cannot be measured in both
clips is never quietly treated as unchanged.

A measured delta is a change in a measured signal between two analysed clips.
It is not an audience outcome, not a result, and not evidence that the
hypothesis was correct — no clip in this system has ever been shown to an
audience.
