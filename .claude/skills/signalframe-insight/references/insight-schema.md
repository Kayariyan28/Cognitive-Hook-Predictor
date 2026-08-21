# Insight evidence bundle, citation grammar, and `insight.v1`

Everything in this file is enforced by `backend/insight/`. If code and this file disagree,
one of them is a defect — fix both in the same commit.

## 1. Evidence bundle

`assemble_evidence_bundle()` builds one deterministic JSON document from **already
completed, already validated** results. It performs no I/O, no model calls, and no
measurement.

```json
{
  "schemaVersion": "insight-evidence-bundle/1",
  "source": {
    "forecastResultId": "…",
    "tribeResultId": "…" ,
    "window": null
  },
  "lanes": {
    "measured": { … }, "nanollava": { … }, "ast": { … }, "vjepa": { … },
    "asr": { … }, "ocr": { … }, "context": { … }, "tribe": { … }
  },
  "inputEvidenceHash": "<sha256 of the canonical form of everything above>"
}
```

`source.window` is `null` for a full bundle and `[start, end]` for a hook evidence card.

### Lane roots

A lane is either present or explicitly absent. **An absent lane is a marker, never a
default value.**

```json
{ "status": "absent", "reason": "<why this lane carries no evidence>" }
```

A present lane always has `"status": "present"`.

| Lane | Source in the forecast/TRIBE result | Present shape |
| --- | --- | --- |
| `measured` | `evidence.videoMetadata` + `evidence.optionalProviders.measuredAudio` | `{status, video:{durationSeconds,sizeBytes,contentType}, audio:{status, descriptors:{…}, energyPeaks:[{index,startSec,endSec,text}]}}` |
| `nanollava` | `evidence.optionalProviders.semanticModel` | `{status, keyframes:[{index,startSec,endSec,text,parsed}], warnings:[…]}` |
| `ast` | `evidence.optionalProviders.audioModel` | `{status, windows:[{index,startSec,endSec,labels:[{label,modelScore}]}], descriptors:{…}}` |
| `vjepa` | `evidence.optionalProviders.vjepa21` | `{status, windows:[{index,startSec,endSec,text,labels:[…]}], descriptors:{…}}` |
| `asr` | `evidence.optionalProviders.asr` | `{status, language, segments:[{index,startSec,endSec,text}]}` |
| `ocr` | `evidence.optionalProviders.ocr` | `{status, engine, frames:[{frameIndex,blocks:[{text,confidence,bbox}]}]}` |
| `context` | `evidence.creatorContext.value` merged with the caller's declared context | `{status, declared:{…}}` |
| `tribe` | a `tribe-cortical-descriptors/1` document | `{status, provenance:{…}, intervals:[…], parcels:[…], phases:[…], rankedBy:"rms"}` |

### Normalization rules

- **Descriptor key prefixes are stripped.** `measured_audio.rms` becomes
  `measured.audio.descriptors.rms`; `audio.ast.mean_top_label_score` becomes
  `ast.descriptors.mean_top_label_score`; `vjepa2_1.temporal_change_mean` becomes
  `vjepa.descriptors.temporal_change_mean`. **Values are copied verbatim** — only the key
  changes.
- **Per-dimension embedding features are excluded.** Any V-JEPA feature matching
  `embedding_\d+` is dropped: it is not evidence a reader or a model can cite meaningfully.
- **Observation times are renamed, never recomputed.** `startTime`/`endTime` become
  `startSec`/`endSec` with identical values.
- **NanoLLaVA observation text is kept verbatim in `text`.** `parsed` holds the decoded
  object when `text` is a JSON object, otherwise `null`.
- **The source-video content hash never enters the bundle.** It is a fingerprint of
  creator media, it is not citable evidence, and the remote-provider payload is built
  from this bundle.
- **The TRIBE tensor never enters the bundle.** Only interval series
  (`{index,startSec,durationSec,magnitude,continuity,changeRate,spatialDistribution}` copied
  from descriptor `frames[]`), the `phases[]` summaries, and at most **8** parcel summaries.
- **Parcels are ranked by `rms` descending**, the duration-weighted dispersion measure the
  repo's descriptor document actually carries, and the lane records `"rankedBy": "rms"`.
  Ties break by hemisphere then `labelIndex`, exactly as `src/tribe/descriptors.js` sorts.

### Hook evidence card

`hook_evidence_card(bundle, window=(0.0, 3.0))` returns a bundle with the same schema whose
lanes are sliced to the window and whose `source.window` is the requested window.

- An item is **inside the window** when `startSec < end` and `endSec > start`. An item that
  straddles the boundary is kept; an item that merely touches it (`startSec == end` or
  `endSec == start`) is not.
- Items with no time (OCR frames, parcels, context, video metadata, descriptors) are kept as
  is: they describe the clip, not a moment. OCR frames are kept only when the source frame
  time is known and inside the window; frames without a time are kept.
- The audio card adds `audio.onset`: `{firstEnergyPeakSec, prePeakSilenceSec}` derived only
  from energy peaks already present in the lane. Both are `null` when no peak is inside the
  window — never `0` as a stand-in.
- A lane that becomes empty after slicing turns into an absent marker with reason
  `"no <lane> evidence falls inside the requested window"`.
- The card recomputes `inputEvidenceHash` over its own content, so a hook card and a full
  bundle never collide in the cache.

### Canonical form and `inputEvidenceHash`

Canonical form is `json.dumps(value, sort_keys=True, separators=(",", ":"),
ensure_ascii=False, allow_nan=False)` with the `inputEvidenceHash` key itself omitted. The
hash is the SHA-256 hex digest of its UTF-8 encoding. Dict insertion order therefore never
changes the hash.

## 2. Citation grammar

```
citation  = lane ":" json-pointer [ "@window(" start "," end ")" ]
lane      = "measured" | "nanollava" | "ast" | "vjepa" | "asr" | "ocr" | "context" | "tribe"
```

- `json-pointer` is RFC 6901, resolved **against the lane root object**, and must start with
  `/`. `measured:/audio/descriptors/rms` resolves `bundle.lanes.measured` then
  `/audio/descriptors/rms`.
- The optional `@window(start,end)` asserts the cited item covers that interval. It resolves
  only when the pointed-at value is an object carrying `startSec`/`endSec` and those bounds
  overlap the asserted window. Numbers are plain decimals; no exponents.
- A citation into an absent lane is `citation_unresolvable`, never silently dropped.
- A citation whose lane is unknown, whose pointer does not start with `/`, or whose window
  is unparsable is `citation_malformed`.

## 3. The `insight.v1` artifact

```json
{
  "schemaVersion": "insight/2",
  "insightId": "…",
  "generatedAt": "…",
  "hookReport": {
    "windowSeconds": [0.0, 3.0],
    "whatTheHookContains": [ { "text": "…", "citations": ["…"] } ],
    "observations":        [ { "text": "…", "citations": ["…"] } ],
    "hypotheses":          [ { "id": "h1", "text": "…", "label": "untested heuristic",
                               "citations": ["…"] } ],
    "experiments":         [ { "id": "x1", "hypothesisId": "h1", "edit": "…",
                               "effort": "low",
                               "expectedSignalShift": [ { "metricPath": "…",
                                                          "direction": "increase" } ],
                               "citations": ["…"] } ]
  },
  "hookRewrites":    [ { "line": "…", "basis": "spoken", "citations": ["…"] } ],
  "phaseCommentary": [ { "phase": "early", "text": "…", "citations": ["…"] } ],
  "tribeNotes":      [ { "text": "…", "citations": ["tribe:/…"] } ],
  "behavioralOutcome": false,
  "limits": "…",
  "provenance": { … }
}
```

### Model-settable versus server-owned

The model returns **only** `hookReport`, `hookRewrites`, `phaseCommentary`, and
`tribeNotes`. Presence of any
other top-level key in the model's JSON is `server_owned_field` (for `insightId`,
`generatedAt`, `behavioralOutcome`, `limits`, `provenance`, `schemaVersion`) or
`unknown_field` (for anything else). The server injects the owned fields after validation.

### Field rules

- Unknown keys anywhere reject with `unknown_field`. Every object is closed.
- `windowSeconds` is `[start, end]`, finite, `start < end`, and must equal the window the
  bundle was sliced to. A full-bundle generation uses `[0.0, 3.0]` only if the caller asked
  for the hook window; otherwise `hookReport` may be absent.
- Text fields: 1–400 characters after stripping, no control characters.
- `line`: 1–160 characters; `basis` ∈ `{"spoken","on-screen"}`; 0–4 rewrites.
- `edit`: 1–400 characters.
- `label` must be exactly `"untested heuristic"`.
- `effort` ∈ `{"low","medium","high"}`.
- `direction` ∈ `{"increase","decrease","unchanged"}`.
- `phase` ∈ `{"early","middle","late"}` — the three phases `src/tribe/descriptors.js`
  actually produces.
- `metricPath` must be a citation-grammar path without a window, pointing at a number.
- `id` matches `^[a-z][a-z0-9-]{0,31}$`; ids are unique within their array;
  `hypothesisId` must name a hypothesis in the same artifact.
- Array bounds: `whatTheHookContains` 1–8, `observations` 0–8, `hypotheses` 0–6,
  `experiments` 0–6, `expectedSignalShift` 1–4, `phaseCommentary` 0–3, `tribeNotes` 0–4,
  `citations` 1–6 per item.
- Every item carries at least one citation; an empty list is `missing_citation`.

## 4. Numeric-copy rule

For each item, collect every number reachable from that item's citations (the pointed-at
value, and recursively every number inside it to a depth of 6). Then extract every numeral
in the item's text and `edit`.

A numeral `n` is accepted when any of the following holds against some reachable value `v`:

1. `n == v` exactly.
2. `n == round_to_significant(v, s)` for some `s` with `2 <= s <= 12` — the model may round,
   but not below two significant figures.
3. The numeral is written with a trailing `%` and rule 1 or 2 holds for `v * 100`.
4. The numeral equals the start or end of the requested window (times equal to the requested
   window are exempt).
5. The numeral sits inside a `hookRewrites` line. A proposed replacement line is a
   suggestion the creator may record, not an assertion about this clip, so the evidence is
   not required to contain its numbers. **Every claim-boundary rule still applies to it in
   full**, and no other field carries this exemption.

Anything else is `numeric_not_in_evidence`. Ordinal words ("first", "second") are words, not
numerals, and are unaffected.

## 5. `reasonCode` enum

| reasonCode | Meaning |
| --- | --- |
| `bundle_unavailable` | Upstream results are missing, incomplete, or unreadable |
| `provider_unavailable` | The configured provider is disabled, unverified, or not importable |
| `provider_error` | The provider was reachable but failed to produce output |
| `output_not_json` | The model output is not a single strict JSON object |
| `output_too_large` | The model output exceeds the configured byte limit |
| `schema_invalid` | Types, enums, lengths, bounds, or references failed |
| `unknown_field` | A key the schema does not define |
| `server_owned_field` | The model tried to set a server-owned field |
| `missing_citation` | An item carries no citation |
| `citation_malformed` | The citation does not parse under the grammar |
| `citation_unresolvable` | The citation parses but does not resolve in the bundle |
| `numeric_not_in_evidence` | A numeral is not a copy of cited evidence |
| `claim_boundary_violation` | See `claim-boundaries.md` |

`validate_insight(raw_json, bundle)` returns either `{"status": "valid", "artifact": {…}}` or
`{"status": "rejected", "reasonCode": …, "detail": …}`. It is all-or-nothing: no partial
acceptance and no repair of malformed JSON.

## 6. Provenance manifest

```json
{
  "schemaVersion": "insight-provenance/1",
  "provider": "mlx-local",
  "modelId": "…", "modelRevision": "…",
  "promptTemplateId": "hook-doctor.v1", "promptHash": "…",
  "temperature": 0, "maxOutputTokens": 1024,
  "inputEvidenceHash": "…", "outputHash": "…",
  "hookOnly": true,
  "startedAt": "…", "completedAt": "…", "elapsedSeconds": 0.0,
  "behavioralOutcome": false
}
```

`outputHash` is the SHA-256 of the canonical form of the validated artifact's model-settable
fields. The cache key is
`sha256(inputEvidenceHash, promptHash, provider, modelRevision, temperature)`.

## 7. Required fixtures

Every fixture below lives in `backend/tests/insight_fixtures/` and runs in the default
backend suite. Adversarial fixtures assert their exact `reasonCode`; valid twins must pass.

| Fixture | Expected |
| --- | --- |
| `valid_full_artifact.json` | valid |
| `valid_hook_only_artifact.json` | valid |
| `unknown_top_level_key.json` | `unknown_field` |
| `unknown_nested_key.json` | `unknown_field` |
| `server_owned_behavioral_outcome.json` | `server_owned_field` |
| `server_owned_limits.json` | `server_owned_field` |
| `server_owned_provenance.json` | `server_owned_field` |
| `empty_citations.json` | `missing_citation` |
| `malformed_citation_no_lane.json` | `citation_malformed` |
| `malformed_citation_bad_window.json` | `citation_malformed` |
| `unknown_lane_citation.json` | `citation_malformed` |
| `absent_lane_citation.json` | `citation_unresolvable` |
| `dangling_pointer_citation.json` | `citation_unresolvable` |
| `window_outside_interval.json` | `citation_unresolvable` |
| `invented_number.json` | `numeric_not_in_evidence` |
| `over_rounded_number.json` | `numeric_not_in_evidence` |
| `rounded_number_valid_twin.json` | valid |
| `window_time_exempt_valid_twin.json` | valid |
| `hypothesis_missing_label.json` | `schema_invalid` |
| `effort_enum_invalid.json` | `schema_invalid` |
| `phase_enum_invalid.json` | `schema_invalid` |
| `text_too_long.json` | `schema_invalid` |
| `too_many_experiments.json` | `schema_invalid` |
| `unknown_hypothesis_reference.json` | `schema_invalid` |
| `metric_path_not_numeric.json` | `schema_invalid` |
