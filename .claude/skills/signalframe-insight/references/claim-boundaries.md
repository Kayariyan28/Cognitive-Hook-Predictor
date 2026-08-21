# Claim boundaries

Term data lives in exactly one module: `backend/insight/claim_terms.py`. The prompt template
embeds it, the validator enforces it, and the frontend static check reads its exported JSON.
Never restate a term list anywhere else.

## Three enforcement layers

1. **Prompt** — the `hook-doctor.v1` template embeds the term lists and the limits summary so
   a compliant model rarely produces a violation.
2. **Deterministic validator** — `claim_boundary_violations(artifact, bundle)` is the only
   layer that decides whether an artifact is published. The prompt is never trusted.
3. **CI fixtures and the optional judge** — every fixture below runs in the default backend
   suite; the env-gated judge is a tripwire, never a gate in the request path.

## Term categories

### `GLOBAL_OUTCOME_TERMS`

Platform or audience outcome claims. Forbidden in **all** insight text, whatever it cites.

`viral`, `virality`, `go viral`, `views`, `view count`, `watch time`, `retention`,
`completion rate`, `engagement rate`, `click-through`, `ctr`, `impressions`, `reach`,
`shares`, `share rate`, `followers`, `subscribers`, `algorithm`, `the algorithm`,
`conversion`, `converts`, `monetization`, `sells`, `will perform`, `outperform`,
`guarantee`, `guaranteed`, `boost`, `drive traffic`

### `GLOBAL_MENTAL_STATE_TERMS`

Attributing a mental state to viewers. Forbidden in **all** insight text.

`attention`, `attentive`, `engaged`, `engagement`, `emotion`, `emotional`, `feels`, `feel`,
`felt`, `mood`, `arousal`, `memory`, `memorable`, `remember`, `bored`, `boredom`,
`curiosity`, `curious`, `excitement`, `excited`, `interest`, `interested`, `hooked`,
`captivated`, `empathy`, `trust`, `desire`, `intent`

`hook` (the product noun) is **not** a term. `hooked` is. Matching is whole-word and
phrase-exact, case-insensitive, so `hook`, `hooks`, and `hook window` stay legal.

### `TRIBE_SCOPED_TERMS`

Reverse inference from cortical values. Forbidden **only** in items that cite `tribe:`,
where they are additionally forbidden on top of the two global lists.

`brain activity`, `brain lights up`, `lights up`, `neural attention`, `neuroscience proves`,
`the brain says`, `subconscious`, `dopamine`, `limbic`, `amygdala`, `viewers' brains`,
`measured brain`, `real brain`, `live brain`, `mind reading`, `what viewers think`

## Scoping

- Rules are **sentence-scoped**: text is split on `.`, `!`, `?`, and newlines; a violation
  names the offending sentence, not the whole item.
- Global rules run on every text field of every item, including `edit` strings.
- TRIBE-scoped rules run only on items whose `citations` include at least one `tribe:`
  citation.
- `tribeNotes` items must additionally carry at least one `tribe:` citation — an untethered
  TRIBE note is `schema_invalid`, not a boundary violation.

## Exemptions

A sentence is exempt for a matched term when either holds:

1. **Whole-sentence limiter.** The sentence contains one of:
   `untested heuristic`, `not audience behavior`, `not a prediction`, `not a measurement`,
   `not a measure of`, `no evidence`, `predicted average-subject cortical bold`,
   `predicted cortical bold`, `not measured`, `does not establish`, `cannot establish`,
   `not an outcome`.
2. **Proximate negation.** One of `not`, `never`, `no`, `cannot`, `isn't`, `aren't`,
   `doesn't`, `don't`, `without` appears in the same sentence within **60 characters** of the
   matched term.

Proximity is measured between character spans in the sentence, in either direction. This is
a deliberately mechanical approximation: it is documented, testable, and biased toward
rejecting rather than accepting.

## Violation shape

```json
{ "reasonCode": "claim_boundary_violation",
  "term": "retention",
  "sentence": "This opening should lift retention.",
  "itemPath": "/hookReport/observations/0/text" }
```

`validate_insight` surfaces the first violation in document order as
`{"status": "rejected", "reasonCode": "claim_boundary_violation", "detail": {…}}` with the
full violation object as the detail.

## Required fixtures

Each adversarial fixture asserts `claim_boundary_violation` and its exact `term`. Each valid
twin must pass validation unchanged.

| Fixture | Expected |
| --- | --- |
| `global_virality_claim.json` | violation, term `go viral` |
| `global_retention_claim.json` | violation, term `retention` |
| `global_views_claim.json` | violation, term `views` |
| `global_algorithm_claim.json` | violation, term `algorithm` |
| `global_mental_state_claim.json` | violation, term `attention` |
| `experiment_edit_outcome_claim.json` | violation, term `watch time` |
| `tribe_mental_state.json` | violation, term `emotion` |
| `tribe_brain_lights_up.json` | violation, term `lights up` |
| `tribe_reverse_inference.json` | violation, term `subconscious` |
| `asr_outcome_claim.json` | violation, term `viral` |
| `ocr_outcome_claim.json` | violation, term `hooked` |
| `negated_virality_valid_twin.json` | valid |
| `negated_retention_valid_twin.json` | valid |
| `tribe_caption_valid_twin.json` | valid |
| `untested_heuristic_valid_twin.json` | valid |
| `hook_noun_valid_twin.json` | valid |
