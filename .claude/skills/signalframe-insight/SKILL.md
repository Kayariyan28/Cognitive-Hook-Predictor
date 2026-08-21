---
name: signalframe-insight
description: >-
  Build and change the SignalFrame LLM insight layer — the evidence bundle assembler, the
  Hook Doctor, the insight validators, the /api/insight/v1 service, the ASR/OCR evidence
  lanes, and the experiment tracker. Use this skill for anything that turns measured
  evidence into natural language, cites evidence, or renders insight text to a creator.
---

# SignalFrame insight layer

The insight layer turns **already-validated evidence** into natural language a creator can
act on. It never becomes a new measurement, a new score, or a new prediction.

Read `CLAUDE.md`, `docs/SCIENTIFIC_LIMITS.md`, and `backend/forecast/README.md` before
writing code. This skill adds insight-specific rules on top of the project invariants; where
they overlap, the project invariants win.

## The five insight invariants

1. **Evidence in, language out.** The model receives a JSON evidence bundle assembled from
   completed, validated results. It never receives media, tensors, embeddings, file paths,
   or upload directories. It never triggers a measurement.
2. **Every claim is cited or it is not published.** Each emitted item carries citations into
   the bundle using the grammar in `references/insight-schema.md`. An unresolvable citation
   rejects the whole artifact.
3. **No numeral the evidence does not contain.** Numbers in insight text must be copied from
   the cited evidence within the documented rounding tolerance. See the numeric-copy rule in
   `references/insight-schema.md`.
4. **Hypotheses are labeled, outcomes are forbidden.** A causal or audience-outcome claim is
   a rejection, not a style problem. TRIBE-cited text carries the strictest limits. See
   `references/claim-boundaries.md`.
5. **The layer is descriptive.** Every insight artifact carries `behavioralOutcome: false`
   and a server-owned `limits` string. The model cannot set either. LLM output can never
   satisfy the behavioral-head approval contract.

## Where the layer lives

| Concern | Module |
| --- | --- |
| Bundle assembly and hook card | `backend/insight/bundle.py` |
| Citation grammar and resolution | `backend/insight/citations.py` |
| Claim-boundary term data (single source) | `backend/insight/claim_terms.py` |
| Schema, numeric-copy, lint, `validate_insight` | `backend/insight/validation.py` |
| Provider interface and adapters | `backend/insight/providers/` |
| Versioned prompt template | `backend/insight/prompts/hook_doctor_v1.py` |
| Provenance manifest builder | `backend/insight/provenance.py` |
| Persistence, caching, service wiring | `backend/insight/service.py`, `backend/insight/router.py` |
| Frontend client, presentation, panel | `src/insight/`, `src/components/InsightPanel.jsx` |
| Tests and red-team fixtures | `backend/tests/test_insight_*.py`, `backend/tests/insight_fixtures/` |

## Non-negotiable implementation rules

- **Fail closed, all or nothing.** `validate_insight` returns a validated artifact or a
  rejection `{reasonCode, detail}`. There is no partial acceptance, no field dropping, and
  no repair of malformed model JSON.
- **Rejections are persisted, never cached as successes.** A rejection record is written so
  an operator can read the offending sentence; a later identical request re-runs the model.
- **Providers fail closed and never chain.** If the configured provider is unavailable, the
  answer is `provider_unavailable`. There is no silent fallback to another provider.
- **Cloud is opt-in and payload-bounded.** The remote provider is gated by
  `INSIGHT_CLOUD_ENABLED=true` plus `ANTHROPIC_API_KEY`, and its payload is the derived JSON
  bundle only. That code path must not import anything that reads upload or frame
  directories.
- **Term lists live in exactly one module.** `backend/insight/claim_terms.py` is the source
  of truth for the prompt template, the validator, and the exported JSON the frontend static
  check reads. Never restate a term list anywhere else.
- **New evidence lanes are evidence branches first.** An ASR or OCR lane follows the existing
  branch contract in `backend/forecast/workers/`: pinned artifact verified before load,
  deterministic preprocessing, strict output schema, branch-local failure,
  `behavioralOutcome: false`, provenance. The insight lane is downstream of that, never a
  shortcut around it.
- **Tighten the prompt, not the validator.** When the model produces a violation, the fix is
  the prompt template or the bundle. Weakening a validator rule to make output pass is a
  defect.

## Working order

The layer was built and must stay buildable in this order; each step is independently
testable without the next one existing.

1. Bundle assembler and hook evidence card (pure functions, no LLM).
2. Validators and red-team fixtures (no LLM).
3. Providers and the versioned prompt template.
4. The HTTP service with caching and atomic persistence.
5. The creator-facing panel.
6. ASR and OCR evidence lanes.
7. The experiment tracker that closes the loop.

## References

- `references/insight-schema.md` — bundle shape, citation grammar, `insight.v1` artifact,
  numeric-copy rule, `reasonCode` enum, provenance manifest, required fixtures.
- `references/claim-boundaries.md` — term categories, scoping rules, exemptions, violation
  shape, required red-team fixtures and their valid twins.
- `references/mac-deps.md` — pinned Mac-only dependencies, availability probes, and the
  no-CUDA rule.
