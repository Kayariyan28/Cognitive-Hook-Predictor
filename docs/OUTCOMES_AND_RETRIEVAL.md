# Design proposal: outcome ingestion and per-creator retrieval

> **Status: proposal. No code has been written for either feature.**
> This document exists to be argued with before anything is built. Nothing in
> it is implemented, and neither feature trains, approves, or enables a
> behavioral head.

Two additions extend the loop the insight lane opens:

1. **Manual outcome ingestion** — a creator imports post-publish metrics they
   chose to record, stored locally and linked to result IDs. These are *labels
   for a future, separately trained calibration head*. They are not a forecast,
   not evidence, and not something the insight lane may read.
2. **Per-creator retrieval** — on insight generation, the backend computes
   simple comparative context from the creator's **own past evidence bundles**
   (for example, where this clip's measured opening energy sits among their last
   N clips) and puts it in the `measured:` lane as ordinary, citable,
   validator-checkable evidence.

They are proposed together because they pull in opposite directions, and the
design only works if that tension is made structural. Retrieval is comparison
between the creator's own **measurements**. Outcome ingestion is the creator's
own **results**. Feature 2 must never be able to reach feature 1's data.

---

## 1. Manual outcome ingestion

### What it is

A CSV the creator exports from a platform's own analytics, or types by hand,
and imports through the local UI. One row is one published clip.

```csv
resultId,platform,postedAt,measuredAt,metric,value,denominator,note
3f1c2b…,reels,2026-02-03T18:00:00Z,2026-02-10T18:00:00Z,views,18432,,
3f1c2b…,reels,2026-02-03T18:00:00Z,2026-02-10T18:00:00Z,retention_5s,0.41,eligible_starts,
```

### What it is not

It is not a metric SignalFrame measured, observed, verified, or can verify. It
is **creator-declared, self-reported, unverified data**, exactly like the
existing `creatorContext`. The schema records that as a first-class field rather
than as a footnote, and the UI says it wherever a number is shown.

### Storage

A store parallel to the existing ones, under the configured private runtime
directory, atomic in the same way (stage → fsync → rename):

```
$INSIGHT_DIR/outcomes/<outcomeSetId>/outcomes.json
```

```json
{
  "schemaVersion": "creator-outcome-set/1",
  "outcomeSetId": "…",
  "importedAt": "…",
  "source": "manual-csv",
  "sourceFileSha256": "…",
  "declaredBy": "creator",
  "verified": false,
  "records": [
    {
      "resultId": "…",
      "platform": "reels",
      "postedAt": "…",
      "measuredAt": "…",
      "metric": "retention_5s",
      "value": 0.41,
      "denominator": "eligible_starts",
      "note": null
    }
  ]
}
```

Import validation is strict and fails the whole file, not row by row: unknown
metric names, a `resultId` with no published forecast result, a non-finite
value, a `measuredAt` earlier than `postedAt`, a probability outside `[0, 1]`,
or a duplicate `(resultId, platform, metric, measuredAt)` rejects the import
with a `reasonCode`. A partially imported outcome set is worse than none,
because it silently biases whatever is eventually trained on it.

### Retention and privacy

- **Local only.** No network call, no platform API, no upload. The import is a
  file read and a JSON write on the operator's machine.
- **Never sent to a provider.** Not to the local model, and specifically not to
  the remote one. The remote payload is built from the evidence bundle, and
  outcomes are structurally excluded from it (see below).
- **Operator-deletable.** `DELETE /api/insight/v1/outcomes/{outcomeSetId}`
  removes the set; a documented retention setting (default: keep until deleted)
  and a `docs/` note tell the operator this is the most sensitive store in the
  repository — it correlates specific creative content with specific
  performance, which neither the clip nor the metric does alone.
- **Not anonymous.** A `resultId` links to a persisted evidence result. Treat an
  outcome store as identifying even though it contains no name.
- **Multi-user deployment is out of scope.** The local server has no
  authentication layer; an outcome store is a reason to add one before exposing
  the service, not something to defer.

### How it stays inside the five insight invariants

| Invariant | How this addition satisfies it |
| --- | --- |
| 1. Evidence in, language out | Outcomes are not evidence and never enter a bundle lane, so the model never receives them. |
| 2. Every claim is cited | No new claims exist; there is nothing for the model to cite. |
| 3. No numeral the evidence does not contain | The model cannot cite an outcome, so it cannot copy one. A numeral matching a stored outcome would fail the numeric-copy rule like any other invented number. |
| 4. Hypotheses labeled, outcomes forbidden | Unchanged. Storing a real outcome does not license a claim about a future one. |
| 5. The layer is descriptive | Unchanged. `behavioralOutcome: false` still holds because no head exists. |

### The structural guarantee

A comment saying "outcomes never reach the prompt" is not a guarantee. The same
technique already used for the remote provider applies here:

- `backend/insight/outcomes.py` is a leaf module. `bundle.py`, `validation.py`,
  `service.py`, and `prompts/` do not import it, and a test walks the import
  graph of each and fails if any of them can reach it — the test that already
  guards `judge.py` and `anthropic_cloud.py`, extended by one entry.
- The bundle's lane list stays closed at the eight documented lanes. There is no
  `outcomes:` lane, so there is no citation that could name one, and
  `parse_citation` rejects an unknown lane as malformed before resolution.
- A red-team fixture asserts that an artifact citing `outcomes:/records/0/value`
  rejects with `citation_malformed`, and keeps asserting it forever.

### Relationship to the behavioral-head gate

These labels are *inputs to a future training run that this repository does not
perform*. Any head trained on them still has to pass the gate that is already
code-owned in `backend/forecast/calibrated_heads.py`, where
`APPROVED_TARGET_CONTRACTS` is deliberately empty and an approval is a reviewed
code change, not a data file. Nothing about having labels shortens that list:

- an immutable feature-contract hash, encoder revisions, weight hashes, and a
  calibrator hash;
- the exact target event, denominator, platform, horizon, population, locale,
  language, duration range, and account segment;
- creator-disjoint and chronological evaluation with near-duplicate grouping;
- locked-test calibration evidence (Brier, log loss, ECE, calibration
  slope/intercept, interval coverage);
- freshness, missing-modality, out-of-distribution, and deployment gates.

And a self-imported CSV from one creator's own clips is, on its own, a
**severely biased sample**: one account, one audience, one distribution history,
survivorship over what they chose to post and chose to record. It is a starting
point for a personal calibration experiment, not a training set for a general
head. The design doc should say so; the UI must too.

---

## 2. Per-creator retrieval

### What it is

Deterministic comparative context, computed by code from evidence the backend
already has, added to the `measured:` lane so the Hook Doctor can say "the
lowest measured opening energy of your last 12 clips" **with a citation** and
have that sentence checked like any other.

### Proposed lane shape

`measured` gains one sub-object, and the bundle version bumps to
`insight-evidence-bundle/2` because the lane shape is closed:

```json
"comparative": {
  "status": "present",
  "corpus": {
    "kind": "local-forecast-results",
    "clipCount": 12,
    "oldestCreatedAt": "…",
    "newestCreatedAt": "…"
  },
  "metrics": [
    {
      "metricPath": "measured:/audio/descriptors/silent_window_fraction",
      "value": 0.1875,
      "percentile": 8.3,
      "rank": 1,
      "outOf": 12,
      "corpusMinimum": 0.1125,
      "corpusMedian": 0.2440,
      "corpusMaximum": 0.4010
    }
  ]
}
```

Everything there is a number computed from measured values across the
operator's own completed results. `percentile`, `rank`, and `outOf` are
citable, so "the lowest of your last 12 clips" resolves to `rank: 1` and
`outOf: 12` and passes the numeric-copy rule.

### Fail-closed rules

- **A small corpus produces no comparison.** Below a documented minimum the
  whole `comparative` block is an absent marker with a reason, not a percentile
  over four clips. The repo already sets this precedent with
  `MINIMUM_TREND_REFERENCE_SIZE = 20` in `src/forecast/contract.js`; the
  proposed default is the same number, and the reason string states it.
- **A metric missing from any corpus clip is excluded**, not imputed, and the
  exclusion is visible in `outOf`.
- **Only metric paths that resolve to a number in every counted clip appear.**
  This reuses the citation resolver rather than adding a second path lookup —
  the same reuse the experiment tracker already makes.
- **The corpus is declared, not implied.** `clipCount` and the date range travel
  with the comparison, so "your last 12 clips" can never be written against a
  corpus of 5.

### The known gap: opening index

The example in the original brief — "percentile of this clip's opening index
among their last N clips" — **cannot be built as stated today.** The opening,
pacing, continuity, and ending indices are computed in the browser by
`src/analysis.js` and never reach the backend; the Phase 0 audit recorded this,
and it is why they are absent from the bundle.

Two honest options, in preference order:

1. **Compare backend-measured descriptors instead** (measured audio, V-JEPA
   summary descriptors, AST label scores). Available now, no new plumbing, and
   the resulting sentences are about signals the server actually measured.
2. **Submit the browser indices as declared job evidence** — the frontend posts
   its measured indices with the job, they are stored in the result under an
   explicitly browser-sourced key, and comparisons over them say so. This is
   more work and adds a client-trusted input, which needs its own contract
   before it is worth doing.

The proposal is to ship option 1 and treat option 2 as a separate decision.

### Multi-creator scope

"Per-creator" is aspirational on a single-operator local server: the corpus is
"every completed forecast result in this job store". That is correct for the
current deployment and wrong the moment two people share one instance. The
corpus definition must therefore be a named, replaceable strategy from day one,
and the docs must say that a shared deployment needs authentication before this
feature means what its name says.

### How it stays inside the five insight invariants

| Invariant | How this addition satisfies it |
| --- | --- |
| 1. Evidence in, language out | Comparative fields are computed from bundles the backend already assembled. No new measurement, no media access, no model. |
| 2. Every claim is cited | The fields live in the `measured:` lane, so a comparative sentence cites `measured:/comparative/metrics/0/rank` like anything else, and an uncited one is `missing_citation`. |
| 3. No numeral the evidence does not contain | `percentile`, `rank`, and `outOf` are in the bundle, so the numeric-copy rule resolves them. A model that guesses "your worst opening in 20 clips" against `outOf: 12` is rejected. |
| 4. Hypotheses labeled, outcomes forbidden | **The model never computes a comparison.** It restates one that code computed. The claim vocabulary is unchanged, so "your best-performing clips" is still a `claim_boundary_violation` — being *relatively quiet* is a measurement; being *better* is not. |
| 5. The layer is descriptive | A percentile among the creator's own measured clips is a measurement about measurements. It is not a prediction and it is not an outcome. |

### The line that must not blur

With both features present, the tempting next step is the forbidden one:
comparing this clip against the creator's **best-performing** past clips. That
would make the insight lane an outcome model with no calibration, no gate, and
no evaluation — precisely what
[SCIENTIFIC_LIMITS.md](SCIENTIFIC_LIMITS.md) and the empty
`APPROVED_TARGET_CONTRACTS` exist to prevent.

The design's answer is not discipline. It is that the retrieval code physically
cannot read the outcome store, and a test proves it.

---

## Schema and configuration changes

| Change | Kind |
| --- | --- |
| `creator-outcome-set/1` | New document schema |
| `insight-evidence-bundle/2` | Bump: `measured.comparative` added to a closed lane |
| `POST /api/insight/v1/outcomes` | New route: import one CSV |
| `GET /api/insight/v1/outcomes` | New route: list imported sets |
| `DELETE /api/insight/v1/outcomes/{id}` | New route: delete one set |
| `INSIGHT_COMPARATIVE_MINIMUM_CLIPS` | New setting, default 20 |
| `INSIGHT_COMPARATIVE_WINDOW` | New setting: how many recent clips form the corpus |
| `INSIGHT_OUTCOME_RETENTION_DAYS` | New setting, default unset (keep until deleted) |

Bumping the bundle version changes `inputEvidenceHash` for every clip, so
existing cache entries miss and regenerate once. Already-published artifacts are
unaffected: each one persists the bundle it was generated from.

The `hook-doctor.v1` prompt template gains one paragraph describing the
`comparative` block, which moves `promptHash` and therefore the cache key. That
is the intended behavior — a model reading a new kind of evidence should not
serve text generated before that evidence existed. The template version becomes
`hook-doctor.v2`.

---

## What I would want decided before writing code

1. **Is comparison against the creator's own past clips wanted at all?** It is
   the feature most likely to be read as a verdict, whatever the caption says.
2. **Option 1 or option 2 for the opening index** (backend descriptors now, or
   client-submitted browser indices later).
3. **The minimum corpus size.** 20 mirrors the existing trend precedent, but a
   creator with 20 clips is not a common case; a smaller number buys usefulness
   with noise.
4. **Whether outcome ingestion should exist before there is any plan to train a
   head.** Storing performance data that nothing consumes is a real privacy cost
   for a hypothetical future benefit. It may be right to defer feature 1 and
   ship feature 2 alone.

My recommendation: **build feature 2, defer feature 1.** Retrieval is useful
immediately, stays inside the invariants without any new structural guard, and
carries no new sensitive data. Outcome ingestion creates the repository's most
sensitive store in exchange for a training set that is, by construction, too
biased to train the head it is meant for.
