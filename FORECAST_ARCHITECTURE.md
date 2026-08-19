# Creator forecast architecture

This application uses late fusion. A video encoder, language model, audio model,
TRIBE v2, and web retrieval are not interchangeable and do not become a
performance predictor merely because their features are available.

```text
uploaded clip
  |-- local visual/audio measurements -------- descriptive clip evidence
  |-- pinned V-JEPA 2.1 Base ------------------ visual-temporal features
  |-- NanoLLaVA sampled keyframes ------------- still-frame semantics
  |-- pinned AST AudioSet --------------------- sound-label evidence
  `-- measured PCM/STFT ----------------------- acoustic signal evidence

creator context
  |-- caption/topic/genre/platform/schedule --- supplied facts
  |-- account history ------------------------- pre-post factual snapshot
  `-- trends/competitors ---------------------- timestamped retrieval snapshot

feature bundle + context snapshot
  `-- one separately trained and calibrated head per behavioral target

TRIBE v2
  `-- independent predicted cortical-BOLD report; forecastContribution=false
```

## Available-now profile

Model weights are not bundled. Implemented adapters become ready only when
their exact hash-verified artifacts and runtimes are configured:

| Branch | Current source | Status | Allowed use |
| --- | --- | --- | --- |
| Visual content | Browser-decoded sampled pixels, motion, cuts, luminance, contrast, color, edges | Available | Measured features and directional content-signal indices |
| Audio content | Browser Web Audio RMS, dynamics, silence, onset cadence, speech-like modulation | Available when the codec decodes | Measured audio descriptors and directional content-signal indices |
| Semantics | Creator-supplied caption, topic, and genre | Available when supplied | Factual context only |
| Account/distribution | Creator-supplied pre-post account context | Available when supplied | Evidence only until an account-aware head is calibrated |
| V-JEPA 2.1 Base | Executable pinned local adapter | Artifact-gated | Visual-temporal representations only |
| NanoLLaVA | Executable isolated MLX adapter over six frames | Artifact-gated | Still-keyframe semantics; never labeled VideoLLaMA |
| AST AudioSet | Executable pinned local adapter | Artifact-gated | Sound-label evidence only |
| Measured audio | Server ffmpeg PCM/STFT measurements | Available with decodable audio | Signal facts only |
| VideoLLaMA 2.1 AV | No executable adapter | Unavailable | Never substituted by NanoLLaVA |
| Live trends/competition | Optional timestamped provider adapter | Not configured by default | Context indices only after source and freshness validation |
| TRIBE v2 + its pinned V-JEPA 2 extractor | Existing local worker | Available separately | Predicted average-subject cortical BOLD; never a forecast feature |

An unavailable encoder may be replaced only by another provider that produces
the same declared evidence type and passes its immutable model, preprocessing,
and quality contract. Missing behavioral heads are never replaced by a
hand-weighted proxy.

## Metric availability

The following creator-facing metrics are independent targets: five-second
retention, completion, replay, sharing, expected first-pass watch, and a defined
seven-day virality event. Each value remains withheld until a matching
`performance-calibration/1` artifact supplies all of the following:

- an immutable feature-contract hash, encoder revisions, weight hashes, and
  calibrator hash;
- the exact target event, denominator, platform, horizon, population, locale,
  language, duration range, and account segment;
- creator-disjoint and chronological evaluation with near-duplicate grouping;
- locked-test calibration evidence (including Brier score, log loss, expected
  calibration error, calibration slope/intercept, and interval coverage);
- freshness, missing-modality, out-of-distribution, and deployment gates.

The UI may show available descriptive content indices while behavioral metrics
are withheld. Input coverage is never displayed as model confidence.

## Artifact identities

- `clipFeatureId`: source-video hash plus exact model and preprocessing pins.
- `contextSnapshotId`: creator inputs plus account and retrieval evidence hashes
  and their `asOf` times.
- `forecastId`: both identities plus the target-head and calibration revisions.

Changing trends or posting time invalidates only contextual forecasts. It does
not rewrite intrinsic clip features or any TRIBE result.

## Activation order

1. Keep the available-now evidence profile working and collect prospective
   creator feedback without presenting behavioral probabilities.
2. Configure the implemented immutable V-JEPA 2.1 Base worker. Do not replace
   the separate V-JEPA 2 checkpoint inside TRIBE.
3. Configure the implemented NanoLLaVA and AST adapters only with their exact
   license, weight, revision, and preprocessing pins. Add a continuous-video
   semantic provider only under its own honest identity.
4. Add a compliant trend/competitor retrieval provider that stores source URLs,
   hashes, query, platform, locale, and `capturedAt`/`asOf` timestamps.
5. Train one target head per platform outcome on legally usable analytics, then
   publish calibration artifacts only after the deployment gates pass.
6. Run prospective shadow validation and creator-disjoint A/B evaluation before
   enabling a numeric behavioral value in production.

## Hard invariants

- Forecast feature names reject `tribe`, `bold`, `cortex`, `parcel`, and brain
  descriptors.
- Removing, failing, or perturbing a TRIBE tensor must leave forecast metrics
  byte-for-byte identical.
- A missing input produces an explicit unavailable reason. Learned weights are
  never silently redistributed unless a separately evaluated fallback head is
  selected.
- No score is labeled retention, completion, replay, sharing, confidence, or
  virality unless its target and calibration provenance are available.
- Account and retrieval evidence must be strictly pre-post (`asOf` no later than
  the intended posting time).
