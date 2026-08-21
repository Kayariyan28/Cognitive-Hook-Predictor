# Scientific and product limits

SignalFrame is built to distinguish a functioning model from a validated
claim. This document defines what each output supports, what it does not
support, and which additional evidence would be required to make a stronger
claim.

## Output taxonomy

| Output | Source | What it supports | What it does not establish |
| --- | --- | --- | --- |
| Modeled Engagement | Browser-measured opening, continuity, pacing, ending, visual, and audio signals | A transparent directional content index | Watch time, retention, views, engagement rate, or probability |
| Virality Outlook | Declared weighted local content features with missing-input renormalization | A comparison heuristic inside this tool | Viral reach, platform distribution, expected views, or causal uplift |
| V-JEPA 2.1 evidence | Learned visual representations over deterministic windows | Descriptive visual-temporal similarity/change features | Attention, story quality, comprehension, retention, or virality |
| NanoLLaVA evidence | Six sampled still frames | Literal scene/action/framing descriptions when schema-valid | Full-video semantics, OCR, soundtrack understanding, audience behavior |
| AST evidence | AudioSet classifier over decoded windows | Candidate sound-event labels | Transcript, music appeal, sentiment, quality, or audience response |
| Measured audio | PCM and STFT calculations | Signal facts such as energy, silence, centroid, flatness, and flux | Learned semantics or performance predictions |
| Spoken transcript | Pinned mlx-whisper over the decoded 16 kHz mono track | Which words were spoken and when, when speech is present and recognizable | Speaker identity, sentiment, tone, meaning, delivery quality, or audience response |
| On-screen text | Apple Vision or Tesseract over the six sampled 384 px keyframes | Which glyphs were recognized on those six frames, and roughly where | Text between the sampled frames, reading order, emphasis, meaning, or whether anyone read it |
| Insight lane | A pinned language model over validated evidence, checked by a deterministic validator | Descriptive restatement of cited evidence and untested editing hypotheses | Any outcome, any mental state, any number the evidence does not contain |
| Behavioral head | Separately approved and calibrated target model | Its exact declared platform/population/horizon probability | Any other outcome, platform, population, or time horizon |
| TRIBE v2 tensor | Released average-subject brain-response model | Model-predicted cortical BOLD on `fsaverage5` | Measured viewer activity, an individual brain, attention, emotion, memory, engagement, or virality |

Behavioral heads are unavailable in this release because no bundled artifact
has passed the code-owned production approval contract. A successful evidence
job does not change that fact.

## Technical validity is not outcome validity

The project validates that:

- the requested official code/model revision and weight digest are present;
- preprocessing follows the declared contract;
- model input and output shapes are exact and finite;
- source-video, result, tensor, and thumbnail hashes agree;
- timing and hemisphere/vertex ordering are preserved; and
- an unavailable or malformed component fails closed.

These checks establish artifact and execution integrity. They do **not** prove
that a model predicts creator outcomes accurately. Outcome validity requires
representative post-level ground truth, a fixed target definition, independent
evaluation, calibration, drift monitoring, and prospective validation.

## TRIBE v2 interpretation

TRIBE v2 predicts an average subject's fMRI BOLD response to naturalistic
stimuli. In this project the tensor contains 20,484 `fsaverage5` cortical
vertices per model interval, ordered left hemisphere then right hemisphere.

### The values are predictions, not measurements

No person is scanned when a creator uploads a clip. The displayed values are
outputs of a trained encoding model in its training-target units. They should
be described as **predicted cortical BOLD**, not “live brain activity,” “neural
attention,” or “what viewers feel.”

### Average-subject output is not an individual

Population-average modeling cannot establish how a particular viewer,
demographic, clinical group, culture, or audience segment will respond.
Individual anatomy, history, language, context, fatigue, expectation, and
measurement noise are not represented by a single average-subject surface.

### BOLD is indirect and delayed

fMRI BOLD is a hemodynamic signal, not direct millisecond-scale neural firing.
The released model uses a five-second hemodynamic offset, which SignalFrame
records in the manifest. One-second output intervals do not imply one-second
causal or cognitive precision.

### Anatomical parcels are not mental-state labels

The Destrieux atlas supplies anatomical names. A higher modeled value in a
parcel does not by itself mean attention, memory, emotion, language
comprehension, purchase intent, or preference. Reverse inference from an
anatomical region to a mental state is not supported here.

### Magnitude is not “better”

Higher predicted magnitude, broader spatial distribution, continuity, or
pattern change is not inherently desirable. The creator report presents these
as descriptive properties and frames editing ideas as hypotheses to test with
real audience outcomes.

### Vision-only is an ablation

The Apple-silicon TRIBE profile disables audio and text and reports
`inferenceMode: vision-only`. It remains genuine output from the official
visual extractor plus cortical encoder, but it is not the full published
trimodal path. Comparisons must use the same inference mode and provenance.

## Transcript interpretation

A transcript is a recognition result, not an understanding of a clip. It
supports statements about which words were spoken and when they were spoken,
and only when speech is present, audible, and in a language the pinned model
handles. Word-error rate varies with accent, dialect, code-switching,
background music, overlapping speech, and recording quality, and the branch
does not report a confidence a creator could calibrate against.

It does not establish who spoke, how many people spoke, how anyone felt, what
was meant, whether delivery was good, or how an audience will respond. The
branch records no speaker identity and no sentiment, by contract and by test.
A missing or unrecognizable speech track leaves the branch unavailable; it is
never filled in from captions, filenames, or creator-declared context.

## On-screen-text interpretation

Text recognition runs on six deterministically sampled frames, not on the
whole clip. Text that appears only between those samples is not in the
evidence, and its absence is not evidence of absence. Apple Vision results
change with the operating-system version, so the branch records the engine and
the macOS version it ran under; a result produced under one version is not
strictly comparable to one produced under another.

The branch reports glyphs, confidences, and bounding boxes. It does not report
what the text means, which text a viewer would read first, whether the wording
is effective, or whether anyone read it at all.

## Frame and interval interpretation

The Frames tab ranks authoritative TRIBE intervals by model-predicted cortical
magnitude or adjacent-pattern change. Its thumbnail is a real source-video
frame decoded at or after the interval's start.

It does not identify an exact “attention frame,” attention span, retention
drop, emotional peak, or interaction event. The ranking is useful for locating
moments whose **predicted cortical pattern** differs under the model. Any
creator hypothesis from that ranking needs a real platform experiment.

If a source frame cannot be decoded for an authoritative interval, the result
marks that thumbnail unavailable. It does not synthesize a replacement.

## A/B Lab interpretation

The A/B Lab compares two uploaded variants' independently validated
descriptive model outputs. Seeking a moment synchronizes the relevant video or
timeline position.

This is not a randomized controlled A/B test. It has no platform impressions,
audience assignment, exposure balance, outcome denominator, or statistical
power calculation. A larger descriptive difference does not mean one cut will
win. Use it to form editing hypotheses, then test those hypotheses on the
target platform with a predeclared metric.

## Behavioral prediction requirements

A useful creator forecast must define each target independently. For example:

- five-second retention among eligible starts;
- completion among starts for a defined duration bucket;
- replay among unique exposed viewers;
- share among eligible views within a fixed horizon;
- expected first-pass watch fraction; or
- a predeclared virality event on one platform within seven days.

Before SignalFrame can display one of these as a probability, the corresponding
head must pin:

1. the event, denominator, platform, horizon, locale, population, and account
   segment;
2. the exact encoder features and missing-modality policy;
3. creator-disjoint and chronological train/validation/test splits;
4. near-duplicate grouping to prevent leakage;
5. held-out Brier score, log loss, expected calibration error, calibration
   slope/intercept, and interval coverage;
6. training-data cutoff, evaluation time, expiry, and drift thresholds; and
7. an approved artifact and feature-contract digest.

Training on public viral clips alone would create severe selection and
survivorship bias. The dataset needs unposted or prospectively scored clips,
their actual exposures, non-viral outcomes, account/distribution context, and
legally usable platform analytics.

## Confidence and uncertainty

The browser's confidence/coverage language means **input coverage**—for
example, whether visual and audio measurements were available. It is not
statistical confidence in audience behavior.

A future behavioral head must report uncertainty tied to its calibrated target
and evaluation domain. “High confidence” cannot be inferred from having more
encoders, a larger neural network, or a more colorful cortical map.

## Context, trends, and distribution

Virality depends on more than clip content. Account history, audience fit,
platform recommendation systems, timing, competition, paid distribution,
caption/topic choices, and live trends can dominate results.

Creator-entered context is treated as declared fact, not verified history.
Trend or competitor evidence must be timestamped, source-linked, and captured
before the intended post time. Missing external context stays missing. It is
not backfilled from the clip or from TRIBE.

## Domain and drift risks

Performance can change across:

- platforms and recommendation policies;
- languages, regions, cultures, and accessibility patterns;
- genres, editing conventions, codecs, aspect ratios, and clip duration;
- accounts with different audience and distribution histories; and
- time as content trends and platform behavior shift.

Even a well-calibrated head must be monitored and disabled when the input is
out of domain or its evaluation evidence expires.

## Medical and high-stakes use

SignalFrame is not a medical device, diagnostic system, neuroimaging analysis
tool for individual patients, lie detector, cognitive assessment, or mental
state classifier. Do not use it for health, employment, insurance, education,
law enforcement, or any decision about a person.

## Privacy

Uploaded videos may contain faces, voices, locations, private text, or
copyrighted material. A localhost deployment is not automatically safe merely
because it is local:

- TRIBE tensors and thumbnails persist by default for verified replay/cache;
- forecast JSON persists until the operator removes it;
- model caches may preserve metadata and consume substantial disk;
- a network-exposed service needs authentication, authorization, encryption,
  tenant isolation, quotas, auditability, and retention/deletion controls.

Only analyze media you are authorized to process.

## Licensing

TRIBE v2 code and weights are CC BY-NC 4.0. Commercial use of the TRIBE path
requires appropriate permission or a separately licensed replacement. Other
models keep their own upstream licenses and dataset terms. A license attached
to model code does not necessarily grant rights to every training dataset,
input video, output use, trademark, or deployment scenario.

See the [README](../README.md#third-party-models-and-licenses) for upstream
links and [backend/README.md](../backend/README.md) for the exact runtime
contract.
