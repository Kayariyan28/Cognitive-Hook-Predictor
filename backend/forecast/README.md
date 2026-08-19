# Creator forecast capability registry and evidence jobs

This package exposes an executable-runtime capability document plus a bounded,
private job API:

```text
GET /api/forecast/v1/status
POST /api/forecast/v1/jobs
GET /api/forecast/v1/jobs/{job_id}
GET /api/forecast/v1/results/{result_id}
```

The response schema is `creator-forecast-capabilities/1`. It reports the live
job route, locally configured evidence adapters, deliberately deferred remote or
large-model readiness, and target-specific executable-head readiness. It never
contains a clip score; only a validated completed job result can contain one.
The job endpoints are the evidence execution surface. They do not alter
the TRIBE router, invoke TRIBE, or treat a cortical tensor as a behavioral
forecast.

Every job and result response uses:

```http
Cache-Control: private, no-store
Pragma: no-cache
X-Content-Type-Options: nosniff
```

## Submit and poll a job

Submit multipart form data with a required `video` field and optional `context`
field. `context` is a strict JSON object, not a client-declared model output.
Duplicate keys, non-finite JSON constants, non-object roots, and oversized
context are rejected.

```bash
curl -i -X POST http://127.0.0.1:8000/api/forecast/v1/jobs \
  -H 'Idempotency-Key: 12345678-1234-4234-9234-123456789abc' \
  -F 'video=@clip.mp4;type=video/mp4' \
  -F 'context={"platform":"youtube-shorts","topic":"travel"}'
```

An accepted upload returns `202`, a `Location` header, and a
`creator-forecast-job/1` document. Poll the returned job URL. The durable state
machine is:

```text
queued -> probing -> running -> complete
                            \-> failed
```

The job reports `stage`, `heartbeatAt`, and real wall-clock `elapsedSeconds`.
It intentionally has no percentage field: model stages do not have an honest
universal percent-complete mapping. `GET /results/{result_id}` returns `409`
until a result is atomically published, and also returns `409` when its job
failed.

`Idempotency-Key` accepts a client-generated UUID. Its normalized value is the
durable job ID, so an ambiguous upload retry must reuse the same key. Replays
return the already-published job and cannot enqueue a second inference. A new
attempt after an authoritative failed job must use a new key.

Uploads are hashed with streaming SHA-256 while being written, and the byte
limit is enforced during that stream. The worker then invokes `ffprobe` without
a shell and requires a decodable video stream with authoritative duration from
10 through 60 seconds inclusive. An out-of-range or undecodable upload becomes
a durable failed job with a safe error code; it never reaches an orchestrator.
Source video bytes are discarded after completion or failure.

Result JSON is written to a private staging directory, fsynced, and renamed
into its final result directory as one atomic publish. On process startup,
persisted `queued`, `probing`, or `running` jobs are marked `failed` with
`worker_interrupted`; they are never mislabeled complete and their abandoned
uploads are removed.

## Orchestrator injection contract

`ForecastJobService` accepts either a callable or an object exposing `run`.
The runner receives exactly:

```python
(
    video_path,       # pathlib.Path; present only while this job is running
    input_sha256,     # streamed upload SHA-256
    media_probe,      # strict dict with ffprobe duration and upload metadata
    context,          # strict JSON object or None
    stage_callback,   # report a real kebab-case stage; no fake percentage
)
```

Async and synchronous runners are supported. A synchronous runner executes in
a worker thread and may call `stage_callback("model-stage")`; an async runner
may `await stage_callback("model-stage")`.

The returned value must be a strict JSON-serializable object containing exactly
`evidence`, `behavioralHeads`, and `boundaries`. Every supported behavioral head
must be present. An unavailable head must use `value: null` and
`validation: "not-calibrated"`. An available value is accepted only as a finite
0–1 probability with `validation: "production-calibrated"`. Malformed,
partial, or non-finite runner output fails the job without publishing a result.

The default orchestrator is deliberately modest. It returns hash-verified
upload metadata, authoritative ffprobe duration, the optional creator context,
and explicit provider availability. All behavioral heads are `unavailable`.
It does not invent hook, retention, completion, rewatch, share, engagement, or
virality values.

## Resource bounds

The defaults can be changed explicitly:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `FORECAST_JOB_DIR` | `backend/.runtime/forecast` | Durable jobs/results root |
| `FORECAST_MAX_UPLOAD_BYTES` | `268435456` | Streamed per-video limit |
| `FORECAST_MAX_CONTEXT_BYTES` | `32768` | UTF-8 JSON context limit |
| `FORECAST_MAX_RESULT_BYTES` | `4194304` | Strict JSON orchestrator-result limit |
| `FORECAST_MAX_ACTIVE_JOBS` | `4` | Combined queued/probing/running admission bound |
| `FORECAST_MAX_WORKERS` | `1` | Concurrent evidence runners |
| `FORECAST_FFPROBE_BINARY` | `ffprobe` | ffprobe executable/path |
| `FORECAST_PROBE_TIMEOUT_SECONDS` | `30` | Per-file probe timeout |
| `FORECAST_HEARTBEAT_SECONDS` | `5` | Durable running heartbeat cadence |
| `FORECAST_JOB_TIMEOUT_SECONDS` | `600` | Maximum orchestrator runtime |

Every numeric bound must be positive, and workers cannot exceed the active-job
bound. Invalid configuration stops router construction instead of silently
falling back.

## Available-now profile

The built-in profile reports only functionality that exists in this project:

- browser-decoded metadata;
- browser-local pixel/frame-change measurements;
- browser-local waveform measurements when an audio track can be decoded; and
- the separately configured pinned V-JEPA2 to TRIBE v2 cortical pipeline.

The browser providers have descriptive roles and `forecastContribution: false`.
Their implementation digest is calculated from the current `src/analysis.js`.
They are not substitutes for V-JEPA 2.1, VideoLLaMA 2.1 AV, a learned audio
model, or a behavioral calibration head.

### Job-only available-model evidence

Forecast jobs also probe three explicitly separated local evidence adapters.
The status branch ledger names their configured/readiness state without
mistaking a fallback for the advertised VideoLLaMA or learned-audio branches:

- `semanticModel` is the pinned
  `mlx-community/nanoLLaVA-1.5-8bit` snapshot at revision
  `6e2bc13b87ab178668313552b9d69026af7d556f`, with
  `model.safetensors` SHA-256
  `f0a1d1517ad9e6e810bc6ac99956643c66e4b87a2f82bd5d1b5cb0966e5c5476`
  and declared Apache-2.0 license. It uses six deterministic, center-sampled
  384×384 ffmpeg PNG frames. Every model response must pass an exact JSON
  contract before it becomes an observation. This is a still-keyframe
  semantic fallback, **not VideoLLaMA**, and it does not consume audio or
  model continuous motion.
- `audioModel` can use the pinned
  `MIT/ast-finetuned-audioset-10-10-0.4593` snapshot at revision
  `f826b80d28226b62986cc218e5cec390b1096902`, with safetensors SHA-256
  `ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d`
  and declared BSD-3-Clause license. AST receives deterministic 16 kHz mono
  PCM windows and returns uncalibrated AudioSet sound-label evidence. It is not a
  transcript, music-quality model, or audience model.
- `measuredAudio` always stays separate from `audioModel`. It reports ffmpeg
  PCM/STFT facts such as RMS, silence-window fraction, spectral centroid,
  flatness, spectral flux, and relative energy peaks. It uses no learned
  model and makes no speech, music, attention, or behavioral inference.

All three adapters recompute the uploaded video's SHA-256 before decoding.
Pinned model weights are also hashed before model loading. A missing snapshot,
missing optional runtime, hash mismatch, invalid model JSON, or invalid scalar
feature fails only that branch. Validated scalar evidence may be consumed by a
release-approved calibrated head, but the adapter itself always reports
`behavioralOutcome: false`: it is never an audience-outcome model. Result
provenance is usage-aware. An unused adapter keeps
`forecastContribution: false`; when one of its exact declared features enters
an available calibrated head, it reports `forecastContribution: true` plus a
`forecastUse` ledger naming the head, feature-contract SHA-256, and consumed
feature names.
The local wrappers publish the SHA-1 content identity of the exact imported
source in protocol v1's 40-character `codeRevision` field; model weights and
preprocessing contracts retain full SHA-256 pins.

No download is performed by this package. Optional locations can be supplied
with `FORECAST_NANOLLAVA_SNAPSHOT`, `FORECAST_AST_SNAPSHOT`,
`FORECAST_NANOLLAVA_FFMPEG_BINARY`, `FORECAST_AUDIO_FFMPEG_BINARY`, and
`FORECAST_AUDIO_FFPROBE_BINARY`.
NanoLLaVA additionally requires `mlx_vlm`; AST requires Torch and Transformers.

Keep MLX-VLM outside the TRIBE/backend Python environment. The optional
`FORECAST_NANOLLAVA_PYTHON` value must be an absolute path to an executable in
a separately managed virtual environment. When it is configured, the backend
does not import `mlx_vlm` in the TRIBE process and does not fall back to that
process's packages. It extracts the same six deterministic PNGs, then starts
exactly one isolated child for the clip. That child loads the pinned local
snapshot once and describes all six frames. The request and response are
bounded strict JSON over stdio; frame bytes, model weights, response order, and
response frame hashes are revalidated. Hugging Face offline modes are forced,
and no model URL or download operation is exposed to the child.
MLX-VLM's bundled `llguidance` processor enforces a bounded JSON Schema during
generation; extra keys and malformed field types cannot become observations.
The child sets `trust_remote_code=False` and uses MLX-VLM's built-in
`llava-qwen2` mapping, so the snapshot's custom Python files are never executed.
`visibleText` is explicitly JSON `null`: NanoLLaVA is not treated as an OCR
provider, and unavailable text evidence is not mislabeled as observed absence.

One isolated Apple-silicon setup, with the MLX-VLM package version pinned to
the current official [v0.6.8 release](https://github.com/Blaizzy/mlx-vlm/releases/tag/v0.6.8),
is:

```bash
python3.11 -m venv /absolute/path/to/nanollava-mlx-vlm-0.6.8
/absolute/path/to/nanollava-mlx-vlm-0.6.8/bin/python -m pip install 'mlx-vlm==0.6.8'

export FORECAST_NANOLLAVA_PYTHON=/absolute/path/to/nanollava-mlx-vlm-0.6.8/bin/python
export FORECAST_NANOLLAVA_SNAPSHOT=/absolute/path/to/models--mlx-community--nanoLLaVA-1.5-8bit/snapshots/6e2bc13b87ab178668313552b9d69026af7d556f
export FORECAST_NANOLLAVA_TIMEOUT_SECONDS=300
./backend/run-local-mac.sh
```

The snapshot must already contain `model.safetensors`, `config.json`, and
`tokenizer.json`; its weights must match the SHA-256 stated above. The backend
and child never download a missing file. `FORECAST_NANOLLAVA_TIMEOUT_SECONDS`
accepts a finite value from 10 through 1800 seconds. An absent, relative, or
non-executable isolated-Python path, a timeout, a child error, or malformed
output leaves only this evidence branch unavailable; it never creates a score.

TRIBE readiness is read from the live `TribeRuntime.status()` result on every
capability request. Its role is `cortical-only`; its V-JEPA2 extractor and
cortical result both have `forecastContribution: false`. The status document
does not infer readiness merely because environment variables are present.

The creator-forecast duration range is 10 to 60 seconds, inclusive. The job
worker enforces it from server-side ffprobe output; client duration claims are
ignored. The capability endpoint truthfully reports that the upload route is
available, but that fact never unlocks a behavioral value in the browser.

## Optional provider registry

Leave both variables unset to report every optional provider and calibration
head as `not-configured`:

```dotenv
# FORECAST_REGISTRY_PATH=/absolute/path/to/creator-forecast-registry.json
# FORECAST_REGISTRY_JSON={"schemaVersion":"creator-forecast-registry/1","branches":{},"heads":{}}
```

Only one variable may be set. Invalid JSON, duplicate keys, unknown fields,
unknown branch/head names, mutable revisions, missing files, and hash
mismatches stop application construction. Configured artifact paths are never
returned by the public API.

The registry root has exactly these fields:

```json
{
  "schemaVersion": "creator-forecast-registry/1",
  "branches": {},
  "heads": {}
}
```

Supported optional branch keys and their exact adapter IDs are:

| Branch | Required `adapterId` |
| --- | --- |
| `vjepa21` | `vjepa21-pytorch` |
| `videoLlama21Av` | `videollama21-av-pytorch` |
| `audioModel` | `speech-music-model` |
| `account` | `account-history-v1` |
| `trends` | `web-trend-evidence-card-v1` |
| `competitors` | `competitor-evidence-card-v1` |

Each branch entry must contain exactly:

```json
{
  "id": "publisher/model-or-provider",
  "revision": "full-40-character-commit-sha",
  "artifactPath": "/absolute/path/to/installed-artifact",
  "artifactSha256": "actual-64-character-file-sha256",
  "license": "declared-license",
  "adapterId": "the-exact-branch-adapter-id"
}
```

A successfully verified entry is reported as
`registered-artifact-runtime-unavailable`. Registration proves artifact
identity and integrity only. It does not enable execution, substitute another
provider, or contribute to a forecast.

Supported target-specific head keys are `hookStrength`,
`retention5Second`, `completionProbability`, `rewatchPotential`,
`sharePotential`, `predictedEngagement`, and `viralityPotential`. A head must
pin its model artifact and a separate calibration artifact and declare:

- `outputKind: "calibrated-probability"`;
- immutable target and denominator IDs, exact definitions, and the SHA-256 of
  each UTF-8 definition;
- platform, domain, locale, and evaluation population;
- a typed horizon (`seconds-after-start`, `clip-end`, `hours-after-post`, or
  `days-after-post`);
- the SHA-256 of an exact feature-to-evidence-source contract;
- a timezone-aware training-data cutoff; and
- held-out calibration method, evaluation-manifest ID, dataset, evaluation and
  expiry timestamps, sample count, Brier score, and expected calibration
  error.

Registry JSON is not an approval authority. Each target is defined in code,
and a registered head can execute only if every model, target, context,
feature-contract, calibration, and fresh evaluation-manifest pin exactly
matches a release-owned `ApprovedTargetContract`. The production approval
table is intentionally empty until a reviewed artifact and evaluation are
shipped. Therefore arbitrary or self-declared JSON cannot unlock a score, and
a registered but unapproved head fails application startup instead of being
quietly treated as calibrated.

When an approved head does return a probability, its result publishes the
complete `targetContract` and `featureContract`. The evidence envelope derives
`usedByHeads` from that exact feature contract. Creator context is separately
marked as eligibility context, and TRIBE is forbidden as a feature source and
always remains outside the forecast.

## Local V-JEPA 2.1 Base feature worker

`backend.forecast.workers.vjepa21` is a code-only worker implementation. It
does not download a repository, checkpoint, or Python package. Supply:

```dotenv
FORECAST_VJEPA21_SOURCE_PATH=/absolute/path/to/clean/vjepa2-checkout
FORECAST_VJEPA21_CHECKPOINT_PATH=/absolute/path/to/vjepa2_1_vitb_dist_vitG_384.pt
FORECAST_VJEPA21_CHECKPOINT_SHA256=<sha256-computed-after-download>
FORECAST_VJEPA21_DEVICE=cpu
# FORECAST_VJEPA21_FFMPEG_BINARY=ffmpeg
# FORECAST_VJEPA21_FFMPEG_TIMEOUT_SECONDS=120
```

For the accuracy-first Apple-silicon route, keep `FORECAST_VJEPA21_DEVICE=cpu`.
On the configured M4 Max host with PyTorch 2.6.0, the exact pinned Base encoder
completed a full float32 `1 x 3 x 64 x 384 x 384` forward in about 10.6 seconds
and returned the required `1 x 18,432 x 768` tensor. The same full-token
attention path on MPS float32 showed severe, nondeterministic latency, including
an operation that remained blocked in `torch.mps.synchronize()` for over a
minute. This is a backend/runtime pathology rather than video decoding or a
smaller-model substitution. Each planned 16-second window requires one encoder
forward, so longer clips require additional forwards.

The worker deliberately does not switch precision or devices behind the
operator's back. Its current inference precision is float32 and is reported in
the status document. MPS float16 autocast can be much faster on this host, but
it changes numerical precision and is not enabled by this worker. Do not select
`mps` for this pinned local runtime unless that exact PyTorch/macOS combination
has been separately benchmarked and its output tolerance accepted.

The source path must be a Git checkout at exactly
`204698b45b3712590f06245fbfba32d3be539812`. Executable/source paths must be
clean. On a case-insensitive macOS volume, the verifier tolerates only the
upstream tree's exact nine known `vitG`/`vitg` YAML collisions, and only after
each uppercase worktree file is byte-matched against its tracked lowercase Git
blob. Any other tracked, staged, or untracked change fails readiness. Keep the
checkpoint outside the checkout so it does not make the verified tree dirty.
The worker verifies the configured checkpoint hash before deserialization,
calls the official local
`vjepa2_1_vit_base_384` entry point with `pretrained=False`, and strict-loads
the cleaned `ema_encoder` state. It never calls the broken pretrained URL in
that upstream commit. Check readiness with:

```bash
python -m backend.forecast.workers.vjepa21 status
```

The extraction CLI consumes the strict `creator-forecast-worker-request/1`
JSON document used by `forecast/adapters.py`. It recomputes the video hash and
requires `requestId`, `inputSha256`, and the uploaded bytes to agree before it
loads the model:

```bash
python -m backend.forecast.workers.vjepa21 extract \
  --video /absolute/path/to/clip.mp4 \
  --request /absolute/path/to/request.json
```

The preprocessing contract digest is
`d505e3350d91fc2b8679ff04b792e94e115dc4f6c825566643ab280c8e9000da`
with ID `vjepa2.1-64f4fps-384-ffmpeg/1`. Each model window contains exactly 64
frames sampled by ffmpeg at 4 fps. Full 16-second windows are placed
deterministically to cover the clip's first and last frames; a source interval
shorter than 16 seconds repeats its final decoded frame. Frames are resized
bilinearly to a 438-pixel short side, center-cropped to 384 square, converted
to RGB float32 in BCTHW layout, and normalized with the official ImageNet
mean/std values.

Output is descriptive learned-visual evidence only: the fixed 768-dimensional
mean-pooled representation (`vjepa2_1.embedding_000` through `_767`), embedding
norms, temporal consistency/change, interval observations, and preprocessing
warnings. `behavioralOutcome` is always `false`. These values are not
attention measurements, retention probabilities, engagement predictions, or
virality scores. The CLI is not the persistent `/v1/status` and `/v1/extract`
HTTP deployment by itself.

The forecast job orchestrator now provides that separately tested in-process
adapter. It is enabled only when
`FORECAST_VJEPA21_SOURCE_PATH`, `FORECAST_VJEPA21_CHECKPOINT_PATH`, and
`FORECAST_VJEPA21_CHECKPOINT_SHA256` are all non-empty. Setting only part of
that required trio stops backend construction; none of the three means the
local branch remains unavailable. A configured remote `vjepa21` worker keeps
execution precedence, although a partial local configuration is still rejected
as an operator error. Model loading stays lazy until job readiness is probed.
Before execution the adapter recomputes the uploaded video's SHA-256, validates
the worker's immutable status provenance, builds the strict request document,
and converts the self-validated wire result to the immutable `BranchOutput`.
The capability-v1 ledger remains unchanged, and calibrated behavioral heads
remain a separate evidence consumer.

## Why registration is fail-closed

Model names and hashes supplied by a registry are not sufficient validation.
Every optional entry requires a full commit revision and the SHA-256 of a real
local artifact. Every calibration head separately hashes its model and
calibration evidence, binds semantic definitions by digest, and must match a
code-owned target plus an exact release approval. Evaluation evidence must be
current at startup. Missing evidence is never filled with the browser formula,
TRIBE output, or another provider, and remaining weights are never silently
renormalized.
