# Real TRIBE v2 inference backend

This service accepts a video, invokes Meta's official `TribeModel` pipeline, and
returns the resulting time-major `(T, 20,484)` cortical prediction. It has no
mock model, synthetic brain activity, browser-derived neural fallback, or
per-frame color normalization. If the model, credentials, output shape, vertex
order, or segment timing cannot be verified, the request fails.

## Important scientific and license boundary

TRIBE v2 predicts an average subject's fMRI BOLD response to naturalistic
stimuli. It does **not** contain a virality, views, sharing, or engagement head.
The API therefore returns `isViralityScore: false`; any product-level virality
estimate must be trained and validated separately and must never be presented as
a measured or predicted brain signal.

The official TRIBE v2 code and weights are licensed **CC BY-NC 4.0**. This
backend is suitable for non-commercial/research use under those terms. A
commercial product needs permission or a separately licensed model. Review the
[official license](https://github.com/facebookresearch/tribev2/blob/main/LICENSE)
before deploying.

## Runtime prerequisites

- Linux with an NVIDIA CUDA GPU is the supported practical target. The released
  config pins its text/audio/video feature extractors to CUDA, and the upstream
  transcription command uses float16.
- Python 3.11 is the safest supported interpreter for the current upstream
  dependency set.
- A large amount of disk space for the TRIBE checkpoint and the gated Llama
  3.2-3B, V-JEPA2, Wav2Vec-BERT, and WhisperX dependencies.
- `ffmpeg`, `git`, and `uvx` on `PATH`. The official event pipeline runs
  `uvx whisperx` for word-level timestamps.
- A Hugging Face account with access to `meta-llama/Llama-3.2-3B`, plus a read
  token in `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`).

For a local Apple-silicon research proof, this project also supports an
explicit `TRIBE_INFERENCE_MODE=vision-only` ablation. It runs the official
V-JEPA2 visual extractor and the released TRIBE cortical encoder on MPS, while
zero-filling the disabled audio and text branches exactly as the released
encoder does for absent modalities. The local configuration uses MPS FP16
autocast for only the V-JEPA2 forward pass; the pinned weights, preprocessing,
temporal sampling, layer/token aggregation, cortical model, and 20,484-vertex
output contract are unchanged. This remains genuine TRIBE output, but it is not
the published full trimodal path. The API and UI label it as video-only.

Install a CUDA build of PyTorch compatible with your host first, then install
the pinned backend dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the appropriate CUDA PyTorch 2.5/2.6 wheel for this machine first.
python -m pip install -r backend/requirements.txt
```

The requirements pin the official TRIBE code to Git commit
`af58661791a351a448a489042a28f6c37e1c14b7`. The model snapshot is independently
pinned at runtime via `TRIBE_MODEL_REVISION`; the service rejects `main`, tags,
and abbreviated SHAs. The current official model commit used in
`.env.example` is `f894e783020944dcd96e5568550afe2aa9743f9f`.
The video backbone is independently pinned to V-JEPA2 snapshot
`875c192b7b704b87d1e1d99345769632dd5f739a`; both the status response and
result manifest expose that revision.

## Configure and run

Load the example variables into your deployment environment without committing
the token:

```bash
set -a
source backend/.env.example
set +a
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

`TRIBE_PRELOAD_MODEL=true` downloads and loads the pinned model at startup. With
the default `false`, the first prediction request loads it. Either way,
`GET /api/tribe/v1/status` reports the honest state: `unconfigured`,
`not_loaded`, `loading`, `ready`, or `error`.

On the configured Apple-silicon machine, `backend/run-local-mac.sh` exports
`backend/.env.local` before Python starts (including the separate bytecode-cache
location) and launches the video-only MPS worker on `127.0.0.1:8000`. The
launcher prefers a persistent Python 3.11 runtime outside macOS File Provider
at `~/Library/Application Support/SignalFrame/py311`; this avoids intermittent
`dataless` placeholder timeouts seen when a virtual environment lives under a
synced Documents folder. Set `SIGNALFRAME_BACKEND_PYTHON` to an absolute
executable path to override it. A physical `backend/.venv/bin/python` remains
the fallback for portable checkouts.

### Video-extractor execution policy

`TRIBE_VIDEO_DEVICE` is explicit and accepts `cpu`, `cuda`, or `mps`.
`TRIBE_VIDEO_PRECISION` accepts `fp32` or `autocast-fp16`; the FP16 autocast
mode is deliberately limited to `mps`. The example CUDA/full configuration
keeps `fp32`, while `.env.local` selects:

```dotenv
TRIBE_VIDEO_DEVICE=mps
TRIBE_VIDEO_PRECISION=autocast-fp16
```

Neuralset's released Pydantic device field does not currently admit `mps`, so
the service constructs the released extractor configuration first and assigns
the validated device afterward, before its lazy V-JEPA2 model is created. The
autocast patch is similarly narrow: it activates only when the actual model is
on MPS and the model name is V-JEPA2. CPU and unrelated video-model calls retain
their upstream behavior.

The worker fails closed during model loading if MPS was requested but the
installed PyTorch build does not report a built and available MPS backend, or
if it cannot enter MPS FP16 autocast. It does not silently fall back to a proxy
or relabel CPU output as MPS output. `PYTORCH_ENABLE_MPS_FALLBACK=1` may still
let PyTorch execute individual unsupported operators on CPU, as designed by
PyTorch. Autocast can introduce the normal small floating-point differences
relative to FP32, so device and precision are recorded in status/results and
are part of the content-addressed result-cache identity.

## Exact repeat acceleration

The first prediction for a new video still performs the complete official
extractor and cortical-model computation. The vision-only local path now runs
the large V-JEPA2 backbone on MPS with the explicit precision policy above. The
backend does not shorten the clip, reduce the vertex count, interpolate frames,
or substitute a proxy to make that first computation look faster.

Exact repeats are accelerated safely:

1. The upload is SHA-256 hashed while its multipart bytes are streamed to disk;
   it is never buffered as one large in-memory value.
2. A cache identity is derived from that input hash, inference mode,
   preprocessing version, TRIBE model revision, TRIBE code revision, audited
   TRIBE checkpoint digest, pinned V-JEPA2 ID/revision/weight digest, and
   V-JEPA2 execution device/precision.
3. During a cache miss, the raw upload is moved to a deterministic
   content-addressed path while the existing process-wide inference lock is
   held. Neuralset/Exca therefore sees the same event filepath and can reuse its
   persistent official feature cache if a completed result is unavailable.
4. A completed result pointer is published atomically only after the manifest
   and `frames.f32` have been written and read back successfully.
5. Every hit rechecks the full manifest provenance, fsaverage5 mapping, shape,
   timing, expected byte length, actual file length, and a freshly streamed
   SHA-256 of `frames.f32`. Any mismatch fails closed to official inference.

Concurrent identical requests are serialized through the same inference lock.
The first request computes and publishes the result; the waiter then validates
and reuses it, so two copies of the model are not run for the same clip inside
one worker process. Changing any pinned model/code/weight input, changing
`TRIBE_INFERENCE_MODE`, or bumping `PREPROCESSING_VERSION` produces a different
identity and cannot reuse the previous artifact.

The content-addressed raw video plus any sibling audio/transcript scratch files
are removed in a `finally` block after inference. Completed cortical tensors,
source-derived thumbnails, and Neuralset feature caches remain persisted under
`TRIBE_RESULT_DIR` and `TRIBE_CACHE_DIR`, respectively; deployments with
retention requirements must manage those two directories accordingly. No raw
uploaded video is deliberately retained by this acceleration layer. The JPEG
thumbnails are capped at 320 pixels wide, but they are still sensitive user
media and must be covered by the deployment's access and deletion policy.

## Creator forecast capability status

`GET /api/forecast/v1/status` is a read-only, score-free readiness contract.
It does not change any `/api/tribe/v1/*` route or artifact. The response reports
the real available-now browser providers; live TRIBE/V-JEPA2 readiness; the
configured state of the executable V-JEPA 2.1, NanoLLaVA, AST, and measured
audio branches; explicitly unavailable VideoLLaMA, account, trend, and
competitor providers; and readiness for each target-specific calibration head.

The status endpoint never accepts an upload or emits a clip score. Forecast
uploads use `POST /api/forecast/v1/jobs`; that durable service enforces the
10–60 second inclusive range with authoritative server-side `ffprobe`, runs
the available evidence adapters, and publishes a validated result. Behavioral
values remain `null` unless the exact target has an executable,
release-approved production-calibrated head.

Optional artifacts can be registered through `FORECAST_REGISTRY_PATH` or
`FORECAST_REGISTRY_JSON`, but not both. Registration requires a known adapter,
an immutable full commit SHA, a real local artifact, and a matching SHA-256.
Calibration heads additionally require exact target/denominator/platform/
population/horizon declarations and hash-verified held-out calibration
evidence. Registration establishes integrity only; it cannot enable an
unimplemented executor or approve a behavioral head by itself. See
[`forecast/README.md`](forecast/README.md) for the full contract.

## API contract

### Predict

`POST /api/tribe/v1/predict` as `multipart/form-data` with one field named
`video`. Supported filename extensions match upstream TRIBE:
`.mp4`, `.avi`, `.mkv`, `.mov`, and `.webm`.

The call is intentionally synchronous because the returned manifest is only
created after real inference and strict validation. Inference is serialized per
process to avoid simultaneous use of the same GPU model.

```bash
curl -F video=@clip.mp4 http://localhost:8000/api/tribe/v1/predict
```

The successful response has this stable shape (values abbreviated):

```json
{
  "schemaVersion": "tribe-fsaverage5/1",
  "resultId": "...",
  "predictionType": "average-subject cortical BOLD response",
  "isViralityScore": false,
  "inferenceMode": "vision-only",
  "modalitiesUsed": ["video"],
  "missingModalities": ["audio", "text"],
  "extractors": {
    "video": {
      "id": "facebook/vjepa2-vitg-fpc64-256",
      "revision": "875c192b7b704b87d1e1d99345769632dd5f739a",
      "weightsSha256": "f205e77aa2ade168db6b09d4bc420d156141f64ab964278a9c181a2bdf2a232b",
      "device": "mps",
      "precision": "autocast-fp16"
    }
  },
  "model": {
    "id": "facebook/tribev2",
    "revision": "<40-character model commit>",
    "weightsSha256": "9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321",
    "codeRevision": "af58661791a351a448a489042a28f6c37e1c14b7",
    "license": "CC-BY-NC-4.0"
  },
  "cache": {
    "identityVersion": "tribe-result-cache/1",
    "key": "<provenance-bound cache key>",
    "preprocessingVersion": "tribev2-neuralset-video-events/1",
    "thumbnailContractVersion": "source-thumbnails/2"
  },
  "surface": {
    "space": "fsaverage5",
    "mappingId": "fsaverage5-half:52eda3c221c9439b320f97663d441390c56a14e69c2cddb7a5073f6d550466cb",
    "vertexCount": 20484,
    "hemispheres": [
      {"id": "left", "offset": 0, "count": 10242},
      {"id": "right", "offset": 10242, "count": 10242}
    ]
  },
  "frames": {
    "url": "/api/tribe/v1/results/<id>/frames.f32",
    "dtype": "float32",
    "byteOrder": "little-endian",
    "layout": "time-major",
    "shape": [53, 20484],
    "byteLength": 4342608,
    "sha256": "...",
    "timesSeconds": [0.0, 1.0],
    "durationsSeconds": [1.0, 1.0],
    "hemodynamicOffsetSeconds": 5.0,
    "displayScale": {
      "type": "robust-diverging-linear",
      "domain": [-1.73, 0.0, 2.04],
      "percentiles": [1.0, 99.0],
      "centerMethod": "zero",
      "scope": "all-frames-all-vertices",
      "clamp": true,
      "units": "TRIBE-v2 predicted BOLD (training-target z-score units)"
    }
  },
  "thumbnails": {
    "status": "available",
    "source": "uploaded-video",
    "captureTimeBasis": "tribe-segment-start",
    "selection": "first decoded source frame at-or-after requested time",
    "format": "image/jpeg",
    "maxWidthPixels": 320,
    "items": [
      {
        "index": 0,
        "requestedTimeSeconds": 0.0,
        "url": "/api/tribe/v1/results/<id>/thumbnails/0000.jpg",
        "byteLength": 18420,
        "sha256": "..."
      }
    ]
  }
}
```

`timesSeconds` and `durationsSeconds` come directly from the segment objects
returned alongside the prediction matrix; the service never replaces them with
frame indices. The 5-second hemodynamic offset is read from the loaded model's
`data.neuro.offset` and must equal the published value or validation fails.

The binary is a flat little-endian float32 array. Value `(frame, vertex)` starts
at byte `4 * (frame * 20484 + vertex)`. Vertex order is left fsaverage5
`0..10241`, then right fsaverage5 `10242..20483`. The fixed display scale makes
every frame in one clip use the same global 1st/99th-percentile bounds, matching
TRIBE's robust plotting convention and preventing second-by-second rescaling.
When zero is inside those bounds it is the diverging center; otherwise the
global median (or, for a tied median, the p1/p99 midpoint) is used. Raw values
remain untouched in the stored artifact. Because the robust bounds are computed
per result, colors should not be compared quantitatively across different
uploads without an additional calibrated analysis.

For `status: "available"`, `thumbnails.items` contains exactly one real
source-video still per TRIBE interval and preserves the same indices as
`frames.timesSeconds`. ffmpeg
uses decode-accurate seeking: because compressed video has discrete frames, the
still is the first decoded frame at or after the requested authoritative segment
start, not a fabricated image or a claim of an impossible continuous-time
frame. Each JPEG is length- and SHA-256-validated when a cached result is reused.

If an authoritative TRIBE start is at or beyond the selected source-video
stream duration and ffmpeg therefore cannot decode a frame there, already
decoded stills are preserved. The manifest returns `status: "partial"`, sparse
`items` for the real JPEGs, and an `unavailableItems` entry of the following
exact form for every out-of-range frame:

```json
{
  "index": 10,
  "requestedTimeSeconds": 10.0,
  "reason": "outside-source-video"
}
```

`items` plus `unavailableItems` must form an exact, non-overlapping partition
of every TRIBE frame index and each time must match `frames.timesSeconds`.
Only manifest-declared JPEG indices are served. An in-range decode failure is
not relabeled as an expected boundary omission: all partial stills are removed
and the valid cortical tensor is retained with the systemic unavailability
contract.

If a codec cannot be decoded or ffmpeg is unavailable, the valid cortical tensor
is retained and the manifest instead returns `status: "unavailable"`,
`reason: "source-frame-extraction-failed"`, and an empty `items` array.

### Read result/status

- `GET /api/tribe/v1/results/{id}/frames.f32`
- `GET /api/tribe/v1/results/{id}/manifest.json`
- `GET /api/tribe/v1/results/{id}/thumbnails/{index}.jpg` — sensitive persisted
  source derivative; served with `Cache-Control: private, no-store`
- `GET /api/tribe/v1/status` — model readiness and last runtime failure
- `GET /api/tribe/v1/health` — process liveness only
- `GET /api/forecast/v1/status` — read-only evidence/calibration capabilities;
  no clip scores
- `POST /api/forecast/v1/jobs` — bounded private 10–60 second evidence upload
- `GET /api/forecast/v1/jobs/{id}` — durable state, stage, and elapsed time
- `GET /api/forecast/v1/results/{id}` — validated atomically published result

## Tests

Tests exercise fsaverage5 shape/finite-value validation, segment timing, exact
little-endian serialization, checksums, source-thumbnail timing/hashing/failure
cleanup, the manifest contract, streamed upload
hashing, cache-identity invalidation, deterministic staging cleanup, corrupt or
truncated artifact rejection, and concurrent repeat deduplication. They import
neither `tribev2` nor model weights:

```bash
python -m unittest discover -s backend/tests -v
```
