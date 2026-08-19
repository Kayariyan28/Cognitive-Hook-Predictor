# macOS setup

This guide takes a clean checkout to a working local SignalFrame installation
on Apple silicon. It deliberately separates the base application from optional
model artifacts so that an unavailable model is visible instead of silently
replaced.

## Supported local profile

| Component | macOS profile |
| --- | --- |
| Interface and browser measurements | Supported |
| Forecast job service and measured audio | Supported |
| V-JEPA 2.1 Base evidence | Supported on CPU, float32 |
| AST AudioSet evidence | Supported on CPU |
| NanoLLaVA keyframe semantics | Supported in a separate MLX environment |
| TRIBE v2 | Supported as a labeled `vision-only` research ablation |
| Full TRIBE video/audio/text inference | Use Linux with an NVIDIA GPU |

The macOS TRIBE path uses the official V-JEPA2 visual extractor and released
cortical encoder. It does not claim to reproduce the full trimodal result. The
separate Forecast Lab V-JEPA **2.1 Base** branch is not the V-JEPA2 backbone
inside TRIBE.

## 1. Clone and enter the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

If you received a local archive, open Terminal in its extracted directory and
continue with the next step.

## 2. Install system dependencies

Install Apple's command-line tools and Homebrew packages:

```bash
xcode-select --install
brew install node python@3.11 ffmpeg git git-lfs
```

Verify the executables:

```bash
node --version
npm --version
python3.11 --version
ffmpeg -version
ffprobe -version
git --version
```

Use Python 3.11 for the backend. Newer Python versions may not be compatible
with the pinned upstream TRIBE dependency graph.

## 3. Install the frontend

```bash
npm ci
```

`npm ci` uses the committed lockfile and should not modify it.

## 4. Create the backend environment

A virtual environment outside a synced Documents/Desktop folder avoids macOS
File Provider placeholder stalls:

```bash
export SIGNALFRAME_RUNTIME="$HOME/Library/Application Support/SignalFrame"
mkdir -p "$SIGNALFRAME_RUNTIME"

python3.11 -m venv "$SIGNALFRAME_RUNTIME/py311"
export SIGNALFRAME_BACKEND_PYTHON="$SIGNALFRAME_RUNTIME/py311/bin/python"

"$SIGNALFRAME_BACKEND_PYTHON" -m pip install --upgrade pip wheel
"$SIGNALFRAME_BACKEND_PYTHON" -m pip install \
  'torch==2.6.0' 'torchvision==0.21.0'
"$SIGNALFRAME_BACKEND_PYTHON" -m pip install -r backend/requirements.txt
"$SIGNALFRAME_BACKEND_PYTHON" -m pip install \
  'timm==1.0.19' 'einops==0.8.2' 'transformers==4.57.6'
```

This is the tested direct dependency set. The project requirements pin the
TRIBE source commit, API versions, NumPy, and Hugging Face client. Torch,
Torchvision, timm, einops, and Transformers are stated explicitly above because
the optional evidence workers also depend on them. Transitive upstream packages
are not supplied as a hash-locked environment; generate and review a complete
lock before treating this as a reproducible production deployment.

## 5. Configure the backend

Create the untracked environment file:

```bash
cp backend/.env.macos.example backend/.env.local
```

Edit `backend/.env.local`. Keep the immutable revision values from the
template, then set this local profile:

```dotenv
SIGNALFRAME_RUNTIME="$HOME/Library/Application Support/SignalFrame"
SIGNALFRAME_BACKEND_PYTHON="$SIGNALFRAME_RUNTIME/py311/bin/python"

TRIBE_MODEL_ID=facebook/tribev2
TRIBE_MODEL_REVISION=f894e783020944dcd96e5568550afe2aa9743f9f
TRIBE_CHECKPOINT_NAME=best.ckpt
TRIBE_DEVICE=mps
TRIBE_INFERENCE_MODE=vision-only
TRIBE_VIDEO_DEVICE=mps
TRIBE_VIDEO_PRECISION=autocast-fp16
TRIBE_PRELOAD_MODEL=false

TRIBE_CORS_ORIGINS=http://localhost:4173,http://127.0.0.1:4173
TRIBE_CORS_ALLOW_CREDENTIALS=false
TRIBE_MAX_UPLOAD_BYTES=536870912

FORECAST_JOB_TIMEOUT_SECONDS=1800
FORECAST_MAX_ACTIVE_JOBS=4
FORECAST_MAX_WORKERS=1
```

Leave `HF_TOKEN` empty unless you set it to a real read token in the ignored
local file. Prefer the Hugging Face credential store:

```bash
"$SIGNALFRAME_RUNTIME/py311/bin/hf" auth login
```

Never put a credential in the repository, a screenshot, a bug report, or a
shell command committed to history. For the first model download, leave
`HF_HUB_OFFLINE` unset or set it to `false`. After every required artifact is
cached, an offline deployment may set `HF_HUB_OFFLINE=1`.

The first TRIBE request loads the pinned model because preload is disabled.
The backend verifies the exact expected checkpoint digest and fails if it does
not match. Full trimodal inference also requires access to its gated language
model; see [the backend guide](../backend/README.md) for the Linux/CUDA profile.

## 6. Optional: install V-JEPA 2.1 Base evidence

The Forecast Lab worker accepts only a clean official checkout at commit
`204698b45b3712590f06245fbfba32d3be539812` and the configured checkpoint
digest.

```bash
export SIGNALFRAME_RUNTIME="$HOME/Library/Application Support/SignalFrame"
mkdir -p "$SIGNALFRAME_RUNTIME/models/vjepa21"

git clone https://github.com/facebookresearch/vjepa2.git \
  "$SIGNALFRAME_RUNTIME/models/vjepa21/source"
git -C "$SIGNALFRAME_RUNTIME/models/vjepa21/source" checkout --detach \
  204698b45b3712590f06245fbfba32d3be539812

curl --fail --location --retry 3 \
  https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt \
  --output "$SIGNALFRAME_RUNTIME/models/vjepa21/vjepa2_1_vitb_dist_vitG_384.pt"

echo '34d5d0ae6f1297511b9d669890350d059dd3ff7b75df23eb845f2aa14e610220  '"$SIGNALFRAME_RUNTIME/models/vjepa21/vjepa2_1_vitb_dist_vitG_384.pt" \
  | shasum -a 256 --check
```

Stop if the checksum does not match. Do not “fix” the environment by inserting
a new hash without reviewing the upstream artifact and compatibility.

Add to `backend/.env.local`:

```dotenv
FORECAST_VJEPA21_SOURCE_PATH="$SIGNALFRAME_RUNTIME/models/vjepa21/source"
FORECAST_VJEPA21_CHECKPOINT_PATH="$SIGNALFRAME_RUNTIME/models/vjepa21/vjepa2_1_vitb_dist_vitG_384.pt"
FORECAST_VJEPA21_CHECKPOINT_SHA256=34d5d0ae6f1297511b9d669890350d059dd3ff7b75df23eb845f2aa14e610220
FORECAST_VJEPA21_DEVICE=cpu
```

CPU float32 is the accuracy-first setting for this worker on the tested Mac.
Its MPS path is not enabled automatically because the observed latency was
nondeterministic. Check configuration after loading the environment:

```bash
set -a
source backend/.env.local
set +a
"$SIGNALFRAME_BACKEND_PYTHON" -m backend.forecast.workers.vjepa21 status
```

## 7. Optional: install AST AudioSet evidence

Download the exact snapshot into the external model directory:

```bash
export SIGNALFRAME_RUNTIME="$HOME/Library/Application Support/SignalFrame"
"$SIGNALFRAME_RUNTIME/py311/bin/hf" download \
  MIT/ast-finetuned-audioset-10-10-0.4593 \
  --revision f826b80d28226b62986cc218e5cec390b1096902 \
  --local-dir "$SIGNALFRAME_RUNTIME/models/ast"

echo 'ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d  '"$SIGNALFRAME_RUNTIME/models/ast/model.safetensors" \
  | shasum -a 256 --check
```

Then add:

```dotenv
FORECAST_AST_SNAPSHOT="$SIGNALFRAME_RUNTIME/models/ast"
```

AST returns independent AudioSet label scores for deterministic ten-second
audio windows. Those labels are not speech transcription, music quality,
retention, engagement, or virality.

## 8. Optional: install NanoLLaVA keyframe semantics

Keep MLX-VLM isolated from the TRIBE environment:

```bash
export SIGNALFRAME_RUNTIME="$HOME/Library/Application Support/SignalFrame"

python3.11 -m venv "$SIGNALFRAME_RUNTIME/mlx-vlm-0.6.8"
"$SIGNALFRAME_RUNTIME/mlx-vlm-0.6.8/bin/python" -m pip install --upgrade pip
"$SIGNALFRAME_RUNTIME/mlx-vlm-0.6.8/bin/python" -m pip install 'mlx-vlm==0.6.8'

"$SIGNALFRAME_RUNTIME/py311/bin/hf" download \
  mlx-community/nanoLLaVA-1.5-8bit \
  --revision 6e2bc13b87ab178668313552b9d69026af7d556f \
  --local-dir "$SIGNALFRAME_RUNTIME/models/nanollava"

echo 'f0a1d1517ad9e6e810bc6ac99956643c66e4b87a2f82bd5d1b5cb0966e5c5476  '"$SIGNALFRAME_RUNTIME/models/nanollava/model.safetensors" \
  | shasum -a 256 --check
```

Then add:

```dotenv
FORECAST_NANOLLAVA_PYTHON="$SIGNALFRAME_RUNTIME/mlx-vlm-0.6.8/bin/python"
FORECAST_NANOLLAVA_SNAPSHOT="$SIGNALFRAME_RUNTIME/models/nanollava"
FORECAST_NANOLLAVA_TIMEOUT_SECONDS=300
```

SignalFrame invokes this branch in one bounded, offline child process for six
deterministic source frames. It disables remote code and accepts only
schema-valid JSON. It is explicitly a still-keyframe fallback, not VideoLLaMA,
OCR, audio understanding, or audience prediction.

## 9. Run SignalFrame

Start the backend in Terminal 1:

```bash
./backend/run-local-mac.sh
```

Start the frontend in Terminal 2:

```bash
npm run dev -- --host 127.0.0.1 --port 4173
```

Open [http://127.0.0.1:4173](http://127.0.0.1:4173) in a native,
hardware-accelerated browser.

## 10. Verify the installation

Confirm both capability documents respond:

```bash
curl --fail http://127.0.0.1:8000/api/tribe/v1/status
curl --fail http://127.0.0.1:8000/api/forecast/v1/status
```

Expected distinctions:

- TRIBE may report `not_loaded` until the first cortical prediction; this is
  different from `unconfigured` or `error`.
- Installed evidence branches report their exact model, revision, weights,
  preprocessing, and readiness.
- VideoLLaMA remains explicitly unavailable; NanoLLaVA never impersonates it.
- `scoreGenerationAvailable` remains false until a release-approved calibrated
  head exists.
- TRIBE and its internal V-JEPA2 extractor report
  `forecastContribution: false`.

Run the test suites:

```bash
npm run build
npm test
npm run test:sites

set -a
source backend/.env.local
set +a
"$SIGNALFRAME_BACKEND_PYTHON" -m unittest discover -s backend/tests -v
```

Build first on a clean checkout because the packaging contract verifies the
generated files under `dist/`.

## Performance expectations

Model latency depends heavily on clip length, thermals, memory pressure, and
the exact Apple chip. Saved local evidence in this repository records:

- a V-JEPA 2.1 Base CPU float32 forward of roughly 10.6 seconds for one
  `64 x 384 x 384` window on the tested M4 Max;
- a complete 15-second four-branch forecast-evidence run of roughly 34 seconds
  on that host; and
- a six-second TRIBE vision-only run of roughly 6 minutes 55 seconds with its
  MPS FP16 visual path.

These are measurements, not service-level guarantees. Exact repeat caching can
reuse a fully revalidated TRIBE artifact, but a new input always runs the real
model path. The UI reports actual stages and elapsed time instead of a made-up
percentage.

## Troubleshooting

### “WebGL is unavailable”

The app does not replace the original 3D cortical viewer with a fake brain.
Use a current native Chrome, Safari, or Firefox with hardware acceleration
enabled. Embedded or sandboxed in-app browsers may disable GPU/WebGL even when
the Mac supports it. Restart the native browser after changing GPU settings.

### The brain stays neutral

This is correct unless a validated TRIBE result is loaded. Check the TRIBE
status route, then inspect the backend terminal for model, hash, shape, or
download errors. Browser-side scores are intentionally forbidden from coloring
the mesh.

### Backend shows `not_loaded`

With `TRIBE_PRELOAD_MODEL=false`, loading is deferred until the first TRIBE
prediction. `not_loaded` is not a forecast-worker failure. If you want startup
to load the model, set `TRIBE_PRELOAD_MODEL=true` and expect a much slower
launch.

### A model snapshot is “unavailable”

Check that the path is absolute after `backend/.env.local` is sourced, every
required file exists, and the exact hash matches. The workers do not download
missing files or accept a mutable model revision during a job.

### Hugging Face returns 401/403

Run `hf auth login`, verify that your account accepted any gated upstream
terms, and retry the download. Do not work around access controls by copying a
token into source code.

### Python imports stall under Documents or Desktop

Confirm `SIGNALFRAME_BACKEND_PYTHON` points to the external environment created
above. Reinstall the environment there instead of relying on offloaded
File Provider placeholders inside the checkout.

### Port 8000 or 4173 is already in use

Stop the conflicting process. If you deliberately move the API port, also set
`VITE_TRIBE_API_URL` in the root `.env.local`, for example:

```dotenv
VITE_TRIBE_API_URL=http://127.0.0.1:8001
```

Then restart Vite so it receives the new value.

### MPS initialization fails

Verify that the installed Torch build reports MPS availability. Do not silently
substitute a proxy. CPU fallback is possible only when you explicitly change
the device and precision settings and accept the substantial latency increase.

## Next steps

- Read [Architecture](ARCHITECTURE.md) before changing provider boundaries.
- Read [Scientific limits](SCIENTIFIC_LIMITS.md) before changing creator-facing
  terminology.
- Read [Publish to GitHub](PUBLISH_TO_GITHUB.md) before making the repository
  public.
