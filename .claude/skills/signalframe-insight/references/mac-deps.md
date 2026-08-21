# Mac-only dependencies for the insight layer

Apple Silicon is the primary target. **No CUDA-only code path may be introduced anywhere.**
Optional acceleration is guarded and falls back explicitly, never silently.

Insight dependencies are **not** in `backend/requirements.txt`. They live in
`backend/requirements-insight.txt` with exact pinned versions so the model-free CI job keeps
installing nothing new.

```
mlx-lm==0.28.4
mlx-whisper==0.4.3
anthropic==0.75.0
ocrmac==1.0.0
pytesseract==0.3.13
```

Every one of these is imported lazily inside the adapter that needs it. An `ImportError`
means that adapter reports `unavailable` with a reason — it never breaks module import, the
service, or another lane.

## Availability probes

An availability probe is cheap and non-destructive: environment variables, `importlib.util.find_spec`, and
a pinned-revision check. It must not download weights, load a model, or make a network call.

| Adapter | Available only when |
| --- | --- |
| `mlx-local` | `mlx_lm` importable, `INSIGHT_LOCAL_MODEL` set, `INSIGHT_LOCAL_MODEL_REVISION` is a 40-hex commit SHA, and the revision verifies before load |
| `anthropic` | `INSIGHT_CLOUD_ENABLED=true`, `ANTHROPIC_API_KEY` present, `anthropic` importable, `INSIGHT_ANTHROPIC_MODEL` set |
| ASR (`mlx-whisper`) | `mlx_whisper` importable and `INSIGHT_ASR_MODEL_REVISION` is a 40-hex commit SHA |
| OCR (`ocrmac`) | `ocrmac` importable and Apple Vision usable; records `sw_vers -productVersion` |
| OCR fallback | `pytesseract` importable **and** the `tesseract` binary on `PATH` |

Providers never chain. A disabled or unverified provider yields `provider_unavailable`; the
service does not quietly try a different one.

## Environment keys

```
INSIGHT_PROVIDER=mlx-local
INSIGHT_CLOUD_ENABLED=false
ANTHROPIC_API_KEY=
INSIGHT_ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
INSIGHT_LOCAL_MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit
INSIGHT_LOCAL_MODEL_REVISION=
INSIGHT_MAX_OUTPUT_TOKENS=1536
INSIGHT_TIMEOUT_SECONDS=120
INSIGHT_ASR_MODEL=mlx-community/whisper-large-v3-turbo
INSIGHT_ASR_MODEL_REVISION=
INSIGHT_OCR_ENGINE=ocrmac
```

`ANTHROPIC_API_KEY` is never echoed by `/api/insight/v1/status`, never logged, and never
written into a provenance manifest or a rejection record.

## Media contracts

- The ASR lane reuses the **existing** deterministic 16 kHz mono decode contract in
  `backend/forecast/workers/measured_audio.py`. Do not add a second ffmpeg path.
- The OCR lane reuses the **existing** six deterministic 384 px keyframes produced by the
  NanoLLaVA preprocessing contract. Do not sample new frames.
- Raw media never leaves the machine. Only derived, validated JSON may be sent to a remote
  provider, and only when the operator opted in.
