# CLAUDE.md — SignalFrame project memory

SignalFrame is an evidence-first, pre-publish analysis lab for short-form video with an
independently verified TRIBE v2 cortical-response viewer. React/Vite frontend, Python 3.11
backend, macOS Apple Silicon is the primary dev target (MLX / MPS paths).

Before writing any code, read: `README.md`, `docs/ARCHITECTURE.md`, `docs/SCIENTIFIC_LIMITS.md`,
`backend/README.md`, and `backend/forecast/README.md`. Discover real module layout, naming, and
test patterns from the code — do not assume framework details that are not in the repo.

## Non-negotiable product invariants

1. **Three evidence lanes stay separate.** Content signals, multimodal encoder evidence, and
   TRIBE v2 descriptors never blend into a single score. TRIBE tensors and brain-derived
   descriptors are **forbidden forecast features** — never feed them into any behavioral head,
   heuristic, ranking, or prompt that outputs an outcome claim.
2. **Fail closed.** A missing model, invalid schema, unverified artifact, failed hash, or WebGL
   failure produces an explicit `unavailable` state with a reason — never a guess, fallback
   score, or decorative substitute.
3. **Provenance travels with everything.** Every adapter records model identity, revision/weight
   digest, preprocessing contract, input hash, output hash, and timing in a manifest.
4. **Every adapter declares `behavioralOutcome: false`.** Only a head that passes the
   code-owned production approval contract may emit a behavioral value. Nothing in this repo
   currently qualifies. LLM output NEVER qualifies.
5. **No fabricated numbers.** UI and text output may only show numbers that were measured,
   returned by a verified model, or copied verbatim from validated evidence.
6. **Creator media is sensitive.** Uploads are bounded, private/no-store, deleted after jobs.
   Raw video/audio/frames never leave the machine to any cloud service. Only derived,
   validated JSON evidence may be sent to a remote LLM, and only when the operator opted in.
7. **TRIBE language discipline.** TRIBE values are "predicted average-subject cortical BOLD."
   Never describe them as attention, emotion, engagement, memory, arousal, what viewers feel,
   or virality. No reverse inference from anatomical parcels to mental states.

## Build, run, test (macOS)

- Frontend: `npm ci`, `npm run dev -- --host 127.0.0.1 --port 4173`, `npm run build`,
  `npm test`, `npm run test:sites`.
- Backend: Python 3.11 venv at `$SIGNALFRAME_RUNTIME/py311` (outside iCloud paths);
  interpreter exported as `SIGNALFRAME_BACKEND_PYTHON`; run via `./backend/run-local-mac.sh`;
  tests via `python -m unittest discover -s backend/tests -v`.
- Config lives in `backend/.env.local` (gitignored). Never commit tokens, keys, or `.env.local`.
- Device settings use MPS on Apple Silicon (`TRIBE_DEVICE=mps`). Never introduce CUDA-only
  code paths; guard optional acceleration and fall back to CPU explicitly.

## Rules for new code

- New evidence branches follow the existing adapter pattern: pinned artifact, verified digest
  before load, deterministic preprocessing contract, strict output schema, branch-local failure
  (one broken branch never breaks the job), `behavioralOutcome: false`, unit tests.
- Results are staged, fsynced, and atomically renamed before a job is `complete`. Interrupted
  work is marked failed on restart.
- Add or update tests for every behavior you add, including the fail-closed paths. Run the
  full frontend and backend suites before declaring a task done.
- Keep third-party licenses honest: update `NOTICE.md` when adding models or scientific data,
  and never bundle weights in the repo.

## LLM insight layer

All work on the insight service, Hook Doctor, ASR/OCR lanes, experiment tracker, or any
feature that turns evidence into natural language MUST follow the
`signalframe-insight` skill at `.claude/skills/signalframe-insight/SKILL.md`. Read it first.
