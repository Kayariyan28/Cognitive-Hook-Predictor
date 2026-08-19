#!/bin/zsh
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"

if [[ -f "$script_dir/.env.local" ]]; then
  set -a
  source "$script_dir/.env.local"
  set +a
fi

# Use the repository-local environment by default. People who keep this
# checkout in iCloud/File Provider can point at a physical Python environment
# elsewhere with SIGNALFRAME_BACKEND_PYTHON.
runtime_python="${SIGNALFRAME_BACKEND_PYTHON:-$script_dir/.venv/bin/python}"

if [[ ! -x "$runtime_python" ]]; then
  print -u2 "No usable backend Python runtime found at: $runtime_python"
  print -u2 "Create backend/.venv with Python 3.11 or set SIGNALFRAME_BACKEND_PYTHON."
  exit 1
fi

export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_dir"
exec "$runtime_python" -m uvicorn backend.app:app --host 127.0.0.1 --port "${PORT:-8000}"
