#!/usr/bin/env bash
# Start SignalFrame locally and open it in your browser.
#
#   ./scripts/start-mac.sh
#
# Checks prerequisites, prepares the model-free Python environment, loads
# backend/.env.local (creating it from the macOS example on first run), starts
# the API and the interface, waits for both to answer, and opens the app.
# Ctrl-C stops both.
set -uo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-4173}"
VENV="${VENV:-$root/.venv}"

say() { printf "\033[1m==>\033[0m %s\n" "$1"; }
warn() { printf "\033[33m !\033[0m %s\n" "$1"; }
die() { printf "\033[31m x\033[0m %s\n" "$1" >&2; exit 1; }

# ---- prerequisites -------------------------------------------------------
command -v node >/dev/null || die "Node.js 20+ is required (brew install node)."

python_bin=""
for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null; then
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null; then
      python_bin="$candidate"; break
    fi
  fi
done
[ -n "$python_bin" ] || die "Python 3.11+ is required (brew install python@3.11)."
say "Using $($python_bin --version)"

if ! command -v ffmpeg >/dev/null; then
  warn "ffmpeg not found. The measured-audio branch and the demo clip need it."
  warn "Install it with: brew install ffmpeg"
fi

port_busy() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }
port_busy "$API_PORT" && die "Port $API_PORT is already in use. Set API_PORT=... to change it."
port_busy "$WEB_PORT" && die "Port $WEB_PORT is already in use. Set WEB_PORT=... to change it."

# ---- python environment --------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  say "Creating the Python environment at $VENV"
  "$python_bin" -m venv "$VENV" || die "Could not create the virtual environment."
fi
say "Installing the model-free backend requirements"
"$VENV/bin/pip" install --quiet --upgrade pip >/dev/null 2>&1
"$VENV/bin/pip" install --quiet -r backend/requirements-local.txt \
  || die "Backend dependency install failed."

# ---- node dependencies ---------------------------------------------------
if [ ! -d node_modules ]; then
  say "Installing frontend dependencies"
  npm ci || die "npm ci failed."
fi

# ---- configuration -------------------------------------------------------
# Plain uvicorn does not read this file, so it is loaded here. Without it every
# optional lane, including TRIBE, stays unconfigured no matter what is set.
if [ ! -f backend/.env.local ]; then
  say "Creating backend/.env.local from the macOS example"
  cp backend/.env.macos.example backend/.env.local
fi
set -a
# shellcheck disable=SC1091
. ./backend/.env.local
set +a

# ---- run -----------------------------------------------------------------
cleanup() {
  say "Stopping"
  [ -n "${api_pid:-}" ] && kill "$api_pid" 2>/dev/null
  [ -n "${web_pid:-}" ] && kill "$web_pid" 2>/dev/null
  wait 2>/dev/null
  exit 0
}
trap cleanup INT TERM

say "Starting the API on http://127.0.0.1:$API_PORT"
PYTHONPATH="$root" "$VENV/bin/python" -m uvicorn backend.app:app \
  --host 127.0.0.1 --port "$API_PORT" > /tmp/signalframe-api.log 2>&1 &
api_pid=$!

say "Starting the interface on http://127.0.0.1:$WEB_PORT"
VITE_TRIBE_API_URL="http://127.0.0.1:$API_PORT" \
  npm run dev -- --host 127.0.0.1 --port "$WEB_PORT" > /tmp/signalframe-web.log 2>&1 &
web_pid=$!

wait_for() {
  for _ in $(seq 1 40); do
    curl -fsS -o /dev/null "$1" 2>/dev/null && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 0.5
  done
  return 1
}

if ! wait_for "http://127.0.0.1:$API_PORT/api/insight/v1/status" "$api_pid"; then
  warn "The API did not come up. Last lines of /tmp/signalframe-api.log:"
  tail -15 /tmp/signalframe-api.log >&2
  cleanup
fi
if ! wait_for "http://127.0.0.1:$WEB_PORT/" "$web_pid"; then
  warn "The interface did not come up. Last lines of /tmp/signalframe-web.log:"
  tail -15 /tmp/signalframe-web.log >&2
  cleanup
fi

echo
say "SignalFrame is running"
printf "    Interface  http://127.0.0.1:%s\n" "$WEB_PORT"
printf "    API        http://127.0.0.1:%s\n" "$API_PORT"
echo
"$VENV/bin/python" - "$API_PORT" <<'PY'
import json, sys, urllib.request
port = sys.argv[1]
def get(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
        return json.load(r)
insight = get("/api/insight/v1/status")
forecast = get("/api/forecast/v1/status")
print("    Lane readiness")
for key, branch in forecast["branches"].items():
    if key.startswith("browserLocal"):
        continue
    print(f"      {key:<16} {branch['state']}")
print(f"      {'insight model':<16} {insight['state']}")
print()
print("    A lane reporting 'not-configured' is working as designed: it names")
print("    the artifact it needs rather than substituting anything.")
PY
echo
if command -v open >/dev/null; then
  open "http://127.0.0.1:$WEB_PORT" 2>/dev/null || true
fi
say "Ctrl-C to stop both."
wait
