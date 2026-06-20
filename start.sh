#!/usr/bin/env bash
#
# DUM-E launcher: starts the FastAPI backend and the Next.js frontend together.
#
#   ./start.sh
#
# Backend  -> http://127.0.0.1:8000   (FastAPI; serves /api and the train WS)
# Frontend -> http://localhost:3000   (Next.js; proxies /api to the backend)
#
# Requires a .env file (MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, HUGGINGFACE_TOKEN,
# OPENROUTER_KEY). The Modal trainer must be deployed once beforehand:
#   modal deploy modal_app/trainer.py
#
# Works on macOS's stock Bash 3.2 (no `wait -n`, no associative arrays).
set -eu

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
PY="${PYTHON:-python3}"

# --- preflight ------------------------------------------------------------
if [ ! -f .env ]; then
  echo "error: .env not found. Create it with MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, HUGGINGFACE_TOKEN, OPENROUTER_KEY." >&2
  exit 1
fi

# Export .env so uvicorn (and Modal lookups) see the credentials.
set -a
# shellcheck disable=SC1091
. ./.env
set +a

if ! "$PY" -c "import uvicorn" >/dev/null 2>&1; then
  echo "error: '$PY' cannot import uvicorn. Install backend deps: pip install -r backend/requirements.txt" >&2
  exit 1
fi

BACKEND_LOG="$ROOT/.backend.log"

# --- cleanup on exit ------------------------------------------------------
BACKEND_PID=""
FRONTEND_PID=""
cleanup() {
  echo ""
  echo "shutting down..."
  [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null || true
  [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null || true
  # also free the ports in case a child re-forked
  if command -v lsof >/dev/null 2>&1; then
    for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
      pids="$(lsof -ti:"$port" 2>/dev/null || true)"
      [ -n "$pids" ] && kill $pids 2>/dev/null || true
    done
  fi
}
trap cleanup INT TERM EXIT

# --- backend --------------------------------------------------------------
echo "[backend] starting on http://${BACKEND_HOST}:${BACKEND_PORT} (log: .backend.log)"
"$PY" -m uvicorn backend.app.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" \
  >"$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

# Wait until the backend actually answers (or its process dies). Old-Bash safe.
echo "[backend] waiting for it to come up..."
ok=""
i=0
while [ "$i" -lt 40 ]; do
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "error: backend process exited during startup. Last lines:" >&2
    tail -n 20 "$BACKEND_LOG" >&2
    exit 1
  fi
  if curl -fsS "http://${BACKEND_HOST}:${BACKEND_PORT}/api/learned?limit=1" >/dev/null 2>&1; then
    ok="yes"
    break
  fi
  sleep 0.5
  i=$((i + 1))
done

if [ -z "$ok" ]; then
  echo "error: backend did not respond on :${BACKEND_PORT} within 20s. Last lines:" >&2
  tail -n 20 "$BACKEND_LOG" >&2
  exit 1
fi
echo "[backend] ready."

# --- frontend -------------------------------------------------------------
if [ ! -d frontend/node_modules ]; then
  echo "[frontend] installing dependencies (first run)..."
  ( cd frontend && npm install )
fi

echo "[frontend] starting on http://localhost:${FRONTEND_PORT}"
( cd frontend && BACKEND_URL="http://${BACKEND_HOST}:${BACKEND_PORT}" npm run dev -- --port "$FRONTEND_PORT" ) &
FRONTEND_PID=$!

echo ""
echo "DUM-E is up. Open http://localhost:${FRONTEND_PORT}"
echo "Press Ctrl-C to stop both."

# Poll both children (Bash 3.2 has no `wait -n`). Exit when either dies.
while kill -0 "$BACKEND_PID" 2>/dev/null && kill -0 "$FRONTEND_PID" 2>/dev/null; do
  sleep 1
done
echo "a process exited; stopping the other."
