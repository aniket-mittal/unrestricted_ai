#!/usr/bin/env bash
#
# DUM-E reset: make the shared model forget everything it was taught.
#
#   ./reset.sh
#
# Clears two things:
#   1. Modal volume "unrestricted-weights" (the CURRENT pointer + every v{N}
#      adapter), so Trainer.generate falls back to the pristine base model.
#   2. Local SQLite learning state (lessons, training_pairs, weights_versions,
#      learned_feed, training_jobs), so the "Recently Learned" feed empties too.
#
# Conversations/messages are kept by default (chat history is harmless); pass
# --all to wipe those as well.
#
# Requires .env (Modal + HF tokens). Works on macOS's stock Bash 3.2.
set -eu

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
WIPE_ALL=""
[ "${1:-}" = "--all" ] && WIPE_ALL="1"

if [ ! -f .env ]; then
  echo "error: .env not found (need MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, HUGGINGFACE_TOKEN)." >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
. ./.env
set +a

echo "This will make the shared DUM-E model FORGET everything it was taught."
printf "Continue? [y/N] "
read -r ans
case "$ans" in
  y|Y|yes|YES) ;;
  *) echo "aborted."; exit 0 ;;
esac

# Where the backend listens (so we can use the authoritative endpoint if up).
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
BASE_URL="http://${BACKEND_HOST}:${BACKEND_PORT}"

# --- Preferred path: authoritative reset via the running backend -----------
# If the backend is up, POST /api/admin/reset DRAINS the training worker before
# wiping (so an in-flight finetune can't re-insert a version after the wipe),
# wipes the volume + warm container memory, clears the DB + broadcasters, and
# restarts the worker — all coordinated in one process. No stale state survives
# and no manual backend restart is needed.
WIPE_QS=""
[ -n "$WIPE_ALL" ] && WIPE_QS="?wipe_chat=true"
HDR=""
[ -n "${RESET_TOKEN:-}" ] && HDR="-H X-Reset-Token:${RESET_TOKEN}"

if curl -fsS "${BASE_URL}/api/learned?limit=1" >/dev/null 2>&1; then
  echo "[reset] backend is up -> authoritative POST /api/admin/reset (drains worker)..."
  # shellcheck disable=SC2086
  if curl -fsS -X POST $HDR "${BASE_URL}/api/admin/reset${WIPE_QS}" >/dev/null 2>&1; then
    echo ""
    echo "Reset complete. DUM-E is back to its base, untaught state."
    exit 0
  fi
  echo "[reset] endpoint call failed; falling back to out-of-band wipe below." >&2
fi

# --- Fallback: backend down / endpoint failed -> wipe directly --------------
# Safe because nothing is mid-flight when the backend is down; on next start the
# backend reconciles its weights pointer to the (now-empty) volume.
echo "[modal] clearing learned weights on the volume..."
"$PY" -m modal run modal_app/trainer.py::reset

echo "[db] clearing local learning state..."
WIPE_ALL="$WIPE_ALL" "$PY" - <<'PYEOF'
import os, sqlite3
from backend.app.config import settings

path = settings.DB_PATH
if not os.path.isfile(path):
    print(f"[db] no database at {path}; nothing to clear.")
    raise SystemExit(0)

conn = sqlite3.connect(path)
tables = ["training_pairs", "weights_versions", "learned_feed", "training_jobs", "lessons"]
if os.environ.get("WIPE_ALL"):
    tables += ["messages", "conversations"]

for t in tables:
    try:
        conn.execute(f"DELETE FROM {t}")
    except sqlite3.OperationalError:
        pass  # table may not exist yet
conn.commit()
conn.close()
print(f"[db] cleared: {', '.join(tables)}")
PYEOF

echo ""
echo "Reset complete. DUM-E is back to its base, untaught state."
echo "(Backend was down; it self-reconciles to the empty volume on next start.)"
