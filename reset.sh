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

# --- 1. Modal volume: clear learned weights -------------------------------
echo "[modal] clearing learned weights on the volume..."
"$PY" -m modal run modal_app/trainer.py::reset

# --- 2. Local DB: clear learning state ------------------------------------
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
echo "(If the backend is running, restart it so it does not hold stale state.)"
