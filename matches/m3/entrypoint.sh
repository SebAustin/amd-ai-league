#!/usr/bin/env bash
# Start the resident model server in the background, then run the given command
# (or stay alive so the harness can `docker exec` app.py calls into a warm container).
set -euo pipefail

SOCKET="${M4_MODEL_SOCKET:-/tmp/amdleague-model.sock}"
if [[ "${M3_DISABLE_LLM:-0}" != "1" && ! -S "$SOCKET" ]]; then
  # Take the start lock first so app.py's lazy autostart never launches a second copy.
  LOCK="${SOCKET%.sock}.starting"
  touch "$LOCK"
  python3 -m amdleague.common.model_server >>/tmp/model_server.log 2>&1 &
  echo $! > "$LOCK"  # lets clients detect a dead loader (stale lock)
fi

if [[ $# -eq 0 ]]; then
  exec sleep infinity
fi
exec "$@"
