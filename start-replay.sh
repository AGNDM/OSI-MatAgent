#!/usr/bin/env bash
set -euo pipefail
REPLAY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$REPLAY_ROOT"
for replay_python in "$REPLAY_ROOT/.venv/bin/python" python3 python; do
  if command -v "$replay_python" >/dev/null 2>&1 &&
     "$replay_python" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    exec "$replay_python" -B "$REPLAY_ROOT/tools/launch_replay.py" "$@"
  fi
done
printf '%s\n' 'Python 3.11 or newer is required on macOS/Linux.'
printf '%s\n' 'On Windows, double-click Start-Replay.cmd to install a local Python automatically.' >&2
exit 1
