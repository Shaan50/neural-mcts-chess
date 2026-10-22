#!/usr/bin/env bash
# Engine executable for lichess-bot: runs the AZChess UCI engine from this repo.
# Override AZCHESS_HOME (repo root), AZCHESS_PYTHON or AZCHESS_WEIGHTS if needed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${AZCHESS_HOME:-$(cd "$HERE/../../.." && pwd)}"
PY="${AZCHESS_PYTHON:-$REPO/.venv/bin/python}"
cd "$REPO"
exec "$PY" -m azchess.uci --weights "${AZCHESS_WEIGHTS:-$REPO/weights/azchess.pt}" "$@"
