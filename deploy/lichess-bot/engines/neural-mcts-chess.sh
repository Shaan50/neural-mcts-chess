#!/usr/bin/env bash
# Engine executable for lichess-bot: runs the neural-mcts-chess UCI engine from this repo.
# Override NEURAL_MCTS_CHESS_HOME (repo root), NEURAL_MCTS_CHESS_PYTHON or NEURAL_MCTS_CHESS_WEIGHTS if needed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${NEURAL_MCTS_CHESS_HOME:-$(cd "$HERE/../../.." && pwd)}"
PY="${NEURAL_MCTS_CHESS_PYTHON:-$REPO/.venv/bin/python}"
cd "$REPO"
exec "$PY" -m neural_mcts_chess.uci --weights "${NEURAL_MCTS_CHESS_WEIGHTS:-$REPO/weights/neural-mcts-chess.pt}" "$@"
