#!/usr/bin/env bash
# Clone lichess-bot next to this directory, install it into its own venv and
# point it at the neural-mcts-chess UCI engine.
#
#   export LICHESS_BOT_TOKEN=...      # token of a Lichess *BOT* account (scope bot:play)
#   deploy/setup_lichess_bot.sh
#   cd deploy/lichess-bot/lichess-bot && .venv/bin/python lichess-bot.py
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/lichess-bot/lichess-bot"
REF="${LICHESS_BOT_REF:-master}"
if [ ! -d "$DEST" ]; then
  git clone --depth 1 --branch "$REF" https://github.com/lichess-bot-devs/lichess-bot.git "$DEST"
fi
python3 -m venv "$DEST/.venv"
"$DEST/.venv/bin/pip" install -q -r "$DEST/requirements.txt"
CFG="$HERE/lichess-bot/config.yml"
[ -f "$CFG" ] || cp "$HERE/lichess-bot/config.yml.example" "$CFG"
if [ -n "${LICHESS_BOT_TOKEN:-}" ]; then
  sed -i.bak "s|^token: .*|token: \"$LICHESS_BOT_TOKEN\"|" "$CFG" && rm -f "$CFG.bak"
fi
ln -sf "$CFG" "$DEST/config.yml"
[ -f "$HERE/../weights/neural-mcts-chess.pt" ] || echo "warning: weights/neural-mcts-chess.pt missing - run: neural-mcts-chess export runs/<run>/best.pt weights/neural-mcts-chess.pt --half"
echo "lichess-bot ready. Start with: cd $DEST && .venv/bin/python lichess-bot.py"
