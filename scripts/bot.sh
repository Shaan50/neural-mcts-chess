#!/usr/bin/env bash
# Run the Lichess bot from the repo root.
#
#   scripts/bot.sh start        run in this terminal (Ctrl+C to stop)
#   scripts/bot.sh start --bg   run in the background, logging to logs/lichess-bot.log
#   scripts/bot.sh stop         ask a running bot to shut down (same as Ctrl+C)
#   scripts/bot.sh stop --force quit immediately, even mid-game
#   scripts/bot.sh status       local process + Lichess online status
#   scripts/bot.sh logs         follow the background log
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOT_DIR="$ROOT/deploy/lichess-bot/lichess-bot"
CONFIG="$ROOT/deploy/lichess-bot/config.yml"
WEIGHTS="$ROOT/weights/neural-mcts-chess.pt"
LOG="$ROOT/logs/lichess-bot.log"

pids() { pgrep -f "lichess-bot.py" || true; }

check_setup() {
  [ -x "$BOT_DIR/.venv/bin/python" ] || { echo "lichess-bot is not installed: run deploy/setup_lichess_bot.sh"; exit 1; }
  [ -f "$CONFIG" ] || { echo "missing $CONFIG: run deploy/setup_lichess_bot.sh"; exit 1; }
  grep -q '^token: "lip_' "$CONFIG" || { echo "no Lichess token in $CONFIG: export LICHESS_BOT_TOKEN=... and rerun deploy/setup_lichess_bot.sh"; exit 1; }
  [ -f "$WEIGHTS" ] || { echo "missing $WEIGHTS: run 'make export'"; exit 1; }
}

username() {
  local token
  token=$(sed -nE 's/^token: *"?([^"]*)"?.*/\1/p' "$CONFIG")
  curl -s -m 10 -H "Authorization: Bearer $token" https://lichess.org/api/account |
    python3 -c 'import json,sys; print(json.load(sys.stdin).get("username",""))' 2>/dev/null || true
}

case "${1:-}" in
  start)
    check_setup
    if [ -n "$(pids)" ]; then echo "bot is already running (pid $(pids | tr '\n' ' '))"; exit 1; fi
    cd "$BOT_DIR"
    if [ "${2:-}" = "--bg" ]; then
      mkdir -p "$(dirname "$LOG")"
      nohup caffeinate -i .venv/bin/python lichess-bot.py >> "$LOG" 2>&1 &
      sleep 2
      echo "bot started in the background (pid $(pids | tr '\n' ' ')); logs: make bot-logs; stop: make bot-stop"
    else
      echo "starting bot in this terminal (Ctrl+C to stop; your Mac stays awake while it runs)"
      exec caffeinate -i .venv/bin/python lichess-bot.py
    fi
    ;;
  stop)
    p=$(pids)
    if [ -z "$p" ]; then echo "bot is not running"; exit 0; fi
    kill -INT $p
    if [ "${2:-}" = "--force" ]; then sleep 1; kill -INT $p 2>/dev/null || true; fi
    for _ in $(seq 1 30); do [ -z "$(pids)" ] && { echo "bot stopped"; exit 0; }; sleep 1; done
    echo "bot is shutting down but still running (likely finishing a game); 'make bot-stop FORCE=1' quits immediately"
    ;;
  status)
    p=$(pids)
    if [ -n "$p" ]; then echo "local process: running (pid $(echo $p | tr '\n' ' '))"; else echo "local process: not running"; fi
    if [ -f "$CONFIG" ]; then
      name=$(username)
      if [ -n "$name" ]; then
        echo -n "lichess: "
        curl -s -m 10 "https://lichess.org/api/users/status?ids=$name" |
          python3 -c 'import json,sys; d=json.load(sys.stdin)[0]; print(d["name"], "online=%s playing=%s" % (d.get("online", False), d.get("playing", False)), " https://lichess.org/@/" + d["name"])'
      fi
    fi
    ;;
  logs)
    touch "$LOG"
    exec tail -f "$LOG"
    ;;
  *)
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
