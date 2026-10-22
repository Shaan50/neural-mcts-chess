# Deploying neural-mcts-chess as a Lichess bot

neural-mcts-chess speaks UCI, so it plugs into the community
[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) bridge, which
handles the Lichess Bot API (challenges, game streams, clocks).

## 1. Export the weights

```bash
make export   # fp16 copy of the best network (self-play best if the loop has run) -> weights/neural-mcts-chess.pt
echo "uci\nisready\nposition startpos\ngo movetime 1000\nquit" | python -m neural_mcts_chess.uci   # smoke test
```

## 2. Create a bot account (one-time, irreversible)

1. Register a **new** Lichess account that has never played a game.
2. Create a personal API token with the `bot:play` scope at
   <https://lichess.org/account/oauth/token>.
3. Upgrade the account to a BOT account. This cannot be undone:
   `curl -d '' https://lichess.org/api/bot/account/upgrade -H "Authorization: Bearer $LICHESS_BOT_TOKEN"`
   (or `python lichess-bot.py -u` after step 3 below).

## 3. Install and run lichess-bot

```bash
export LICHESS_BOT_TOKEN=lip_xxx
deploy/setup_lichess_bot.sh          # clones lichess-bot, creates its venv, writes config.yml
cd deploy/lichess-bot/lichess-bot && .venv/bin/python lichess-bot.py
```

`deploy/lichess-bot/config.yml.example` accepts standard-chess challenges from
1+0 to 30+20, one game at a time (the GPU search is shared), resigns at a
sustained Q of about -0.92, and offers or accepts draws in dead-equal endgames.
`engines/neural-mcts-chess.sh` launches `python -m neural_mcts_chess.uci` with `weights/neural-mcts-chess.pt`.
Override with `NEURAL_MCTS_CHESS_WEIGHTS=/path/to/net.pt`.

## Day-to-day commands (from the repo root)

```bash
make bot          # run in this terminal; Ctrl+C to stop (keeps the Mac awake)
make bot-bg       # run in the background, logging to logs/lichess-bot.log
make bot-logs     # follow the background log
make bot-status   # local process + whether Lichess shows the bot online
make bot-stop     # graceful stop (same as Ctrl+C); FORCE=1 quits even mid-game
```

## Docker

```bash
docker build -f deploy/Dockerfile -t neural-mcts-chess-bot .
docker run -e LICHESS_BOT_TOKEN=lip_xxx neural-mcts-chess-bot
```

The image builds the C++ extension and uses CPU inference. The 10x128 network
reaches several hundred nodes per second on a few cores, which is enough for blitz
and rapid. For bullet, run on a machine with a GPU (CUDA base image) or Apple silicon.
