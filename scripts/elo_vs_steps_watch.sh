#!/usr/bin/env bash
# Evaluate each medium snapshot (raw policy) as soon as it is written, while training runs.
set -u
PY=.venv/bin/python
while true; do
  running=0; pgrep -f "configs/sl_medium.yaml" >/dev/null && running=1
  $PY -m neural_mcts_chess.tools.recalibrate runs/sl_medium/snapshots/step_*.pt
  $PY -m neural_mcts_chess.experiments.ablations steps --run runs/sl_medium --sims 0 --concurrency 8
  [ $running = 0 ] && break
  sleep 300
done
echo "[watch] done"
