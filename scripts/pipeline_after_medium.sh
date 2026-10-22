#!/usr/bin/env bash
# Post-pretraining pipeline, sequenced for a 16 GB machine (one heavy GPU job at a time,
# plus at most one light evaluation process alongside it).
set -euo pipefail
PY=.venv/bin/python
log() { echo "[pipeline $(date +%H:%M:%S)] $*"; }

log "recalibrate BN of all medium/tiny checkpoints"
$PY -m azchess.tools.recalibrate runs/sl_medium/snapshots/*.pt runs/sl_medium/best.pt runs/sl_medium/final.pt \
    runs/sl_tiny/snapshots/*.pt runs/sl_tiny/best.pt runs/sl_tiny/final.pt
$PY -m azchess.tools.export runs/sl_medium/best.pt weights/azchess.pt --half

log "small model training (alone on the GPU) + light evals alongside"
$PY -u -m azchess.train.supervised --config configs/sl_small.yaml workers=3 > logs/sl_small.log 2>&1 &
TRAIN=$!
$PY -m azchess.experiments.accuracy runs/sl_tiny/best.pt runs/sl_medium/best.pt
$PY -m azchess.experiments.tactics medium=runs/sl_medium/best.pt tiny=runs/sl_tiny/best.pt --sims 0 64 256 800 3200
$PY -m azchess.experiments.ablations sims --model runs/sl_medium/best.pt --sims 0 16 64 256 800 --concurrency 16
wait $TRAIN
$PY -m azchess.tools.recalibrate runs/sl_small/snapshots/*.pt runs/sl_small/best.pt runs/sl_small/final.pt
$PY -m azchess.experiments.accuracy runs/sl_small/best.pt

log "self-play reinforcement loop"
$PY -u -m azchess.selfplay.loop --config configs/selfplay.yaml > logs/selfplay.log 2>&1

log "remaining ablations"
$PY -m azchess.experiments.ablations size --models tiny=runs/sl_tiny/best.pt small=runs/sl_small/best.pt medium=runs/sl_medium/best.pt --sims 0 256
$PY -m azchess.experiments.ablations selfplay --sl runs/sl_medium/best.pt --loop runs/selfplay --sims 256 --h2h-games 300
$PY -m azchess.experiments.revalidate runs/sl_tiny runs/sl_small runs/sl_medium
log "done"
