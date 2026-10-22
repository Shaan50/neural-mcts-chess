# Reproduction pipeline. `make help` lists targets. Variables can be overridden:
#   make data MONTH=2025-02 BYTES=6G
PY      ?= .venv/bin/python
MONTH   ?= 2025-01
BYTES   ?= 3G
SF      ?= bin/stockfish

.PHONY: help setup build test lint data train-tiny train-small train-medium selfplay accuracy ablations tactics plots report export bench uci

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

setup: ## create venv, install deps, build C++ core
	python3 -m venv .venv && $(PY) -m pip install -U pip && $(PY) -m pip install -e ".[dev]"
	$(MAKE) build

build: ## compile the C++ extension in place
	$(PY) setup.py build_ext --inplace

test: ## run the test suite
	$(PY) -m pytest -q -p no:warnings

lint:
	$(PY) -m ruff check azchess tests && $(PY) -m ruff format --check azchess tests

data: ## download a Lichess month prefix, filter 2000+ games, build shards
	$(PY) -m azchess.data.download --month $(MONTH) --max-bytes $(BYTES)
	$(PY) -m azchess.data.filter data/raw/lichess_$(MONTH).pgn.zst data/raw/lichess_$(MONTH)_2000plus.pgn.zst
	$(PY) -m azchess.data.build data/raw/lichess_$(MONTH)_2000plus.pgn.zst data/shards

train-tiny: ## supervised 4x32
	$(PY) -m azchess.train.supervised --config configs/sl_tiny.yaml
train-small: ## supervised 6x64
	$(PY) -m azchess.train.supervised --config configs/sl_small.yaml
train-medium: ## supervised 10x128 (main network)
	$(PY) -m azchess.train.supervised --config configs/sl_medium.yaml

selfplay: ## self-play refinement loop with gating
	$(PY) -m azchess.selfplay.loop --config configs/selfplay.yaml

accuracy: ## held-out move-prediction accuracy
	$(PY) -m azchess.experiments.accuracy runs/sl_tiny/best.pt runs/sl_small/best.pt runs/sl_medium/best.pt

ablations: ## Elo ablations vs Stockfish (search, size, training step, self-play)
	$(PY) -m azchess.experiments.ablations sims --model runs/sl_medium/best.pt --sims 0 16 64 256 800
	$(PY) -m azchess.experiments.ablations size --models tiny=runs/sl_tiny/best.pt small=runs/sl_small/best.pt medium=runs/sl_medium/best.pt --sims 0 256
	$(PY) -m azchess.experiments.ablations steps --run runs/sl_medium --sims 0
	$(PY) -m azchess.experiments.ablations selfplay --sl runs/sl_medium/best.pt --loop runs/selfplay --sims 256

tactics: ## Win At Chess solve rate vs visits
	$(PY) -m azchess.experiments.tactics medium=runs/sl_medium/best.pt --sims 0 64 256 800 3200

plots: ## render docs/figures
	$(PY) -m azchess.experiments.plots
report: ## markdown tables for the README
	$(PY) -m azchess.experiments.report > results/RESULTS.md

export: ## export deployable weights
	$(PY) -m azchess.tools.export runs/sl_medium/best.pt weights/azchess.pt --half
bench: ## throughput benchmarks
	$(PY) -m azchess.tools.bench --weights runs/sl_medium/best.pt --out results/bench.json
uci: ## run the UCI engine
	$(PY) -m azchess.uci --weights weights/azchess.pt
