# AZChess: an AlphaZero-style chess engine on a student budget

A residual CNN policy/value network plus batched Monte-Carlo tree search,
pretrained on 540k high-rated Lichess games, refined by self-play with gating,
measured in Elo against calibrated Stockfish, and deployable as a Lichess bot.
Everything was trained on a single laptop GPU (Apple M5, 16 GB, PyTorch MPS).

<!-- RESULTS-SUMMARY -->

## Highlights

- **C++17 core with pybind11 bindings.** A bitboard move generator (64M perft
  nodes/s, 41× python-chess) and an arena-allocated batched PUCT search. The
  pure-Python reference implementation produces **visit counts identical** to
  the C++ search in tests.
- **Batched GPU inference across hundreds of concurrent games.** Virtual loss,
  collision handling, a shared NN-evaluation cache, tree reuse, power-of-two
  batch padding, legal-move logit gathering on the device, and an optional
  double-buffered pipeline that overlaps CPU tree search with GPU inference.
  Tree work takes under 10% of search time, so search is GPU-bound.
- **Data pipeline.** Resumable range download of a 33 GB Lichess dump prefix,
  then a header filter (~145k games/s), then parallel SAN → tensor encoding in
  C++ (~1M positions/s). Positions are stored as a **100-byte bit-packed record**
  (56× smaller than float planes) and expanded to planes **on the GPU**.
- **Training.** fp16/bf16 autocast with loss scaling, EMA weights with decay
  warmup, legal-move-masked policy loss, a WDL value head, cosine LR,
  full-state checkpoint/resume, TensorBoard, JSONL and W&B logging, and
  held-out validation split by game.
- **Self-play RL loop.** Dirichlet noise, a temperature schedule, KataGo
  playout-cap randomization, resignation with a no-resign control group to
  measure false positives, a replay-buffer window mixed with human data, Q+z
  value targets, and gating with SPRT.
- **Evaluation.** A concurrent match runner (Stockfish thinks in background
  threads while all of our games share one GPU batch), paired openings, an
  adaptive Stockfish ladder, an MLE rating with Fisher-information and
  bootstrap CIs, and per-opponent performance with Wilson-style intervals.
- **UCI engine and Lichess bot.** Time management with smart pruning, tree
  reuse across moves, WDL reporting, a lichess-bot config, an engine wrapper,
  a setup script and a Dockerfile.
- **About 100 tests**, run in CI on Linux and macOS. They cover perft on six
  standard positions, C++/python-chess equivalence on more than 5,000 random
  positions, move-index bijection and mirror symmetry, encoding round-trips,
  MCTS invariants, Elo statistics, the data pipeline, training/resume, and
  end-to-end UCI and self-play runs.

## Architecture

```
 Lichess dump (.pgn.zst, 33 GB) ──range download──▶ prefix (3 GB)
        │ header filter: both ≥2000, ≥3+0, normal termination   (145k games/s)
        ▼
 filtered PGN ──multiprocess──▶ C++ encode_game (SAN parse + legal moves + encode)
        ▼
 shards/*.npz: bb[12]×u64 · meta[4]×u8 · move · legal moves · result · ply · elo
        │                                    (split by game hash: train/val/test)
        ▼
 IterableDataset → whole-batch vectorised gather → GPU: bits → 22×8×8 planes
        ▼
 ┌────────────── ResNet (B blocks × F filters, optional SE) ──────────────┐
 │ policy head: conv → 73×8×8 = 4672 logits   value head: conv → FC → W/D/L │
 └─────────────────────────────────────────────────────────────────────────┘
        │ supervised: CE(masked policy, human move) + 0.5·CE(WDL, result)
        ▼
 Batched MCTS (C++) ◀──── evaluator (one GPU call per batch of leaves) ────┐
   gather leaves from N trees (virtual loss) ──▶ NN ──▶ expand + backup ───┘
        │
        ├──▶ self-play (noise, PCR, resign) ──▶ replay buffer ──▶ retrain ──▶ gate vs best
        ├──▶ match runner vs Stockfish ladder ──▶ MLE Elo ± CI, ablations, charts
        └──▶ UCI engine ──▶ lichess-bot ──▶ lichess.org
```

### Encodings

**Moves.** AlphaZero's 8×8×73 scheme: 56 queen-move planes (8 directions × 7
distances), 8 knight planes and 9 under-promotion planes. Index = `plane·64 +
from_square`, which matches a convolutional policy head flattened in
plane-major order. Squares are oriented to the side to move, with the board
mirrored vertically for black, so the network always plays "up the board".
1,858 of the 4,672 slots are geometrically reachable, the same count Lc0 uses.

**Board.** 22 planes: our/their P N B R Q K (12), two repetition planes, an
en-passant square (only when the capture is legal), four castling planes, a
side-to-move colour plane, a halfmove clock / 100 plane, and a constant plane
that marks the board edge for zero-padded convolutions. The compact form
(12 bitboards + 4 bytes) is what is stored, batched and sent to the GPU.

**Game rules** (identical in both backends, checked in this order):
checkmate/stalemate, insufficient material (a port of python-chess's rules),
the fifty-move rule and threefold repetition, both applied automatically. Inside the search tree, a
repetition of a position reached *after* the root scores as a draw (the
Stockfish/Lc0 two-fold rule).

### Search

PUCT with AlphaZero's growing exploration constant
`c(N) = c_init + log((N + c_base + 1)/c_base)`, Lc0-style first-play urgency
(`Q_parent − r·√(visited policy)`), virtual loss for in-flight leaves, and
collision detection. Children are stored contiguously in a flat arena. When
the tree is reused between moves, the chosen subtree is copied breadth-first
into a fresh arena. The NN cache key includes the halfmove clock and
repetition count, because the network sees them.

`gather_batch` walks every game's tree and returns encoded leaves plus flat
legal-move indices. The evaluator runs one forward pass, gathers only the
legal logits on the device and returns ~35 floats per position.
`apply_batch` expands and backs up the results. Both release the GIL.

<!-- RESULTS -->

## Reproducing

```bash
# environment (Apple silicon: make sure the Python is arm64, otherwise MPS is unavailable)
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" && .venv/bin/python setup.py build_ext --inplace
mkdir -p bin && cp $(which stockfish) bin/stockfish     # or download Stockfish 19
make test

make data MONTH=2025-01 BYTES=3G          # ~30 min download at 2 MB/s, <2 min processing
make train-tiny train-small train-medium  # 30k steps × 1024 each
make accuracy                              # held-out move prediction
make selfplay                              # 5 generations of self-play + gating
make ablations                             # Elo vs Stockfish ladders
make plots report export
make uci                                   # or: deploy/setup_lichess_bot.sh
```

All commands are also available as `azchess <command>` (`azchess --help`).
Every long-running step checkpoints and resumes: downloads use HTTP ranges,
training restores the full optimizer/scheduler/scaler/RNG state, the self-play
loop keeps a phase-level `state.json`, and ablation results are written per
configuration.

### Compute budget

| Step | Hardware | Wall-clock |
|---|---|---|
<!-- COMPUTE -->

`configs/sl_large_cuda.yaml` is the config for a rented GPU: a 15×192
SE-ResNet in bf16 over about 200M positions. At $0.5–1/h for an RTX 4090-class
card, it fits the < $100 budget together with a longer self-play phase.

## Project layout

```
csrc/                 C++17: bitboard move generator, batched MCTS, pybind11 bindings
azchess/
  encoding/           move (8x8x73) and board (22-plane / 100-byte) encodings
  game.py             python-chess backend + backend factory
  data/               download, header filter, shard builder, shard format, batch loader
  model/              ResNet (SE optional, WDL/scalar heads), batched GPU evaluator
  train/              losses, trainer (AMP, EMA, resume, logging), supervised entry point
  mcts/               Python reference search, backend-agnostic batched driver
  selfplay/           concurrent self-play generator, generation loop with gating
  eval/               players, match runner, openings, Elo/SPRT statistics, eval CLI
  experiments/        ablations, held-out accuracy, plots, report tables
  uci/                UCI engine
  tools/              export (fp16 / TorchScript / ONNX), benchmarks
configs/              training / self-play configs
deploy/               lichess-bot config, engine wrapper, setup script, Dockerfile
tests/                pytest suite (~100 tests)
```

## Limitations and honest caveats

<!-- CAVEATS -->
