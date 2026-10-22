"""Throughput benchmarks: move generation, network inference, batched search.

python -m neural_mcts_chess.tools.bench [--weights runs/sl_medium/best.pt] [--out results/bench.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import chess
import numpy as np

from neural_mcts_chess.game import PyPosition, cpp_available, make_position

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


def _py_perft(board: chess.Board, depth: int) -> int:
    if depth == 1:
        return board.legal_moves.count()
    n = 0
    for m in board.legal_moves:
        board.push(m)
        n += _py_perft(board, depth - 1)
        board.pop()
    return n


def bench_movegen() -> dict:
    out = {}
    t = time.perf_counter()
    n = _py_perft(chess.Board(KIWIPETE), 3)
    out["python_chess_perft_nps"] = n / (time.perf_counter() - t)
    if cpp_available():
        from neural_mcts_chess import _core

        t = time.perf_counter()
        n = _core.Position(KIWIPETE).perft(4)
        out["cpp_perft_nps"] = n / (time.perf_counter() - t)
    # Realistic search workload: legal moves + push/pop + encode for random playouts.
    rng = np.random.default_rng(0)
    for name, backend in [("python", "python"), ("cpp", "cpp")]:
        if backend == "cpp" and not cpp_available():
            continue
        pos = make_position(chess.STARTING_FEN, backend)
        t, ops = time.perf_counter(), 0
        while time.perf_counter() - t < 1.0:
            for _ in range(40):
                moves = pos.legal_moves()
                if not moves or pos.terminal():
                    break
                pos.encode()
                pos.push(moves[int(rng.integers(len(moves)))])
                ops += 1
            while pos.ply() > 0:
                pos.pop()
        out[f"{name}_node_ops_per_s"] = ops / (time.perf_counter() - t)
    return out


def bench_network(weights: str, batch_sizes=(1, 16, 64, 256, 1024)) -> dict:
    from neural_mcts_chess.model.inference import Evaluator

    ev = Evaluator.from_checkpoint(weights)
    ev.warmup(max(batch_sizes))
    pos = PyPosition()
    bb, meta = pos.encode()
    legal = pos.policy_indices(pos.legal_moves())
    out = {"device": str(ev.device)}
    for b in batch_sizes:
        bbs = np.repeat(bb[None], b, 0).view(np.int64)
        metas = np.repeat(meta[None], b, 0)
        flat = (np.arange(b)[:, None] * 4672 + legal[None]).reshape(-1)
        t, n = time.perf_counter(), 0
        while time.perf_counter() - t < 1.0:
            ev(bbs, metas, flat)
            n += b
        out[f"nn_evals_per_s_b{b}"] = n / (time.perf_counter() - t)
    return out


def bench_search(weights: str) -> dict:
    from neural_mcts_chess.mcts.pysearch import SearchParams
    from neural_mcts_chess.mcts.search import make_search, run_batched
    from neural_mcts_chess.model.inference import Evaluator

    ev = Evaluator.from_checkpoint(weights)
    ev.warmup(1024)
    out = {}
    for games, leaves, pipeline in [(1, 16, False), (64, 8, False), (256, 4, False), (256, 4, True)]:
        searches = []
        for _ in range(games):
            s = make_search(SearchParams())
            s.set_position(make_position())
            s.begin(False)
            searches.append(s)
        visits = 800 if games == 1 else 200
        t = time.perf_counter()
        st = run_batched(searches, ev, visits, leaves, pipeline=pipeline)
        el = time.perf_counter() - t
        out[f"search_g{games}_l{leaves}{'_pipe' if pipeline else ''}_visits_per_s"] = (
            sum(s.root_visits() for s in searches) / el
        )
        out[f"search_g{games}_l{leaves}{'_pipe' if pipeline else ''}_mean_batch"] = st.evals / max(st.batches, 1)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default=None)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    res = {"movegen": bench_movegen()}
    if a.weights:
        res["network"] = bench_network(a.weights)
        res["search"] = bench_search(a.weights)
    print(json.dumps(res, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
