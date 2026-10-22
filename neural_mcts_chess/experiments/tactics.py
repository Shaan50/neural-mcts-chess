"""Tactical test suite: fraction of Win At Chess (WAC, 300 positions) solved.

All positions are searched simultaneously (one batched MCTS) at each visit
budget; a position counts as solved when the chosen move is one of the EPD's
``bm`` (best move) entries. Budget 0 is the raw policy's top move.

    python -m neural_mcts_chess.experiments.tactics runs/sl_medium/best.pt --sims 0 64 256 800 3200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import chess
import numpy as np
import requests

from neural_mcts_chess.game import make_position
from neural_mcts_chess.mcts.pysearch import SearchParams
from neural_mcts_chess.mcts.search import choose_move, make_search, run_batched
from neural_mcts_chess.model.inference import Evaluator, legal_flat_from_lists

WAC_URL = "https://raw.githubusercontent.com/jdart1/arasan-chess/master/tests/wacnew.epd"


def load_epd(path: Path) -> list[tuple[str, set[str], str]]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(requests.get(WAC_URL, timeout=60).text)
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        board, ops = chess.Board.from_epd(line)
        best = {m.uci() for m in ops.get("bm", [])}
        if best:
            out.append((board.fen(), best, ops.get("id", "")))
    return out


def solve(evaluator: Evaluator, suite, sims: int, leaves: int = 8) -> list[bool]:
    positions = [make_position(fen) for fen, _, _ in suite]
    if sims <= 1:
        encs = [p.encode() for p in positions]
        moves = [p.legal_moves() for p in positions]
        idx = [p.policy_indices(m) for p, m in zip(positions, moves)]
        bb = np.stack([e[0] for e in encs]).view(np.int64)
        meta = np.stack([e[1] for e in encs])
        logits, _ = evaluator(bb, meta, legal_flat_from_lists(idx))
        chosen, off = [], 0
        for p, ms in zip(positions, moves):
            chosen.append(p.uci(ms[int(np.argmax(logits[off : off + len(ms)]))]))
            off += len(ms)
    else:
        searches = []
        for p in positions:
            s = make_search(SearchParams())
            s.set_position(p)
            s.begin(False)
            searches.append(s)
        run_batched(searches, evaluator, sims, leaves, pipeline=True)
        chosen = [s.position().uci(choose_move(s)) for s in searches]
    return [c in best for c, (_, best, _) in zip(chosen, suite)]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", help="checkpoint paths (name=path to label)")
    p.add_argument("--sims", type=int, nargs="+", default=[0, 64, 256, 800, 3200])
    p.add_argument("--epd", type=Path, default=Path("data/epd/wac.epd"))
    p.add_argument("--out", type=Path, default=Path("results/tactics.json"))
    a = p.parse_args(argv)
    suite = load_epd(a.epd)
    res = json.loads(a.out.read_text()) if a.out.exists() else {}
    for item in a.models:
        name, _, path = item.rpartition("=") if "=" in item else (item, "", item)
        ev = Evaluator.from_checkpoint(path)
        ev.warmup(1024)
        for sims in a.sims:
            t = time.time()
            ok = solve(ev, suite, sims)
            res.setdefault(name, {})[str(sims)] = {"solved": int(sum(ok)), "total": len(ok), "seconds": time.time() - t}
            print(f"[tactics] {name} sims={sims}: {sum(ok)}/{len(ok)} solved ({time.time() - t:.0f}s)", flush=True)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
