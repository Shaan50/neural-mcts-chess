"""UCI protocol front-end.

Supports ``uci``, ``isready``, ``setoption``, ``ucinewgame``, ``position``,
``go`` (wtime/btime/winc/binc/movestogo/movetime/nodes/infinite), ``stop``,
``quit``, plus the non-standard ``d`` (show position) and ``bench``.

The search runs in a background thread so ``stop`` is honoured immediately.
The tree is reused between moves whenever the new ``position`` command
extends the previous one. Scores are reported both as centipawns (Lc0's
Q -> cp mapping) and, with ``UCI_ShowWDL``, as win/draw/loss permille.

    python -m azchess.uci --weights weights/azchess.pt
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from dataclasses import dataclass

from azchess.game import make_position
from azchess.mcts.pysearch import SearchParams
from azchess.mcts.search import SearchStats, choose_move, make_cache, make_search, root_policy, run_batched

ENGINE_NAME = "AZChess 0.1"
STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def q_to_cp(q: float) -> int:
    q = max(-0.999, min(0.999, q))
    return int(round(90.0 * math.tan(1.5637541897 * q)))


@dataclass
class Option:
    type: str
    default: object
    min: int | None = None
    max: int | None = None

    def uci(self, name: str) -> str:
        s = f"option name {name} type {self.type} default {self.default if self.type != 'check' else str(self.default).lower()}"
        if self.min is not None:
            s += f" min {self.min} max {self.max}"
        return s


OPTIONS = {
    "WeightsFile": Option("string", os.environ.get("AZCHESS_WEIGHTS", "weights/azchess.pt")),
    "Device": Option("string", "auto"),
    "Nodes": Option("spin", 0, 0, 10_000_000),  # 0 = use the clock
    "MinibatchSize": Option("spin", 16, 1, 256),
    "CPuct": Option("string", "1.25"),
    "FPUReduction": Option("string", "0.25"),
    "PolicyTemperature": Option("string", "1.0"),
    "NNCacheSize": Option("spin", 200_000, 0, 10_000_000),
    "MoveOverhead": Option("spin", 100, 0, 5000),
    "SlowMover": Option("spin", 100, 10, 1000),
    "SmartPruning": Option("check", True),
    "UCI_ShowWDL": Option("check", True),
}


class UCIEngine:
    def __init__(self, out=None, weights: str | None = None):
        self.out = out or sys.stdout
        self.opts = {k: v.default for k, v in OPTIONS.items()}
        if weights:
            self.opts["WeightsFile"] = weights
        self.evaluator = None
        self.cache = None
        self.search = None
        self.root_fen = STARTPOS
        self.moves: list[str] = []
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    # ----------------------------------------------------------------- io
    def send(self, msg: str) -> None:
        with self.lock:
            self.out.write(msg + "\n")
            self.out.flush()

    def ensure_loaded(self) -> None:
        if self.evaluator is None:
            from azchess.model.inference import Evaluator

            self.evaluator = Evaluator.from_checkpoint(self.opts["WeightsFile"], self.opts["Device"])
            self.cache = make_cache(int(self.opts["NNCacheSize"]))
            t0 = time.perf_counter()
            self.evaluator.warmup(max_batch=max(256, 2 * int(self.opts["MinibatchSize"])))
            self.send(
                f"info string loaded {self.opts['WeightsFile']} on {self.evaluator.device} (warmup {time.perf_counter() - t0:.1f}s)"
            )

    def params(self) -> SearchParams:
        return SearchParams(
            cpuct_init=float(self.opts["CPuct"]),
            fpu_reduction=float(self.opts["FPUReduction"]),
            policy_temperature=float(self.opts["PolicyTemperature"]),
        )

    # ----------------------------------------------------------- commands
    def handle(self, line: str) -> bool:
        tokens = line.strip().split()
        if not tokens:
            return True
        cmd, args = tokens[0], tokens[1:]
        if cmd == "uci":
            self.send(f"id name {ENGINE_NAME}")
            self.send("id author Shaan Cheruvu")
            for name, opt in OPTIONS.items():
                self.send(opt.uci(name))
            self.send("uciok")
        elif cmd == "isready":
            self.ensure_loaded()
            self.send("readyok")
        elif cmd == "setoption":
            self.cmd_setoption(args)
        elif cmd == "ucinewgame":
            self.wait()
            self.search = None
            if self.cache is not None:
                self.cache.clear()
        elif cmd == "position":
            self.wait()
            self.cmd_position(args)
        elif cmd == "go":
            self.wait()
            self.cmd_go(args)
        elif cmd == "stop":
            self.stop_event.set()
            self.wait()
        elif cmd == "ponderhit":
            pass
        elif cmd == "quit":
            self.stop_event.set()
            self.wait()
            return False
        elif cmd == "d":
            self.send(f"info string fen {self.current_position().fen()}")
        elif cmd == "bench":
            self.cmd_bench()
        elif cmd == "debug":
            pass
        else:
            self.send(f"info string unknown command {cmd}")
        return True

    def wait(self) -> None:
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def cmd_setoption(self, args: list[str]) -> None:
        if "name" not in args:
            return
        i = args.index("name")
        j = args.index("value") if "value" in args else len(args)
        name = " ".join(args[i + 1 : j])
        value = " ".join(args[j + 1 :])
        key = next((k for k in OPTIONS if k.lower() == name.lower()), None)
        if key is None:
            self.send(f"info string unknown option {name}")
            return
        opt = OPTIONS[key]
        if opt.type == "spin":
            self.opts[key] = max(opt.min, min(opt.max, int(value)))
        elif opt.type == "check":
            self.opts[key] = value.lower() == "true"
        else:
            self.opts[key] = value
        if key in ("WeightsFile", "Device"):
            self.evaluator = None
            self.search = None
        if key in ("CPuct", "FPUReduction", "PolicyTemperature"):
            self.search = None

    def cmd_position(self, args: list[str]) -> None:
        if not args:
            return
        if args[0] == "startpos":
            fen, rest = STARTPOS, args[1:]
        elif args[0] == "fen":
            k = args.index("moves") if "moves" in args else len(args)
            fen, rest = " ".join(args[1:k]), args[k:]
        else:
            return
        moves = rest[1:] if rest and rest[0] == "moves" else []
        # Reuse the tree if the new position extends the previous one.
        if self.search is not None and fen == self.root_fen and moves[: len(self.moves)] == self.moves:
            for m in moves[len(self.moves) :]:
                self.search.advance_uci(m)
        else:
            self.search = None
        self.root_fen, self.moves = fen, moves

    def current_position(self):
        pos = make_position(self.root_fen)
        for m in self.moves:
            pos.push_uci(m)
        return pos

    def _budget(self, limits: dict, white: bool) -> float | None:
        if "movetime" in limits:
            return max(0.01, limits["movetime"] / 1000 - self.opts["MoveOverhead"] / 1000)
        if limits.get("infinite") or (self.opts["Nodes"] and not any(k in limits for k in ("wtime", "btime"))):
            return None
        remaining = limits.get("wtime" if white else "btime")
        if remaining is None:
            return None if limits.get("nodes") or self.opts["Nodes"] else 1.0
        inc = limits.get("winc" if white else "binc", 0)
        mtg = limits.get("movestogo") or max(20, 40 - len(self.moves) // 4)
        t = (remaining / mtg + 0.75 * inc) * self.opts["SlowMover"] / 100
        t = min(t, remaining * 0.25) - self.opts["MoveOverhead"]
        return max(0.02, t / 1000)

    def cmd_go(self, args: list[str]) -> None:
        self.ensure_loaded()
        limits: dict = {}
        i = 0
        while i < len(args):
            a = args[i]
            if a in ("wtime", "btime", "winc", "binc", "movestogo", "movetime", "nodes", "depth", "mate"):
                limits[a] = int(args[i + 1])
                i += 2
            else:
                limits[a] = True
                i += 1
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._search_and_report, args=(limits,), daemon=True)
        self.thread.start()

    def _search_and_report(self, limits: dict) -> None:
        try:
            self._search_and_report_inner(limits)
        except Exception as e:  # noqa: BLE001 - never leave the GUI without a bestmove
            self.send(f"info string search error: {e!r}")
            pos = self.current_position()
            legal = pos.legal_moves()
            self.send(f"bestmove {pos.uci(legal[0]) if legal else '0000'}")

    def _search_and_report_inner(self, limits: dict) -> None:
        pos = self.current_position()
        legal = pos.legal_moves()
        if not legal or pos.terminal():
            self.send("bestmove 0000")
            return
        if self.search is None:
            self.search = make_search(self.params(), cache=self.cache)
            self.search.set_position(pos)
        s = self.search
        s.begin(False)
        start = time.perf_counter()
        budget = self._budget(limits, pos.turn)
        node_limit = limits.get("nodes") or self.opts["Nodes"] or 10**9
        target = s.root_visits() + node_limit if limits.get("nodes") or self.opts["Nodes"] else 10**9
        deadline = None if budget is None else start + budget
        stats = SearchStats()
        visits0 = s.root_visits()
        last_info = start
        while not self.stop_event.is_set():
            now = time.perf_counter()
            chunk_end = now + 0.25
            if deadline is not None:
                chunk_end = min(chunk_end, deadline)
            run_batched(
                [s],
                self.evaluator,
                target,
                int(self.opts["MinibatchSize"]),
                deadline=chunk_end,
                stop=self.stop_event,
                stats=stats,
            )
            now = time.perf_counter()
            if len(legal) == 1 and not limits.get("infinite"):
                break
            if s.done(target) or (deadline is not None and now >= deadline):
                break
            if (
                deadline is not None
                and self.opts["SmartPruning"]
                and now - start > 0.1 * budget
                and s.root_visits() - visits0 >= 32
                and self._cannot_change(s, stats, deadline - now)
            ):
                break
            if now - last_info >= 1.0:
                self._info(s, stats, now - start, visits0)
                last_info = now
        if limits.get("infinite"):
            self.stop_event.wait()
        elapsed = time.perf_counter() - start
        self._info(s, stats, elapsed, visits0)
        mv = choose_move(s) if s.root_children() else legal[0]
        self.send(f"bestmove {s.position().uci(mv)}")

    @staticmethod
    def _cannot_change(s, stats: SearchStats, remaining: float) -> bool:
        """Lc0-style smart pruning: stop when the runner-up cannot catch up in the remaining time."""
        _m, _i, visits, _q = root_policy(s)
        if len(visits) < 2 or stats.seconds <= 0.05:
            return False
        top2 = sorted(visits)[-2:]
        return top2[1] - top2[0] > stats.evals_per_s * remaining * 1.1

    def _info(self, s, stats: SearchStats, elapsed: float, visits0: int) -> None:
        q = s.root_q()
        d = s.root_d()
        w = max(0.0, (1 - d + q) / 2)
        loss = max(0.0, (1 - d - q) / 2)
        nodes = s.root_visits()
        pos = s.position()
        pv = " ".join(_pv_uci(pos, s.pv(16)))
        msg = (
            f"info depth {max(1, round(s.mean_depth()))} seldepth {s.max_depth()} nodes {nodes} "
            f"nps {int((nodes - visits0) / max(elapsed, 1e-3))} score cp {q_to_cp(q)}"
        )
        if self.opts["UCI_ShowWDL"]:
            msg += f" wdl {round(w * 1000)} {round(d * 1000)} {round(loss * 1000)}"
        msg += f" time {int(elapsed * 1000)} pv {pv}"
        self.send(msg)

    def cmd_bench(self) -> None:
        self.ensure_loaded()
        fens = [
            STARTPOS,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
        ]
        total, t0 = 0, time.perf_counter()
        for fen in fens:
            s = make_search(self.params(), cache=None)
            s.set_position(make_position(fen))
            s.begin(False)
            run_batched([s], self.evaluator, 2000, int(self.opts["MinibatchSize"]))
            total += s.root_visits()
        el = time.perf_counter() - t0
        self.send(f"info string bench nodes {total} time {el:.2f}s nps {total / el:.0f}")


def _pv_uci(pos, moves) -> list[str]:
    out = []
    p = pos.copy()
    for m in moves:
        out.append(p.uci(m))
        p.push(m)
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="AZChess UCI engine")
    p.add_argument("--weights", default=None)
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    engine = UCIEngine(weights=a.weights)
    if a.device:
        engine.opts["Device"] = a.device
    for line in sys.stdin:
        if not engine.handle(line):
            break


if __name__ == "__main__":
    main()
