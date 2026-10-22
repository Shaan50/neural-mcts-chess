"""Players for engine matches.

Two kinds of players exist:

* *batched* players (``MCTSPlayer``, ``PolicyPlayer``) choose moves for many
  concurrent games in one call, so their network evaluations share GPU batches;
* *async* players (``StockfishPlayer``, ``RandomPlayer``) answer one game at a
  time, in background threads, while the batched players search.

Every player is told about every move played in each of its games through
``observe`` (the MCTS player uses this to reuse its search tree).
"""

from __future__ import annotations

import os
import random
import shutil
from concurrent.futures import Future, ThreadPoolExecutor

import chess
import chess.engine
import numpy as np

from azchess.game import make_position
from azchess.mcts.pysearch import SearchParams
from azchess.mcts.search import SearchStats, choose_move, make_cache, make_search, run_batched
from azchess.model.inference import Evaluator, legal_flat_from_lists


class MCTSPlayer:
    batched = True

    def __init__(
        self,
        evaluator: Evaluator,
        sims: int = 400,
        params: SearchParams | None = None,
        leaves_per_search: int = 8,
        name: str | None = None,
        backend: str = "auto",
        cache_size: int = 300_000,
        temperature: float = 0.0,
        pipeline: bool = True,
        seed: int = 0,
    ):
        self.evaluator = evaluator
        self.sims = sims
        self.params = params or SearchParams()
        self.leaves = leaves_per_search
        self.name = name or f"mcts{sims}"
        self.backend = backend
        self.cache = make_cache(cache_size, backend)
        self.cache_size = cache_size
        self.temperature = temperature
        self.pipeline = pipeline
        self.rng = np.random.default_rng(seed)
        self.searches: dict[int, object] = {}
        self.stats = SearchStats()

    def new_game(self, gid: int, fen: str, moves: list[str]) -> None:
        pos = make_position(fen, "cpp" if self.backend != "python" else "python")
        for m in moves:
            pos.push_uci(m)
        s = make_search(self.params, self.backend, self.cache, self.cache_size)
        s.set_position(pos)
        self.searches[gid] = s

    def observe(self, gid: int, uci: str) -> None:
        self.searches[gid].advance_uci(uci)

    def select(self, gids: list[int]) -> list[tuple[str, dict]]:
        searches = [self.searches[g] for g in gids]
        for s in searches:
            s.begin(False)
        run_batched(searches, self.evaluator, self.sims, self.leaves, pipeline=self.pipeline, stats=self.stats)
        out = []
        for s in searches:
            mv = choose_move(s, self.temperature, self.rng)
            out.append((s.position().uci(mv), {"q": s.root_q(), "visits": s.root_visits(), "depth": s.mean_depth()}))
        return out

    def end_game(self, gid: int) -> None:
        self.searches.pop(gid, None)

    def close(self) -> None:
        pass


class PolicyPlayer:
    """Plays the network's highest-probability legal move (no search)."""

    batched = True

    def __init__(self, evaluator: Evaluator, name: str = "policy", temperature: float = 0.0, seed: int = 0):
        self.evaluator = evaluator
        self.name = name
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.positions: dict[int, object] = {}

    def new_game(self, gid: int, fen: str, moves: list[str]) -> None:
        pos = make_position(fen)
        for m in moves:
            pos.push_uci(m)
        self.positions[gid] = pos

    def observe(self, gid: int, uci: str) -> None:
        self.positions[gid].push_uci(uci)

    def select(self, gids: list[int]) -> list[tuple[str, dict]]:
        poss = [self.positions[g] for g in gids]
        encs = [p.encode() for p in poss]
        moves = [p.legal_moves() for p in poss]
        idx = [p.policy_indices(m) for p, m in zip(poss, moves)]
        bb = np.stack([e[0] for e in encs]).view(np.int64)
        meta = np.stack([e[1] for e in encs])
        logits, wdl = self.evaluator(bb, meta, legal_flat_from_lists(idx))
        out, off = [], 0
        for p, ms, w in zip(poss, moves, wdl):
            lg = logits[off : off + len(ms)]
            off += len(ms)
            if self.temperature > 0:
                z = np.exp((lg - lg.max()) / self.temperature)
                k = int(self.rng.choice(len(ms), p=z / z.sum()))
            else:
                k = int(np.argmax(lg))
            out.append((p.uci(ms[k]), {"q": float(w[0] - w[2])}))
        return out

    def end_game(self, gid: int) -> None:
        self.positions.pop(gid, None)

    def close(self) -> None:
        pass


def find_stockfish(path: str | None = None) -> str:
    for cand in [path, os.environ.get("STOCKFISH_PATH"), "bin/stockfish", shutil.which("stockfish")]:
        if cand and os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "Stockfish binary not found: pass --stockfish, set STOCKFISH_PATH, or put it in bin/stockfish"
    )


class StockfishPlayer:
    """Stockfish weakened with UCI_LimitStrength/UCI_Elo or Skill Level.

    Stockfish's UCI_Elo scale is calibrated against CCRL 40/4 ratings (anchor:
    Stash engines at 120s+1s); it bottoms out at 1320.
    """

    batched = False

    def __init__(
        self,
        elo: int | None = None,
        skill: int | None = None,
        movetime: float | None = 0.1,
        nodes: int | None = None,
        depth: int | None = None,
        path: str | None = None,
        threads: int = 1,
        hash_mb: int = 16,
        name: str | None = None,
        max_workers: int = 16,
    ):
        self.path = find_stockfish(path)
        self.elo, self.skill = elo, skill
        self.limit = chess.engine.Limit(
            time=movetime if nodes is None and depth is None else None, nodes=nodes, depth=depth
        )
        self.options = {"Threads": threads, "Hash": hash_mb}
        if elo is not None:
            self.options.update({"UCI_LimitStrength": True, "UCI_Elo": int(elo)})
        elif skill is not None:
            self.options["Skill Level"] = int(skill)
        self.name = name or (f"sf_elo{elo}" if elo else f"sf_skill{skill}" if skill is not None else "stockfish")
        self.pool = ThreadPoolExecutor(max_workers)
        self.free: list[chess.engine.SimpleEngine] = []
        self.engines: dict[int, chess.engine.SimpleEngine] = {}
        self.boards: dict[int, chess.Board] = {}

    def _engine(self) -> chess.engine.SimpleEngine:
        if self.free:
            return self.free.pop()
        eng = chess.engine.SimpleEngine.popen_uci(self.path)
        eng.configure(self.options)
        return eng

    def new_game(self, gid: int, fen: str, moves: list[str]) -> None:
        board = chess.Board(fen)
        for m in moves:
            board.push_uci(m)
        self.boards[gid] = board
        self.engines[gid] = self._engine()

    def observe(self, gid: int, uci: str) -> None:
        self.boards[gid].push_uci(uci)

    def select_async(self, gid: int) -> Future:
        board, eng = self.boards[gid].copy(), self.engines[gid]
        return self.pool.submit(lambda: (eng.play(board, self.limit, game=gid).move.uci(), {}))

    def end_game(self, gid: int) -> None:
        self.boards.pop(gid, None)
        eng = self.engines.pop(gid, None)
        if eng is not None:
            self.free.append(eng)

    def close(self) -> None:
        for eng in self.free + list(self.engines.values()):
            try:
                eng.quit()
            except Exception:  # noqa: BLE001 - engine already gone
                pass
        self.free, self.engines = [], {}
        self.pool.shutdown(wait=False)


class RandomPlayer:
    batched = False

    def __init__(self, name: str = "random", seed: int = 0):
        self.name = name
        self.rng = random.Random(seed)
        self.boards: dict[int, chess.Board] = {}
        self.pool = ThreadPoolExecutor(1)

    def new_game(self, gid: int, fen: str, moves: list[str]) -> None:
        b = chess.Board(fen)
        for m in moves:
            b.push_uci(m)
        self.boards[gid] = b

    def observe(self, gid: int, uci: str) -> None:
        self.boards[gid].push_uci(uci)

    def select_async(self, gid: int) -> Future:
        mv = self.rng.choice(list(self.boards[gid].legal_moves)).uci()
        f: Future = Future()
        f.set_result((mv, {}))
        return f

    def end_game(self, gid: int) -> None:
        self.boards.pop(gid, None)

    def close(self) -> None:
        self.pool.shutdown(wait=False)


def parse_spec(spec: str) -> tuple[str, dict]:
    """'kind:key=value,key=value' -> (kind, {key: value})."""
    kind, _, rest = spec.partition(":")
    kv = {}
    for part in filter(None, rest.split(",")):
        k, _, v = part.partition("=")
        try:
            kv[k] = int(v)
        except ValueError:
            try:
                kv[k] = float(v)
            except ValueError:
                kv[k] = v
    return kind, kv


def build_player(spec: str, device: str = "auto", evaluators: dict | None = None):
    """Build a player from a spec string, e.g.::

    nn:path=runs/medium/best.pt,sims=400      MCTS with 400 visits per move
    nn:path=runs/medium/best.pt,sims=0        raw policy (no search)
    nn:path=...,ema=0                         use the raw (non-EMA) weights
    sf:elo=1500,movetime=0.1                  Stockfish at UCI_Elo 1500
    sf:skill=3,depth=5                        Stockfish skill level 3
    random
    """
    kind, kv = parse_spec(spec)
    evaluators = evaluators if evaluators is not None else {}
    if kind == "nn":
        path = kv.pop("path")
        use_ema = bool(kv.pop("ema", 1))
        key = path if use_ema else f"{path}#raw"
        if key not in evaluators:
            evaluators[key] = Evaluator.from_checkpoint(path, device, use_ema=use_ema)
        path = key
        sims = kv.pop("sims", 400)
        name = kv.pop("name", None)
        if sims <= 1:
            return PolicyPlayer(evaluators[path], name=name or "policy", temperature=kv.pop("temperature", 0.0))
        params = SearchParams(**{k: kv.pop(k) for k in list(kv) if k in SearchParams.__dataclass_fields__})
        return MCTSPlayer(evaluators[path], sims=sims, params=params, name=name or f"mcts{sims}", **kv)
    if kind == "sf":
        return StockfishPlayer(**kv)
    if kind == "random":
        return RandomPlayer(**kv)
    raise ValueError(f"unknown player spec {spec!r}")
