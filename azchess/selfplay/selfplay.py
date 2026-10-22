"""Self-play game generation.

Hundreds of games are played simultaneously; every move step runs one batched
MCTS over all of them so the network sees large batches. Techniques:

* Dirichlet noise at the root and visit-count temperature for the opening
  plies (AlphaZero);
* playout cap randomisation (KataGo): only a random ``full_search_prob``
  fraction of moves get a full search, with noise, and become training
  samples; the rest use a cheap search without noise and are not recorded.
  This multiplies the number of games (= value targets) per GPU-hour;
* resignation when the root value stays below ``resign_threshold`` for
  ``resign_moves`` consecutive moves, disabled in a ``no_resign_frac`` control
  group used to measure the false-positive resignation rate (AlphaZero);
* a shared NN cache across games (common openings are evaluated once) and
  tree reuse between moves.

Samples are written in the standard shard format with sparse policy targets
(visit distribution over legal moves), the game result and the root Q.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import chess
import numpy as np

from azchess.data.shards import ShardBuffer
from azchess.eval.openings import builtin_openings
from azchess.game import CHECKMATE, TERMINAL_NAMES, make_position
from azchess.mcts.pysearch import SearchParams
from azchess.mcts.search import SearchStats, make_cache, make_search, root_policy, run_batched


@dataclass
class SelfPlayConfig:
    games: int = 512
    concurrency: int = 128
    sims: int = 200
    fast_sims: int = 40
    full_search_prob: float = 0.25
    leaves_per_search: int = 4
    temperature: float = 1.0
    temp_plies: int = 30
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    cpuct_init: float = 1.25
    fpu_reduction: float = 0.25
    resign_threshold: float = -0.92
    resign_moves: int = 3
    no_resign_frac: float = 0.1
    max_plies: int = 400
    book_frac: float = 0.0  # fraction of games starting from a random book opening
    cache_size: int = 500_000
    pipeline: bool = True
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> SelfPlayConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class _Game:
    gid: int
    search: object
    full: bool = False
    resign_ok: bool = True
    low_streak: list[int] = field(default_factory=lambda: [0, 0])
    would_resign: int | None = None  # colour (0 white, 1 black) that would have resigned
    samples: list[dict] = field(default_factory=list)
    moves: list[str] = field(default_factory=list)
    ply: int = 0


@dataclass
class SelfPlayStats:
    games: int = 0
    positions: int = 0
    moves: int = 0
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    resigned: int = 0
    resign_false_positives: int = 0
    resign_checks: int = 0
    terminations: dict = field(default_factory=dict)
    seconds: float = 0.0
    evals: int = 0
    cache_hit_rate: float = 0.0

    def summary(self) -> dict:
        d = asdict(self)
        d["games_per_hour"] = self.games / self.seconds * 3600 if self.seconds else 0
        d["evals_per_s"] = self.evals / self.seconds if self.seconds else 0
        d["mean_plies"] = self.moves / self.games if self.games else 0
        d["resign_fp_rate"] = self.resign_false_positives / self.resign_checks if self.resign_checks else None
        return d


def generate(
    evaluator, cfg: SelfPlayConfig, out_dir: str | Path, prefix: str = "sp", verbose: bool = True
) -> SelfPlayStats:
    rng = np.random.default_rng(cfg.seed)
    params = SearchParams(
        cpuct_init=cfg.cpuct_init,
        fpu_reduction=cfg.fpu_reduction,
        dirichlet_alpha=cfg.dirichlet_alpha,
        dirichlet_epsilon=cfg.dirichlet_epsilon,
        seed=cfg.seed,
    )
    cache = make_cache(cfg.cache_size)
    buf = ShardBuffer(Path(out_dir), prefix, shard_size=200_000, seed=cfg.seed)
    stats = SelfPlayStats()
    search_stats = SearchStats()
    book = builtin_openings()
    started = 0
    active: list[_Game] = []
    t0 = last_print = time.time()
    games_log = open(Path(out_dir) / f"{prefix}_games.jsonl", "a")

    def start_game() -> _Game:
        nonlocal started
        p = SearchParams(**{**asdict(params), "seed": int(rng.integers(1 << 62))})
        s = make_search(p, cache=cache)
        pos = make_position(chess.STARTING_FEN)
        moves = []
        if cfg.book_frac > 0 and rng.random() < cfg.book_frac:
            moves = list(book[int(rng.integers(len(book)))])
            for m in moves:
                pos.push_uci(m)
        s.set_position(pos)
        g = _Game(started, s, resign_ok=rng.random() >= cfg.no_resign_frac, moves=moves, ply=len(moves))
        started += 1
        return g

    def finish(g: _Game, result_white: float, termination: str) -> None:
        for smp in g.samples:
            smp["result"] = np.int8(result_white if smp.pop("white") else -result_white)
        if g.samples:
            n = len(g.samples)
            width = max(len(s["policy_idx"]) for s in g.samples)
            pidx = np.full((n, width), -1, np.int16)
            pp = np.zeros((n, width), np.float16)
            for i, s in enumerate(g.samples):
                pidx[i, : len(s["policy_idx"])] = s["policy_idx"]
                pp[i, : len(s["policy_p"])] = s["policy_p"]
            buf.add(
                {
                    "bb": np.stack([s["bb"] for s in g.samples]),
                    "meta": np.stack([s["meta"] for s in g.samples]),
                    "policy_idx": pidx,
                    "policy_p": pp,
                    "result": np.array([s["result"] for s in g.samples], np.int8),
                    "q": np.array([s["q"] for s in g.samples], np.float16),
                    "legal": np.concatenate([s["legal"] for s in g.samples]),
                    "n_legal": np.array([len(s["legal"]) for s in g.samples], np.uint8),
                    "ply": np.array([s["ply"] for s in g.samples], np.int16),
                    "elo": np.zeros(n, np.int16),
                }
            )
        stats.games += 1
        stats.positions += len(g.samples)
        stats.moves += g.ply
        stats.terminations[termination] = stats.terminations.get(termination, 0) + 1
        if result_white > 0:
            stats.white_wins += 1
        elif result_white < 0:
            stats.black_wins += 1
        else:
            stats.draws += 1
        if g.would_resign is not None and not g.resign_ok:
            stats.resign_checks += 1
            loser_white = g.would_resign == 0
            if (loser_white and result_white >= 0) or (not loser_white and result_white <= 0):
                stats.resign_false_positives += 1
        games_log.write(
            json.dumps(
                {"gid": g.gid, "result": result_white, "termination": termination, "plies": g.ply, "moves": g.moves}
            )
            + "\n"
        )

    while active or started < cfg.games:
        while started < cfg.games and len(active) < cfg.concurrency:
            active.append(start_game())
        targets = []
        for g in active:
            g.full = bool(rng.random() < cfg.full_search_prob)
            g.search.begin(add_noise=g.full)
            targets.append(cfg.sims if g.full else cfg.fast_sims)
        run_batched(
            [g.search for g in active],
            evaluator,
            targets,
            cfg.leaves_per_search,
            pipeline=cfg.pipeline,
            stats=search_stats,
        )

        still = []
        for g in active:
            s = g.search
            pos = s.position()
            white = pos.turn
            moves, idx, visits, _q = root_policy(s)
            q = s.root_q()
            if g.full and visits.sum() > 0:
                bb, meta = pos.encode()
                nz = visits > 0
                g.samples.append(
                    {
                        "bb": bb,
                        "meta": meta,
                        "legal": idx.astype(np.int16),
                        "policy_idx": idx[nz].astype(np.int16),
                        "policy_p": (visits[nz] / visits.sum()).astype(np.float16),
                        "q": q,
                        "ply": g.ply,
                        "white": white,
                    }
                )
            # resignation (tracked per colour)
            c = 0 if white else 1
            g.low_streak[c] = g.low_streak[c] + 1 if q < cfg.resign_threshold else 0
            if g.low_streak[c] >= cfg.resign_moves:
                if g.resign_ok:
                    stats.resigned += 1
                    finish(g, -1.0 if white else 1.0, "resignation")
                    continue
                if g.would_resign is None:
                    g.would_resign = c
            # move selection
            if g.ply < cfg.temp_plies and cfg.temperature > 0:
                p = visits.astype(np.float64) ** (1.0 / cfg.temperature)
                k = int(rng.choice(len(moves), p=p / p.sum()))
            else:
                k = int(np.lexsort((_q, visits))[-1])
            uci = pos.uci(moves[k])
            s.advance(moves[k])
            g.moves.append(uci)
            g.ply += 1
            after = s.position()
            code = after.terminal()
            if code:
                loser_white = after.turn
                res = (-1.0 if loser_white else 1.0) if code == CHECKMATE else 0.0
                finish(g, res, TERMINAL_NAMES[code])
            elif g.ply >= cfg.max_plies:
                finish(g, 0.0, "max plies")
            else:
                still.append(g)
        active = still
        if verbose and time.time() - last_print > 30:
            last_print = time.time()
            el = last_print - t0
            print(
                f"[selfplay] games {stats.games}/{cfg.games} positions {stats.positions + sum(len(g.samples) for g in active)} "
                f"evals/s {search_stats.evals / max(search_stats.seconds, 1e-9):,.0f} games/h {stats.games / el * 3600:,.0f}",
                flush=True,
            )
    buf.flush()
    games_log.close()
    stats.seconds = time.time() - t0
    stats.evals = search_stats.evals
    if cache is not None and hasattr(cache, "lookups") and cache.lookups:
        stats.cache_hit_rate = cache.hits / cache.lookups
    if verbose:
        print(f"[selfplay] done: {json.dumps(stats.summary())}", flush=True)
    return stats
