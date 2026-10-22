"""Backend-agnostic driver for batched MCTS.

A ``Search`` (C++ ``neural_mcts_chess._core.Search`` or the Python reference
``PySearch``) only walks trees; this module owns the loop that alternates
between collecting leaves from many searches and evaluating them in one
network call:

    gather leaves from all searches (virtual loss)  ->  one batched NN call
          ^                                                    |
          +------------------- apply + backup  <---------------+

With ``pipeline=True`` searches are split into two groups so the CPU gathers
leaves for one group while the GPU evaluates the other (the C++ gather
releases the GIL, so the two genuinely overlap).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from neural_mcts_chess.game import cpp_available
from neural_mcts_chess.mcts.pysearch import PySearch, SearchParams, py_apply_batch, py_gather_batch


@dataclass
class SearchStats:
    evals: int = 0
    batches: int = 0
    seconds: float = 0.0
    batch_sizes: list[int] = field(default_factory=list)

    @property
    def evals_per_s(self) -> float:
        return self.evals / self.seconds if self.seconds else 0.0


def use_cpp(backend: str) -> bool:
    return backend == "cpp" or (backend == "auto" and cpp_available())


def make_cache(capacity: int, backend: str = "auto"):
    if capacity <= 0:
        return None
    if use_cpp(backend):
        from neural_mcts_chess import _core

        return _core.NNCache(capacity)
    return OrderedDict()


def make_search(params: SearchParams, backend: str = "auto", cache=None, cache_size: int = 0):
    if use_cpp(backend):
        from neural_mcts_chess import _core

        return _core.Search(params.to_cpp(), cache)
    return PySearch(params, cache, cache_size)


def _fns(searches):
    if searches and not isinstance(searches[0], PySearch):
        from neural_mcts_chess import _core

        return _core.gather_batch, _core.apply_batch
    return py_gather_batch, py_apply_batch


def run_batched(
    searches: list,
    evaluator,
    target_visits: int | list[int],
    leaves_per_search: int = 8,
    deadline: float | None = None,
    stop: threading.Event | None = None,
    pipeline: bool = False,
    stats: SearchStats | None = None,
) -> SearchStats:
    """Search every tree until it has ``target_visits`` root visits (an int, or
    one per search) or until ``deadline`` / ``stop``. Leaves no evaluations pending."""
    stats = stats or SearchStats()
    if not searches:
        return stats
    gather, apply = _fns(searches)
    t0 = time.perf_counter()

    def halted() -> bool:
        return (deadline is not None and time.perf_counter() >= deadline) or (stop is not None and stop.is_set())

    if not pipeline or len(searches) < 2:
        while not halted():
            bb, meta, legal, _n_legal, _per, not_done = gather(searches, leaves_per_search, target_visits)
            if bb.shape[0]:
                lg, wdl = evaluator(bb, meta, legal)
                apply(searches, lg, wdl)
                stats.evals += bb.shape[0]
                stats.batches += 1
                stats.batch_sizes.append(bb.shape[0])
            elif not_done == 0:
                break
    else:
        targets = [target_visits] * len(searches) if isinstance(target_visits, int) else list(target_visits)
        groups = [searches[0::2], searches[1::2]]
        group_targets = [targets[0::2], targets[1::2]]
        futures: list = [None, None]
        with ThreadPoolExecutor(1) as pool:
            while True:
                active = False
                for gi, group in enumerate(groups):
                    if futures[gi] is not None:
                        n, fut = futures[gi]
                        lg, wdl = fut.result()
                        apply(group, lg, wdl)
                        futures[gi] = None
                        stats.evals += n
                        stats.batches += 1
                        stats.batch_sizes.append(n)
                    if halted():
                        continue
                    bb, meta, legal, _n_legal, _per, not_done = gather(group, leaves_per_search, group_targets[gi])
                    if bb.shape[0]:
                        futures[gi] = (bb.shape[0], pool.submit(evaluator, bb, meta, legal))
                        active = True
                    elif not_done:
                        active = True
                if not active and all(f is None for f in futures):
                    break
    stats.seconds += time.perf_counter() - t0
    return stats


def root_policy(search) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """(moves, policy indices, visit counts, child q) of the root."""
    ch = search.root_children()
    if not ch:
        return [], np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    moves = [c[0] for c in ch]
    idx = np.array([c[1] for c in ch], dtype=np.int64)
    visits = np.array([c[2] for c in ch], dtype=np.int64)
    q = np.array([c[3] for c in ch], dtype=np.float64)
    return moves, idx, visits, q


def choose_move(search, temperature: float = 0.0, rng: np.random.Generator | None = None):
    """Pick a root move: most visits (ties broken by Q) or sampled ~ N^(1/T)."""
    moves, _idx, visits, q = root_policy(search)
    if not moves:
        raise ValueError("search has no expanded root")
    if visits.sum() == 0:  # root expanded but no child visited yet: trust the policy prior
        priors = np.array([c[4] for c in search.root_children()])
        return moves[int(np.argmax(priors))]
    if temperature > 1e-3 and visits.sum() > 0:
        p = visits.astype(np.float64) ** (1.0 / temperature)
        p /= p.sum()
        return moves[int((rng or np.random.default_rng()).choice(len(moves), p=p))]
    order = np.lexsort((q, visits))  # primary key visits, secondary q
    return moves[int(order[-1])]
