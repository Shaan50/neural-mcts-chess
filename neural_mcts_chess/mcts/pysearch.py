"""Pure-Python reference implementation of the batched PUCT search.

This mirrors ``csrc/mcts.cpp`` operation for operation (same selection
formula, FPU, virtual loss, terminal handling and backup) so that the fast C++
search can be validated against a readable implementation, and so the engine
still works when the extension is not built. It works with either position
backend.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import asdict, dataclass

import numpy as np

from neural_mcts_chess.encoding.moves import POLICY_SIZE
from neural_mcts_chess.game import CHECKMATE, ONGOING, THREEFOLD

UNEXPANDED, PENDING, EXPANDED, TERMINAL = range(4)


@dataclass
class SearchParams:
    cpuct_init: float = 1.25
    cpuct_base: float = 19652.0
    fpu_reduction: float = 0.25
    fpu_reduction_root: float = 0.0
    policy_temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    virtual_loss: float = 1.0
    max_collisions: int = 8
    twofold_draw: bool = True
    seed: int = 0

    def to_cpp(self):
        from neural_mcts_chess import _core

        p = _core.SearchParams()
        for k, v in asdict(self).items():
            setattr(p, k, v)
        return p


class Node:
    __slots__ = ("parent", "children", "move", "prior", "n", "n_inflight", "w", "d", "state", "terminal_v")

    def __init__(self, parent: Node | None = None, move=None, prior: float = 0.0):
        self.parent = parent
        self.children: list[Node] = []
        self.move = move
        self.prior = prior
        self.n = 0
        self.n_inflight = 0
        self.w = 0.0
        self.d = 0.0
        self.state = UNEXPANDED
        self.terminal_v = 0.0


class PyBatch:
    def __init__(self) -> None:
        self.bb: list[np.ndarray] = []
        self.meta: list[np.ndarray] = []
        self.legal: list[np.ndarray] = []
        self.rows = 0


def _position_key(pos) -> tuple:
    key = pos.hash() if hasattr(pos, "hash") else pos.board._transposition_key()
    return (
        key,
        min(pos.board.halfmove_clock if hasattr(pos, "board") else pos.halfmove_clock(), 255),
        min(pos.repetition_count(), 3),
    )


class PySearch:
    def __init__(self, params: SearchParams, cache: OrderedDict | None = None, cache_size: int = 0):
        self.params = params
        self.cache = cache
        self.cache_size = cache_size
        self.rng = np.random.default_rng(params.seed)
        self.pos = None
        self.root = Node()
        self.pending_: list[tuple[Node, tuple, list]] = []
        self.want_noise = False
        self.noise_applied = False
        self._max_depth = 0
        self._depth_sum = 0
        self._depth_count = 0

    # --------------------------------------------------------------- setup
    def set_position(self, pos) -> None:
        self.pos = pos.copy()
        self.root = Node()
        self.pending_ = []
        self.noise_applied = False

    def position(self):
        return self.pos.copy()

    def begin(self, add_noise: bool = False) -> None:
        self.want_noise = add_noise
        self.noise_applied = False
        self._max_depth = self._depth_sum = self._depth_count = 0

    def advance(self, move) -> None:
        if self.pending_:
            raise RuntimeError("advance() while evaluations are pending")
        self.pos.push(move)
        child = None
        if self.root.state == EXPANDED:
            for c in self.root.children:
                if c.move == move:
                    child = c
        if child is None or child.state == PENDING:
            self.root = Node()
        else:
            child.parent = None
            self.root = child
            if child.state == TERMINAL and self.pos.terminal() == ONGOING:
                self.root = Node()
        self.noise_applied = False

    def advance_uci(self, uci: str) -> None:
        self.advance(self.pos.parse_uci(uci))

    # ------------------------------------------------------------ search
    def _select(self, node: Node) -> Node:
        p = self.params
        child_visits = 0
        visited_policy = 0.0
        for c in node.children:
            cn = c.n + c.n_inflight
            child_visits += cn
            if cn > 0:
                visited_policy += c.prior
        pn = float(child_visits)
        cpuct = p.cpuct_init + math.log((pn + p.cpuct_base + 1.0) / p.cpuct_base)
        sqrt_n = math.sqrt(max(pn, 1.0))
        parent_q = -node.w / node.n if node.n > 0 else 0.0
        red = p.fpu_reduction_root if node is self.root else p.fpu_reduction
        fpu = parent_q - red * math.sqrt(visited_policy)
        best, best_c = -math.inf, node.children[0]
        for c in node.children:
            cn = c.n + c.n_inflight
            q = (c.w - p.virtual_loss * c.n_inflight) / cn if cn > 0 else fpu
            u = cpuct * c.prior * sqrt_n / (1.0 + cn)
            if q + u > best:
                best, best_c = q + u, c
        return best_c

    def _expand(self, node: Node, moves: list, logits: np.ndarray) -> None:
        z = np.asarray(logits, dtype=np.float64)
        z = np.exp((z - z.max()) / self.params.policy_temperature)
        priors = (z / z.sum()).astype(np.float32)
        node.children = [Node(node, m, float(pr)) for m, pr in zip(moves, priors)]
        node.state = EXPANDED

    @staticmethod
    def _backup(node: Node | None, v: float, d: float, from_inflight: bool) -> None:
        while node is not None:
            node.n += 1
            node.w += -v
            node.d += d
            if from_inflight:
                node.n_inflight -= 1
            v = -v
            node = node.parent

    def _apply_noise(self) -> None:
        k = len(self.root.children)
        if k == 0:
            return
        noise = self.rng.gamma(self.params.dirichlet_alpha, 1.0, size=k)
        if noise.sum() <= 0:
            return
        noise /= noise.sum()
        eps = self.params.dirichlet_epsilon
        for c, x in zip(self.root.children, noise):
            c.prior = float(np.float32((1 - eps) * c.prior + eps * x))
        self.noise_applied = True

    def gather(self, batch: PyBatch, max_leaves: int, target_visits: int) -> int:
        collected = collisions = 0
        pos = self.pos
        while collected < max_leaves and collisions <= self.params.max_collisions:
            r = self.root
            if r.state == TERMINAL or r.n + r.n_inflight >= target_visits:
                break
            node, depth = r, 0
            while node.state == EXPANDED:
                if node is r and self.want_noise and not self.noise_applied:
                    self._apply_noise()
                node = self._select(node)
                pos.push(node.move)
                depth += 1
            self._max_depth = max(self._max_depth, depth)
            self._depth_sum += depth
            self._depth_count += 1
            try:
                if node.state == PENDING:
                    collisions += 1
                    self._depth_count -= 1
                    self._depth_sum -= depth
                    continue
                if node.state == TERMINAL:
                    v = node.terminal_v
                    self._backup(node, v, 1.0 if v == 0.0 else 0.0, False)
                    continue
                moves = pos.legal_moves()
                t = pos.terminal()
                if t == ONGOING and self.params.twofold_draw and depth > 0 and pos.repeated_within(depth):
                    t = THREEFOLD
                if t != ONGOING:
                    node.state = TERMINAL
                    node.terminal_v = -1.0 if t == CHECKMATE else 0.0
                    self._backup(node, node.terminal_v, 0.0 if t == CHECKMATE else 1.0, False)
                    continue
                key = _position_key(pos) if self.cache is not None else None
                if key is not None and key in self.cache:
                    logits, wdl = self.cache[key]
                    self._expand(node, moves, logits)
                    self._backup(node, float(wdl[0]) - float(wdl[2]), float(wdl[1]), False)
                    continue
                node.state = PENDING
                x = node
                while x is not None:
                    x.n_inflight += 1
                    x = x.parent
                bb, meta = pos.encode()
                batch.bb.append(bb)
                batch.meta.append(meta)
                batch.legal.append(pos.policy_indices(moves))
                batch.rows += 1
                self.pending_.append((node, key, moves))
                collected += 1
            finally:
                for _ in range(depth):
                    pos.pop()
        return collected

    def apply(self, legal_logits: np.ndarray, wdl: np.ndarray) -> int:
        off = 0
        for i, (node, key, moves) in enumerate(self.pending_):
            k = len(moves)
            lg = legal_logits[off : off + k]
            w = wdl[i]
            if self.cache is not None:
                self.cache[key] = (np.array(lg, dtype=np.float32), np.array(w, dtype=np.float32))
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
            self._expand(node, moves, lg)
            self._backup(node, float(w[0]) - float(w[2]), float(w[1]), True)
            off += k
        self.pending_ = []
        return off

    # ----------------------------------------------------------- queries
    def pending(self) -> int:
        return len(self.pending_)

    def done(self, target_visits: int) -> bool:
        return self.root.state == TERMINAL or self.root.n >= target_visits

    def root_visits(self) -> int:
        return self.root.n

    def root_q(self) -> float:
        return -self.root.w / self.root.n if self.root.n > 0 else 0.0

    def root_d(self) -> float:
        return self.root.d / self.root.n if self.root.n > 0 else 0.0

    def root_children(self) -> list[tuple]:
        if self.root.state != EXPANDED:
            return []
        idx = self.pos.policy_indices([c.move for c in self.root.children])
        return [(c.move, int(i), c.n, c.w / c.n if c.n > 0 else 0.0, c.prior) for c, i in zip(self.root.children, idx)]

    def pv(self, max_len: int = 32) -> list:
        out, node = [], self.root
        while len(out) < max_len and node.state == EXPANDED:
            best = max(node.children, key=lambda c: c.n)
            if best.n == 0:
                break
            out.append(best.move)
            node = best
        return out

    def tree_size(self) -> int:
        stack, n = [self.root], 0
        while stack:
            x = stack.pop()
            n += 1
            stack.extend(x.children)
        return n

    def max_depth(self) -> int:
        return self._max_depth

    def mean_depth(self) -> float:
        return self._depth_sum / self._depth_count if self._depth_count else 0.0


def py_gather_batch(searches: list[PySearch], max_leaves: int, target_visits):
    targets = [target_visits] * len(searches) if isinstance(target_visits, int) else list(target_visits)
    batch = PyBatch()
    per = np.array([s.gather(batch, max_leaves, t) for s, t in zip(searches, targets)], dtype=np.int32)
    not_done = sum(not s.done(t) for s, t in zip(searches, targets))
    if batch.rows:
        bb = np.stack(batch.bb).view(np.int64)
        meta = np.stack(batch.meta)
        n_legal = np.array([len(x) for x in batch.legal], dtype=np.int32)
        rows = np.repeat(np.arange(batch.rows, dtype=np.int64), n_legal)
        legal = rows * POLICY_SIZE + np.concatenate(batch.legal).astype(np.int64)
    else:
        bb, meta = np.zeros((0, 12), np.int64), np.zeros((0, 4), np.uint8)
        legal, n_legal = np.zeros(0, np.int64), np.zeros(0, np.int32)
    return bb, meta, legal, n_legal, per, not_done


def py_apply_batch(searches: list[PySearch], legal_logits: np.ndarray, wdl: np.ndarray) -> None:
    off = row = 0
    for s in searches:
        k = s.pending()
        off += s.apply(legal_logits[off:], wdl[row : row + k])
        row += k
