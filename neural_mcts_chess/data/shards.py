"""On-disk shard format for training positions.

Each shard is an uncompressed ``.npz`` holding N positions:

==========  ==============  ===================================================
key         dtype / shape   meaning
==========  ==============  ===================================================
bb          uint64 (N, 12)  oriented piece bitboards (see encoding/board.py)
meta        uint8  (N, 4)   ep square, castling/colour flags, halfmove, reps
policy_idx  int16  (N, K)   policy target indices (K=1 for supervised data)
policy_p    float16 (N, K)  probabilities for ``policy_idx`` (-1 index = pad)
result      int8   (N,)     game result for the side to move (+1 / 0 / -1)
q           float16 (N,)    search value of the position (NaN if unknown)
legal       int16  (L,)     concatenated legal-move policy indices
n_legal     uint8  (N,)     legal moves per position (offsets = cumsum)
ply         int16  (N,)     ply number in the game
elo         int16  (N,)     rating of the side to move (0 if unknown)
==========  ==============  ===================================================

Storing the full legal-move list (~35 int16 per position) lets training mask
illegal moves out of the policy softmax and lets validation report legal
top-k accuracy, for ~70 extra bytes per position. Self-play shards use the
same format with K > 1 (sparse visit distributions).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KEYS = ("bb", "meta", "policy_idx", "policy_p", "result", "q", "legal", "n_legal", "ply", "elo")


@dataclass
class ShardBuffer:
    """Accumulates per-game arrays and flushes them as shuffled shards."""

    out_dir: Path
    prefix: str
    shard_size: int = 1_000_000
    seed: int = 0
    shuffle: bool = True
    parts: dict[str, list[np.ndarray]] = field(default_factory=lambda: {k: [] for k in KEYS})
    count: int = 0
    shard_index: int = 0
    written: int = 0

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.out_dir.glob(f"{self.prefix}_*.npz"))
        self.shard_index = len(existing)
        self.rng = np.random.default_rng(self.seed + self.shard_index)

    def add(self, arrays: dict[str, np.ndarray]) -> None:
        n = len(arrays["bb"])
        if n == 0:
            return
        for k in KEYS:
            self.parts[k].append(arrays[k])
        self.count += n
        if self.count >= self.shard_size:
            self.flush()

    def flush(self) -> Path | None:
        if self.count == 0:
            return None
        data = concat_positions({k: v for k, v in self.parts.items()})
        if self.shuffle:
            data = take_positions(data, self.rng.permutation(len(data["bb"])))
        path = self.out_dir / f"{self.prefix}_{self.shard_index:05d}.npz"
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **data)
        os.replace(tmp, path)
        self.written += len(data["bb"])
        self.shard_index += 1
        self.parts = {k: [] for k in KEYS}
        self.count = 0
        return path


def pad_policy(parts: list[np.ndarray], fill) -> np.ndarray:
    width = max(p.shape[1] for p in parts)
    if all(p.shape[1] == width for p in parts):
        return np.concatenate(parts)
    out = []
    for p in parts:
        if p.shape[1] < width:
            pad = np.full((p.shape[0], width - p.shape[1]), fill, dtype=p.dtype)
            p = np.concatenate([p, pad], axis=1)
        out.append(p)
    return np.concatenate(out)


def concat_positions(parts: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    out = {}
    for k, v in parts.items():
        if k == "policy_idx":
            out[k] = pad_policy(v, -1)
        elif k == "policy_p":
            out[k] = pad_policy(v, 0)
        else:
            out[k] = np.concatenate(v)
    return out


def legal_offsets(n_legal: np.ndarray) -> np.ndarray:
    off = np.zeros(len(n_legal) + 1, dtype=np.int64)
    np.cumsum(n_legal, out=off[1:])
    return off


def gather_ragged(values: np.ndarray, offsets: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gather the ragged rows ``rows`` from (values, offsets). Returns
    (flat values, per-row lengths), fully vectorised."""
    starts = offsets[rows]
    lens = offsets[rows + 1] - starts
    total = int(lens.sum())
    if total == 0:
        return values[:0], lens
    row_starts = np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
    idx = row_starts + np.arange(total)
    return values[idx], lens


def take_positions(data: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, np.ndarray]:
    out = {k: v[rows] for k, v in data.items() if k not in ("legal", "n_legal")}
    off = legal_offsets(data["n_legal"])
    out["legal"], lens = gather_ragged(data["legal"], off, rows)
    out["n_legal"] = lens.astype(np.uint8)
    return out


def load_shard(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def list_shards(pattern: str | Path | list) -> list[str]:
    if isinstance(pattern, (list, tuple)):
        return sorted(p for pat in pattern for p in list_shards(pat))
    pattern = str(pattern)
    if os.path.isdir(pattern):
        pattern = os.path.join(pattern, "*.npz")
    return sorted(p for p in glob.glob(pattern) if not p.endswith(".tmp.npz"))


def shard_stats(paths: list[str]) -> dict[str, float]:
    n = 0
    for p in paths:
        with np.load(p) as z:
            n += z["result"].shape[0]
    return {"shards": len(paths), "positions": n}
