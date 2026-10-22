"""Streaming batch loader over position shards.

Per-sample ``__getitem__`` datasets spend most of their time in Python at the
~10k positions/s we need, so this loader yields whole batches assembled with
vectorised numpy indexing. Each DataLoader worker owns a disjoint subset of
the shards, keeps ``shards_in_memory`` of them loaded at once and samples
batches from a random permutation of their union (a cross-shard shuffle
buffer). Compact encodings stay compact; planes are built on the GPU.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from neural_mcts_chess.data.shards import concat_positions, gather_ragged, legal_offsets, list_shards, load_shard
from neural_mcts_chess.encoding.moves import POLICY_SIZE


def make_batch(data: dict[str, np.ndarray], offsets: np.ndarray, rows: np.ndarray) -> dict[str, torch.Tensor]:
    legal_vals, lens = gather_ragged(data["legal"], offsets, rows)
    row_ids = np.repeat(np.arange(len(rows), dtype=np.int64), lens)
    flat = row_ids * POLICY_SIZE + legal_vals.astype(np.int64)
    return {
        "bb": torch.from_numpy(np.ascontiguousarray(data["bb"][rows]).view(np.int64)),
        "meta": torch.from_numpy(data["meta"][rows]),
        "policy_idx": torch.from_numpy(data["policy_idx"][rows].astype(np.int64)),
        "policy_p": torch.from_numpy(data["policy_p"][rows].astype(np.float32)),
        "result": torch.from_numpy(data["result"][rows].astype(np.int64)),
        "q": torch.from_numpy(data["q"][rows].astype(np.float32)),
        "legal_flat": torch.from_numpy(flat),
    }


class ShardStream(IterableDataset):
    def __init__(
        self,
        paths: list[str],
        batch_size: int,
        shuffle: bool = True,
        infinite: bool = True,
        shards_in_memory: int = 2,
        seed: int = 0,
        max_positions: int | None = None,
    ):
        super().__init__()
        if not paths:
            raise ValueError("no shards given")
        self.paths = list(paths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.infinite = infinite
        self.shards_in_memory = shards_in_memory
        self.seed = seed
        self.max_positions = max_positions

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng((self.seed, wid))
        if len(self.paths) >= nw:
            mine, stride = self.paths[wid::nw], 1
        else:  # fewer shards than workers: share shards, split rows
            mine, stride = self.paths, nw
        emitted = 0
        while True:
            order = rng.permutation(len(mine)) if self.shuffle else np.arange(len(mine))
            for g in range(0, len(order), self.shards_in_memory):
                group = [load_shard(mine[i]) for i in order[g : g + self.shards_in_memory]]
                data = concat_positions({k: [d[k] for d in group] for k in group[0]}) if len(group) > 1 else group[0]
                offsets = legal_offsets(data["n_legal"])
                n = len(data["bb"])
                rows = rng.permutation(n) if self.shuffle else np.arange(n)
                if stride > 1:
                    rows = rows[wid::stride]
                for i in range(0, len(rows) - self.batch_size + 1 if self.shuffle else len(rows), self.batch_size):
                    sel = rows[i : i + self.batch_size]
                    if self.max_positions is not None:
                        sel = sel[: max(0, self.max_positions - emitted)]
                        if len(sel) == 0:
                            return
                    emitted += len(sel)
                    yield make_batch(data, offsets, sel)
            if not self.infinite:
                return


def make_loader(
    patterns,
    batch_size: int,
    workers: int = 4,
    shuffle: bool = True,
    infinite: bool = True,
    seed: int = 0,
    max_positions: int | None = None,
    shards_in_memory: int = 2,
) -> DataLoader:
    paths = list_shards(patterns)
    ds = ShardStream(paths, batch_size, shuffle, infinite, shards_in_memory, seed, max_positions)
    return DataLoader(
        ds,
        batch_size=None,
        num_workers=workers,
        persistent_workers=workers > 0 and infinite,
        prefetch_factor=4 if workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
    )
