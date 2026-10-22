"""Convert a filtered Lichess PGN dump into training shards.

Games are split into train / val / test by a hash of the Lichess game id, so
every position of a game lands in the same split (no leakage of near-identical
positions between training and validation). Parsing is parallelised across
processes; each worker tokenises movetext in Python and hands the SAN list to
the C++ ``encode_game`` (SAN parsing, legal-move generation and encoding in a
single call). A python-chess fallback is used if the extension is missing.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import time
import zlib
from pathlib import Path

import chess
import numpy as np

from neural_mcts_chess.data.filter import iter_raw_games, open_truncated_zst
from neural_mcts_chess.data.shards import ShardBuffer
from neural_mcts_chess.game import cpp_available

COMMENT_RE = re.compile(r"\{[^}]*\}")
MOVENUM_RE = re.compile(r"\d+\.(?:\.\.)?")
RESULTS = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


def split_of(game_id: str, val_permille: int, test_permille: int) -> str:
    h = zlib.crc32(game_id.encode()) % 1000
    if h < val_permille:
        return "val"
    if h < val_permille + test_permille:
        return "test"
    return "train"


def movetext_tokens(raw: str) -> list[str]:
    body = "".join(line for line in raw.splitlines(keepends=True) if not line.startswith("["))
    body = MOVENUM_RE.sub(" ", COMMENT_RE.sub(" ", body))
    return [t for t in body.split() if t not in RESULTS and t != "*" and not t.startswith("$")]


def _encode_python(sans: list[str]):
    from neural_mcts_chess.encoding.moves import move_to_index
    from neural_mcts_chess.game import PyPosition

    pos = PyPosition()
    bbs, metas, moves, legal, n_legal = [], [], [], [], []
    for san in sans:
        try:
            mv = pos.board.parse_san(san)
        except ValueError:
            break
        bb, meta = pos.encode()
        bbs.append(bb)
        metas.append(meta)
        moves.append(move_to_index(mv, pos.turn))
        lm = pos.legal_moves()
        legal.extend(pos.policy_indices(lm).tolist())
        n_legal.append(len(lm))
        pos.push(mv)
    n = len(moves)
    return (
        np.array(bbs, dtype=np.uint64).reshape(n, 12),
        np.array(metas, dtype=np.uint8).reshape(n, 4),
        np.array(moves, dtype=np.int16),
        np.array(legal, dtype=np.int16),
        np.array(n_legal, dtype=np.int32),
        int(n == len(sans)),
    )


def encode_games(batch: list[tuple[dict, str]]) -> list[tuple[str, dict]]:
    use_cpp = cpp_available()
    if use_cpp:
        from neural_mcts_chess import _core
    out = []
    for headers, raw in batch:
        sans = movetext_tokens(raw)
        if len(sans) < 2:
            continue
        if use_cpp:
            bb, meta, mv, legal, n_legal, complete, _term = _core.encode_game(chess.STARTING_FEN, sans)
        else:
            bb, meta, mv, legal, n_legal, complete = _encode_python(sans)
        if not complete or len(mv) == 0:
            continue  # malformed game; skip entirely
        n = len(mv)
        res_white = RESULTS[headers["Result"]]
        sign = np.where(np.arange(n) % 2 == 0, 1, -1).astype(np.int8)
        we, be = int(headers.get("WhiteElo", 0)), int(headers.get("BlackElo", 0))
        out.append(
            (
                headers.get("Site", ""),
                {
                    "bb": bb,
                    "meta": meta,
                    "policy_idx": mv.reshape(n, 1),
                    "policy_p": np.ones((n, 1), dtype=np.float16),
                    "result": (sign * res_white).astype(np.int8),
                    "q": np.full(n, np.nan, dtype=np.float16),
                    "legal": legal,
                    "n_legal": n_legal.astype(np.uint8),
                    "ply": np.arange(n, dtype=np.int16),
                    "elo": np.where(np.arange(n) % 2 == 0, we, be).astype(np.int16),
                },
            )
        )
    return out


def _chunks(it, size):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def build(
    src: Path,
    out_dir: Path,
    shard_size: int = 1_000_000,
    val_permille: int = 10,
    test_permille: int = 10,
    workers: int = max(1, mp.cpu_count() - 2),
    max_games: int | None = None,
    seed: int = 0,
) -> dict[str, int]:
    buffers = {
        split: ShardBuffer(out_dir / split, split, shard_size if split == "train" else shard_size // 4, seed=seed)
        for split in ("train", "val", "test")
    }
    games = iter_raw_games(open_truncated_zst(src) if src.suffix == ".zst" else open(src))
    if max_games:
        games = (g for i, g in zip(range(max_games), games))
    stats = {"games": 0, "positions": 0}
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for results in pool.imap(encode_games, _chunks(games, 256), chunksize=1):
            for site, arrays in results:
                buffers[split_of(site, val_permille, test_permille)].add(arrays)
                stats["games"] += 1
                stats["positions"] += len(arrays["bb"])
            if stats["games"] % 20_000 < 256:
                rate = stats["positions"] / (time.time() - t0)
                print(
                    f"[build] games {stats['games']:,} positions {stats['positions']:,} ({rate:,.0f} pos/s)", flush=True
                )
    for split, buf in buffers.items():
        buf.flush()
        stats[f"{split}_positions"] = buf.written
    print(f"[build] done in {time.time() - t0:.0f}s: {stats}")
    return stats


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path, help="filtered .pgn or .pgn.zst")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--shard-size", type=int, default=1_000_000)
    p.add_argument("--val-permille", type=int, default=10)
    p.add_argument("--test-permille", type=int, default=10)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    p.add_argument("--max-games", type=int, default=None)
    a = p.parse_args(argv)
    build(a.src, a.out_dir, a.shard_size, a.val_permille, a.test_permille, a.workers, a.max_games)


if __name__ == "__main__":
    main()
