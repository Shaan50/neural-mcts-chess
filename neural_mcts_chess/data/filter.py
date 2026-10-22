"""Stream-filter a (possibly truncated) Lichess .pgn.zst dump by game headers.

Only headers are inspected, so filtering runs at roughly decompression speed
(no move parsing). Surviving games are written verbatim to a zstd-compressed
PGN that the (much slower, parallel) shard builder consumes.

Default filter: both players rated >= 2000, standard chess, time control of at
least blitz (estimated duration base + 40*increment >= 180 s), and a "Normal"
termination (drops time forfeits and abandonments, whose results are noisy
value targets).
"""

from __future__ import annotations

import argparse
import io
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import zstandard

HEADER_RE = re.compile(r'^\[(\w+) "(.*)"\]\s*$')


@dataclass
class GameFilter:
    min_elo: int = 2000
    max_elo_diff: int | None = 400
    min_duration_s: int = 180
    require_normal_termination: bool = True

    def accept(self, h: dict[str, str]) -> bool:
        try:
            we, be = int(h.get("WhiteElo", "0")), int(h.get("BlackElo", "0"))
        except ValueError:  # "?" for unrated/provisional
            return False
        if we < self.min_elo or be < self.min_elo:
            return False
        if self.max_elo_diff is not None and abs(we - be) > self.max_elo_diff:
            return False
        if h.get("Result") not in ("1-0", "0-1", "1/2-1/2"):
            return False
        if h.get("Variant", "Standard") != "Standard" or "FEN" in h:
            return False
        if self.require_normal_termination and h.get("Termination") != "Normal":
            return False
        tc = h.get("TimeControl", "-")
        if tc == "-":  # correspondence
            return True
        try:
            base, inc = (int(x) for x in tc.split("+"))
        except ValueError:
            return False
        return base + 40 * inc >= self.min_duration_s


def iter_raw_games(text: io.TextIOBase) -> Iterator[tuple[dict[str, str], str]]:
    """Yield (headers, raw_pgn_text) for each game without parsing moves."""
    headers: dict[str, str] = {}
    lines: list[str] = []
    in_moves = False
    for line in text:
        if line.startswith("["):
            if in_moves:  # a new game starts
                yield headers, "".join(lines)
                headers, lines, in_moves = {}, [], False
            m = HEADER_RE.match(line)
            if m:
                headers[m.group(1)] = m.group(2)
            lines.append(line)
        else:
            if line.strip():
                in_moves = True
            lines.append(line)
    if headers and in_moves:
        yield headers, "".join(lines)


def open_truncated_zst(path: Path) -> io.TextIOBase:
    """Text stream over a zstd file that may be cut off mid-frame."""

    class SafeReader(io.RawIOBase):
        def __init__(self) -> None:
            self._fh = open(path, "rb")
            self._r = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(self._fh, read_across_frames=True)

        def readable(self) -> bool:
            return True

        def readinto(self, b) -> int:  # type: ignore[override]
            try:
                # read1: at most one read from the file, so a truncated tail only
                # loses the last partial chunk instead of a whole buffer's worth
                data = self._r.read1(len(b))
            except zstandard.ZstdError:  # truncated tail of a partial download
                return 0
            b[: len(data)] = data
            return len(data)

    return io.TextIOWrapper(io.BufferedReader(SafeReader(), 1 << 20), encoding="utf-8", errors="replace")


def filter_dump(src: Path, dst: Path, flt: GameFilter, max_games: int | None = None) -> dict[str, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    stats = {"seen": 0, "kept": 0}
    t0 = time.time()
    cctx = zstandard.ZstdCompressor(level=6, threads=-1)
    with open(dst, "wb") as fout, cctx.stream_writer(fout) as w:
        pending = None
        for headers, raw in iter_raw_games(open_truncated_zst(src)):
            # Hold back one game so the final (possibly truncated) game is dropped.
            if pending is not None:
                ph, praw = pending
                stats["seen"] += 1
                if flt.accept(ph):
                    w.write(praw.encode())
                    stats["kept"] += 1
                    if max_games and stats["kept"] >= max_games:
                        pending = None
                        break
                if stats["seen"] % 500_000 == 0:
                    el = time.time() - t0
                    print(
                        f"[filter] seen {stats['seen']:,} kept {stats['kept']:,} ({stats['kept'] / stats['seen']:.2%})  {stats['seen'] / el:,.0f} games/s",
                        flush=True,
                    )
            pending = (headers, raw)
    print(f"[filter] done: seen {stats['seen']:,} kept {stats['kept']:,} -> {dst}")
    return stats


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--min-elo", type=int, default=2000)
    p.add_argument("--max-elo-diff", type=int, default=400)
    p.add_argument("--min-duration", type=int, default=180)
    p.add_argument("--any-termination", action="store_true")
    p.add_argument("--max-games", type=int, default=None)
    a = p.parse_args(argv)
    flt = GameFilter(a.min_elo, a.max_elo_diff, a.min_duration, not a.any_termination)
    filter_dump(a.src, a.dst, flt, a.max_games)


if __name__ == "__main__":
    main()
