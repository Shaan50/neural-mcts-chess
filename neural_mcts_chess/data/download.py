"""Resumable download of (a prefix of) a monthly Lichess database dump.

Monthly dumps are ~30 GB of zstd-compressed PGN. We rarely need the whole
file: high-rated games are spread uniformly through the month, so the first
few GB contain a representative sample. This module downloads the first
``max_bytes`` of the archive with HTTP range requests, resuming from a partial
file if interrupted. The truncated archive is decoded by ``neural_mcts_chess.data.filter``
which treats an unexpected end-of-stream as a clean EOF.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests

LICHESS_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
CHUNK = 1 << 20


def parse_size(text: str) -> int:
    text = text.strip().upper()
    units = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
    if text[-1] in units:
        return int(float(text[:-1]) * units[text[-1]])
    if text.endswith("B") and text[-2] in units:
        return int(float(text[:-2]) * units[text[-2]])
    return int(text)


def remote_size(url: str) -> int:
    r = requests.head(url, allow_redirects=True, timeout=30)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def download_prefix(month: str, out_dir: Path, max_bytes: int | None, retries: int = 20) -> Path:
    url = LICHESS_URL.format(month=month)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"lichess_{month}.pgn.zst"
    total = remote_size(url)
    target = total if max_bytes is None else min(total, max_bytes)
    done = out.stat().st_size if out.exists() else 0
    print(
        f"[download] {url}\n[download] remote={total / 1e9:.2f} GB target={target / 1e9:.2f} GB have={done / 1e9:.2f} GB"
    )

    attempt = 0
    t0, start_bytes = time.time(), done
    while done < target:
        try:
            headers = {"Range": f"bytes={done}-{target - 1}"}
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(out, "ab") as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        done += len(chunk)
                        if done // (64 * CHUNK) != (done - len(chunk)) // (64 * CHUNK):
                            rate = (done - start_bytes) / max(time.time() - t0, 1e-6)
                            eta = (target - done) / max(rate, 1)
                            print(
                                f"[download] {done / 1e9:6.2f}/{target / 1e9:.2f} GB  {rate / 1e6:5.1f} MB/s  eta {eta / 60:5.1f} min",
                                flush=True,
                            )
                        if done >= target:
                            break
            attempt = 0
        except (requests.RequestException, OSError) as e:  # network hiccup: back off and resume
            attempt += 1
            if attempt > retries:
                raise
            wait = min(60, 2**attempt)
            print(f"[download] error {e!r}; retry {attempt}/{retries} in {wait}s", file=sys.stderr)
            time.sleep(wait)
    if out.stat().st_size > target:  # trim overshoot from the final chunk
        with open(out, "r+b") as f:
            f.truncate(target)
    print(f"[download] complete: {out} ({os.path.getsize(out) / 1e9:.2f} GB)")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--month", default="2025-01", help="YYYY-MM of the Lichess standard rated dump")
    p.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--max-bytes", type=parse_size, default=None, help="download only this prefix, e.g. 2G")
    args = p.parse_args(argv)
    download_prefix(args.month, args.out_dir, args.max_bytes)


if __name__ == "__main__":
    main()
