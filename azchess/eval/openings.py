"""Opening book for evaluation matches.

Every opening is played twice with colours reversed, which cancels most of the
first-move advantage and opening-specific bias and keeps deterministic
engines from replaying the same game.
"""

from __future__ import annotations

import functools

import chess

# 50 common, roughly balanced openings (4-10 plies, SAN).
OPENINGS_SAN = [
    "e4 e5 Nf3 Nc6 Bb5 a6",
    "e4 e5 Nf3 Nc6 Bc4 Bc5",
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6",
    "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6",
    "e4 c5 Nc3 Nc6 g3 g6",
    "e4 e6 d4 d5 Nc3 Nf6",
    "e4 e6 d4 d5 e5 c5",
    "e4 c6 d4 d5 e5 Bf5",
    "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5",
    "e4 d5 exd5 Qxd5 Nc3 Qa5",
    "e4 d6 d4 Nf6 Nc3 g6",
    "e4 Nf6 e5 Nd5 d4 d6",
    "e4 e5 Nf3 Nf6 Nxe5 d6",
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6",
    "e4 e5 f4 exf4 Nf3 g5",
    "e4 e5 Nc3 Nf6 f4 d5",
    "e4 g6 d4 Bg7 Nc3 d6",
    "d4 d5 c4 e6 Nc3 Nf6",
    "d4 d5 c4 c6 Nf3 Nf6",
    "d4 d5 c4 dxc4 Nf3 Nf6",
    "d4 Nf6 c4 e6 Nc3 Bb4",
    "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6",
    "d4 Nf6 c4 e6 Nf3 b6",
    "d4 Nf6 c4 c5 d5 b5",
    "d4 Nf6 c4 g6 Nc3 d5",
    "d4 d5 Bf4 Nf6 e3 c5",
    "d4 Nf6 Nf3 e6 Bg5 h6",
    "d4 f5 g3 Nf6 Bg2 g6",
    "d4 e6 c4 Bb4+ Bd2 Qe7",
    "c4 e5 Nc3 Nf6 Nf3 Nc6",
    "c4 c5 Nc3 Nc6 g3 g6",
    "c4 Nf6 Nc3 e6 e4 d5",
    "Nf3 d5 g3 Nf6 Bg2 c6",
    "Nf3 Nf6 c4 g6 b3 Bg7",
    "e4 c5 c3 Nf6 e5 Nd5",
    "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6",
    "e4 e5 Nf3 Nc6 Nc3 Nf6",
    "e4 e5 Bc4 Nf6 d3 c6",
    "d4 d5 Nf3 Nf6 c4 e6 Bg5 Be7",
    "d4 d5 c4 e5 dxe5 d4",
    "e4 c6 d4 d5 exd5 cxd5 c4 Nf6",
    "e4 e6 d4 d5 Nd2 c5",
    "d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6",
    "b3 e5 Bb2 Nc6 e3 d5",
    "g3 d5 Bg2 Nf6 Nf3 c6",
    "e4 e5 Nf3 d6 d4 exd4 Nxd4 Nf6",
    "e4 Nc6 d4 d5 Nc3 dxe4",
    "f4 d5 Nf3 Nf6 e3 g6",
    "d4 Nf6 Bg5 e6 e4 h6",
    "e4 c5 Nf3 g6 d4 cxd4 Nxd4 Nc6",
]


@functools.cache
def builtin_openings() -> list[tuple[str, ...]]:
    """Openings as tuples of UCI moves from the standard start position."""
    out = []
    for line in OPENINGS_SAN:
        b = chess.Board()
        out.append(tuple(b.push_san(san).uci() for san in line.split()))
    return out


def opening_name(moves: tuple[str, ...]) -> str:
    """Opening as numbered SAN, e.g. '1. e4 e5 2. Nf3 Nc6'."""
    return chess.Board().variation_san([chess.Move.from_uci(m) for m in moves])
