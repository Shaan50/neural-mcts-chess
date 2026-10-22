"""Game-state backends.

Two interchangeable implementations of the same small interface are provided:

* :class:`PyPosition` – a thin wrapper around ``python-chess`` (reference,
  always available).
* ``azchess._core.Position`` – a C++17 bitboard move generator exposed via
  pybind11 (10-100x faster); built from ``csrc/``.

Both expose: ``legal_moves()``, ``policy_indices(moves)``, ``push(move)``,
``pop()``, ``push_uci(str)``, ``uci(move)``, ``terminal()``, ``encode()``,
``fen()``, ``copy()``, ``turn`` (True = white), ``is_check()``, ``ply()`` and
``repetition_count()``. Moves are opaque objects as far as callers are
concerned (``chess.Move`` for python-chess, 16-bit ints for C++).

Game termination rules (identical for both backends, checked in this order):
no legal moves -> checkmate/stalemate; insufficient material; halfmove clock
>= 100 (fifty-move rule, applied automatically); third occurrence of a
position (threefold repetition, applied automatically).
"""

from __future__ import annotations

from collections import Counter

import chess
import numpy as np

from azchess.encoding.board import encode_compact
from azchess.encoding.moves import move_to_index

ONGOING, CHECKMATE, STALEMATE, INSUFFICIENT, FIFTY_MOVES, THREEFOLD = range(6)
TERMINAL_NAMES = {
    ONGOING: "ongoing",
    CHECKMATE: "checkmate",
    STALEMATE: "stalemate",
    INSUFFICIENT: "insufficient material",
    FIFTY_MOVES: "fifty-move rule",
    THREEFOLD: "threefold repetition",
}


class PyPosition:
    """python-chess backed game state with repetition tracking."""

    backend = "python"

    def __init__(self, fen: str = chess.STARTING_FEN):
        self.board = chess.Board(fen)
        self._counts: Counter = Counter([self.board._transposition_key()])
        self._keys: list = [self.board._transposition_key()]

    @classmethod
    def from_board(cls, board: chess.Board) -> PyPosition:
        root = board.root()
        pos = cls(root.fen())
        for mv in board.move_stack:
            pos.push(mv)
        return pos

    def copy(self) -> PyPosition:
        new = PyPosition.__new__(PyPosition)
        new.board = self.board.copy(stack=True)
        new._counts = self._counts.copy()
        new._keys = list(self._keys)
        return new

    @property
    def turn(self) -> bool:
        return self.board.turn

    def legal_moves(self) -> list[chess.Move]:
        return list(self.board.legal_moves)

    def policy_indices(self, moves) -> np.ndarray:
        turn = self.board.turn
        return np.fromiter((move_to_index(m, turn) for m in moves), dtype=np.int64, count=len(moves))

    def push(self, move: chess.Move) -> None:
        self.board.push(move)
        key = self.board._transposition_key()
        self._keys.append(key)
        self._counts[key] += 1

    def pop(self) -> chess.Move:
        key = self._keys.pop()
        self._counts[key] -= 1
        return self.board.pop()

    def push_uci(self, uci: str) -> None:
        move = chess.Move.from_uci(uci)
        if move not in self.board.legal_moves:
            raise ValueError(f"illegal move {uci} in {self.board.fen()}")
        self.push(move)

    def uci(self, move: chess.Move) -> str:
        return move.uci()

    def parse_uci(self, uci: str) -> chess.Move:
        return chess.Move.from_uci(uci)

    def is_check(self) -> bool:
        return self.board.is_check()

    def repetition_count(self) -> int:
        return self._counts[self._keys[-1]]

    def ply(self) -> int:
        return len(self.board.move_stack)

    def fen(self) -> str:
        return self.board.fen()

    def terminal(self) -> int:
        b = self.board
        if not any(b.generate_legal_moves()):
            return CHECKMATE if b.is_check() else STALEMATE
        if b.is_insufficient_material():
            return INSUFFICIENT
        if b.halfmove_clock >= 100:
            return FIFTY_MOVES
        if self.repetition_count() >= 3:
            return THREEFOLD
        return ONGOING

    def encode(self) -> tuple[np.ndarray, np.ndarray]:
        return encode_compact(self.board, self.repetition_count() - 1)


def cpp_available() -> bool:
    try:
        from azchess import _core  # noqa: F401
    except ImportError:
        return False
    return True


def make_position(fen: str = chess.STARTING_FEN, backend: str = "auto"):
    """Create a position with the requested backend ("auto" prefers C++)."""
    if backend == "python" or (backend == "auto" and not cpp_available()):
        return PyPosition(fen)
    from azchess import _core

    return _core.Position(fen)


def terminal_value(code: int) -> float:
    """Value of a terminal position from the side to move's perspective."""
    return -1.0 if code == CHECKMATE else 0.0
