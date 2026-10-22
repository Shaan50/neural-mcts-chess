"""AlphaZero-style 8x8x73 move encoding.

A move is identified by its origin square and one of 73 "move planes":

* planes  0-55: queen-like moves, ``direction * 7 + (distance - 1)`` with
  directions N, NE, E, SE, S, SW, W, NW (queen promotions use these planes)
* planes 56-63: the 8 knight jumps
* planes 64-72: under-promotions, ``64 + (file_delta + 1) * 3 + piece`` with
  piece in (knight, bishop, rook)

The flat policy index is ``plane * 64 + from_square`` so that a convolutional
policy head emitting a (73, 8, 8) tensor can be flattened directly. Squares
are always expressed from the side to move's perspective: for black the board
is mirrored vertically (``square ^ 56``), so "N" always means "forward".
"""

from __future__ import annotations

import chess
import numpy as np

NUM_MOVE_PLANES = 73
POLICY_SIZE = NUM_MOVE_PLANES * 64  # 4672

QUEEN_DIRS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]  # (dfile, drank)
KNIGHT_DIRS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
UNDERPROMOTIONS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

# promo slot: 0 = none / queen, 1 = knight, 2 = bishop, 3 = rook, 4 = queen
_PROMO_SLOT = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}


def _plane(df: int, dr: int, promo_slot: int) -> int:
    if promo_slot in (1, 2, 3):
        if dr != 1 or df not in (-1, 0, 1):
            return -1
        return 64 + (df + 1) * 3 + (promo_slot - 1)
    if (df, dr) in KNIGHT_DIRS:
        return 56 + KNIGHT_DIRS.index((df, dr))
    if df == 0 and dr == 0:
        return -1
    if df != 0 and dr != 0 and abs(df) != abs(dr):
        return -1
    dist = max(abs(df), abs(dr))
    direction = QUEEN_DIRS.index((int(np.sign(df)), int(np.sign(dr))))
    return direction * 7 + dist - 1


def _build_tables() -> tuple[np.ndarray, np.ndarray]:
    """MOVE_TABLE[from, to, promo_slot] -> policy index (or -1) in oriented coordinates,
    INDEX_TABLE[index] -> (from, to, promo_slot) or (-1, -1, -1) if off-board."""
    move_table = np.full((64, 64, 5), -1, dtype=np.int16)
    index_table = np.full((POLICY_SIZE, 3), -1, dtype=np.int16)
    for frm in range(64):
        for to in range(64):
            df, dr = (to & 7) - (frm & 7), (to >> 3) - (frm >> 3)
            for slot in range(5):
                if slot in (1, 2, 3) and (frm >> 3, to >> 3) != (6, 7):
                    continue  # under-promotions only start on the 7th rank
                plane = _plane(df, dr, slot)
                if plane < 0:
                    continue
                idx = plane * 64 + frm
                move_table[frm, to, slot] = idx
                if slot in (0, 1, 2, 3):  # queen promo shares the slot-0 plane
                    index_table[idx] = (frm, to, slot)
    return move_table, index_table


MOVE_TABLE, INDEX_TABLE = _build_tables()


def orient(square: int, turn: chess.Color) -> int:
    return square if turn == chess.WHITE else square ^ 56


def move_to_index(move: chess.Move, turn: chess.Color) -> int:
    """Policy index of ``move`` for the side ``turn`` to move."""
    frm, to = orient(move.from_square, turn), orient(move.to_square, turn)
    idx = int(MOVE_TABLE[frm, to, _PROMO_SLOT[move.promotion]])
    if idx < 0:
        raise ValueError(f"move {move} is not encodable")
    return idx


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Inverse of :func:`move_to_index`; needs the board to recognise queen promotions."""
    frm, to, slot = (int(x) for x in INDEX_TABLE[index])
    if frm < 0:
        raise ValueError(f"policy index {index} does not correspond to an on-board move")
    frm, to = orient(frm, board.turn), orient(to, board.turn)
    promo = {0: None, 1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK}[slot]
    if promo is None and board.piece_type_at(frm) == chess.PAWN and chess.square_rank(to) in (0, 7):
        promo = chess.QUEEN
    return chess.Move(frm, to, promo)


def legal_move_indices(board: chess.Board) -> tuple[list[chess.Move], np.ndarray]:
    moves = list(board.legal_moves)
    idx = np.fromiter((move_to_index(m, board.turn) for m in moves), dtype=np.int64, count=len(moves))
    return moves, idx


def legal_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(POLICY_SIZE, dtype=bool)
    mask[legal_move_indices(board)[1]] = True
    return mask
