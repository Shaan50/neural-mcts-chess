"""Board -> network input encoding.

Positions are stored and transported in a compact, bit-packed form and only
expanded to dense float planes right before the network (on the GPU when
available). A position costs 100 bytes instead of 22*64*4 = 5632 bytes.

Compact form (always from the side to move's perspective, board mirrored
vertically when black is to move):

* ``bb``   uint64[12]  piece bitboards: ours P N B R Q K, then theirs P N B R Q K
* ``meta`` uint8[4]
    0: en-passant target square (oriented), 64 if none / not legally capturable
    1: flags: bit0 our O-O, bit1 our O-O-O, bit2 their O-O, bit3 their O-O-O,
       bit4 side to move is black
    2: halfmove clock (capped at 255)
    3: number of earlier occurrences of this position (capped at 2)

Dense planes (``NUM_PLANES`` = 22, each 8x8, row = rank, col = file):

    0-11  piece planes                 12  repeated once before
    13    repeated twice before        14  en-passant square
    15-18 castling rights (ours K, Q, theirs K, Q)
    19    side to move is black        20  halfmove clock / 100
    21    constant ones (lets convolutions see the board edge)
"""

from __future__ import annotations

import chess
import numpy as np

NUM_PLANES = 22
NO_EP = 64

_PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def _flip(bb: int) -> int:
    return chess.flip_vertical(bb)


def encode_compact(board: chess.Board, prior_occurrences: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Encode a python-chess board. ``prior_occurrences`` is the number of times
    the exact position (transposition key) appeared earlier in the game."""
    us, them = board.turn, not board.turn
    bb = np.zeros(12, dtype=np.uint64)
    for i, pt in enumerate(_PIECE_ORDER):
        ours = board.pieces_mask(pt, us)
        theirs = board.pieces_mask(pt, them)
        if us == chess.BLACK:
            ours, theirs = _flip(ours), _flip(theirs)
        bb[i], bb[6 + i] = ours, theirs

    meta = np.zeros(4, dtype=np.uint8)
    ep = board.ep_square if board.has_legal_en_passant() else None
    meta[0] = NO_EP if ep is None else (ep if us == chess.WHITE else ep ^ 56)
    flags = 0
    flags |= int(board.has_kingside_castling_rights(us)) << 0
    flags |= int(board.has_queenside_castling_rights(us)) << 1
    flags |= int(board.has_kingside_castling_rights(them)) << 2
    flags |= int(board.has_queenside_castling_rights(them)) << 3
    flags |= int(us == chess.BLACK) << 4
    meta[1] = flags
    meta[2] = min(board.halfmove_clock, 255)
    meta[3] = min(prior_occurrences, 2)
    return bb, meta


_BIT_SHIFTS = np.arange(64, dtype=np.uint64)


def decode_planes(bb: np.ndarray, meta: np.ndarray) -> np.ndarray:
    """Vectorised compact -> dense planes. bb: (N, 12) uint64, meta: (N, 4) uint8."""
    bb = np.asarray(bb, dtype=np.uint64)
    meta = np.asarray(meta, dtype=np.uint8)
    n = bb.shape[0]
    out = np.zeros((n, NUM_PLANES, 64), dtype=np.float32)
    out[:, :12, :] = ((bb[:, :, None] >> _BIT_SHIFTS) & np.uint64(1)).astype(np.float32)
    rep = meta[:, 3]
    out[:, 12, :] = (rep >= 1)[:, None]
    out[:, 13, :] = (rep >= 2)[:, None]
    has_ep = meta[:, 0] < NO_EP
    out[np.nonzero(has_ep)[0], 14, meta[has_ep, 0]] = 1.0
    flags = meta[:, 1]
    for k in range(4):
        out[:, 15 + k, :] = ((flags >> k) & 1)[:, None]
    out[:, 19, :] = ((flags >> 4) & 1)[:, None]
    out[:, 20, :] = (meta[:, 2].astype(np.float32) / 100.0)[:, None]
    out[:, 21, :] = 1.0
    return out.reshape(n, NUM_PLANES, 8, 8)


def planes_from_compact_torch(bb, meta):
    """Same as :func:`decode_planes` but on torch tensors (runs on GPU).

    ``bb`` is int64 (the uint64 bit pattern reinterpreted), ``meta`` uint8/int.
    Bit extraction with arithmetic shifts is still correct for bit 63 because
    we mask with 1 after shifting.
    """
    import torch

    n = bb.shape[0]
    device = bb.device
    shifts = torch.arange(64, device=device, dtype=torch.int64)
    pieces = ((bb.unsqueeze(-1) >> shifts) & 1).to(torch.float32)  # (N, 12, 64)
    meta = meta.to(torch.int64)
    rep = meta[:, 3]
    flags = meta[:, 1]
    scal = torch.stack(
        [
            (rep >= 1).float(),
            (rep >= 2).float(),
            torch.zeros(n, device=device),
            (flags & 1).float(),
            ((flags >> 1) & 1).float(),
            ((flags >> 2) & 1).float(),
            ((flags >> 3) & 1).float(),
            ((flags >> 4) & 1).float(),
            meta[:, 2].float() / 100.0,
            torch.ones(n, device=device),
        ],
        dim=1,
    )  # (N, 10)
    rest = scal.unsqueeze(-1).expand(n, 10, 64).clone()
    ep = meta[:, 0]
    ep_plane = torch.zeros(n, 65, device=device)  # column 64 swallows "no ep"
    ep_plane.scatter_(1, ep.clamp(max=64).unsqueeze(1), 1.0)
    rest[:, 2, :] = ep_plane[:, :64]
    return torch.cat([pieces, rest], dim=1).view(n, NUM_PLANES, 8, 8)


def bb_as_int64(bb: np.ndarray) -> np.ndarray:
    """Reinterpret uint64 bitboards as int64 (torch has limited uint64 support)."""
    return np.ascontiguousarray(bb, dtype=np.uint64).view(np.int64)
