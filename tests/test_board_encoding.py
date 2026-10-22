import chess
import numpy as np
import pytest
import torch

from azchess.encoding.board import (
    NO_EP,
    NUM_PLANES,
    bb_as_int64,
    decode_planes,
    encode_compact,
    planes_from_compact_torch,
)
from azchess.game import PyPosition

from .conftest import random_boards


def _encode(board: chess.Board):
    return PyPosition.from_board(board).encode()


def test_start_position_planes():
    bb, meta = encode_compact(chess.Board())
    planes = decode_planes(bb[None], meta[None])[0]
    assert planes.shape == (NUM_PLANES, 8, 8)
    assert planes[0, 1].sum() == 8  # our pawns on rank 2
    assert planes[6, 6].sum() == 8  # their pawns on rank 7
    assert planes[5, 0, 4] == 1  # our king e1
    assert planes[11, 7, 4] == 1  # their king e8
    assert planes[15:19].min() == 1  # all castling rights
    assert planes[19].max() == 0  # white to move
    assert planes[14].max() == 0  # no ep
    assert planes[21].min() == 1
    assert meta[0] == NO_EP


def test_black_to_move_is_mirrored():
    b = chess.Board()
    b.push_uci("e2e4")
    bb, meta = encode_compact(b)
    planes = decode_planes(bb[None], meta[None])[0]
    # Black's pawns are now "ours" and on (oriented) rank 2.
    assert planes[0, 1].sum() == 8
    assert planes[5, 0, 4] == 1  # black king e8 -> oriented e1
    assert planes[6, 4, 4] == 1  # white pawn e4 -> oriented e5 (row 4)
    assert planes[19].min() == 1


def test_mirror_invariance():
    """Encoding a position and its colour-flipped mirror differs only in the colour plane."""
    for b in random_boards(60, seed=3):
        bb1, m1 = encode_compact(b)
        bb2, m2 = encode_compact(b.mirror())
        assert np.array_equal(bb1, bb2)
        assert m1[0] == m2[0] and m1[2] == m2[2]
        assert (m1[1] & 0x0F) == (m2[1] & 0x0F)
        assert (m1[1] >> 4) != (m2[1] >> 4)


def test_en_passant_plane_only_when_legal():
    legal = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    bb, meta = encode_compact(legal)
    assert meta[0] == chess.F6
    pinned = chess.Board("8/8/8/K2pP2r/8/8/8/7k w - d6 0 1")
    assert encode_compact(pinned)[1][0] == NO_EP


def test_repetition_planes():
    pos = PyPosition()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        pos.push_uci(uci)
    bb, meta = pos.encode()
    assert meta[3] == 2  # start position seen twice before
    planes = decode_planes(bb[None], meta[None])[0]
    assert planes[12].min() == 1 and planes[13].min() == 1


def test_halfmove_plane():
    b = chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 37 50")
    bb, meta = encode_compact(b)
    planes = decode_planes(bb[None], meta[None])[0]
    assert np.allclose(planes[20], 0.37)


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
def test_torch_decoder_matches_numpy(device):
    boards = list(random_boards(30, seed=4))
    encs = [_encode(b) for b in boards]
    bb = np.stack([e[0] for e in encs])
    meta = np.stack([e[1] for e in encs])
    ref = decode_planes(bb, meta)
    got = planes_from_compact_torch(torch.from_numpy(bb_as_int64(bb)).to(device), torch.from_numpy(meta).to(device))
    np.testing.assert_allclose(got.cpu().numpy(), ref, atol=1e-6)


def test_bit63_roundtrip():
    """h8 (bit 63) must survive the int64 reinterpretation used for torch."""
    b = chess.Board("4k2r/8/8/8/8/8/8/4K3 w - - 0 1")
    bb, meta = encode_compact(b)
    assert bb[9] == 1 << 63  # their rook on h8
    planes = planes_from_compact_torch(torch.from_numpy(bb_as_int64(bb[None])), torch.from_numpy(meta[None]))
    assert planes[0, 9, 7, 7] == 1
