import chess
import numpy as np
import pytest

from azchess.encoding.moves import (
    INDEX_TABLE,
    MOVE_TABLE,
    POLICY_SIZE,
    index_to_move,
    legal_mask,
    move_to_index,
)

from .conftest import TRICKY_FENS, random_boards


def test_policy_size_and_table_shapes():
    assert POLICY_SIZE == 4672
    assert MOVE_TABLE.shape == (64, 64, 5)
    assert INDEX_TABLE.shape == (POLICY_SIZE, 3)


def test_number_of_geometrically_valid_indices():
    # 1792 queen/knight moves + 66 under-promotions (the same count Lc0 uses).
    assert int((INDEX_TABLE[:, 0] >= 0).sum()) == 1858


def test_known_indices():
    b = chess.Board()
    # e2e4: direction N (0), distance 2 -> plane 1, from e2 (12)
    assert move_to_index(chess.Move.from_uci("e2e4"), chess.WHITE) == 1 * 64 + 12
    # g1f3: knight (-1, +2) is KNIGHT_DIRS[7] -> plane 63
    assert move_to_index(chess.Move.from_uci("g1f3"), chess.WHITE) == 63 * 64 + 6
    b.push_uci("e2e4")
    # black's e7e5 is the mirror image of white's e2e4
    assert move_to_index(chess.Move.from_uci("e7e5"), chess.BLACK) == 1 * 64 + 12


def test_underpromotion_planes():
    assert move_to_index(chess.Move.from_uci("a7a8n"), chess.WHITE) == (64 + 1 * 3 + 0) * 64 + 48
    assert move_to_index(chess.Move.from_uci("a7b8r"), chess.WHITE) == (64 + 2 * 3 + 2) * 64 + 48
    # queen promotion uses the ordinary queen-move plane
    assert move_to_index(chess.Move.from_uci("a7a8q"), chess.WHITE) == move_to_index(
        chess.Move.from_uci("a7a8"), chess.WHITE
    )
    # black under-promotion capturing towards the h-file (+1 file from its perspective too)
    assert move_to_index(chess.Move.from_uci("b2c1b"), chess.BLACK) == (64 + 2 * 3 + 1) * 64 + (9 ^ 56)


@pytest.mark.parametrize("fen", TRICKY_FENS)
def test_roundtrip_tricky(fen):
    b = chess.Board(fen)
    for mv in b.legal_moves:
        idx = move_to_index(mv, b.turn)
        assert 0 <= idx < POLICY_SIZE
        assert index_to_move(idx, b) == mv


def test_roundtrip_and_uniqueness_random_positions():
    n_positions = 0
    for b in random_boards(150, seed=1):
        if b.is_game_over():
            continue
        idx = [move_to_index(m, b.turn) for m in b.legal_moves]
        assert len(set(idx)) == len(idx), "two legal moves share a policy index"
        for m, i in zip(b.legal_moves, idx):
            assert index_to_move(i, b) == m
        mask = legal_mask(b)
        assert mask.sum() == len(idx)
        n_positions += 1
    assert n_positions > 1000


def test_color_mirror_symmetry():
    """A position and its colour-flipped mirror have identical policy indices."""
    for b in random_boards(40, seed=2):
        m = b.mirror()
        a = sorted(move_to_index(mv, b.turn) for mv in b.legal_moves)
        c = sorted(move_to_index(mv, m.turn) for mv in m.legal_moves)
        assert a == c


def test_off_board_indices_rejected():
    invalid = np.nonzero(INDEX_TABLE[:, 0] < 0)[0]
    with pytest.raises(ValueError):
        index_to_move(int(invalid[0]), chess.Board())
