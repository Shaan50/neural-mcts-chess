"""Cross-checks the C++ move generator against python-chess."""

import pickle
import random

import chess
import numpy as np
import pytest

from neural_mcts_chess.encoding.moves import move_to_index
from neural_mcts_chess.game import PyPosition

from .conftest import TRICKY_FENS, random_boards, requires_cpp

pytestmark = requires_cpp

PERFT = [
    (chess.STARTING_FEN, [20, 400, 8902, 197281, 4865609]),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", [48, 2039, 97862, 4085603]),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43238, 674624]),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", [6, 264, 9467, 422333]),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [44, 1486, 62379, 2103487]),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", [46, 2079, 89890, 3894594]),
]


@pytest.fixture(scope="module")
def core():
    from neural_mcts_chess import _core

    return _core


@pytest.mark.parametrize("fen,counts", PERFT)
def test_perft(core, fen, counts):
    pos = core.Position(fen)
    for depth, expected in enumerate(counts, start=1):
        assert pos.perft(depth) == expected, f"perft({depth}) of {fen}"


@pytest.mark.parametrize("fen", TRICKY_FENS)
def test_fen_roundtrip(core, fen):
    assert core.Position(fen).fen() == chess.Board(fen).fen()


def _check_equal(core_pos, board: chess.Board, py_pos: PyPosition):
    assert core_pos.fen() == board.fen()
    cm = sorted(core_pos.uci(m) for m in core_pos.legal_moves())
    pm = sorted(m.uci() for m in board.legal_moves)
    assert cm == pm
    assert core_pos.is_check() == board.is_check()
    assert core_pos.terminal() == py_pos.terminal()
    assert core_pos.repetition_count() == py_pos.repetition_count()
    cbb, cmeta = core_pos.encode()
    pbb, pmeta = py_pos.encode()
    assert np.array_equal(cbb, pbb) and np.array_equal(cmeta, pmeta)
    moves = core_pos.legal_moves()
    cidx = dict(zip((core_pos.uci(m) for m in moves), core_pos.policy_indices(moves)))
    for m in board.legal_moves:
        assert cidx[m.uci()] == move_to_index(m, board.turn)


def test_random_games_match_python_chess(core):
    rng = random.Random(7)
    n = 0
    for _game in range(120):
        fen = rng.choice(TRICKY_FENS)
        board = chess.Board(fen)
        py_pos = PyPosition(fen)
        cpos = core.Position(fen)
        for _ply in range(200):
            _check_equal(cpos, board, py_pos)
            n += 1
            if py_pos.terminal() != 0:
                break
            moves = list(board.legal_moves)
            mv = rng.choice(moves)
            board.push(mv)
            py_pos.push(mv)
            cpos.push_uci(mv.uci())
        # unwind a few moves and re-check pop()
        for _ in range(min(10, len(board.move_stack))):
            board.pop()
            py_pos.pop()
            cpos.pop()
            _check_equal(cpos, board, py_pos)
    assert n > 5000


def test_repetition_and_draw_codes(core):
    cpos, py_pos = core.Position(), PyPosition()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        cpos.push_uci(uci)
        py_pos.push_uci(uci)
    assert cpos.repetition_count() == py_pos.repetition_count() == 3
    assert cpos.terminal() == py_pos.terminal() == 5
    assert cpos.repeated_within(4)
    assert core.Position("8/8/8/4k3/8/8/8/4K1N1 w - - 0 1").terminal() == 3  # insufficient
    assert core.Position("8/8/8/4k3/8/8/8/R3K3 w - - 100 80").terminal() == 4  # fifty moves
    assert core.Position("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1").terminal() == 1  # checkmate
    assert core.Position("7k/8/6QK/8/8/8/8/8 b - - 0 1").terminal() == 2  # stalemate


def test_insufficient_material_matches(core):
    for b in random_boards(300, seed=11, max_plies=300):
        assert core.Position(b.fen()).is_insufficient_material() == b.is_insufficient_material(), b.fen()


def test_san_parsing(core):
    for b in random_boards(80, seed=5):
        if b.is_game_over():
            continue
        cpos = core.Position(b.fen())
        for m in b.legal_moves:
            assert cpos.uci(cpos.parse_san(b.san(m))) == m.uci()


def test_encode_game_matches_python(core):
    rng = random.Random(9)
    board = chess.Board()
    sans = []
    for _ in range(120):
        moves = list(board.legal_moves)
        if not moves:
            break
        mv = rng.choice(moves)
        sans.append(board.san(mv))
        board.push(mv)
    bb, meta, move_idx, legal, n_legal, complete, term = core.encode_game(chess.STARTING_FEN, sans)
    assert complete == 1 and len(move_idx) == len(sans)
    py_pos = PyPosition()
    off = 0
    replay = chess.Board()
    for i, san in enumerate(sans):
        pbb, pmeta = py_pos.encode()
        assert np.array_equal(bb[i], pbb) and np.array_equal(meta[i], pmeta)
        mv = replay.parse_san(san)
        assert move_idx[i] == move_to_index(mv, replay.turn)
        expect = sorted(move_to_index(m, replay.turn) for m in replay.legal_moves)
        assert sorted(legal[off : off + n_legal[i]].tolist()) == expect
        off += n_legal[i]
        replay.push(mv)
        py_pos.push(mv)


def test_encode_game_stops_on_illegal(core):
    out = core.encode_game(chess.STARTING_FEN, ["e4", "e5", "Ke3"])
    assert len(out[2]) == 2 and out[5] == 0


def test_pickle_and_copy(core):
    pos = core.Position()
    for uci in ["e2e4", "c7c5", "g1f3"]:
        pos.push_uci(uci)
    clone = pickle.loads(pickle.dumps(pos))
    assert clone.fen() == pos.fen() and clone.move_stack() == pos.move_stack()
    cp = pos.copy()
    cp.push_uci("d7d6")
    assert cp.fen() != pos.fen()
