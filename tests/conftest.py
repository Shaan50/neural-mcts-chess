import random

import chess
import pytest

from neural_mcts_chess.game import cpp_available

# Positions that exercise castling, en passant, promotions and pins.
TRICKY_FENS = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # kiwipete
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",  # legal ep
    "8/8/8/K2pP2r/8/8/8/7k w - d6 0 1",  # ep capture illegal (horizontal pin)
    "4k3/1P6/8/8/8/8/6p1/4K3 w - - 0 1",  # promotions both ways
]

requires_cpp = pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")


def random_boards(n_games: int, seed: int = 0, max_plies: int = 160):
    """Yield python-chess boards visited during random games (biased towards
    captures/promotions so that endgames and special moves appear)."""
    rng = random.Random(seed)
    for _ in range(n_games):
        board = chess.Board(rng.choice(TRICKY_FENS))
        for _ in range(rng.randint(1, max_plies)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over(claim_draw=False):
                break
            special = [m for m in moves if m.promotion or board.is_capture(m) or board.is_castling(m)]
            mv = rng.choice(special) if special and rng.random() < 0.4 else rng.choice(moves)
            board.push(mv)
            yield board
