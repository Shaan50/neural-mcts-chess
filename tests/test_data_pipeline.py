import io

import chess
import chess.pgn
import numpy as np
import zstandard

from neural_mcts_chess.data.build import build, movetext_tokens, split_of
from neural_mcts_chess.data.filter import GameFilter, filter_dump, iter_raw_games
from neural_mcts_chess.data.loader import ShardStream, make_batch
from neural_mcts_chess.data.shards import gather_ragged, legal_offsets, list_shards, load_shard
from neural_mcts_chess.encoding.moves import POLICY_SIZE, move_to_index

PGN = """[Event "Rated Blitz game"]
[Site "https://lichess.org/aaaaaaaa"]
[White "a"]
[Black "b"]
[Result "1-0"]
[WhiteElo "2210"]
[BlackElo "2150"]
[TimeControl "180+2"]
[Termination "Normal"]

1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] } 2. Qh5?! Nc6 3. Bc4 Nf6?? 4. Qxf7# 1-0

[Event "Rated Bullet game"]
[Site "https://lichess.org/bbbbbbbb"]
[White "c"]
[Black "d"]
[Result "0-1"]
[WhiteElo "2500"]
[BlackElo "2450"]
[TimeControl "60+0"]
[Termination "Normal"]

1. d4 d5 2. c4 e6 0-1

[Event "Rated Rapid game"]
[Site "https://lichess.org/cccccccc"]
[White "e"]
[Black "f"]
[Result "1/2-1/2"]
[WhiteElo "2300"]
[BlackElo "2350"]
[TimeControl "600+5"]
[Termination "Normal"]

1. d4 Nf6 2. c4 g6 3. Nc3 d5 4. cxd5 Nxd5 5. e4 Nxc3 6. bxc3 Bg7 1/2-1/2

[Event "Rated Blitz game"]
[Site "https://lichess.org/dddddddd"]
[White "g"]
[Black "h"]
[Result "1-0"]
[WhiteElo "1900"]
[BlackElo "2300"]
[TimeControl "300+0"]
[Termination "Time forfeit"]

1. e4 c5 1-0

"""


def test_iter_raw_games_and_filter():
    games = list(iter_raw_games(io.StringIO(PGN)))
    assert len(games) == 4
    flt = GameFilter(min_elo=2000)
    kept = [h["Site"][-8:] for h, _ in games if flt.accept(h)]
    assert kept == ["aaaaaaaa", "cccccccc"]  # bullet, low-rated and time forfeit rejected


def test_movetext_tokens_strip_comments_and_annotations():
    _h, raw = next(iter_raw_games(io.StringIO(PGN)))
    toks = movetext_tokens(raw)
    assert toks == ["e4", "e5", "Qh5?!", "Nc6", "Bc4", "Nf6??", "Qxf7#"]


def test_split_is_deterministic_and_balanced():
    ids = [f"https://lichess.org/{i:08d}" for i in range(20000)]
    splits = [split_of(i, 10, 10) for i in ids]
    assert splits == [split_of(i, 10, 10) for i in ids]
    frac_val = splits.count("val") / len(splits)
    assert 0.005 < frac_val < 0.015


def _write_zst(path, text):
    path.write_bytes(zstandard.ZstdCompressor().compress(text.encode()))


def random_pgn(n_games: int, seed: int = 0) -> str:
    """Distinct random games (realistic compressibility, unlike repeated text)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_games):
        g = chess.pgn.Game()
        g.headers.update(
            Site=f"https://lichess.org/r{i:07d}",
            Result="1/2-1/2",
            WhiteElo="2100",
            BlackElo="2200",
            TimeControl="300+3",
            Termination="Normal",
        )
        node, board = g, chess.Board()
        for _ in range(60):
            moves = list(board.legal_moves)
            if not moves:
                break
            mv = moves[int(rng.integers(len(moves)))]
            node = node.add_variation(mv)
            board.push(mv)
        out.append(str(g))
    return "\n\n".join(out) + "\n"


def test_filter_dump_handles_truncation(tmp_path):
    src = tmp_path / "dump.pgn.zst"
    data = zstandard.ZstdCompressor().compress(random_pgn(1500).encode())
    src.write_bytes(data[: int(len(data) * 0.7)])  # simulate a partial download
    stats = filter_dump(src, tmp_path / "out.pgn.zst", GameFilter())
    assert 500 < stats["kept"] < 1500


def test_build_end_to_end(tmp_path):
    src = tmp_path / "games.pgn"
    src.write_text(PGN)
    stats = build(src, tmp_path / "shards", shard_size=1000, val_permille=0, test_permille=0, workers=1)
    assert stats["games"] == 4  # build does not filter; that is filter.py's job
    d = load_shard(list_shards(tmp_path / "shards" / "train")[0])
    assert len(d["bb"]) == 7 + 4 + 12 + 2
    # Reconstruct the first game's positions and check move indices and results.
    first = np.nonzero(d["elo"] == 2210)[0]  # white positions of game 1
    assert set(d["result"][first].tolist()) == {1}
    assert set(d["result"][d["elo"] == 2150].tolist()) == {-1}
    assert set(d["result"][d["elo"] == 2350].tolist()) == {0}  # black in the drawn game
    board = chess.Board()
    expect = set()
    for san in ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"]:
        mv = board.parse_san(san)
        expect.add((board.ply(), move_to_index(mv, board.turn)))
        board.push(mv)
    game1 = (d["elo"] == 2210) | (d["elo"] == 2150)
    got = {(int(p), int(m)) for p, m in zip(d["ply"][game1], d["policy_idx"][game1, 0])}
    assert got == expect


def test_gather_ragged_matches_naive():
    rng = np.random.default_rng(0)
    lens = rng.integers(0, 6, size=50)
    vals = np.arange(lens.sum())
    off = legal_offsets(lens)
    rows = rng.choice(50, size=20)
    got, got_lens = gather_ragged(vals, off, rows)
    naive = np.concatenate([vals[off[r] : off[r + 1]] for r in rows])
    assert np.array_equal(got, naive) and np.array_equal(got_lens, lens[rows])


def test_loader_batches(tmp_path):
    src = tmp_path / "games.pgn"
    src.write_text(PGN * 3)
    build(src, tmp_path / "shards", shard_size=1000, val_permille=0, test_permille=0, workers=1)
    paths = list_shards(tmp_path / "shards" / "train")
    data = load_shard(paths[0])
    batch = make_batch(data, legal_offsets(data["n_legal"]), np.arange(5))
    assert batch["bb"].dtype.is_floating_point is False and batch["bb"].shape == (5, 12)
    rows = batch["legal_flat"] // POLICY_SIZE
    assert rows.max() == 4 and (np.diff(rows.numpy()) >= 0).all()
    # the played move is always among the legal moves
    for r in range(5):
        legal = set((batch["legal_flat"][rows == r] % POLICY_SIZE).tolist())
        assert int(batch["policy_idx"][r, 0]) in legal
    # a non-shuffled finite pass visits every position exactly once
    seen = sum(b["bb"].shape[0] for b in ShardStream(paths, 8, shuffle=False, infinite=False))
    assert seen == len(data["bb"])
