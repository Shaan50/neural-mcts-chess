"""Integration tests with a tiny randomly initialised network on CPU:
UCI protocol, match runner and self-play generation."""

import io
import json

import chess
import numpy as np
import pytest
import torch

from azchess.data.shards import legal_offsets, list_shards, load_shard
from azchess.eval.match import play_match, summarize
from azchess.eval.players import MCTSPlayer, PolicyPlayer, RandomPlayer
from azchess.mcts.pysearch import SearchParams
from azchess.model.inference import Evaluator
from azchess.model.network import AZNet, ModelConfig
from azchess.selfplay.selfplay import SelfPlayConfig, generate
from azchess.uci.engine import UCIEngine, q_to_cp


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    torch.manual_seed(0)
    cfg = ModelConfig(blocks=1, filters=8, value_channels=4, value_hidden=16)
    path = tmp_path_factory.mktemp("w") / "tiny.pt"
    torch.save({"model": AZNet(cfg).state_dict(), "ema": None, "model_config": cfg.to_dict()}, path)
    return str(path)


@pytest.fixture(scope="module")
def evaluator(weights):
    return Evaluator.from_checkpoint(weights, "cpu")


def run_uci(engine, lines):
    for line in lines:
        engine.handle(line)
    engine.wait()
    return engine.out.getvalue().splitlines()


def test_uci_protocol(weights):
    e = UCIEngine(out=io.StringIO(), weights=weights)
    e.opts["Device"] = "cpu"
    out = run_uci(e, ["uci", "isready", "position startpos moves e2e4", "go nodes 60"])
    assert "uciok" in out and "readyok" in out
    best = [line for line in out if line.startswith("bestmove")][-1].split()[1]
    b = chess.Board()
    b.push_uci("e2e4")
    assert chess.Move.from_uci(best) in b.legal_moves
    infos = [line for line in out if line.startswith("info depth")]
    assert infos and " wdl " in infos[-1] and " pv " in infos[-1]


def test_uci_tree_reuse_and_options(weights):
    e = UCIEngine(out=io.StringIO(), weights=weights)
    e.opts["Device"] = "cpu"
    run_uci(e, ["isready", "setoption name MinibatchSize value 4", "position startpos", "go nodes 100"])
    assert e.opts["MinibatchSize"] == 4
    first = e.out.getvalue().splitlines()[-1].split()[1]
    s = e.search
    run_uci(e, [f"position startpos moves {first} e7e5", "go nodes 50"])
    assert e.search is s  # same tree object, advanced by two plies


def test_uci_movetime_and_mate(weights):
    e = UCIEngine(out=io.StringIO(), weights=weights)
    e.opts["Device"] = "cpu"
    out = run_uci(e, ["position fen 7k/8/6K1/8/8/8/8/R7 w - - 0 1", "go movetime 300"])
    assert out[-1] == "bestmove a1a8"  # mate in one is found even by a random network
    out = run_uci(e, ["position fen 7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "go movetime 100"])
    assert out[-1] == "bestmove 0000"  # checkmated: no move


def test_q_to_cp_monotonic():
    qs = np.linspace(-0.99, 0.99, 50)
    cps = [q_to_cp(q) for q in qs]
    assert all(a <= b for a, b in zip(cps, cps[1:])) and q_to_cp(0) == 0


def test_match_runner_alternates_colours(evaluator):
    a = PolicyPlayer(evaluator, name="policy")
    b = RandomPlayer(seed=1)
    recs = play_match(a, b, 6, concurrency=3, verbose=False, max_plies=60)
    assert len(recs) == 6
    assert [r.white for r in recs] == ["policy", "random"] * 3
    assert recs[0].opening == recs[1].opening
    s = summarize(recs, "policy")
    assert s["wins"] + s["draws"] + s["losses"] == 6


def test_mcts_player_vs_itself(evaluator):
    a = MCTSPlayer(evaluator, sims=16, params=SearchParams(), name="a", pipeline=False)
    b = MCTSPlayer(evaluator, sims=8, params=SearchParams(), name="b", pipeline=False)
    recs = play_match(a, b, 4, concurrency=4, verbose=False, max_plies=40)
    for r in recs:
        board = chess.Board()
        for m in r.moves:
            assert chess.Move.from_uci(m) in board.legal_moves
            board.push_uci(m)


def test_selfplay_generation(evaluator, tmp_path):
    cfg = SelfPlayConfig(
        games=6, concurrency=3, sims=12, fast_sims=4, full_search_prob=0.5, max_plies=40, pipeline=False
    )
    stats = generate(evaluator, cfg, tmp_path, verbose=False)
    assert stats.games == 6
    d = load_shard(list_shards(tmp_path)[0])
    assert len(d["bb"]) == stats.positions > 0
    p = d["policy_p"].astype(np.float32).sum(1)
    assert np.allclose(p, 1.0, atol=2e-3)
    assert set(np.unique(d["result"])) <= {-1, 0, 1}
    # visited moves are a subset of the legal moves stored for each position
    off = legal_offsets(d["n_legal"])
    for i in range(len(d["bb"])):
        legal = set(d["legal"][off[i] : off[i + 1]].tolist())
        assert set(d["policy_idx"][i][d["policy_idx"][i] >= 0].tolist()) <= legal
    lines = (tmp_path / "sp_games.jsonl").read_text().splitlines()
    assert len(lines) == 6 and all("moves" in json.loads(x) for x in lines)
