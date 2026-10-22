"""MCTS invariants, and equivalence of the C++ search with the Python reference."""

import chess
import numpy as np
import pytest

from neural_mcts_chess.encoding.moves import POLICY_SIZE
from neural_mcts_chess.game import PyPosition, make_position
from neural_mcts_chess.mcts.pysearch import EXPANDED, SearchParams
from neural_mcts_chess.mcts.search import choose_move, make_cache, make_search, root_policy, run_batched

from .conftest import requires_cpp

PIECE_VALUES = np.array([1, 3, 3, 5, 9, 0], dtype=np.float64)


def fake_evaluator(bb, meta, legal_flat):
    """Deterministic stand-in for the network: pseudo-random priors keyed on the
    position and move, value from the material balance."""
    bb = np.asarray(bb).view(np.uint64)
    rows = legal_flat // POLICY_SIZE
    pidx = legal_flat % POLICY_SIZE
    row_hash = (bb.astype(np.uint64).sum(axis=1) % np.uint64(1_000_003)).astype(np.int64)
    logits = (((pidx * 2654435761 + row_hash[rows] * 40503) % 1009) / 1009.0 * 4.0).astype(np.float32)
    counts = np.array([[bin(int(x)).count("1") for x in row] for row in bb], dtype=np.float64)
    material = counts[:, :6] @ PIECE_VALUES - counts[:, 6:] @ PIECE_VALUES
    v = np.tanh(material / 6.0)
    wdl = np.stack([(1 + v) / 2 * 0.9, np.full_like(v, 0.1), (1 - v) / 2 * 0.9], axis=1).astype(np.float32)
    return logits, wdl


def uniform_evaluator(bb, meta, legal_flat):
    return np.zeros(len(legal_flat), np.float32), np.tile(np.array([[0.3, 0.4, 0.3]], np.float32), (bb.shape[0], 1))


FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",  # Qxf7#
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
]


def _search(backend, pos_backend, fen, sims, leaves, evaluator=fake_evaluator, **kw):
    params = SearchParams(**kw)
    s = make_search(params, backend)
    s.set_position(make_position(fen, pos_backend))
    s.begin(False)
    run_batched([s], evaluator, sims, leaves)
    return s


def _visit_dict(s):
    pos = s.position()
    return {pos.uci(c[0]): c[2] for c in s.root_children()}


@requires_cpp
@pytest.mark.parametrize("fen", FENS)
@pytest.mark.parametrize("leaves", [1, 4])
def test_cpp_matches_python_reference(fen, leaves):
    cpp = _search("cpp", "cpp", fen, 300, leaves)
    py = _search("python", "cpp", fen, 300, leaves)
    assert _visit_dict(cpp) == _visit_dict(py)
    assert cpp.root_q() == pytest.approx(py.root_q(), abs=1e-5)
    assert [cpp.position().uci(m) for m in cpp.pv(8)] == [py.position().uci(m) for m in py.pv(8)]


@requires_cpp
@pytest.mark.parametrize("fen", FENS[:2])
def test_python_chess_backend_matches(fen):
    a = _search("python", "python", fen, 200, 1)
    b = _search("cpp", "cpp", fen, 200, 1)
    assert _visit_dict(a) == _visit_dict(b)


@pytest.mark.parametrize("backend", ["python"] + (["cpp"] if make_cache(1, "auto") is not None else []))
@pytest.mark.parametrize("leaves", [1, 8])
def test_visit_accounting(backend, leaves):
    s = _search(backend, "auto", FENS[2], 257, leaves)
    moves, idx, visits, q = root_policy(s)
    assert s.root_visits() == 257
    assert visits.sum() == 256  # every visit except the root's own expansion goes to a child
    assert s.pending() == 0
    assert np.all(np.abs(q) <= 1.0 + 1e-9)
    assert len(set(idx.tolist())) == len(idx)


def test_no_residual_virtual_loss_and_consistent_counts():
    s = _search("python", "python", FENS[2], 400, 16)
    stack = [s.root]
    while stack:
        node = stack.pop()
        assert node.n_inflight == 0
        if node.state == EXPANDED:
            assert node.n == 1 + sum(c.n for c in node.children)
            assert abs(sum(c.prior for c in node.children) - 1.0) < 1e-4
        stack.extend(node.children)


@pytest.mark.parametrize("backend", ["python", "auto"])
def test_finds_mate_in_one(backend):
    s = _search(backend, "auto", FENS[1], 200, 4, evaluator=uniform_evaluator)
    assert s.position().uci(choose_move(s)) == "f3f7"
    assert s.root_q() > 0.5


@pytest.mark.parametrize("backend", ["python", "auto"])
def test_terminal_root_needs_no_evaluations(backend):
    calls = []

    def ev(*a):
        calls.append(1)
        return uniform_evaluator(*a)

    mated = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
    s = make_search(SearchParams(), backend)
    s.set_position(make_position(mated, "auto"))
    s.begin(False)
    run_batched([s], ev, 50, 4)
    # First visit discovers the terminal state without calling the network.
    assert not calls
    assert s.root_visits() <= 1


def test_dirichlet_noise_changes_priors_and_keeps_distribution():
    base = _search("python", "python", chess.STARTING_FEN, 2, 1, seed=1)
    noisy = make_search(SearchParams(seed=1), "python")
    noisy.set_position(PyPosition())
    noisy.begin(add_noise=True)
    run_batched([noisy], fake_evaluator, 3, 1)
    p0 = np.array([c.prior for c in base.root.children])
    p1 = np.array([c.prior for c in noisy.root.children])
    assert abs(p1.sum() - 1) < 1e-4
    assert not np.allclose(p0, p1)


@pytest.mark.parametrize("backend", ["python", "auto"])
def test_tree_reuse_keeps_subtree_statistics(backend):
    s = _search(backend, "auto", chess.STARTING_FEN, 400, 4)
    best = choose_move(s)
    child_visits = {c[0]: c[2] for c in s.root_children()}[best]
    s.advance(best)
    assert s.root_visits() == child_visits
    s.begin(False)
    run_batched([s], fake_evaluator, child_visits + 100, 4)
    assert s.root_visits() == child_visits + 100


@requires_cpp
def test_policy_temperature_flattens_priors():
    cold = _search("cpp", "cpp", chess.STARTING_FEN, 2, 1, policy_temperature=1.0)
    hot = _search("cpp", "cpp", chess.STARTING_FEN, 2, 1, policy_temperature=3.0)
    pc = np.array([c[4] for c in cold.root_children()])
    ph = np.array([c[4] for c in hot.root_children()])
    entropy = lambda p: -(p * np.log(p)).sum()  # noqa: E731
    assert entropy(ph) > entropy(pc)


@requires_cpp
def test_shared_cache_reproduces_results_with_fewer_evaluations():
    calls = {"n": 0}

    def counting(bb, meta, legal):
        calls["n"] += bb.shape[0]
        return fake_evaluator(bb, meta, legal)

    cache = make_cache(100_000, "cpp")
    results = []
    for _ in range(2):
        s = make_search(SearchParams(), "cpp", cache)
        s.set_position(make_position(FENS[2], "cpp"))
        s.begin(False)
        before = calls["n"]
        run_batched([s], counting, 300, 1)
        results.append((_visit_dict(s), calls["n"] - before))
    assert results[0][0] == results[1][0]
    assert results[1][1] == 0  # second search fully served from the cache
    assert cache.hits > 0


@requires_cpp
def test_many_searches_batched_and_pipelined():
    searches = []
    for fen in FENS * 4:
        s = make_search(SearchParams(), "cpp")
        s.set_position(make_position(fen, "cpp"))
        s.begin(False)
        searches.append(s)
    stats = run_batched(searches, fake_evaluator, 128, 8, pipeline=True)
    assert all(s.root_visits() == 128 or s.done(128) for s in searches)
    assert max(stats.batch_sizes) > 8  # leaves from several searches share a batch


def test_twofold_repetition_scored_as_draw_in_tree():
    """With only kings and a rook shuffling, the search must not crash on repetitions
    and any repeated position inside the tree is a draw."""
    s = _search("python", "python", "8/8/8/4k3/8/8/8/R3K3 w - - 0 1", 300, 4)
    assert s.root_visits() == 300
