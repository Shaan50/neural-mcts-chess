import math

import numpy as np
import pytest

from azchess.eval.elo import elo_from_score, expected_score, mle_rating, performance, sprt
from azchess.experiments.ablations import SF_MAX, SF_MIN, ladder_levels


def test_expected_score_and_inverse():
    assert expected_score(0) == pytest.approx(0.5)
    assert expected_score(400) == pytest.approx(10 / 11)
    assert expected_score(-200) == pytest.approx(1 - expected_score(200))
    for d in (-500, -100, 0, 50, 300):
        assert elo_from_score(float(expected_score(d))) == pytest.approx(d, abs=1e-6)


def test_performance_ci():
    p = performance(50, 0, 50)
    assert p.elo == pytest.approx(0, abs=1e-9) and p.lo < 0 < p.hi
    p = performance(30, 40, 30)  # draws shrink the variance
    q = performance(50, 0, 50)
    assert (p.hi - p.lo) < (q.hi - q.lo)
    z = performance(0, 0, 10)
    assert math.isfinite(z.elo) and math.isfinite(z.hi) and z.lo < z.hi


def test_mle_recovers_true_rating():
    rng = np.random.default_rng(1)
    true = 1850.0
    opps = np.repeat([1500, 1700, 1900, 2100], 400)
    p_win = expected_score(true - opps)
    scores = (rng.random(len(opps)) < p_win).astype(float)
    r = mle_rating(opps, scores, bootstrap=200)
    assert abs(r.elo - true) < 3 * r.se
    assert r.boot_lo < r.elo < r.boot_hi
    # the MLE satisfies the score equation: expected total score == actual total score
    assert expected_score(r.elo - opps).sum() == pytest.approx(scores.sum(), rel=0.02)


def test_mle_all_wins_is_finite():
    r = mle_rating([1500] * 20, [1.0] * 20)
    assert math.isfinite(r.elo) and r.elo > 1700


def test_sprt_decisions():
    assert sprt(300, 200, 100).decision == "H1"
    assert sprt(100, 200, 300).decision == "H0"
    assert sprt(5, 10, 5).decision == "continue"


@pytest.mark.parametrize("est", [800, 1100, 1480, 1530, 2000, 2975, 3300])
def test_ladder_levels_stay_in_stockfish_range(est):
    lv = ladder_levels(est)
    assert len(lv) == 3 and lv == sorted(lv)
    assert SF_MIN <= lv[0] and lv[-1] <= SF_MAX
    assert lv[1] - lv[0] == 200
