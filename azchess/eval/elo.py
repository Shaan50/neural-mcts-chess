"""Elo estimation from match results.

* :func:`performance` - Elo difference implied by a W/D/L record against one
  opponent, with a 95% CI from the trinomial (W/D/L) variance of the score.
* :func:`mle_rating` - maximum-likelihood rating against several opponents of
  known rating (e.g. a ladder of calibrated Stockfish levels), with a CI from
  the Fisher information and optionally a bootstrap over games.
* :func:`sprt` - generalised sequential probability ratio test (normal
  approximation, as used by fishtest) for "is the new net stronger?" gating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

LN10_400 = math.log(10) / 400


def expected_score(elo_diff: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(elo_diff) / 400.0))


def elo_from_score(score: float) -> float:
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / score - 1.0)


@dataclass
class Performance:
    elo: float
    lo: float
    hi: float
    score: float
    games: int

    def __str__(self) -> str:
        return f"{self.elo:+.0f} [{self.lo:+.0f}, {self.hi:+.0f}] (score {self.score:.3f}, n={self.games})"


def performance(wins: int, draws: int, losses: int, z: float = 1.96) -> Performance:
    """Elo difference from a W/D/L record. The CI is a Wilson-style score
    interval using the trinomial variance, so it stays sensible (and
    non-degenerate) for 0% / 100% scores."""
    n = wins + draws + losses
    if n == 0:
        return Performance(0.0, -math.inf, math.inf, 0.5, 0)
    s = (wins + 0.5 * draws) / n
    var = (wins * (1 - s) ** 2 + draws * (0.5 - s) ** 2 + losses * s**2) / n
    denom = 1 + z * z / n
    center = (s + z * z / (2 * n)) / denom
    half = z * math.sqrt(var / n + z * z / (4 * n * n)) / denom
    eps = 0.5 / n  # clamp so perfect scores give finite (if wide) estimates
    clamp = lambda x: min(max(x, eps), 1 - eps)  # noqa: E731
    return Performance(
        elo_from_score(clamp(s)), elo_from_score(clamp(center - half)), elo_from_score(clamp(center + half)), s, n
    )


@dataclass
class Rating:
    elo: float
    se: float
    lo: float
    hi: float
    games: int
    boot_lo: float | None = None
    boot_hi: float | None = None

    def __str__(self) -> str:
        s = f"{self.elo:.0f} ± {1.96 * self.se:.0f} (95% CI [{self.lo:.0f}, {self.hi:.0f}], n={self.games})"
        if self.boot_lo is not None:
            s += f" bootstrap [{self.boot_lo:.0f}, {self.boot_hi:.0f}]"
        return s


def mle_rating(
    opp_elos, scores, z: float = 1.96, bootstrap: int = 0, seed: int = 0, prior_games: float = 0.5
) -> Rating:
    """MLE of a player's rating from per-game (opponent rating, score in {0, .5, 1}).

    The score likelihood is the usual Bernoulli/Bradley-Terry quasi-likelihood
    (draws count as half a win). ``prior_games`` adds that many virtual draws
    against the mean opponent so that 100% or 0% scores stay finite.
    """
    opp = np.asarray(opp_elos, dtype=np.float64)
    sc = np.asarray(scores, dtype=np.float64)
    n = len(sc)
    if prior_games > 0:
        opp = np.concatenate([opp, [opp.mean()] * 2])
        sc = np.concatenate([sc, [0.5, 0.5]])
        w = np.concatenate([np.ones(n), [prior_games / 2] * 2])
    else:
        w = np.ones(n)
    r, info = _mle_weighted(opp, sc, w, float(opp.mean()))
    se = 1.0 / math.sqrt(max(info, 1e-12))
    rating = Rating(r, se, r - z * se, r + z * se, n)
    if bootstrap > 0 and n > 1:
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(bootstrap):
            idx = rng.integers(0, n, n)
            o = np.concatenate([opp[idx], opp[n:]])
            s = np.concatenate([sc[idx], sc[n:]])
            boots.append(_mle_weighted(o, s, w, r)[0])
        rating.boot_lo, rating.boot_hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    return rating


def _mle_weighted(opp, score, weight, init, iters: int = 200) -> tuple[float, float]:
    r = init
    info = 1e-9
    for _ in range(iters):
        e = expected_score(r - opp)
        grad = LN10_400 * np.sum(weight * (score - e))
        info = LN10_400**2 * np.sum(weight * e * (1 - e))
        step = float(np.clip(grad / max(info, 1e-12), -200, 200))
        r += step
        if abs(step) < 1e-7:
            break
    return r, info


@dataclass
class SPRTResult:
    llr: float
    lower: float
    upper: float
    decision: str  # "H1" (stronger), "H0" (not stronger), "continue"


def sprt(
    wins: int, draws: int, losses: int, elo0: float = 0.0, elo1: float = 30.0, alpha: float = 0.05, beta: float = 0.05
) -> SPRTResult:
    """GSPRT with normal approximation on per-game scores (logistic Elo)."""
    lower, upper = math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)
    n = wins + draws + losses
    if n == 0 or wins + losses == 0:
        return SPRTResult(0.0, lower, upper, "continue")
    s = (wins + 0.5 * draws) / n
    var = (wins * (1 - s) ** 2 + draws * (0.5 - s) ** 2 + losses * s**2) / n
    if var <= 0:
        return SPRTResult(0.0, lower, upper, "continue")
    s0, s1 = float(expected_score(elo0)), float(expected_score(elo1))
    llr = n * (s1 - s0) * (2 * s - s0 - s1) / (2 * var)
    decision = "H1" if llr >= upper else "H0" if llr <= lower else "continue"
    return SPRTResult(llr, lower, upper, decision)
