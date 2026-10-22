"""Evaluate a player against a ladder of opponents and estimate its Elo.

    python -m neural_mcts_chess.eval.run --player nn:path=runs/medium/best.pt,sims=400 \
        --opponents sf:elo=1320 sf:elo=1600 sf:elo=1900 --games 40 --out results/medium_400.json

Opponents with a known rating (Stockfish ``elo=``) contribute to a joint
maximum-likelihood rating on Stockfish's UCI_Elo scale; every opponent also
gets a per-opponent performance rating with a 95% confidence interval.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from neural_mcts_chess.eval.elo import mle_rating, performance
from neural_mcts_chess.eval.match import play_match, records_to_json, summarize
from neural_mcts_chess.eval.players import build_player, parse_spec


def known_elo(spec: str) -> float | None:
    kind, kv = parse_spec(spec)
    if kind == "sf" and "elo" in kv:
        return float(kv["elo"])
    if "rating" in kv:
        return float(kv["rating"])
    return None


def evaluate(
    player_spec: str,
    opponent_specs: list[str],
    games: int = 40,
    concurrency: int = 16,
    device: str = "auto",
    pgn: str | None = None,
    bootstrap: int = 1000,
    max_plies: int = 400,
    evaluators: dict | None = None,
    verbose: bool = True,
) -> dict:
    evaluators = evaluators if evaluators is not None else {}
    player = build_player(player_spec, device, evaluators)
    per_game_opp, per_game_score = [], []
    opponents = []
    t0 = time.time()
    for spec in opponent_specs:
        opp = build_player(spec, device, evaluators)
        if opp.name == player.name:
            opp.name += "_opp"
        try:
            recs = play_match(
                player, opp, games, concurrency=concurrency, verbose=verbose, pgn_path=pgn, max_plies=max_plies
            )
        finally:
            opp.close()
        s = summarize(recs, player.name)
        perf = performance(s["wins"], s["draws"], s["losses"])
        elo = known_elo(spec)
        entry = {
            "spec": spec,
            "name": opp.name,
            "known_elo": elo,
            **s,
            "perf_diff": perf.elo,
            "perf_lo": perf.lo,
            "perf_hi": perf.hi,
        }
        if elo is not None:
            entry["perf_rating"] = elo + perf.elo
            per_game_opp += [elo] * len(recs)
            per_game_score += [r.score_for(player.name) for r in recs]
        terminations: dict[str, int] = {}
        for r in recs:
            terminations[r.termination] = terminations.get(r.termination, 0) + 1
        entry["terminations"] = terminations
        entry["games_detail"] = records_to_json(recs)
        opponents.append(entry)
        if verbose:
            print(
                f"[eval] {player.name} vs {opp.name}: +{s['wins']} ={s['draws']} -{s['losses']}  perf {perf}",
                flush=True,
            )
    result = {"player": player_spec, "name": player.name, "opponents": opponents, "seconds": time.time() - t0}
    if per_game_opp:
        r = mle_rating(per_game_opp, per_game_score, bootstrap=bootstrap)
        result["rating"] = {
            "elo": r.elo,
            "se": r.se,
            "lo": r.lo,
            "hi": r.hi,
            "boot_lo": r.boot_lo,
            "boot_hi": r.boot_hi,
            "games": r.games,
        }
        if verbose:
            print(f"[eval] {player.name}: rating {r}")
    stats = getattr(player, "stats", None)
    if stats is not None and stats.seconds:
        result["search"] = {
            "evals": stats.evals,
            "evals_per_s": stats.evals_per_s,
            "mean_batch": stats.evals / max(stats.batches, 1),
        }
    player.close()
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--player", required=True)
    p.add_argument("--opponents", nargs="+", required=True)
    p.add_argument("--games", type=int, default=40, help="games per opponent (even: each opening with both colours)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--pgn", default=None)
    a = p.parse_args(argv)
    result = evaluate(a.player, a.opponents, a.games, a.concurrency, a.device, a.pgn, a.bootstrap, a.max_plies)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2))
        print(f"[eval] wrote {a.out}")


if __name__ == "__main__":
    main()
