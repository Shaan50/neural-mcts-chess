"""Ablation experiments. Each writes/extends a JSON file in results/ keyed by a
label, so interrupted experiments resume where they stopped.

    # raw policy vs MCTS at increasing visit counts
    python -m azchess.experiments.ablations sims --model runs/sl_medium/best.pt --sims 0 16 64 256 800

    # network size vs strength (same data, same number of training steps)
    python -m azchess.experiments.ablations size --models tiny=runs/sl_tiny/best.pt small=runs/sl_small/best.pt \
        medium=runs/sl_medium/best.pt --sims 0 256

    # Elo vs training step from a run's weight snapshots
    python -m azchess.experiments.ablations steps --run runs/sl_medium --sims 0

    # pretraining-only vs self-play-refined
    python -m azchess.experiments.ablations selfplay --sl runs/sl_medium/best.pt --loop runs/selfplay --sims 256

Strength is measured against Stockfish 19 with UCI_LimitStrength (UCI_Elo
scale). An adaptive ladder first probes the player's strength with a few games,
then plays full matches against levels bracketing the estimate.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from azchess.eval.elo import mle_rating, performance
from azchess.eval.match import play_match, summarize
from azchess.eval.players import build_player
from azchess.eval.run import evaluate
from azchess.model.inference import Evaluator

SF_MIN, SF_MAX = 1320, 3190


class Results:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def has(self, key: str) -> bool:
        return key in self.data

    def put(self, key: str, value: dict) -> None:
        self.data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)


def sf_spec(elo: int, movetime: float) -> str:
    return f"sf:elo={int(elo)},movetime={movetime}"


def adaptive_ladder(
    player_spec: str,
    probe_games: int = 12,
    games: int = 32,
    levels: int = 3,
    spacing: int = 200,
    start: int = 1700,
    movetime: float = 0.1,
    concurrency: int = 16,
    evaluators: dict | None = None,
) -> dict:
    """Probe to find the player's strength, then evaluate against ``levels``
    Stockfish settings spaced ``spacing`` apart around the estimate."""
    evaluators = evaluators if evaluators is not None else {}
    probes = []
    elo = start
    for _ in range(4):
        res = evaluate(
            player_spec,
            [sf_spec(elo, movetime)],
            probe_games,
            concurrency,
            evaluators=evaluators,
            bootstrap=0,
            verbose=False,
        )
        o = res["opponents"][0]
        probes.append(o)
        print(f"[ladder] probe vs SF {elo}: +{o['wins']} ={o['draws']} -{o['losses']}", flush=True)
        if 0 < o["score"] < 1 or elo in (SF_MIN, SF_MAX):
            break
        elo = min(SF_MAX, elo + 400) if o["score"] == 1 else max(SF_MIN, elo - 400)
    opp = [p["known_elo"] for p in probes for _ in range(p["games"])]
    sc = [g_score for p in probes for g_score in _scores(p)]
    est = mle_rating(opp, sc).elo
    centre = int(round(min(max(est, SF_MIN + spacing * (levels // 2)), SF_MAX - spacing * (levels // 2)) / 50.0) * 50)
    ladder = [centre + spacing * (k - levels // 2) for k in range(levels)]
    print(f"[ladder] estimate {est:.0f} -> ladder {ladder}", flush=True)
    full = evaluate(
        player_spec, [sf_spec(e, movetime) for e in ladder], games, concurrency, evaluators=evaluators, bootstrap=500
    )
    # Combine probe and ladder games in one MLE.
    opp += [o["known_elo"] for o in full["opponents"] for _ in range(o["games"])]
    sc += [s for o in full["opponents"] for s in _scores(o)]
    r = mle_rating(opp, sc, bootstrap=500)
    full["probes"] = [_strip(p) for p in probes]
    full["opponents"] = [_strip(o) for o in full["opponents"]]
    full["rating"] = {
        "elo": r.elo,
        "se": r.se,
        "lo": r.lo,
        "hi": r.hi,
        "boot_lo": r.boot_lo,
        "boot_hi": r.boot_hi,
        "games": r.games,
    }
    print(f"[ladder] {player_spec}: {r}", flush=True)
    return full


def _scores(opp_entry: dict) -> list[float]:
    name = opp_entry["name"]
    out = []
    for g in opp_entry["games_detail"]:
        if g["result"] == "1/2-1/2":
            out.append(0.5)
        else:
            our_white = g["black"] == name
            out.append(1.0 if (g["result"] == "1-0") == our_white else 0.0)
    return out


def _strip(entry: dict) -> dict:
    e = dict(entry)
    e["games_detail"] = [
        {k: g[k] for k in ("white", "black", "result", "termination", "opening")} | {"plies": len(g["moves"])}
        for g in entry["games_detail"]
    ]
    return e


def head_to_head(spec_a: str, spec_b: str, games: int, concurrency: int = 32, evaluators: dict | None = None) -> dict:
    evaluators = evaluators if evaluators is not None else {}
    a = build_player(spec_a, evaluators=evaluators)
    b = build_player(spec_b, evaluators=evaluators)
    if a.name == b.name:
        b.name += "_b"
    recs = play_match(a, b, games, concurrency=concurrency)
    s = summarize(recs, a.name)
    perf = performance(s["wins"], s["draws"], s["losses"])
    return {**s, "a": spec_a, "b": spec_b, "elo_diff": perf.elo, "elo_lo": perf.lo, "elo_hi": perf.hi}


# --------------------------------------------------------------------- cli
def cmd_sims(a) -> None:
    res = Results(a.out or "results/ablation_sims.json")
    evaluators: dict = {}
    for sims in a.sims:
        key = f"sims={sims}"
        if res.has(key):
            continue
        r = adaptive_ladder(
            f"nn:path={a.model},sims={sims}", a.probe_games, a.games, movetime=a.movetime, evaluators=evaluators
        )
        res.put(key, {"sims": sims, "model": a.model, **r})


def cmd_size(a) -> None:
    res = Results(a.out or "results/ablation_size.json")
    evaluators: dict = {}
    for item in a.models:
        name, path = item.split("=", 1)
        for sims in a.sims:
            key = f"{name}/sims={sims}"
            if res.has(key):
                continue
            ev = evaluators.get(path) or Evaluator.from_checkpoint(path)
            evaluators[path] = ev
            r = adaptive_ladder(
                f"nn:path={path},sims={sims}", a.probe_games, a.games, movetime=a.movetime, evaluators=evaluators
            )
            res.put(
                key,
                {
                    "model": name,
                    "path": path,
                    "sims": sims,
                    "params": ev.model.num_parameters(),
                    "config": ev.model.cfg.to_dict(),
                    **r,
                },
            )


def cmd_steps(a) -> None:
    run = Path(a.run)
    res = Results(a.out or f"results/elo_vs_steps_{run.name}.json")
    snaps = sorted((run / "snapshots").glob("step_*.pt"))
    if a.every > 1:
        snaps = snaps[a.every - 1 :: a.every] + (
            [snaps[-1]] if snaps and snaps[-1] not in snaps[a.every - 1 :: a.every] else []
        )
    evaluators: dict = {}
    for snap in snaps:
        step = int(re.search(r"step_(\d+)", snap.name).group(1))
        for sims in a.sims:
            key = f"step={step}/sims={sims}"
            if res.has(key):
                continue
            r = adaptive_ladder(
                f"nn:path={snap},sims={sims}", a.probe_games, a.games, movetime=a.movetime, evaluators=evaluators
            )
            res.put(key, {"step": step, "sims": sims, "path": str(snap), **r})
            evaluators.clear()


def cmd_selfplay(a) -> None:
    res = Results(a.out or "results/ablation_selfplay.json")
    loop = Path(a.loop)
    state = json.loads((loop / "state.json").read_text())
    rl = state["best"]
    evaluators: dict = {}
    for name, path in [("pretrained", a.sl), ("selfplay", rl)]:
        key = f"{name}/sims={a.sims}"
        if not res.has(key):
            r = adaptive_ladder(
                f"nn:path={path},sims={a.sims}", a.probe_games, a.games, movetime=a.movetime, evaluators=evaluators
            )
            res.put(key, {"model": name, "path": path, "sims": a.sims, **r})
    key = f"h2h/sims={a.sims}"
    if not res.has(key) and a.h2h_games:
        res.put(
            key,
            head_to_head(
                f"nn:path={rl},sims={a.sims},name=selfplay",
                f"nn:path={a.sl},sims={a.sims},name=pretrained",
                a.h2h_games,
                evaluators=evaluators,
            ),
        )
    res.put("loop_history", {"history": state["history"], "relative_elo": state["elo"], "best_gen": state["best_gen"]})


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--games", type=int, default=32, help="games per ladder level")
        sp.add_argument("--probe-games", type=int, default=12)
        sp.add_argument("--movetime", type=float, default=0.1, help="Stockfish seconds per move")
        sp.add_argument("--out", default=None)

    s = sub.add_parser("sims")
    s.add_argument("--model", required=True)
    s.add_argument("--sims", type=int, nargs="+", default=[0, 16, 64, 256, 800])
    common(s)
    s = sub.add_parser("size")
    s.add_argument("--models", nargs="+", required=True, help="name=path")
    s.add_argument("--sims", type=int, nargs="+", default=[0, 256])
    common(s)
    s = sub.add_parser("steps")
    s.add_argument("--run", required=True)
    s.add_argument("--sims", type=int, nargs="+", default=[0])
    s.add_argument("--every", type=int, default=1, help="use every k-th snapshot")
    common(s)
    s = sub.add_parser("selfplay")
    s.add_argument("--sl", required=True)
    s.add_argument("--loop", required=True)
    s.add_argument("--sims", type=int, default=256)
    s.add_argument("--h2h-games", type=int, default=100)
    common(s)
    a = p.parse_args(argv)
    {"sims": cmd_sims, "size": cmd_size, "steps": cmd_steps, "selfplay": cmd_selfplay}[a.cmd](a)


if __name__ == "__main__":
    main()
