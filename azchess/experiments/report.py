"""Render markdown tables from results/*.json (pasted into the README).

python -m azchess.experiments.report > results/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _ci(e: dict) -> str:
    r = e["rating"]
    lo = r.get("boot_lo") if r.get("boot_lo") is not None else r["lo"]
    hi = r.get("boot_hi") if r.get("boot_hi") is not None else r["hi"]
    return f"**{r['elo']:.0f}** [{lo:.0f}, {hi:.0f}]"


def _record(e: dict) -> str:
    w = sum(o["wins"] for o in e["opponents"]) + sum(o["wins"] for o in e.get("probes", []))
    d = sum(o["draws"] for o in e["opponents"]) + sum(o["draws"] for o in e.get("probes", []))
    l = sum(o["losses"] for o in e["opponents"]) + sum(o["losses"] for o in e.get("probes", []))
    return f"+{w} ={d} −{l}"


def _ladder(e: dict) -> str:
    return ", ".join(f"{int(o['known_elo'])}: {o['score']:.2f}" for o in e["opponents"])


def accuracy_table(res: Path) -> str:
    d = _load(res / "accuracy.json")
    if not d:
        return ""
    rows = ["| Model | Params | Top-1 | Top-5 | Policy CE | Value MSE |", "|---|---:|---:|---:|---:|---:|"]
    for _path, r in sorted(d.items(), key=lambda kv: kv[1]["params"]):
        o = r["overall"]
        c = r["config"]
        rows.append(
            f"| {c['blocks']}×{c['filters']} | {r['params'] / 1e6:.2f}M | {o['top1']:.1%} | {o['top5']:.1%} | {o['policy_ce']:.3f} | {o['value_mse']:.3f} |"
        )
    big = max(d.values(), key=lambda r: r["params"])
    rows += ["", f"Breakdown for the {big['config']['blocks']}×{big['config']['filters']} network:", ""]
    rows += ["| Slice | Positions | Top-1 | Top-5 |", "|---|---:|---:|---:|"]
    for name, key in [("ply", "by_ply"), ("rating", "by_elo")]:
        for b, o in big[key].items():
            rows.append(f"| {name} {b} | {o['positions']:,} | {o['top1']:.1%} | {o['top5']:.1%} |")
    return "\n".join(rows)


def sims_table(res: Path) -> str:
    d = _load(res / "ablation_sims.json")
    if not d:
        return ""
    rows = ["| Search | Elo [95% CI] | Games (W/D/L) | Score by Stockfish level |", "|---|---|---|---|"]
    for e in sorted(d.values(), key=lambda e: e["sims"]):
        name = "raw policy (no search)" if e["sims"] <= 1 else f"MCTS, {e['sims']} visits/move"
        rows.append(f"| {name} | {_ci(e)} | {_record(e)} | {_ladder(e)} |")
    return "\n".join(rows)


def size_table(res: Path) -> str:
    d = _load(res / "ablation_size.json")
    if not d:
        return ""
    rows = ["| Network | Params | Search | Elo [95% CI] | Games (W/D/L) |", "|---|---:|---|---|---|"]
    for e in sorted(d.values(), key=lambda e: (e["sims"], e["params"])):
        c = e["config"]
        s = "raw policy" if e["sims"] <= 1 else f"MCTS {e['sims']}"
        rows.append(
            f"| {e['model']} ({c['blocks']}×{c['filters']}) | {e['params'] / 1e6:.2f}M | {s} | {_ci(e)} | {_record(e)} |"
        )
    return "\n".join(rows)


def steps_table(res: Path, run: str = "sl_medium") -> str:
    d = _load(res / f"elo_vs_steps_{run}.json")
    if not d:
        return ""
    rows = ["| Training step | Positions | Search | Elo [95% CI] |", "|---:|---:|---|---|"]
    for e in sorted(d.values(), key=lambda e: (e["sims"], e["step"])):
        s = "raw policy" if e["sims"] <= 1 else f"MCTS {e['sims']}"
        rows.append(f"| {e['step']:,} | {e['step'] * 1024 / 1e6:.1f}M | {s} | {_ci(e)} |")
    return "\n".join(rows)


def selfplay_table(res: Path) -> str:
    d = _load(res / "ablation_selfplay.json")
    if not d:
        return ""
    rows = ["| Network | Elo vs Stockfish [95% CI] | Games (W/D/L) |", "|---|---|---|"]
    for k, e in d.items():
        if k.startswith(("pretrained", "selfplay")):
            rows.append(f"| {e['model']} (MCTS {e['sims']}) | {_ci(e)} | {_record(e)} |")
    for k, e in d.items():
        if k.startswith("h2h"):
            rows += [
                "",
                f"Head-to-head, self-play-refined vs pretrained ({e['games']} games, paired openings): "
                f"+{e['wins']} ={e['draws']} −{e['losses']}, score {e['score']:.3f}, "
                f"Elo difference **{e['elo_diff']:+.0f}** [{e['elo_lo']:+.0f}, {e['elo_hi']:+.0f}]",
            ]
    hist = d.get("loop_history", {}).get("history", [])
    if hist:
        rows += [
            "",
            "| Gen | Self-play games | Positions recorded | Gating score (vs current best) | Elo diff [95% CI] | SPRT | Accepted |",
            "|---:|---:|---:|---:|---|---|---|",
        ]
        for h in hist:
            g, sp = h["gating"], h["selfplay"]
            rows.append(
                f"| {h['generation']} | {sp['games']:,} | {sp['positions']:,} | {g['score']:.3f} | "
                f"{g['elo_diff']:+.0f} [{g['elo_lo']:+.0f}, {g['elo_hi']:+.0f}] | {g['sprt']} | {'yes' if g['accepted'] else 'no'} |"
            )
    return "\n".join(rows)


def tactics_table(res: Path) -> str:
    d = _load(res / "tactics.json")
    if not d:
        return ""
    budgets = sorted({int(s) for r in d.values() for s in r}, key=int)
    head = "| Network | " + " | ".join("raw policy" if b <= 1 else f"{b} visits" for b in budgets) + " |"
    rows = [head, "|---|" + "---:|" * len(budgets)]
    for name, r in d.items():
        cells = [f"{r[str(b)]['solved']}/{r[str(b)]['total']}" if str(b) in r else "–" for b in budgets]
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results"))
    a = p.parse_args(argv)
    for title, fn in [
        ("Held-out move prediction", accuracy_table),
        ("Raw policy vs MCTS", sims_table),
        ("Tactics (Win At Chess, 300 positions)", tactics_table),
        ("Network size vs strength", size_table),
        ("Elo vs training step", steps_table),
        ("Pretraining-only vs self-play-refined", selfplay_table),
    ]:
        body = fn(a.results)
        if body:
            print(f"### {title}\n\n{body}\n")


if __name__ == "__main__":
    main()
