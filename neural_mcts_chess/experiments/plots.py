"""Render README figures from runs/ and results/.

    python -m neural_mcts_chess.experiments.plots --out docs/figures

Every figure degrades gracefully: missing inputs are skipped with a note.
Colours: the first three slots of a CVD-validated categorical palette (blue,
orange, aqua); series are also distinguished by marker shape and direct labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "font.size": 10,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "lines.markersize": 6,
        }
    )


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _metrics(run: Path) -> list[dict]:
    f = run / "metrics.jsonl"
    return [json.loads(line) for line in open(f)] if f.exists() else []


def _label_end(ax, x, y, text, color) -> None:
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", color=INK_2, fontsize=9)


def _val_points(name: str, run: Path, results: Path) -> list[dict]:
    """Prefer per-snapshot revalidation (one consistent procedure); fall back to
    the validation entries logged during training (last entry per step)."""
    snap = _load(results / f"snapshot_val_{run.name}.json")
    if snap:
        return [{"step": int(k), **v} for k, v in sorted(snap.items(), key=lambda kv: int(kv[0]))]
    by_step = {m["step"]: m for m in _metrics(run) if m["split"] == "val"}
    return [by_step[k] for k in sorted(by_step)]


def training_curves(runs: dict[str, Path], out: Path, results: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, (name, run) in enumerate(runs.items()):
        val = _val_points(name, run, results)
        if not val:
            continue
        steps = [m["step"] * 1024 / 1e6 for m in val]
        for ax, key in zip(axes, ["policy_top1", "value_loss"]):
            ys = [m[key] for m in val]
            ax.plot(steps, ys, color=SERIES[k], marker=MARKERS[k], markersize=4, label=name)
            _label_end(ax, steps[-1], ys[-1], name, SERIES[k])
    axes[0].set_title("Held-out move-prediction accuracy (top-1)")
    axes[0].set_ylabel("top-1 accuracy")
    axes[1].set_title("Held-out value loss (WDL cross-entropy)")
    axes[1].set_ylabel("cross-entropy")
    for ax in axes:
        ax.set_xlabel("training positions (millions)")
        ax.legend(loc="lower right" if ax is axes[0] else "upper right")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=160)
    plt.close(fig)


def _rating(entry: dict) -> tuple[float, float, float]:
    r = entry["rating"]
    lo = r.get("boot_lo") if r.get("boot_lo") is not None else r["lo"]
    hi = r.get("boot_hi") if r.get("boot_hi") is not None else r["hi"]
    return r["elo"], r["elo"] - lo, hi - r["elo"]


def elo_vs_steps(path: Path, out: Path) -> None:
    data = _load(path)
    if not data:
        print(f"[plots] skip elo_vs_steps ({path} missing)")
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    by_sims: dict[int, list] = {}
    for e in data.values():
        by_sims.setdefault(e["sims"], []).append(e)
    for k, (sims, entries) in enumerate(sorted(by_sims.items())):
        entries.sort(key=lambda e: e["step"])
        xs = [e["step"] * 1024 / 1e6 for e in entries]
        ys, lo, hi = zip(*[_rating(e) for e in entries])
        ax.fill_between(
            xs,
            [y - a for y, a in zip(ys, lo)],
            [y + b for y, b in zip(ys, hi)],
            color=SERIES[k],
            alpha=0.15,
            linewidth=0,
        )
        label = "raw policy" if sims <= 1 else f"MCTS {sims} visits"
        ax.plot(xs, ys, color=SERIES[k], marker=MARKERS[k], label=label)
        _label_end(ax, xs[-1], ys[-1], f"{ys[-1]:.0f}", SERIES[k])
    ax.set_title("Raw-policy Elo vs training (95% CI)" if len(by_sims) == 1 else "Elo vs training (95% CI)")
    ax.set_xlabel("training positions (millions)")
    ax.set_ylabel("Elo (Stockfish UCI_Elo scale)")
    if len(by_sims) > 1:
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "elo_vs_steps.png", dpi=160)
    plt.close(fig)


def elo_vs_sims(path: Path, out: Path) -> None:
    data = _load(path)
    if not data:
        print(f"[plots] skip elo_vs_sims ({path} missing)")
        return
    entries = sorted(data.values(), key=lambda e: e["sims"])
    xs = [max(e["sims"], 1) for e in entries]
    ys, lo, hi = zip(*[_rating(e) for e in entries])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.errorbar(xs, ys, yerr=[lo, hi], color=SERIES[0], marker="o", capsize=3, elinewidth=1.2)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.0f}", (x, y), xytext=(8, -4), textcoords="offset points", ha="left", color=INK_2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["raw\npolicy" if e["sims"] <= 1 else str(e["sims"]) for e in entries])
    ax.set_title("Search scaling: Elo vs MCTS visits per move (95% CI)")
    ax.set_xlabel("visits per move (log scale)")
    ax.set_ylabel("Elo")
    fig.tight_layout()
    fig.savefig(out / "elo_vs_sims.png", dpi=160)
    plt.close(fig)


def elo_vs_size(path: Path, out: Path) -> None:
    data = _load(path)
    if not data:
        print(f"[plots] skip elo_vs_size ({path} missing)")
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    by_sims: dict[int, list] = {}
    for e in data.values():
        by_sims.setdefault(e["sims"], []).append(e)
    for k, (sims, entries) in enumerate(sorted(by_sims.items())):
        entries.sort(key=lambda e: e["params"])
        xs = [e["params"] / 1e6 for e in entries]
        ys, lo, hi = zip(*[_rating(e) for e in entries])
        label = "raw policy" if sims <= 1 else f"MCTS {sims} visits"
        ax.errorbar(xs, ys, yerr=[lo, hi], color=SERIES[k], marker=MARKERS[k], capsize=3, elinewidth=1.2, label=label)
        for x, y, e in zip(xs, ys, entries):
            ax.annotate(e["model"], (x, y), xytext=(6, -10), textcoords="offset points", color=INK_2, fontsize=8)
    ax.set_xscale("log")
    sizes = sorted({e["params"] / 1e6 for e in data.values()})
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{x:.2g}M" for x in sizes])
    ax.minorticks_off()
    ax.set_title("Network size vs strength (95% CI)")
    ax.set_xlabel("parameters (millions, log scale)")
    ax.set_ylabel("Elo")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "elo_vs_size.png", dpi=160)
    plt.close(fig)


def tactics(path: Path, out: Path) -> None:
    data = _load(path)
    if not data:
        print(f"[plots] skip tactics ({path} missing)")
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for k, (name, res) in enumerate(list(data.items())[:3]):
        pts = sorted((int(s), v["solved"] / v["total"]) for s, v in res.items())
        xs = [max(s, 1) for s, _ in pts]
        ys = [y * 100 for _, y in pts]
        ax.plot(xs, ys, color=SERIES[k], marker=MARKERS[k], label=name)
        _label_end(ax, xs[-1], ys[-1], f"{ys[-1]:.0f}%", SERIES[k])
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["raw\npolicy" if s <= 1 else str(s) for s, _ in pts])
    ax.set_title("Win At Chess tactics solved (300 positions)")
    ax.set_xlabel("visits per position (log scale)")
    ax.set_ylabel("% solved")
    if len(data) > 1:
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "tactics.png", dpi=160)
    plt.close(fig)


def selfplay_generations(loop_dir: Path, out: Path, results: Path) -> None:
    state = _load(loop_dir / "state.json")
    if not state or not state.get("history"):
        print(f"[plots] skip selfplay ({loop_dir} has no history)")
        return
    hist = state["history"]
    gens = [h["generation"] for h in hist]
    diffs = [h["gating"]["elo_diff"] for h in hist]
    lo = [h["gating"]["elo_diff"] - h["gating"]["elo_lo"] for h in hist]
    hi = [h["gating"]["elo_hi"] - h["gating"]["elo_diff"] for h in hist]
    accepted = [h["gating"]["accepted"] for h in hist]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.errorbar(gens, diffs, yerr=[lo, hi], color=SERIES[0], fmt="none", capsize=3, elinewidth=1.2)
    ax.scatter(
        [g for g, a in zip(gens, accepted) if a],
        [d for d, a in zip(diffs, accepted) if a],
        color=SERIES[0],
        marker="o",
        s=40,
        label="accepted",
        zorder=3,
    )
    ax.scatter(
        [g for g, a in zip(gens, accepted) if not a],
        [d for d, a in zip(diffs, accepted) if not a],
        facecolor=SURFACE,
        edgecolor=SERIES[0],
        marker="o",
        s=40,
        label="rejected",
        zorder=3,
    )
    ax.set_title("Gating: candidate vs current best")
    ax.set_xlabel("generation")
    ax.set_ylabel("Elo difference (95% CI)")
    ax.set_xticks(gens)
    ax.legend(loc="best")
    ax = axes[1]
    elo = state["elo"]
    xs = sorted(int(g) for g in elo) + [max(gens)]
    ys = [elo[str(g)] for g in sorted(int(g) for g in elo)]
    ys.append(ys[-1])
    ax.step(xs, ys, where="post", color=SERIES[1], label="gating estimate")
    ax.scatter(xs[:-1], ys[:-1], color=SERIES[1], marker="s", zorder=3)
    h2h = (_load(results / "ablation_selfplay.json") or {}).get("h2h/sims=256")
    if h2h:
        x = max(gens) + 0.3
        ax.errorbar(
            [x],
            [h2h["elo_diff"]],
            yerr=[[h2h["elo_diff"] - h2h["elo_lo"]], [h2h["elo_hi"] - h2h["elo_diff"]]],
            color=SERIES[0],
            marker="o",
            capsize=3,
            elinewidth=1.2,
            label=f"re-test, {h2h['games']} games",
        )
        ax.annotate(
            f"{h2h['elo_diff']:+.0f}",
            (x, h2h["elo_diff"]),
            xytext=(8, -4),
            textcoords="offset points",
            color=INK_2,
            fontsize=9,
        )
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_title("Best network vs pretrained (Elo)")
    ax.set_xlabel("generation")
    ax.set_ylabel("Elo relative to generation 0")
    ax.set_xticks(range(0, max(gens) + 1))
    ax.legend(loc="center left", bbox_to_anchor=(0.4, 0.5))
    fig.tight_layout()
    fig.savefig(out / "selfplay_generations.png", dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("docs/figures"))
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--runs", nargs="*", default=["tiny=runs/sl_tiny", "small=runs/sl_small", "medium=runs/sl_medium"])
    p.add_argument("--loop", type=Path, default=Path("runs/selfplay"))
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    _style()
    runs = {k: Path(v) for k, v in (r.split("=", 1) for r in a.runs)}
    training_curves(runs, a.out, a.results)
    elo_vs_steps(a.results / "elo_vs_steps_sl_medium.json", a.out)
    elo_vs_sims(a.results / "ablation_sims.json", a.out)
    elo_vs_size(a.results / "ablation_size.json", a.out)
    tactics(a.results / "tactics.json", a.out)
    selfplay_generations(a.loop, a.out, a.results)
    print(f"[plots] wrote figures to {a.out}")


if __name__ == "__main__":
    main()
