"""Single entry point: ``azchess <command> [args...]``."""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "download": ("azchess.data.download", "download a prefix of a monthly Lichess dump"),
    "filter": ("azchess.data.filter", "filter a dump by rating / time control / termination"),
    "build": ("azchess.data.build", "convert filtered PGN into training shards"),
    "train": ("azchess.train.supervised", "supervised pretraining"),
    "selfplay": ("azchess.selfplay.loop", "self-play reinforcement loop with gating"),
    "eval": ("azchess.eval.run", "play matches and estimate Elo"),
    "ablate": ("azchess.experiments.ablations", "run ablation experiments"),
    "tactics": ("azchess.experiments.tactics", "Win At Chess tactical test suite"),
    "accuracy": ("azchess.experiments.accuracy", "held-out move-prediction accuracy"),
    "plots": ("azchess.experiments.plots", "render figures from results/"),
    "report": ("azchess.experiments.report", "render markdown result tables"),
    "uci": ("azchess.uci.engine", "run the UCI engine"),
    "export": ("azchess.tools.export", "export weights for deployment"),
    "bench": ("azchess.tools.bench", "benchmark move generation, network and search"),
}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print("usage: azchess <command> [args]\n\ncommands:")
        for name, (_mod, doc) in COMMANDS.items():
            print(f"  {name:10s} {doc}")
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 1)
    module = importlib.import_module(COMMANDS[argv[0]][0])
    module.main(argv[1:])


if __name__ == "__main__":
    main()
