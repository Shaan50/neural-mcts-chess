"""Single entry point: ``neural-mcts-chess <command> [args...]``."""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "download": ("neural_mcts_chess.data.download", "download a prefix of a monthly Lichess dump"),
    "filter": ("neural_mcts_chess.data.filter", "filter a dump by rating / time control / termination"),
    "build": ("neural_mcts_chess.data.build", "convert filtered PGN into training shards"),
    "train": ("neural_mcts_chess.train.supervised", "supervised pretraining"),
    "selfplay": ("neural_mcts_chess.selfplay.loop", "self-play reinforcement loop with gating"),
    "eval": ("neural_mcts_chess.eval.run", "play matches and estimate Elo"),
    "ablate": ("neural_mcts_chess.experiments.ablations", "run ablation experiments"),
    "tactics": ("neural_mcts_chess.experiments.tactics", "Win At Chess tactical test suite"),
    "accuracy": ("neural_mcts_chess.experiments.accuracy", "held-out move-prediction accuracy"),
    "plots": ("neural_mcts_chess.experiments.plots", "render figures from results/"),
    "report": ("neural_mcts_chess.experiments.report", "render markdown result tables"),
    "uci": ("neural_mcts_chess.uci.engine", "run the UCI engine"),
    "export": ("neural_mcts_chess.tools.export", "export weights for deployment"),
    "bench": ("neural_mcts_chess.tools.bench", "benchmark move generation, network and search"),
}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print("usage: neural-mcts-chess <command> [args]\n\ncommands:")
        for name, (_mod, doc) in COMMANDS.items():
            print(f"  {name:10s} {doc}")
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 1)
    module = importlib.import_module(COMMANDS[argv[0]][0])
    module.main(argv[1:])


if __name__ == "__main__":
    main()
