"""Supervised pretraining on human games.

python -m neural_mcts_chess.train.supervised --config configs/train_medium.yaml [overrides key=value ...]
"""

from __future__ import annotations

import argparse

import yaml

from neural_mcts_chess.data.loader import make_loader
from neural_mcts_chess.train.trainer import TrainConfig, Trainer


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, value = pair.split("=", 1)
        out[key] = yaml.safe_load(value)
    return out


def load_config(path: str | None, overrides: list[str]) -> TrainConfig:
    d = yaml.safe_load(open(path)) if path else {}
    for k, v in parse_overrides(overrides).items():
        if k.startswith("model."):
            d.setdefault("model", {})[k[6:]] = v
        else:
            d[k] = v
    return TrainConfig.from_dict(d)


def run(cfg: TrainConfig) -> Trainer:
    trainer = Trainer(cfg)
    print(
        f"[train] device={trainer.device} params={trainer.model.num_parameters() / 1e6:.2f}M model={trainer.model_cfg}"
    )
    train = make_loader(cfg.train_data, cfg.batch_size, workers=cfg.workers, seed=cfg.seed + trainer.step)

    def val_factory():
        return make_loader(
            cfg.val_data, 2048, workers=0, shuffle=False, infinite=False, max_positions=cfg.val_positions
        )

    try:
        trainer.fit(train, val_factory)
    finally:
        trainer.close()
    return trainer


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("overrides", nargs="*", help="key=value overrides, e.g. steps=1000 model.blocks=4")
    a = p.parse_args(argv)
    run(load_config(a.config, a.overrides))


if __name__ == "__main__":
    main()
