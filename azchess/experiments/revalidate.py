"""Held-out validation metrics for every weight snapshot of a run, computed
with one consistent procedure (used for the training-curve figure).

    python -m azchess.experiments.revalidate runs/sl_medium
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from azchess.data.loader import make_loader
from azchess.model.inference import load_checkpoint, pick_device
from azchess.train.trainer import TrainConfig, Trainer


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+")
    p.add_argument("--data", default="data/shards/val")
    p.add_argument("--positions", type=int, default=200_000)
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    a = p.parse_args(argv)
    dev = pick_device("auto")
    for run in map(Path, a.runs):
        out = a.out_dir / f"snapshot_val_{run.name}.json"
        res = json.loads(out.read_text()) if out.exists() else {}
        for snap in sorted((run / "snapshots").glob("step_*.pt")):
            step = int(re.search(r"step_(\d+)", snap.name).group(1))
            if str(step) in res:
                continue
            model, ckpt = load_checkpoint(snap, dev)
            tr = Trainer.__new__(Trainer)  # only need validate(); avoid creating a run dir
            tr.device, tr.amp_dtype, tr.cfg = dev, None, TrainConfig()
            loader = make_loader(a.data, 2048, workers=0, shuffle=False, infinite=False, max_positions=a.positions)
            m = tr.validate(loader, model)
            m["bn_recalibrated"] = bool(ckpt.get("bn_recalibrated"))
            res[str(step)] = m
            print(
                f"[revalidate] {run.name} step {step}: top1 {m['policy_top1']:.4f} value {m['value_loss']:.4f}",
                flush=True,
            )
        a.out_dir.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
