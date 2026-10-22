"""Re-estimate BatchNorm statistics of a checkpoint's EMA weights in place.

Averaged weights need statistics computed *for those weights* (as SWA's
``update_bn`` does). Snapshots written before the EMA began averaging BN
buffers paired averaged weights with the live model's statistics; running this
over every snapshot gives them all the same, correct treatment.

    python -m azchess.tools.recalibrate runs/sl_medium/snapshots/*.pt runs/sl_medium/best.pt
"""

from __future__ import annotations

import argparse
import itertools
import os

import torch

from azchess.data.loader import make_loader
from azchess.encoding.board import planes_from_compact_torch
from azchess.model.inference import pick_device
from azchess.model.network import AZNet, ModelConfig


@torch.no_grad()
def recalibrate_bn(model: AZNet, batches, device) -> None:
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    momenta = [bn.momentum for bn in bns]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average over all batches
    model.train()
    for b in batches:
        model(planes_from_compact_torch(b["bb"].to(device), b["meta"].to(device)))
    for bn, mom in zip(bns, momenta):
        bn.momentum = mom
    model.eval()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--data", default="data/shards/train")
    p.add_argument("--batches", type=int, default=64)
    p.add_argument("--force", action="store_true", help="recalibrate even if already done")
    a = p.parse_args(argv)
    dev = pick_device("auto")
    for path in a.checkpoints:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("bn_recalibrated") and not a.force:
            print(f"[recalibrate] {path}: already done")
            continue
        model = AZNet(ModelConfig.from_dict(ckpt["model_config"]))
        model.load_state_dict(ckpt["ema"] if ckpt.get("ema") is not None else ckpt["model"])
        model.to(dev)
        loader = make_loader(a.data, 1024, workers=0, seed=1234, shards_in_memory=1)
        recalibrate_bn(model, itertools.islice(iter(loader), a.batches), dev)
        key = "ema" if ckpt.get("ema") is not None else "model"
        ckpt[key] = {k: v.cpu() for k, v in model.state_dict().items()}
        ckpt["bn_recalibrated"] = True
        tmp = f"{path}.tmp"
        torch.save(ckpt, tmp)
        os.replace(tmp, path)
        print(f"[recalibrate] {path}: BN statistics re-estimated on {a.batches} batches")


if __name__ == "__main__":
    main()
