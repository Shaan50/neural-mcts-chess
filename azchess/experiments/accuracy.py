"""Held-out move-prediction accuracy of one or more networks.

Reports policy top-1 / top-5 accuracy, policy cross-entropy and value metrics
on the test split (games never seen in training), overall and broken down by
game phase (ply) and by the rating of the player to move.

    python -m azchess.experiments.accuracy runs/sl_tiny/best.pt runs/sl_small/best.pt runs/sl_medium/best.pt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from azchess.data.loader import make_batch
from azchess.data.shards import concat_positions, legal_offsets, list_shards, load_shard
from azchess.encoding.board import planes_from_compact_torch
from azchess.model.inference import autocast_dtype, load_checkpoint, pick_device
from azchess.train.losses import legal_mask

PLY_BUCKETS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 1000)]
ELO_BUCKETS = [(2000, 2200), (2200, 2400), (2400, 2600), (2600, 4000)]


@torch.no_grad()
def evaluate_model(path: str, data: dict, batch_size: int = 2048, device: str = "auto") -> dict:
    dev = pick_device(device)
    model, ckpt = load_checkpoint(path, dev)
    amp = autocast_dtype(dev)
    offsets = legal_offsets(data["n_legal"])
    n = len(data["bb"])
    per_pos = {
        "top1": np.zeros(n, bool),
        "top5": np.zeros(n, bool),
        "pl": np.zeros(n, np.float32),
        "mse": np.zeros(n, np.float32),
    }
    for i in range(0, n, batch_size):
        rows = np.arange(i, min(n, i + batch_size))
        b = {k: v.to(dev) for k, v in make_batch(data, offsets, rows).items()}
        x = planes_from_compact_torch(b["bb"], b["meta"])
        with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
            p, v = model(x)
        logits = p.float()
        mask = legal_mask(b["legal_flat"], len(rows), dev)
        logits = logits.masked_fill(~mask, -1e9)
        target = b["policy_idx"][:, 0]
        top5 = logits.topk(5, -1).indices
        per_pos["top1"][rows] = (top5[:, 0] == target).cpu().numpy()
        per_pos["top5"][rows] = (top5 == target[:, None]).any(-1).cpu().numpy()
        per_pos["pl"][rows] = torch.nn.functional.cross_entropy(logits, target, reduction="none").cpu().numpy()
        if model.cfg.value_head == "wdl":
            wdl = torch.softmax(v.float(), -1)
            e = wdl[:, 0] - wdl[:, 2]
        else:
            e = v.float().view(-1)
        per_pos["mse"][rows] = ((e - b["result"].float()) ** 2).cpu().numpy()

    def agg(mask) -> dict:
        return {
            "positions": int(mask.sum()),
            "top1": float(per_pos["top1"][mask].mean()),
            "top5": float(per_pos["top5"][mask].mean()),
            "policy_ce": float(per_pos["pl"][mask].mean()),
            "value_mse": float(per_pos["mse"][mask].mean()),
        }

    out = {
        "path": path,
        "params": model.num_parameters(),
        "config": model.cfg.to_dict(),
        "step": ckpt.get("step"),
        "overall": agg(np.ones(n, bool)),
    }
    out["by_ply"] = {f"{lo}-{hi}": agg((data["ply"] >= lo) & (data["ply"] < hi)) for lo, hi in PLY_BUCKETS}
    out["by_elo"] = {f"{lo}-{hi}": agg((data["elo"] >= lo) & (data["elo"] < hi)) for lo, hi in ELO_BUCKETS}
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+")
    p.add_argument("--data", default="data/shards/test")
    p.add_argument("--max-positions", type=int, default=400_000)
    p.add_argument("--out", type=Path, default=Path("results/accuracy.json"))
    a = p.parse_args(argv)
    shards = [load_shard(s) for s in list_shards(a.data)]
    data = concat_positions({k: [s[k] for s in shards] for k in shards[0]}) if len(shards) > 1 else shards[0]
    if len(data["bb"]) > a.max_positions:
        from azchess.data.shards import take_positions

        data = take_positions(data, np.arange(a.max_positions))
    results = defaultdict(dict)
    if a.out.exists():
        results.update(json.loads(a.out.read_text()))
    for m in a.models:
        r = evaluate_model(m, data)
        results[m] = r
        o = r["overall"]
        print(
            f"{m}: top1 {o['top1']:.4f} top5 {o['top5']:.4f} CE {o['policy_ce']:.4f} value MSE {o['value_mse']:.4f} ({o['positions']:,} positions)"
        )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
