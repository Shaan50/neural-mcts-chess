"""Training loop used by both supervised pretraining and self-play refinement.

Features: mixed precision (fp16 + loss scaling on MPS/CUDA, bf16 on Ampere+),
EMA weights (used for evaluation and play), warmup + cosine LR schedule,
weight decay excluding norms/biases, gradient clipping, full checkpoint /
resume (model, EMA, optimiser, scheduler, scaler, RNG, data position), periodic
weight snapshots for Elo-vs-training-step evaluation, TensorBoard + JSONL
metrics (and Weights & Biases when installed and enabled).
"""

from __future__ import annotations

import copy
import json
import math
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from azchess.encoding.board import planes_from_compact_torch
from azchess.model.inference import autocast_dtype, pick_device
from azchess.model.network import AZNet, ModelConfig
from azchess.train.losses import compute_losses


@dataclass
class TrainConfig:
    run_dir: str = "runs/supervised"
    train_data: list[str] = field(default_factory=lambda: ["data/shards/train"])
    val_data: list[str] = field(default_factory=lambda: ["data/shards/val"])
    model: dict = field(default_factory=dict)
    batch_size: int = 1024
    steps: int = 20_000
    optimizer: str = "adamw"  # "adamw" | "sgd"
    lr: float = 2e-3
    min_lr_ratio: float = 0.02
    warmup_steps: int = 500
    weight_decay: float = 1e-4
    momentum: float = 0.9
    value_weight: float = 1.0
    q_ratio: float = 0.0
    grad_clip: float = 10.0
    amp: bool = True
    ema_decay: float = 0.999
    workers: int = 4
    log_every: int = 50
    val_every: int = 2000
    val_positions: int = 200_000
    ckpt_every: int = 2000
    snapshot_every: int = 2000
    seed: int = 0
    device: str = "auto"
    init_from: str | None = None
    wandb: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown training config keys: {sorted(unknown)}")
        return cls(**d)


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_lambda(cfg: TrainConfig):
    def f(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        t = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps))
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t))

    return f


class EMA:
    """Exponential moving average of weights with a warmup on the decay
    (``min(decay, (1 + n) / (10 + n))``) so early snapshots are usable.

    BatchNorm running statistics are averaged with the same decay. Copying
    them from the live model instead pairs averaged weights with statistics
    of different weights, which badly miscalibrates the value head (measured:
    held-out value loss 0.95 vs 0.87 for the raw weights)."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        ema_p = list(self.module.parameters())
        torch._foreach_lerp_(ema_p, list(model.parameters()), 1.0 - d)
        for eb, mb in zip(self.module.buffers(), model.buffers()):
            if eb.is_floating_point():
                eb.lerp_(mb, 1.0 - d)
            else:
                eb.copy_(mb)  # num_batches_tracked


class MetricLogger:
    def __init__(self, run_dir: Path, cfg: TrainConfig):
        from torch.utils.tensorboard import SummaryWriter

        self.tb = SummaryWriter(str(run_dir / "tb"))
        self.jsonl = open(run_dir / "metrics.jsonl", "a")
        self.wandb = None
        if cfg.wandb:
            try:
                import wandb

                self.wandb = wandb.init(
                    project="azchess", dir=str(run_dir), config=asdict(cfg), name=run_dir.name, resume="allow"
                )
            except ImportError:
                print("[train] wandb not installed; continuing with TensorBoard only")

    def log(self, step: int, prefix: str, values: dict[str, float]) -> None:
        for k, v in values.items():
            self.tb.add_scalar(f"{prefix}/{k}", v, step)
        self.jsonl.write(json.dumps({"step": step, "split": prefix, **values}) + "\n")
        self.jsonl.flush()
        if self.wandb is not None:
            self.wandb.log({f"{prefix}/{k}": v for k, v in values.items()}, step=step)

    def close(self) -> None:
        self.tb.close()
        self.jsonl.close()
        if self.wandb is not None:
            self.wandb.finish()


class Trainer:
    def __init__(self, cfg: TrainConfig, model: AZNet | None = None):
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "snapshots").mkdir(exist_ok=True)
        torch.manual_seed(cfg.seed)
        self.device = pick_device(cfg.device)
        self.model_cfg = model.cfg if model is not None else ModelConfig.from_dict(cfg.model)
        self.model = (model or AZNet(self.model_cfg)).to(self.device)
        if cfg.init_from:
            ckpt = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt.get("ema") or ckpt["model"])
            print(f"[train] initialised weights from {cfg.init_from}")
        if cfg.optimizer == "sgd":
            self.opt = torch.optim.SGD(
                param_groups(self.model, cfg.weight_decay), lr=cfg.lr, momentum=cfg.momentum, nesterov=True
            )
        else:
            self.opt = torch.optim.AdamW(param_groups(self.model, cfg.weight_decay), lr=cfg.lr, betas=(0.9, 0.99))
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda(cfg))
        self.amp_dtype = autocast_dtype(self.device) if cfg.amp else None
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp_dtype == torch.float16)
        self.ema = EMA(self.model, cfg.ema_decay) if cfg.ema_decay > 0 else None
        self.step = 0
        self.positions_seen = 0
        self.best_val = float("inf")
        self.logger = MetricLogger(self.run_dir, cfg)
        (self.run_dir / "config.json").write_text(
            json.dumps({"train": asdict(cfg), "model": self.model_cfg.to_dict()}, indent=2)
        )
        last = self.run_dir / "last.pt"
        if last.exists():
            self.load(last)

    # ------------------------------------------------------------------ io
    def eval_model(self) -> AZNet:
        return self.ema.module if self.ema is not None else self.model

    def state(self, full: bool = True) -> dict:
        s = {
            "model": self.model.state_dict(),
            "ema": self.ema.module.state_dict() if self.ema is not None else None,
            "model_config": self.model_cfg.to_dict(),
            "step": self.step,
            "positions_seen": self.positions_seen,
        }
        if full:
            s.update(
                optimizer=self.opt.state_dict(),
                scheduler=self.sched.state_dict(),
                scaler=self.scaler.state_dict(),
                best_val=self.best_val,
                train_config=asdict(self.cfg),
                rng=torch.get_rng_state(),
            )
        return s

    def save(self, path: Path, full: bool = True) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save(self.state(full), tmp)
        tmp.replace(path)

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if self.ema is not None and ckpt.get("ema") is not None:
            self.ema.module.load_state_dict(ckpt["ema"])
            self.ema.updates = ckpt["step"]
        self.opt.load_state_dict(ckpt["optimizer"])
        self.sched.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.step = ckpt["step"]
        self.positions_seen = ckpt.get("positions_seen", 0)
        self.best_val = ckpt.get("best_val", float("inf"))
        torch.set_rng_state(ckpt["rng"])
        print(f"[train] resumed from {path} at step {self.step}")

    # ------------------------------------------------------------- compute
    def _to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _forward(self, model: AZNet, batch: dict[str, torch.Tensor]):
        x = planes_from_compact_torch(batch["bb"], batch["meta"])
        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
            p, v = model(x)
        return compute_losses(p, v, batch, model.cfg.value_head, self.cfg.q_ratio)

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.model.train()
        batch = self._to_device(batch)
        pl, vl, metrics = self._forward(self.model, batch)
        loss = pl + self.cfg.value_weight * vl
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()
        if self.ema is not None:
            self.ema.update(self.model)
        self.step += 1
        self.positions_seen += batch["bb"].shape[0]
        metrics["loss"] = loss.detach()
        metrics["grad_norm"] = gnorm.detach()
        return metrics

    @torch.no_grad()
    def validate(self, batches: Iterable[dict[str, torch.Tensor]], model: AZNet | None = None) -> dict[str, float]:
        model = model or self.eval_model()
        model.eval()
        sums: dict[str, float] = defaultdict(float)
        n = 0
        for batch in batches:
            batch = self._to_device(batch)
            bsz = batch["bb"].shape[0]
            _, _, metrics = self._forward(model, batch)
            for k, v in metrics.items():
                sums[k] += float(v) * bsz
            n += bsz
        return {k: v / max(n, 1) for k, v in sums.items()} | {"positions": n}

    def fit(self, train_batches: Iterable, val_factory=None, steps: int | None = None) -> None:
        cfg = self.cfg
        target = steps if steps is not None else cfg.steps
        running: dict[str, float] = defaultdict(float)
        count = 0
        t0, pos0 = time.time(), self.positions_seen
        it = iter(train_batches)
        while self.step < target:
            metrics = self.train_step(next(it))
            for k, v in metrics.items():
                running[k] += float(v)
            count += 1
            if self.step % cfg.log_every == 0:
                elapsed = time.time() - t0
                vals = {k: v / count for k, v in running.items()}
                vals["lr"] = self.sched.get_last_lr()[0]
                vals["pos_per_s"] = (self.positions_seen - pos0) / max(elapsed, 1e-9)
                vals["positions_seen"] = self.positions_seen
                self.logger.log(self.step, "train", vals)
                print(
                    f"[train] step {self.step:6d} loss {vals['loss']:.4f} pol {vals['policy_loss']:.4f} "
                    f"val {vals['value_loss']:.4f} top1 {vals['policy_top1']:.3f} lr {vals['lr']:.2e} "
                    f"{vals['pos_per_s']:,.0f} pos/s",
                    flush=True,
                )
                running.clear()
                count = 0
                t0, pos0 = time.time(), self.positions_seen
            if val_factory is not None and (self.step % cfg.val_every == 0 or self.step == target):
                vm = self.validate(val_factory())
                self.logger.log(self.step, "val", vm)
                print(
                    f"[val]   step {self.step:6d} pol {vm['policy_loss']:.4f} top1 {vm['policy_top1']:.4f} "
                    f"top5 {vm['policy_top5']:.4f} value {vm['value_loss']:.4f} mse {vm['value_mse']:.4f}",
                    flush=True,
                )
                score = vm["policy_loss"] + cfg.value_weight * vm["value_loss"]
                if score < self.best_val:
                    self.best_val = score
                    self.save(self.run_dir / "best.pt", full=False)
            if self.step % cfg.snapshot_every == 0 or self.step == target:
                self.save(self.run_dir / "snapshots" / f"step_{self.step:07d}.pt", full=False)
            if self.step % cfg.ckpt_every == 0 or self.step == target:
                self.save(self.run_dir / "last.pt")
        self.save(self.run_dir / "last.pt")
        self.save(self.run_dir / "final.pt", full=False)

    def close(self) -> None:
        self.logger.close()
