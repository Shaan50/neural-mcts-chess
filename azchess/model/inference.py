"""Batched network inference for search and play.

``Evaluator`` turns compact encodings (as produced by the C++ batch gatherer or
the Python backends) into legal-move logits and WDL probabilities:

* compact -> dense planes conversion happens on the device;
* batches are padded to power-of-two buckets so MPS / CUDA graph caches and
  cuDNN autotuning see a handful of shapes rather than hundreds;
* only the logits of legal moves are gathered on the device and copied back,
  ~35 floats per position instead of 4672.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from azchess.encoding.board import planes_from_compact_torch
from azchess.encoding.moves import POLICY_SIZE
from azchess.model.network import AZNet, ModelConfig, value_to_wdl


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device: torch.device) -> torch.dtype | None:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return None


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu", use_ema: bool = True) -> tuple[AZNet, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["model_config"])
    model = AZNet(cfg)
    state = ckpt.get("ema") if use_ema and ckpt.get("ema") is not None else ckpt["model"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model, ckpt


def _bucket(n: int, minimum: int = 8) -> int:
    b = minimum
    while b < n:
        b *= 2
    return b


class Evaluator:
    def __init__(self, model: AZNet, device: torch.device | str | None = None, half: bool = True, pad: bool = True):
        self.model = model.eval()
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model.to(self.device)
        self.amp = autocast_dtype(self.device) if half else None
        self.pad = pad
        self.value_head = model.cfg.value_head
        self.positions = 0
        self.calls = 0

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto", use_ema: bool = True, **kw) -> Evaluator:
        dev = pick_device(device)
        model, _ = load_checkpoint(path, dev, use_ema=use_ema)
        return cls(model, dev, **kw)

    @torch.inference_mode()
    def warmup(self, max_batch: int = 256) -> None:
        """Run every padded batch shape once so kernel compilation (MPS graph
        building, cuDNN autotuning) does not eat into the first timed search."""
        b = 8
        while b <= max(8, max_batch):
            bb = np.zeros((b, 12), np.int64)
            meta = np.zeros((b, 4), np.uint8)
            self._forward(bb, meta)
            b *= 2
        self.positions = self.calls = 0

    def _forward(self, bb: np.ndarray, meta: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        n = bb.shape[0]
        size = _bucket(n) if self.pad else n
        bb_t = torch.zeros((size, 12), dtype=torch.int64)
        meta_t = torch.zeros((size, 4), dtype=torch.uint8)
        bb_t[:n] = torch.from_numpy(np.ascontiguousarray(bb).view(np.int64))
        meta_t[:n] = torch.from_numpy(np.ascontiguousarray(meta))
        x = planes_from_compact_torch(
            bb_t.to(self.device, non_blocking=True), meta_t.to(self.device, non_blocking=True)
        )
        with torch.autocast(self.device.type, dtype=self.amp, enabled=self.amp is not None):
            logits, value = self.model(x)
        self.positions += n
        self.calls += 1
        return logits[:n].float(), value_to_wdl(value[:n], self.value_head)

    @torch.inference_mode()
    def evaluate_legal(self, bb: np.ndarray, meta: np.ndarray, legal_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (logits of the legal moves in ``legal_flat`` order, wdl (B, 3))."""
        if bb.shape[0] == 0:
            return np.zeros(0, np.float32), np.zeros((0, 3), np.float32)
        logits, wdl = self._forward(bb, meta)
        idx = torch.from_numpy(np.ascontiguousarray(legal_flat, dtype=np.int64)).to(self.device)
        legal = logits.reshape(-1).index_select(0, idx)
        return legal.cpu().numpy(), wdl.cpu().numpy()

    @torch.inference_mode()
    def evaluate_full(self, bb: np.ndarray, meta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits, wdl = self._forward(bb, meta)
        return logits.cpu().numpy(), wdl.cpu().numpy()

    __call__ = evaluate_legal


def legal_flat_from_lists(policy_lists: list[np.ndarray]) -> np.ndarray:
    """Concatenate per-position legal policy indices into flat (row * 4672 + idx) form."""
    if not policy_lists:
        return np.zeros(0, np.int64)
    rows = np.repeat(np.arange(len(policy_lists), dtype=np.int64), [len(p) for p in policy_lists])
    return rows * POLICY_SIZE + np.concatenate(policy_lists).astype(np.int64)
