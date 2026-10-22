"""Residual policy/value network.

Architecture (AlphaZero with the Leela Chess Zero refinements that matter at
small scale):

* input convolution 3x3, ``filters`` channels, batch norm, ReLU
* ``blocks`` residual blocks: conv-BN-ReLU-conv-BN, optional squeeze-and-
  excitation (SE) that produces a per-channel scale *and* bias, skip, ReLU
* policy head: conv 3x3 -> ReLU -> conv 1x1 to 73 planes = 8x8x73 = 4672 logits
  (one per move-encoding slot, see encoding/moves.py)
* value head: conv 1x1 (32) -> FC 128 -> FC 3 for win/draw/loss logits
  ("wdl", default) or FC 1 + tanh ("scalar", the original AlphaZero head)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_mcts_chess.encoding.board import NUM_PLANES
from neural_mcts_chess.encoding.moves import NUM_MOVE_PLANES, POLICY_SIZE


@dataclass
class ModelConfig:
    blocks: int = 6
    filters: int = 64
    se_ratio: int = 4  # 0 disables squeeze-and-excitation
    value_head: str = "wdl"  # "wdl" or "scalar"
    value_channels: int = 32
    value_hidden: int = 128
    policy_channels: int = 0  # 0 -> same as filters
    input_planes: int = NUM_PLANES

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


def conv_bn(cin: int, cout: int, k: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout))


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, ratio: int):
        super().__init__()
        hidden = max(channels // ratio, 8)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, 2 * channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # reshape+sum and a fused addcmul are ~20% faster on MPS than mean(dim=(2,3)) + broadcasting
        b, c = x.shape[:2]
        s = x.reshape(b, c, 64).sum(-1) * (1.0 / 64)
        s = self.fc2(F.relu(self.fc1(s)))
        scale, bias = s.view(b, 2 * c, 1, 1).split(c, dim=1)
        return torch.addcmul(bias, x, torch.sigmoid(scale))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, se_ratio: int):
        super().__init__()
        self.conv1 = conv_bn(channels, channels, 3)
        self.conv2 = conv_bn(channels, channels, 3)
        self.se = SqueezeExcite(channels, se_ratio) if se_ratio > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x))
        y = self.se(self.conv2(y))
        return F.relu(x + y)


class AZNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        f = cfg.filters
        self.stem = conv_bn(cfg.input_planes, f, 3)
        self.tower = nn.Sequential(*[ResidualBlock(f, cfg.se_ratio) for _ in range(cfg.blocks)])
        pc = cfg.policy_channels or f
        self.policy_conv = conv_bn(f, pc, 3)
        self.policy_out = nn.Conv2d(pc, NUM_MOVE_PLANES, 1)
        self.value_conv = conv_bn(f, cfg.value_channels, 1)
        self.value_fc1 = nn.Linear(cfg.value_channels * 64, cfg.value_hidden)
        self.value_fc2 = nn.Linear(cfg.value_hidden, 3 if cfg.value_head == "wdl" else 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero-init the last BN of every residual branch so blocks start as identity.
        for blk in self.tower:
            nn.init.zeros_(blk.conv2[1].weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.stem(x))
        x = self.tower(x)
        p = self.policy_out(F.relu(self.policy_conv(x))).flatten(1)  # (B, 73*64) plane-major
        v = F.relu(self.value_conv(x)).flatten(1)
        v = self.value_fc2(F.relu(self.value_fc1(v)))
        if self.cfg.value_head == "scalar":
            v = torch.tanh(v)
        return p, v

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def value_to_wdl(v: torch.Tensor, value_head: str) -> torch.Tensor:
    """Network value output -> (B, 3) win/draw/loss probabilities."""
    if value_head == "wdl":
        return torch.softmax(v.float(), dim=-1)
    s = v.float().view(-1)
    return torch.stack([(1 + s) / 2, torch.zeros_like(s), (1 - s) / 2], dim=-1)


assert POLICY_SIZE == NUM_MOVE_PLANES * 64
