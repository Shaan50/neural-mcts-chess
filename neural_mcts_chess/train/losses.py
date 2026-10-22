"""Policy/value losses and metrics shared by supervised and self-play training."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from neural_mcts_chess.encoding.moves import POLICY_SIZE

MASK_VALUE = -1e9


def legal_mask(legal_flat: torch.Tensor, batch: int, device) -> torch.Tensor:
    mask = torch.zeros(batch * POLICY_SIZE, dtype=torch.bool, device=device)
    mask[legal_flat] = True
    return mask.view(batch, POLICY_SIZE)


def policy_targets(policy_idx: torch.Tensor, policy_p: torch.Tensor) -> torch.Tensor:
    """Sparse (B, K) indices/probabilities -> dense (B, 4672) distribution. Index -1 = padding."""
    valid = policy_idx >= 0
    dense = torch.zeros(policy_idx.shape[0], POLICY_SIZE, device=policy_idx.device)
    dense.scatter_add_(1, policy_idx.clamp(min=0), policy_p * valid)
    return dense / dense.sum(-1, keepdim=True).clamp(min=1e-8)


def value_targets(result: torch.Tensor, q: torch.Tensor, q_ratio: float) -> torch.Tensor:
    """WDL target distribution from the game result, optionally blended with the
    search value q (``(1 - q_ratio) * z + q_ratio * q``) where q is known."""
    z = F.one_hot((1 - result).long(), 3).float()  # result +1 -> class 0 (win)
    if q_ratio <= 0:
        return z
    qv = q.clamp(-1, 1)
    q_wdl = torch.stack([qv.clamp(min=0), 1 - qv.abs(), (-qv).clamp(min=0)], dim=-1)
    known = ~torch.isnan(q)
    mix = torch.where(known[:, None], (1 - q_ratio) * z + q_ratio * torch.nan_to_num(q_wdl), z)
    return mix


def compute_losses(
    policy_logits: torch.Tensor,
    value_out: torch.Tensor,
    batch: dict[str, torch.Tensor],
    value_head: str = "wdl",
    q_ratio: float = 0.0,
    mask_illegal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    b = policy_logits.shape[0]
    logits = policy_logits.float()
    mask = legal_mask(batch["legal_flat"], b, logits.device)
    if mask_illegal:
        logits = logits.masked_fill(~mask, MASK_VALUE)
    logp = F.log_softmax(logits, dim=-1)
    target = policy_targets(batch["policy_idx"], batch["policy_p"])
    policy_loss = -(target * logp).sum(-1).mean()

    result = batch["result"]
    if value_head == "wdl":
        vt = value_targets(result, batch["q"], q_ratio)
        vlogp = F.log_softmax(value_out.float(), dim=-1)
        value_loss = -(vt * vlogp).sum(-1).mean()
        wdl = vlogp.exp()
        expected = wdl[:, 0] - wdl[:, 2]
    else:
        expected = value_out.float().view(-1)
        z = result.float()
        if q_ratio > 0:
            q = batch["q"]
            z = torch.where(torch.isnan(q), z, (1 - q_ratio) * z + q_ratio * q)
        value_loss = F.mse_loss(expected, z)

    with torch.no_grad():
        masked = logits if mask_illegal else logits.masked_fill(~mask, MASK_VALUE)
        best = target.argmax(-1)
        top5 = masked.topk(5, dim=-1).indices
        metrics = {
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "policy_top1": (top5[:, 0] == best).float().mean(),
            "policy_top5": (top5 == best[:, None]).any(-1).float().mean(),
            # entropy of the predicted move distribution (legal moves only)
            "policy_entropy": -(F.softmax(masked, -1) * F.log_softmax(masked, -1)).sum(-1).mean(),
            "value_mse": F.mse_loss(expected, result.float()),
            "value_sign_acc": (
                (expected > 0.33) & (result > 0)
                | (expected < -0.33) & (result < 0)
                | (expected.abs() <= 0.33) & (result == 0)
            )
            .float()
            .mean(),
        }
    return policy_loss, value_loss, metrics
