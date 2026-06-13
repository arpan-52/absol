"""Losses: focal BCE (edge), BCE (antenna), dice+BCE (sample mask).

Transient-protected chunks contribute as hard negatives with configurable
extra weight (spec: 2x).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_bce_logits(
    logits: torch.Tensor, targets: torch.Tensor, gamma: float,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = bce * (1 - p_t).clamp(min=1e-6) ** gamma
    if weight is not None:
        loss = loss * weight
        return loss.sum() / weight.sum().clamp(min=1)
    return loss.mean()


def dice_loss(probs: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None = None,
              eps: float = 1.0) -> torch.Tensor:
    if weight is not None:
        probs, targets = probs * weight, targets * weight
    num = 2 * (probs * targets).sum() + eps
    den = probs.sum() + targets.sum() + eps
    return 1 - num / den


def compute_losses(out: dict, labels: dict, cfg_train: dict) -> dict[str, torch.Tensor]:
    lw = cfg_train["loss_weights"]
    gamma = float(cfg_train["focal_gamma"])

    edge = focal_bce_logits(
        out["edge_logits"], labels["edge_target"], gamma, weight=labels["edge_weight"]
    )
    ant = F.binary_cross_entropy_with_logits(
        out["ant_logits"], labels["ant_target"].expand_as(out["ant_logits"])
    )
    if out["mask_logits"] is not None and labels["mask_target"].numel() > 0:
        mp = torch.sigmoid(out["mask_logits"])
        mt, mw = labels["mask_target"], labels["mask_weight"]
        mask = (
            F.binary_cross_entropy_with_logits(out["mask_logits"], mt, weight=mw)
            + dice_loss(mp, mt, weight=mw)
        )
    else:
        mask = torch.zeros((), device=out["edge_logits"].device)

    total = (
        float(lw["edge_chunk"]) * edge
        + float(lw["antenna"]) * ant
        + float(lw["sample_mask"]) * mask
    )
    return {"total": total, "edge": edge.detach(), "antenna": ant.detach(),
            "mask": mask.detach()}
