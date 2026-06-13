"""Chunk CNN encoder: delay-fringe tile + physics features -> embedding."""
from __future__ import annotations

import torch
from torch import nn


class TileEncoder(nn.Module):
    """CNN over the [2, t_c, f_c] tile; physics features concatenated after
    global average pooling. Shared across all bl-nodes."""

    def __init__(
        self, in_ch: int = 2, channels: tuple[int, ...] = (16, 32, 64),
        embed_dim: int = 64, n_phys: int = 14,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_ch
        for k, ch in enumerate(channels):
            stride = (1, 2) if k == 0 else 2
            layers += [
                nn.Conv2d(prev, ch, 3, stride=stride, padding=1),
                nn.GroupNorm(min(4, ch), ch),
                nn.GELU(),
            ]
            prev = ch
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(prev + n_phys, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, tile: torch.Tensor, phys: torch.Tensor) -> torch.Tensor:
        h = self.pool(self.cnn(tile)).flatten(1)
        return self.proj(torch.cat([h, phys], dim=1))
