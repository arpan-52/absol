"""Per-baseline temporal transformer over the time-chunk axis."""
from __future__ import annotations

import math

import torch
from torch import nn


def _sincos(n: int, d: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(n, device=device).float()[:, None]
    i = torch.arange(0, d, 2, device=device).float()
    div = torch.exp(-math.log(10000.0) * i / d)
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class TemporalRefiner(nn.Module):
    """Refines per-(bl-node) embeddings and logits across time-chunk columns."""

    def __init__(self, d_model: int = 96, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
            batch_first=True, dropout=0.0, activation="gelu", norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.delta = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h [N, Nt, d] -> (refined h [N, Nt, d], delta logits [N, Nt])."""
        hr = self.enc(h + _sincos(h.shape[1], h.shape[2], h.device)[None])
        return hr, self.delta(hr).squeeze(-1)
