"""ABSOL model: tile encoder + hetero-GNN + temporal refiner + mask decoder."""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import HeteroData

from absol.features import N_PHYS
from absol.model.encoder import TileEncoder
from absol.model.gnn import HeteroGNN
from absol.model.temporal import TemporalRefiner


class MaskDecoder(nn.Module):
    """Tiny ConvTranspose decoder: bl-node embedding -> [t_c, f_c] sample-mask
    logits. Applied only where p_chunk > gate at inference.

    Works for ANY chunk size: deconv 8x from the ceil(t_c/8) x ceil(f_c/8)
    seed grid, then bilinear-resize to the exact (t_c, f_c). So
    chunk_time_samples / chunk_freq_channels are freely configurable.
    """

    def __init__(self, d_in: int, t_c: int, f_c: int):
        super().__init__()
        self.t_c, self.f_c = t_c, f_c
        self.t0, self.f0 = -(-t_c // 8), -(-f_c // 8)        # ceil-div
        self.lin = nn.Linear(d_in, 32 * self.t0 * self.f0)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.lin(h).reshape(-1, 32, self.t0, self.f0)
        y = self.deconv(x)                                   # [-1, 1, 8*t0, 8*f0]
        if y.shape[-2:] != (self.t_c, self.f_c):
            y = torch.nn.functional.interpolate(
                y, size=(self.t_c, self.f_c), mode="bilinear", align_corners=False
            )
        return y.squeeze(1)


class AbsolModel(nn.Module):
    """Full detector. forward() consumes the per-column hetero-graphs of one
    scene/scan and returns logits per the data contract."""

    def __init__(self, model_cfg: dict, t_c: int, f_c: int):
        super().__init__()
        enc, gnn, tmp = model_cfg["encoder"], model_cfg["gnn"], model_cfg["temporal"]
        self.encoder = TileEncoder(
            in_ch=2, channels=tuple(enc["cnn_channels"]),
            embed_dim=int(enc["embed_dim"]), n_phys=N_PHYS,
        )
        self.gnn = HeteroGNN(
            in_dim=int(enc["embed_dim"]), hidden=int(gnn["hidden_dim"]),
            n_rounds=int(gnn["n_rounds"]), heads=int(gnn["heads"]),
        )
        self.temporal = TemporalRefiner(
            d_model=int(gnn["hidden_dim"]), n_layers=int(tmp["n_layers"]),
            n_heads=int(tmp["n_heads"]),
        )
        self.mask_decoder = MaskDecoder(int(gnn["hidden_dim"]), t_c, f_c)

    def forward(
        self,
        columns: list[HeteroData],
        mask_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """mask_index: flat indices into [N_bl_nodes * Nt] (node-major) for
        which sample-mask logits are decoded; None decodes none.

        Returns: edge_logits [Nbl, Nt], ant_logits [A, Nt],
        mask_logits [M, t_c, f_c] (or None), h [Nbl, Nt, d].
        """
        hs, e0, al = [], [], []
        for g in columns:
            emb = self.encoder(g["bl"].tile, g["bl"].phys)
            out = self.gnn(g, emb)
            hs.append(out["h_bl"])
            e0.append(out["edge_logit"])
            al.append(out["ant_logit"])
        h = torch.stack(hs, dim=1)                       # [Nbl, Nt, d]
        e0 = torch.stack(e0, dim=1)                      # [Nbl, Nt]
        hr, delta = self.temporal(h)
        edge_logits = e0 + delta
        mask_logits = None
        if mask_index is not None and mask_index.numel() > 0:
            flat = hr.reshape(-1, hr.shape[-1])
            mask_logits = self.mask_decoder(flat[mask_index])
        return {
            "edge_logits": edge_logits,
            "ant_logits": torch.stack(al, dim=1),        # [A, Nt]
            "mask_logits": mask_logits,
            "h": hr,
        }

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
