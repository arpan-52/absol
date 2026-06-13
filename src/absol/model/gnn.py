"""Heterogeneous message passing (GATv2 per relation) + output heads."""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv

from absol.graphs import REL_REV, REL_SAME_DIR, REL_SAME_UV, REL_SHARES


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))


class HeteroGNN(nn.Module):
    def __init__(self, in_dim: int = 64, hidden: int = 96, n_rounds: int = 3,
                 heads: int = 4, ant_in: int = 4):
        super().__init__()
        assert hidden % heads == 0
        self.lin_bl = nn.Linear(in_dim, hidden)
        self.lin_ant = nn.Linear(ant_in, hidden)
        out = hidden // heads
        self.rounds = nn.ModuleList()
        for _ in range(n_rounds):
            self.rounds.append(HeteroConv({
                REL_SAME_DIR: GATv2Conv(hidden, out, heads=heads, add_self_loops=False),
                REL_SAME_UV: GATv2Conv(hidden, out, heads=heads, add_self_loops=False),
                REL_SHARES: GATv2Conv((hidden, hidden), out, heads=heads, add_self_loops=False),
                REL_REV: GATv2Conv((hidden, hidden), out, heads=heads, add_self_loops=False),
            }, aggr="sum"))
        self.norms = nn.ModuleList([
            nn.ModuleDict({"bl": nn.LayerNorm(hidden), "antenna": nn.LayerNorm(hidden)})
            for _ in range(n_rounds)
        ])
        self.edge_head = _mlp(hidden, hidden, 1)
        self.ant_head = _mlp(hidden, hidden, 1)

    def forward(self, data: HeteroData, bl_emb: torch.Tensor) -> dict[str, torch.Tensor]:
        h = {"bl": self.lin_bl(bl_emb), "antenna": self.lin_ant(data["antenna"].x)}
        edge_index_dict = {k: data[k].edge_index for k in data.edge_types}
        for conv, norm in zip(self.rounds, self.norms):
            out = conv(h, edge_index_dict)
            h = {
                nt: norm[nt](h[nt] + torch.nn.functional.gelu(out[nt]))
                if nt in out else h[nt]
                for nt in h
            }
        return {
            "h_bl": h["bl"],
            "h_ant": h["antenna"],
            "edge_logit": self.edge_head(h["bl"]).squeeze(-1),
            "ant_logit": self.ant_head(h["antenna"]).squeeze(-1),
        }
