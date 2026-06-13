"""Shared preprocessing pipeline: visibilities -> tiles/features -> graphs.

Used verbatim by the training dataset and by MS inference, so the model
sees identical preprocessing in both. Pre-existing flags (weights == 0) are
respected throughout (see features.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import HeteroData

from absol.features import (
    chunk_grid,
    delay_fringe_tiles,
    mad_normalize,
    physics_features,
    savgol_residual,
)
from absol.geometry import Array
from absol.graphs import StaticRelations, build_column_graph, build_static


@dataclass
class Prepared:
    columns: list[HeteroData]    # one hetero-graph per time-chunk column
    chunks: torch.Tensor         # [B, Nt, Nf, t_c, f_c, P] normalized
    wchunks: torch.Tensor        # [B, Nt, Nf, t_c, f_c]
    validity: torch.Tensor       # [B, Nt, Nf]
    tiles: torch.Tensor          # [B, Nt, Nf, 2, t_c, f_c]
    sigma_mad: torch.Tensor      # [B, P]
    static: StaticRelations
    n_t: int                     # original (unpadded) T
    n_f: int                     # original (unpadded) F
    t_c: int
    f_c: int


def prepare(
    vis: torch.Tensor,           # [B, T, F, P] raw (fringe-stopped frame)
    weights: torch.Tensor,       # [B, T, F]
    array: Array,
    ha_rad: np.ndarray,
    dec_rad: float,
    freqs_hz: np.ndarray,
    sim_cfg: dict,
    model_cfg: dict,
    seed: int = 0,
) -> Prepared:
    ch_cfg = sim_cfg["chunking"]
    t_c, f_c = int(ch_cfg["chunk_time_samples"]), int(ch_cfg["chunk_freq_channels"])
    res_cfg = sim_cfg.get("residual", {})
    win = int(res_cfg.get("savgol_window_dumps", 25))
    order = int(res_cfg.get("savgol_order", 3))

    res = savgol_residual(vis, weights, win, order) if win > 0 else vis
    vis_n, sigma = mad_normalize(res, weights)
    chunks, wch, validity = chunk_grid(vis_n, weights, t_c, f_c)
    tiles = delay_fringe_tiles(chunks, wch)

    ha_mid = float(ha_rad[len(ha_rad) // 2])
    nf = chunks.shape[2]
    static = build_static(array, ha_mid, dec_rad, freqs_hz, nf, model_cfg["graph"])

    feats = physics_features(
        chunks, wch, tiles, validity, static.pairs, array.n_ant,
        static.uv_dist_m, seed=seed,
    )
    columns = [
        build_column_graph(
            static, tiles[:, nt], feats[:, nt], validity[:, nt],
            array.integration_s, model_cfg["graph"],
        )
        for nt in range(chunks.shape[1])
    ]
    return Prepared(
        columns=columns, chunks=chunks, wchunks=wch, validity=validity,
        tiles=tiles, sigma_mad=sigma, static=static,
        n_t=vis.shape[1], n_f=vis.shape[2], t_c=t_c, f_c=f_c,
    )
