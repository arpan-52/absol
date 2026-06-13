"""Hetero-graph construction (relations A/B/C) for PyG.

One `HeteroData` per time-chunk column. Node types:
- ``antenna`` [A]: position features.
- ``bl`` [B * Nf]: line-graph node per (baseline, freq-tile), carrying the
  delay-fringe tile + physics features. Flat index = b * Nf + nf (this
  ordering is relied on by the temporal stage - do not change).

Relations:
- A ``(bl, shares, antenna)`` + reverse: each bl-node to its two antennas.
- B ``(bl, same_dir, bl)``: connect bl-nodes whose MEASURED tile fringe
  rates are both within tolerance of the SAME direction bucket
  (`geometry.direction_buckets` - single source of truth), degree-capped.
  Data-dependent; the static bucket rate table is precomputed and cached.
- C ``(bl, same_uvcell, bl)``: same/adjacent UV cell (cell size from
  config), static per scene.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import HeteroData

from absol.geometry import Array, DirectionBuckets, direction_buckets, uvw

REL_SAME_DIR = ("bl", "same_dir", "bl")
REL_SAME_UV = ("bl", "same_uvcell", "bl")
REL_SHARES = ("bl", "shares", "antenna")
REL_REV = ("antenna", "rev_shares", "bl")


@dataclass
class StaticRelations:
    """Per-scene static graph parts (cacheable per array/pointing/lst cell)."""

    pairs: np.ndarray            # [B, 2]
    n_ant: int
    n_freq_tiles: int
    shares: np.ndarray           # [2, E] bl-node -> antenna
    uv_edges: np.ndarray         # [2, E] bl-node -> bl-node (same column pattern)
    buckets: DirectionBuckets
    uv_dist_m: np.ndarray        # [B]
    tile_freq_hz: np.ndarray     # [Nf] tile centre frequencies
    ant_x: np.ndarray            # [A, 4] antenna node features


def build_static(
    array: Array,
    ha_mid_rad: float,
    dec_rad: float,
    freqs_hz: np.ndarray,
    n_freq_tiles: int,
    cfg_graph: dict,
) -> StaticRelations:
    pairs = array.baselines()
    n_b, a = pairs.shape[0], array.n_ant
    nf = n_freq_tiles
    n_nodes = n_b * nf

    # relation A: bl-node -> its two antennas (per freq tile)
    node = np.arange(n_nodes)
    b_of = node // nf
    src = np.concatenate([node, node])
    dst = np.concatenate([pairs[b_of, 0], pairs[b_of, 1]])
    shares = np.stack([src, dst])

    # relation C: same/adjacent uv cell, capped nearest-by-uv, same freq tile
    uv = uvw(array, ha_mid_rad, dec_rad)[:, :2]                  # [B, 2] metres
    cell = float(cfg_graph.get("uv_cell_m", 400.0))
    cap = int(cfg_graph.get("degree_cap", 16))
    cu, cv = np.floor(uv[:, 0] / cell).astype(int), np.floor(uv[:, 1] / cell).astype(int)
    cells: dict[tuple[int, int], list[int]] = {}
    for b in range(n_b):
        cells.setdefault((cu[b], cv[b]), []).append(b)
    e_src, e_dst = [], []
    for b in range(n_b):
        cand: list[int] = []
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                cand += cells.get((cu[b] + du, cv[b] + dv), [])
        cand = [c for c in cand if c != b]
        if not cand:
            continue
        d = np.hypot(uv[cand, 0] - uv[b, 0], uv[cand, 1] - uv[b, 1])
        for c in np.asarray(cand)[np.argsort(d)[:cap]]:
            e_src.append(b)
            e_dst.append(int(c))
    base = np.stack([np.asarray(e_src, dtype=np.int64), np.asarray(e_dst, dtype=np.int64)]) \
        if e_src else np.zeros((2, 0), dtype=np.int64)
    uv_edges = np.concatenate(
        [base * nf + k for k in range(nf)], axis=1
    ) if base.size else base

    buckets = direction_buckets(
        array, ha_mid_rad, dec_rad,
        grid_deg=float(cfg_graph.get("direction_grid_deg", 5.0)),
    )
    f_lo = freqs_hz.reshape(-1)
    tile_f = np.array([
        f_lo[min(k * (len(f_lo) // nf) + (len(f_lo) // nf) // 2, len(f_lo) - 1)]
        for k in range(nf)
    ]) if nf > 0 else np.array([])

    enu = array.enu / max(np.abs(array.enu).max(), 1.0)
    ant_x = np.concatenate([enu, np.ones((a, 1))], axis=1).astype(np.float32)

    return StaticRelations(
        pairs=pairs, n_ant=a, n_freq_tiles=nf, shares=shares, uv_edges=uv_edges,
        buckets=buckets, uv_dist_m=np.hypot(uv[:, 0], uv[:, 1]),
        tile_freq_hz=tile_f, ant_x=ant_x,
    )


def _same_dir_edges(
    tiles_col: np.ndarray,       # [B, Nf, 2, t_c, f_c]
    validity_col: np.ndarray,    # [B, Nf]
    static: StaticRelations,
    integration_s: float,
    tol_hz: float,
    cap: int,
    max_gated: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """Relation B edges + per-node best direction bucket (-1 = none).

    Measured rate: peak of the fringe-rate power profile (tile power summed
    over the delay axis), gated on prominence and on being off-centre.
    Matched against bucket rate predictions scaled to the tile frequency.
    """
    n_b, nf, _, t_c, f_c = tiles_col.shape
    n_nodes = n_b * nf
    fr_bins = np.fft.fftshift(np.fft.fftfreq(t_c, d=integration_s))
    power = (10.0 ** tiles_col[:, :, 0]) ** 2                    # [B, Nf, t_c, f_c]
    prof = power.sum(-1)                                         # [B, Nf, t_c]
    peak = prof.argmax(-1)
    pmax = prof.max(-1)
    pmed = np.median(prof, axis=-1) + 1e-30
    gate = (peak != t_c // 2) & (pmax / pmed > 4.0) & (validity_col > 0.3)
    rate_hat = fr_bins[peak]                                     # [B, Nf]

    dir_bucket = np.full(n_nodes, -1, dtype=np.int64)
    idx = np.argwhere(gate)
    if idx.shape[0] < 2:
        return np.zeros((2, 0), dtype=np.int64), dir_bucket
    if idx.shape[0] > max_gated:
        order = np.argsort(-(pmax / pmed)[gate])[:max_gated]
        idx = idx[order]

    rates_k = static.buckets.rates_hz                            # [K, B] at bucket freq
    nu0 = static.buckets.freq_hz
    edges: set[tuple[int, int]] = set()
    deg = np.zeros(n_nodes, dtype=np.int64)

    # group gated nodes by freq tile (rate predictions scale with tile freq)
    for nf_i in range(nf):
        sub = idx[idx[:, 1] == nf_i]
        if sub.shape[0] < 2:
            continue
        bs = sub[:, 0]
        nodes = bs * nf + nf_i
        scale = static.tile_freq_hz[nf_i] / nu0 if static.tile_freq_hz.size else 1.0
        # predicted rates alias into the per-dump Nyquist window, same as the
        # measured FFT peak: compare on the aliasing circle
        fs = 1.0 / integration_s
        pred = rates_k[:, bs] * scale
        delta = np.abs(((pred - rate_hat[bs, nf_i][None, :]) + fs / 2) % fs - fs / 2)
        resid = delta                                                # [K, n]
        match = resid < tol_hz
        best = np.where(match.any(0), np.where(match, resid, np.inf).argmin(0), -1)
        dir_bucket[nodes] = best
        for k in np.unique(best[best >= 0]):
            members = np.flatnonzero(best == k)
            if members.size < 2:
                continue
            members = members[np.argsort(resid[k, members])][:32]
            mn = nodes[members]
            for ai in range(mn.size):
                for bi in range(ai + 1, mn.size):
                    u, v = int(mn[ai]), int(mn[bi])
                    if deg[u] >= cap or deg[v] >= cap:
                        continue
                    if (u, v) not in edges:
                        edges.add((u, v))
                        edges.add((v, u))
                        deg[u] += 1
                        deg[v] += 1
    if not edges:
        return np.zeros((2, 0), dtype=np.int64), dir_bucket
    e = np.array(sorted(edges), dtype=np.int64).T
    return e, dir_bucket


def build_column_graph(
    static: StaticRelations,
    tiles_col: torch.Tensor,     # [B, Nf, 2, t_c, f_c]
    phys_col: torch.Tensor,      # [B, Nf, 14]
    validity_col: torch.Tensor,  # [B, Nf]
    integration_s: float,
    cfg_graph: dict,
) -> HeteroData:
    n_b, nf = tiles_col.shape[0], tiles_col.shape[1]
    n_nodes = n_b * nf
    data = HeteroData()
    data["bl"].tile = tiles_col.reshape(n_nodes, *tiles_col.shape[2:]).float()
    data["bl"].phys = phys_col.reshape(n_nodes, -1).float()
    data["bl"].validity = validity_col.reshape(n_nodes).float()
    data["bl"].num_nodes = n_nodes
    data["antenna"].x = torch.as_tensor(static.ant_x)
    data["antenna"].num_nodes = static.n_ant

    same_dir, dir_bucket = _same_dir_edges(
        tiles_col.detach().cpu().numpy(), validity_col.detach().cpu().numpy(),
        static, integration_s,
        tol_hz=float(cfg_graph.get("direction_tol_hz", 0.008)),
        cap=int(cfg_graph.get("degree_cap", 16)),
    )
    data["bl"].dir_bucket = torch.as_tensor(dir_bucket)
    data[REL_SAME_DIR].edge_index = torch.as_tensor(same_dir, dtype=torch.long)
    data[REL_SAME_UV].edge_index = torch.as_tensor(static.uv_edges, dtype=torch.long)
    data[REL_SHARES].edge_index = torch.as_tensor(static.shares, dtype=torch.long)
    data[REL_REV].edge_index = torch.as_tensor(static.shares[::-1].copy(), dtype=torch.long)
    return data
