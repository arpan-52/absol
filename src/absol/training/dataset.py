"""On-the-fly IterableDataset over simulated scenes.

Each item is a fully prepared scene: column hetero-graphs + training labels.
Scenes are never stored (1-2 GB each at full GMRT scale); generation happens
in dataloader workers with per-worker seeding.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from absol.geometry import Array
from absol.pipeline import Prepared, prepare
from absol.simulator.scenes import Scene, SceneGenerator


def chunk_truth_fractions(
    mask: torch.Tensor, weights: torch.Tensor, t_c: int, f_c: int
) -> torch.Tensor:
    """Fraction of VALID samples per chunk that are True, [B, Nt, Nf]."""
    b, t, f = mask.shape
    nt, nf = -(-t // t_c), -(-f // f_c)
    mp = torch.zeros((b, nt * t_c, nf * f_c), dtype=torch.float32, device=mask.device)
    wp = torch.zeros_like(mp)
    mp[:, :t, :f] = mask.float() * (weights > 0)
    wp[:, :t, :f] = (weights > 0).float()
    mc = mp.reshape(b, nt, t_c, nf, f_c).permute(0, 1, 3, 2, 4).sum((-2, -1))
    wc = wp.reshape(b, nt, t_c, nf, f_c).permute(0, 1, 3, 2, 4).sum((-2, -1))
    return mc / wc.clamp(min=1)


def scene_to_item(
    scene: Scene, sim_cfg: dict, model_cfg: dict, seed: int = 0,
    max_mask_chunks: int = 512,
) -> dict:
    prep: Prepared = prepare(
        scene.vis, scene.weights_in, scene.array, scene.ha_rad, scene.dec_rad,
        scene.freqs_hz, sim_cfg, model_cfg, seed=seed,
    )
    t_c, f_c = prep.t_c, prep.f_c
    cf = chunk_truth_fractions(scene.truth_mask, scene.weights_in, t_c, f_c)
    pf = chunk_truth_fractions(scene.protected_mask, scene.weights_in, t_c, f_c)
    b, nt, nf = cf.shape

    edge_target = (cf > 0.01).float()                       # [B, Nt, Nf]
    pnw = float(model_cfg["training"].get("protected_negative_weight", 2.0))
    edge_weight = torch.ones_like(edge_target)
    edge_weight = torch.where((pf > 0.01) & (edge_target < 0.5), pnw * edge_weight, edge_weight)
    edge_weight = edge_weight * (prep.validity > 0.05).float()

    # node-major flat layout to match model: index = (b * Nf + nf) * Nt + nt
    et = edge_target.permute(0, 2, 1).reshape(b * nf, nt)
    ew = edge_weight.permute(0, 2, 1).reshape(b * nf, nt)

    # sample-mask supervision on (capped) truth-positive chunks
    pos = torch.nonzero(et.reshape(-1) > 0.5).squeeze(-1)
    if pos.numel() > max_mask_chunks:
        pos = pos[torch.randperm(pos.numel())[:max_mask_chunks]]
    mask_target = torch.zeros((0, t_c, f_c))
    mask_weight = torch.zeros((0, t_c, f_c))
    if pos.numel() > 0:
        tm, _, _ = _chunked_bool(scene.truth_mask, scene.weights_in, t_c, f_c)
        wm = prep.wchunks                                    # [B, Nt, Nf, t_c, f_c]
        tm_flat = tm.permute(0, 2, 1, 3, 4).reshape(b * nf * nt, t_c, f_c)
        wm_flat = wm.permute(0, 2, 1, 3, 4).reshape(b * nf * nt, t_c, f_c)
        mask_target = tm_flat[pos].float()
        mask_weight = (wm_flat[pos] > 0).float()

    return {
        "columns": prep.columns,
        "edge_target": et,
        "edge_weight": ew,
        "ant_target": scene.truth_antennas.float()[:, None],   # [A, 1]
        "mask_index": pos,
        "mask_target": mask_target,
        "mask_weight": mask_weight,
        "meta": scene.meta,
        "validity": prep.validity,
    }


def _chunked_bool(mask: torch.Tensor, weights: torch.Tensor, t_c: int, f_c: int):
    b, t, f = mask.shape
    nt, nf = -(-t // t_c), -(-f // f_c)
    mp = torch.zeros((b, nt * t_c, nf * f_c), dtype=torch.bool, device=mask.device)
    mp[:, :t, :f] = mask
    mc = mp.reshape(b, nt, t_c, nf, f_c).permute(0, 1, 3, 2, 4)
    return mc, nt, nf


class SceneDataset(IterableDataset):
    """Infinite stream of prepared scenes for one curriculum stage."""

    def __init__(self, sim_cfg: dict, array: Array, model_cfg: dict,
                 stage: int, seed: int = 0, device: str = "cpu"):
        super().__init__()
        self.sim_cfg, self.array, self.model_cfg = sim_cfg, array, model_cfg
        self.stage, self.seed = stage, seed
        self.device = device

    def __iter__(self) -> Iterator[dict]:
        info = get_worker_info()
        wid = info.id if info is not None else 0
        # CUDA tensors are unsafe in forked workers: generate on GPU only in
        # the main process (num_workers=0); workers always fall back to CPU.
        dev = "cpu" if info is not None else self.device
        gen = SceneGenerator(self.sim_cfg, self.array, seed=self.seed, device=dev)
        gen.reseed(wid)
        item_rng = np.random.default_rng(self.seed + wid)
        while True:
            scene = gen.sample(self.stage)
            yield scene_to_item(
                scene, self.sim_cfg, self.model_cfg, seed=int(item_rng.integers(2**31))
            )
