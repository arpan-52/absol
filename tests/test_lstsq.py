"""Spec test 6: lstsq antenna aggregation recovers a hub-coupled antenna set.

This is the regression test for the whole premise: RFI amplitude factorizes
over baselines as a_i * a_j, so per-antenna log-couplings are recoverable
from per-baseline scores by least squares. Run on the real 30-antenna GMRT
layout (the feasibility/ROC demo).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from absol.features import lstsq_antenna_scores
from absol.geometry import Array

REPO = Path(__file__).resolve().parents[1]


def _auc(scores: np.ndarray, truth: np.ndarray) -> float:
    pos, neg = scores[truth], scores[~truth]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    cmp = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    return float(cmp)


def test_6_hub_recovery_on_gmrt_layout():
    array = Array.from_yaml(REPO / "configs" / "array_gmrt.yaml")
    assert array.n_ant == 30
    pairs = array.baselines()
    rng = np.random.default_rng(11)
    aucs = []
    for trial in range(10):
        truth = np.zeros(array.n_ant, dtype=bool)
        truth[rng.choice(array.n_ant, size=5, replace=False)] = True
        a = np.where(truth, 10.0 ** (-rng.random(array.n_ant) * 1.0), 1e-3)
        v = a[pairs[:, 0]] * a[pairs[:, 1]]
        v = v * np.exp(0.3 * rng.standard_normal(v.size))      # measurement noise
        scores = lstsq_antenna_scores(v, pairs, array.n_ant)
        aucs.append(_auc(scores, truth))
    assert np.mean(aucs) > 0.95, f"mean AUC {np.mean(aucs):.3f}"


def test_6b_full_sim_hub_recovery(tiny_array, tiny_sim_cfg):
    """End-to-end: simulate a subset-coupled scene, score baselines by raw
    excess power, recover the coupled antennas."""
    import torch

    from absol.simulator.scenes import SceneGenerator

    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in tiny_sim_cfg.items()}
    cfg["rfi"] = dict(cfg["rfi"],
                      mechanisms={"ground_narrowband": 1.0},
                      strength_sigma={"dist": "loguniform", "min": 50, "max": 100},
                      coupling_decades=1)
    cfg["antenna_dropout"] = {"min_frac": 0.0, "max_frac": 0.0}
    cfg["flag_augmentation"] = {"scattered_p": 0, "contiguous_p": 0, "block_p": 0}
    from absol.features import mad_normalize, savgol_residual

    aucs = []
    for seed in range(6):
        gen = SceneGenerator(cfg, tiny_array, seed=seed)
        scene = gen.sample(stage=1)
        truth = scene.truth_antennas.numpy()
        if truth.sum() in (0, tiny_array.n_ant):
            continue
        # RFI amplitude statistic that factorizes as a_i a_j: top-quantile of
        # the robust-normalized residual, with the common noise floor removed
        res = savgol_residual(scene.vis, scene.weights_in, 11)
        vn, _ = mad_normalize(res, scene.weights_in)
        n_b = vn.shape[0]
        q = torch.quantile(vn.abs().reshape(n_b, -1), 0.999, dim=1).numpy()
        v = np.clip(q - np.median(q), 1e-3, None)
        scores = lstsq_antenna_scores(v, scene.array.baselines(), scene.array.n_ant)
        aucs.append(_auc(scores, truth))
    assert len(aucs) >= 2
    assert np.mean(aucs) > 0.8, f"sim hub AUC {np.mean(aucs):.3f}"
