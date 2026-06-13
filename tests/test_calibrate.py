"""Spec test 9: temperature scaling improves ECE on miscalibrated logits."""
from __future__ import annotations

import numpy as np
import torch

from absol.training.calibrate import ece, fit_temperature


def test_9_temperature_scaling_improves_ece():
    rng = np.random.default_rng(0)
    n = 20000
    z = rng.normal(0, 1.5, n)                       # true logits
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(np.float32)
    logits = torch.as_tensor(3.0 * z, dtype=torch.float32)   # overconfident x3
    labels = torch.as_tensor(y)

    t_fit = fit_temperature(logits, labels)
    assert 2.0 < t_fit < 4.5, f"recovered T {t_fit}"

    p_raw = torch.sigmoid(logits).numpy()
    p_cal = torch.sigmoid(logits / t_fit).numpy()
    e0, e1 = ece(p_raw, y), ece(p_cal, y)
    assert e1 < e0, f"ECE did not improve: {e0:.4f} -> {e1:.4f}"
    assert e1 < 0.02
