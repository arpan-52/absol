"""Smooth random complex antenna gains.

g_k(t) = (1 + amp_rms * x_k(t)) * exp(i * phase_rms_rad * y_k(t)) with x, y
unit-variance Gaussian processes realized as Gaussian-filtered white noise of
correlation timescale ``timescale_s``. Frequency-independent in v0.

v0 SIMPLIFICATION (documented per spec): gains are applied multiplicatively
to the TOTAL visibility (sky + noise + RFI). In reality additive RFI enters
post-beam but pre-gain, so it should also see the gains, while noise should
not; the error committed here is second order for |g|-1 ~ 5%. Revisit later.
"""
from __future__ import annotations

import numpy as np
import torch


def smooth_noise(
    n_series: int, n_time: int, timescale_dumps: float, rng: np.random.Generator
) -> np.ndarray:
    """[n_series, n_time] unit-variance Gaussian, correlation ~ timescale_dumps."""
    pad = max(int(4 * timescale_dumps), 4)
    x = rng.standard_normal((n_series, n_time + 2 * pad))
    t = np.arange(-pad, pad + 1, dtype=np.float64)
    sig = max(timescale_dumps, 1e-3)
    k = np.exp(-0.5 * (t / sig) ** 2)
    k /= np.sqrt((k**2).sum())          # preserve unit variance after convolution
    out = np.stack([np.convolve(row, k, mode="same") for row in x], axis=0)
    return out[:, pad:pad + n_time]


def antenna_gains(
    n_ant: int,
    n_time: int,
    amp_rms: float,
    phase_rms_deg: float,
    timescale_s: float,
    integration_s: float,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Complex gains g[A, T] (complex64)."""
    ts = timescale_s / integration_s
    amp = 1.0 + amp_rms * smooth_noise(n_ant, n_time, ts, rng)
    ph = np.deg2rad(phase_rms_deg) * smooth_noise(n_ant, n_time, ts, rng)
    g = amp * np.exp(1j * ph)
    return torch.as_tensor(g, dtype=torch.complex64, device=device)


def apply_gains(vis: torch.Tensor, gains: torch.Tensor, pairs: np.ndarray) -> torch.Tensor:
    """vis[B,T,F,P] *= g_i(t) g_j(t)^* for baseline (i, j)."""
    gi = gains[pairs[:, 0]]            # [B, T]
    gj = gains[pairs[:, 1]].conj()
    return vis * (gi * gj)[:, :, None, None]
