"""Feature extraction: residual step, chunking, normalization, physics features.

All functions respect pre-existing flags via ``weights`` (0 = flagged):
flagged samples are excluded from the Savitzky-Golay fit, the MAD scale,
and zero-weighted inside FFT tiles; chunk validity tracks the unflagged
fraction. This module is shared verbatim by training and MS inference.

Physics feature vector (float32[14], FIXED order - data contract):
  0  spectral-kurtosis excess SK-1 (generalized SK over chunk powers; 0 for Gaussian)
  1  off-peak fringe power fraction (power outside central 3x3 of the delay-fringe tile)
  2  log10(max/median |V|) within chunk
  3  occupancy fraction of samples with |V| > 3 (robust-sigma units)
  4  log10 pol power ratio P0/P1 (clipped +-2)
  5  GridFlag residual z-score of chunk power vs same-(uv-bin, freq-tile) population
  6  closure-phase circular scatter (mean over sampled triangles through this baseline)
  7  validity fraction (unflagged fraction of the chunk)
  8  zero-fringe excess: central tile pixel z-score vs tile distribution
  9  chunk time index, normalized to [0, 1]
  10 chunk freq index, normalized to [0, 1]
  11 baseline |uv| length / max
  12 reserved (0)
  13 reserved (0)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import savgol_coeffs

N_PHYS = 14

# Savitzky-Golay FIR kernels are fixed per (window, order); cache the tiny
# coefficient vectors so the residual step stays a pure-GPU conv1d.
_SG_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _sg_kernel(window: int, order: int) -> np.ndarray:
    key = (window, order)
    if key not in _SG_CACHE:
        _SG_CACHE[key] = savgol_coeffs(window, order).astype(np.float32)
    return _SG_CACHE[key]


# --------------------------------------------------------------------------
def savgol_residual(
    vis: torch.Tensor, weights: torch.Tensor, window: int, order: int = 3
) -> torch.Tensor:
    """Subtract a smooth time component per (baseline, channel, pol).

    GPU-native: the Savitzky-Golay smoother is the fixed FIR kernel
    ``savgol_coeffs(window, order)`` applied along time with ``F.conv1d``
    (replicate-padded edges), so no data ever leaves the device. Flagged
    samples are replaced by the per-(b, f, p) unflagged mean before filtering
    so flags cannot drag the fit. If the scan is too short for the window,
    the weighted time mean is subtracted instead.
    """
    b, t, f, p = vis.shape
    w = weights[..., None].expand_as(vis.real)
    wsum = w.sum(1).clamp(min=1e-9)
    mean = (vis * w).sum(1) / wsum                                  # [B, F, P]
    if t < 7 or window < 5:
        return vis - mean[:, None]
    win = min(window if window % 2 == 1 else window + 1, t if t % 2 == 1 else t - 1)
    ordr = min(order, win - 2)
    filled = torch.where(w > 0, vis, mean[:, None].expand_as(vis))

    k = torch.as_tensor(_sg_kernel(win, ordr), device=vis.device).reshape(1, 1, -1)
    pad = win // 2
    x = filled.permute(0, 2, 3, 1).reshape(-1, 1, t)               # [B*F*P, 1, T]
    xr = F.pad(x.real, (pad, pad), mode="replicate")
    xi = F.pad(x.imag, (pad, pad), mode="replicate")
    sm = (F.conv1d(xr, k) + 1j * F.conv1d(xi, k)).to(torch.complex64)
    sm = sm.reshape(b, f, p, t).permute(0, 3, 1, 2)
    return vis - sm


def mad_normalize(
    vis: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Divide by 1.4826 * MAD(|residual|) per baseline & pol over the scan.

    Returns (normalized vis [B,T,F,P], sigma [B,P]). All downstream
    thresholds (occupancy > 3, simulator strength grid) are in these units.
    """
    b, t, f, p = vis.shape
    x = vis.abs()
    x = torch.where(weights[..., None] > 0, x, torch.full_like(x, float("nan")))
    flat = x.permute(0, 3, 1, 2).reshape(b, p, t * f)
    med = flat.nanmedian(dim=2).values                              # [B, P]
    mad = (flat - med[..., None]).abs().nanmedian(dim=2).values
    sigma = 1.4826 * mad
    sigma = torch.where(torch.isfinite(sigma) & (sigma > 1e-12), sigma, torch.ones_like(sigma))
    return vis / sigma[:, None, None, :], sigma


def chunk_grid(
    vis: torch.Tensor, weights: torch.Tensor, t_c: int, f_c: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tile the (T, F) plane into (t_c, f_c) chunks, zero-padding edges.

    Returns (chunks [B,Nt,Nf,t_c,f_c,P], wchunks [B,Nt,Nf,t_c,f_c],
    validity [B,Nt,Nf]). Padded samples carry weight 0 so they count
    against the validity fraction (data contract).
    """
    b, t, f, p = vis.shape
    nt, nf = -(-t // t_c), -(-f // f_c)
    vp = torch.zeros((b, nt * t_c, nf * f_c, p), dtype=vis.dtype, device=vis.device)
    wp = torch.zeros((b, nt * t_c, nf * f_c), dtype=weights.dtype, device=vis.device)
    vp[:, :t, :f] = vis
    wp[:, :t, :f] = weights
    chunks = vp.reshape(b, nt, t_c, nf, f_c, p).permute(0, 1, 3, 2, 4, 5).contiguous()
    wch = wp.reshape(b, nt, t_c, nf, f_c).permute(0, 1, 3, 2, 4).contiguous()
    return chunks, wch, wch.mean(dim=(3, 4))


def delay_fringe_tiles(chunks: torch.Tensor, wchunks: torch.Tensor) -> torch.Tensor:
    """2D FFT over (t, f) of the pol-mean, flag-zeroed chunk.

    Returns [B,Nt,Nf,2,t_c,f_c]: channel 0 = log10(|X|+1e-6), channel 1 =
    phase. fftshifted, so fringe-rate 0 / delay 0 sits at the tile centre.
    """
    x = chunks.mean(dim=-1) * wchunks
    xf = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    return torch.stack(
        [torch.log10(xf.abs() + 1e-6), torch.angle(xf)], dim=3
    ).to(torch.float32)


# --------------------------------------------------------------------------
def _pair_index_map(pairs: np.ndarray, n_ant: int) -> np.ndarray:
    m = np.full((n_ant, n_ant), -1, dtype=np.int64)
    for b, (i, j) in enumerate(pairs):
        m[i, j] = b
        m[j, i] = b
    return m


def _closure_scatter(
    chunks: torch.Tensor, wchunks: torch.Tensor, pairs: np.ndarray, n_ant: int,
    n_tri: int = 8, n_samp: int = 128, seed: int = 0,
) -> torch.Tensor:
    """Feature 6: circular std of closure phase per (baseline, chunk).

    Subsamples n_tri third antennas per baseline and n_samp positions per
    chunk (documented approximation to bound memory at full GMRT scale).
    """
    dev = chunks.device
    b, nt, nf, t_c, f_c, p = chunks.shape
    nc = nt * nf
    v = chunks.mean(-1).reshape(b, nc, t_c * f_c)
    w = (wchunks.reshape(b, nc, t_c * f_c) > 0).float()
    rng = np.random.default_rng(seed)
    samp = torch.as_tensor(
        rng.choice(t_c * f_c, size=min(n_samp, t_c * f_c), replace=False), device=dev
    )
    v, w = v[:, :, samp], w[:, :, samp]
    pmap = _pair_index_map(pairs, n_ant)
    acc = torch.zeros((b, nc), device=dev)
    cnt = torch.zeros((b, nc), device=dev)
    n_tri = min(n_tri, max(n_ant - 2, 0))
    pi, pj = pairs[:, 0], pairs[:, 1]
    for _ in range(n_tri):
        # vectorized: draw a third antenna per baseline, redraw only collisions
        ks = rng.integers(0, n_ant, size=b)
        bad = (ks == pi) | (ks == pj)
        while bad.any():
            ks = np.where(bad, rng.integers(0, n_ant, size=b), ks)
            bad = (ks == pi) | (ks == pj)
        i, j = pi, pj
        b_jk, b_ik = pmap[j, ks], pmap[i, ks]
        c_jk = torch.where(
            torch.as_tensor(j < ks, device=dev)[:, None, None], v[b_jk], v[b_jk].conj()
        )
        c_ik = torch.where(
            torch.as_tensor(i < ks, device=dev)[:, None, None], v[b_ik], v[b_ik].conj()
        )
        bisp = v * c_jk * c_ik.conj()
        ww = w * w[b_jk] * w[b_ik]
        z = bisp / bisp.abs().clamp(min=1e-30)
        r = (z * ww).sum(-1).abs() / ww.sum(-1).clamp(min=1)
        scat = torch.sqrt(-2.0 * torch.log(r.clamp(min=1e-6, max=1.0)))
        ok = ww.sum(-1) > 8
        acc += torch.where(ok, scat, torch.zeros_like(scat))
        cnt += ok.float()
    return acc / cnt.clamp(min=1)


def physics_features(
    chunks: torch.Tensor,
    wchunks: torch.Tensor,
    tiles: torch.Tensor,
    validity: torch.Tensor,
    pairs: np.ndarray,
    n_ant: int,
    uv_dist_m: np.ndarray,
    seed: int = 0,
) -> torch.Tensor:
    """The 14 physics features, [B, Nt, Nf, 14] (ordering documented above)."""
    dev = chunks.device
    b, nt, nf, t_c, f_c, p = chunks.shape
    nc = nt * nf
    m_samp = t_c * f_c
    feats = torch.zeros((b, nc, N_PHYS), dtype=torch.float32, device=dev)

    amp = chunks.abs().mean(-1).reshape(b, nc, m_samp)              # pol-mean |V|
    w = (wchunks.reshape(b, nc, m_samp) > 0).float()
    nval = w.sum(-1)

    # 0: generalized spectral kurtosis excess, per pol (power of one pol of a
    # complex-Gaussian visibility is exponential => E[SK] = 1), then pol-mean.
    xp = (chunks.abs() ** 2).reshape(b, nc, m_samp, p) * w[..., None]
    s1p, s2p = xp.sum(2), (xp**2).sum(2)
    mp_ = nval.clamp(min=2)[..., None]
    skp = ((mp_ + 1) / (mp_ - 1)) * (mp_ * s2p / s1p.clamp(min=1e-30) ** 2 - 1)
    sk = skp.mean(-1)
    feats[:, :, 0] = torch.where(nval > 8, sk - 1.0, torch.zeros_like(sk)).clamp(-5, 20)
    x = (chunks.abs() ** 2).mean(-1).reshape(b, nc, m_samp) * w   # pol-mean power (feat 5)

    # 1, 8: delay-fringe tile statistics
    power = (10.0 ** tiles[:, :, :, 0]) ** 2
    power = power.reshape(b, nc, t_c, f_c)
    ct, cf = t_c // 2, f_c // 2
    central = power[:, :, max(ct - 1, 0):ct + 2, max(cf - 1, 0):cf + 2].sum((-2, -1))
    tot = power.sum((-2, -1)).clamp(min=1e-30)
    feats[:, :, 1] = (1.0 - central / tot).clamp(0, 1)
    logm = tiles[:, :, :, 0].reshape(b, nc, t_c * f_c)
    mu, sd = logm.mean(-1), logm.std(-1).clamp(min=1e-6)
    feats[:, :, 8] = ((logm[:, :, (ct * f_c + cf)] - mu) / sd).clamp(-10, 30)

    # 2: log10 max/median; 3: occupancy > 3 sigma
    a_nan = torch.where(w > 0, amp, torch.full_like(amp, float("nan")))
    med = a_nan.nanmedian(-1).values.clamp(min=1e-12)
    mx = torch.nan_to_num(a_nan, nan=0.0).amax(-1)
    feats[:, :, 2] = torch.log10((mx / med).clamp(min=1e-3, max=1e4))
    feats[:, :, 3] = ((amp > 3.0).float() * w).sum(-1) / nval.clamp(min=1)

    # 4: pol power ratio
    if p >= 2:
        pw = (chunks.abs() ** 2).reshape(b, nc, m_samp, p)
        p0 = (pw[..., 0] * w).sum(-1) / nval.clamp(min=1)
        p1 = (pw[..., 1] * w).sum(-1) / nval.clamp(min=1)
        feats[:, :, 4] = torch.log10(p0.clamp(min=1e-30) / p1.clamp(min=1e-30)).clamp(-2, 2)

    # 5: GridFlag z-score vs same-(uv-bin, freq-tile) population
    pw_mean = (x.sum(-1) / nval.clamp(min=1)).clamp(min=1e-30).log10()   # [B, Nc]
    uv = torch.as_tensor(uv_dist_m, dtype=torch.float32, device=dev)
    edges = torch.logspace(
        np.log10(max(float(uv.min()), 1.0)), np.log10(float(uv.max()) + 1.0), 16, device=dev
    )
    uv_bin = torch.bucketize(uv, edges)                                   # [B]
    f_idx = torch.arange(nc, device=dev) % nf
    gid = (uv_bin[:, None] * nf + f_idx[None, :]).reshape(-1)
    val = pw_mean.reshape(-1)
    ok = (validity.reshape(-1) > 0.05).float()
    ng = int(gid.max()) + 1
    cnt = torch.zeros(ng, device=dev).index_add_(0, gid, ok)
    s = torch.zeros(ng, device=dev).index_add_(0, gid, val * ok)
    s2 = torch.zeros(ng, device=dev).index_add_(0, gid, val**2 * ok)
    gmu = s / cnt.clamp(min=1)
    gsd = (s2 / cnt.clamp(min=1) - gmu**2).clamp(min=1e-12).sqrt()
    z = (val - gmu[gid]) / gsd[gid]
    feats[:, :, 5] = torch.where(cnt[gid] > 4, z, torch.zeros_like(z)).reshape(b, nc).clamp(-10, 10)

    # 6: closure-phase scatter
    feats[:, :, 6] = _closure_scatter(chunks, wchunks, pairs, n_ant, seed=seed)

    # 7, 9, 10, 11: validity, position, baseline length
    feats[:, :, 7] = validity.reshape(b, nc)
    t_idx = torch.arange(nc, device=dev) // nf
    feats[:, :, 9] = (t_idx / max(nt - 1, 1)).float()[None, :]
    feats[:, :, 10] = (f_idx / max(nf - 1, 1)).float()[None, :]
    feats[:, :, 11] = (uv / uv.max().clamp(min=1e-9))[:, None]

    feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    feats = feats * (validity.reshape(b, nc, 1) > 0).float()
    return feats.reshape(b, nt, nf, N_PHYS)


# --------------------------------------------------------------------------
def lstsq_antenna_scores(
    values: np.ndarray, pairs: np.ndarray, n_ant: int, ridge: float = 1e-3
) -> np.ndarray:
    """Feasibility baseline: solve log v_ij ~ x_i + x_j for per-antenna x.

    With RFI amplitude factorizing as a_i a_j over baselines, the recovered
    x rank antennas by log-coupling. This is the regression test for the
    whole antenna-aggregation premise (spec test 6).
    """
    y = np.log(np.clip(np.asarray(values, dtype=np.float64), 1e-12, None))
    a_inc = np.zeros((pairs.shape[0], n_ant))
    a_inc[np.arange(pairs.shape[0]), pairs[:, 0]] = 1.0
    a_inc[np.arange(pairs.shape[0]), pairs[:, 1]] = 1.0
    a_full = np.vstack([a_inc, np.sqrt(ridge) * np.eye(n_ant)])
    y_full = np.concatenate([y, np.zeros(n_ant)])
    x, *_ = np.linalg.lstsq(a_full, y_full, rcond=None)
    return x
