"""Point-source sky model and visibility prediction (fringe-stopped frame).

Physics: for a point source at direction cosines (l, m) relative to the phase
centre with flux S(nu) [Jy], the fringe-stopped visibility on baseline ij is

    V_ij(t, nu) = S(nu) * exp{ 2 pi i (nu / c) [ u_ij(t) l + v_ij(t) m ] }

with (u, v) in metres from `geometry.uvw` evolving with hour angle. A source
at the phase centre (l = m = 0) gives a constant visibility (unit-tested).
The w-(n-1) term is negligible for the ~1 deg field considered in v0 and is
omitted; document any change here in the data contract.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from absol.geometry import C_LIGHT, Array, uvw


@dataclass
class SkySources:
    l: np.ndarray            # [S] direction cosine
    m: np.ndarray            # [S]
    flux_jy: np.ndarray      # [S] at band centre
    spec_index: np.ndarray   # [S] S(nu) = S0 (nu/nu0)^alpha


def sample_sky(sky_cfg: dict, rng: np.random.Generator) -> SkySources:
    """Draw point sources: power-law fluxes, uniform disc positions."""
    lo, hi = int(sky_cfg["n_sources"]["min"]), int(sky_cfg["n_sources"]["max"])
    n = int(rng.integers(lo, hi + 1))
    f = sky_cfg["flux_jy"]
    alpha, smin, smax = float(f["alpha"]), float(f["min"]), float(f["max"])
    # Inverse-CDF sampling of p(S) ~ S^alpha on [smin, smax] (alpha != -1).
    u = rng.random(n)
    a1 = alpha + 1.0
    flux = (smin**a1 + u * (smax**a1 - smin**a1)) ** (1.0 / a1)
    r_max = np.deg2rad(float(sky_cfg["field_radius_deg"]))
    r = r_max * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    return SkySources(
        l=r * np.cos(th), m=r * np.sin(th), flux_jy=flux,
        spec_index=rng.normal(-0.7, 0.3, size=n),
    )


def sky_visibilities(
    array: Array,
    sources: SkySources,
    ha_rad: np.ndarray,        # [T]
    dec_rad: float,
    freqs_hz: np.ndarray,      # [F]
    device: torch.device,
) -> torch.Tensor:
    """Noiseless sky visibilities, complex64 [B, T, F] (identical in both pols).

    Computed source-by-source to bound memory at full GMRT scale.
    """
    uvw_t = uvw(array, ha_rad, dec_rad)                    # [T, B, 3]
    u = torch.as_tensor(uvw_t[..., 0].T, dtype=torch.float64, device=device)  # [B, T]
    v = torch.as_tensor(uvw_t[..., 1].T, dtype=torch.float64, device=device)
    nu = torch.as_tensor(freqs_hz, dtype=torch.float64, device=device)        # [F]
    nu0 = float(freqs_hz.mean())
    out = torch.zeros((u.shape[0], u.shape[1], nu.shape[0]), dtype=torch.complex64, device=device)
    for s in range(sources.l.size):
        # phase[B,T,F] = 2 pi (nu/c) (u l + v m)
        geom = (u * sources.l[s] + v * sources.m[s])[:, :, None] * (nu / C_LIGHT)
        spec = float(sources.flux_jy[s]) * (nu / nu0) ** float(sources.spec_index[s])
        out += (spec.to(torch.complex128)
                * torch.exp(2j * torch.pi * geom.to(torch.complex128))).to(torch.complex64)
    return out
