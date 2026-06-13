"""Geometry sanity: uvw rotation, fringe-rate zero at phase centre, buckets."""
from __future__ import annotations

import numpy as np

from absol.geometry import (
    azel_to_hadec,
    direction_buckets,
    fringe_rate,
    hadec_to_azel,
    uvw,
)


def test_phase_centre_fringe_rate_zero(tiny_array):
    ha, dec = 0.21, np.deg2rad(33.0)
    r = fringe_rate(tiny_array, ha, dec, 650e6, ha, dec)
    assert np.allclose(r, 0.0, atol=1e-12)


def test_uvw_baseline_lengths_preserved(tiny_array):
    """|uvw| must equal the physical baseline length for any (ha, dec)."""
    d = tiny_array.baseline_xyz()
    for ha, dec in [(0.0, 0.5), (-0.7, -0.2), (1.0, 1.1)]:
        u = uvw(tiny_array, ha, dec)
        assert np.allclose(np.linalg.norm(u, axis=1), np.linalg.norm(d, axis=1),
                           rtol=1e-12)


def test_azel_hadec_roundtrip(tiny_array):
    lat = tiny_array.lat_rad
    for az, el in [(10, 5), (123, 40), (260, 70), (350, 12)]:
        ha, dec = azel_to_hadec(np.deg2rad(az), np.deg2rad(el), lat)
        az2, el2 = hadec_to_azel(ha, dec, lat)
        assert abs(np.rad2deg(el2) - el) < 1e-9
        assert abs((np.rad2deg(az2) - az + 180) % 360 - 180) < 1e-9


def test_direction_buckets_cached_and_sane(tiny_array):
    b1 = direction_buckets(tiny_array, 0.1, 0.4, grid_deg=10.0)
    b2 = direction_buckets(tiny_array, 0.1, 0.4, grid_deg=10.0)
    assert b1 is b2                       # cache hit
    n_b = tiny_array.baselines().shape[0]
    assert b1.rates_hz.shape == (b1.az_deg.size, n_b)
    assert (b1.el_deg > 0).all()
    assert np.isfinite(b1.rates_hz).all()
