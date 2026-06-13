"""Array geometry: ENU positions, uvw, fringe rates, direction buckets.

This module is the single source of truth for residual fringe rates. The
simulator (RFI phase generation), graph relation B (`same_dir`), and the
coincidence validator must all derive rates from here.

Conventions
-----------
- Antenna positions: ENU metres relative to the array reference antenna.
- Local equatorial frame (X, Y, Z): X toward (H=0, dec=0), Y toward east
  (H=-6h), Z toward the north celestial pole. For latitude ``lat``:
      X = -sin(lat) N + cos(lat) U,   Y = E,   Z = cos(lat) N + sin(lat) U
- uvw (metres) for hour angle H, declination dec (all angles radians):
      u =  sinH X + cosH Y
      v = -sindec cosH X + sindec sinH Y + cosdec Z
      w =  cosdec cosH X - cosdec sinH Y + sindec Z
- Geometric phase of a far-field source in direction s on baseline ij:
      phi = 2 pi (nu / c) w_s        [rad]
  In the fringe-stopped frame the tracked phase-centre w is subtracted, so
  the residual fringe rate of direction s at frequency nu is
      f_ij = (nu / c) d/dt [ w_s - w_pc ]      [Hz]
  with dH/dt = OMEGA_EARTH. The phase centre has f = 0 by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

C_LIGHT = 299_792_458.0          # m/s
OMEGA_EARTH = 7.292_115_855_3e-5  # sidereal rotation rate, rad/s


@dataclass(frozen=True)
class Array:
    """Fixed array geometry loaded from ``array_*.yaml``."""

    name: str
    names: tuple[str, ...]
    enu: np.ndarray                # [A, 3] float64, metres
    latitude_deg: float
    longitude_deg: float
    freq_start_hz: float
    freq_end_hz: float
    n_channels: int
    n_pol: int
    integration_s: float

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: float) -> Array:
        cfg = yaml.safe_load(Path(path).read_text())
        arr, band, obs = cfg["array"], cfg["band"], cfg["observation"]
        ants = arr["antennas"]
        enu = np.array([[a["e"], a["n"], a["u"]] for a in ants], dtype=np.float64)
        kw = dict(
            name=arr["name"],
            names=tuple(a["id"] for a in ants),
            enu=enu,
            latitude_deg=float(arr["latitude_deg"]),
            longitude_deg=float(arr["longitude_deg"]),
            freq_start_hz=float(band["freq_start_hz"]),
            freq_end_hz=float(band["freq_end_hz"]),
            n_channels=int(band["n_channels"]),
            n_pol=int(band["n_pol"]),
            integration_s=float(obs["integration_s"]),
        )
        kw.update(overrides)
        return cls(**kw)

    @property
    def n_ant(self) -> int:
        return len(self.names)

    @property
    def lat_rad(self) -> float:
        return float(np.deg2rad(self.latitude_deg))

    @property
    def freqs_hz(self) -> np.ndarray:
        """Channel centre frequencies [F]."""
        edges = np.linspace(self.freq_start_hz, self.freq_end_hz, self.n_channels + 1)
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def chan_width_hz(self) -> float:
        return (self.freq_end_hz - self.freq_start_hz) / self.n_channels

    def baselines(self) -> np.ndarray:
        """Antenna index pairs (i < j), shape [B, 2]."""
        a = self.n_ant
        i, j = np.triu_indices(a, k=1)
        return np.stack([i, j], axis=1)

    def subset(self, keep: np.ndarray) -> Array:
        """Sub-array of the antennas where ``keep`` (bool [A]) is True."""
        idx = np.flatnonzero(keep)
        return Array(
            name=self.name,
            names=tuple(self.names[k] for k in idx),
            enu=self.enu[idx],
            latitude_deg=self.latitude_deg,
            longitude_deg=self.longitude_deg,
            freq_start_hz=self.freq_start_hz,
            freq_end_hz=self.freq_end_hz,
            n_channels=self.n_channels,
            n_pol=self.n_pol,
            integration_s=self.integration_s,
        )

    def xyz(self) -> np.ndarray:
        """Antenna positions in the local equatorial frame, [A, 3] metres."""
        return enu_to_xyz(self.enu, self.lat_rad)

    def baseline_xyz(self) -> np.ndarray:
        """Baseline vectors xyz_j - xyz_i, [B, 3] metres."""
        xyz = self.xyz()
        pairs = self.baselines()
        return xyz[pairs[:, 1]] - xyz[pairs[:, 0]]


def enu_to_xyz(enu: np.ndarray, lat_rad: float) -> np.ndarray:
    """ENU [.., 3] -> local equatorial XYZ [.., 3] (metres)."""
    e, n, u = enu[..., 0], enu[..., 1], enu[..., 2]
    sl, cl = np.sin(lat_rad), np.cos(lat_rad)
    x = -sl * n + cl * u
    y = e
    z = cl * n + sl * u
    return np.stack([x, y, z], axis=-1)


def uvw(array: Array, ha_rad: float | np.ndarray, dec_rad: float) -> np.ndarray:
    """Baseline uvw in metres for hour angle(s) ``ha_rad``.

    Returns [B, 3] for scalar ha, [T, B, 3] for a vector of hour angles.
    """
    d = array.baseline_xyz()                       # [B, 3]
    ha = np.atleast_1d(np.asarray(ha_rad, dtype=np.float64))
    sh, ch = np.sin(ha)[:, None], np.cos(ha)[:, None]
    sd, cd = np.sin(dec_rad), np.cos(dec_rad)
    x, y, z = d[:, 0][None], d[:, 1][None], d[:, 2][None]
    u = sh * x + ch * y
    v = -sd * ch * x + sd * sh * y + cd * z
    w = cd * ch * x - cd * sh * y + sd * z
    out = np.stack([u, v, w], axis=-1)             # [T, B, 3]
    return out[0] if np.isscalar(ha_rad) or np.ndim(ha_rad) == 0 else out


def w_metres(array: Array, ha_rad: float | np.ndarray, dec_rad: float) -> np.ndarray:
    """w coordinate only: [B] or [T, B] metres."""
    return uvw(array, ha_rad, dec_rad)[..., 2]


def _dw_dt(array: Array, ha_rad: float, dec_rad: float) -> np.ndarray:
    """d(w)/dt per baseline [B], metres/second (dH/dt = OMEGA_EARTH)."""
    d = array.baseline_xyz()
    sh, ch = np.sin(ha_rad), np.cos(ha_rad)
    cd = np.cos(dec_rad)
    dw_dh = -cd * sh * d[:, 0] - cd * ch * d[:, 1]
    return dw_dh * OMEGA_EARTH


def fringe_rate(
    array: Array,
    ha_rad: float,
    dec_rad: float,
    freq_hz: float,
    ha_pc_rad: float | None = None,
    dec_pc_rad: float | None = None,
) -> np.ndarray:
    """Residual fringe rate [B] in Hz of direction (ha, dec) at ``freq_hz``.

    The tracking rate of the phase centre (ha_pc, dec_pc) is subtracted, so
    the phase centre itself has fringe rate 0 by construction. With no phase
    centre given, the un-stopped geometric rate is returned.
    """
    rate = _dw_dt(array, ha_rad, dec_rad)
    if ha_pc_rad is not None and dec_pc_rad is not None:
        rate = rate - _dw_dt(array, ha_pc_rad, dec_pc_rad)
    return rate * freq_hz / C_LIGHT


def azel_to_hadec(az_rad: float, el_rad: float, lat_rad: float) -> tuple[float, float]:
    """Topocentric (az from N through E, el) -> (hour angle, declination), radians."""
    sl, cl = np.sin(lat_rad), np.cos(lat_rad)
    sd = sl * np.sin(el_rad) + cl * np.cos(el_rad) * np.cos(az_rad)
    sd = np.clip(sd, -1.0, 1.0)
    dec = np.arcsin(sd)
    cd = np.cos(dec)
    if abs(cd) < 1e-12:
        return 0.0, float(dec)
    sin_h = -np.sin(az_rad) * np.cos(el_rad) / cd
    cos_h = (np.sin(el_rad) - sl * sd) / (cl * cd)
    return float(np.arctan2(sin_h, cos_h)), float(dec)


def azel_to_hadec_array(
    az_rad: np.ndarray, el_rad: np.ndarray, lat_rad: float
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized :func:`azel_to_hadec` over arrays -> (ha[K], dec[K])."""
    az, el = np.asarray(az_rad), np.asarray(el_rad)
    sl, cl = np.sin(lat_rad), np.cos(lat_rad)
    sd = np.clip(sl * np.sin(el) + cl * np.cos(el) * np.cos(az), -1.0, 1.0)
    dec = np.arcsin(sd)
    cd = np.cos(dec)
    safe = np.abs(cd) > 1e-12
    cd_s = np.where(safe, cd, 1.0)
    sin_h = np.where(safe, -np.sin(az) * np.cos(el) / cd_s, 0.0)
    cos_h = np.where(safe, (np.sin(el) - sl * sd) / (cl * cd_s), 1.0)
    return np.arctan2(sin_h, cos_h), dec


def fringe_rate_grid(
    array: Array,
    ha_rad: np.ndarray,
    dec_rad: np.ndarray,
    freq_hz: float,
    ha_pc_rad: float,
    dec_pc_rad: float,
) -> np.ndarray:
    """Vectorized residual fringe rate [K, B] for K directions (ha[K], dec[K])."""
    d = array.baseline_xyz()                              # [B, 3]
    dx, dy = d[:, 0], d[:, 1]
    ha, dec = np.atleast_1d(ha_rad), np.atleast_1d(dec_rad)
    sh, ch, cd = np.sin(ha), np.cos(ha), np.cos(dec)      # [K]
    dw = (-cd * sh)[:, None] * dx[None, :] + (-cd * ch)[:, None] * dy[None, :]  # [K, B]
    shp, chp, cdp = np.sin(ha_pc_rad), np.cos(ha_pc_rad), np.cos(dec_pc_rad)
    dw_pc = -cdp * shp * dx - cdp * chp * dy              # [B]
    return (dw - dw_pc[None, :]) * OMEGA_EARTH * freq_hz / C_LIGHT


def hadec_to_azel(ha_rad: float, dec_rad: float, lat_rad: float) -> tuple[float, float]:
    """(hour angle, declination) -> (az from N through E, el), radians."""
    sl, cl = np.sin(lat_rad), np.cos(lat_rad)
    se = sl * np.sin(dec_rad) + cl * np.cos(dec_rad) * np.cos(ha_rad)
    se = np.clip(se, -1.0, 1.0)
    el = np.arcsin(se)
    ce = np.cos(el)
    if abs(ce) < 1e-12:
        return 0.0, float(el)
    sin_a = -np.sin(ha_rad) * np.cos(dec_rad) / ce
    cos_a = (np.sin(dec_rad) - sl * se) / (cl * ce)
    return float(np.arctan2(sin_a, cos_a) % (2 * np.pi)), float(el)


@dataclass(frozen=True)
class DirectionBuckets:
    """Az/el grid above the horizon with per-baseline residual fringe rates.

    rates_hz[k, b]: expected residual fringe rate of bucket k on baseline b at
    ``freq_hz`` while tracking the phase centre. Scale by (nu / freq_hz) for
    other frequencies (fringe rate is linear in frequency).
    """

    az_deg: np.ndarray      # [K]
    el_deg: np.ndarray      # [K]
    rates_hz: np.ndarray    # [K, B]
    freq_hz: float
    grid_deg: float
    ha_pc_rad: float = field(default=0.0)
    dec_pc_rad: float = field(default=0.0)


_BUCKET_CACHE: dict[tuple, DirectionBuckets] = {}


def direction_buckets(
    array: Array,
    ha_pc_rad: float,
    dec_pc_rad: float,
    freq_hz: float | None = None,
    grid_deg: float = 5.0,
) -> DirectionBuckets:
    """Expected residual fringe rates for an (az, el) grid above the horizon.

    Precomputed per (array, pointing, hour-angle cell): the cache key rounds
    ha to the grid timescale so a scan reuses one bucket set. Used by graph
    relation B, by the simulator's RFI phase generation, and by the satellite
    coincidence validator.
    """
    if freq_hz is None:
        freq_hz = 0.5 * (array.freq_start_hz + array.freq_end_hz)
    key = (
        array.name, array.n_ant, round(float(ha_pc_rad), 3),
        round(float(dec_pc_rad), 4), round(float(freq_hz), -3), grid_deg,
        hash(array.names),
    )
    hit = _BUCKET_CACHE.get(key)
    if hit is not None:
        return hit

    azs = np.arange(grid_deg / 2, 360.0, grid_deg)
    els = np.arange(grid_deg / 2, 90.0, grid_deg)
    az_g, el_g = [a.ravel() for a in np.meshgrid(azs, els)]
    ha_g, dec_g = azel_to_hadec_array(np.deg2rad(az_g), np.deg2rad(el_g), array.lat_rad)
    rates = fringe_rate_grid(array, ha_g, dec_g, freq_hz, ha_pc_rad, dec_pc_rad)
    out = DirectionBuckets(
        az_deg=az_g, el_deg=el_g, rates_hz=rates, freq_hz=float(freq_hz),
        grid_deg=grid_deg, ha_pc_rad=float(ha_pc_rad), dec_pc_rad=float(dec_pc_rad),
    )
    if len(_BUCKET_CACHE) > 256:
        _BUCKET_CACHE.clear()
    _BUCKET_CACHE[key] = out
    return out
