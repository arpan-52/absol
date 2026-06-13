"""Coincidence tests: TLE satellite transits & 100 Hz powerline folding.

(a) Satellite: user supplies a TLE catalogue; sgp4 (optional dep, [validate]
extra) propagates each satellite to the dump times; transit windows are
dumps where a satellite is above the horizon within ``pointing_ring_deg`` of
the pointing. Statistic: excess of flagged fraction inside vs outside the
windows; significance via circular time shuffles.

(b) Powerline: fold the per-dump flagged fraction at the (aliased) 100 Hz
modulation frequency and report the folded-profile chi^2 significance.
NOTE: with 8 s dumps 100 Hz aliases near 0; this test is most meaningful
for high-time-resolution data — the caveat is recorded in the result.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class CoincidenceResult:
    satellite_excess: float = float("nan")
    satellite_sigma: float = float("nan")
    n_window_dumps: int = 0
    powerline_chi2_sigma: float = float("nan")
    fold_hz: float = 100.0
    caveats: list[str] = field(default_factory=list)
    figure: str = ""


def _flag_series_from_sidecar(sidecar: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-dump flagged fraction (p_sample > 0.5) from the inference sidecar."""
    with h5py.File(sidecar, "r") as h5:
        scans = sorted(k for k in h5 if k.startswith("scan_"))
        fracs, times = [], []
        for k in scans:
            p = h5[k]["p_sample"][...]
            fracs.append((p > 0.5).mean(axis=(0, 2)))
            times.append(np.arange(p.shape[1]))
    return np.concatenate(times), np.concatenate(fracs)


def _sat_windows(tle_file: str, times_mjd_s: np.ndarray, lat_deg: float,
                 lon_deg: float, ring_deg: float) -> np.ndarray:
    from sgp4.api import Satrec  # lazy optional dep

    sats = []
    lines = Path(tle_file).read_text().strip().splitlines()
    for i in range(0, len(lines) - 1):
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            sats.append(Satrec.twoline2rv(lines[i], lines[i + 1]))
    windows = np.zeros(times_mjd_s.size, dtype=bool)
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    for k, t_s in enumerate(times_mjd_s):
        mjd = t_s / 86400.0
        jd = mjd + 2400000.5
        jdi, fr = int(jd), jd - int(jd)
        gmst = (280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360.0
        for sat in sats:
            err, r_teme, _ = sat.sgp4(jdi + 0.5, fr - 0.5)
            if err != 0:
                continue
            th = np.deg2rad(gmst)
            x = np.cos(th) * r_teme[0] + np.sin(th) * r_teme[1]
            y = -np.sin(th) * r_teme[0] + np.cos(th) * r_teme[1]
            z = r_teme[2]
            re = 6378.137
            obs = np.array([
                re * np.cos(lat) * np.cos(lon), re * np.cos(lat) * np.sin(lon),
                re * np.sin(lat),
            ])
            d = np.array([x, y, z]) - obs
            up = obs / np.linalg.norm(obs)
            el = np.rad2deg(np.arcsin(np.dot(d / np.linalg.norm(d), up)))
            if el > 0 and el < 90 - 0 and el > 90 - 90:  # above horizon
                if el > 90 - ring_deg * 3:               # generous sidelobe ring
                    windows[k] = True
                    break
    return windows


def run_coincidence(
    sidecar: str, cfg: dict, out_dir: str,
    times_mjd_s: np.ndarray | None = None,
    lat_deg: float = 19.0963, lon_deg: float = 74.05,
) -> CoincidenceResult:
    ccfg = cfg["coincidence"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = CoincidenceResult(fold_hz=float(ccfg.get("powerline_fold_hz", 100.0)))

    t_idx, frac = _flag_series_from_sidecar(sidecar)

    # (a) satellite transit excess
    tle = ccfg.get("tle_file")
    if tle and times_mjd_s is not None:
        win = _sat_windows(tle, times_mjd_s, lat_deg, lon_deg,
                           float(ccfg.get("pointing_ring_deg", 30.0)))
        result.n_window_dumps = int(win.sum())
        if 0 < win.sum() < win.size:
            obs = frac[win].mean() - frac[~win].mean()
            n_sh = int(ccfg.get("n_time_shuffles", 1000))
            rng = np.random.default_rng(0)
            null = np.empty(n_sh)
            for s in range(n_sh):
                w = np.roll(win, rng.integers(1, win.size))
                null[s] = frac[w].mean() - frac[~w].mean()
            result.satellite_excess = float(obs)
            result.satellite_sigma = float((obs - null.mean()) / (null.std() + 1e-12))
    else:
        result.caveats.append("satellite test skipped: no TLE file or times")

    # (b) powerline fold
    dump_s = 8.0
    f_alias = abs(result.fold_hz - round(result.fold_hz * dump_s) / dump_s)
    if f_alias < 1e-6:
        result.caveats.append(
            f"100 Hz aliases to ~0 at {dump_s}s dumps; powerline fold not significant by construction"
        )
    phase = (t_idx * dump_s * max(f_alias, 1e-9)) % 1.0
    nb = 8
    prof = np.array([frac[(phase >= k / nb) & (phase < (k + 1) / nb)].mean()
                     if ((phase >= k / nb) & (phase < (k + 1) / nb)).any() else np.nan
                     for k in range(nb)])
    good = np.isfinite(prof)
    if good.sum() > 2:
        mu = np.nanmean(prof)
        sd = np.nanstd(prof) + 1e-12
        chi2 = float(np.nansum(((prof - mu) / sd) ** 2))
        result.powerline_chi2_sigma = float((chi2 - good.sum()) / np.sqrt(2 * good.sum()))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(frac)
    ax[0].set_title("per-dump flagged fraction")
    ax[1].plot(prof, "o-")
    ax[1].set_title(f"folded @ {result.fold_hz} Hz (aliased)")
    fig.tight_layout()
    fig_path = out / "coincidence.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    result.figure = str(fig_path)
    (out / "coincidence.json").write_text(json.dumps(asdict(result), indent=2))
    return result
