"""Noise-vs-integration-time test.

The user images the data externally (CASA/WSClean) at cumulative
integration times and provides, per flagging scheme, either a CSV with
columns ``t_seconds,sigma_jy`` or FITS image paths (lazy astropy). We fit
sigma(t) = sigma_1 / sqrt(t) on the early points and report the integration
time at which the measured curve departs by more than ``departure_frac``.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class NoiseCurveResult:
    labels: list[str] = field(default_factory=list)
    departure_time_s: dict[str, float] = field(default_factory=dict)
    sigma_1s_jy: dict[str, float] = field(default_factory=dict)
    departure_frac: float = 0.1
    figure: str = ""


def _read_series(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    if "csv" in entry:
        ts, sig = [], []
        with open(entry["csv"]) as f:
            for row in csv.DictReader(f):
                ts.append(float(row["t_seconds"]))
                sig.append(float(row["sigma_jy"]))
        return np.asarray(ts), np.asarray(sig)
    if "fits" in entry:
        from astropy.io import fits as afits  # lazy
        ts, sig = [], []
        for t_s, path in entry["fits"]:
            with afits.open(path) as hdul:
                data = hdul[0].data.squeeze()
            ts.append(float(t_s))
            sig.append(float(1.4826 * np.median(np.abs(data - np.median(data)))))
        return np.asarray(ts), np.asarray(sig)
    raise ValueError("series entry needs 'csv' or 'fits'")


def run_noise_curve(cfg: dict, out_dir: str) -> NoiseCurveResult:
    ncfg = cfg["noise_curve"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = NoiseCurveResult(departure_frac=float(ncfg.get("departure_frac", 0.1)))

    fig, ax = plt.subplots(figsize=(6, 4))
    for entry in ncfg.get("series", []):
        label = entry["label"]
        t, sig = _read_series(entry)
        order = np.argsort(t)
        t, sig = t[order], sig[order]
        n_fit = max(2, len(t) // 4)
        sigma_1 = float(np.median(sig[:n_fit] * np.sqrt(t[:n_fit])))
        expect = sigma_1 / np.sqrt(t)
        dep = np.abs(sig / expect - 1) > result.departure_frac
        t_dep = float(t[dep][0]) if dep.any() else float(t[-1])
        result.labels.append(label)
        result.departure_time_s[label] = t_dep
        result.sigma_1s_jy[label] = sigma_1
        ax.loglog(t, sig, "o-", label=label)
        ax.loglog(t, expect, "--", alpha=0.5)
    ax.set_xlabel("integration time (s)")
    ax.set_ylabel("image noise (Jy)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = out / "noise_curve.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    result.figure = str(fig_path)
    (out / "noise_curve.json").write_text(json.dumps(asdict(result), indent=2))
    return result
