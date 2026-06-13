"""Post-flagging residual statistics: Gaussianity per baseline class.

After applying ABSOL weights, the surviving data should be thermal-noise
like: spectral kurtosis ~ Gaussian expectation and no excess power at zero
or off fringe rates. The known central-square vs arm contamination asymmetry
must appear in the probabilities, NOT in the weighted residuals.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from absol.features import (
    chunk_grid,
    delay_fringe_tiles,
    mad_normalize,
    physics_features,
    savgol_residual,
)
from absol.graphs import build_static
from absol.training.loop import _array_from_raw
from absol.utils import load_yaml


@dataclass
class ResidualStatsResult:
    sk_excess_central: float = float("nan")
    sk_excess_arm: float = float("nan")
    offfringe_central: float = float("nan")
    offfringe_arm: float = float("nan")
    kurtosis_pvalue_proxy: float = float("nan")
    figure: str = ""
    notes: list[str] = field(default_factory=list)


def run_residual_stats(
    ms_path: str, run_dir: str, sidecar: str, cfg: dict, out_dir: str | None = None,
) -> ResidualStatsResult:
    import h5py

    from absol.inference.ms_io import read_ms

    run = Path(run_dir)
    out = Path(out_dir or run / "validation")
    out.mkdir(parents=True, exist_ok=True)
    sim_cfg = load_yaml(run / "sim.yaml")
    model_cfg = load_yaml(run / "model.yaml")
    array = _array_from_raw(load_yaml(run / "array.yaml"), run)
    scan = read_ms(ms_path, array, max_scans=1)[0]

    with h5py.File(sidecar, "r") as h5:
        p_sample = h5[f"scan_{scan.scan_id}"]["p_sample"][...].astype(np.float32)

    w_eff = scan.weights_in * torch.as_tensor(1.0 - p_sample)
    res = savgol_residual(scan.vis, scan.weights_in,
                          int(sim_cfg["residual"]["savgol_window_dumps"]))
    vis_n, _ = mad_normalize(res, scan.weights_in)
    t_c = int(sim_cfg["chunking"]["chunk_time_samples"])
    f_c = int(sim_cfg["chunking"]["chunk_freq_channels"])
    chunks, wch, validity = chunk_grid(vis_n, (w_eff > 0.5).float(), t_c, f_c)
    tiles = delay_fringe_tiles(chunks, wch)
    static = build_static(
        scan.array, float(scan.ha_rad[len(scan.ha_rad) // 2]), scan.dec_rad,
        scan.freqs_hz, chunks.shape[2], model_cfg["graph"],
    )
    feats = physics_features(chunks, wch, tiles, validity, static.pairs,
                             scan.array.n_ant, static.uv_dist_m)
    b, nt, nf, _ = feats.shape
    sk = feats[..., 0].reshape(b, -1)
    off = feats[..., 1].reshape(b, -1)
    val = validity.reshape(b, -1) > 0.3

    cmax = float(cfg["residual_stats"].get("central_square_max_baseline_m", 2000.0))
    central = torch.as_tensor(static.uv_dist_m < cmax)
    res_out = ResidualStatsResult()

    def _m(x, sel_b):
        sel = val & sel_b[:, None]
        return float(x[sel].mean()) if sel.any() else float("nan")

    res_out.sk_excess_central = _m(sk, central)
    res_out.sk_excess_arm = _m(sk, ~central)
    res_out.offfringe_central = _m(off, central)
    res_out.offfringe_arm = _m(off, ~central)
    res_out.kurtosis_pvalue_proxy = float(abs(res_out.sk_excess_central)
                                          + abs(res_out.sk_excess_arm))
    if abs(res_out.sk_excess_central - res_out.sk_excess_arm) > 0.5:
        res_out.notes.append(
            "central-square vs arm SK differ post-weighting: contamination "
            "asymmetry leaked into residuals"
        )

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].hist([sk[central].flatten().numpy(), sk[~central].flatten().numpy()],
               bins=50, label=["central", "arm"], density=True, histtype="step")
    ax[0].set_xlabel("SK - 1 (post-weighting)")
    ax[0].legend()
    ax[1].hist([off[central].flatten().numpy(), off[~central].flatten().numpy()],
               bins=50, label=["central", "arm"], density=True, histtype="step")
    ax[1].set_xlabel("off-fringe power fraction")
    fig.tight_layout()
    fig_path = out / "residual_stats.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    res_out.figure = str(fig_path)
    (out / "residual_stats.json").write_text(json.dumps(asdict(res_out), indent=2))
    return res_out
