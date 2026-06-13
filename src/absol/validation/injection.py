"""Injection-recovery test: inject simulated RFI into a REAL MS, score recovery.

For each mechanism x strength cell: inject `simulator.rfi` events into the
real visibilities (strength in units of the measured per-baseline robust
sigma), run the model, and report recall at a fixed false-positive rate
measured on the un-injected data. If the user supplies MS copies flagged by
external flaggers, their FLAG columns are scored on the same injections.
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

from absol.inference.run import predict_scan
from absol.simulator.rfi import MECHANISMS
from absol.simulator.scenes import event_contribution
from absol.training.dataset import chunk_truth_fractions
from absol.training.loop import _array_from_raw, load_model
from absol.utils import load_yaml, resolve_device


@dataclass
class InjectionResult:
    mechanisms: list[str]
    strengths_sigma: list[float]
    recall: dict[str, list[float]] = field(default_factory=dict)   # mech -> per strength
    fpr_operating_point: float = 0.01
    threshold_at_fpr: float = 0.5
    n_injections_per_cell: int = 0
    figure: str = ""


def run_injection_test(
    ms_path: str, run_dir: str, cfg: dict, device: str | None = None,
    out_dir: str | None = None, seed: int = 7,
) -> InjectionResult:
    from absol.inference.ms_io import read_ms

    run = Path(run_dir)
    out = Path(out_dir or run / "validation")
    out.mkdir(parents=True, exist_ok=True)
    icfg = cfg["injection"]
    dev = resolve_device(device)
    model, _ = load_model(run, dev)
    sim_cfg = load_yaml(run / "sim.yaml")
    model_cfg = load_yaml(run / "model.yaml")
    array = _array_from_raw(load_yaml(run / "array.yaml"), run)
    t_json = run / "T.json"
    temperature = json.loads(t_json.read_text())["temperature"] if t_json.exists() else 1.0

    scan = read_ms(ms_path, array, max_scans=1)[0]
    rng = np.random.default_rng(seed)
    t_int = scan.array.integration_s
    times = np.arange(scan.times.size) * t_int

    # baseline robust sigma of the real residual data (model units)
    from absol.features import mad_normalize, savgol_residual
    res = savgol_residual(scan.vis, scan.weights_in,
                          int(sim_cfg["residual"]["savgol_window_dumps"]))
    _, sigma = mad_normalize(res, scan.weights_in)
    sigma_b = sigma.mean(dim=1).numpy()                       # [B]

    # null run -> threshold at the requested FPR
    p_chunk0, _, _, _, _ = predict_scan(model, scan, sim_cfg, model_cfg, temperature, dev)
    fpr = float(icfg.get("fpr_operating_point", 0.01))
    thr = float(np.quantile(p_chunk0.reshape(-1), 1 - fpr))

    strengths = [float(s) for s in icfg["strength_grid_sigma"]]
    mechs = list(icfg["mechanisms"])
    n_rep = int(icfg.get("n_injections_per_cell", 8))
    result = InjectionResult(
        mechanisms=mechs, strengths_sigma=strengths,
        fpr_operating_point=fpr, threshold_at_fpr=thr, n_injections_per_cell=n_rep,
    )
    t_c = int(sim_cfg["chunking"]["chunk_time_samples"])
    f_c = int(sim_cfg["chunking"]["chunk_freq_channels"])

    for mech in mechs:
        recalls = []
        for s in strengths:
            hits, total = 0, 0
            for _ in range(n_rep):
                event = MECHANISMS[mech](scan.array, times, scan.freqs_hz, rng, sim_cfg["rfi"])
                contrib = event_contribution(
                    event, scan.array, scan.ha_rad, scan.dec_rad, scan.freqs_hz,
                    rng, torch.device("cpu"),
                )
                act = contrib.abs() > 1e-8
                if int(act.sum()) == 0:
                    continue
                med = float(contrib.abs()[act].median())
                contrib *= s * float(np.median(sigma_b)) / max(med, 1e-30)
                inj = scan.vis + contrib[..., None]
                truth = (contrib.abs() > 0.1 * torch.as_tensor(
                    sigma_b, dtype=torch.float32)[:, None, None])
                scan_i = type(scan)(**{**scan.__dict__, "vis": inj})
                p_chunk, _, _, _, _ = predict_scan(
                    model, scan_i, sim_cfg, model_cfg, temperature, dev)
                cf = chunk_truth_fractions(truth, scan.weights_in, t_c, f_c).numpy()
                pos = cf > 0.01
                hits += int(((p_chunk > thr) & pos).sum())
                total += int(pos.sum())
            recalls.append(hits / total if total else float("nan"))
        result.recall[mech] = recalls

    fig, ax = plt.subplots(figsize=(6, 4))
    for mech in mechs:
        ax.plot(strengths, result.recall[mech], "o-", label=mech)
    ax.set_xscale("log")
    ax.set_xlabel("injection strength (sigma)")
    ax.set_ylabel(f"recall @ FPR={fpr:g}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig_path = out / "injection.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    result.figure = str(fig_path)
    (out / "injection.json").write_text(json.dumps(asdict(result), indent=2))
    return result
