"""Temperature scaling on a frozen model + reliability diagrams.

Outputs ``T.json`` and reliability PNGs (overall + per RFI-strength bin)
into the run directory. Calibrated probability = sigmoid(logit / T).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from absol.training.dataset import scene_to_item
from absol.training.loop import _array_from_raw, _move_item, load_model
from absol.utils import load_yaml, resolve_device

STRENGTH_BINS = [(0.0, 1.0), (1.0, 5.0), (5.0, 100.0), (100.0, np.inf)]


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error (equal-width bins)."""
    edges = np.linspace(0, 1, n_bins + 1)
    out, n = 0.0, len(probs)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (probs >= lo) & (probs < hi)
        if sel.sum() == 0:
            continue
        out += sel.sum() / n * abs(probs[sel].mean() - labels[sel].mean())
    return float(out)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits / log_t.exp(), labels
        )
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def _reliability_png(probs, labels, path: Path, title: str) -> None:
    edges = np.linspace(0, 1, 16)
    mids, accs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (probs >= lo) & (probs < hi)
        if sel.sum() > 0:
            mids.append(probs[sel].mean())
            accs.append(labels[sel].mean())
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(mids, accs, "o-")
    ax.set_xlabel("predicted p")
    ax.set_ylabel("empirical fraction")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def calibrate(run_dir: str, n_scenes: int | None = None, seed: int = 99991) -> dict:
    run = Path(run_dir)
    sim_cfg = load_yaml(run / "sim.yaml")
    model_cfg = load_yaml(run / "model.yaml")
    array = _array_from_raw(load_yaml(run / "array.yaml"), run)
    device = resolve_device(model_cfg["training"].get("device"))
    model, _ = load_model(run, device)

    n = int(n_scenes or model_cfg["calibration"]["holdout_scenes"])
    from absol.simulator.scenes import SceneGenerator
    gen = SceneGenerator(sim_cfg, array, seed=seed)

    logits_all, labels_all, strength_all = [], [], []
    with torch.no_grad():
        for _ in range(n):
            scene = gen.sample(4)
            item = _move_item(scene_to_item(scene, sim_cfg, model_cfg), device)
            out = model(item["columns"], mask_index=None)
            valid = item["edge_weight"] > 0
            logits_all.append(out["edge_logits"][valid].cpu())
            labels_all.append(item["edge_target"][valid].cpu())
            strength_all.append(
                torch.full((int(valid.sum()),), float(scene.meta["max_strength_sigma"]))
            )
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    strength = torch.cat(strength_all).numpy()

    t_val = fit_temperature(logits, labels)
    p_raw = torch.sigmoid(logits).numpy()
    p_cal = torch.sigmoid(logits / t_val).numpy()
    y = labels.numpy()
    result = {
        "temperature": t_val,
        "ece_before": ece(p_raw, y),
        "ece_after": ece(p_cal, y),
        "n_scenes": n,
        "n_chunks": int(len(y)),
    }
    (run / "T.json").write_text(json.dumps(result, indent=2))
    _reliability_png(p_cal, y, run / "reliability_overall.png", f"all (T={t_val:.2f})")
    for lo, hi in STRENGTH_BINS:
        sel = (strength >= lo) & (strength < hi)
        if sel.sum() > 100:
            _reliability_png(
                p_cal[sel], y[sel], run / f"reliability_s{lo:g}-{hi:g}.png",
                f"strength {lo:g}-{hi:g} sigma",
            )
    return result
