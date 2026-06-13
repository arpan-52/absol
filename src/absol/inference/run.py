"""Inference orchestration: MS scan -> probabilities -> writeback + sidecar."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from absol.geometry import Array
from absol.pipeline import prepare
from absol.training.loop import _array_from_raw, load_model
from absol.utils import load_yaml, resolve_device


def predict_scan(model, scan, sim_cfg, model_cfg, temperature, device):
    """Returns (p_chunk [B,Nt,Nf], p_sample [B,T,F], ant_scores [A,Nt],
    dir_attrib list-of-arrays, prep)."""
    prep = prepare(
        scan.vis.to(device), scan.weights_in.to(device), scan.array,
        scan.ha_rad, scan.dec_rad, scan.freqs_hz, sim_cfg, model_cfg,
    )
    gate = float(model_cfg.get("inference", {}).get("chunk_prob_mask_gate", 0.2))
    b, nt, nf = prep.validity.shape
    t_c, f_c = prep.t_c, prep.f_c
    with torch.no_grad():
        cols = [g.to(device) for g in prep.columns]
        out = model(cols, mask_index=None)
        edge_logits = out["edge_logits"] / temperature        # [B*Nf, Nt]
        p_node = torch.sigmoid(edge_logits)
        sel = torch.nonzero((p_node > gate).reshape(-1)).squeeze(-1)
        mask_p = None
        if sel.numel() > 0:
            ml = model.mask_decoder(out["h"].reshape(-1, out["h"].shape[-1])[sel])
            mask_p = torch.sigmoid(ml / temperature)

    # node-major [B*Nf, Nt] -> [B, Nt, Nf]
    p_chunk = p_node.reshape(b, nf, nt).permute(0, 2, 1).contiguous()
    ant_scores = torch.sigmoid(out["ant_logits"] / temperature)

    # paint p_sample [B, T, F]
    p_pad = p_chunk[:, :, :, None, None].expand(b, nt, nf, t_c, f_c).clone()
    if mask_p is not None:
        flat = p_pad.permute(0, 2, 1, 3, 4).reshape(b * nf * nt, t_c, f_c)
        flat[sel] = mask_p
        p_pad = flat.reshape(b, nf, nt, t_c, f_c).permute(0, 2, 1, 3, 4)
    p_full = p_pad.permute(0, 1, 3, 2, 4).reshape(b, nt * t_c, nf * f_c)
    p_sample = p_full[:, :prep.n_t, :prep.n_f]
    p_sample = p_sample * (scan.weights_in.to(device) > 0)    # pre-flagged -> 0 weight path

    dir_bucket = torch.stack(
        [prep.columns[k]["bl"].dir_bucket.reshape(b, nf) for k in range(nt)], dim=1
    )                                                          # [B, Nt, Nf]
    return (
        p_chunk.cpu().numpy(), p_sample.cpu().numpy(), ant_scores.cpu().numpy(),
        dir_bucket.cpu().numpy(), prep,
    )


def quicklook_png(path: Path, p_chunk, ant_scores, p_sample, ant_names):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    b, nt, nf = p_chunk.shape
    axes[0].imshow(p_chunk.reshape(b, nt * nf), aspect="auto", vmin=0, vmax=1, cmap="inferno")
    axes[0].set_title("p_chunk (baseline x chunk)")
    axes[0].set_xlabel("chunk (t-major)")
    axes[0].set_ylabel("baseline")
    axes[1].bar(range(len(ant_names)), ant_scores.mean(axis=1))
    axes[1].set_xticks(range(len(ant_names)))
    axes[1].set_xticklabels(ant_names, rotation=90, fontsize=6)
    axes[1].set_title("antenna score (scan mean)")
    axes[2].plot((p_sample > 0.5).mean(axis=(0, 1)))
    axes[2].set_title("flag fraction per channel")
    axes[2].set_xlabel("channel")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_inference(
    ms_path: str,
    run_dir: str,
    write_flags: bool = False,
    threshold: float | None = None,
    data_column: str = "auto",
    subtract_model: bool = False,
    device: str | None = None,
    max_scans: int | None = None,
) -> Path:
    from absol.inference.ms_io import read_ms, write_back

    run = Path(run_dir)
    dev = resolve_device(device)
    model, ckpt = load_model(run, dev)
    sim_cfg = load_yaml(run / "sim.yaml")
    model_cfg = load_yaml(run / "model.yaml")
    array_raw = load_yaml(run / "array.yaml")
    t_json = run / "T.json"
    temperature = json.loads(t_json.read_text())["temperature"] if t_json.exists() else 1.0
    if threshold is None:
        threshold = float(model_cfg.get("inference", {}).get("flag_threshold", 0.5))

    array = _array_from_raw(array_raw, run)
    # the MS defines the channelization at inference time
    scans = read_ms(ms_path, array, data_column=data_column,
                    subtract_model=subtract_model, max_scans=max_scans)
    if not scans:
        raise RuntimeError(f"no usable scans found in {ms_path}")
    ms_freqs = scans[0].freqs_hz
    array = Array(
        name=array.name, names=array.names, enu=array.enu,
        latitude_deg=array.latitude_deg, longitude_deg=array.longitude_deg,
        freq_start_hz=float(ms_freqs.min()), freq_end_hz=float(ms_freqs.max()),
        n_channels=int(ms_freqs.size), n_pol=array.n_pol,
        integration_s=float(np.median(np.diff(scans[0].times))) if scans[0].times.size > 1
        else array.integration_s,
    )

    sidecar = Path(str(ms_path).rstrip("/") + ".absol.h5")
    p_samples = []
    with h5py.File(sidecar, "w") as h5:
        h5.attrs["run"] = str(run)
        h5.attrs["temperature"] = temperature
        for scan in scans:
            p_chunk, p_sample, ant_scores, dir_bucket, prep = predict_scan(
                model, scan, sim_cfg, model_cfg, temperature, dev
            )
            p_samples.append(p_sample)
            g = h5.create_group(f"scan_{scan.scan_id}")
            g.create_dataset("p_chunk", data=p_chunk, compression="gzip")
            g.create_dataset("p_sample", data=p_sample.astype(np.float16), compression="gzip")
            g.create_dataset("antenna_scores", data=ant_scores)
            g.create_dataset("dir_bucket", data=dir_bucket)
            g.create_dataset("bucket_az_deg", data=prep.static.buckets.az_deg)
            g.create_dataset("bucket_el_deg", data=prep.static.buckets.el_deg)
            g.attrs["antenna_names"] = list(scan.array.names)
            quicklook_png(
                Path(str(ms_path).rstrip("/") + f".absol_scan{scan.scan_id}.png"),
                p_chunk, ant_scores, p_sample, scan.array.names,
            )

    write_back(ms_path, scans, p_samples, write_flags=write_flags, threshold=threshold)
    return sidecar
