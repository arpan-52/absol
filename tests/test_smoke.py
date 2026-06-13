"""Spec test 8: end-to-end smoke on CPU < 60 s, all data contracts asserted.

Also asserts total model params < 5M (spec 5.8) and that a short overfit
run decreases the loss (CI smoke-train, spec milestone M4).
"""
from __future__ import annotations

import time

import torch

from absol.features import N_PHYS
from absol.model import AbsolModel
from absol.simulator.scenes import SceneGenerator
from absol.training.dataset import scene_to_item
from absol.training.losses import compute_losses


def test_8_end_to_end_contracts(tiny_array, tiny_sim_cfg, model_cfg):
    t0 = time.time()
    gen = SceneGenerator(tiny_sim_cfg, tiny_array, seed=42)
    scene = gen.sample(stage=4)

    a = scene.array.n_ant
    n_b = a * (a - 1) // 2
    n_t, n_f, n_p = 24, 256, 2
    # section 4 contracts
    assert scene.vis.shape == (n_b, n_t, n_f, n_p) and scene.vis.dtype == torch.complex64
    assert scene.weights_in.shape == (n_b, n_t, n_f)
    assert scene.truth_mask.shape == (n_b, n_t, n_f) and scene.truth_mask.dtype == torch.bool
    assert scene.truth_antennas.shape == (a,)

    item = scene_to_item(scene, tiny_sim_cfg, model_cfg)
    t_c, f_c = 16, 128
    nt, nf = 2, 2
    cols = item["columns"]
    assert len(cols) == nt
    assert cols[0]["bl"].tile.shape == (n_b * nf, 2, t_c, f_c)
    assert cols[0]["bl"].phys.shape == (n_b * nf, N_PHYS)
    assert item["edge_target"].shape == (n_b * nf, nt)

    model = AbsolModel(model_cfg, t_c, f_c)
    n_par = model.count_params()
    assert n_par < 5_000_000, f"params {n_par}"

    out = model(cols, mask_index=item["mask_index"])
    assert out["edge_logits"].shape == (n_b * nf, nt)
    assert out["ant_logits"].shape == (a, nt)
    p_chunk = torch.sigmoid(out["edge_logits"]).reshape(n_b, nf, nt).permute(0, 2, 1)
    assert p_chunk.shape == (n_b, nt, nf)            # p_chunk [B, Nc] contract
    if item["mask_index"].numel() > 0:
        assert out["mask_logits"].shape[1:] == (t_c, f_c)

    losses = compute_losses(out, item, model_cfg["training"])
    losses["total"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert torch.isfinite(losses["total"])
    assert time.time() - t0 < 60, "smoke must run in < 60 s on CPU"


def test_8b_overfit_loss_decreases(tiny_array, tiny_sim_cfg, model_cfg, tmp_path):
    import yaml

    from absol.training.loop import train

    sim_p = tmp_path / "sim.yaml"
    sim_p.write_text(yaml.safe_dump(tiny_sim_cfg))
    arr_p = tmp_path / "array.yaml"
    arr_p.write_text(yaml.safe_dump({
        "array": {"name": "GMRT-tiny", "latitude_deg": 19.0963, "longitude_deg": 74.05,
                  "antennas": [
                      {"id": n, "e": float(e), "n": float(nn), "u": float(u)}
                      for n, (e, nn, u) in zip(tiny_array.names, tiny_array.enu)
                  ]},
        "band": {"freq_start_hz": 550.0e6, "freq_end_hz": 750.0e6,
                 "n_channels": 256, "n_pol": 2},
        "observation": {"integration_s": 8.0},
    }))
    mdl_p = tmp_path / "model.yaml"
    mdl_p.write_text(yaml.safe_dump(model_cfg))

    out = train(str(sim_p), str(arr_p), str(mdl_p), str(tmp_path / "run"),
                overfit_one=True, steps=60, num_workers=0)
    import json
    recs = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    done = [r for r in recs if r.get("event") == "done"][-1]
    assert done["last_loss"] < done["first_loss"], (
        f"loss did not decrease: {done['first_loss']} -> {done['last_loss']}"
    )
    assert (out / "model.pt").exists()
