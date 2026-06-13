"""Training loop: curriculum stages, AdamW + cosine, AMP, JSONL logging.

No external tracker dependency; metrics stream to ``<out>/log.jsonl``.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from absol.geometry import Array
from absol.model import AbsolModel
from absol.training.dataset import SceneDataset, scene_to_item
from absol.training.losses import compute_losses
from absol.utils import apply_override, load_yaml, resolve_device, set_seed


def _move_item(item: dict, device: torch.device) -> dict:
    out = dict(item)
    out["columns"] = [g.to(device) for g in item["columns"]]
    for k in ("edge_target", "edge_weight", "ant_target", "mask_index",
              "mask_target", "mask_weight"):
        out[k] = item[k].to(device)
    return out


def _step(model: AbsolModel, item: dict, cfg_train: dict) -> dict:
    out = model(item["columns"], mask_index=item["mask_index"])
    labels = {
        "edge_target": item["edge_target"], "edge_weight": item["edge_weight"],
        "ant_target": item["ant_target"], "mask_target": item["mask_target"],
        "mask_weight": item["mask_weight"],
    }
    return compute_losses(out, labels, cfg_train)


def load_configs(sim_cfg_path: str, array_cfg_path: str, model_cfg_path: str,
                 overrides: list[tuple[str, object]] | None = None):
    sim_cfg = load_yaml(sim_cfg_path)
    array_raw = load_yaml(array_cfg_path)
    model_cfg = load_yaml(model_cfg_path)
    for key, val in overrides or []:
        root = key.split(".", 1)[0]
        if root in ("band", "observation", "array"):
            apply_override(array_raw, key, val)
        elif root in ("encoder", "gnn", "temporal", "graph", "training",
                      "calibration", "inference"):
            apply_override(model_cfg, key, val)
        else:
            apply_override(sim_cfg, key, val)
    return sim_cfg, array_raw, model_cfg


def _array_from_raw(array_raw: dict, tmpdir: Path) -> Array:
    import yaml
    p = tmpdir / "_array_resolved.yaml"
    p.write_text(yaml.safe_dump(array_raw))
    return Array.from_yaml(p)


def train(
    sim_cfg_path: str,
    array_cfg_path: str,
    model_cfg_path: str,
    out_dir: str,
    overrides: list[tuple[str, object]] | None = None,
    overfit_one: bool = False,
    steps: int | None = None,
    num_workers: int = 0,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sim_cfg, array_raw, model_cfg = load_configs(
        sim_cfg_path, array_cfg_path, model_cfg_path, overrides
    )
    array = _array_from_raw(array_raw, out)
    cfg_t = model_cfg["training"]
    set_seed(int(cfg_t.get("seed", 1234)))
    device = resolve_device(cfg_t.get("device"))
    use_amp = bool(cfg_t.get("amp", True)) and device.type == "cuda"

    # snapshot configs into the run dir (inference reads them from here)
    for src, name in ((sim_cfg_path, "sim.yaml"), (model_cfg_path, "model.yaml")):
        shutil.copy(src, out / name)
    import yaml
    (out / "sim.yaml").write_text(yaml.safe_dump(sim_cfg))
    (out / "model.yaml").write_text(yaml.safe_dump(model_cfg))
    (out / "array.yaml").write_text(yaml.safe_dump(array_raw))

    t_c = int(sim_cfg["chunking"]["chunk_time_samples"])
    f_c = int(sim_cfg["chunking"]["chunk_freq_channels"])
    model = AbsolModel(model_cfg, t_c, f_c).to(device)
    n_par = model.count_params()
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(cfg_t["lr"]),
        weight_decay=float(cfg_t.get("weight_decay", 0.01)),
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)
    log_f = (out / "log.jsonl").open("a")

    def log(rec: dict) -> None:
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    log({"event": "start", "params": n_par, "device": str(device), "overfit": overfit_one})

    batch_scenes = int(cfg_t.get("batch_scenes", 4))
    stages_steps = list(cfg_t["steps_per_stage"])
    if overfit_one:
        stages_steps = [int(steps or 300)]
    elif steps is not None:
        stages_steps = [min(int(steps), s) for s in stages_steps]

    # GPU-native by default: with num_workers=0 scenes are simulated in-process
    # on the training device (no CPU hop). Multi-worker generation must stay on
    # CPU (CUDA is unsafe in forked workers).
    gen_device = str(device) if num_workers == 0 else "cpu"
    log({"event": "gen", "gen_device": gen_device, "num_workers": num_workers})

    global_step = 0
    first_loss = last_loss = None
    for stage_i, n_steps in enumerate(stages_steps, start=1):
        stage = min(stage_i, 4) if not overfit_one else 2
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_steps, 1))
        if overfit_one:
            from absol.simulator.scenes import SceneGenerator
            gen = SceneGenerator(sim_cfg, array, seed=int(cfg_t.get("seed", 1234)),
                                 device=gen_device)
            item_fixed = scene_to_item(gen.sample(stage), sim_cfg, model_cfg)
            iterator = iter(lambda: item_fixed, None)
        else:
            ds = SceneDataset(sim_cfg, array, model_cfg, stage,
                              seed=int(cfg_t.get("seed", 1234)) + 7919 * stage_i,
                              device=gen_device)
            if num_workers == 0:
                iterator = iter(ds)               # in-process, on-device generation
            else:
                loader = DataLoader(ds, batch_size=None, num_workers=num_workers,
                                    persistent_workers=True)
                iterator = iter(loader)

        for step in range(n_steps):
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            comps = {"edge": 0.0, "antenna": 0.0, "mask": 0.0}
            n_acc = 1 if overfit_one else batch_scenes
            for _ in range(n_acc):
                item = _move_item(next(iterator), device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    losses = _step(model, item, cfg_t)
                scaler.scale(losses["total"] / n_acc).backward()
                tot += float(losses["total"].detach()) / n_acc
                for k in comps:
                    comps[k] += float(losses[k]) / n_acc
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            global_step += 1
            if first_loss is None:
                first_loss = tot
            last_loss = tot
            if global_step % int(cfg_t.get("log_every", 50)) == 0 or step == 0:
                log({"step": global_step, "stage": stage, "loss": tot, **comps,
                     "lr": sched.get_last_lr()[0], "t": time.time()})
            if global_step % int(cfg_t.get("checkpoint_every", 1000)) == 0:
                _save(model, model_cfg, t_c, f_c, out / "last.pt", global_step)
        _save(model, model_cfg, t_c, f_c, out / f"stage{stage_i}.pt", global_step)

    _save(model, model_cfg, t_c, f_c, out / "model.pt", global_step)
    log({"event": "done", "steps": global_step,
         "first_loss": first_loss, "last_loss": last_loss})
    log_f.close()
    return out


def _save(model, model_cfg, t_c, f_c, path: Path, step: int) -> None:
    torch.save({"state_dict": model.state_dict(), "model_cfg": model_cfg,
                "t_c": t_c, "f_c": f_c, "step": step}, path)


def load_model(run_dir: str | Path, device: torch.device) -> tuple[AbsolModel, dict]:
    run = Path(run_dir)
    ckpt = torch.load(run / "model.pt", map_location=device, weights_only=False)
    model = AbsolModel(ckpt["model_cfg"], ckpt["t_c"], ckpt["f_c"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
