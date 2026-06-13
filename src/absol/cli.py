"""ABSOL command-line interface.

    absol simulate   --sim-config ... --array-config ... --out scenes.h5 --n 100
    absol train      --sim-config ... --array-config ... --model-config ... --out runs/x
    absol calibrate  --run runs/x
    absol infer      --ms obs.ms --run runs/x [--write-flags --threshold 0.5]
    absol validate injection|coincidence|residuals|report --ms ... --run ... --cfg ...
"""
from __future__ import annotations

import argparse
import json
import sys

from absol.utils import load_yaml, parse_overrides


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--override", action="append", default=[],
                   help="dotpath config override, e.g. band.n_channels=1024")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="absol",
                                 description="Array-Based Signal Outlier Locator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("simulate", help="generate frozen scenes to HDF5")
    p.add_argument("--sim-config", required=True)
    p.add_argument("--array-config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--stage", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    _add_common(p)

    p = sub.add_parser("train", help="curriculum training on on-the-fly scenes")
    p.add_argument("--sim-config", required=True)
    p.add_argument("--array-config", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--overfit-one", action="store_true")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--workers", type=int, default=0,
                   help="0 = generate scenes in-process on the training device "
                        "(GPU-native, default); >0 = CPU dataloader workers")
    _add_common(p)

    p = sub.add_parser("calibrate", help="temperature scaling on holdout scenes")
    p.add_argument("--run", required=True)
    p.add_argument("--n", type=int, default=None)

    p = sub.add_parser("infer", help="run on a Measurement Set")
    p.add_argument("--ms", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--write-flags", action="store_true")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--data-column", default="auto",
                   help="DATA | CORRECTED_DATA | auto")
    p.add_argument("--subtract-model", action="store_true",
                   help="subtract MODEL_DATA before detection")
    p.add_argument("--device", default=None)
    p.add_argument("--max-scans", type=int, default=None)

    p = sub.add_parser("validate", help="label-free validation campaign")
    p.add_argument("which", choices=["injection", "coincidence", "residuals", "report"])
    p.add_argument("--ms", default=None)
    p.add_argument("--run", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--sidecar", default=None)

    args = ap.parse_args(argv)

    if args.cmd == "simulate":
        return _simulate(args)
    if args.cmd == "train":
        from absol.training.loop import train
        out = train(args.sim_config, args.array_config, args.model_config, args.out,
                    overrides=parse_overrides(args.override),
                    overfit_one=args.overfit_one, steps=args.steps,
                    num_workers=args.workers)
        print(f"run dir: {out}")
        return 0
    if args.cmd == "calibrate":
        from absol.training.calibrate import calibrate
        res = calibrate(args.run, n_scenes=args.n)
        print(json.dumps(res, indent=2))
        return 0
    if args.cmd == "infer":
        from absol.inference.run import run_inference
        sidecar = run_inference(
            args.ms, args.run, write_flags=args.write_flags,
            threshold=args.threshold, data_column=args.data_column,
            subtract_model=args.subtract_model, device=args.device,
            max_scans=args.max_scans,
        )
        print(f"sidecar: {sidecar}")
        return 0
    if args.cmd == "validate":
        cfg = load_yaml(args.cfg)
        if args.which == "injection":
            from absol.validation.injection import run_injection_test
            res = run_injection_test(args.ms, args.run, cfg)
            print(json.dumps(res.recall, indent=2))
        elif args.which == "coincidence":
            from absol.validation.coincidence import run_coincidence
            sidecar = args.sidecar or str(args.ms).rstrip("/") + ".absol.h5"
            res = run_coincidence(sidecar, cfg, args.run + "/validation")
            print(f"satellite: {res.satellite_sigma} sigma; "
                  f"powerline: {res.powerline_chi2_sigma} sigma")
        elif args.which == "residuals":
            from absol.validation.residual_stats import run_residual_stats
            sidecar = args.sidecar or str(args.ms).rstrip("/") + ".absol.h5"
            res = run_residual_stats(args.ms, args.run, sidecar, cfg)
            print(json.dumps({k: v for k, v in res.__dict__.items() if k != "figure"},
                             indent=2, default=str))
        else:
            from absol.validation.report import render_report
            out = render_report(args.run, cfg)
            print(f"report: {out}")
        return 0
    return 1


def _simulate(args) -> int:
    from pathlib import Path

    import h5py

    from absol.simulator.scenes import SceneGenerator
    from absol.training.loop import _array_from_raw, load_configs

    sim_cfg, array_raw, _ = load_configs(
        args.sim_config, args.array_config, args.sim_config,
        parse_overrides(args.override),
    )
    array = _array_from_raw(array_raw, Path(args.out).parent)
    gen = SceneGenerator(sim_cfg, array, seed=args.seed)
    with h5py.File(args.out, "w") as h5:
        for k in range(args.n):
            s = gen.sample(args.stage)
            g = h5.create_group(f"scene_{k:05d}")
            g.create_dataset("vis", data=s.vis.numpy(), compression="gzip")
            g.create_dataset("weights_in", data=s.weights_in.numpy(), compression="gzip")
            g.create_dataset("truth_mask", data=s.truth_mask.numpy(), compression="gzip")
            g.create_dataset("protected_mask", data=s.protected_mask.numpy(),
                             compression="gzip")
            g.create_dataset("truth_antennas", data=s.truth_antennas.numpy())
            g.create_dataset("ha_rad", data=s.ha_rad)
            g.create_dataset("freqs_hz", data=s.freqs_hz)
            g.attrs["dec_rad"] = s.dec_rad
            g.attrs["meta"] = json.dumps(s.meta)
            g.attrs["antenna_names"] = list(s.array.names)
            print(f"scene {k + 1}/{args.n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
