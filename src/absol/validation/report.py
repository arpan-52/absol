"""Aggregate all validation metrics into ``validation_report.md``."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path


def render_report(run_dir: str, cfg: dict, out_path: str | None = None) -> Path:
    run = Path(run_dir)
    vdir = run / "validation"
    out = Path(out_path or run / "validation_report.md")
    acc = cfg.get("acceptance", {})

    def _load(name: str) -> dict | None:
        p = vdir / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    inj = _load("injection")
    noise = _load("noise_curve")
    coin = _load("coincidence")
    resid = _load("residual_stats")

    lines = [
        "# ABSOL validation report",
        f"run: `{run}` — generated {date.today().isoformat()}",
        "",
        "## Pre-registered acceptance table",
        "",
        "| Criterion | Target | Measured | Status |",
        "|---|---|---|---|",
    ]

    def row(name, target, measured, ok):
        status = "—" if ok is None else ("PASS" if ok else "FAIL")
        lines.append(f"| {name} | {target} | {measured} | {status} |")

    # injection: recall in the 0.5-2 sigma bin vs AOFlagger margin
    margin = acc.get("injection_recall_margin_vs_aoflagger", 0.15)
    if inj:
        s = inj["strengths_sigma"]
        sel = [k for k, v in enumerate(s) if 0.5 <= v <= 2.0]
        vals = [inj["recall"][m][k] for m in inj["recall"] for k in sel
                if inj["recall"][m][k] == inj["recall"][m][k]]
        meas = f"{sum(vals) / len(vals):.2f} (mean recall 0.5-2σ)" if vals else "n/a"
        row(f"recall@{inj['fpr_operating_point']:.0%}FPR ≥ AOFlagger + {margin}",
            f"+{margin} margin", meas, None)
    else:
        row("injection recall", f"AOFlagger + {margin}", "not run", None)

    factor = acc.get("noise_curve_tracking_factor", 4.0)
    if noise and noise["labels"]:
        meas = ", ".join(f"{k}: {v:.0f}s" for k, v in noise["departure_time_s"].items())
        row(f"thermal 1/sqrt(t) tracking ≥ {factor}x longer", f"{factor}x", meas, None)
    else:
        row("noise-curve tracking", f"{factor}x longer", "not run", None)

    sat_t = acc.get("satellite_coincidence_sigma", 5.0)
    if coin:
        sig = coin["satellite_sigma"]
        ok = None if sig != sig else sig >= sat_t
        row("satellite coincidence", f"≥ {sat_t}σ",
            "n/a" if sig != sig else f"{sig:.1f}σ", ok)
        row("powerline fold", "reported",
            f"{coin['powerline_chi2_sigma']:.1f}σ" if coin["powerline_chi2_sigma"] ==
            coin["powerline_chi2_sigma"] else "n/a", None)
    else:
        row("satellite coincidence", f"≥ {sat_t}σ", "not run", None)

    row("data loss", acc.get("data_loss_constraint", "≤ default pipeline"),
        "external imaging comparison", None)

    if resid:
        lines += [
            "",
            "## Residual statistics (post-weighting)",
            f"- SK excess central / arm: {resid['sk_excess_central']:.3f} / "
            f"{resid['sk_excess_arm']:.3f}",
            f"- off-fringe fraction central / arm: {resid['offfringe_central']:.3f} / "
            f"{resid['offfringe_arm']:.3f}",
        ] + [f"- NOTE: {n}" for n in resid.get("notes", [])]

    lines.append("")
    lines.append("## Figures")
    for r in (inj, noise, coin, resid):
        if r and r.get("figure"):
            lines.append(f"![]({r['figure']})")
    for r in (coin,):
        if r:
            for c in r.get("caveats", []):
                lines.append(f"- caveat: {c}")

    out.write_text("\n".join(lines) + "\n")
    return out
