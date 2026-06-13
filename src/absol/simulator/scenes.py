"""Scenario generator: full scenes -> training samples.

Physics model per scene, in the fringe-stopped frame (spec section 5.2):

    V_ij(t,nu) = g_i g_j^* [ sky_ij(t,nu) + n_ij(t,nu) + R_ij(t,nu) ]

- sky: point sources via `simulator.sky` (phase-centre source constant).
- noise: complex Gaussian, per-component std from the radiometer equation
      sigma_ij = sqrt(SEFD_i SEFD_j / (2 dnu t_int))    [Jy]
  with per-antenna SEFD jitter.
- RFI: sum over mechanism events m of
      a_i(t) a_j(t) E_m(t,nu) exp(i phi_ij(t,nu)) pol_p
  phases from `geometry` (see `event_contribution`); strength is calibrated
  so that the median |R| over the event's active samples equals
  strength_sigma x median(sigma_ij).
- truth mask: |sum of non-protected R| > truth.mask_threshold_sigma x sigma_ij.

v0 simplification: gains multiply everything incl. RFI (see gains.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from absol.geometry import (
    C_LIGHT,
    OMEGA_EARTH,
    Array,
    azel_to_hadec,
    fringe_rate,
    w_metres,
)
from absol.simulator.gains import antenna_gains, apply_gains
from absol.simulator.rfi import MECHANISMS, RFIEvent
from absol.simulator.sky import sample_sky, sky_visibilities


@dataclass
class Scene:
    """All tensors of the data contract (section 4) + metadata."""

    vis: torch.Tensor             # complex64 [B, T, F, P]
    weights_in: torch.Tensor      # float32  [B, T, F]; 0 = pre-flagged
    truth_mask: torch.Tensor      # bool     [B, T, F]
    protected_mask: torch.Tensor  # bool     [B, T, F]
    truth_antennas: torch.Tensor  # bool     [A]
    array: Array                  # surviving sub-array
    antenna_present: np.ndarray   # bool [A_full]
    ha_rad: np.ndarray            # [T] phase-centre hour angle
    dec_rad: float
    freqs_hz: np.ndarray          # [F]
    sigma_thermal: torch.Tensor   # float32 [B] per-component noise std (Jy)
    meta: dict


def event_contribution(
    event: RFIEvent,
    array: Array,
    ha_rad: np.ndarray,
    dec_rad: float,
    freqs_hz: np.ndarray,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Unpolarized RFI visibility term [B, T, F] complex64 (unit pol factor).

    Phase model (all rates from `geometry` - single source of truth):
    - directional: instantaneous (ha_s, dec_s) from the event az/el track;
      fringe phase integrates fringe_rate over time, scaled linearly in nu;
      a constant per-baseline delay phase 2 pi (nu/c) w_s gives the
      closure-consistent delay structure.
    - zero_fringe: random constant per-baseline phase.
    - at_phase_centre: phase 0 (fringe-stopped).
    """
    pairs = array.baselines()
    n_b = pairs.shape[0]
    n_t, n_f = ha_rad.size, freqs_hz.size
    nu0 = float(freqs_hz.mean())
    env = torch.as_tensor(event.envelope, dtype=torch.float32, device=device)  # [T, F]
    coup = torch.as_tensor(event.couplings, dtype=torch.float32, device=device)  # [A, T]
    aiaj = coup[pairs[:, 0]] * coup[pairs[:, 1]]                                 # [B, T]

    if event.at_phase_centre:
        phase = torch.zeros((n_b, n_t, 1), dtype=torch.float32, device=device)
        scale_f = torch.ones(n_f, dtype=torch.float32, device=device)
    elif event.zero_fringe:
        phi0 = torch.as_tensor(rng.uniform(0, 2 * np.pi, n_b), dtype=torch.float32, device=device)
        phase = phi0[:, None, None].expand(n_b, n_t, 1).contiguous()
        scale_f = torch.ones(n_f, dtype=torch.float32, device=device)
    else:
        dt = (ha_rad[1] - ha_rad[0]) / OMEGA_EARTH if n_t > 1 else 8.0
        rates = np.empty((n_t, n_b))
        w_s = np.empty((n_t, n_b))
        for t in range(n_t):
            ha_s, dec_s = azel_to_hadec(
                np.deg2rad(event.az_deg[t]), np.deg2rad(event.el_deg[t]), array.lat_rad
            )
            rates[t] = fringe_rate(array, ha_s, dec_s, nu0, float(ha_rad[t]), dec_rad)
            w_s[t] = w_metres(array, ha_s, dec_s)
        # integrate the (slowly varying) rate; remove the mean to keep phases small
        phi_t = 2 * np.pi * np.cumsum(rates - rates.mean(0, keepdims=True) * 0.0, axis=0) * dt
        phi_t += 2 * np.pi * (nu0 / C_LIGHT) * w_s          # delay structure (mod 2 pi)
        phase = torch.as_tensor(phi_t.T[:, :, None], dtype=torch.float32, device=device)
        scale_f = torch.as_tensor(freqs_hz / nu0, dtype=torch.float32, device=device)

    out = torch.empty((n_b, n_t, n_f), dtype=torch.complex64, device=device)
    # phase[b,t,1] * scale_f[f] -> full phase; chunk over baselines to bound memory
    step = max(1, int(4e7 // max(n_t * n_f, 1)))
    for b0 in range(0, n_b, step):
        sl = slice(b0, min(b0 + step, n_b))
        ph = phase[sl] * scale_f[None, None, :]
        out[sl] = (aiaj[sl][:, :, None] * env[None]) * torch.exp(1j * ph)
    return out


class SceneGenerator:
    """Curriculum-gated random scene sampler. Picklable / worker-safe:
    call :meth:`reseed` with a per-worker offset before iterating."""

    def __init__(self, sim_cfg: dict, array: Array, seed: int = 0, device: str = "cpu"):
        self.cfg = sim_cfg
        self.array = array
        self.seed = int(seed)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(self.seed)

    def reseed(self, offset: int) -> None:
        self.rng = np.random.default_rng(self.seed + 1_000_003 * (offset + 1))

    # ---------------- curriculum gating ----------------
    def _stage_params(self, stage: int) -> dict:
        cfg = self.cfg
        s = int(np.clip(stage, 1, 4))
        all_w = cfg["rfi"]["mechanisms"]
        allowed = {
            1: ["ground_narrowband", "powerline_arcing"],
            2: ["ground_narrowband", "powerline_arcing", "satellite", "pulsed"],
            3: [m for m in all_w if m != "transient_protected"],
            4: list(all_w.keys()),
        }[s]
        allowed = [m for m in allowed if m in all_w] or list(all_w.keys())
        smin = {1: 5.0, 2: 1.0}.get(s, float(cfg["rfi"]["strength_sigma"]["min"]))
        n_max = {1: 1, 2: 3}.get(s, int(cfg["scene"]["n_rfi_sources"]["max"]))
        n_min = 1 if s == 1 else int(cfg["scene"]["n_rfi_sources"]["min"])
        drop_max = {1: 0.0, 2: 0.5 * float(cfg["antenna_dropout"]["max_frac"])}.get(
            s, float(cfg["antenna_dropout"]["max_frac"])
        )
        clean = {1: 0.0, 2: 0.1}.get(s, float(cfg["scene"]["clean_scene_fraction"]))
        return dict(
            mechanisms={m: all_w[m] for m in allowed},
            strength_min=smin, n_rfi=(n_min, n_max), drop_max=drop_max,
            clean_frac=clean, flag_aug=(s >= 2),
        )

    # ---------------- main entry ----------------
    def sample(self, stage: int) -> Scene:
        cfg, rng, dev = self.cfg, self.rng, self.device
        sp = self._stage_params(stage)
        full = self.array
        t_int = full.integration_s
        n_t = int(cfg["scene"]["n_time"])
        times = np.arange(n_t) * t_int

        # antenna dropout (graph is built on the surviving subset)
        frac = rng.uniform(float(cfg["antenna_dropout"]["min_frac"]), sp["drop_max"])
        keep = rng.random(full.n_ant) >= frac
        while keep.sum() < min(6, full.n_ant):
            keep[rng.integers(0, full.n_ant)] = True
        array = full.subset(keep) if not keep.all() else full
        a, pairs = array.n_ant, array.baselines()
        n_b = pairs.shape[0]
        freqs = array.freqs_hz
        n_f = freqs.size

        dec = float(np.deg2rad(rng.uniform(
            cfg["scene"]["declination_deg"]["min"], cfg["scene"]["declination_deg"]["max"]
        )))
        ha0 = rng.uniform(-np.pi / 6, np.pi / 12)
        ha = ha0 + OMEGA_EARTH * times

        # ---- sky + noise + gains ----
        sky = sample_sky(cfg["sky"], rng)
        vis_sky = sky_visibilities(array, sky, ha, dec, freqs, dev)       # [B,T,F]
        sefd = float(cfg["noise"]["sefd_jy"]) * (
            1 + float(cfg["noise"].get("sefd_jitter_frac", 0.0)) * rng.standard_normal(a)
        )
        sefd = np.clip(sefd, 50.0, None)
        dnu = array.chan_width_hz
        sigma = np.sqrt(sefd[pairs[:, 0]] * sefd[pairs[:, 1]] / (2.0 * dnu * t_int))  # [B]
        sig_t = torch.as_tensor(sigma, dtype=torch.float32, device=dev)
        n_p = array.n_pol
        noise = sig_t[:, None, None, None] * torch.view_as_complex(
            torch.as_tensor(
                rng.standard_normal((n_b, n_t, n_f, n_p, 2)), dtype=torch.float32
            ).to(dev)
        )
        vis = vis_sky[..., None] + noise
        del noise, vis_sky

        # ---- RFI ----
        truth = torch.zeros((n_b, n_t, n_f), dtype=torch.bool, device=dev)
        prot = torch.zeros_like(truth)
        truth_ant = np.zeros(a, dtype=bool)
        rfi_sum = torch.zeros((n_b, n_t, n_f, n_p), dtype=torch.complex64, device=dev)
        prot_sum = torch.zeros_like(rfi_sum)
        events_meta: list[dict] = []
        max_strength = 0.0

        n_rfi = 0 if rng.random() < sp["clean_frac"] else int(
            rng.integers(sp["n_rfi"][0], sp["n_rfi"][1] + 1)
        )
        mech_names = list(sp["mechanisms"].keys())
        mech_w = np.array([sp["mechanisms"][m] for m in mech_names], dtype=float)
        mech_w /= mech_w.sum()
        s_cfg = cfg["rfi"]["strength_sigma"]

        for _ in range(n_rfi):
            name = mech_names[rng.choice(len(mech_names), p=mech_w)]
            event = MECHANISMS[name](array, times, freqs, rng, cfg["rfi"])
            if event.envelope.max() <= 0:
                continue
            contrib = event_contribution(event, array, ha, dec, freqs, rng, dev)
            strength = float(np.exp(rng.uniform(
                np.log(max(sp["strength_min"], float(s_cfg["min"]))), np.log(float(s_cfg["max"]))
            )))
            # calibrate: median |R| over active samples = strength x median sigma
            active = contrib.abs() > 1e-8
            n_act = int(active.sum())
            if n_act == 0:
                continue
            vals = contrib.abs()[active]
            if vals.numel() > 1_000_000:
                idx = torch.randint(0, vals.numel(), (1_000_000,), device=dev)
                vals = vals[idx]
            med = float(vals.median())
            scale = strength * float(np.median(sigma)) / max(med, 1e-30)
            contrib *= scale
            pf = torch.as_tensor(event.pol_factors, dtype=torch.float32, device=dev)
            term = contrib[..., None] * pf
            if event.protected:
                prot_sum += term
            else:
                rfi_sum += term
                truth_ant |= event.coupled_mask
            del contrib, term
            max_strength = max(max_strength, strength)
            events_meta.append({
                "name": name, "strength_sigma": strength,
                "coupling_mode": event.coupling_mode,
                "az_deg": None if event.az_deg is None else float(event.az_deg[0]),
                "el_deg": None if event.el_deg is None else float(event.el_deg[0]),
                "protected": event.protected, **event.meta,
            })

        thr = float(cfg["truth"]["mask_threshold_sigma"]) * sig_t[:, None, None]
        truth = rfi_sum.abs().amax(-1) > thr
        prot = prot_sum.abs().amax(-1) > thr
        vis = vis + rfi_sum + prot_sum
        del rfi_sum, prot_sum

        g = antenna_gains(
            a, n_t, float(cfg["gains"]["amp_rms"]), float(cfg["gains"]["phase_rms_deg"]),
            float(cfg["gains"]["timescale_s"]), t_int, rng, dev,
        )
        vis = apply_gains(vis, g, pairs)

        # ---- pre-existing flag augmentation ----
        weights = torch.ones((n_b, n_t, n_f), dtype=torch.float32, device=dev)
        if sp["flag_aug"]:
            fa = cfg["flag_augmentation"]
            if rng.random() < float(fa["scattered_p"]):
                p = rng.uniform(0.005, 0.05)
                weights *= (torch.as_tensor(rng.random((n_b, n_t, n_f))) >= p).float().to(dev)
            if rng.random() < float(fa["contiguous_p"]):
                if rng.random() < 0.5:
                    t0 = rng.integers(0, n_t)
                    w = int(rng.integers(1, max(2, n_t // 8)))
                    weights[:, t0:t0 + w, :] = 0.0
                else:
                    f0 = rng.integers(0, n_f)
                    w = int(rng.integers(5, max(6, n_f // 8)))
                    weights[:, :, f0:f0 + w] = 0.0
            if rng.random() < float(fa["block_p"]):
                nb = max(1, int(0.03 * n_b))
                weights[rng.choice(n_b, nb, replace=False)] = 0.0

        return Scene(
            vis=vis, weights_in=weights, truth_mask=truth, protected_mask=prot,
            truth_antennas=torch.as_tensor(truth_ant), array=array,
            antenna_present=keep, ha_rad=ha, dec_rad=dec, freqs_hz=freqs,
            sigma_thermal=sig_t,
            meta={
                "stage": int(stage), "events": events_meta, "n_rfi": n_rfi,
                "max_strength_sigma": max_strength, "dec_deg": float(np.rad2deg(dec)),
                "n_sky_sources": int(sky.l.size),
            },
        )
