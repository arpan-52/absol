"""RFI mechanism library.

Each mechanism samples an :class:`RFIEvent`: a dimensionless envelope
``E(t, nu) >= 0`` (arbitrary scale; the scene generator rescales it to the
target strength in thermal-sigma units), a direction, a coupling mode and
per-antenna couplings ``a_k(t)``. The scene generator turns this into the
additive visibility term

    R_ij(t, nu) = a_i(t) a_j(t) * E(t, nu) * exp(i phi_ij(t, nu)) * pol_p

with the phase derived from `geometry` (single source of truth).

v0 DIRECTION MODEL (documented approximation): all directional RFI is
modelled as a source at its instantaneous celestial (ha, dec), i.e. with the
residual fringe rate `geometry.fringe_rate` predicts for that direction.
For a transmitter truly fixed on the ground the geometric delay is constant
and only the (baseline-dependent) phase-centre tracking rate survives; v0
trades that exactness for a single self-consistent rate model shared by the
simulator, graph relation B and the validators. Revisit in v1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from absol.geometry import Array
from absol.simulator.gains import smooth_noise


@dataclass
class RFIEvent:
    name: str
    envelope: np.ndarray                  # [T, F] float32 >= 0
    pol_factors: tuple[float, float]
    az_deg: np.ndarray | None             # [T] (constant for static sources)
    el_deg: np.ndarray | None
    coupling_mode: str
    coupled_mask: np.ndarray              # [A] bool
    couplings: np.ndarray                 # [A, T] float32 (0 for uncoupled)
    zero_fringe: bool = False
    at_phase_centre: bool = False
    protected: bool = False
    meta: dict = field(default_factory=dict)


def _sample_couplings(
    array: Array,
    n_time: int,
    rng: np.random.Generator,
    cfg_rfi: dict,
    integration_s: float,
    force_mode: str | None = None,
    bias_enu: np.ndarray | None = None,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Coupling mode, coupled mask [A], couplings a_k(t) [A, T].

    Per-antenna base amplitudes are log-uniform over ``coupling_decades``
    (normalized to max 1) with smooth positive modulation of timescale
    60-600 s. ``bias_enu`` biases subset membership toward antennas nearest
    a ground point (1/d^2 weights) - central square ends up worst, as real.
    """
    a = array.n_ant
    modes = cfg_rfi["coupling_mode_probs"]
    if force_mode is not None:
        mode = force_mode
    else:
        names = list(modes.keys())
        p = np.array([modes[k] for k in names], dtype=float)
        mode = names[rng.choice(len(names), p=p / p.sum())]

    mask = np.zeros(a, dtype=bool)
    if mode == "all_antennas":
        mask[:] = True
    elif mode == "single":
        mask[rng.integers(0, a)] = True
    else:  # subset
        k = int(rng.integers(2, max(3, a // 3) + 1))
        if bias_enu is not None:
            d2 = ((array.enu - bias_enu[None, :]) ** 2).sum(1)
            w = 1.0 / (d2 + 1e4)        # +100 m softening
            w /= w.sum()
            mask[rng.choice(a, size=min(k, a), replace=False, p=w)] = True
        else:
            mask[rng.choice(a, size=min(k, a), replace=False)] = True

    decades = float(cfg_rfi["coupling_decades"])
    base = np.zeros(a)
    base[mask] = 10.0 ** (-rng.random(mask.sum()) * decades)
    if base.max() > 0:
        base /= base.max()

    ts = cfg_rfi.get("modulation_timescale_s", {"min": 60, "max": 600})
    tau = rng.uniform(float(ts["min"]), float(ts["max"])) / integration_s
    mod = np.exp(0.4 * smooth_noise(a, n_time, tau, rng))
    coup = (base[:, None] * mod).astype(np.float32)
    coup[~mask] = 0.0
    return mode, mask, coup


def _gauss_band(freqs: np.ndarray, f0: float, bw: float) -> np.ndarray:
    return np.exp(-0.5 * ((freqs - f0) / (bw / 2.355)) ** 2)


def _telegraph(n_time: int, rng: np.random.Generator, mean_seg: float = 10.0) -> np.ndarray:
    """Random on/off time profile, smoothed by one dump."""
    out = np.zeros(n_time)
    t, state = 0, bool(rng.random() < 0.5)
    while t < n_time:
        seg = max(1, int(rng.exponential(mean_seg)))
        out[t:t + seg] = float(state)
        t += seg
        state = not state
    if out.max() == 0:
        out[rng.integers(0, n_time):] = 1.0
    k = np.array([0.25, 0.5, 0.25])
    return np.convolve(out, k, mode="same")


def ground_narrowband(
    array: Array, times_s: np.ndarray, freqs_hz: np.ndarray, rng: np.random.Generator, cfg_rfi: dict
) -> RFIEvent:
    """Fixed terrestrial transmitter: az free, el <= 10 deg; 0.05-5 MHz wide;
    persistent or slow on/off; subset coupling biased toward antennas nearest
    a random ground point within ~20 km."""
    n_t = times_s.size
    az = float(rng.uniform(0, 360))
    el = float(rng.uniform(0.5, 10.0))
    f0 = float(rng.uniform(freqs_hz.min(), freqs_hz.max()))
    bw = float(np.exp(rng.uniform(np.log(0.05e6), np.log(5e6))))
    spec = _gauss_band(freqs_hz, f0, bw)
    tprof = np.ones(n_t) if rng.random() < 0.5 else _telegraph(n_t, rng)
    ground = rng.normal(0.0, 7000.0, size=3)
    ground[2] = 0.0
    mode, mask, coup = _sample_couplings(
        array, n_t, rng, cfg_rfi, times_s[1] - times_s[0] if n_t > 1 else 8.0,
        force_mode="subset", bias_enu=ground,
    )
    env = (tprof[:, None] * spec[None, :]).astype(np.float32)
    return RFIEvent(
        name="ground_narrowband", envelope=env, pol_factors=(1.0, float(rng.uniform(0.5, 1.0))),
        az_deg=np.full(n_t, az), el_deg=np.full(n_t, el),
        coupling_mode=mode, coupled_mask=mask, couplings=coup,
        meta={"f0_hz": f0, "bw_hz": bw},
    )


def powerline_arcing(
    array: Array, times_s: np.ndarray, freqs_hz: np.ndarray, rng: np.random.Generator, cfg_rfi: dict
) -> RFIEvent:
    """Broadband (>= 50 MHz) impulsive arcing, amplitude-modulated at 100 Hz.

    100 Hz is far above the dump rate, so per-dump it appears as raised power
    with a random duty cycle; the true modulation flag is stored in ``meta``
    for the coincidence validator. Strongly polarized (2-10x imbalance).
    """
    n_t = times_s.size
    span = freqs_hz.max() - freqs_hz.min()
    bw = float(rng.uniform(min(50e6, span), span))
    f0 = float(rng.uniform(freqs_hz.min() + bw / 4, freqs_hz.max() - bw / 4))
    spec = _gauss_band(freqs_hz, f0, 2 * bw)
    ripple = np.exp(0.5 * smooth_noise(1, freqs_hz.size, 100.0, rng)[0])
    spec = spec * ripple
    tprof = np.zeros(n_t)
    for _ in range(1 + rng.poisson(2)):
        t0 = rng.integers(0, n_t)
        dur = int(rng.integers(1, 6))
        duty = rng.uniform(0.05, 0.5)
        tprof[t0:t0 + dur] += duty * (1 + 0.5 * np.abs(rng.standard_normal(min(dur, n_t - t0))))
    r = float(rng.uniform(2.0, 10.0))
    pol = (1.0, 1.0 / r) if rng.random() < 0.5 else (1.0 / r, 1.0)
    ground = rng.normal(0.0, 7000.0, size=3)
    ground[2] = 0.0
    mode, mask, coup = _sample_couplings(
        array, n_t, rng, cfg_rfi, times_s[1] - times_s[0] if n_t > 1 else 8.0,
        force_mode="subset", bias_enu=ground,
    )
    return RFIEvent(
        name="powerline_arcing", envelope=(tprof[:, None] * spec[None, :]).astype(np.float32),
        pol_factors=pol, az_deg=np.full(n_t, float(rng.uniform(0, 360))),
        el_deg=np.full(n_t, float(rng.uniform(0.5, 5.0))),
        coupling_mode=mode, coupled_mask=mask, couplings=coup,
        meta={"modulated_100hz": True},
    )


def satellite(
    array: Array, times_s: np.ndarray, freqs_hz: np.ndarray, rng: np.random.Generator, cfg_rfi: dict
) -> RFIEvent:
    """Direction sweeps linearly in az/el (0.01-0.5 deg/s); 1-30 MHz band;
    all-antenna coupling; fringe rate recomputed along the track."""
    n_t = times_s.size
    speed = rng.uniform(0.01, 0.5)
    psi = rng.uniform(0, 2 * np.pi)
    az = (rng.uniform(0, 360) + speed * np.cos(psi) * (times_s - times_s[0])) % 360.0
    el = rng.uniform(10, 70) + speed * np.sin(psi) * (times_s - times_s[0])
    above = el > 0.5
    bw = float(rng.uniform(1e6, 30e6))
    f0 = float(rng.uniform(freqs_hz.min(), freqs_hz.max()))
    spec = _gauss_band(freqs_hz, f0, bw)
    dt = times_s[1] - times_s[0] if n_t > 1 else 8.0
    tprof = np.exp(0.5 * smooth_noise(1, n_t, 120.0 / dt, rng)[0]) * above
    mode, mask, coup = _sample_couplings(
        array, n_t, rng, cfg_rfi, dt, force_mode="all_antennas",
    )
    return RFIEvent(
        name="satellite", envelope=(tprof[:, None] * spec[None, :]).astype(np.float32),
        pol_factors=(1.0, float(rng.uniform(0.7, 1.0))),
        az_deg=az, el_deg=np.clip(el, 0.0, 90.0),
        coupling_mode=mode, coupled_mask=mask, couplings=coup,
        meta={"track_deg_s": speed},
    )


def pulsed(
    array: Array, times_s: np.ndarray, freqs_hz: np.ndarray, rng: np.random.Generator, cfg_rfi: dict
) -> RFIEvent:
    """Radar/DME-like ms pulses: per-dump duty cycle 0.1-10 %, narrowband or
    hopping across a 5-50 MHz span."""
    n_t, n_f = times_s.size, freqs_hz.size
    duty = float(np.exp(rng.uniform(np.log(1e-3), np.log(0.1))))
    power = duty * np.clip(1 + 0.7 * rng.standard_normal(n_t), 0.0, None)
    env = np.zeros((n_t, n_f), dtype=np.float32)
    if rng.random() < 0.5:                       # hopping
        span = rng.uniform(5e6, 50e6)
        fc = rng.uniform(freqs_hz.min() + span / 2, freqs_hz.max() - span / 2)
        width = rng.uniform(0.2e6, 2e6)
        for t in range(n_t):
            env[t] = power[t] * _gauss_band(freqs_hz, fc + rng.uniform(-span / 2, span / 2), width)
        hop = True
    else:
        f0 = rng.uniform(freqs_hz.min(), freqs_hz.max())
        width = rng.uniform(0.05e6, 1e6)
        env[:] = power[:, None] * _gauss_band(freqs_hz, f0, width)[None, :]
        hop = False
    mode, mask, coup = _sample_couplings(
        array, n_t, rng, cfg_rfi, times_s[1] - times_s[0] if n_t > 1 else 8.0,
    )
    return RFIEvent(
        name="pulsed", envelope=env, pol_factors=(1.0, float(rng.uniform(0.5, 1.0))),
        az_deg=np.full(n_t, float(rng.uniform(0, 360))),
        el_deg=np.full(n_t, float(rng.uniform(0.5, 20.0))),
        coupling_mode=mode, coupled_mask=mask, couplings=coup,
        meta={"duty": duty, "hopping": hop},
    )


def internal_zero_fringe(
    array: Array, times_s: np.ndarray, freqs_hz: np.ndarray, rng: np.random.Generator, cfg_rfi: dict
) -> RFIEvent:
    """Correlated digital leakage: fringe rate identically 0, channel comb
    every k-th channel, subset coupling. Exists to break the
    'RFI != 0 fringe rate' shortcut."""
    n_t, n_f = times_s.size, freqs_hz.size
    k = int(rng.choice([32, 64, 128, 256]))
    k = min(k, max(2, n_f // 4))
    off = int(rng.integers(0, k))
    spec = np.zeros(n_f)
    teeth = np.arange(off, n_f, k)
    spec[teeth] = rng.uniform(0.5, 1.5, size=teeth.size)
    dt = times_s[1] - times_s[0] if n_t > 1 else 8.0
    tprof = np.exp(0.3 * smooth_noise(1, n_t, 200.0 / dt, rng)[0])
    mode, mask, coup = _sample_couplings(array, n_t, rng, cfg_rfi, dt, force_mode="subset")
    return RFIEvent(
        name="internal_zero_fringe", envelope=(tprof[:, None] * spec[None, :]).astype(np.float32),
        pol_factors=(1.0, 1.0), az_deg=None, el_deg=None, zero_fringe=True,
        coupling_mode=mode, coupled_mask=mask, couplings=coup,
        meta={"comb_period": k},
    )


def transient_protected(
    array: Array, times_s: np.ndarray, freqs_hz: np.ndarray, rng: np.random.Generator, cfg_rfi: dict
) -> RFIEvent:
    """Celestial dispersed transient AT the phase centre: cold-plasma nu^-2
    delay t(nu) = 4.149 ms * DM * (nu/GHz)^-2, DM 50-2000. Fringe-stopped, so
    phase = 0; truth label = NOT contaminated (the model must not flag it)."""
    n_t = times_s.size
    dm = float(rng.uniform(50, 2000))
    dt = times_s[1] - times_s[0] if n_t > 1 else 8.0
    t0 = times_s[0] + rng.uniform(-0.2, 0.8) * (times_s[-1] - times_s[0] + dt)
    width = rng.uniform(0.5, 2.0) * dt
    t_arr = t0 + 4.149e-3 * dm * (freqs_hz / 1e9) ** -2          # [F] seconds
    env = np.exp(-0.5 * ((times_s[:, None] - t_arr[None, :]) / width) ** 2)
    spec = (freqs_hz / freqs_hz.mean()) ** rng.normal(-1.5, 0.5)
    a = array.n_ant
    return RFIEvent(
        name="transient_protected", envelope=(env * spec[None, :]).astype(np.float32),
        pol_factors=(1.0, 1.0), az_deg=None, el_deg=None, at_phase_centre=True,
        coupling_mode="all_antennas", coupled_mask=np.ones(a, dtype=bool),
        couplings=np.ones((a, n_t), dtype=np.float32), protected=True,
        meta={"dm": dm, "t0_s": float(t0)},
    )


MECHANISMS = {
    "ground_narrowband": ground_narrowband,
    "powerline_arcing": powerline_arcing,
    "satellite": satellite,
    "pulsed": pulsed,
    "internal_zero_fringe": internal_zero_fringe,
    "transient_protected": transient_protected,
}
