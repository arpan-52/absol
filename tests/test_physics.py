"""Physics invariants (spec tests 1-5) - the most important tests."""
from __future__ import annotations

import numpy as np
import torch

from absol.features import delay_fringe_tiles, physics_features
from absol.geometry import OMEGA_EARTH, azel_to_hadec, fringe_rate
from absol.simulator.rfi import RFIEvent, internal_zero_fringe
from absol.simulator.scenes import SceneGenerator, event_contribution
from absol.simulator.sky import SkySources, sky_visibilities


def test_1_phase_centre_source_constant(tiny_array):
    """Phase-centre source => |V| constant in time to numerical precision."""
    src = SkySources(l=np.zeros(1), m=np.zeros(1), flux_jy=np.array([1.7]),
                     spec_index=np.zeros(1))
    ha = -0.3 + OMEGA_EARTH * np.arange(50) * 8.0
    vis = sky_visibilities(tiny_array, src, ha, np.deg2rad(20.0),
                           tiny_array.freqs_hz, torch.device("cpu"))
    assert torch.allclose(vis, torch.full_like(vis, 1.7 + 0j), atol=1e-4)
    assert float(vis.abs().std()) < 1e-5


def test_2_noise_sigma_radiometer(tiny_array, tiny_sim_cfg):
    """Simulated noise sigma matches the radiometer equation within 2%."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in tiny_sim_cfg.items()}
    cfg["sky"] = dict(cfg["sky"], n_sources={"min": 0, "max": 0})
    cfg["scene"] = dict(cfg["scene"], clean_scene_fraction=1.0)
    cfg["noise"] = {"sefd_jy": 390, "sefd_jitter_frac": 0.0}
    cfg["gains"] = {"amp_rms": 0.0, "phase_rms_deg": 0.0, "timescale_s": 300}
    cfg["antenna_dropout"] = {"min_frac": 0.0, "max_frac": 0.0}
    cfg["flag_augmentation"] = {"scattered_p": 0, "contiguous_p": 0, "block_p": 0}
    gen = SceneGenerator(cfg, tiny_array, seed=1)
    scene = gen.sample(stage=3)             # stage 3 honors clean_scene_fraction
    assert scene.meta["n_rfi"] == 0
    expected = 390.0 / np.sqrt(2.0 * tiny_array.chan_width_hz * 8.0)
    measured = scene.vis.real.std(dim=(1, 2, 3))      # per baseline
    assert torch.allclose(
        measured, torch.full_like(measured, expected), rtol=0.02
    ), f"measured {measured.mean():.4f} vs expected {expected:.4f}"


def _strong_event(array, n_t, n_f, az, el):
    return RFIEvent(
        name="test", envelope=np.ones((n_t, n_f), dtype=np.float32),
        pol_factors=(1.0, 1.0), az_deg=np.full(n_t, az), el_deg=np.full(n_t, el),
        coupling_mode="all_antennas", coupled_mask=np.ones(array.n_ant, bool),
        couplings=np.ones((array.n_ant, n_t), dtype=np.float32),
    )


def test_3_rfi_fringe_rate_matches_geometry(tiny_array):
    """FFT peak of simulated RFI matches geometry.fringe_rate within one bin."""
    n_t, t_int = 64, tiny_array.integration_s
    az, el, dec = 132.0, 7.0, np.deg2rad(25.0)
    ha = -0.1 + OMEGA_EARTH * np.arange(n_t) * t_int
    freqs = np.array([650e6])
    rng = np.random.default_rng(0)
    ev = _strong_event(tiny_array, n_t, 1, az, el)
    contrib = event_contribution(ev, tiny_array, ha, dec, freqs, rng,
                                 torch.device("cpu"))
    ha_s, dec_s = azel_to_hadec(np.deg2rad(az), np.deg2rad(el), tiny_array.lat_rad)
    mid = n_t // 2
    expected = fringe_rate(tiny_array, ha_s, dec_s, 650e6,
                           float(ha[mid]), dec)               # [B]
    fr_bins = np.fft.fftshift(np.fft.fftfreq(n_t, d=t_int))
    bin_w = fr_bins[1] - fr_bins[0]
    spec = np.fft.fftshift(np.abs(np.fft.fft(contrib[:, :, 0].numpy(), axis=1)), axes=1)
    measured = fr_bins[spec.argmax(axis=1)]
    # at 8 s dumps long-baseline rates alias: compare on the aliasing circle
    fs = 1.0 / t_int
    err = np.abs(((measured - expected) + fs / 2) % fs - fs / 2)
    assert np.all(err <= bin_w + 1e-9), (
        f"max err {err.max():.2e} Hz, bin {bin_w:.2e}"
    )


def test_4_zero_fringe_peaks_at_bin0(tiny_array, tiny_sim_cfg):
    """internal_zero_fringe mechanism peaks at exactly fringe-rate bin 0."""
    n_t = 32
    times = np.arange(n_t) * 8.0
    freqs = tiny_array.freqs_hz[:64]
    rng = np.random.default_rng(3)
    ev = internal_zero_fringe(tiny_array, times, freqs, rng, tiny_sim_cfg["rfi"])
    ha = OMEGA_EARTH * times
    contrib = event_contribution(ev, tiny_array, ha, np.deg2rad(10.0), freqs, rng,
                                 torch.device("cpu"))
    comb = np.flatnonzero(ev.envelope[0] > 0)
    b_sel = int(np.argmax(np.abs(contrib[:, 0, comb[0]]).numpy()))
    spec = np.abs(np.fft.fft(contrib[b_sel, :, comb[0]].numpy()))
    assert spec.argmax() == 0


def test_5_spectral_kurtosis(tiny_array):
    """SK ~ expectation for pure noise; significant excess for intermittent CW."""
    rng = np.random.default_rng(7)
    b, t_c, f_c, p = 6, 16, 128, 2
    pairs = tiny_array.baselines()[:b]
    noise = (rng.standard_normal((b, 1, 1, t_c, f_c, p))
             + 1j * rng.standard_normal((b, 1, 1, t_c, f_c, p))) / np.sqrt(2)
    chunks = torch.as_tensor(noise.astype(np.complex64))
    wch = torch.ones((b, 1, 1, t_c, f_c))
    tiles = delay_fringe_tiles(chunks, wch)
    val = torch.ones((b, 1, 1))
    feats = physics_features(chunks, wch, tiles, val, pairs, tiny_array.n_ant,
                             uv_dist_m=np.linspace(100, 5000, b))
    sk_noise = feats[..., 0].mean()
    assert abs(float(sk_noise)) < 0.15, f"pure-noise SK excess {sk_noise}"

    cw = noise.copy()
    cw[:, :, :, :3, 40] += 30.0          # CW on 3 of 16 dumps -> intermittent
    chunks_c = torch.as_tensor(cw.astype(np.complex64))
    tiles_c = delay_fringe_tiles(chunks_c, wch)
    feats_c = physics_features(chunks_c, wch, tiles_c, val, pairs, tiny_array.n_ant,
                               uv_dist_m=np.linspace(100, 5000, b))
    assert float(feats_c[..., 0].mean()) > 1.0, "contaminated SK should show excess"
