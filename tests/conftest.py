"""Shared fixtures: tiny 8-antenna GMRT sub-array + tiny scene config (CPU)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from absol.geometry import Array
from absol.utils import load_yaml

REPO = Path(__file__).resolve().parents[1]

TINY_ANTS = [
    {"id": "C00", "e": 687.88, "n": -21.21, "u": 0.01},
    {"id": "C01", "e": 326.45, "n": -42.46, "u": -0.68},
    {"id": "C02", "e": 0.00, "n": 0.00, "u": 0.00},
    {"id": "C03", "e": -372.71, "n": 142.99, "u": -4.68},
    {"id": "C05", "e": 67.81, "n": -258.90, "u": -5.91},
    {"id": "W01", "e": -1591.95, "n": 624.65, "u": 3.18},
    {"id": "E02", "e": 2814.55, "n": 1015.10, "u": -17.02},
    {"id": "S01", "e": 633.96, "n": -2960.03, "u": -26.88},
]


@pytest.fixture(scope="session")
def tiny_array_yaml(tmp_path_factory) -> Path:
    cfg = {
        "array": {"name": "GMRT-tiny", "latitude_deg": 19.0963,
                  "longitude_deg": 74.05, "antennas": TINY_ANTS},
        "band": {"freq_start_hz": 550.0e6, "freq_end_hz": 750.0e6,
                 "n_channels": 256, "n_pol": 2},
        "observation": {"integration_s": 8.0},
    }
    p = tmp_path_factory.mktemp("cfg") / "array_tiny.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


@pytest.fixture(scope="session")
def tiny_array(tiny_array_yaml) -> Array:
    return Array.from_yaml(tiny_array_yaml)


@pytest.fixture(scope="session")
def tiny_sim_cfg() -> dict:
    return {
        "scene": {
            "duration_s": 192, "n_time": 24,
            "declination_deg": {"min": -10, "max": 40},
            "clean_scene_fraction": 0.0,
            "n_rfi_sources": {"min": 1, "max": 2},
        },
        "sky": {
            "n_sources": {"min": 1, "max": 3},
            "flux_jy": {"dist": "powerlaw", "alpha": -1.6, "min": 0.05, "max": 1.0},
            "field_radius_deg": 1.0,
        },
        "noise": {"sefd_jy": 390, "sefd_jitter_frac": 0.15},
        "gains": {"amp_rms": 0.05, "phase_rms_deg": 10, "timescale_s": 300},
        "flag_augmentation": {"scattered_p": 0.3, "contiguous_p": 0.3, "block_p": 0.1},
        "antenna_dropout": {"min_frac": 0.0, "max_frac": 0.25},
        "rfi": {
            "strength_sigma": {"dist": "loguniform", "min": 0.3, "max": 1000},
            "coupling_decades": 3,
            "coupling_mode_probs": {"subset": 0.5, "all_antennas": 0.3, "single": 0.2},
            "modulation_timescale_s": {"min": 60, "max": 600},
            "mechanisms": {
                "ground_narrowband": 0.30, "powerline_arcing": 0.20,
                "satellite": 0.20, "pulsed": 0.15,
                "internal_zero_fringe": 0.10, "transient_protected": 0.05,
            },
        },
        "truth": {"mask_threshold_sigma": 0.1},
        "chunking": {"chunk_time_samples": 16, "chunk_freq_channels": 128},
        "residual": {"savgol_window_dumps": 11, "savgol_order": 3},
    }


@pytest.fixture(scope="session")
def model_cfg() -> dict:
    cfg = load_yaml(REPO / "configs" / "model_default.yaml")
    cfg["training"]["device"] = "cpu"
    cfg["training"]["amp"] = False
    return cfg
