"""Small shared helpers: YAML loading, dotpath overrides, seeding."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def apply_override(cfg: dict, dotpath: str, value: Any) -> None:
    """Set ``cfg['a']['b'] = parsed(value)`` for dotpath ``'a.b'`` (in place)."""
    keys = dotpath.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = yaml.safe_load(str(value))


def parse_overrides(pairs: list[str]) -> list[tuple[str, Any]]:
    """['a.b=1', ...] -> [('a.b', 1), ...]."""
    out = []
    for p in pairs:
        k, _, v = p.partition("=")
        out.append((k.strip(), yaml.safe_load(v.strip())))
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def resolve_device(name: str | None) -> torch.device:
    if name in (None, "auto"):
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        name = "cpu"
    return torch.device(name)
