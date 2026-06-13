"""Spec test 7: graph construction invariants."""
from __future__ import annotations

import numpy as np

from absol.graphs import REL_REV, REL_SAME_DIR, REL_SAME_UV, REL_SHARES
from absol.pipeline import prepare
from absol.simulator.scenes import SceneGenerator


def _prep(tiny_array, tiny_sim_cfg, model_cfg, drop=False, seed=5):
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in tiny_sim_cfg.items()}
    if drop:
        cfg["antenna_dropout"] = {"min_frac": 0.3, "max_frac": 0.3}
    else:
        cfg["antenna_dropout"] = {"min_frac": 0.0, "max_frac": 0.0}
    gen = SceneGenerator(cfg, tiny_array, seed=seed)
    scene = gen.sample(stage=3 if drop else 1)
    prep = prepare(scene.vis, scene.weights_in, scene.array, scene.ha_rad,
                   scene.dec_rad, scene.freqs_hz, cfg, model_cfg)
    return scene, prep


def test_7_shared_antenna_degree(tiny_array, tiny_sim_cfg, model_cfg):
    scene, prep = _prep(tiny_array, tiny_sim_cfg, model_cfg)
    g = prep.columns[0]
    a = scene.array.n_ant
    nf = prep.static.n_freq_tiles
    n_b = scene.array.baselines().shape[0]
    assert g["bl"].num_nodes == n_b * nf
    assert g["antenna"].num_nodes == a

    ei = g[REL_SHARES].edge_index.numpy()
    # each bl-node connects to exactly its 2 antennas
    counts = np.bincount(ei[0], minlength=n_b * nf)
    assert (counts == 2).all()
    # distinct other baselines sharing an antenna = 2(A-2) per bl-node
    node_ants = {n: set() for n in range(n_b * nf)}
    for s, d in ei.T:
        node_ants[s].add(d)
    for node in range(0, n_b * nf, nf):           # one node per baseline (tile 0)
        others = sum(
            1 for o in range(0, n_b * nf, nf)
            if o != node and node_ants[o] & node_ants[node]
        )
        assert others == 2 * (a - 2)

    # reverse relation is the exact transpose
    rev = g[REL_REV].edge_index.numpy()
    assert (rev[::-1] == ei).all()


def test_7_same_dir_symmetric(tiny_array, tiny_sim_cfg, model_cfg):
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in tiny_sim_cfg.items()}
    cfg["rfi"] = dict(cfg["rfi"], mechanisms={"ground_narrowband": 1.0},
                      strength_sigma={"dist": "loguniform", "min": 100, "max": 200})
    gen = SceneGenerator(cfg, tiny_array, seed=2)
    scene = gen.sample(stage=1)
    prep = prepare(scene.vis, scene.weights_in, scene.array, scene.ha_rad,
                   scene.dec_rad, scene.freqs_hz, cfg, model_cfg)
    for g in prep.columns:
        e = g[REL_SAME_DIR].edge_index.numpy()
        fwd = set(map(tuple, e.T))
        assert all((v, u) in fwd for (u, v) in fwd), "relation B must be symmetric"
        uvw_e = g[REL_SAME_UV].edge_index.numpy()
        assert uvw_e.shape[1] > 0                    # uv relation always populated


def test_7_antenna_dropout_removes_incident_nodes(tiny_array, tiny_sim_cfg, model_cfg):
    scene, prep = _prep(tiny_array, tiny_sim_cfg, model_cfg, drop=True)
    a_s = scene.array.n_ant
    assert a_s < tiny_array.n_ant                     # something was dropped
    n_b = a_s * (a_s - 1) // 2
    g = prep.columns[0]
    assert g["antenna"].num_nodes == a_s
    assert g["bl"].num_nodes == n_b * prep.static.n_freq_tiles
    assert int(g[REL_SHARES].edge_index[1].max()) < a_s
