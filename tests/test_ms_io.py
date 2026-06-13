"""Spec test 10: ms_io round-trip on a tiny synthetic MS (skip if no casacore)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

casacore = pytest.importorskip("casacore.tables")

N_ANT, N_T, N_F, N_CORR = 4, 8, 64, 2
ANT_NAMES = ["C00", "C01", "C02", "C03"]


def _make_ms(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Minimal single-spw MS with XX/YY, returns (data, flags) per row."""
    from casacore.tables import (
        default_ms,
        makearrcoldesc,
        maketabdesc,
        table,
    )

    ms = default_ms(path)
    ms.addcols(maketabdesc(makearrcoldesc(
        "DATA", 0.0 + 0.0j, ndim=2, shape=[N_F, N_CORR], valuetype="complex"
    )))
    ms.close()

    ant = table(f"{path}/ANTENNA", readonly=False, ack=False)
    ant.addrows(N_ANT)
    ant.putcol("NAME", ANT_NAMES)
    ant.putcol("DISH_DIAMETER", np.full(N_ANT, 45.0))
    ant.putcol("POSITION", np.tile([1656342.0, 5797947.0, 2073243.0], (N_ANT, 1)))
    ant.close()

    spw = table(f"{path}/SPECTRAL_WINDOW", readonly=False, ack=False)
    spw.addrows(1)
    freqs = np.linspace(550e6, 650e6, N_F)
    spw.putcell("CHAN_FREQ", 0, freqs)
    spw.putcell("CHAN_WIDTH", 0, np.full(N_F, freqs[1] - freqs[0]))
    spw.putcell("NUM_CHAN", 0, N_F)
    spw.putcell("NAME", 0, "band4")
    spw.close()

    pol = table(f"{path}/POLARIZATION", readonly=False, ack=False)
    pol.addrows(1)
    pol.putcell("CORR_TYPE", 0, np.array([9, 12]))          # XX, YY
    pol.putcell("CORR_PRODUCT", 0, np.array([[0, 0], [1, 1]]))
    pol.putcell("NUM_CORR", 0, N_CORR)
    pol.close()

    dd = table(f"{path}/DATA_DESCRIPTION", readonly=False, ack=False)
    dd.addrows(1)
    dd.putcell("SPECTRAL_WINDOW_ID", 0, 0)
    dd.putcell("POLARIZATION_ID", 0, 0)
    dd.close()

    fld = table(f"{path}/FIELD", readonly=False, ack=False)
    fld.addrows(1)
    fld.putcell("PHASE_DIR", 0, np.array([[1.0, 0.35]]))
    fld.putcell("DELAY_DIR", 0, np.array([[1.0, 0.35]]))
    fld.putcell("REFERENCE_DIR", 0, np.array([[1.0, 0.35]]))
    fld.putcell("NAME", 0, "test")
    fld.close()

    pairs = [(i, j) for i in range(N_ANT) for j in range(i + 1, N_ANT)]
    n_rows = N_T * len(pairs)
    main = table(path, readonly=False, ack=False)
    main.addrows(n_rows)
    rng = np.random.default_rng(0)
    data = (rng.standard_normal((n_rows, N_F, N_CORR))
            + 1j * rng.standard_normal((n_rows, N_F, N_CORR))).astype(np.complex64)
    flags = np.zeros((n_rows, N_F, N_CORR), dtype=bool)
    flags[:, 10:14, :] = True                              # band-stop pre-flag
    t0 = 5e9
    r = 0
    for ti in range(N_T):
        for (i, j) in pairs:
            main.putcell("TIME", r, t0 + 8.0 * ti)
            main.putcell("TIME_CENTROID", r, t0 + 8.0 * ti)
            main.putcell("ANTENNA1", r, i)
            main.putcell("ANTENNA2", r, j)
            main.putcell("DATA_DESC_ID", r, 0)
            main.putcell("FIELD_ID", r, 0)
            main.putcell("SCAN_NUMBER", r, 1)
            main.putcell("EXPOSURE", r, 8.0)
            main.putcell("INTERVAL", r, 8.0)
            main.putcell("UVW", r, np.zeros(3))
            main.putcell("DATA", r, data[r])
            main.putcell("FLAG", r, flags[r])
            main.putcell("FLAG_ROW", r, False)
            main.putcell("WEIGHT", r, np.ones(N_CORR))
            main.putcell("SIGMA", r, np.ones(N_CORR))
            r += 1
    main.close()
    return data, flags


def test_10_ms_roundtrip(tmp_path, tiny_array):
    from absol.inference.ms_io import read_ms, write_back

    ms_path = str(tmp_path / "tiny.ms")
    data, flags = _make_ms(ms_path)

    scans = read_ms(ms_path, tiny_array)
    assert len(scans) == 1
    scan = scans[0]
    n_b = N_ANT * (N_ANT - 1) // 2
    assert scan.vis.shape == (n_b, N_T, N_F, 2)
    assert scan.array.n_ant == N_ANT                       # subset of the 8-ant cfg

    # values round-trip (row 0 is baseline (0,1) at t0 in our construction)
    b0 = 0
    assert torch.allclose(
        scan.vis[b0, 0], torch.as_tensor(data[0].astype(np.complex64)), atol=1e-6
    )
    # ALL pre-existing flags respected -> zero weight there
    assert (scan.weights_in[:, :, 10:14] == 0).all()
    assert (scan.weights_in[:, :, 20:] == 1).all()

    # writeback: ABSOL_WEIGHT + FLAG OR-update; old flags must survive
    p = np.zeros((n_b, N_T, N_F), dtype=np.float32)
    p[0, :, 30:33] = 0.9                                    # detector hits
    write_back(ms_path, scans, [p], write_flags=True, threshold=0.5)

    t = casacore.table(ms_path, ack=False)
    assert "ABSOL_WEIGHT" in t.colnames()
    w = t.getcol("ABSOL_WEIGHT")
    fl = t.getcol("FLAG")
    t.close()
    assert np.isclose(w[0, 30, 0], 0.1, atol=1e-5)          # 1 - 0.9
    assert (w[:, 10:14, :] == 0).all()                      # pre-flagged -> weight 0
    assert fl[:, 10:14, :].all()                            # old flags retained
    assert fl[0, 30:33, :].all()                            # new flags added
    assert not fl[1, 30:33, :].any()                        # only where p > thr
