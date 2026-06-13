"""Measurement Set I/O (lazy casacore import).

Reading: DATA/CORRECTED_DATA (optionally minus MODEL_DATA), FLAG, FLAG_ROW
-> per-scan tensors on the array's antenna ordering. ALL pre-existing flags
are respected: any flagged correlation => weights_in = 0 for that sample;
rows missing from the time grid get weight 0; autocorrelations are skipped.

Writeback: (a) ``ABSOL_WEIGHT`` float column [nchan, ncorr] = 1 - p_sample
(0 where pre-flagged), (b) optional FLAG OR-update at a threshold - flags
are only ever ADDED, never cleared.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from absol.geometry import Array


def _tables():
    try:
        from casacore import tables
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "python-casacore is required for MS I/O: install the [ms] extra "
            "(pixi install -e ms)"
        ) from e
    return tables


def gmst_rad(mjd_sec: np.ndarray) -> np.ndarray:
    """Greenwich mean sidereal time (rad) from MJD seconds (UTC ~ UT1 ok here).

    Approximate polynomial (good to ~seconds of time, far below the
    direction-bucket resolution)."""
    d = mjd_sec / 86400.0 - 51544.5
    gmst_h = (18.697374558 + 24.06570982441908 * d) % 24.0
    return gmst_h / 24.0 * 2 * np.pi


@dataclass
class ScanData:
    vis: torch.Tensor            # [B, T, F, P] complex64 (parallel hands)
    weights_in: torch.Tensor     # [B, T, F] float32
    times: np.ndarray            # [T] MJD seconds
    ha_rad: np.ndarray           # [T]
    dec_rad: float
    freqs_hz: np.ndarray         # [F]
    array: Array                 # sub-array actually present in this scan
    scan_id: int
    row_index: np.ndarray        # [B, T] MS row or -1
    pol_idx: np.ndarray          # indices of parallel-hand correlations in MS
    n_corr_ms: int


def read_ms(
    ms_path: str,
    array: Array,
    data_column: str = "auto",
    subtract_model: bool = False,
    max_scans: int | None = None,
) -> list[ScanData]:
    t = _tables()
    main = t.table(ms_path, readonly=True, ack=False)
    if data_column == "auto":
        data_column = "CORRECTED_DATA" if "CORRECTED_DATA" in main.colnames() else "DATA"

    ant_tab = t.table(f"{ms_path}/ANTENNA", ack=False)
    ms_names = [n.strip() for n in ant_tab.getcol("NAME")]
    ant_tab.close()
    name_to_cfg = {n: i for i, n in enumerate(array.names)}
    ms_to_cfg = np.array([name_to_cfg.get(n, -1) for n in ms_names])
    present = np.zeros(array.n_ant, dtype=bool)
    present[ms_to_cfg[ms_to_cfg >= 0]] = True

    spw = t.table(f"{ms_path}/SPECTRAL_WINDOW", ack=False)
    freqs = spw.getcol("CHAN_FREQ")[0].astype(np.float64)
    spw.close()
    field = t.table(f"{ms_path}/FIELD", ack=False)
    phase_dir = field.getcol("PHASE_DIR")
    field.close()
    pol_tab = t.table(f"{ms_path}/POLARIZATION", ack=False)
    corr_types = pol_tab.getcol("CORR_TYPE")[0]
    pol_tab.close()
    # parallel hands: XX/YY (9, 12) or RR/LL (5, 8)
    par = [k for k, c in enumerate(corr_types) if c in (9, 12, 5, 8)]
    pol_idx = np.array(par[:2] if len(par) >= 2 else [0], dtype=int)

    scans = []
    scan_numbers = np.unique(main.getcol("SCAN_NUMBER"))
    if max_scans is not None:
        scan_numbers = scan_numbers[:max_scans]
    for scan_no in scan_numbers:
        sub = main.query(f"SCAN_NUMBER == {scan_no} AND DATA_DESC_ID == 0")
        if sub.nrows() == 0:
            sub.close()
            continue
        a1 = ms_to_cfg[sub.getcol("ANTENNA1")]
        a2 = ms_to_cfg[sub.getcol("ANTENNA2")]
        times_all = sub.getcol("TIME")
        fid = int(sub.getcol("FIELD_ID")[0])
        ra, dec = float(phase_dir[fid, 0, 0]), float(phase_dir[fid, 0, 1])

        scan_present = np.zeros(array.n_ant, dtype=bool)
        ok_rows = (a1 >= 0) & (a2 >= 0) & (a1 != a2)
        scan_present[a1[ok_rows]] = True
        scan_present[a2[ok_rows]] = True
        sub_array = array.subset(scan_present)
        sub_idx = {int(g): k for k, g in enumerate(np.flatnonzero(scan_present))}
        pairs = sub_array.baselines()
        pair_to_b = {(int(i), int(j)): b for b, (i, j) in enumerate(pairs)}

        utimes = np.unique(times_all)
        n_t, n_b, n_f = utimes.size, pairs.shape[0], freqs.size
        t_of = {tv: k for k, tv in enumerate(utimes)}
        n_p = min(array.n_pol, pol_idx.size)

        vis = torch.zeros((n_b, n_t, n_f, n_p), dtype=torch.complex64)
        w = torch.zeros((n_b, n_t, n_f), dtype=torch.float32)
        row_index = np.full((n_b, n_t), -1, dtype=np.int64)

        data = sub.getcol(data_column)
        if subtract_model and "MODEL_DATA" in main.colnames():
            data = data - sub.getcol("MODEL_DATA")
        flag = sub.getcol("FLAG")
        flag_row = sub.getcol("FLAG_ROW")
        for r in range(sub.nrows()):
            if not ok_rows[r]:
                continue
            i, j = sub_idx[int(a1[r])], sub_idx[int(a2[r])]
            conj = i > j
            key = (j, i) if conj else (i, j)
            b = pair_to_b[key]
            ti = t_of[times_all[r]]
            row_index[b, ti] = r
            d = data[r][:, pol_idx[:n_p]]
            if conj:
                d = np.conj(d)
            vis[b, ti] = torch.as_tensor(d.astype(np.complex64))
            f_any = flag[r][:, pol_idx[:n_p]].any(axis=1) | bool(flag_row[r])
            w[b, ti] = torch.as_tensor((~f_any).astype(np.float32))
        sub.close()

        lst = gmst_rad(utimes) + np.deg2rad(array.longitude_deg)
        ha = ((lst - ra + np.pi) % (2 * np.pi)) - np.pi
        scans.append(ScanData(
            vis=vis, weights_in=w, times=utimes, ha_rad=ha, dec_rad=dec,
            freqs_hz=freqs, array=sub_array, scan_id=int(scan_no),
            row_index=row_index, pol_idx=pol_idx, n_corr_ms=len(corr_types),
        ))
    main.close()
    return scans


def write_back(
    ms_path: str,
    scans: list[ScanData],
    p_samples: list[np.ndarray],      # per scan [B, T, F] in [0, 1]
    write_flags: bool = False,
    threshold: float = 0.5,
    column: str = "ABSOL_WEIGHT",
) -> None:
    t = _tables()
    main = t.table(ms_path, readonly=False, ack=False)
    nchan = scans[0].freqs_hz.size
    ncorr = scans[0].n_corr_ms
    if column not in main.colnames():
        desc = t.makearrcoldesc(column, 1.0, ndim=2, shape=[nchan, ncorr],
                                valuetype="float")
        main.addcols(t.maketabdesc(desc))
        full = np.ones((main.nrows(), nchan, ncorr), dtype=np.float32)
        main.putcol(column, full)

    for scan, p in zip(scans, p_samples):
        rows = scan.row_index
        b_idx, t_idx = np.nonzero(rows >= 0)
        flat_rows = rows[b_idx, t_idx]
        order = np.argsort(flat_rows)
        b_idx, t_idx, flat_rows = b_idx[order], t_idx[order], flat_rows[order]
        w_in = scan.weights_in.numpy()
        for b, ti, r in zip(b_idx, t_idx, flat_rows):
            wgt = (1.0 - p[b, ti]).astype(np.float32)
            wgt = wgt * (w_in[b, ti] > 0)            # pre-flagged stays weight 0
            main.putcell(column, int(r), np.repeat(wgt[:, None], ncorr, axis=1))
            if write_flags:
                fl = main.getcell("FLAG", int(r))
                new = p[b, ti] > threshold
                fl |= new[:, None]                   # OR-update: only ever add flags
                main.putcell("FLAG", int(r), fl)
    main.close()
