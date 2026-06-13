# ABSOL — Array-Based Signal Outlier Locator

Graph-based visibility-domain RFI detection for GMRT band-4.

Per-array RFI detector for GMRT band-4 (550–750 MHz, 30 antennas, 4096 channels).
Train a heterogeneous GNN on simulated contaminated visibilities for the fixed GMRT
geometry; at inference read a Measurement Set, output calibrated per-(baseline, time,
channel) contamination probabilities + per-antenna scores, and write them back as an
`ABSOL_WEIGHT` column and optional `FLAG` OR-update. **Pre-existing flags are always
respected**: flagged samples are excluded from normalization/features, carry zero
weight into the model (validity feature), and writeback only ever adds flags.

## Install

```bash
pixi install              # default env: torch + PyG (CUDA if available)
pixi install -e ms        # + python-casacore (MS I/O) + sgp4 + dev tools
pixi run -e ms pytest     # run the test suite (CPU)
```

## Happy path

1. Antenna positions ship in `configs/array_gmrt.yaml` (real GMRT Antenna.def values,
   converted to ENU; the two mock entries C07/S05 are excluded). Edit only this file
   if positions change — everything reads geometry from it.

2. Train (curriculum stages 1–4, on-the-fly simulated scenes):

```bash
pixi run absol train --sim-config configs/sim_gmrt_band4.yaml \
    --array-config configs/array_gmrt.yaml \
    --model-config configs/model_default.yaml --out runs/v0
```

   On a small GPU, train cheaper by reducing the **span**, not the resolution.
   The model only sees fixed 16×128 tiles, so it transfers to full-band data —
   **but only if the per-cell resolution matches**: channel width Δν and dump
   time Δt must stay equal to the real data (GMRT band-4: 48.83 kHz/ch, 8 s).
   So narrow the band proportionally to the channel cut (keep 48.83 kHz/ch),
   and keep `observation.integration_s` and the chunk sizes fixed:

```bash
# 1024 ch over 50 MHz = same 48.83 kHz/ch as full 4096-ch / 200 MHz band
pixi run absol train ... \
    --override band.n_channels=1024 \
    --override band.freq_start_hz=550.0e+6 --override band.freq_end_hz=600.0e+6 \
    --override scene.n_time=48 --override training.batch_scenes=2
```

   (Reducing `n_channels` while leaving the band at 550–750 MHz would widen Δν
   and break the train/inference match — don't do that.) Everything is
   config-adjustable: `band.n_channels`, `band.freq_start/end_hz`,
   `scene.n_time`, `observation.integration_s`, and
   `chunking.chunk_time_samples` / `chunking.chunk_freq_channels` all flow from
   YAML and accept `--override`.

   Quick sanity check that the whole stack learns (a few minutes, also useful on CPU):

```bash
pixi run absol train ... --out runs/sanity --overfit-one --steps 300
```

3. Calibrate probabilities (temperature scaling on holdout scenes):

```bash
pixi run absol calibrate --run runs/v0
```

4. Inference on a real MS (requires `-e ms` env):

```bash
pixi run -e ms absol infer --ms obs.ms --run runs/v0 --write-flags --threshold 0.5
```

   Outputs: `ABSOL_WEIGHT` column (1 − p_sample), optional FLAG OR-update, an HDF5
   sidecar (`obs.ms.absol.h5`: p_chunk, antenna scores, direction attributions), and a
   quicklook PNG.

5. Label-free validation on real data:

```bash
pixi run -e ms absol validate injection --ms obs.ms --run runs/v0 --cfg configs/validate_default.yaml
pixi run -e ms absol validate coincidence --ms obs.ms --run runs/v0 --cfg configs/validate_default.yaml
pixi run -e ms absol validate residuals --ms obs.ms --run runs/v0 --cfg configs/validate_default.yaml
pixi run -e ms absol validate report --run runs/v0 --cfg configs/validate_default.yaml
```

`validate report` aggregates all metrics into `validation_report.md` with the
pre-registered acceptance table.

## Repository layout

See `plan.md` spec. `src/absol/geometry.py` is the single source of truth for fringe
rates (`direction_buckets`): the simulator, graph relation B, and the coincidence
validator all call it.

## v0 non-goals

RFI subtraction, real-time/streaming, SBI, multi-array generalization, image-domain
automation (imaging runs externally in CASA/WSClean).
