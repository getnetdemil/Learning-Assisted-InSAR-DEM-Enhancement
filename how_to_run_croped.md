# How to Run — Learning-Assisted InSAR DEM Enhancement

All commands assume you are in the repo root and use the `torch-gpu` conda env.
Always set the runtime environment first:

```bash
export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement
alias py="conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu python"
```

All scripts must be run from the repo root (`/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement`).

---

## Pipeline Overview

```
Step 1  scripts/download_subset.py            Build SLC manifest + download raw data
Step 2  src/insar_processing/pair_graph.py    Build pair graph + compute B_perp (Python API)
Step 3  scripts/preprocess_pairs.py           Coregistration → interferogram → Goldstein filter
Step 4  scripts/select_triplet_completing_pairs.py   Find pairs needed to close triplets
Step 4b scripts/patch_coreg_meta.py           Add FiLM metadata fields to coreg_meta.json
Step 4c scripts/assess_coreg_quality.py       Retroactive quality metrics from processed pairs
Step 5  scripts/unwrap_snaphu.py              Phase unwrapping with SNAPHU
Step 6  experiments/enhanced/train_film_unet.py      Train FiLMUNet (self-supervised)
Step 7  eval/compute_metrics.py               All 5 contest metrics vs baseline
Step 8  experiments/baseline/run_baseline.py  Classical phase-to-height baseline (legacy)
Step 9  experiments/enhanced/train_unet.py    Legacy U-Net baseline (superseded)
```

---

## Step 1 — Build the SLC Manifest + Download Raw Data

**Script**: `scripts/download_subset.py`

**Purpose**: Two modes in one script:
1. `--index_only` — Crawl the Capella STAC catalog, assign 39 AOI labels to all 791 SLCs, and save a metadata manifest. No data is downloaded.
2. Download mode — Parallel S3 download of SLC GeoTIFFs + extended JSON sidecars for a chosen AOI. No AWS credentials needed (public bucket, unsigned access).

**Prerequisite**: `pip install boto3 pystac` (already in `torch-gpu` env).

```bash
# 1a. Build the full metadata index (no download, ~2 min):
py scripts/download_subset.py \
    --index_only \
    --out_manifest data/manifests/full_index.parquet

# 1b. Download all Hawaii (AOI_000) SLC collects — ~497 GB, 8 workers:
py scripts/download_subset.py \
    --manifest data/manifests/full_index.parquet \
    --aoi_filter AOI_000 \
    --out_dir data/raw/ \
    --n_workers 8 \
    --assets slc,metadata

# 1c. Download only spotlight ascending collects, cap at 50:
py scripts/download_subset.py \
    --manifest data/manifests/full_index.parquet \
    --aoi_filter AOI_000 \
    --mode_filter spotlight \
    --orbit_filter ascending \
    --max_collects 50 \
    --out_dir data/raw/ \
    --n_workers 8

# 1d. Download SLCs + preview thumbnails:
py scripts/download_subset.py \
    --manifest data/manifests/full_index.parquet \
    --aoi_filter AOI_000 \
    --out_dir data/raw/ \
    --assets slc,metadata,preview \
    --n_workers 4
```

**All flags**:

| Flag | Default | Description |
|------|---------|-------------|
| `--index_only` | off | Build manifest only, skip download |
| `--out_manifest` | `data/manifests/full_index.parquet` | Output manifest path |
| `--manifest` | — | Existing manifest to use for download (skip re-crawl) |
| `--aoi_filter` | — | AOI label(s) to download, e.g. `AOI_000` or `AOI_000,AOI_008` |
| `--mode_filter` | — | Instrument mode filter, e.g. `spotlight` |
| `--orbit_filter` | — | `ascending` or `descending` |
| `--max_collects` | — | Cap number of SLC collects to download |
| `--out_dir` | `data/raw/` | Root directory for downloaded files |
| `--n_workers` | 4 | Parallel download workers |
| `--assets` | `slc,metadata` | Asset keys: `slc`, `metadata`, `preview` |

**Outputs**:
- `data/manifests/full_index.parquet` — 791 rows, columns: `id`, `collect_id`, `datetime`, `platform`, `instrument_mode`, `orbit_state`, `look_direction`, `orbital_plane`, `incidence_angle_deg`, `lon`, `lat`, `aoi`, etc.
- `data/raw/AOI_000/<collect_id>/` — per-collect subdirectory with `*.tif` (CInt16 SLC) and `*_extended.json` metadata

**Current status**: DONE — 221 Hawaii SLCs in `data/raw/AOI_000/`.

---

## Step 2 — Build the Pair Graph + Compute B_perp

**Module**: `src/insar_processing/pair_graph.py` (Python API, no standalone CLI)

**Purpose**: Enumerate all valid interferometric pairs from the SLC manifest, score each by `Q_ij = 1/(Δt+1) × 1/(1+Δinc)`, and enumerate all valid closure triplets (a→b, b→c, a→c) within temporal/incidence constraints. Pairs with same orbit state and look direction only.

```bash
py -c "
from src.insar_processing.pair_graph import PairGraphConfig, build_pair_graph, find_triplets
import pandas as pd

manifest = pd.read_parquet('data/manifests/full_index.parquet')
hawaii = manifest[manifest.aoi == 'AOI_000']

# Build pair graph: all pairs with Δt ≤ 365 days, Δinc ≤ 5°
cfg = PairGraphConfig(dt_max_days=365, dinc_max_deg=5.0)
pairs_df = build_pair_graph(hawaii, cfg)
pairs_df.to_parquet('data/manifests/hawaii_pairs.parquet', index=False)

# Find all closed triplets from the pair graph
triplets_df = find_triplets(pairs_df)
triplets_df.to_parquet('data/manifests/hawaii_triplets_strict.parquet', index=False)

print(f'{len(pairs_df)} pairs, {len(triplets_df)} triplets')
"
```

**Outputs**:
- `data/manifests/hawaii_pairs.parquet` — columns: `id_ref`, `id_sec`, `datetime_ref`, `datetime_sec`, `dt_days`, `dinc_deg`, `orbit_state`, `look_direction`, `incidence_ref`, `incidence_sec`, `orbital_plane_ref`, `orbital_plane_sec`, `q_score`, `aoi`
- `data/manifests/hawaii_triplets_strict.parquet` — columns: `id_a`, `id_b`, `id_c`

**Current status**: DONE — 8,834 pairs, 24,171 triplets (built with `dt_max_days=365, dinc_max_deg=5`).
Preprocessing (Step 3) applies a tighter `--dt_max 7` filter → 224 pairs actually processed.

---

## Step 3 — Preprocess Pairs (Coregistration → Interferogram → Coherence → Goldstein)

**Script**: `scripts/preprocess_pairs.py`

**Purpose**: Full end-to-end preprocessing for each pair:
1. Load SLC patches (4096×4096 complex GeoTIFF)
2. Sub-pixel coregistration via phase cross-correlation
3. Form complex interferogram
4. Estimate coherence with 5×5 box-car window
5. Apply Goldstein-Werner spectral filter (fixed or coherence-adaptive)
6. Save outputs + `coreg_meta.json`

Skips pairs where output directory already exists and has `coreg_meta.json`.

**Important**: pair output directories use the full pair ID as the directory name (up to 104 characters). Do NOT change this — a prior bug with 80-char truncation caused silent overwrites.

```bash
# Process top-100 pairs by row order from manifest (recommended for training):
py scripts/preprocess_pairs.py \
    --pairs_manifest data/manifests/hawaii_pairs.parquet \
    --raw_dir data/raw/AOI_000 \
    --out_dir data/processed/pairs \
    --max_pairs 100 \
    --patch_size 4096 \
    --looks_range 5 \
    --looks_azimuth 5 \
    --goldstein_alpha 0.5 \
    --adaptive \
    --n_workers 4

# Process with fixed Goldstein alpha (no coherence adaptation):
py scripts/preprocess_pairs.py \
    --pairs_manifest data/manifests/hawaii_pairs.parquet \
    --raw_dir data/raw/AOI_000 \
    --out_dir data/processed/pairs \
    --max_pairs 100 \
    --goldstein_alpha 0.7 \
    --n_workers 4

# Process a specific subset from a different manifest (e.g. triplet-completing pairs):
py scripts/preprocess_pairs.py \
    --pairs_manifest data/manifests/triplet_completing_pairs.parquet \
    --raw_dir data/raw/AOI_000 \
    --out_dir data/processed/pairs \
    --patch_size 4096 \
    --adaptive \
    --n_workers 4

# Single pair for debugging (no --max_pairs = all pairs; use 1 for one):
py scripts/preprocess_pairs.py \
    --pairs_manifest data/manifests/hawaii_pairs.parquet \
    --raw_dir data/raw/AOI_000 \
    --out_dir data/processed/pairs \
    --max_pairs 1
```

**All flags**:

| Flag | Default | Description |
|------|---------|-------------|
| `--pairs_manifest` | required | Path to pairs parquet file |
| `--raw_dir` | required | Root dir with per-collect SLC subdirectories |
| `--out_dir` | required | Output directory for processed pairs |
| `--max_pairs` | — | Cap number of pairs to process (rows 0..N after filter) |
| `--dt_max` | 7.0 | Max temporal baseline filter (days) |
| `--dinc_max` | 2.0 | Max incidence angle difference filter (degrees) |
| `--patch_size` | 4096 | SLC patch size in pixels (square) |
| `--looks_range` | 5 | Range looks for coherence window |
| `--looks_azimuth` | 5 | Azimuth looks for coherence window |
| `--goldstein_alpha` | 0.5 | Goldstein filter strength α ∈ [0,1] (used when `--adaptive` is off) |
| `--adaptive` | off | Use coherence-adaptive α (stronger in low-coherence regions) |
| `--coreg_n_grid` | 3 | Grid dimension for multi-patch coregistration (3 = 3×3=9 patches; 1 = single centre patch) |
| `--n_workers` | 4 | Parallel worker threads |

**Output per pair** in `data/processed/pairs/<id_ref>__<id_sec>/`:

| File | Format | Content |
|------|--------|---------|
| `ifg_raw.tif` | 2-band float32 GeoTIFF | Re, Im of unnormalised interferogram |
| `ifg_goldstein.tif` | 2-band float32 GeoTIFF | Goldstein-filtered Re, Im |
| `coherence.tif` | 1-band float32 GeoTIFF | Coherence values ∈ [0,1] |
| `coreg_meta.json` | JSON | `id_ref`, `id_sec`, `dt_days`, `dinc_deg`, `q_score`, `bperp_m`, `row_offset_px`, `col_offset_px`, `patch_size`, `patch_row_ref`, `patch_col_ref`, `cc_peak_mean`, `cc_peak_min`, `n_coreg_patches`, `offset_row_std_px`, `offset_col_std_px`, `incidence_angle_deg`, `mode`, `look_direction`, `snr_proxy` (last 4 added by Step 4b) |

**Performance**: ~17 min for 62 pairs with 4 workers on a 4096×4096 patch size.

**Current status**: DONE — 162 unique pairs processed (100 original top-q + 62 triplet-completing).

---

## Step 4 — Select Triplet-Completing Pairs

**Script**: `scripts/select_triplet_completing_pairs.py`

**Purpose**: After preprocessing a q-score-based subset, identify the minimal set of additional pairs needed so that every available 2-leg triplet (a→b, b→c) gains its missing 3rd leg (a→c). This is necessary because q-score selection picks short-Δt pairs that form no closed triplets — the diagonal a→c legs are longer-Δt and need to be added explicitly. Also verifies the resulting network is overconstrained (P > T epochs) for SBAS.

Run this **after** Step 3 to find which pairs to add, then run Step 3 again on the output manifest.

```bash
# Identify the missing triplet-completing pairs:
py scripts/select_triplet_completing_pairs.py \
    --pairs_manifest    data/manifests/hawaii_pairs.parquet \
    --triplets_manifest data/manifests/hawaii_triplets_strict.parquet \
    --processed_dir     data/processed/pairs \
    --out_parquet       data/manifests/triplet_completing_pairs.parquet \
    --out_csv           data/manifests/triplet_completing_pairs.csv

# Then preprocess the identified pairs (Step 3 with the new manifest):
py scripts/preprocess_pairs.py \
    --pairs_manifest data/manifests/triplet_completing_pairs.parquet \
    --raw_dir data/raw/AOI_000 \
    --out_dir data/processed/pairs \
    --patch_size 4096 \
    --adaptive \
    --n_workers 4
```

**All flags**:

| Flag | Default | Description |
|------|---------|-------------|
| `--pairs_manifest` | `data/manifests/hawaii_pairs.parquet` | Full pair manifest to search |
| `--triplets_manifest` | `data/manifests/hawaii_triplets_strict.parquet` | Triplet definitions |
| `--processed_dir` | `data/processed/pairs` | Directory with already-processed pair subdirs |
| `--out_parquet` | `data/manifests/triplet_completing_pairs.parquet` | Output parquet of missing pairs |
| `--out_csv` | `data/manifests/triplet_completing_pairs.csv` | Output CSV (same content) |

**Expected stdout**:
```
Processed pairs found:  100
2-leg triplets found:    62
Unique missing pairs:    62
After processing: P=162, T=138 → OVERCONSTRAINED (+24)
All 62 pairs verified in hawaii_pairs.parquet ✓
Saved 62 rows → data/manifests/triplet_completing_pairs.parquet
```

**Outputs**:
- `data/manifests/triplet_completing_pairs.parquet` — same schema as `hawaii_pairs.parquet`, subset of rows
- `data/manifests/triplet_completing_pairs.csv` — same content in CSV

**Current status**: DONE — 62 pairs identified and preprocessed. Network has 162 pairs, 138 epochs (overconstrained by +24).

---

## Step 4b — Patch FiLM Metadata into coreg_meta.json

**Script**: `scripts/patch_coreg_meta.py`

**Purpose**: The preprocessing step (Step 3) saves `coreg_meta.json` with core geometry fields, but omits four fields required for proper FiLM conditioning: `incidence_angle_deg`, `mode`, `look_direction`, and `snr_proxy`. This script reads the missing fields from `hawaii_pairs.parquet` and writes them into each pair's `coreg_meta.json` in-place.

Run this once after all preprocessing is complete, before training or evaluation.

```bash
# Patch all 224 coreg_meta.json files:
py scripts/patch_coreg_meta.py \
    --pairs_dir data/processed/pairs \
    --manifest  data/manifests/hawaii_pairs.parquet
```

**All flags**:

| Flag | Default | Description |
|------|---------|-------------|
| `--pairs_dir` | required | Root directory containing per-pair subdirectories |
| `--manifest` | required | `hawaii_pairs.parquet` (source of incidence/look/q_score) |

**Expected output**:
```
Patched 224 / 224 coreg_meta.json files (0 not in manifest)
```

**Fields added to each** `coreg_meta.json`:

| Field | Source | Example |
|-------|--------|---------|
| `incidence_angle_deg` | mean of `incidence_ref` + `incidence_sec` | `55.8` |
| `mode` | hardcoded `"SL"` (all Capella Spotlight) | `"SL"` |
| `look_direction` | `look_direction` column from manifest | `"right"` |
| `snr_proxy` | `q_score` from manifest (quality proxy ∈ [0,1]) | `0.252` |

**Current status**: DONE — 224/224 files patched (2026-03-18).

---

## Step 4c — Retroactive Coregistration Quality Assessment

**Script**: `scripts/assess_coreg_quality.py`

**Purpose**: Estimates coregistration quality metrics for all already-processed pairs without re-running coregistration. Reads `ifg_goldstein.tif` + `coherence.tif` from each pair directory and computes:
- `mean_coherence` — average coherence across the full patch
- `coherence_p10` — 10th percentile coherence (flags poorly-coregistered pairs)
- `phase_spatial_std` — std of wrapped phase over coherent pixels (proxy for fringe density vs. noise)
- `cc_peak_mean` / `n_coreg_patches` — copied from `coreg_meta.json` if present

Takes < 5 min for 224 pairs.

```bash
py scripts/assess_coreg_quality.py \
    --pairs_dir data/processed/pairs \
    --out_csv   data/manifests/coreg_quality.csv
```

**All flags**:

| Flag | Default | Description |
|------|---------|-------------|
| `--pairs_dir` | required | Root directory containing per-pair subdirectories |
| `--out_csv` | `data/manifests/coreg_quality.csv` | Output CSV path |
| `--coh_flag_threshold` | 0.15 | Flag pairs with mean_coherence below this value |

**Output**: `data/manifests/coreg_quality.csv` — one row per pair with quality metrics and `flag_low_coherence` boolean column.

**Current status**: Available for use. New pairs processed with `--coreg_n_grid 3` will also have `cc_peak_mean`, `cc_peak_min`, `n_coreg_patches`, `offset_row_std_px`, `offset_col_std_px` in their `coreg_meta.json`.

---

## Step 5 — Phase Unwrapping with SNAPHU

**Script**: `scripts/unwrap_snaphu.py`

**Purpose**: Convert the wrapped interferometric phase (−π to π) to absolute unwrapped phase for each preprocessed pair. Reads `ifg_goldstein.tif` + `coherence.tif`, masks incoherent pixels, runs SNAPHU, and saves `unw_phase.tif`. Skips pairs that already have `unw_phase.tif`. Required for Metrics 2, 3, and 4.

**Prerequisite** — install `snaphu-py` (Python wrapper, no separate binary):
```bash
conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu \
    pip install snaphu
# or:
conda install --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu \
    -c conda-forge snaphu-py
# Verify: conda run --prefix ... python -c "import snaphu; print(snaphu.__version__)"
```

**Important**: use the direct Python path (not `conda run`) for real-time log output on large runs.
All Capella pairs are 4096×4096 pixels; the script automatically uses 2×2 tiling for these.

```bash
# Unwrap all processed pairs with DEFO mode, 4 workers (recommended):
DIRECT_PY=/scratch/gdemil24/hrwsi_s3client/torch-gpu/bin/python
$DIRECT_PY -u scripts/unwrap_snaphu.py \
    --pairs_dir data/processed/pairs \
    --mode DEFO \
    --nlooks 9.0 \
    --coh_threshold 0.1 \
    --workers 4

# Use TOPO mode for DEM estimation:
$DIRECT_PY -u scripts/unwrap_snaphu.py \
    --pairs_dir data/processed/pairs \
    --mode TOPO \
    --workers 4

# Limit to 50 pairs, stricter coherence mask:
$DIRECT_PY -u scripts/unwrap_snaphu.py \
    --pairs_dir data/processed/pairs \
    --max_pairs 50 \
    --mode DEFO \
    --coh_threshold 0.2 \
    --nlooks 25.0 \
    --workers 4

# Write unwrapped outputs to a separate directory:
$DIRECT_PY -u scripts/unwrap_snaphu.py \
    --pairs_dir data/processed/pairs \
    --out_dir data/processed/unwrapped \
    --mode DEFO \
    --workers 4
```

**All flags**:

| Flag | Default | Description |
|------|---------|-------------|
| `--pairs_dir` | required | Root directory containing per-pair subdirectories |
| `--out_dir` | same as `pairs_dir` | Output directory for `unw_phase.tif` files |
| `--max_pairs` | — | Cap number of pairs to process |
| `--mode` | `DEFO` | SNAPHU cost mode: `DEFO` (deformation) or `TOPO` (topographic) |
| `--coh_threshold` | 0.1 | Coherence below which pixels are masked to NaN in output |
| `--nlooks` | 9.0 | Equivalent number of independent looks (5×5 window → ~25) |
| `--workers` | 2 | Parallel worker threads |

**Output per pair**: `unw_phase.tif` — float32 GeoTIFF, unwrapped phase in radians, NaN at masked/incoherent pixels.

**Current status**: DONE — 224/224 pairs unwrapped (`unw_phase.tif` in all pair dirs).
Critical fix applied (2026-03-18): pairs ≥4096 px use `ntiles=(4,4)` (not `(2,2)`) to
avoid "Exceeded maximum secondary arcs" error. Unwrap success rate (Goldstein) = 0.256.

---

## Step 6 — Train FiLMUNet (Primary Model)

**Script**: `experiments/enhanced/train_film_unet.py`

**Purpose**: Self-supervised training of the FiLM-conditioned U-Net on preprocessed Capella interferogram tiles. No clean reference interferograms needed — uses Noise2Noise (N2N) via sub-look splits + physics-consistency losses (closure, temporal, gradient). Reads pairs from `data/processed/pairs/`, splits temporally into train/val/test, and trains for the configured number of epochs. Checkpoints every 5 epochs + best validation closure.

Three YAML configs control all behaviour:
- `configs/data/capella_aoi_selection.yaml` — data paths and split settings
- `configs/model/film_unet.yaml` — architecture (features, embed_dim, metadata_dim)
- `configs/train/contest.yaml` — learning rate, epochs, batch size, loss weights

```bash
# Standard training run with closure loss (foreground, timestamped log):
TASK_NAME="train_film_unet" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python experiments/enhanced/train_film_unet.py --data_config configs/data/capella_aoi_selection.yaml --model_config configs/model/film_unet.yaml --train_config configs/train/contest.yaml --run_name raw2gold_closure --triplets_manifest data/manifests/hawaii_triplets_strict.parquet | tee "$LOG"
```

```bash
# Resume from an existing checkpoint:
TASK_NAME="train_film_unet_resume" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python experiments/enhanced/train_film_unet.py --data_config configs/data/capella_aoi_selection.yaml --model_config configs/model/film_unet.yaml --train_config configs/train/contest.yaml --run_name raw2gold_closure --triplets_manifest data/manifests/hawaii_triplets_strict.parquet --resume experiments/enhanced/checkpoints/film_unet/raw2gold_closure_20260321_1852/raw2gold_closure_20260321_1852_final.pt | tee "$LOG"
```

**All flags**:

| Flag | Default | Description |
|------|---------|-------------|
| `--data_config` | required | Path to data YAML |
| `--model_config` | required | Path to model YAML |
| `--train_config` | required | Path to training YAML |
| `--run_name` | `None` | Tag for checkpoint/log filenames (timestamp appended) |
| `--resume` | `None` | Path to `.pt` checkpoint to resume from |
| `--triplets_manifest` | `None` | Parquet of triplets; enables closure loss via TripletTileDataset |
| `--epochs` | `None` | Override `num_epochs` from train config |
| `--loss_n2n` | `None` | Override `loss_weights.n2n` |
| `--loss_unc` | `None` | Override `loss_weights.unc` |
| `--loss_closure` | `None` | Override `loss_weights.closure` |
| `--loss_temporal` | `None` | Override `loss_weights.temporal` |
| `--loss_grad` | `None` | Override `loss_weights.grad` |
| `--zero_film` | off | Disable FiLM conditioning (ablation: geometry-blind U-Net) |

**Key config knobs** (edit the YAML files directly):

`configs/data/capella_aoi_selection.yaml`:
```yaml
pairs_dir: data/processed/pairs
tile_size: 256
stride: 128
min_coherence: 0.15
in_channels: 3         # Re, Im, coherence
temporal_split:
  train_frac: 0.70
  val_frac: 0.15       # rest is test (held out)
```

`configs/model/film_unet.yaml`:
```yaml
in_channels: 3
out_channels: 3        # Re_denoised, Im_denoised, log_variance
metadata_dim: 7        # [Δt, θ_inc, θ_graze, B_perp, mode, look, SNR_proxy]
features: [32, 64, 128, 256]
embed_dim: 64
```

`configs/train/contest.yaml`:
```yaml
lr: 1.0e-4
epochs: 50
batch_size: 8
weight_decay: 1.0e-5
seed: 42
loss_weights:
  n2n: 1.0
  uncertainty_nll: 0.5
  closure: 0.3
  temporal: 0.2
  gradient: 0.1
```

**Checkpoints** written to `experiments/enhanced/checkpoints/film_unet/{run_tag}/`:

| File pattern | Saved when |
|---|---|
| `{run_tag}_epoch_005.pt`, `_epoch_010.pt`, … | Every 5 epochs |
| `{run_tag}_best_closure.pt` | Best validation closure loss |
| `{run_tag}_final.pt` | End of last epoch |
| `{run_tag}_training_summary.json` | End of training |
| `logs/{run_tag}.log` | Live training log (same base name as checkpoint) |

where `run_tag = {--run_name}_{YYYYMMDD_HHMM}` (timestamp auto-set at script start).

Each checkpoint stores: model state dict, optimizer state, epoch number, all three config dicts, git commit hash.

**Current status**: DONE — final checkpoint `raw2gold_closure_20260321_1852_final.pt` (50 epochs, closure loss active).
M5 temporal residual improvement: −68.3% (1.158 → 0.367 rad). See REPRODUCIBILITY.md for full metric table.

---

## Step 7 — Compute All 5 Contest Metrics

**Script**: `eval/compute_metrics.py`

**Purpose**: Evaluate the full IEEE GRSS 2026 contest metric suite comparing Goldstein-filtered baseline against FiLMUNet, over all processed pairs. Optionally runs FiLMUNet inference on pairs that don't yet have `ifg_film_unet.tif`. Writes a CSV comparison table and three publication-quality figures.

The five metrics:
1. **Triplet closure error** — `median(|wrap(φ_ab + φ_bc − φ_ac)|)` over complete triplets (lower is better)
2. **Unwrap success rate** — fraction of high-coherence pixels with valid unwrapped phase (requires `unw_phase.tif`)
3. **Usable pairs fraction** — fraction passing coherence > 0.35 AND closure gate (requires triplets)
4. **DEM NMAD** — `1.4826 × median(|err − median(err)|)` (requires reference DEM)
5. **Temporal consistency residual** — `‖W(Ax − φ̂)‖₂` from weighted SBAS inversion (requires P > T)

```bash
# Full evaluation with inference (requires checkpoint + preprocessed pairs):
py eval/compute_metrics.py \
    --checkpoint  experiments/enhanced/checkpoints/film_unet/best_closure.pt \
    --pairs_dir   data/processed/pairs \
    --triplets_manifest data/manifests/hawaii_triplets_strict.parquet \
    --out_dir     experiments/enhanced/outputs

# Skip inference (reuse existing ifg_film_unet.tif from a prior run):
py eval/compute_metrics.py \
    --checkpoint  experiments/enhanced/checkpoints/film_unet/best_closure.pt \
    --pairs_dir   data/processed/pairs \
    --triplets_manifest data/manifests/hawaii_triplets_strict.parquet \
    --out_dir     experiments/enhanced/outputs \
    --skip_inference

# Skip metrics that need unw_phase.tif (metrics 2/3/4):
py eval/compute_metrics.py \
    --checkpoint  experiments/enhanced/checkpoints/film_unet/best_closure.pt \
    --pairs_dir   data/processed/pairs \
    --triplets_manifest data/manifests/hawaii_triplets_strict.parquet \
    --out_dir     experiments/enhanced/outputs \
    --skip_snaphu_metrics

# Evaluate on held-out test split only (last 15% by date):
py eval/compute_metrics.py \
    --checkpoint  experiments/enhanced/checkpoints/film_unet/best_closure.pt \
    --pairs_dir   data/processed/pairs \
    --triplets_manifest data/manifests/hawaii_triplets_strict.parquet \
    --out_dir     experiments/enhanced/outputs \
    --test_only \
    --test_frac 0.15

# Force CPU (e.g. no GPU on login node):
py eval/compute_metrics.py \
    --checkpoint  experiments/enhanced/checkpoints/film_unet/best_closure.pt \
    --pairs_dir   data/processed/pairs \
    --triplets_manifest data/manifests/hawaii_triplets_strict.parquet \
    --out_dir     experiments/enhanced/outputs \
    --device cpu
```

**All flags**:

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | required | Path to FiLMUNet `.pt` checkpoint |
| `--pairs_dir` | required | Root directory of processed pair subdirectories |
| `--triplets_manifest` | required | Path to triplets parquet file |
| `--out_dir` | `experiments/enhanced/outputs` | Directory for CSV + figures |
| `--tile_size` | 256 | Inference tile size (pixels) |
| `--stride` | 128 | Inference tile stride (pixels) |
| `--test_frac` | 0.15 | Fraction of pairs in held-out test split |
| `--skip_inference` | off | Skip FiLMUNet forward pass; reuse `ifg_film_unet.tif` if it exists |
| `--force_inference` | off | Re-run FiLMUNet inference even if `ifg_film_unet.tif` already exists |
| `--skip_snaphu_metrics` | off | Skip metrics 2/3/4 (which need `unw_phase.tif`) |
| `--copernicus_dem_dir` | `None` | Directory with Copernicus GLO-30 tiles; enables M4 DEM NMAD |
| `--test_only` | off | Evaluate on the last `test_frac` pairs by date only |
| `--device` | `None` (auto-detect) | PyTorch device string (`cuda`, `cpu`, `cuda:1`, …) |

**Outputs** in `{out_dir}/`:

| File pattern | Content |
|---|---|
| `metrics_{eval_tag}.csv` | All 5 metrics × {goldstein, film_unet, improvement_pct} |
| `figures/{eval_tag}_closure_histogram.png` | Distribution of per-triplet closure errors |
| `figures/{eval_tag}_phase_comparison.png` | Side-by-side raw / Goldstein / FiLMUNet phase |
| `figures/{eval_tag}_temporal_residual_bar.png` | Bar chart of Metric 5 for both methods |
| `logs/eval_{eval_tag}.log` | Full eval log (same timestamp as output files) |

where `eval_tag = eval_{checkpoint_stem}_{YYYYMMDD_HHMM}`.

**Current metric values** (2026-03-21 full eval, 224 pairs, `raw2gold_30ep_20260319_2139_final.pt`):

| Metric | Goldstein | FiLMUNet | Notes |
|--------|-----------|----------|-------|
| M1 Triplet Closure | 1.018 rad | 1.021 rad | No improvement — closure loss inactive during training |
| M2 Unwrap Success Rate | 0.256 | N/A | SNAPHU only run on Goldstein; needs re-run on denoised |
| M3 Usable Pairs | 0.000 | 0.000 | M1 still > 0.5 gate |
| M4 DEM NMAD | N/A | N/A | No reference DEM available |
| M5 Temporal Residual | 0.050 rad | 2.977 rad | FiLMUNet value is INVALID — M5 bug being fixed |

Pending: retrain with active closure loss → re-eval expected to show real M1/M5 improvement.

---

## Step 8 — Classical Baseline DEM (Legacy)

**Script**: `experiments/baseline/run_baseline.py`

**Purpose**: Simplified phase-to-height conversion using the classical InSAR formula `h = φ / (k × sin(θ))`, where k is the wavenumber and θ is the incidence angle. Reads pre-unwrapped phase from a GeoTIFF and writes a relative height DEM. Useful as the no-ML baseline to compare against.

```bash
py experiments/baseline/run_baseline.py \
    --config configs/experiment/baseline_sentinel1.yaml
```

**Config** (`configs/experiment/baseline_sentinel1.yaml`):
```yaml
interferogram_path: data/processed/pairs/<pair_dir>/ifg_goldstein.tif
unwrapped_phase_path: data/processed/pairs/<pair_dir>/unw_phase.tif
coherence_path: data/processed/pairs/<pair_dir>/coherence.tif
output_dem_path: data/reference/baseline_dem.tif
wavelength_m: 0.031   # X-band Capella (0.031 m = 31 mm)
incidence_angle_deg: 45.0
perpendicular_baseline_m: 200.0
```

**Output**: GeoTIFF at `output_dem_path` — relative height in metres (no absolute reference). Requires pre-unwrapped phase; run Step 5 first.

---

## Step 9 — Legacy U-Net Baseline (Superseded)

**Script**: `experiments/enhanced/train_unet.py`

**Purpose**: Early proof-of-concept supervised U-Net for InSAR DEM enhancement. Requires co-registered interferogram, coherence, and a reference DEM. Superseded by `train_film_unet.py` (Step 6) which is fully self-supervised and FiLM-conditioned. Kept for ablation comparison.

```bash
py experiments/enhanced/train_unet.py \
    --data_config  configs/data/sentinel1_example.yaml \
    --model_config configs/model/unet_baseline.yaml \
    --train_config configs/train/default.yaml
```

**Config** (`configs/data/sentinel1_example.yaml`):
```yaml
interferogram_path: data/processed/example/ifg_goldstein.tif
coherence_path:     data/processed/example/coherence.tif
reference_dem_path: data/reference/reference_dem.tif
tile_size: 256
stride: 256
```

**Output**: checkpoint at `experiments/enhanced/checkpoints/unet_baseline_final.pt`.

**Note**: `reference_dem_path` is required but not available for the Capella contest dataset. Use `train_film_unet.py` instead for all contest work.

---

## GitHub Issue Workflow

```bash
# Create all project issues (idempotent — safe to re-run):
bash scripts/create_github_issues.sh

# List open issues:
gh issue list --repo getnetdemil/Learning-Assisted-InSAR-DEM-Enhancement

# View a specific issue:
gh issue view 20

# Close an issue manually:
gh issue close 20 --comment "Done — metrics verified."
```

---

## Typical Full Pipeline (end-to-end reproduction)

```bash
export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement
PREFIX=/scratch/gdemil24/hrwsi_s3client/torch-gpu

# ── 1. Build manifest (DONE — skip if full_index.parquet exists) ──────────────
conda run --prefix $PREFIX python scripts/download_subset.py \
    --index_only --out_manifest data/manifests/full_index.parquet

# ── 2. Download Hawaii SLCs (DONE — 221 SLCs in data/raw/AOI_000/) ───────────
conda run --prefix $PREFIX python scripts/download_subset.py \
    --manifest data/manifests/full_index.parquet \
    --aoi_filter AOI_000 --out_dir data/raw/ --n_workers 8

# ── 3. Build pair graph (DONE — 8834 pairs, 24171 triplets) ──────────────────
conda run --prefix $PREFIX python -c "
from src.insar_processing.pair_graph import PairGraphConfig, build_pair_graph, find_triplets
import pandas as pd
m = pd.read_parquet('data/manifests/full_index.parquet')
hawaii = m[m.aoi == 'AOI_000']
cfg = PairGraphConfig(dt_max_days=365, dinc_max_deg=5.0)
pairs_df = build_pair_graph(hawaii, cfg)
pairs_df.to_parquet('data/manifests/hawaii_pairs.parquet', index=False)
triplets_df = find_triplets(pairs_df)
triplets_df.to_parquet('data/manifests/hawaii_triplets_strict.parquet', index=False)
"

# ── 4. Preprocess top-100 pairs (DONE) ───────────────────────────────────────
conda run --prefix $PREFIX python scripts/preprocess_pairs.py \
    --pairs_manifest data/manifests/hawaii_pairs.parquet \
    --raw_dir data/raw/AOI_000 \
    --out_dir data/processed/pairs \
    --max_pairs 100 --adaptive --n_workers 4

# ── 5. Select + preprocess triplet-completing pairs (DONE — 62 pairs) ────────
conda run --prefix $PREFIX python scripts/select_triplet_completing_pairs.py \
    --pairs_manifest    data/manifests/hawaii_pairs.parquet \
    --triplets_manifest data/manifests/hawaii_triplets_strict.parquet \
    --processed_dir     data/processed/pairs \
    --out_parquet       data/manifests/triplet_completing_pairs.parquet

conda run --prefix $PREFIX python scripts/preprocess_pairs.py \
    --pairs_manifest data/manifests/triplet_completing_pairs.parquet \
    --raw_dir data/raw/AOI_000 \
    --out_dir data/processed/pairs \
    --patch_size 4096 --adaptive --n_workers 4

# ── 5b. Patch FiLM metadata (DONE — 224/224 files) ───────────────────────────
conda run --prefix $PREFIX python scripts/patch_coreg_meta.py \
    --pairs_dir data/processed/pairs \
    --manifest  data/manifests/hawaii_pairs.parquet

# ── 6. Unwrap with SNAPHU (RUNNING — 224 pairs, ~23h) ────────────────────────
# Use direct Python path for real-time output visibility on large jobs:
$PREFIX/bin/python -u scripts/unwrap_snaphu.py \
    --pairs_dir data/processed/pairs \
    --mode DEFO --coh_threshold 0.1 --nlooks 9.0 --workers 4

# ── 7. Train FiLMUNet (RUNNING — epoch 5/50) ─────────────────────────────────
conda run --prefix $PREFIX python experiments/enhanced/train_film_unet.py \
    --data_config  configs/data/capella_aoi_selection.yaml \
    --model_config configs/model/film_unet.yaml \
    --train_config configs/train/contest.yaml

# ── 8. Evaluate all 5 contest metrics ────────────────────────────────────────
conda run --prefix $PREFIX python eval/compute_metrics.py \
    --checkpoint  experiments/enhanced/checkpoints/film_unet/best_closure.pt \
    --pairs_dir   data/processed/pairs \
    --triplets_manifest data/manifests/hawaii_triplets_strict.parquet \
    --out_dir     experiments/enhanced/outputs
```

---

## Step 10 — Multi-AOI Zero-Shot Transfer

Apply the trained Hawaii checkpoint to additional AOIs without retraining.

| AOI | Location | Scenes | Pairs | Triplets | Status |
|-----|----------|--------|-------|----------|--------|
| AOI_022 | Spain (Pyrenees) | 13 | 13 | 8 | manifests ✓ |
| AOI_024 | W. Australia | ~100 | 2025 | 28126 | manifests ✓ |
| AOI_004 | SF Bay Area | — | TBD | TBD | download ✓ |
| AOI_005 | Sierra Nevada, CA | 36 | TBD | TBD | download ✓ |
| AOI_008 | Los Angeles, CA | — | TBD | TBD | download ✓ |
| AOI_009 | Detroit / Great Lakes | 10 | TBD | TBD | download ✓ |
| AOI_017 | Vermont / White Mtns | 2 | TBD | TBD | download ✓ |
| AOI_033 | Tasmania (alpine) | 15 | TBD | TBD | download ✓ |

### 10a — Download raw SLCs (skip if already present)

```bash
# AOI_004 (SF Bay)
TASK_NAME="download_aoi004" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/download_subset.py --manifest data/manifests/full_index.parquet --aoi_filter AOI_004 --out_dir data/raw/ --n_workers 6 | tee "$LOG"
```

```bash
# AOI_005 (Sierra Nevada)
TASK_NAME="download_aoi005" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/download_subset.py --manifest data/manifests/full_index.parquet --aoi_filter AOI_005 --out_dir data/raw/ --n_workers 6 | tee "$LOG"
```

```bash
# AOI_008 (Los Angeles)
TASK_NAME="download_aoi008" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/download_subset.py --manifest data/manifests/full_index.parquet --aoi_filter AOI_008 --out_dir data/raw/ --n_workers 6 | tee "$LOG"
```

```bash
# AOI_009 (Detroit / Great Lakes)
TASK_NAME="download_aoi009" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/download_subset.py --manifest data/manifests/full_index.parquet --aoi_filter AOI_009 --out_dir data/raw/ --n_workers 6 | tee "$LOG"
```

```bash
# AOI_017 (Vermont / White Mtns)
TASK_NAME="download_aoi017" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/download_subset.py --manifest data/manifests/full_index.parquet --aoi_filter AOI_017 --out_dir data/raw/ --n_workers 6 | tee "$LOG"
```

```bash
# AOI_022 (Spain — Pyrenees)
TASK_NAME="download_aoi022" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/download_subset.py --manifest data/manifests/full_index.parquet --aoi_filter AOI_022 --out_dir data/raw/ --n_workers 6 | tee "$LOG"
```

```bash
# AOI_024 (W. Australia)
TASK_NAME="download_aoi024" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/download_subset.py --manifest data/manifests/full_index.parquet --aoi_filter AOI_024 --out_dir data/raw/ --n_workers 6 | tee "$LOG"
```

```bash
# AOI_033 (Tasmania)
TASK_NAME="download_aoi033" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/download_subset.py --manifest data/manifests/full_index.parquet --aoi_filter AOI_033 --out_dir data/raw/ --n_workers 6 | tee "$LOG"
```

### 10b — Build pair + triplet manifests (all 8 AOIs)

`pair_graph.py` is a library — write the build script to a temp file first, then run it.

```bash
# Step 1: write the manifest-build script
cat > /tmp/build_manifests_all_aois.py << 'PYEOF'
import os, sys
os.chdir("/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement")
sys.path.insert(0, os.getcwd())
import pandas as pd
from src.insar_processing.pair_graph import PairGraphConfig, build_pair_graph, find_triplets

df = pd.read_parquet("data/manifests/full_index.parquet")
cfg = PairGraphConfig(dt_max_days=365.0, dinc_max_deg=5.0, require_same_orbit=True, require_same_look=True)

aois = [("AOI_004","aoi004"),("AOI_005","aoi005"),("AOI_008","aoi008"),("AOI_009","aoi009"),
        ("AOI_017","aoi017"),("AOI_022","aoi022"),("AOI_024","aoi024"),("AOI_033","aoi033")]
for aoi, tag in aois:
    sub = df[df["aoi"] == aoi]
    print(f"\n{aoi}: {len(sub)} acquisitions")
    edges    = build_pair_graph(sub, cfg)
    triplets = find_triplets(edges)
    edges.to_parquet(f"data/manifests/{tag}_pairs.parquet", index=False)
    triplets.to_parquet(f"data/manifests/{tag}_triplets.parquet", index=False)
    print(f"  → {len(edges)} pairs, {len(triplets)} triplets")
PYEOF
```

```bash
# Step 2: run it (AOI_022 + AOI_024 manifests already exist — will overwrite, which is fine)
TASK_NAME="build_manifests_all_aois" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python /tmp/build_manifests_all_aois.py | tee "$LOG"
```

Verify: `ls data/manifests/aoi*_pairs.parquet` — should list 8 files.

### 10c — Preprocess pairs

```bash
# AOI_004 (SF Bay)
TASK_NAME="preprocess_aoi004" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/preprocess_pairs.py --pairs_manifest data/manifests/aoi004_pairs.parquet --raw_dir data/raw/AOI_004 --out_dir data/processed/AOI_004 --dt_max 365 --n_workers 4 | tee "$LOG"
```

```bash
# AOI_005 (Sierra Nevada)
TASK_NAME="preprocess_aoi005" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/preprocess_pairs.py --pairs_manifest data/manifests/aoi005_pairs.parquet --raw_dir data/raw/AOI_005 --out_dir data/processed/AOI_005 --dt_max 365 --n_workers 4 | tee "$LOG"
```

```bash
# AOI_008 (Los Angeles)
TASK_NAME="preprocess_aoi008" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/preprocess_pairs.py --pairs_manifest data/manifests/aoi008_pairs.parquet --raw_dir data/raw/AOI_008 --out_dir data/processed/AOI_008 --dt_max 365 --n_workers 4 | tee "$LOG"
```

```bash
# AOI_009 (Detroit)
TASK_NAME="preprocess_aoi009" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/preprocess_pairs.py --pairs_manifest data/manifests/aoi009_pairs.parquet --raw_dir data/raw/AOI_009 --out_dir data/processed/AOI_009 --dt_max 365 --n_workers 4 | tee "$LOG"
```

```bash
# AOI_017 (Vermont)
TASK_NAME="preprocess_aoi017" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/preprocess_pairs.py --pairs_manifest data/manifests/aoi017_pairs.parquet --raw_dir data/raw/AOI_017 --out_dir data/processed/AOI_017 --dt_max 365 --n_workers 4 | tee "$LOG"
```

```bash
# AOI_022 (Spain)
TASK_NAME="preprocess_aoi022" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/preprocess_pairs.py --pairs_manifest data/manifests/aoi022_pairs.parquet --raw_dir data/raw/AOI_022 --out_dir data/processed/AOI_022 --dt_max 365 --n_workers 4 | tee "$LOG"
```

```bash
# AOI_024 (W. Australia — large: 2025 pairs, use more workers)
TASK_NAME="preprocess_aoi024" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/preprocess_pairs.py --pairs_manifest data/manifests/aoi024_pairs.parquet --raw_dir data/raw/AOI_024 --out_dir data/processed/AOI_024 --dt_max 365 --n_workers 6 | tee "$LOG"
```

```bash
# AOI_033 (Tasmania)
TASK_NAME="preprocess_aoi033" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/preprocess_pairs.py --pairs_manifest data/manifests/aoi033_pairs.parquet --raw_dir data/raw/AOI_033 --out_dir data/processed/AOI_033 --dt_max 365 --n_workers 4 | tee "$LOG"
```

### 10d — Patch coreg metadata

```bash
# AOI_004
TASK_NAME="patch_meta_aoi004" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/patch_coreg_meta.py --pairs_dir data/processed/AOI_004 --manifest data/manifests/full_index.parquet | tee "$LOG"
```

```bash
# AOI_005
TASK_NAME="patch_meta_aoi005" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/patch_coreg_meta.py --pairs_dir data/processed/AOI_005 --manifest data/manifests/full_index.parquet | tee "$LOG"
```

```bash
# AOI_008
TASK_NAME="patch_meta_aoi008" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/patch_coreg_meta.py --pairs_dir data/processed/AOI_008 --manifest data/manifests/full_index.parquet | tee "$LOG"
```

```bash
# AOI_009
TASK_NAME="patch_meta_aoi009" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/patch_coreg_meta.py --pairs_dir data/processed/AOI_009 --manifest data/manifests/full_index.parquet | tee "$LOG"
```

```bash
# AOI_017
TASK_NAME="patch_meta_aoi017" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/patch_coreg_meta.py --pairs_dir data/processed/AOI_017 --manifest data/manifests/full_index.parquet | tee "$LOG"
```

```bash
# AOI_022
TASK_NAME="patch_meta_aoi022" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/patch_coreg_meta.py --pairs_dir data/processed/AOI_022 --manifest data/manifests/full_index.parquet | tee "$LOG"
```

```bash
# AOI_024
TASK_NAME="patch_meta_aoi024" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/patch_coreg_meta.py --pairs_dir data/processed/AOI_024 --manifest data/manifests/full_index.parquet | tee "$LOG"
```

```bash
# AOI_033
TASK_NAME="patch_meta_aoi033" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/patch_coreg_meta.py --pairs_dir data/processed/AOI_033 --manifest data/manifests/full_index.parquet | tee "$LOG"
```

### 10e — SNAPHU unwrapping

```bash
# AOI_004
TASK_NAME="unwrap_aoi004" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python -u scripts/unwrap_snaphu.py --pairs_dir data/processed/AOI_004 --mode DEFO --nlooks 9.0 --coh_threshold 0.1 --workers 2 | tee "$LOG"
```

```bash
# AOI_005 (Sierra Nevada)
TASK_NAME="unwrap_aoi005" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python -u scripts/unwrap_snaphu.py --pairs_dir data/processed/AOI_005 --mode DEFO --nlooks 9.0 --coh_threshold 0.1 --workers 3 | tee "$LOG"
```

```bash
# AOI_008 (Los Angeles)
TASK_NAME="unwrap_aoi008" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python -u scripts/unwrap_snaphu.py --pairs_dir data/processed/AOI_008 --mode DEFO --nlooks 9.0 --coh_threshold 0.1 --workers 3 | tee "$LOG"
```

```bash
# AOI_009 (Detroit)
TASK_NAME="unwrap_aoi009" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python -u scripts/unwrap_snaphu.py --pairs_dir data/processed/AOI_009 --mode DEFO --nlooks 9.0 --coh_threshold 0.1 --workers 2 | tee "$LOG"
```

```bash
# AOI_017 (Vermont)
TASK_NAME="unwrap_aoi017" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python -u scripts/unwrap_snaphu.py --pairs_dir data/processed/AOI_017 --mode DEFO --nlooks 9.0 --coh_threshold 0.1 --workers 2 | tee "$LOG"
```

```bash
# AOI_022 (Spain)
TASK_NAME="unwrap_aoi022" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python -u scripts/unwrap_snaphu.py --pairs_dir data/processed/AOI_022 --mode DEFO --nlooks 9.0 --coh_threshold 0.1 --workers 2 | tee "$LOG"
```

```bash
# AOI_024 (W. Australia — large)
TASK_NAME="unwrap_aoi024" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python -u scripts/unwrap_snaphu.py --pairs_dir data/processed/AOI_024 --mode DEFO --nlooks 9.0 --coh_threshold 0.1 --workers 4 | tee "$LOG"
```

```bash
# AOI_033 (Tasmania)
TASK_NAME="unwrap_aoi033" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python -u scripts/unwrap_snaphu.py --pairs_dir data/processed/AOI_033 --mode DEFO --nlooks 9.0 --coh_threshold 0.1 --workers 2 | tee "$LOG"
```

### 10f — Zero-shot eval with Hawaii checkpoint

```bash
# Set checkpoint once
CKPT="experiments/enhanced/checkpoints/film_unet/raw2gold_closure_20260321_1852/raw2gold_closure_20260321_1852_final.pt"
```

```bash
# AOI_004
TASK_NAME="eval_zeroshot_aoi004" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python eval/compute_metrics.py --checkpoint "$CKPT" --pairs_dir data/processed/AOI_004 --triplets_manifest data/manifests/aoi004_triplets.parquet --out_dir experiments/enhanced/outputs/aoi004_eval | tee "$LOG"
```

```bash
# AOI_005
TASK_NAME="eval_zeroshot_aoi005" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python eval/compute_metrics.py --checkpoint "$CKPT" --pairs_dir data/processed/AOI_005 --triplets_manifest data/manifests/aoi005_triplets.parquet --out_dir experiments/enhanced/outputs/aoi005_eval | tee "$LOG"
```

```bash
# AOI_008
TASK_NAME="eval_zeroshot_aoi008" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python eval/compute_metrics.py --checkpoint "$CKPT" --pairs_dir data/processed/AOI_008 --triplets_manifest data/manifests/aoi008_triplets.parquet --out_dir experiments/enhanced/outputs/aoi008_eval | tee "$LOG"
```

```bash
# AOI_009
TASK_NAME="eval_zeroshot_aoi009" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python eval/compute_metrics.py --checkpoint "$CKPT" --pairs_dir data/processed/AOI_009 --triplets_manifest data/manifests/aoi009_triplets.parquet --out_dir experiments/enhanced/outputs/aoi009_eval | tee "$LOG"
```

```bash
# AOI_017
TASK_NAME="eval_zeroshot_aoi017" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python eval/compute_metrics.py --checkpoint "$CKPT" --pairs_dir data/processed/AOI_017 --triplets_manifest data/manifests/aoi017_triplets.parquet --out_dir experiments/enhanced/outputs/aoi017_eval | tee "$LOG"
```

```bash
# AOI_022 (Spain)
TASK_NAME="eval_zeroshot_aoi022" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python eval/compute_metrics.py --checkpoint "$CKPT" --pairs_dir data/processed/AOI_022 --triplets_manifest data/manifests/aoi022_triplets.parquet --out_dir experiments/enhanced/outputs/aoi022_eval | tee "$LOG"
```

```bash
# AOI_024 (W. Australia)
TASK_NAME="eval_zeroshot_aoi024" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python eval/compute_metrics.py --checkpoint "$CKPT" --pairs_dir data/processed/AOI_024 --triplets_manifest data/manifests/aoi024_triplets.parquet --out_dir experiments/enhanced/outputs/aoi024_eval | tee "$LOG"
```

```bash
# AOI_033 (Tasmania)
TASK_NAME="eval_zeroshot_aoi033" && DATE=$(date +%Y%m%d_%H%M%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python eval/compute_metrics.py --checkpoint "$CKPT" --pairs_dir data/processed/AOI_033 --triplets_manifest data/manifests/aoi033_triplets.parquet --out_dir experiments/enhanced/outputs/aoi033_eval | tee "$LOG"
```

Results land in `experiments/enhanced/outputs/{tag}_eval/metrics_*.csv` with M1–M5 columns for Goldstein vs FiLMUNet.
