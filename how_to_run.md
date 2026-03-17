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

**Purpose**: Enumerate all valid interferometric pairs from the SLC manifest, score each by `Q_ij = 1/(Δt+1) × 1/(1+Δinc)`, compute perpendicular baseline `B_perp` from satellite state vectors, and enumerate all valid closure triplets (a→b, b→c, a→c) within temporal/incidence constraints.

```bash
py -c "
from insar_processing.pair_graph import build_pair_graph, enumerate_triplets
import pandas as pd

manifest = pd.read_parquet('data/manifests/full_index.parquet')
hawaii = manifest[manifest.aoi == 'AOI_000']

# Build pair graph: all pairs with Δt ≤ 365 days, Δinc ≤ 5°
pairs_df = build_pair_graph(hawaii, dt_max=365, dinc_max=5.0)
pairs_df.to_parquet('data/manifests/hawaii_pairs.parquet', index=False)

# Enumerate strict triplets: Δt ≤ 60 days, Δinc ≤ 2°
triplets_df = enumerate_triplets(pairs_df, dt_max=60, dinc_max=2.0)
triplets_df.to_parquet('data/manifests/hawaii_triplets_strict.parquet', index=False)

print(f'{len(pairs_df)} pairs, {len(triplets_df)} triplets')
"
```

**Outputs**:
- `data/manifests/hawaii_pairs.parquet` — columns: `id_ref`, `id_sec`, `datetime_ref`, `datetime_sec`, `dt_days`, `dinc_deg`, `orbit_state`, `look_direction`, `incidence_ref`, `incidence_sec`, `q_score`, `aoi`, `bperp_m`
- `data/manifests/hawaii_triplets_strict.parquet` — columns: `id_a`, `id_b`, `id_c`

**Current status**: DONE — 8,834 pairs, 24,171 triplets.

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
| `--dt_max` | 60.0 | Max temporal baseline filter (days) |
| `--dinc_max` | 2.0 | Max incidence angle difference filter (degrees) |
| `--patch_size` | 4096 | SLC patch size in pixels (square) |
| `--looks_range` | 5 | Range looks for coherence window |
| `--looks_azimuth` | 5 | Azimuth looks for coherence window |
| `--goldstein_alpha` | 0.5 | Goldstein filter strength α ∈ [0,1] (used when `--adaptive` is off) |
| `--adaptive` | off | Use coherence-adaptive α (stronger in low-coherence regions) |
| `--n_workers` | 4 | Parallel worker threads |

**Output per pair** in `data/processed/pairs/<id_ref>__<id_sec>/`:

| File | Format | Content |
|------|--------|---------|
| `ifg_raw.tif` | 2-band float32 GeoTIFF | Re, Im of unnormalised interferogram |
| `ifg_goldstein.tif` | 2-band float32 GeoTIFF | Goldstein-filtered Re, Im |
| `coherence.tif` | 1-band float32 GeoTIFF | Coherence values ∈ [0,1] |
| `coreg_meta.json` | JSON | `id_ref`, `id_sec`, `dt_days`, `dinc_deg`, `q_score`, `bperp_m`, `row_offset_px`, `col_offset_px`, `patch_size`, `patch_row_ref`, `patch_col_ref` |

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
```

```bash
# Unwrap all processed pairs with DEFO mode (2 workers):
py scripts/unwrap_snaphu.py \
    --pairs_dir data/processed/pairs \
    --mode DEFO \
    --workers 2

# Use TOPO mode for DEM estimation:
py scripts/unwrap_snaphu.py \
    --pairs_dir data/processed/pairs \
    --mode TOPO \
    --workers 2

# Limit to 50 pairs, stricter coherence mask, custom nlooks:
py scripts/unwrap_snaphu.py \
    --pairs_dir data/processed/pairs \
    --max_pairs 50 \
    --mode DEFO \
    --coh_threshold 0.2 \
    --nlooks 25.0 \
    --workers 4

# Write unwrapped outputs to a separate directory:
py scripts/unwrap_snaphu.py \
    --pairs_dir data/processed/pairs \
    --out_dir data/processed/unwrapped \
    --mode DEFO \
    --workers 2
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

**Current status**: PENDING — snaphu-py not yet installed.

---

## Step 6 — Train FiLMUNet (Primary Model)

**Script**: `experiments/enhanced/train_film_unet.py`

**Purpose**: Self-supervised training of the FiLM-conditioned U-Net on preprocessed Capella interferogram tiles. No clean reference interferograms needed — uses Noise2Noise (N2N) via sub-look splits + physics-consistency losses (closure, temporal, gradient). Reads pairs from `data/processed/pairs/`, splits temporally into train/val/test, and trains for the configured number of epochs. Checkpoints every 5 epochs + best validation closure.

Three YAML configs control all behaviour:
- `configs/data/capella_aoi_selection.yaml` — data paths and split settings
- `configs/model/film_unet.yaml` — architecture (features, embed_dim, metadata_dim)
- `configs/train/contest.yaml` — learning rate, epochs, batch size, loss weights

```bash
# Fresh training run:
py experiments/enhanced/train_film_unet.py \
    --data_config  configs/data/capella_aoi_selection.yaml \
    --model_config configs/model/film_unet.yaml \
    --train_config configs/train/contest.yaml

# Resume from a saved checkpoint (continues from the saved epoch):
py experiments/enhanced/train_film_unet.py \
    --data_config  configs/data/capella_aoi_selection.yaml \
    --model_config configs/model/film_unet.yaml \
    --train_config configs/train/contest.yaml \
    --resume experiments/enhanced/checkpoints/film_unet/epoch_020.pt
```

**All flags**:

| Flag | Description |
|------|-------------|
| `--data_config` | Path to data YAML (required) |
| `--model_config` | Path to model YAML (required) |
| `--train_config` | Path to training YAML (required) |
| `--resume` | Path to `.pt` checkpoint to resume from |

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

**Checkpoints** written to `experiments/enhanced/checkpoints/film_unet/`:

| File | Saved when |
|------|-----------|
| `epoch_005.pt`, `epoch_010.pt`, … | Every 5 epochs |
| `best_closure.pt` | Best validation closure loss |
| `best_unwrap.pt` | Best validation unwrap loss (placeholder) |
| `final.pt` | End of last epoch |

Each checkpoint stores: model state dict, optimizer state, epoch number, all three config dicts, git commit hash.

**Current status**: IN PROGRESS — checkpoint at `experiments/enhanced/checkpoints/film_unet/best_closure.pt`.

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
| `--skip_snaphu_metrics` | off | Skip metrics 2/3/4 (which need `unw_phase.tif`) |
| `--test_only` | off | Evaluate on the last `test_frac` pairs by date only |
| `--device` | auto | PyTorch device string (`cuda`, `cpu`, `cuda:1`, …) |

**Outputs** in `experiments/enhanced/outputs/`:

| File | Content |
|------|---------|
| `metrics_comparison.csv` | All 5 metrics × {goldstein, film_unet, improvement_pct} |
| `figures/closure_histogram.png` | Distribution of per-triplet closure errors |
| `figures/phase_comparison.png` | Side-by-side raw / Goldstein / FiLMUNet phase images |
| `figures/temporal_residual_bar.png` | Bar chart of Metric 5 for both methods |

**Current metric values** (2026-03-17, `--skip_inference --skip_snaphu_metrics`, 162 pairs):

| Metric | Goldstein | FiLMUNet | Status |
|--------|-----------|----------|--------|
| M1 Triplet Closure Error | 1.018 rad | — (needs inference) | 62 triplets ✓ |
| M2 Unwrap Success Rate | N/A | N/A | Needs SNAPHU |
| M3 Usable Pairs Fraction | 0.000 | — | Needs inference |
| M4 DEM NMAD | N/A | N/A | Needs ref DEM |
| M5 Temporal Residual | 0.050 rad | — (needs inference) | P=162>T=138 ✓ |

**Note**: M3=0.0 for Goldstein is correct — the raw closure error (1.018 rad) exceeds the 0.5 rad usable-pair gate. FiLMUNet inference is expected to reduce M1 below the gate threshold, yielding M3 > 0.

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
from insar_processing.pair_graph import build_pair_graph, enumerate_triplets
import pandas as pd
m = pd.read_parquet('data/manifests/full_index.parquet')
hawaii = m[m.aoi == 'AOI_000']
pairs_df = build_pair_graph(hawaii, dt_max=365, dinc_max=5.0)
pairs_df.to_parquet('data/manifests/hawaii_pairs.parquet', index=False)
triplets_df = enumerate_triplets(pairs_df, dt_max=60, dinc_max=2.0)
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

# ── 6. Install snaphu-py then unwrap (PENDING) ───────────────────────────────
conda run --prefix $PREFIX pip install snaphu
conda run --prefix $PREFIX python scripts/unwrap_snaphu.py \
    --pairs_dir data/processed/pairs \
    --mode DEFO --coh_threshold 0.1 --nlooks 25 --workers 4

# ── 7. Train FiLMUNet (IN PROGRESS) ──────────────────────────────────────────
conda run --prefix $PREFIX python experiments/enhanced/train_film_unet.py \
    --data_config  configs/data/capella_aoi_selection.yaml \
    --model_config configs/model/film_unet.yaml \
    --train_config configs/train/contest.yaml \
    --resume experiments/enhanced/checkpoints/film_unet/best_closure.pt

# ── 8. Evaluate all 5 contest metrics ────────────────────────────────────────
conda run --prefix $PREFIX python eval/compute_metrics.py \
    --checkpoint  experiments/enhanced/checkpoints/film_unet/best_closure.pt \
    --pairs_dir   data/processed/pairs \
    --triplets_manifest data/manifests/hawaii_triplets_strict.parquet \
    --out_dir     experiments/enhanced/outputs
```
