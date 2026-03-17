Learning-Assisted InSAR DEM Enhancement
========================================

Contest submission for the **[IEEE GRSS 2026 Data Fusion Contest](https://www.grss-ieee.org/community/technical-committees/2026-ieee-grss-data-fusion-contest/)**
— *deadline: April 06, 2026.*

This repository implements a **stack-aware, self-supervised deep learning pipeline** for improving Interferometric SAR (InSAR) products over large, geometry-diverse Capella Space X-band SAR stacks, with downstream improvements in DEM quality and time-series consistency.

---

## Overview

The contest provides a Capella Space X-band SAR dataset: ~791 SLC collects across 39 AOIs, enabling thousands of interferometric pairs with substantial diversity in acquisition mode, incidence angle, look direction, and orbital geometry.

A competitive entry must be **pair-graph-aware** and **geometry-conditioned**. Our approach:

1. **Pair-graph construction** — Build a graph (nodes = SLC collects, edges = candidate pairs), score each edge by `Q_ij = 1/(Δt+1) × 1/(1+Δinc)`, compute perpendicular baseline `B_perp` from satellite state vectors, and enumerate closure triplets for network health tracking.
2. **Baseline InSAR products** — Sub-pixel coregistration via phase cross-correlation, interferogram formation, coherence estimation, Goldstein-Werner spectral filtering (coherence-adaptive), SNAPHU phase unwrapping.
3. **Self-supervised DL enhancement** — A FiLM-conditioned U-Net (`FiLMUNet`) trained without ground-truth labels using Noise2Noise via sub-look splits + closure-consistency + temporal-consistency + fringe-preservation losses. Outputs a denoised complex interferogram + per-pixel log-variance uncertainty.
4. **Uncertainty-weighted inversion** — Predicted uncertainty `σ²(p)` weights both SNAPHU unwrapping and SBAS stack inversion.
5. **Evaluation** — Five physics-linked contest metrics computed for Goldstein baseline vs. FiLMUNet.

---

## Contest Metrics

| # | Metric | Definition | Target |
|---|--------|-----------|--------|
| 1 | Triplet closure error | `median|wrap(φ_ab + φ_bc − φ_ac)|` over all complete triplets | ↓ ≥ 30% |
| 2 | Unwrap success rate | Fraction of interferograms with ≥ 90% connected unwrap coverage | ↑ ≥ 15 pp |
| 3 | Percent usable pairs | Fraction passing coherence > 0.35 + unwrap + closure gates | ↑ ≥ 25 pp |
| 4 | DEM NMAD | `1.4826 × median(|e − median(e)|)` over stable terrain | ↓ ≥ 15% |
| 5 | Temporal consistency residual | SBAS inversion residual `‖W(Ax − φ̂)‖₂` | ↓ ≥ 20% |

**Current baseline values** (Goldstein, 162 processed pairs, Hawaii AOI, 2026-03-17):

| Metric | Goldstein | Status |
|--------|-----------|--------|
| M1 Triplet closure error | 1.018 rad | 62 complete triplets ✓ |
| M2 Unwrap success rate | — | Pending SNAPHU |
| M3 Usable pairs fraction | 0.000 | Pending FiLMUNet inference |
| M4 DEM NMAD | — | Pending reference DEM |
| M5 Temporal consistency | 0.050 rad | P=162 > T=138 ✓ |

---

## Environment

This project uses a shared conda environment at `/scratch/gdemil24/hrwsi_s3client/torch-gpu`.

```bash
# All scripts must be run as:
export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement
conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu python <script> <args>

# Verify GPU + key packages:
conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu python -c "
import torch; print('CUDA:', torch.cuda.is_available())
import rasterio; print('rasterio:', rasterio.__version__)
import boto3, pystac; print('boto3/pystac OK')
"
```

**Installed**: Python 3.10, PyTorch 2.4.0 + CUDA, rasterio, boto3, pystac, numpy, scipy, pandas, matplotlib, pyyaml.

To recreate from scratch:
```bash
conda create -n insar-dem python=3.10
conda activate insar-dem
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
pip install boto3 pystac snaphu
```

---

## Data Access

Data is on a **public AWS S3 bucket** — no credentials required.

| Resource | URL |
|----------|-----|
| S3 bucket | `s3://capella-open-data/` (region: `us-west-2`) |
| STAC catalog | `https://capella-open-data.s3.us-west-2.amazonaws.com/stac/catalog.json` |
| Contest collection | `capella-open-data-ieee-data-contest/collection.json` |

The catalog is a **static STAC** (JSON files, no `/search` endpoint) — use `pystac` directly. Access is unsigned (no AWS credentials needed).

```bash
# Verify access:
aws s3 ls --no-sign-request s3://capella-open-data/data/ | head -5
```

**Current data status**:
- 791 SLC items crawled, 39 AOIs assigned → `data/manifests/full_index.parquet`
- **AOI_000 (Hawaii)**: 221 SLCs downloaded → `data/raw/AOI_000/` (~497 GB)
- 8,834 pairs computed → `data/manifests/hawaii_pairs.parquet`
- 24,171 strict triplets (Δt ≤ 60d, Δinc ≤ 2°) → `data/manifests/hawaii_triplets_strict.parquet`
- 162 pairs fully preprocessed → `data/processed/pairs/` (224 directories; 162 unique by `coreg_meta.json`)

---

## Repository Structure

```
src/
  insar_processing/
    io.py                  # Rasterio raster I/O (load_raster, save_raster, resample_raster)
    baseline.py            # BaselineConfig + classical phase-to-height DEM (run_baseline)
    dataset_preparation.py # Sliding-window tiling (TileConfig, sliding_window, prepare_dem_tiles)
    pair_graph.py          # Pair-graph construction, Q_ij scoring, B_perp from state vectors
    geometry.py            # B_perp computation from satellite orbit state vectors
    filters.py             # Goldstein filter, coherence-adaptive Goldstein, boxcar coherence
    sublook.py             # FFT sub-look splitting for Noise2Noise training
  models/
    unet_baseline.py       # Vanilla U-Net (legacy, supervised, needs reference DEM)
    film_unet.py           # FiLM-conditioned U-Net — contest primary model
  losses/
    physics_losses.py      # N2N loss, uncertainty NLL, closure, temporal, gradient losses
  evaluation/
    dem_metrics.py         # RMSE, MAE, bias (with optional mask)
    closure_metrics.py     # All 5 contest metrics
  visualization/
    plots.py               # DEM comparison + error histogram figures

scripts/
  download_subset.py                  # STAC crawl → manifest; S3 parallel download
  preprocess_pairs.py                 # Coreg → interferogram → coherence → Goldstein
  select_triplet_completing_pairs.py  # Find pairs to close open 2-leg triplets
  unwrap_snaphu.py                    # SNAPHU phase unwrapping (snaphu-py backend)

eval/
  compute_metrics.py       # All 5 contest metrics: Goldstein vs FiLMUNet comparison table

experiments/
  baseline/
    run_baseline.py        # Classical phase-to-height DEM (baseline comparison)
  enhanced/
    train_film_unet.py     # FiLMUNet self-supervised training (primary)
    train_unet.py          # Legacy supervised U-Net (superseded)
    checkpoints/
      film_unet/
        best_closure.pt    # Best validation checkpoint (used for metric evaluation)
        epoch_050.pt       # Final epoch checkpoint
        final.pt           # End-of-training checkpoint

configs/
  data/
    capella_aoi_selection.yaml  # pairs_dir, tile_size, stride, temporal split fractions
    sentinel1_example.yaml      # Legacy data config for train_unet.py
  model/
    film_unet.yaml              # features, embed_dim, metadata_dim, in/out channels
    unet_baseline.yaml          # Legacy model config
  train/
    contest.yaml                # lr, epochs, batch_size, loss weights (N2N, closure, …)
    default.yaml                # Legacy train config
  experiment/
    baseline_sentinel1.yaml     # Wavelength, incidence, B_perp for run_baseline.py

data/
  manifests/
    full_index.parquet                    # 791 SLCs, 39 AOIs
    hawaii_pairs.parquet                  # 8,834 pairs with q_score, bperp_m
    hawaii_triplets_strict.parquet        # 24,171 closure triplets
    triplet_completing_pairs.parquet      # 62 pairs added to close open triplets
    overwritten_original_pairs.parquet    # 62 pairs re-processed after naming fix
  raw/AOI_000/                            # 221 SLC dirs (CInt16 GeoTIFF + extended JSON)
  processed/pairs/                        # 224 dirs; 162 unique pairs (coreg_meta.json)

experiments/enhanced/outputs/
  metrics_comparison.csv         # M1–M5 values for Goldstein vs FiLMUNet
  figures/
    closure_histogram.png
    phase_comparison.png
    temporal_residual_bar.png
```

---

## Quick Start

```bash
export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement
PY="conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu python"

# 1. Compute all 5 contest metrics (data already processed):
$PY eval/compute_metrics.py \
    --checkpoint  experiments/enhanced/checkpoints/film_unet/best_closure.pt \
    --pairs_dir   data/processed/pairs \
    --triplets_manifest data/manifests/hawaii_triplets_strict.parquet \
    --out_dir     experiments/enhanced/outputs

# 2. Run SNAPHU to unlock metrics 2/3/4 (install snaphu-py first):
$PY scripts/unwrap_snaphu.py \
    --pairs_dir data/processed/pairs \
    --mode DEFO --nlooks 25 --workers 4

# 3. Re-train FiLMUNet from scratch:
$PY experiments/enhanced/train_film_unet.py \
    --data_config  configs/data/capella_aoi_selection.yaml \
    --model_config configs/model/film_unet.yaml \
    --train_config configs/train/contest.yaml
```

See [`how_to_run.md`](how_to_run.md) for the complete step-by-step guide covering all 9 scripts with every CLI flag.

---

## Pipeline

```
SLC catalog (STAC)
       │
       ▼  scripts/download_subset.py
data/raw/AOI_000/<collect_id>/{*.tif, *_extended.json}
       │
       ▼  src/insar_processing/pair_graph.py
data/manifests/hawaii_pairs.parquet          (8,834 pairs, Q_ij scored)
data/manifests/hawaii_triplets_strict.parquet (24,171 closure triplets)
       │
       ▼  scripts/preprocess_pairs.py  +  scripts/select_triplet_completing_pairs.py
data/processed/pairs/<id_ref>__<id_sec>/
    ifg_raw.tif          (2-band float32: Re, Im)
    ifg_goldstein.tif    (2-band float32: Goldstein-filtered Re, Im)
    coherence.tif        (1-band float32: [0,1])
    coreg_meta.json      (dt_days, bperp_m, q_score, offsets, …)
       │
       ├──▶  scripts/unwrap_snaphu.py
       │         unw_phase.tif  (1-band float32: unwrapped phase, NaN=masked)
       │
       ▼  experiments/enhanced/train_film_unet.py
experiments/enhanced/checkpoints/film_unet/best_closure.pt
       │
       ▼  eval/compute_metrics.py
experiments/enhanced/outputs/
    metrics_comparison.csv
    figures/{closure_histogram, phase_comparison, temporal_residual_bar}.png
```

---

## Model

**FiLMUNet** (`src/models/film_unet.py`) — a geometry-conditioned encoder-decoder.

| Property | Value |
|----------|-------|
| Input | `(B, 3, H, W)` — Re, Im, coherence |
| Conditioning | `[Δt, θ_inc, θ_graze, B_perp, mode, look, SNR_proxy]` (7-dim, z-scored) |
| Output | `(B, 3, H, W)` — Re_denoised, Im_denoised, log_variance |
| Architecture | Encoder-decoder with skip connections; FiLM layers inject metadata at each scale |
| Features | `[32, 64, 128, 256]` (4 scales) |
| Parameters | ~4M |

**Training** (`src/losses/physics_losses.py`) — fully self-supervised:

| Loss | Weight | Purpose |
|------|--------|---------|
| Noise2Noise (sub-look MSE) | 1.0 | Main supervision — even/odd sub-look splits |
| Uncertainty NLL | 0.5 | Calibrate predicted log-variance |
| Closure consistency | 0.3 | Enforce `wrap(φ_ab + φ_bc − φ_ac) → 0` |
| Temporal consistency | 0.2 | Penalise deviation from SBAS phase rates |
| Gradient preservation | 0.1 | Preserve phase fringe sharpness |

---

## Implementation Status

| Component | File | Status |
|-----------|------|--------|
| STAC crawl + S3 download | `scripts/download_subset.py` | ✅ Done |
| Pair graph + B_perp | `src/insar_processing/pair_graph.py`, `geometry.py` | ✅ Done |
| Sub-look splitting | `src/insar_processing/sublook.py` | ✅ Done |
| Goldstein + adaptive filter | `src/insar_processing/filters.py` | ✅ Done |
| Preprocessing pipeline | `scripts/preprocess_pairs.py` | ✅ Done |
| Triplet network repair | `scripts/select_triplet_completing_pairs.py` | ✅ Done |
| FiLM-conditioned U-Net | `src/models/film_unet.py` | ✅ Done |
| Physics loss suite | `src/losses/physics_losses.py` | ✅ Done |
| FiLMUNet training | `experiments/enhanced/train_film_unet.py` | ✅ Done (50 epochs) |
| All 5 contest metrics | `src/evaluation/closure_metrics.py` | ✅ Done |
| Metric evaluation script | `eval/compute_metrics.py` | ✅ Done |
| SNAPHU unwrapping | `scripts/unwrap_snaphu.py` | ⏳ Pending (snaphu-py install) |
| Reference DEM (M4) | — | ⏳ Pending |
| Ablation studies (5 variants) | — | ⏳ Mar 21–24 |
| Zero-shot transfer AOI_008 | — | ⏳ Mar 25–27 |
| REPRODUCIBILITY.md | — | ⏳ Mar 28–Apr 1 |
| Contest paper (4 pages) | — | ⏳ Apr 1–5 |

---

## Key Design Decisions

- **Self-supervised only** — no ground-truth clean interferograms are used. Noise2Noise training uses FFT sub-look splits of each SLC acquisition.
- **FiLM conditioning** — Feature-wise Linear Modulation injects acquisition geometry metadata at every decoder scale, enabling the model to adapt filter strength to incidence angle, temporal baseline, B_perp, and look direction.
- **Model output is a denoised interferogram, not a DEM** — absolute DEM requires SNAPHU unwrapping + SBAS inversion downstream.
- **Triplet-aware pair selection** — the top-q-score pairs alone form no closed triplets (all Δt=3d; diagonal Δt=6d legs needed). `select_triplet_completing_pairs.py` identifies and adds the minimal set.
- **Full pair ID for output directories** — 80-char truncation caused silent data overwrites. Full 104-char pair IDs are used (well under the 255-char filesystem limit).
- **AOI-based train/val/test splits** — temporal ordering prevents geographic leakage; no random tile splitting.
- **Primary AOI**: AOI_000 (Hawaii) — 221 collects, both orbits, inc 35.8–56.3°, volcanic terrain ideal for InSAR.

---

## Development Plan

See [`plan.md`](plan.md) for the week-by-week implementation schedule, detailed phase descriptions, metric definitions, DL loss formulas, and ablation study design.

See [`how_to_run.md`](how_to_run.md) for the complete step-by-step execution guide with all CLI flags.

## Reproducibility

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) (to be created by Mar 28) for the STAC root URL, contest collection ID, exact download manifest with checksums, fixed random seeds, and deterministic training settings — all required for contest submission.
