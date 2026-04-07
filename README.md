Learning-Assisted InSAR DEM Enhancement
========================================

**IEEE GRSS 2026 Data Fusion Contest** submission — *Self-Supervised InSAR Phase Denoising via Geometry-Conditioned Noise2Noise and Closure Consistency.*

This repository implements **FiLMUNet**, a stack-aware, self-supervised deep learning pipeline for improving Interferometric SAR (InSAR) phase quality over large, geometry-diverse Capella Space X-band SAR stacks, with downstream improvements in DEM NMAD and temporal stack consistency.

---

## Key Results

FiLMUNet is evaluated against Goldstein filtering across three AOIs, including one zero-shot transfer site. The headline result is a **68% reduction in SBAS temporal residual** on the primary Hawaii stack.

| Metric | Gold AOI000 | Gold AOI024 | Gold AOI008 | Film AOI000 (Δ) | Film AOI024 (Δ) | Film AOI008† (Δ) |
|--------|:-----------:|:-----------:|:-----------:|:---------------:|:---------------:|:----------------:|
| Closure (rad) ↓ | 1.018 | 0.536 | 0.769 | **0.915** (−10%) | **0.468** (−6%) | 0.771 (0%) |
| Unwrap rate ↑ | 0.256 | 0.531 | 0.256 | **0.258** (+0.2pp) | **0.608** (+7pp) | 0.248 (−0.2pp) |
| DEM NMAD (m) ↓ | 40.13 | 18.32 | 40.13 | **39.44** (−2%) | **12.64** (−31%) | **39.40** (−2%) |
| Temporal res. (rad) ↓ | 1.158 | 1.069 | 1.486 | **0.367** (−68%) | **0.361** (−66%) | **1.450** (−2%) |

† AOI008 is zero-shot: the AOI024-trained checkpoint is applied without retraining.

- **AOI000** (Hawaii): cropped SLC, primary training + evaluation
- **AOI024** (W. Australia): full-scale SLC, separate training
- **AOI008** (Los Angeles): zero-shot transfer, no retraining

---

## Method

### FiLMUNet

A **FiLM-conditioned encoder-decoder** that denoises complex interferograms while conditioning on per-pair acquisition geometry.

| Property | Value |
|----------|-------|
| Input | `(B, 3, H, W)` — Re, Im, coherence |
| FiLM conditioning | `[Δt, θ_inc, θ_graze, B_perp, mode, look, SNR]` (7-dim, z-scored) |
| Output | `(B, 3, H, W)` — Re_denoised, Im_denoised, log_variance |
| Architecture | 4-scale encoder-decoder with skip connections; FiLM layers at each scale |
| Parameters | ~4 M |

### Self-Supervised Training

No clean reference interferograms are required. The loss combines:

| Component | Weight | Mechanism |
|-----------|--------|-----------|
| Noise2Noise (sub-look MSE) | 1.0 | Even/odd FFT sub-look splits share phase, have independent speckle |
| Uncertainty NLL | 0.5 | Calibrates the predicted per-pixel log-variance |
| Triplet closure consistency | 0.3 | Enforces `wrap(φ_ab + φ_bc − φ_ac) → 0` |
| Temporal consistency | 0.2 | Penalises deviation from the SBAS phase model |
| Gradient preservation | 0.1 | Preserves fringe sharpness |

The predicted uncertainty `σ²` propagates downstream as weights in SNAPHU phase unwrapping and SBAS stack inversion.

---

## Data

All data comes from the **public Capella Space AWS S3 bucket** — no credentials required.

| Resource | Value |
|----------|-------|
| STAC collection | `https://capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-ieee-data-contest/collection.json` |
| S3 bucket | `s3://capella-open-data/data/` (region: `us-west-2`) |
| Dataset | 791 X-band spotlight SLCs across 39 AOIs |

```bash
# Verify access (no authentication required):
aws s3 ls --no-sign-request s3://capella-open-data/data/ | head -5
```

---

## Environment

```bash
conda create -n insar-dem python=3.10
conda activate insar-dem
pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
pip install boto3 pystac snaphu-py==0.4.1
```

Confirmed working: Python 3.10, PyTorch 2.4.0, CUDA 12.1, rasterio 1.3.x, numpy 1.26.x.

> **Note:** If rasterio fails with a `GLIBCXX` error, prepend `export LD_LIBRARY_PATH=<your_env>/lib:$LD_LIBRARY_PATH` before running any script.

---

## Quick Start

```bash
export REPO=/path/to/Learning-Assisted-InSAR-DEM-Enhancement
export ENV=/path/to/your/conda/env
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH

# 1. Build full metadata index (STAC crawl — no download)
PYTHONPATH=$REPO conda run --prefix $ENV python \
    scripts/download_subset.py --index_only \
    --out_manifest data/manifests/full_index.parquet

# 2. Build interferometric pair graph for an AOI
PYTHONPATH=$REPO conda run --prefix $ENV python \
    scripts/build_pairs_manifest.py \
    --manifest data/manifests/full_index.parquet \
    --aoi AOI_000 --no-require-same-platform \
    --out data/manifests/aoi000_pairs.parquet \
    --triplets data/manifests/aoi000_triplets.parquet

# 3. Preprocess pairs (coregistration → interferogram → coherence → Goldstein)
PYTHONPATH=$REPO conda run --prefix $ENV --no-capture-output python -u \
    scripts/preprocess_pairs_full_image.py \
    --pairs_manifest data/manifests/aoi000_pairs.parquet \
    --raw_dir data/raw/AOI_000 \
    --out-dir data/processed/aoi000_pairs \
    --batch-workers 2 --min-coherence-mean 0.3

# 4. Train FiLMUNet
PYTHONPATH=$REPO conda run --prefix $ENV --no-capture-output python -u \
    experiments/enhanced/train_film_unet.py \
    --data_config  configs/data/capella_aoi_selection.yaml \
    --model_config configs/model/film_unet.yaml \
    --train_config configs/train/contest.yaml \
    --triplets_manifest data/manifests/aoi000_triplets.parquet

# 5. Evaluate (all 5 contest metrics)
PYTHONPATH=$REPO conda run --prefix $ENV --no-capture-output python -u \
    eval/compute_metrics.py \
    --checkpoint experiments/enhanced/checkpoints/film_unet/<run>/<run>_final.pt \
    --pairs_dir   data/processed/aoi000_pairs \
    --triplets_manifest data/manifests/aoi000_triplets.parquet \
    --out_dir     experiments/enhanced/outputs/aoi000
```

For the complete step-by-step guide with all CLI flags, logging patterns, and multi-AOI examples, see [`How_2_run_full_SLC.md`](How_2_run_full_SLC.md).

---

## Repository Structure

```
scripts/
  download_subset.py              # STAC crawl → manifest; S3 parallel download
  build_pairs_manifest.py         # Pair-graph construction with temporal/geometric filters
  preprocess_pairs_full_image.py  # Coreg → interferogram → coherence → Goldstein (full image)
  unwrap_snaphu.py                # SNAPHU phase unwrapping (snaphu-py backend)
  sbas_dem.py                     # Weighted multi-baseline DEM inversion
  plot_confidence_map.py          # 4-panel: raw / Goldstein / FiLMUNet / uncertainty
  plot_coherence_confidence.py    # Coherence vs FiLMUNet confidence scatter plot
  download_copernicus_dem.py      # Download Copernicus GLO-30 reference DEM tiles

src/
  insar_processing/
    filters.py          # Goldstein + coherence-adaptive Goldstein
    sublook.py          # FFT sub-look splitting for Noise2Noise training
    pair_graph.py       # Q_ij scoring, B_perp from orbit state vectors
    geometry.py         # Perpendicular baseline geometry
  models/
    film_unet.py        # FiLM-conditioned U-Net (primary model)
  losses/
    physics_losses.py   # N2N, uncertainty NLL, closure, temporal, gradient losses
  evaluation/
    closure_metrics.py  # All 5 contest metrics

eval/
  compute_metrics.py    # Goldstein vs FiLMUNet comparison: M1–M5 + CSV + figures

experiments/enhanced/
  train_film_unet.py    # Self-supervised FiLMUNet training
  checkpoints/film_unet/
    raw2gold_closure_20260321_1852/   # Hawaii checkpoint (SHA-256 in REPRODUCIBILITY.md)
    aoi024_finetune_closure_20260406_1503/  # AOI024 + zero-shot AOI008 checkpoint

configs/
  data/capella_aoi024_full_image.yaml   # AOI024 data config
  model/film_unet.yaml                  # Architecture: features, embed_dim, metadata_dim=7
  train/contest.yaml                    # lr=1e-4, epochs=50, seed=42, loss weights

data/
  manifests/            # Parquet pair/triplet manifests (SHA-256 in REPRODUCIBILITY.md)
  raw/                  # Downloaded SLCs (not in git)
  processed/            # Preprocessed pair directories (not in git)
  reference/            # Copernicus GLO-30 reference DEMs (not in git)
```

---

## Reproducibility

All results are reproducible from the public Capella S3 endpoint.

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for:
- STAC collection URL and S3 bucket
- SHA-256 checksums for all manifests and model checkpoints
- Random seeds and exact environment versions
- Step-by-step commands to reproduce each AOI result

---

## Citation

If you use this work, please cite the IEEE GRSS 2026 Data Fusion Contest dataset and this repository.
