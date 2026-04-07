# Full SLC-to-DEM Pipeline: Step-by-Step Guide

End-to-end pipeline for Capella Space X-band SAR InSAR DEM enhancement using FiLMUNet.
Covers STAC download → pair formation → preprocessing → training → evaluation → visualization.

---

## Prerequisites

Install the environment and set these three shell variables **once** before running any step:

```bash
export REPO=/path/to/Learning-Assisted-InSAR-DEM-Enhancement
export ENV=/path/to/your/conda/env
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
```

**User's values:**
```bash
export REPO=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement
export ENV=/scratch/gdemil24/hrwsi_s3client/torch-gpu
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
```

All commands below use `$REPO` and `$ENV`. The log pattern `mkdir -p logs; LOG=...` is reused throughout — each run produces a timestamped log file in `logs/`.

**Common run prefix** (abbreviated as `<RUN>` in general commands):
```bash
LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH \
PYTHONPATH=$REPO \
conda run --prefix $ENV --no-capture-output python -u
```

---

## Step 1: Build Metadata Index

> **Purpose:** Crawl the Capella STAC catalog and build a local manifest (`full_index.parquet`) with metadata for all 791 SLC collects. No data is downloaded. Run this once to discover available AOIs and collects.

**General command:**
```bash
PYTHONPATH=$REPO conda run --prefix $ENV python \
    scripts/download_subset.py \
    --index_only \
    --out_manifest data/manifests/full_index.parquet
```

**Example:**
```bash
PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement \
conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu python \
    scripts/download_subset.py \
    --index_only \
    --out_manifest data/manifests/full_index.parquet
```

**Output:** `data/manifests/full_index.parquet` — one row per SLC collect with AOI, mode, orbit, bbox, center_freq_ghz.

---

## Step 2: Download SLC Data

> **Purpose:** Download SLC GeoTIFF + extended JSON metadata for a specific AOI from the public Capella S3 bucket. No authentication required.

**General command:**
```bash
mkdir -p logs; LOG="logs/<AOI_TAG>_download_$(date +%Y%m%d_%H%M).log"; \
PYTHONPATH=$REPO conda run --prefix $ENV --no-capture-output python -u \
    scripts/download_subset.py \
    --manifest data/manifests/full_index.parquet \
    --aoi_filter <AOI_NAME> \
    --max_collects <N> \
    --out_dir data/raw/ \
    --n_workers 4 2>&1 | tee "$LOG"
```

**Example — AOI_024 (Western Australia):**
```bash
mkdir -p logs; LOG="logs/AOI024_download_$(date +%Y%m%d_%H%M).log"; \
PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement \
conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output python -u \
    scripts/download_subset.py \
    --manifest data/manifests/full_index.parquet \
    --aoi_filter AOI_024 \
    --out_dir data/raw/ \
    --n_workers 4 2>&1 | tee "$LOG"
```

**Output:** `data/raw/<AOI_ID>/` — one subdirectory per collect with `.tif` + `_extended.json`.

**Notes:** Omit `--max_collects` to download all collects for the AOI. Each full-resolution Capella SLC is ~1–4 GB.

---

## Step 3: Build Interferometric Pair Graph

> **Purpose:** Filter the metadata index and form interferometric pairs for a given AOI based on temporal baseline, incidence angle, and look direction. Also generates the triplet list for closure-loss training and M1/M5 evaluation.

**General command:**
```bash
mkdir -p logs; TASK_NAME="build_pairs_manifest" DATE=$(date +%Y%m%d_%H%M%S) \
LOG="logs/${TASK_NAME}_${DATE}.log"; \
LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH \
PYTHONPATH=$REPO \
conda run --prefix $ENV python \
    scripts/build_pairs_manifest.py \
    --manifest data/manifests/full_index.parquet \
    --aoi <AOI_ID> \
    --no-require-same-platform \
    --out data/manifests/<AOI_TAG>_full_index_full_image.parquet \
    --triplets data/manifests/<AOI_TAG>_full_index_triplets_full_image.parquet \
    2>&1 | tee "$LOG"
```

**Example — AOI_000 (Hawaii):**
```bash
mkdir -p logs && TASK_NAME="build_pairs_manifest" DATE=$(date +%Y%m%d_%H%M%S) LOG="logs/${TASK_NAME}_${DATE}.log" LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu python scripts/build_pairs_manifest.py --manifest data/manifests/full_index.parquet --aoi AOI_000 --no-require-same-platform --out data/manifests/full_index_full_image.parquet --triplets data/manifests/full_index_triplets_full_image.parquet 2>&1 | tee "$LOG"
```

**Example — AOI_024 (Western Australia):**
```bash
mkdir -p logs && TASK_NAME="build_pairs_manifest" DATE=$(date +%Y%m%d_%H%M%S) LOG="logs/${TASK_NAME}_${DATE}.log" LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu python scripts/build_pairs_manifest.py --manifest data/manifests/full_index.parquet --aoi AOI_024 --no-require-same-platform --out data/manifests/AOI024_full_index_full_image.parquet --triplets data/manifests/AOI024_full_index_triplets_full_image.parquet 2>&1 | tee "$LOG"
```

**Example — AOI_008 (Los Angeles):**
```bash
mkdir -p logs && TASK_NAME="build_pairs_manifest" DATE=$(date +%Y%m%d_%H%M%S) LOG="logs/${TASK_NAME}_${DATE}.log" LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu python scripts/build_pairs_manifest.py --manifest data/manifests/full_index.parquet --aoi AOI_008 --no-require-same-platform --out data/manifests/AOI008_full_index_full_image.parquet --triplets data/manifests/AOI008_full_index_triplets_full_image.parquet 2>&1 | tee "$LOG"
```

**Output:**
- `data/manifests/<AOI_TAG>_full_index_full_image.parquet` — pair list with B_perp, temporal baseline, incidence
- `data/manifests/<AOI_TAG>_full_index_triplets_full_image.parquet` — triplet list (pair i→j, j→k, i→k)

---

## Step 4: Preprocess Pairs — Full Image

> **Purpose:** For each pair in the manifest: coregister the SLC pair, form the wrapped interferogram, estimate coherence, and apply Goldstein adaptive phase filtering. Produces full-resolution (no tiling) GeoTIFFs.

**General command:**
```bash
mkdir -p logs; LOG="logs/<AOI_TAG>_preprocess_full_image_$(date +%Y%m%d_%H%M).log"; \
LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH \
PYTHONPATH=$REPO \
conda run --prefix $ENV --no-capture-output python -u \
    scripts/preprocess_pairs_full_image.py \
    --pairs_manifest data/manifests/<AOI_TAG>_full_index_full_image.parquet \
    --raw_dir data/raw/<AOI_ID> \
    --out-dir data/processed/<AOI_TAG>_pairs_full_image \
    --batch-workers 2 \
    --max-pairs <N> \
    --min-coherence-mean 0.3 \
    --interp lanczos \
    --remap-tile 4096 \
    --tp-workers 1 \
    --coherence-backend scipy \
    --skip-pass2 \
    --max-slc-gb 4.0 2>&1 | tee "$LOG"
```

**Example — AOI_000 (Hawaii):**
```bash
mkdir -p logs; LOG="logs/AOI000_preprocess_full_image_$(date +%Y%m%d_%H%M).log"; LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output python -u scripts/preprocess_pairs_full_image.py --pairs_manifest data/manifests/AOI000_full_index_full_image.parquet --raw_dir data/raw/AOI_000 --out-dir data/processed/AOI000_pairs_full_image --batch-workers 2 --max-pairs 200 --min-coherence-mean 0.3 --interp lanczos --remap-tile 4096 --tp-workers 1 --coherence-backend scipy --skip-pass2 --max-slc-gb 4.0 2>&1 | tee "$LOG"
```

**Example — AOI_024 (Western Australia):**
```bash
mkdir -p logs; LOG="logs/AOI024_preprocess_full_image_$(date +%Y%m%d_%H%M).log"; LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output python -u scripts/preprocess_pairs_full_image.py --pairs_manifest data/manifests/AOI024_full_index_full_image.parquet --raw_dir data/raw/AOI_024 --out-dir data/processed/AOI024_pairs_full_image --batch-workers 2 --max-pairs 300 --min-coherence-mean 0.3 --interp lanczos --remap-tile 4096 --tp-workers 1 --coherence-backend scipy --skip-pass2 --max-slc-gb 4.0 2>&1 | tee "$LOG"
```

**Example — AOI_008 (Los Angeles):**
```bash
mkdir -p logs; LOG="logs/AOI008_preprocess_full_image_$(date +%Y%m%d_%H%M).log"; LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output python -u scripts/preprocess_pairs_full_image.py --pairs_manifest data/manifests/AOI008_full_index_full_image.parquet --raw_dir data/raw/AOI_008 --out-dir data/processed/AOI008_pairs_full_image --batch-workers 2 --max-pairs 300 --min-coherence-mean 0.3 --interp lanczos --remap-tile 4096 --tp-workers 1 --coherence-backend scipy --skip-pass2 --max-slc-gb 4.0 2>&1 | tee "$LOG"
```

**Output per pair dir** `data/processed/<AOI_TAG>_pairs_full_image/<ref_date>__<sec_date>/`:
- `ifg_raw_complex_real_imag.tif` — raw wrapped interferogram (2-band: real, imag)
- `ifg_goldstein_complex_real_imag.tif` — Goldstein-filtered interferogram
- `coherence.tif` — interferometric coherence [0, 1]
- `coreg_meta.json` — coregistration metadata (B_perp, incidence, SNR proxy, bbox)

**Notes:** `--max-slc-gb 4.0` skips SLC pairs where either scene exceeds 4 GB (prevents OOM on large images). Adjust `--batch-workers` and `--max-pairs` to control parallelism and subset size.

---

## Step 5: Goldstein Phase Unwrapping (Baseline)

> **Purpose:** Unwrap the Goldstein-filtered interferograms using SNAPHU. This produces the baseline unwrapped phase used for DEM NMAD and as a training reference.

**General command:**
```bash
mkdir -p logs; LOG="logs/<AOI_TAG>_unwrap_goldstein_$(date +%Y%m%d_%H%M).log"; \
conda run --prefix $ENV --no-capture-output \
    env LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH \
        PYTHONPATH=$REPO \
    python -u scripts/unwrap_snaphu.py \
    --pairs_dir data/processed/<AOI_TAG>_pairs_full_image \
    --input_ifg ifg_goldstein_complex_real_imag.tif \
    2>&1 | tee "$LOG"
```

**Example — AOI_024 (Western Australia):**
```bash
mkdir -p logs; LOG="logs/AOI024_unwrap_full_image_$(date +%Y%m%d_%H%M).log"; conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python -u scripts/unwrap_snaphu.py --pairs_dir data/processed/AOI024_pairs_full_image --input_ifg ifg_goldstein_complex_real_imag.tif 2>&1 | tee "$LOG"
```

**Output:** `unw_phase.tif` written into each pair directory alongside the input interferogram.

**Notes:** Add `--workers <N>` to parallelize across pairs (default: 1). For full-image Capella pairs this is memory-intensive — use `--workers 1` or `--workers 2` on a node with ≥64 GB RAM.

---

## Step 6: Train FiLMUNet

> **Purpose:** Self-supervised training of the FiLM-conditioned U-Net using Noise2Noise (sub-look splitting), closure consistency, and temporal losses. No reference clean interferograms required.

**General command:**
```bash
mkdir -p logs; LOG="logs/<AOI_TAG>_train_film_unet_$(date +%Y%m%d_%H%M).log"; \
LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH \
PYTHONPATH=$REPO \
conda run --prefix $ENV --no-capture-output python -u \
    experiments/enhanced/train_film_unet.py \
    --data_config configs/data/<your_data_config>.yaml \
    --model_config configs/model/film_unet.yaml \
    --train_config configs/train/contest.yaml \
    --triplets_manifest data/manifests/<AOI_TAG>_full_index_triplets_full_image.parquet \
    --run_name <run_name> 2>&1 | tee "$LOG"
```

**Example — AOI_024 (Western Australia):**
```bash
mkdir -p logs; LOG="logs/AOI024_train_film_unet_$(date +%Y%m%d_%H%M).log"; LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output python -u experiments/enhanced/train_film_unet.py --data_config configs/data/capella_aoi024_full_image.yaml --model_config configs/model/film_unet.yaml --train_config configs/train/contest.yaml --triplets_manifest data/manifests/AOI024_full_index_triplets_full_image.parquet --run_name aoi024_full_image 2>&1 | tee "$LOG"
```

**Output:** `experiments/enhanced/checkpoints/film_unet/<run_name>_<date>/<run_name>_<date>_final.pt`

**Notes:** The data config must point to your processed pairs dir (`pairs_dir`) and set `metadata_dim: 7` for FiLM conditioning on `[Δt, θ_inc, θ_graze, B_perp, mode, look, SNR]`.

---

## Step 7: Fine-tune FiLMUNet with Closure Loss (Optional)

> **Purpose:** Continue training from a checkpoint with increased closure-consistency loss weight. Use this if the initial training did not include `--triplets_manifest`, or to boost M1/M5 metric performance.

**General command:**
```bash
mkdir -p logs; LOG="logs/<AOI_TAG>_finetune_closure_$(date +%Y%m%d_%H%M).log"; \
LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH \
PYTHONPATH=$REPO \
conda run --prefix $ENV --no-capture-output python -u \
    experiments/enhanced/train_film_unet.py \
    --data_config configs/data/<your_data_config>.yaml \
    --model_config configs/model/film_unet.yaml \
    --train_config configs/train/contest.yaml \
    --resume <path/to/checkpoint_final.pt> \
    --triplets_manifest data/manifests/<AOI_TAG>_full_index_triplets_full_image.parquet \
    --run_name <run_name>_finetune_closure \
    --loss_closure 0.8 \
    --epochs 10 2>&1 | tee "$LOG"
```

**Example — AOI_024 (fine-tune after training without triplets):**
```bash
mkdir -p logs; LOG="logs/AOI024_finetune_closure_$(date +%Y%m%d_%H%M).log"; LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output python -u experiments/enhanced/train_film_unet.py --data_config configs/data/capella_aoi024_full_image.yaml --model_config configs/model/film_unet.yaml --train_config configs/train/contest.yaml --run_name aoi024_finetune_closure --resume experiments/enhanced/checkpoints/film_unet/aoi024_full_image_20260405_1433/aoi024_full_image_20260405_1433_final.pt --triplets_manifest data/manifests/AOI024_full_index_triplets_full_image.parquet --loss_closure 0.8 --epochs 10 2>&1 | tee "$LOG"
```

**Output:** New checkpoint at `experiments/enhanced/checkpoints/film_unet/<run_name>_finetune_closure_<date>/<...>_final.pt`

---

## Step 8: FiLMUNet Phase Unwrapping

> **Purpose:** Run FiLMUNet inference on all pairs (produces `ifg_film_unet.tif` + `log_var.tif`), then unwrap the FiLMUNet-denoised interferogram with SNAPHU. Uncertainty (`log_var.tif`) is used as SNAPHU weights and for M4 NMAD weighting.

**Inference + unwrapping are handled together by `unwrap_snaphu.py` when `--input_ifg ifg_film_unet.tif` is set — but inference must be run first via `compute_metrics.py --skip_snaphu_metrics`, or the pair directory must already contain `ifg_film_unet.tif`.**

**Unwrapping general command (after inference):**
```bash
mkdir -p logs; LOG="logs/<AOI_TAG>_unet_unwrap_snaphu_$(date +%Y%m%d_%H%M).log"; \
LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH \
PYTHONPATH=$REPO \
conda run --prefix $ENV --no-capture-output python -u \
    scripts/unwrap_snaphu.py \
    --pairs_dir data/processed/<AOI_TAG>_pairs_full_image/<SELECTED_SUBDIR> \
    --input_ifg ifg_film_unet.tif \
    --output_name unw_phase_film_unet.tif \
    --workers <N> 2>&1 | tee "$LOG"
```

**Example — AOI_024 selected pairs:**
```bash
mkdir -p logs; LOG="logs/AOI024_unet_unwrap_snaphu_$(date +%Y%m%d_%H%M).log"; LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output python -u scripts/unwrap_snaphu.py --pairs_dir data/processed/AOI024_pairs_full_image/AOI024_selected --input_ifg ifg_film_unet.tif --output_name unw_phase_film_unet.tif --workers 4 2>&1 | tee "$LOG"
```

**Output per pair dir:**
- `ifg_film_unet.tif` — FiLMUNet denoised interferogram (2-band complex)
- `log_var.tif` — per-pixel log-variance (uncertainty)
- `unw_phase_film_unet.tif` — unwrapped FiLMUNet phase

---

## Step 9: Download Copernicus Reference DEM

> **Purpose:** Download and mosaic Copernicus GLO-30 tiles (~30 m) for a bounding box. Used as ground-truth elevation for DEM NMAD (M4) computation. No authentication required (public S3 bucket).

**General command:**
```bash
conda run --prefix $ENV python -u \
    scripts/download_copernicus_dem.py \
    --bbox_w <WEST> --bbox_s <SOUTH> --bbox_e <EAST> --bbox_n <NORTH> \
    --out_dir data/reference/copernicus_dem \
    --merged_name <aoi_tag>_dem.tif
```

**Example — AOI_024 (Western Australia):**
```bash
conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu python -u scripts/download_copernicus_dem.py --bbox_w 118.70 --bbox_s -23.25 --bbox_e 118.85 --bbox_n -23.10 --out_dir data/reference/copernicus_dem --merged_name aoi024_dem.tif
```

**Example — AOI_008 (Los Angeles):**
```bash
conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu python -u scripts/download_copernicus_dem.py --bbox_w -118.15 --bbox_s 34.75 --bbox_e -118.00 --bbox_n 34.90 --out_dir data/reference/copernicus_dem --merged_name aoi008_dem.tif
```

**Output:** `data/reference/copernicus_dem/<aoi_tag>_dem.tif` — merged GeoTIFF, float32, ~30 m resolution.

**Notes:** Bounding box should match or slightly exceed the footprint of your processed pairs. The script skips ocean tiles gracefully. Re-runs are safe (skips already-downloaded tiles).

---

## Step 10: Compute All Contest Metrics — Main AOI

> **Purpose:** Run FiLMUNet inference on all pairs and compute all 5 metrics: M1 triplet closure, M2 unwrap success rate, M3 usable pairs, M4 DEM NMAD, M5 temporal residual. Produces `metrics_comparison.csv` and summary figures.

**General command:**
```bash
mkdir -p logs; LOG="logs/<AOI_TAG>_eval_final_$(date +%Y%m%d_%H%M).log"; \
LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH \
PYTHONPATH=$REPO \
conda run --prefix $ENV --no-capture-output python -u \
    eval/compute_metrics.py \
    --checkpoint <path/to/checkpoint_final.pt> \
    --pairs_dir data/processed/<AOI_TAG>_pairs_full_image \
    --triplets_manifest data/manifests/<AOI_TAG>_full_index_triplets_full_image.parquet \
    --out_dir experiments/enhanced/outputs/<AOI_TAG>_full_image \
    --stride 256 \
    --max_pairs <N> \
    --batch_size 64 \
    --force_inference \
    --skip_snaphu_metrics 2>&1 | tee "$LOG"
```

> **Tip:** Use `--skip_snaphu_metrics` to run inference + M1/M5 only (fast). Once `unw_phase.tif` and `unw_phase_film_unet.tif` exist, rerun with `--snaphu_only` to compute M2/M3/M4 without re-running inference.

**Example — AOI_024 (inference + M1/M5):**
```bash
mkdir -p logs; LOG="logs/AOI024_eval_final_$(date +%Y%m%d_%H%M).log"; LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output python -u eval/compute_metrics.py --checkpoint experiments/enhanced/checkpoints/film_unet/aoi024_finetune_closure_20260406_1503/aoi024_finetune_closure_20260406_1503_final.pt --pairs_dir data/processed/AOI024_pairs_full_image --triplets_manifest data/manifests/AOI024_full_index_triplets_full_image.parquet --out_dir experiments/enhanced/outputs/aoi024_full_image --stride 256 --max_pairs 3 --batch_size 64 --force_inference --skip_snaphu_metrics 2>&1 | tee "$LOG"
```

**Output:**
- `experiments/enhanced/outputs/<AOI_TAG>_full_image/metrics_comparison.csv`
- `experiments/enhanced/outputs/<AOI_TAG>_full_image/figures/` — M1 closure histogram, M4 NMAD scatter, M5 residual bar chart

---

## Step 11: Zero-Shot Evaluation — New AOI

> **Purpose:** Evaluate the trained model on a held-out AOI (no fine-tuning on that AOI). Demonstrates cross-geography generalization. Uses the same `compute_metrics.py` but points to a different `--pairs_dir` without `--force_inference` (if inference already ran) or with it (fresh run).

**General command:**
```bash
mkdir -p logs; LOG="logs/<AOI_TAG>_0Shot_eval_$(date +%Y%m%d_%H%M).log"; \
LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH \
PYTHONPATH=$REPO \
conda run --prefix $ENV --no-capture-output python -u \
    eval/compute_metrics.py \
    --checkpoint <path/to/source_aoi_checkpoint_final.pt> \
    --pairs_dir data/processed/<TARGET_AOI_TAG>_pairs_full_image \
    --triplets_manifest data/manifests/<TARGET_AOI_TAG>_full_index_triplets_full_image.parquet \
    --out_dir experiments/enhanced/outputs/<TARGET_AOI_TAG>_0Shot_full_image \
    --stride 256 \
    --max_pairs <N> \
    --batch_size 64 \
    --skip_snaphu_metrics 2>&1 | tee "$LOG"
```

**Example — AOI_008 (LA), using AOI_024-trained checkpoint:**
```bash
mkdir -p logs; LOG="logs/AOI008_0Shot_eval_final_$(date +%Y%m%d_%H%M).log"; LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output python -u eval/compute_metrics.py --checkpoint experiments/enhanced/checkpoints/film_unet/aoi024_finetune_closure_20260406_1503/aoi024_finetune_closure_20260406_1503_final.pt --pairs_dir data/processed/AOI008_pairs_full_image --triplets_manifest data/manifests/AOI008_full_index_triplets_full_image.parquet --out_dir experiments/enhanced/outputs/aoi008_0Shot_full_image --stride 256 --max_pairs 3 --batch_size 64 --skip_snaphu_metrics 2>&1 | tee "$LOG"
```

**Output:** Same structure as Step 10 but in `<TARGET_AOI_TAG>_0Shot_full_image/`.

---

## Step 12: Confidence Map Visualization

> **Purpose:** Generate 4-panel figures (raw phase / Goldstein / FiLMUNet denoised / FiLMUNet confidence map) for each processed pair. Confidence = exp(−σ̂) where σ̂ = √exp(log_var). Requires `ifg_film_unet.tif` and `log_var.tif` to exist (run inference first).

**General command:**
```bash
LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH \
conda run --prefix $ENV python \
    scripts/plot_confidence_map.py \
    --pairs_dir data/processed/<AOI_TAG>_pairs_full_image/<SUBDIR> \
    --out_dir experiments/enhanced/outputs/figures/confidence_maps \
    --tile_size 0
```

**Example — AOI_024 selected pairs (full image):**
```bash
LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu python scripts/plot_confidence_map.py --pairs_dir data/processed/AOI024_pairs_full_image/AOI024_selected --out_dir experiments/enhanced/outputs/figures/confidence_maps --tile_size 0
```

**Output:** `experiments/enhanced/outputs/figures/confidence_maps/confidence_map_<pair_name>.png` — one figure per pair.

**Notes:** `--tile_size 0` uses the full image. Use `--tile_size 1024 --tile_row 0 --tile_col 0` to crop to a 1024×1024 window for quicker inspection.

---

## Step 13: Coherence vs Confidence Scatter Plot

> **Purpose:** Plot interferometric coherence (x-axis) vs FiLMUNet confidence (y-axis) to verify that uncertainty is physics-grounded: low coherence → low confidence. Produces hexbin density + per-pair scatter.

**General command:**
```bash
mkdir -p logs; LOG="logs/<AOI_TAG>_plot_coherence_confidence_$(date +%Y%m%d_%H%M).log"; \
LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH \
conda run --prefix $ENV python \
    scripts/plot_coherence_confidence.py \
    --pairs_dir data/processed/<AOI_TAG>_pairs_full_image/<SUBDIR> \
    --out_dir experiments/enhanced/outputs/figures \
    --sample_rate 500 2>&1 | tee "$LOG"
```

**Example — AOI_024 selected pairs:**
```bash
mkdir -p logs; LOG="logs/AOI024_plot_coherence_confidence_$(date +%Y%m%d_%H%M).log"; LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu python scripts/plot_coherence_confidence.py --pairs_dir data/processed/AOI024_pairs_full_image/AOI024_selected --out_dir experiments/enhanced/outputs/figures --sample_rate 500 2>&1 | tee "$LOG"
```

**Output:** `experiments/enhanced/outputs/figures/coherence_vs_confidence.png`

**Notes:** `--sample_rate 500` keeps 1 in 500 pixels per pair (fast). Lower values (e.g. 100) give denser scatter at the cost of runtime.

---

## Step 14: SBAS Multi-Baseline DEM Generation

> **Purpose:** Weighted multi-baseline height inversion: jointly combines all available unwrapped pairs into a single best-estimate DEM per pixel. FiLMUNet confidence weights (`exp(−σ̂)`) reduce the NMAD relative to coherence-weighted Goldstein inversion. Outputs DEM GeoTIFFs + 3-panel comparison figure.

**General command:**
```bash
mkdir -p logs; LOG="logs/<AOI_TAG>_sbas_dem_$(date +%Y%m%d_%H%M).log"; \
LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH \
PYTHONPATH=$REPO \
conda run --prefix $ENV python -u \
    scripts/sbas_dem.py \
    --pairs_dir data/processed/<AOI_TAG>_pairs_full_image/<SUBDIR> \
    --out_dir experiments/enhanced/outputs/sbas_dem_<aoi_tag> \
    --copernicus_dem_dir data/reference/copernicus_dem \
    --aoi <AOI_TAG> \
    [--max_pairs <N>] 2>&1 | tee "$LOG"
```

**Example — AOI_024, 1 pair (fast demo):**
```bash
mkdir -p logs; LOG="logs/AOI024_sbas_dem_$(date +%Y%m%d_%H%M).log"; LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu python -u scripts/sbas_dem.py --pairs_dir data/processed/AOI024_pairs_full_image/AOI024_selected --out_dir experiments/enhanced/outputs/sbas_dem_aoi024 --copernicus_dem_dir data/reference/copernicus_dem --aoi AOI024 --max_pairs 1 2>&1 | tee "$LOG"
```

**Output:**
- `experiments/enhanced/outputs/sbas_dem_<aoi_tag>/dem_goldstein.tif` — coherence-weighted DEM
- `experiments/enhanced/outputs/sbas_dem_<aoi_tag>/dem_filmunet.tif` — confidence-weighted DEM
- `experiments/enhanced/outputs/sbas_dem_<aoi_tag>/sbas_dem_comparison.png` — 3-panel: Goldstein | FiLMUNet | Difference

**Notes:** `--max_pairs` automatically prioritises pairs that have both `unw_phase.tif` and `unw_phase_film_unet.tif`, so the comparison is always fair. Omit `--max_pairs` to use all available pairs for the best-quality DEM.

---

## Quick Reference

| Step | Script | Key Input | Key Output |
|------|--------|-----------|------------|
| 1 | `download_subset.py --index_only` | STAC URL | `full_index.parquet` |
| 2 | `download_subset.py` | `full_index.parquet` | `data/raw/<AOI_ID>/` |
| 3 | `build_pairs_manifest.py` | `full_index.parquet` | `*_full_image.parquet` + `*_triplets.parquet` |
| 4 | `preprocess_pairs_full_image.py` | pairs parquet + raw SLCs | `ifg_*.tif`, `coherence.tif`, `coreg_meta.json` |
| 5 | `unwrap_snaphu.py` (goldstein) | `ifg_goldstein_complex_real_imag.tif` | `unw_phase.tif` |
| 6 | `train_film_unet.py` | processed pairs + triplets | `*_final.pt` checkpoint |
| 7 | `train_film_unet.py --resume` | previous checkpoint | fine-tuned `*_final.pt` |
| 8 | `unwrap_snaphu.py` (film_unet) | `ifg_film_unet.tif` | `unw_phase_film_unet.tif` |
| 9 | `download_copernicus_dem.py` | bbox | `<aoi>_dem.tif` |
| 10 | `compute_metrics.py` | checkpoint + pairs | `metrics_comparison.csv` + figures |
| 11 | `compute_metrics.py` (zero-shot) | foreign AOI pairs | `metrics_comparison.csv` (cross-AOI) |
| 12 | `plot_confidence_map.py` | `ifg_film_unet.tif` + `log_var.tif` | `confidence_map_*.png` |
| 13 | `plot_coherence_confidence.py` | `coherence.tif` + `log_var.tif` | `coherence_vs_confidence.png` |
| 14 | `sbas_dem.py` | `unw_phase*.tif` stack | `dem_*.tif` + `sbas_dem_comparison.png` |
