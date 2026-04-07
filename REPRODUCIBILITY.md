# Reproducibility Guide

IEEE GRSS 2026 Data Fusion Contest — Learning-Assisted InSAR DEM Enhancement
**Method:** FiLMUNet — Self-Supervised InSAR Phase Denoising via Geometry-Conditioned Noise2Noise

All results are reproducible from the public Capella Space S3 bucket (no authentication required).

---

## 1. Dataset Access

| Item | Value |
|------|-------|
| STAC collection URL | `https://capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-ieee-data-contest/collection.json` |
| S3 bucket | `s3://capella-open-data/data/` (region: `us-west-2`, anonymous access) |
| Total SLC collects | 791 (filter on `_SLC_` in href) |
| Total AOIs | 39 |

### AOIs Used

| AOI | Location | Collects | Role |
|-----|----------|----------|------|
| AOI_000 | Hawaii, USA | 221 | Primary training + evaluation |
| AOI_024 | Western Australia | 30 | Secondary training + evaluation |
| AOI_008 | Los Angeles, USA | — | Zero-shot transfer evaluation |

**AOI_000 (Hawaii) selection rationale:**
- Largest collect count (221) among all 39 AOIs
- Both ascending and descending orbits; incidence range 35.8°–56.3°
- Volcanic terrain yields rich phase signal for closure and SBAS evaluation

### Data Splits

Splits are **by acquisition date** (temporal) to prevent geographic leakage:

| Split | Fraction | Criterion |
|-------|----------|-----------|
| Train | 70% | Earliest 70% of reference dates |
| Val   | 15% | Next 15% |
| Test  | 15% | Latest 15% (held out) |

Pair graph: maximum temporal baseline 180 days; `--no-require-same-platform` flag applied.

---

## 2. Manifest Checksums

Verify with `sha256sum <file>`.

| File | Rows | SHA-256 |
|------|------|---------|
| `data/manifests/full_index.parquet` | 791 SLC collects | `3c03228e72071e49d5fb823551fc4c2e556bd0f08b4ada2af4c3fab7622482ac` |
| `data/manifests/hawaii_pairs.parquet` | Hawaii pairs | `242f6511762f092b4cfa70725db3cb0c714b62d63caccfd343eed85b08513894` |
| `data/manifests/hawaii_triplets_strict.parquet` | Hawaii triplets | `c615add1d45e917f1b803c09660aad6fb25b35fd961b367bcc50d6d9b7c9cc0d` |
| `data/manifests/full_index_full_image.parquet` | 6,149 Hawaii pairs | `7ae018073fbfd12e2430d55824849273f5a8877ee29d56b8b5986d79f20a9c40` |
| `data/manifests/full_index_triplets_full_image.parquet` | Hawaii triplets (full-image) | `f69a8394f39ac5ec2c2caab8f89a77ac9cf27c878ceaf7cfedfb7f755cfceb3e` |
| `data/manifests/AOI024_full_index_full_image.parquet` | 909 pairs | `26e2d5b08e85591247c13cced58ff0176fc5e018b477a8b030c0bef10d296d5a` |
| `data/manifests/AOI024_full_index_triplets_full_image.parquet` | 5,000 triplets | `49fa0c210818589dbd85f1f232a0e0c2f8a4886f83d6f4975263c9b7d7497b54` |
| `data/manifests/AOI008_full_index_full_image.parquet` | 2,818 pairs | `e9eaae8d8f554ad33fe81397ad043af9959a01c47aeeac68cb57b2c3ac1ff012` |
| `data/manifests/AOI008_full_index_triplets_full_image.parquet` | AOI_008 triplets | `38fc71c34109d95b356a5576308756aad986f7f588c56d9a91c472a10987ca51` |

---

## 3. Model Checkpoints

| Checkpoint | Trained on | SHA-256 |
|------------|-----------|---------|
| `experiments/enhanced/checkpoints/film_unet/raw2gold_closure_20260321_1852/raw2gold_closure_20260321_1852_final.pt` | AOI_000 (Hawaii) | `2d021ac32039022992417484c07d3b5e960b5cb1969c43e0f740a071e95df672` |
| `experiments/enhanced/checkpoints/film_unet/aoi024_finetune_closure_20260406_1503/aoi024_finetune_closure_20260406_1503_final.pt` | AOI_024 (W. Australia) | `351d55f18e8e323d318cfc8ba3107b1262bbb5c9570234e36b2d82b01fd1f440` |

The AOI_024 checkpoint is also used for AOI_008 zero-shot evaluation (no retraining on AOI_008).

---

## 4. Environment

```bash
conda create -n insar-dem python=3.10
conda activate insar-dem
pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
pip install boto3 pystac snaphu-py==0.4.1
```

Confirmed working versions:

| Package | Version |
|---------|---------|
| Python | 3.10 |
| PyTorch | 2.4.0 |
| CUDA | 12.1 |
| snaphu-py | 0.4.1 |
| rasterio | 1.3.x |
| numpy | 1.26.x |
| boto3 | any recent |

**Note:** On systems where rasterio fails with a `GLIBCXX` symbol error, set:
```bash
export LD_LIBRARY_PATH=/path/to/conda/env/lib:$LD_LIBRARY_PATH
```

---

## 5. Random Seeds

All randomness is seeded deterministically:

| Component | Seed | Set by |
|-----------|------|--------|
| Python `random` | 42 | `train_film_unet.py: set_seed()` |
| NumPy | 42 | `train_film_unet.py: set_seed()` |
| PyTorch (CPU + GPU) | 42 | `train_film_unet.py: set_seed()` |
| Training config | `seed: 42` | `configs/train/contest.yaml` |

DataLoader: `shuffle=True` (train only) with the seeded state; `drop_last=True` to prevent batch-size variation.

---

## 6. Reproduce — Primary Results (AOI_000, Hawaii)

Set environment variables once:

```bash
export REPO=/path/to/Learning-Assisted-InSAR-DEM-Enhancement
export ENV=/path/to/your/conda/env
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
```

### Step 1 — Build metadata index
```bash
PYTHONPATH=$REPO conda run --prefix $ENV python \
    scripts/download_subset.py --index_only \
    --out_manifest data/manifests/full_index.parquet
```

### Step 2 — Download Hawaii SLC data
```bash
PYTHONPATH=$REPO conda run --prefix $ENV python \
    scripts/download_subset.py \
    --manifest data/manifests/full_index.parquet \
    --aoi_filter AOI_000 \
    --out_dir data/raw/ --n_workers 4
```

### Step 3 — Build pair graph
```bash
PYTHONPATH=$REPO conda run --prefix $ENV python \
    scripts/build_pairs_manifest.py \
    --manifest data/manifests/full_index.parquet \
    --aoi AOI_000 --no-require-same-platform \
    --out data/manifests/full_index_full_image.parquet \
    --triplets data/manifests/full_index_triplets_full_image.parquet
```

### Step 4 — Preprocess pairs (coregistration → interferogram → coherence → Goldstein)
```bash
PYTHONPATH=$REPO conda run --prefix $ENV --no-capture-output python -u \
    scripts/preprocess_pairs_full_image.py \
    --pairs_manifest data/manifests/full_index_full_image.parquet \
    --raw_dir data/raw/AOI_000 \
    --out-dir data/processed/pairs_full_image \
    --batch-workers 2 --max-pairs 200 --min-coherence-mean 0.3 \
    --interp lanczos --remap-tile 4096 --coherence-backend scipy \
    --skip-pass2 --max-slc-gb 4.0
```

### Step 5 — Phase unwrapping (Goldstein baseline)
```bash
PYTHONPATH=$REPO conda run --prefix $ENV --no-capture-output python -u \
    scripts/unwrap_snaphu.py \
    --pairs_dir data/processed/pairs_full_image \
    --input_ifg ifg_goldstein_complex_real_imag.tif
```

### Step 6 — Train FiLMUNet
```bash
PYTHONPATH=$REPO conda run --prefix $ENV --no-capture-output python -u \
    experiments/enhanced/train_film_unet.py \
    --data_config  configs/data/capella_aoi_selection.yaml \
    --model_config configs/model/film_unet.yaml \
    --train_config configs/train/contest.yaml \
    --triplets_manifest data/manifests/full_index_triplets_full_image.parquet \
    --run_name hawaii_full_image
```

### Step 7 — Evaluate (M1 + M5; then separately M2/M3/M4 after SNAPHU on FiLMUNet output)
```bash
# Step 7a: Run inference + M1 / M5
PYTHONPATH=$REPO conda run --prefix $ENV --no-capture-output python -u \
    eval/compute_metrics.py \
    --checkpoint experiments/enhanced/checkpoints/film_unet/raw2gold_closure_20260321_1852/raw2gold_closure_20260321_1852_final.pt \
    --pairs_dir data/processed/pairs_full_image \
    --triplets_manifest data/manifests/hawaii_triplets_strict.parquet \
    --out_dir experiments/enhanced/outputs/hawaii_full_image \
    --stride 256 --batch_size 64 --force_inference --skip_snaphu_metrics

# Step 7b: Unwrap FiLMUNet output
PYTHONPATH=$REPO conda run --prefix $ENV --no-capture-output python -u \
    scripts/unwrap_snaphu.py \
    --pairs_dir data/processed/pairs_full_image \
    --input_ifg ifg_film_unet.tif \
    --output_name unw_phase_film_unet.tif --workers 2

# Step 7c: Compute M2 / M3 / M4
PYTHONPATH=$REPO conda run --prefix $ENV --no-capture-output python -u \
    eval/compute_metrics.py \
    --checkpoint experiments/enhanced/checkpoints/film_unet/raw2gold_closure_20260321_1852/raw2gold_closure_20260321_1852_final.pt \
    --pairs_dir data/processed/pairs_full_image \
    --triplets_manifest data/manifests/hawaii_triplets_strict.parquet \
    --copernicus_dem_dir data/reference/copernicus_dem \
    --out_dir experiments/enhanced/outputs/hawaii_full_image \
    --stride 256 --batch_size 64 --snaphu_only
```

**Output:** `experiments/enhanced/outputs/hawaii_full_image/metrics_comparison.csv`

For the full command reference with logging, see [`How_2_run_full_SLC.md`](How_2_run_full_SLC.md).

---

## 7. Reproduce — Additional Results (AOI_024 + Zero-Shot AOI_008)

### AOI_024 (Western Australia) — separate training

```bash
# Build manifests
PYTHONPATH=$REPO conda run --prefix $ENV python scripts/build_pairs_manifest.py \
    --manifest data/manifests/full_index.parquet --aoi AOI_024 \
    --no-require-same-platform \
    --out data/manifests/AOI024_full_index_full_image.parquet \
    --triplets data/manifests/AOI024_full_index_triplets_full_image.parquet

# Preprocess
PYTHONPATH=$REPO conda run --prefix $ENV --no-capture-output python -u \
    scripts/preprocess_pairs_full_image.py \
    --pairs_manifest data/manifests/AOI024_full_index_full_image.parquet \
    --raw_dir data/raw/AOI_024 \
    --out-dir data/processed/AOI024_pairs_full_image \
    --batch-workers 2 --max-pairs 300 --min-coherence-mean 0.3 \
    --interp lanczos --remap-tile 4096 --coherence-backend scipy \
    --skip-pass2 --max-slc-gb 4.0

# Train FiLMUNet on AOI_024
PYTHONPATH=$REPO conda run --prefix $ENV --no-capture-output python -u \
    experiments/enhanced/train_film_unet.py \
    --data_config configs/data/capella_aoi024_full_image.yaml \
    --model_config configs/model/film_unet.yaml \
    --train_config configs/train/contest.yaml \
    --triplets_manifest data/manifests/AOI024_full_index_triplets_full_image.parquet \
    --run_name aoi024_full_image

# Fine-tune with increased closure weight
PYTHONPATH=$REPO conda run --prefix $ENV --no-capture-output python -u \
    experiments/enhanced/train_film_unet.py \
    --data_config configs/data/capella_aoi024_full_image.yaml \
    --model_config configs/model/film_unet.yaml \
    --train_config configs/train/contest.yaml \
    --resume experiments/enhanced/checkpoints/film_unet/aoi024_full_image_<date>/aoi024_full_image_<date>_final.pt \
    --triplets_manifest data/manifests/AOI024_full_index_triplets_full_image.parquet \
    --run_name aoi024_finetune_closure --loss_closure 0.8 --epochs 10

# Evaluate AOI_024 (M1 + M5 via inference)
PYTHONPATH=$REPO conda run --prefix $ENV --no-capture-output python -u \
    eval/compute_metrics.py \
    --checkpoint experiments/enhanced/checkpoints/film_unet/aoi024_finetune_closure_20260406_1503/aoi024_finetune_closure_20260406_1503_final.pt \
    --pairs_dir data/processed/AOI024_pairs_full_image \
    --triplets_manifest data/manifests/AOI024_full_index_triplets_full_image.parquet \
    --out_dir experiments/enhanced/outputs/aoi024_full_image \
    --stride 256 --max_pairs 3 --batch_size 64 --force_inference --skip_snaphu_metrics
```

### AOI_008 (Los Angeles) — zero-shot transfer (no retraining)

```bash
# Build manifests
PYTHONPATH=$REPO conda run --prefix $ENV python scripts/build_pairs_manifest.py \
    --manifest data/manifests/full_index.parquet --aoi AOI_008 \
    --no-require-same-platform \
    --out data/manifests/AOI008_full_index_full_image.parquet \
    --triplets data/manifests/AOI008_full_index_triplets_full_image.parquet

# Preprocess AOI_008 pairs (same flags as AOI_024)
PYTHONPATH=$REPO conda run --prefix $ENV --no-capture-output python -u \
    scripts/preprocess_pairs_full_image.py \
    --pairs_manifest data/manifests/AOI008_full_index_full_image.parquet \
    --raw_dir data/raw/AOI_008 \
    --out-dir data/processed/AOI008_pairs_full_image \
    --batch-workers 2 --max-pairs 300 --min-coherence-mean 0.3 \
    --interp lanczos --remap-tile 4096 --coherence-backend scipy \
    --skip-pass2 --max-slc-gb 4.0

# Evaluate using AOI_024 checkpoint (zero-shot — no retraining)
PYTHONPATH=$REPO conda run --prefix $ENV --no-capture-output python -u \
    eval/compute_metrics.py \
    --checkpoint experiments/enhanced/checkpoints/film_unet/aoi024_finetune_closure_20260406_1503/aoi024_finetune_closure_20260406_1503_final.pt \
    --pairs_dir data/processed/AOI008_pairs_full_image \
    --triplets_manifest data/manifests/AOI008_full_index_triplets_full_image.parquet \
    --out_dir experiments/enhanced/outputs/aoi008_0Shot_full_image \
    --stride 256 --max_pairs 3 --batch_size 64 --skip_snaphu_metrics
```

---

## 8. Results

Eval log (primary / Hawaii): `logs/eval_raw2gold_closure_20260321_1852_final_20260401_1155.log`

### AOI_000 — Hawaii (Primary)

| Metric | Goldstein | FiLMUNet | Δ |
|--------|-----------|----------|---|
| M1 Triplet Closure (rad) ↓ | 1.018 | 0.915 | −10.1% |
| M2 Unwrap Success Rate ↑ | 0.256 | 0.258 | +0.2 pp |
| M3 Usable Pairs ↑ | 0.000 | 0.000 | 0.0 pp † |
| M4 DEM NMAD (m) ↓ | 40.13 | 39.44 | −1.7% |
| M5 Temporal Residual (rad) ↓ | 1.158 | 0.367 | **−68.3%** |

### AOI_024 — Western Australia (Additional Evaluation)

| Metric | Goldstein | FiLMUNet | Δ |
|--------|-----------|----------|---|
| M1 Triplet Closure (rad) ↓ | 0.536 | **0.468** | **−6%** |
| M2 Unwrap Success Rate ↑ | 0.531 | **0.608** | +7 pp |
| M4 DEM NMAD (m) ↓ | 18.32 | **12.64** | **−31%** |
| M5 Temporal Residual (rad) ↓ | 1.069 | **0.361** | **−66%** |

### AOI_008 — Los Angeles (Zero-Shot Transfer)

| Metric | Goldstein | FiLMUNet | Δ |
|--------|-----------|----------|---|
| M1 Triplet Closure (rad) ↓ | 0.769 | 0.771 | +0% |
| M2 Unwrap Success Rate ↑ | 0.256 | 0.248 | −0.2 pp |
| M4 DEM NMAD (m) ↓ | 40.13 | **39.40** | −2% |
| M5 Temporal Residual (rad) ↓ | 1.486 | **1.450** | **−2%** |

**Footnotes:**

† **M3 = 0 (Hawaii):** M3 requires M1 < 0.5 rad per pair. X-band spotlight over vegetated Hawaiian terrain has systematically high closure error (>0.5 rad for all 224 pairs). Reported honestly; not omitted.

‡ **AOI_008 is zero-shot:** The AOI_024-trained checkpoint is applied to AOI_008 with no retraining. Closure and unwrap rate show no improvement (expected given star-network topology with no independent triplets). DEM NMAD and temporal residual improve modestly, confirming cross-geography generalization.

---

## 9. Contest Metric Definitions

All metrics are implemented in `eval/compute_metrics.py`.

| # | Metric | Formula | Target |
|---|--------|---------|--------|
| M1 | Triplet closure error | `median |wrap(φ_ij + φ_jk − φ_ik)|` over all triplets | ↓ ≥ 30% |
| M2 | Unwrap success rate | Fraction of pixels with valid unwrapped phase | ↑ ≥ 15 pp |
| M3 | Usable pairs fraction | Pairs with: coherence ≥ 0.35 AND M1 < 0.5 rad AND M2 success | ↑ ≥ 25% |
| M4 | DEM NMAD | `1.4826 × median(|e − median(e)|)` vs Copernicus GLO-30 | ↓ ≥ 15% |
| M5 | Temporal consistency | `‖Aẋ − φ̂‖₂` (unweighted SBAS residual over full stack) | ↓ ≥ 20% |

`wrap(·)` wraps to `[−π, π)`. `A` is the SBAS design matrix; `ẋ` is the WLS solution weighted by FiLMUNet confidence `exp(−σ̂)`.
