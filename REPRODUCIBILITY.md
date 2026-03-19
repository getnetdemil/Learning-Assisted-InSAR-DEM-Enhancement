# Reproducibility Guide

IEEE GRSS 2026 Data Fusion Contest — Learning-Assisted InSAR DEM Enhancement

## Dataset

**Source**: Capella Space X-band SAR open dataset, accessed via public AWS S3.

| Item | Value |
|------|-------|
| STAC collection URL | `https://capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-ieee-data-contest/collection.json` |
| S3 bucket | `s3://capella-open-data/data/` (region: `us-west-2`, no auth required) |
| Total SLC items | 791 (filtered on `_SLC_` in href) |
| Primary AOI | AOI_000 (Hawaii) — 221 collects, both orbits |
| Secondary AOI | AOI_008 (Los Angeles) — zero-shot transfer only |

### AOI Selection Criteria

AOI_000 (Hawaii) was selected because:
1. Largest collect count (221 acquisitions) among all 39 AOIs
2. Both ascending and descending orbits available
3. Incidence angle range 35.8°–56.3° spans typical Capella SL mode
4. Volcanic terrain provides rich deformation signal for phase closure testing
5. High coherence inland; decorrelated coast provides natural mask testing

### Data Splits

Splits are **temporal** (by reference acquisition date) to prevent geographic leakage:

| Split | Fraction | Date range |
|-------|----------|------------|
| Train | 70% | Earliest 70% by reference date |
| Val   | 15% | Next 15% by reference date |
| Test  | 15% | Latest 15% by reference date (held out) |

Pair graph construction: maximum temporal baseline 180 days, perpendicular
baseline filtering applied per `src/insar_processing/pair_graph.py`.

---

## Data Checksums

SHA-256 checksums for key manifest files (computed at submission time):

| File | SHA-256 |
|------|---------|
| `data/manifests/full_index.parquet` | `3c03228e72071e49d5fb823551fc4c2e556bd0f08b4ada2af4c3fab7622482ac` |
| `data/manifests/hawaii_pairs.parquet` | `242f6511762f092b4cfa70725db3cb0c714b62d63caccfd343eed85b08513894` |
| `data/manifests/hawaii_triplets_strict.parquet` | `c615add1d45e917f1b803c09660aad6fb25b35fd961b367bcc50d6d9b7c9cc0d` |

To verify:
```bash
sha256sum data/manifests/full_index.parquet \
          data/manifests/hawaii_pairs.parquet \
          data/manifests/hawaii_triplets_strict.parquet
```

---

## Random Seeds

All randomness is seeded deterministically:

| Component | Seed | Set by |
|-----------|------|--------|
| Python `random` | 42 | `train_film_unet.py:set_seed()` |
| NumPy | 42 | `train_film_unet.py:set_seed()` |
| PyTorch (CPU+GPU) | 42 | `train_film_unet.py:set_seed()` |
| Training config | `seed: 42` | `configs/train/contest.yaml` |

DataLoader uses `shuffle=True` (train) with the seeded state, and
`drop_last=True` to avoid batch-size variation.

---

## Environment

```bash
conda create -n insar-dem python=3.10
conda activate insar-dem
pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
pip install boto3 pystac snaphu-py==0.4.1
conda install -c conda-forge isce3
```

Key versions confirmed working:
- PyTorch 2.4.0, CUDA 12.1
- snaphu-py 0.4.1
- rasterio 1.3.x
- numpy 1.26.x

---

## One-Command Pipeline: S3 → Metrics

```bash
# 0. Set up env
export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH
PYTHON="conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu python"

# 1. Crawl STAC → build full_index.parquet + hawaii_pairs.parquet + triplets
$PYTHON scripts/download_subset.py \
    --aoi AOI_000 \
    --out_dir data/raw/hawaii

# 2. Build pair graph (if not already done from manifests)
# Manifests are checked into data/manifests/ — skip if present

# 3. Preprocess pairs: coregistration → interferogram → coherence → Goldstein
$PYTHON scripts/preprocess_pairs.py \
    --manifest data/manifests/hawaii_pairs.parquet \
    --pairs_dir data/processed/pairs \
    --workers 4

# 4. Patch coreg_meta with incidence/mode/look/snr (after preprocess)
$PYTHON scripts/patch_coreg_meta.py \
    --pairs_dir data/processed/pairs

# 5. Phase unwrapping with SNAPHU
$PYTHON -u scripts/unwrap_snaphu.py \
    --pairs_dir data/processed/pairs \
    --workers 4 --mode DEFO --nlooks 9.0 --coh_threshold 0.1

# 6. Train FiLMUNet (self-supervised, ~23h on single A100)
$PYTHON experiments/enhanced/train_film_unet.py \
    --data_config  configs/data/capella_aoi_selection.yaml \
    --model_config configs/model/film_unet.yaml \
    --train_config configs/train/contest.yaml

# 7. Evaluate: all 5 contest metrics
$PYTHON eval/compute_metrics.py \
    --checkpoint    experiments/enhanced/checkpoints/film_unet/best_closure.pt \
    --pairs_dir     data/processed/pairs \
    --triplets_manifest data/manifests/hawaii_triplets_strict.parquet \
    --out_dir       experiments/enhanced/outputs
```

Expected outputs:
- `experiments/enhanced/outputs/metrics_comparison.csv`
- `experiments/enhanced/outputs/figures/closure_histogram.png`
- `experiments/enhanced/outputs/figures/phase_comparison.png`
- `experiments/enhanced/outputs/figures/temporal_residual_bar.png`

---

## Ablation Studies

```bash
# Run all 5 ablation variants (5 × 20 epochs each)
bash scripts/run_ablations.sh

# Collect results into a table
python scripts/collect_ablation_results.py
```

Output: `experiments/enhanced/outputs/ablation_table.csv`

---

## Zero-Shot Transfer (AOI_008 — Los Angeles)

```bash
# Select top-30 AOI_008 acquisitions
python eval/zero_shot_transfer.py --phase select \
    --full_index data/manifests/full_index.parquet \
    --out_manifest data/manifests/aoi008_pairs.parquet \
    --n_pairs 30

# Preprocess AOI_008 pairs (see printed command above)
# Then evaluate with the Hawaii checkpoint (no retraining)
python eval/zero_shot_transfer.py --phase eval \
    --checkpoint experiments/enhanced/checkpoints/film_unet/best_closure.pt \
    --pairs_dir data/processed/pairs_aoi008 \
    --out_dir experiments/enhanced/outputs/zero_shot_aoi008
```

---

## Contest Metric Definitions

All metrics are implemented in `src/evaluation/closure_metrics.py`.

| # | Metric | Formula | Target |
|---|--------|---------|--------|
| M1 | Triplet closure error | median |wrap(φ_ij + φ_jk − φ_ik)| | ↓ ≥30% |
| M2 | Unwrap success rate | fraction of coherent pixels with valid unw | ↑ ≥15 pp |
| M3 | Usable pairs fraction | coh≥0.35 AND closure<0.5 rad AND unw success | ↑ ≥25% |
| M4 | DEM NMAD | 1.4826 × median(|e − median(e)|) | ↓ ≥15% |
| M5 | Temporal consistency | ‖W(Ax − φ̂)‖₂ via SBAS | ↓ ≥20% |

---

## Baseline Results (Goldstein filter, 2026-03-17)

| Metric | Value | Notes |
|--------|-------|-------|
| M1 Triplet closure error | 1.018 rad | 62 complete triplets |
| M5 Temporal residual | 0.050 rad | 162 pairs, 138 epochs |
| M2/M3 | pending | Requires SNAPHU unwrapping |
| M4 | N/A | No reference DEM available |
