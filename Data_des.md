# Data Description — IEEE GRSS 2026 Data Fusion Contest

## Table of Contents
1. [Contest Dataset Overview](#1-contest-dataset-overview)
2. [Why Hawaii (AOI_000)?](#2-why-hawaii-aoi_000)
3. [SLC Inventory — 221 Scenes](#3-slc-inventory--221-scenes)
4. [Pair Construction — 8,834 Candidates](#4-pair-construction--8834-candidates)
5. [Processed Pairs — 224 on Disk](#5-processed-pairs--224-on-disk)
6. [Triplets — 24,171 Manifest / 62 Complete](#6-triplets--24171-manifest--62-complete)
7. [Tile Generation — 215,264 Patches](#7-tile-generation--215264-patches)
8. [SBAS Network — 127 Epochs / 162 Pairs](#8-sbas-network--127-epochs--162-pairs)
9. [File Layout](#9-file-layout)
10. [Scripts Reference](#10-scripts-reference)

---

## 1. Contest Dataset Overview

| Item | Value |
|------|-------|
| Contest | IEEE GRSS 2026 Data Fusion Contest |
| Sensor | Capella Space X-band SAR (Spotlight mode) |
| S3 bucket | `s3://capella-open-data/data/` (public, no auth) |
| STAC catalog | `s3://capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-ieee-data-contest/collection.json` |
| Total SLC scenes | **791** across **39 AOIs** worldwide |
| Scene type used | SLC (Single-Look Complex) only — GEO products ignored |
| Radar frequency | 9.65 GHz (X-band), wavelength λ ≈ 0.031 m |
| Polarization | HH (single-pol) |

The contest provides 791 SLC scenes spanning multiple geographic regions. Only SLC products
are used for InSAR processing; GEO (geocoded) products are discarded.

**Why SLC and not GEO?**
InSAR requires the raw complex phase — geocoding resamples and interpolates the SLC,
destroying the phase coherence between acquisitions. SLC-to-SLC interferogram formation
preserves sub-pixel coregistration accuracy.

---

## 2. Why Hawaii (AOI_000)?

**Primary AOI selected: AOI_000 (Hawaii — Big Island)**

| Criterion | Hawaii advantage |
|-----------|-----------------|
| Scene count | 221 SLCs — largest single AOI in the contest |
| Temporal span | 17 months (Jun 2024 – Nov 2025) — longest temporal baseline |
| Both orbits | 113 ascending + 108 descending — enables multi-look geometry |
| Incidence range | 35.8° – 56.3° — large range for FiLM conditioning signal |
| Terrain | Volcanic relief (Mauna Kea: 4,207 m) — challenging for InSAR, high B_perp sensitivity |
| Deformation | Active volcanism → real temporal signal for M5 temporal residual validation |
| Phase diversity | Mixed coherence (urban, lava fields, vegetation) → varied training signal |

**Secondary AOIs** (not yet processed):
- AOI_008 (Los Angeles) — planned for zero-shot transfer demonstration
- AOI_024 (W. Australia) — flat, stable terrain; planned as stable-baseline reference

---

## 3. SLC Inventory — 221 Scenes

### How the index was built

```bash
# Step 1: Crawl STAC catalog → full_index.parquet
python scripts/download_subset.py \
    --index_only \
    --out_manifest data/manifests/full_index.parquet
```

`download_subset.py` walks the STAC PySTAC collection, filters items containing `_SLC_`
in the asset href, and writes one row per scene to a Parquet manifest.

### Breakdown

| Property | Value |
|----------|-------|
| Total Hawaii SLCs | **221** |
| Ascending orbit | **113** scenes |
| Descending orbit | **108** scenes |
| Primary satellite | Capella-13 (201 / 221 = 91%) |
| Other satellites | Capella-14 (16), Capella-10 (3), Capella-9 (1) |
| Date range | 2024-06-04 → 2025-11-12 (~17 months) |
| Pixel spacing (range) | 0.55 m |
| Pixel spacing (azimuth) | 0.05 m (10× oversampled; focused resolution ≈ 0.5 m) |

### Why mostly Capella-13?

Capella-13 was the primary constellation node tasked over Hawaii for the contest. The
other satellites (Capella-9, -10, -14) added opportunistic fills but on different orbital
planes, reducing their interferometric pairing compatibility with the main C13 stack.

### Key columns in `full_index.parquet`

| Column | Description |
|--------|-------------|
| `id` | Scene ID (e.g. `CAPELLA_C13_SP_SLC_HH_20250108020832_20250108020837`) |
| `datetime` | Acquisition UTC timestamp |
| `orbit_state` | `ascending` or `descending` |
| `orbital_plane` | Integer orbital plane index |
| `incidence_angle_deg` | Local incidence angle at scene center |
| `center_freq_ghz` | Radar center frequency (9.65 GHz) |
| `px_spacing_rg_m` | Slant-range pixel spacing (m) |
| `px_spacing_az_m` | Azimuth pixel spacing (m) |
| `lon`, `lat` | Scene center longitude/latitude |
| `bbox_w/s/e/n` | Scene bounding box (degrees) |
| `aoi` | Assigned AOI label (e.g. `AOI_000`) |
| `slc_href` | S3 URL to the SLC GeoTIFF |

---

## 4. Pair Construction — 8,834 Candidates

### From 221 SLCs to 24,310 possible pairs

Every combination of two scenes forms a potential interferometric pair:

```
C(221, 2) = 221 × 220 / 2 = 24,310 total possible pairs
```

### Why only 8,834 survive?

**Filter 1 — Cross-orbit rejection (removes 12,204 pairs)**

Ascending and descending orbits view the terrain from opposite sides. Their SAR geometries
are incompatible for repeat-pass InSAR (different look angles, opposite range directions).
All ascending×descending combinations are discarded:

```
Cross-orbit pairs:  113 × 108 = 12,204  (all rejected)
Intra-orbit pairs:  C(113,2) + C(108,2) = 6,328 + 5,778 = 12,106  (candidates)
```

**Filter 2 — Temporal baseline cutoff (removes ~3,272 pairs)**

Long temporal baselines cause temporal decorrelation (vegetation, weather, surface change).
The pair graph enforces a maximum temporal separation of **307 days**. Pairs exceeding this
threshold are not included in the manifest. Over a 17-month archive, many scene combinations
exceed 10 months → rejected.

After both filters: **8,834 pairs** survive with:
- Temporal baseline: 3.0 – 307.4 days (median: 94.6 days)
- Q-score (coherence proxy): continuous score used to rank pairs for processing

### Manifest file

```bash
# Generated by pair_graph.py (called inside preprocess_pairs.py setup)
# Output: data/manifests/hawaii_pairs.parquet
```

**Key columns in `hawaii_pairs.parquet`**

| Column | Description |
|--------|-------------|
| `id_ref` | Reference (primary) scene ID |
| `id_sec` | Secondary scene ID |
| `dt_days` | Temporal baseline (days) |
| `bperp_m` | Perpendicular baseline (m) — estimated from orbit state vectors |
| `orbit_state` | `ascending` or `descending` |
| `incidence_ref/sec` | Incidence angles at ref and secondary |
| `q_score` | Interferometric quality score ∈ [0, 1] |
| `aoi` | AOI label |

> **Note on manifest B_perp values**: The manifest shows extreme B_perp values
> (−9,143 to +40,244 m) because they are estimated from orbit state-vector interpolation
> without full ISCE3 geometry. The actual B_perp from coregistration (in `coreg_meta.json`)
> is accurate: 1.8 – 780 m. Use `coreg_meta.json` values for any height-of-ambiguity
> calculations.

---

## 5. Processed Pairs — 224 on Disk

### Why 224 out of 8,834?

Full coregistration + interferogram formation takes ~5 minutes per pair on CPU. Processing
all 8,834 pairs would require ~30 days. The top-224 pairs by Q-score were selected to:
1. Build a well-connected SBAS network (each epoch appears in ≥2 pairs)
2. Maximise triplet closure coverage (for training the closure loss)
3. Cover both ascending and descending geometries

A second pass (`select_triplet_completing_pairs.py`) identified missing "third legs" of
triplets to ensure complete (A→B, B→C, A→C) triangles for M1 closure evaluation.

### Preprocessing pipeline

```bash
python scripts/preprocess_pairs.py \
    --pairs_manifest data/manifests/hawaii_pairs.parquet \
    --raw_dir        data/raw/AOI_000 \
    --out_dir        data/processed/pairs \
    --max_pairs      224 \
    --patch_size     4096 \
    --n_workers      4

# After preprocessing: patch FiLM conditioning fields
python scripts/patch_coreg_meta.py \
    --pairs_dir data/processed/pairs \
    --manifest  data/manifests/hawaii_pairs.parquet
```

**What `preprocess_pairs.py` does per pair:**
1. Load both SLC GeoTIFFs (CInt16 complex) via rasterio
2. Extract a 4096×4096 pixel patch from the scene center
3. Estimate sub-pixel coregistration offsets via phase cross-correlation
4. Resample secondary SLC to reference geometry
5. Form complex interferogram: `ifg = SLC_ref × conj(SLC_sec_resampled)`
6. Estimate coherence: windowed complex correlation (9×9 box)
7. Apply Goldstein phase filter (adaptive, exponent α=0.5)
8. Save outputs to `data/processed/pairs/{pair_id}/`

**What `patch_coreg_meta.py` adds to `coreg_meta.json`:**
- `incidence_angle_deg` — mean of ref + sec incidence angles from `full_index.parquet`
- `mode` — always `"SL"` (Spotlight)
- `look_direction` — `"right"` or `"left"` from manifest
- `snr_proxy` — Q-score (used as SNR proxy for FiLM conditioning)

### Statistics of the 224 processed pairs

| Statistic | Value |
|-----------|-------|
| Pairs processed | **224** |
| Unique SLC scenes used | **138** (of 221 available) |
| Temporal baseline | 3.0 – 5.9 days (mean: 4.6 days) |
| |B_perp| (accurate, from coreg) | 1.8 – 780.1 m (mean: 163.9 m) |
| Pairs with |B_perp| > 10 m | **212** (usable for DEM height conversion) |
| Q-score | 0.145 – 0.253 (mean: 0.193) |

> **Why such short temporal baselines (3–6 days)?**
> The Q-score ranking heavily favors short Δt pairs (lower temporal decorrelation).
> With Capella-13 revisiting Hawaii every ~3 days, the best-quality pairs are always
> consecutive observations.

### Output files per pair directory

```
data/processed/pairs/{pair_id}/
├── ifg_raw.tif           # Raw complex interferogram (2-band: Re, Im) float32
├── ifg_goldstein.tif     # Goldstein-filtered complex interferogram (2-band)
├── coherence.tif         # Coherence estimate (1-band, float32, range [0,1])
├── coreg_meta.json       # All geometry metadata (see below)
├── ifg_film_unet.tif     # FiLMUNet denoised interferogram (written by compute_metrics.py)
├── log_var.tif           # Per-pixel log-variance uncertainty from FiLMUNet
├── unw_phase.tif         # Unwrapped phase — Goldstein input (from unwrap_snaphu.py)
└── unw_phase_film_unet.tif  # Unwrapped phase — FiLMUNet input (from unwrap_snaphu.py)
```

**`coreg_meta.json` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id_ref` | str | Reference SLC scene ID |
| `id_sec` | str | Secondary SLC scene ID |
| `dt_days` | float | Temporal baseline (days) |
| `bperp_m` | float | Perpendicular baseline (m) — from coregistration geometry |
| `incidence_angle_deg` | float | Mean incidence angle (degrees) |
| `mode` | str | `"SL"` (Spotlight) |
| `look_direction` | str | `"right"` |
| `snr_proxy` | float | Q-score as SNR proxy |
| `q_score` | float | Interferometric quality score |
| `row_offset_px` | float | Sub-pixel row shift (coregistration result) |
| `col_offset_px` | float | Sub-pixel column shift |
| `patch_size` | int | Patch size in pixels (4096) |
| `patch_row_ref` | int | Top-left row of patch in original SLC |
| `patch_col_ref` | int | Top-left column of patch in original SLC |
| `dinc_deg` | float | Incidence angle difference ref–sec (degrees) |

---

## 6. Triplets — 24,171 Manifest / 62 Complete

### What is a triplet?

A triplet is a set of three acquisition dates (A, B, C) with all three interferometric pairs
formed: A→B, B→C, and A→C. The **triplet closure phase** is:

```
φ_closure = wrap(φ_AB + φ_BC − φ_AC)
```

Ideally zero for a noise-free, deformation-free stack. Non-zero closure indicates phase noise
or unwrapping errors — the signal that the model learns to suppress (M1).

### From 8,834 pairs to 24,171 triplets

```bash
# Generated during pair graph construction
# Output: data/manifests/hawaii_triplets_strict.parquet
```

Every set of three scenes (A, B, C) where all three pairs (A→B, B→C, A→C) exist in the
8,834-pair manifest forms a valid triplet. With 8,834 pairs over 221 scenes:

```
24,171 triplets survive the "strict" filter:
  - All 3 legs exist in hawaii_pairs.parquet
  - All 3 temporal baselines ≤ 307 days
  - Same orbit direction (all three scenes must be asc or all desc)
```

**Key columns in `hawaii_triplets_strict.parquet`:**

| Column | Description |
|--------|-------------|
| `id_a` | First scene ID (earliest date) |
| `id_b` | Middle scene ID |
| `id_c` | Last scene ID (latest date) |

### From 24,171 manifest triplets to 62 complete on disk

Only **62 triplets** have all three legs available in `data/processed/pairs/`:

```
24,171 manifest triplets
      │
      └── requires all 3 pairs to be in data/processed/pairs/
                │
                └── 62 complete triplets
                     (used for M1 closure eval and training closure loss)
```

The 62 complete triplets were intentionally selected by `select_triplet_completing_pairs.py`,
which found the missing "third legs" and added them to the processing queue:

```bash
python scripts/select_triplet_completing_pairs.py \
    --pairs_manifest    data/manifests/hawaii_pairs.parquet \
    --triplets_manifest data/manifests/hawaii_triplets_strict.parquet \
    --processed_dir     data/processed/pairs \
    --out_parquet       data/manifests/triplet_completing_pairs.parquet
```

### Triplet-tiles in training

The `TripletTileDataset` loads tiles from all three pairs of each complete triplet. With
61,150 triplet-tiles loaded:

```
51,150 triplet-tiles ≈ 62 triplets × 961 tiles/pair × ~86% valid tile fraction
```

The ~14% loss comes from tiles rejected for: all-NaN phase, coherence below threshold,
or patch boundary overlap.

---

## 7. Tile Generation — 215,264 Patches

### Why tile the 4096×4096 patches?

The FiLMUNet model processes 256×256 pixel tiles (GPU memory constraint). Tiling with 50%
overlap provides spatial diversity and augmentation-equivalent coverage without duplicating
entire pairs.

### Tiling parameters

| Parameter | Value |
|-----------|-------|
| Patch size (input) | 4096 × 4096 pixels |
| Tile size | 256 × 256 pixels |
| Stride | 128 pixels (50% overlap) |
| Tiles per axis | floor((4096 − 256) / 128) + 1 = **31** |
| Tiles per pair | 31 × 31 = **961** |

### Dataset splits

Splits are **AOI-based** (by acquisition date order), NOT random tile splits. This prevents
geographic leakage where the same terrain appears in both train and test.

| Split | Pairs | Tiles | Fraction |
|-------|-------|-------|----------|
| Train | **156** | **149,916** | 69.6% |
| Validation | **33** | **31,713** | 14.7% |
| Test (held out) | **35** | **33,635** | 15.7% |
| **Total** | **224** | **215,264** | 100% |

### Why AOI-based splits and not random?

Random tile-level splits would allow the model to see one tile from a pair in training
and an adjacent tile from the same pair in validation — effectively leaking the terrain
and deformation pattern. AOI-based splits ensure that **entire pairs** (and their geographic
footprint) appear in only one split.

---

## 8. SBAS Network — 127 Epochs / 162 Pairs

### SBAS (Small Baseline Subset) inversion

SBAS jointly inverts the interferometric phase time-series to recover surface deformation
velocities. The linear system is:

```
A × x = φ_unwrapped

where:
  A ∈ R^{P × (T-1)}  — design matrix (1 per arc in each pair's time span)
  x ∈ R^{T-1}        — unknown displacement increments between epochs
  φ ∈ R^P            — unwrapped phase observations (one per pair)
  P = number of pairs
  T = number of unique acquisition epochs
```

### Network statistics

| Quantity | Value | Reason |
|----------|-------|--------|
| Unique SLC scenes used | 138 | Scenes appearing in ≥1 processed pair |
| Unique acquisition dates (T) | **127** | Some scenes share acquisition date within hours |
| Pairs in SBAS (P) | **162** | Of 224 total, 62 are triplet-completing extras; valid SBAS graph |
| SBAS unknowns | T − 1 = **126** | Relative displacements between consecutive epochs |
| Over-determination | P − (T−1) = 162 − 126 = **+36** | 36 redundant observations → least-squares solution |

### Why overconstrained (+36) is good

An overconstrained SBAS system (`P > T-1`) is **desirable**:
- Least-squares inversion averages out phase noise across redundant paths
- The residual `‖φ_unwrapped − A × x_star‖` (M5 metric) is small when the network is
  internally consistent
- FiLMUNet reduces phase noise → lower M5 residual → better SBAS solution

### Phase weighting for SBAS

FiLMUNet outputs a per-pixel log-variance `log_var(p)` alongside the denoised phase.
This is used to weight the SBAS observations:

```
W(p) = 1 / σ²(p) = 1 / exp(log_var(p))
```

High-confidence pixels (low log_var) contribute more to the inversion. However, for the
**M5 metric**, the residual is reported **unweighted** in phase space:

```
M5 = ‖φ_stack − A × x_star‖₂   (unweighted, in radians)
```

This avoids inflating the metric by the weight scale.

---

## 9. File Layout

```
data/
├── raw/                            # Downloaded SLC GeoTIFFs (not in git)
│   └── {scene_id}/{scene_id}.tif
│
├── manifests/                      # All index and pair manifests
│   ├── full_index.parquet          # 791 SLC rows, 39 AOIs
│   ├── hawaii_pairs.parquet        # 8,834 candidate pairs for AOI_000
│   ├── hawaii_triplets_strict.parquet  # 24,171 triplets from 8,834 pairs
│   └── triplet_completing_pairs.parquet  # Pairs added to complete triplets
│
├── processed/
│   └── pairs/                      # 224 processed pair directories
│       └── {pair_id}/
│           ├── ifg_raw.tif
│           ├── ifg_goldstein.tif
│           ├── coherence.tif
│           ├── coreg_meta.json
│           ├── ifg_film_unet.tif       (written by eval)
│           ├── log_var.tif             (written by eval)
│           ├── unw_phase.tif           (written by unwrap_snaphu.py, Goldstein)
│           └── unw_phase_film_unet.tif (written by unwrap_snaphu.py, FiLMUNet)
│
└── reference/
    └── copernicus_dem/
        ├── tiles/                  # Individual GLO-30 1°×1° tiles
        └── hawaii_dem.tif          # Merged Copernicus DEM for Hawaii
```

---

## 10. Scripts Reference

| Script | Purpose | Key output |
|--------|---------|------------|
| `scripts/download_subset.py` | Crawl STAC catalog; build `full_index.parquet`; optionally download SLCs | `data/manifests/full_index.parquet` |
| `scripts/preprocess_pairs.py` | Coregistration → interferogram → coherence → Goldstein filter | `data/processed/pairs/{id}/` |
| `scripts/patch_coreg_meta.py` | Add FiLM conditioning fields to `coreg_meta.json` | Updated `coreg_meta.json` |
| `scripts/select_triplet_completing_pairs.py` | Find missing 3rd legs of triplets for processing | `data/manifests/triplet_completing_pairs.parquet` |
| `scripts/unwrap_snaphu.py` | SNAPHU phase unwrapping (Goldstein or FiLMUNet input) | `unw_phase.tif` or `unw_phase_film_unet.tif` |
| `scripts/download_copernicus_dem.py` | Download Copernicus GLO-30 DEM tiles for Hawaii | `data/reference/copernicus_dem/hawaii_dem.tif` |
| `experiments/enhanced/train_film_unet.py` | Train FiLMUNet with N2N + closure + temporal losses | `checkpoints/film_unet/{run}/final.pt` |
| `eval/compute_metrics.py` | Compute all 5 contest metrics (M1–M5) | `experiments/enhanced/outputs/metrics_*.csv` |

### Typical end-to-end run sequence

```bash
# 1. Build manifest
python scripts/download_subset.py --index_only \
    --out_manifest data/manifests/full_index.parquet

# 2. Preprocess top-224 Hawaii pairs
python scripts/preprocess_pairs.py \
    --pairs_manifest data/manifests/hawaii_pairs.parquet \
    --raw_dir data/raw/AOI_000 \
    --out_dir data/processed/pairs \
    --max_pairs 224 --patch_size 4096 --n_workers 4

# 3. Patch FiLM metadata fields
python scripts/patch_coreg_meta.py \
    --pairs_dir data/processed/pairs \
    --manifest  data/manifests/hawaii_pairs.parquet

# 4. Find and process triplet-completing pairs
python scripts/select_triplet_completing_pairs.py \
    --pairs_manifest    data/manifests/hawaii_pairs.parquet \
    --triplets_manifest data/manifests/hawaii_triplets_strict.parquet \
    --processed_dir     data/processed/pairs \
    --out_parquet       data/manifests/triplet_completing_pairs.parquet

python scripts/preprocess_pairs.py \
    --pairs_manifest data/manifests/triplet_completing_pairs.parquet \
    --raw_dir data/raw/AOI_000 \
    --out_dir data/processed/pairs --n_workers 4

# 5. Train FiLMUNet
python experiments/enhanced/train_film_unet.py \
    --data_config  configs/data/capella_aoi_selection.yaml \
    --model_config configs/model/film_unet.yaml \
    --train_config configs/train/contest.yaml \
    --run_name     raw2gold_closure \
    --triplets_manifest data/manifests/hawaii_triplets_strict.parquet

# 6. SNAPHU unwrapping (Goldstein)
python scripts/unwrap_snaphu.py \
    --pairs_dir data/processed/pairs --workers 4

# 7. FiLMUNet inference + SNAPHU on FiLMUNet output
python eval/compute_metrics.py \
    --checkpoint experiments/enhanced/checkpoints/film_unet/.../final.pt \
    --pairs_dir data/processed/pairs \
    --triplets_manifest data/manifests/hawaii_triplets_strict.parquet \
    --out_dir experiments/enhanced/outputs \
    --force_inference

python scripts/unwrap_snaphu.py \
    --pairs_dir data/processed/pairs \
    --input_ifg ifg_film_unet.tif \
    --output_name unw_phase_film_unet.tif \
    --workers 4

# 8. Download reference DEM
python scripts/download_copernicus_dem.py \
    --bbox_w -158 --bbox_s 18 --bbox_e -154 --bbox_n 22 \
    --out_dir data/reference/copernicus_dem

# 9. Final evaluation (all 5 metrics)
python eval/compute_metrics.py \
    --checkpoint experiments/enhanced/checkpoints/film_unet/.../final.pt \
    --pairs_dir data/processed/pairs \
    --triplets_manifest data/manifests/hawaii_triplets_strict.parquet \
    --out_dir experiments/enhanced/outputs \
    --skip_inference \
    --copernicus_dem_dir data/reference/copernicus_dem
```

---

*Last updated: 2026-03-31 | Contest deadline: 2026-04-06*
