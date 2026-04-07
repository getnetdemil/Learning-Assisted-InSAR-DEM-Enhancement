# IEEE GRSS 2026 Data Fusion Contest — Development Plan

**Submission deadline: April 06, 2026 (23:59 AoE)**
**Paper deadline: April 20 (internal) / April 28 (final)**
**Today: March 18, 2026 — 19 days to submission**

---

## Table of Contents
1. [Contest Overview and Objectives](#1-contest-overview-and-objectives)
2. [Dataset: Capella Space SAR Open Data](#2-dataset-capella-space-sar-open-data)
3. [Success Metrics (Official)](#3-success-metrics-official)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Week-by-Week Implementation Plan](#5-week-by-week-implementation-plan)
6. [Phase 1: Data Access and Pair-Graph Construction](#6-phase-1-data-access-and-pair-graph-construction)
7. [Phase 2: Baseline InSAR Products](#7-phase-2-baseline-insar-products)
8. [Phase 3: Deep Learning Method](#8-phase-3-deep-learning-method)
9. [Phase 4: Evaluation and Paper](#9-phase-4-evaluation-and-paper)
10. [Reproducibility Checklist](#10-reproducibility-checklist)
11. [Technical Specifications](#11-technical-specifications)

---

## 1. Contest Overview and Objectives

### 1.1 Context

The IEEE GRSS 2026 Data Fusion Contest (DFC26) provides a **Capella Space commercial SAR stack**: ~1582 unique collects, enabling 17,000+ possible interferometric pairs across multiple AOIs, with substantial diversity in mode, incidence angle, look direction, and orbital geometry.

Evaluation is **committee-based** (not leaderboard), rewarding:
- Soundness and originality
- Insight and effective dataset use
- Scalability (optional but favorable)
- Reproducibility (public code, one-command pipeline)

**A winning approach must be explicitly stack-aware and geometry-aware.**

### 1.2 Contest Objectives (maps to evaluation criteria)

| ID | Objective |
|----|-----------|
| O1 | Increase stack usability under geometry diversity by improving interferogram quality and unwrapping reliability (closure + unwrap metrics) |
| O2 | Improve downstream products: DEM (where geometry permits) and time-series consistency via uncertainty-weighted inversion |
| O3 | Produce interpretable "temporal storytelling" artifacts: pair graphs, maps, timelines showing why the method helps |
| O4 | Reproducibility: one-command pipeline from S3 download → results figures, plus trained weights and configs |

### 1.3 Key Dates

| Date | Event |
|------|-------|
| Feb 04, 2026 | Data released |
| **Mar 05, 2026** | **Today — start of active development** |
| **Apr 06, 2026** | **Submission deadline (23:59 AoE)** |
| Apr 13, 2026 | Winners announced |
| Apr 20, 2026 | Internal paper deadline |
| Apr 28, 2026 | Final paper deadline |

---

## 2. Dataset: Capella Space SAR Open Data

### 2.1 Data Access

```bash
# Validate open access (no authentication required)
aws s3 ls --no-sign-request s3://capella-open-data/data/

# Inventory dataset size
aws s3 ls --no-sign-request --recursive --summarize s3://capella-open-data/data/ > s3_inventory.txt
```

- **S3 bucket**: `s3://capella-open-data/data/` (region: `us-west-2`)
- **STAC root**: `https://capella-open-data.s3.us-west-2.amazonaws.com/stac/catalog.json`
- **Contest collection**: child link `"IEEE Data Contest 2026"` → `capella-open-data-ieee-data-contest/collection.json`
- The catalog is **static STAC** (JSON files linked via root/child/item), NOT a STAC API — use `pystac`, not `pystac-client` unless building a local index.

### 2.2 Product Format (Capella SLC)

- **Format**: Cloud-Optimized GeoTIFF (`.tif`) + STAC JSON + extended JSON sidecar
- **Pixel type**: Complex CInt16 (real Int16 + imaginary Int16 = 32 bits/pixel)
- **Radiometric convention**: SLC relates to Beta Nought; a `scale_factor` in extended metadata converts to radar brightness
- **Metadata parsing**: use `capella-reader` which provides orbit, Doppler, and ISCE3 adapter

```python
from capella_reader import CapellaSLC
slc = CapellaSLC.from_file("path/to/slc.tif")
orbit = slc.adapted.isce3_orbit  # ISCE3 orbit object
```

### 2.3 Compute/Storage Tiers

| Tier | SLC count | Pair count | Storage | GPUs |
|------|-----------|------------|---------|------|
| Minimal (proof-of-pipeline) | 40–120 | 300–1500 | 1–3 TB | 1×24 GB |
| Competitive (3–6 AOIs, multi-mode) | 200–600 | 3k–15k | 6–20 TB | 2×48 GB |
| Ideal (many AOIs/modes) | 800–1582 | 10k–30k | 20–60 TB | 4–8×80 GB |

**Target**: Competitive tier minimum to claim cross-geometry generalization.

### 2.4 Required Additional Packages

Beyond `requirements.txt`:
```bash
pip install boto3 pystac capella-reader dask geopandas
# ISCE3 via conda-forge or from source
conda install -c conda-forge isce3
```

---

## 3. Success Metrics (Official)

All five metrics must be implemented in `src/evaluation/` and reported per-AOI and aggregated.

### 3.1 Triplet Closure Error

**Definition**: For three acquisitions (i, j, k): `c_ijk(p) = wrap(φ_ij(p) + φ_jk(p) − φ_ik(p))`

Aggregate as `median_p(|c_ijk(p)|)` then median over triplets.

**Target**: Median closure error ↓ **30–50%** on stable pixels across multiple AOIs/modes.

```python
# src/evaluation/closure_metrics.py
def triplet_closure_error(phi_ij, phi_jk, phi_ik):
    """Returns per-pixel closure; report median and quantiles."""
    closure = np.angle(np.exp(1j * (phi_ij + phi_jk - phi_ik)))
    return closure
```

### 3.2 Unwrap Success Rate

**Definition**: Pass/fail per interferogram:
1. Unwrapped connected component covers ≥90% of stable mask
2. Triplet closure gate: 95th percentile < 1 rad for triplets containing that edge

**Baseline**: SNAPHU (statistical-cost network-flow MAP)

**Target**: Unwrap success rate ↑ **+15–25 percentage points**.

### 3.3 Percent Usable Pairs

**Definition**: % of candidate edges passing ALL gates:
- Median coherence > 0.35
- Unwrap success
- Closure gate

**Target**: Percent usable pairs ↑ **+20–40%**.

### 3.4 DEM NMAD

**Definition**: `NMAD = 1.4826 × median(|e − median(e)|)` where `e = h_pred − h_ref`

Computed on stable terrain mask; external reference DEM used **only for evaluation**, never as a training target.

**Target**: DEM NMAD ↓ **15–25%** (where baseline geometry allows DEM sensitivity).

### 3.5 Temporal Consistency Residual

**Definition**: Stack inversion residual — solve `Ax ≈ φ` (SBAS-like); compute `‖W(Ax − φ)‖₂` or per-edge residual quantiles.

**Target**: Temporal consistency residual ↓ **≥20%**.

### 3.6 Recommended Evaluation Thresholds (Targets)

| Metric | Target improvement |
|--------|-------------------|
| Closure error median | ↓ ≥30% |
| Unwrap success rate | ↑ ≥15 pp |
| Usable pairs | ↑ ≥25% |
| Temporal residual | ↓ ≥20% |
| DEM NMAD | ↓ ≥15% |

---

## 4. Pipeline Architecture

```
S3 (Capella SAR) → STAC crawl → metadata parquet
        |
        v
   Pair-graph construction (nodes=collects, edges=candidate pairs)
   Edge scoring: Q_ij = f(Δt, Δθ_inc, Δθ_graze, B_perp, SNR_proxy)
   Pair selection: time-series subgraph + DEM-sensitivity subgraph
        |
        v
   SLC coregistration (ISCE3 + capella-reader)
   Interferogram formation: I_ij = S_i · S_j*
   Coherence estimation
   Classical baselines: Goldstein, NL-InSAR, BM3D
   Unwrapping: SNAPHU
        |
        v
   DL model: f_θ(I_ij | metadata) → (Î_ij, log σ²)
   Self-supervised training:
     - Noise2Noise (sub-look splits)
     - Heteroscedastic NLL (uncertainty)
     - Closure-consistency triplet loss
     - Temporal-consistency (SBAS residual) loss
     - Spectral/fringe-preservation loss
        |
        v
   Uncertainty-weighted unwrapping (SNAPHU with weights)
   Uncertainty-weighted stack inversion (SBAS)
   DEM generation (where geometry permits)
        |
        v
   Evaluation: closure / unwrap / usable pairs / DEM NMAD / temporal residual
   Visualization: pair graphs, maps, timelines ("temporal storytelling")
   4-page paper + public GitHub
```

---

## 5. Week-by-Week Implementation Plan

```
Week 1 (Mar 05–10): Data access + STAC index + pair-graph skeleton        ✓ DONE
Week 2 (Mar 10–18): Baseline InSAR products + model + eval pipeline        ✓ DONE
  ↳ 221 SLCs downloaded (497 GB), 162 pairs preprocessed
  ↳ FiLMUNet + losses built; eval/compute_metrics.py implemented
  ↳ Goldstein baseline: M1=1.018 rad, M5=0.050 rad (M2/M3/M4 need SNAPHU)
Week 3 (Mar 18–24): SNAPHU unwrapping + FiLMUNet training + full metrics   ← CURRENT
  ↳ Task A DONE: patch_coreg_meta.py — 224 coreg_meta.json patched with incidence/mode/look/snr
  ↳ Task B RUNNING: unwrap_snaphu.py (PID 2973463, started 13:15 Mar 18, ~23 hrs total)
  ↳ Task C RUNNING: train_film_unet.py (PID 2905041, epoch 5/50 at 13:02 Mar 18, ~21 hrs total)
  ↳ Task D PENDING: eval/compute_metrics.py — run after B+C complete (~9 AM Mar 19)
  ↳ Fixes: unwrap_snaphu.py: ntiles threshold >= 4096, tile_overlap=max(128,5%)
Week 4 (Mar 24–Apr 02): Ablation studies + paper figures + zero-shot AOI_008
Week 5 (Apr 02–06):  Final paper + repo cleanup + submit
```

---

## 6. Phase 1: Data Access and Pair-Graph Construction

### 6.1 STAC Crawl and Local Index

**Task 1.1: Crawl contest collection and build metadata parquet**

```python
# scripts/download_subset.py
import pystac

root = pystac.Catalog.from_file(
    "https://capella-open-data.s3.us-west-2.amazonaws.com/stac/catalog.json"
)
contest = pystac.Collection.from_file(
    "https://capella-open-data.s3.us-west-2.amazonaws.com/stac/"
    "capella-open-data-ieee-data-contest/collection.json"
)

# Traverse items, write each to data/stac_cache/items/<item_id>.json
# Extract to parquet: item_id, datetime, bbox, mode, look_direction,
#   flight_direction, incidence, asset hrefs + sizes
```

**Task 1.2: Subset selection strategy**
- Inspect mode/incidence/AOI distribution from parquet
- Select 3–6 diverse AOIs covering different land cover and incidence ranges
- Document selection rationale (reproducibility requirement)
- Download manifest with checksums → `data/manifests/subset_manifest.csv`

### 6.2 Pair-Graph Construction

**Task 1.3: Node and edge definitions**

- **Node** v_i: `collect_id, datetime, mode, look_direction, flight_direction, incidence_deg`
- **Edge** e_ij: `Δt, Δθ_inc, Δθ_graze, B_perp (computed from orbits), SNR_proxy`

**Task 1.4: Perpendicular baseline computation**

`B_perp` is NOT in the contest CSV — compute from orbit/state vectors using ISCE3:
```python
# src/insar_processing/geometry.py
from isce3.geometry import compute_perpendicular_baseline
```

**Task 1.5: Edge quality score formula**

```
Q_ij = 1[mode_compat] × 1[look_compat] × 1[flight_compat]
       × exp(−α_t × Δt/τ_t)
       × exp(−α_inc × |Δθ_inc|/τ_inc)
       × exp(−α_g × |Δθ_graze|/τ_g)
       × exp(−α_b × |B_perp|/τ_b)
       × σ(β₀ + β₁ × SNR_proxy_ij)
```

Default hyperparameters: `τ_t=30d, τ_inc=5°, τ_g=3°, τ_b=300m`

**Task 1.6: Pair selection strategy**

For each AOI, select:
1. **Time-series subgraph**: emphasize small Δt and high Q_ij (for temporal consistency)
2. **DEM-sensitivity subgraph**: emphasize diverse B_perp while keeping Q_ij ≥ threshold

Cap total edges to compute budget. Document budget in `configs/data/pair_selection.yaml`.

**Deliverables Phase 1:**
- [x] `data/stac_cache/` — local STAC item JSONs (791 SLC items)
- [x] `data/manifests/full_index.parquet` — 791 SLC rows, 39 AOIs assigned
- [x] `data/manifests/hawaii_pairs.parquet` — 8,834 pairs with B_perp
- [x] `data/manifests/hawaii_triplets_strict.parquet` — 24,171 triplets
- [x] `src/insar_processing/pair_graph.py` — graph construction + scoring (Q_ij)
- [x] `src/insar_processing/geometry.py` — B_perp from state-vector interpolation
- [x] `src/insar_processing/sublook.py` — FFT sub-look split for N2N (phase corr = 0.001)
- [x] `src/insar_processing/filters.py` — Goldstein + adaptive Goldstein + boxcar coherence
- [ ] `notebooks/01_pair_graph_exploration.ipynb` — visual exploration ("temporal storytelling")

---

## 7. Phase 2: Baseline InSAR Products

### 7.1 SLC Coregistration (ISCE3-first)

**Task 2.1: Per-pair coregistration**

```bash
# scripts/preprocess_pairs.py
python scripts/preprocess_pairs.py \
    --manifest data/manifests/subset_manifest.csv \
    --out_dir data/processed/
```

Steps per pair:
1. Parse metadata with `capella-reader` (orbit, Doppler)
2. Coarse geometric alignment using ISCE3 orbit geometry
3. Refine with correlation offsets
4. Save misregistration metrics (become features for usability predictor)

### 7.2 Interferogram and Coherence

**Task 2.2**: Form `I_ij = S_i · conj(S_j)` and estimate coherence in sliding window

Product outputs per pair (GeoTIFF):
- `interferogram_real.tif`, `interferogram_imag.tif` (or stacked complex)
- `coherence.tif`
- `amplitude_ref.tif`, `amplitude_sec.tif`

### 7.3 Classical Filtering Baselines

**Task 2.3**: Implement three non-DL baselines for fair comparison:
1. **Goldstein adaptive filter** — `src/insar_processing/filters.py`
2. **NL-InSAR** (nonlocal estimator) — `src/insar_processing/filters.py`
3. **BM3D-like denoiser** (applied to complex channels) — `src/insar_processing/filters.py`

### 7.4 Phase Unwrapping

**Task 2.4: SNAPHU baseline**

```bash
# scripts/unwrap_snaphu.py
snaphu -f snaphu.conf wrapped_phase.bin <width>
```

Use consistent coherence mask threshold (report mask percentage). SNAPHU is the required baseline for reproducibility.

### 7.5 Closure Metrics on Baseline Products

**Task 2.5**: Compute all 5 contest metrics on classical-baseline products to establish numbers to beat.

```python
# src/evaluation/closure_metrics.py
def compute_all_contest_metrics(pair_products, triplets, stable_mask, ref_dem=None):
    """Returns dict with all 5 contest metrics for a set of pairs."""
```

**Deliverables Phase 2:**
- [x] `data/processed/pairs/` — 162 unique pairs (ifg_raw.tif, ifg_goldstein.tif, coherence.tif, coreg_meta.json)
- [x] `scripts/preprocess_pairs.py` — coreg → interferogram → Goldstein → coherence
- [x] `scripts/select_triplet_completing_pairs.py` — 62 triplet-completing pairs selected + preprocessed
- [x] `src/insar_processing/filters.py` — Goldstein + adaptive baseline (NL-InSAR/BM3D optional)
- [x] `scripts/unwrap_snaphu.py` — SNAPHU unwrapping (RUNNING, ~23 hrs, 224 pairs × 4 workers)
- [x] `scripts/patch_coreg_meta.py` — FiLM metadata patch (224/224 done, Mar 18)
- [x] `eval/compute_metrics.py` — all 5 contest metrics implemented (FiLMUNet inference + tables)
- [x] Goldstein baseline numbers: M1=1.018 rad (62 triplets), M5=0.050 rad; M2/M3/M4 need SNAPHU

---

## 8. Phase 3: Deep Learning Method

### 8.1 Model Architecture

**Primary choice: U-Net with FiLM metadata conditioning**

- **Input**: Complex interferogram as 2 channels (Re I_ij, Im I_ij); optionally + coherence channel
- **Conditioning**: FiLM layers conditioned on `[Δt, Δθ_inc, Δθ_graze, B_perp, mode_embed, look_embed, SNR_proxy]`
- **Output**: (1) Denoised complex interferogram Î_ij (2 channels); (2) Per-pixel log-variance `log σ²(p)` (1 channel)

```python
# src/models/film_unet.py
class FiLMUNet(nn.Module):
    """U-Net with Feature-wise Linear Modulation for geometry conditioning."""
    def __init__(self, in_channels=2, metadata_dim=7, features=[32,64,128,256]):
        ...
    def forward(self, x, metadata):
        # metadata: (B, metadata_dim) tensor
        # returns: denoised (B, 2, H, W), log_var (B, 1, H, W)
```

**Optional extension (if time permits): Graph-conditioned FiLM** — compute collect-level node embeddings, use edge embedding as FiLM input. Most contest-aligned novel contribution.

### 8.2 Self-Supervised Training (No Reference DEM Required)

Training is **fully self-supervised** — the contest provides no ground-truth clean interferograms.

**Task 3.1: Noise2Noise dataset creation**

Construct two noisy views (I^(a), I^(b)) per pair using:
- Sub-look / aperture splits from the SLC
- OR stochastic multilook windowing with different random windows

**Task 3.2: Loss functions**

```python
# src/losses/physics_losses.py

# L_N2N: Noise2Noise L1
L_n2n = E[|Î_ij − I^(b)_ij|₁]

# L_unc: Heteroscedastic NLL (uncertainty)
L_unc = Σ_p [ |Î(p) − I^(b)(p)|² / σ²(p) + log σ²(p) ]

# L_closure: Closure-consistency triplet loss
c_ijk(p) = wrap(φ̂_ij(p) + φ̂_jk(p) − φ̂_ik(p))
L_closure = Σ_p w(p) × (1 − cos(c_ijk(p)))   # w from uncertainty/coherence

# L_temporal: Stack inversion residual (SBAS-like)
x* = argmin_x ‖W(Ax − φ̂)‖²
L_temporal = ‖W(Ax* − φ̂)‖²

# L_grad: Spectral/fringe-preservation
L_grad = ‖∇φ̂ − ∇φ_raw‖₁

# Combined
L = λ₁L_n2n + λ₂L_unc + λ₃L_closure + λ₄L_temporal + λ₅L_grad
```

Default: `λ₁=1.0, λ₂=0.5, λ₃=0.3, λ₄=0.2, λ₅=0.1`

### 8.3 Training Configuration

**AOI-based splits** (prevents geographic leakage):
- Train AOIs: 60–70%
- Validation AOIs: 10–20%
- Test AOIs: 20–30% (held out until final evaluation)

**Augmentations** (physically safe):
- 90° rotations, flips
- Global phase offset: `φ → wrap(φ + Δ)` (arbitrary Δ)
- Amplitude scaling in log space
- Random multilook window jitter for Noise2Noise

**Batch strategy**: sample AOI → sample edge → sample patch. Upsample rare modes and high incidence-difference pairs.

**Checkpointing**: Save (1) best-by-validation closure error, (2) best-by-validation unwrap success, (3) final epoch. Store `config.yaml`, git hash, dataset manifest per run.

### 8.4 Uncertainty-Weighted Downstream Integration

**Task 3.3**: Use predicted `σ²(p)` to weight:
1. SNAPHU inputs: coherence-like weight map from `1/σ²(p)`
2. SBAS stack inversion: design matrix weights `W = diag(1/σ²)`
3. DEM generation: weighted multi-baseline combination

### 8.5 Training Script

```bash
python experiments/enhanced/train_film_unet.py \
    --data_config configs/data/capella_aoi_selection.yaml \
    --model_config configs/model/film_unet.yaml \
    --train_config configs/train/contest.yaml
```

**Deliverables Phase 3:**
- [x] `src/models/film_unet.py` — FiLM-conditioned U-Net (7.96M params, smoke-tested)
- [x] `src/losses/physics_losses.py` — all 5 loss components (N2N, NLL, closure, temporal, grad)
- [x] `src/losses/__init__.py` — package init
- [x] `src/insar_processing/sublook.py` — sub-look splitting for N2N
- [x] `experiments/enhanced/train_film_unet.py` — training script with AOI-based splits
- [ ] Trained model checkpoint + config logged  ← **NEXT** (after SNAPHU)
- [ ] First DL-vs-baseline closure metric comparison (requires trained checkpoint)

---

## 9. Phase 4: Evaluation and Paper

### 9.1 Ablation Studies (Minimum Set)

| Ablation | Description |
|----------|-------------|
| N2N-only | Remove closure, temporal, gradient losses |
| +closure | Add L_closure |
| +temporal | Add L_temporal |
| +uncertainty | Add L_unc, integrate uncertainty downstream |
| +FiLM | Full model with metadata conditioning |
| −FiLM | Remove metadata conditioning (FiLM off) |

### 9.2 Metrics Tables to Produce

1. **Main metrics table** per AOI and aggregated:
   - closure error (median, 95th pct)
   - unwrap success rate
   - usable pairs %
   - temporal residual norm
   - DEM NMAD (where applicable)

2. **Pair-selection table**: edges by heuristic vs predictor, fraction usable, compute cost

3. **Ablation table**: each loss component contribution to each metric

### 9.3 Visualization ("Temporal Storytelling")

- Pair-graph diagrams (nodes=collects, edges colored by Q_ij and usability)
- Spatial maps: coherence, closure error, unwrap success, per-pixel uncertainty
- Time-series residual maps
- Side-by-side: raw interferogram / Goldstein / DL-denoised / with closure phase overlay

### 9.4 Paper Structure (4 pages, excluding references)

1. **Problem framing**: geometry-diverse temporal InSAR stacks; why pair-graph + closure matters
2. **Method diagram** + key equations (pair scoring; DL loss; uncertainty integration)
3. **Main results table** (metrics) + 2–3 striking visual comparisons
4. **Ablation table** (compact)
5. **Scalability discussion** (1–2 sentences)

### 9.5 Repository Scripts Required

```
scripts/
  download_subset.py       # STAC crawl → manifest → S3 download
  preprocess_pairs.py      # coreg → interferogram → coherence
  unwrap_snaphu.py         # SNAPHU unwrapping
eval/
  compute_metrics.py       # all 5 contest metrics → tables + figures
REPRODUCIBILITY.md         # STAC root URL, collection ID, download manifest,
                           # checksums, random seeds, deterministic settings
```

**Deliverables Phase 4:**
- [ ] All 5 contest metrics computed for all methods (baseline + DL variants)
- [ ] Ablation table complete
- [ ] Pair-graph visualizations
- [ ] 4-page paper draft submitted
- [ ] Public GitHub repo with one-command reproduction

---

## 10. Reproducibility Checklist

The contest **explicitly disqualifies** entries with missing code or unclear reproducibility.

- [ ] `README.md` — one-line purpose, environment creation, one-command reproduction
- [ ] `REPRODUCIBILITY.md` — STAC root URL, contest collection ID, exact download manifest + checksums, fixed random seeds, deterministic settings
- [ ] `configs/` — default configs for baseline and DL runs; AOI split definitions
- [ ] `scripts/download_subset.py` — full pipeline from STAC → download
- [ ] `scripts/preprocess_pairs.py` — coreg → interferogram → coherence
- [ ] `scripts/unwrap_snaphu.py` — reproducible unwrapping
- [ ] `eval/compute_metrics.py` — all metrics + all paper figures
- [ ] Trained model weights uploaded (Hugging Face or GitHub release)
- [ ] Git hash stored in each checkpoint config
- [ ] `environment.yml` or `requirements_contest.txt` with pinned versions

---

## 11. Technical Specifications

### 11.1 Software Stack

```yaml
python: "3.10"
pytorch: "2.1+"
cuda: "12.1"
key_packages:
  - capella-reader     # Capella SLC metadata + ISCE3 adapter
  - isce3              # Coregistration, geometry, orbit
  - pystac             # Static STAC catalog traversal
  - boto3              # AWS S3 access
  - rasterio           # Raster I/O
  - pyproj             # Coordinate transforms
  - numpy, scipy, xarray, dask
  - geopandas, shapely # Geospatial ops
  - snaphu             # Phase unwrapping
  - wandb              # Experiment tracking (optional but recommended)
```

### 11.2 Key Physical Constants (Capella X-band)

| Parameter | Value |
|-----------|-------|
| Band | X-band |
| Wavelength | ~3.1 cm |
| SLC pixel format | CInt16 |
| Calibration | Beta Nought (SLC), Sigma Nought (GEO/GEC) |

### 11.3 Current Repository Status vs Contest Needs

| Component | Current Status | Contest Need |
|-----------|----------------|--------------|
| Data source | **Done** — 221 Hawaii SLCs, 497 GB downloaded | Capella X-band via S3/STAC ✓ |
| ML approach | **Done** — self-supervised N2N + physics losses | Self-supervised ✓ |
| Model input | **Done** — complex ifg → denoised + uncertainty | ✓ |
| Metrics | **Done** — all 5 in `eval/compute_metrics.py` | ✓ |
| Training splits | **Done** — AOI-based splits in training script | ✓ |
| `src/insar_processing/pair_graph.py` | **Done** — 8,834 pairs, 24,171 triplets | ✓ |
| `src/insar_processing/geometry.py` | **Done** — B_perp from state-vector interpolation | ✓ |
| `src/insar_processing/sublook.py` | **Done** — FFT sub-look N2N splits (phase corr=0.001) | ✓ |
| `src/insar_processing/filters.py` | **Done** — Goldstein + adaptive Goldstein | ✓ |
| `scripts/preprocess_pairs.py` | **Done** — 162 pairs preprocessed | ✓ |
| `scripts/select_triplet_completing_pairs.py` | **Done** — fixed zero-triplet gap (+62 pairs) | ✓ |
| `src/models/film_unet.py` | **Done** — 7.96M params, smoke-tested | ✓ |
| `src/losses/physics_losses.py` | **Done** — N2N, NLL, closure, temporal, grad | ✓ |
| `experiments/enhanced/train_film_unet.py` | **Done** — full training script | ✓ |
| `eval/compute_metrics.py` | **Done** — M1=1.018 rad, M5=0.050 rad (Goldstein) | Partial (M2/M3/M4 need SNAPHU) |
| `scripts/unwrap_snaphu.py` | **Missing** ← BUILD NEXT | Required for M2/M3/M4 |
| Trained model checkpoint | **Missing** ← TRAIN NEXT | Required for DL comparison |
| `REPRODUCIBILITY.md` | Does not exist | Contest requirement (due last week) |

### 11.4 Current Next Steps (Mar 17–24)

1. [ ] `scripts/unwrap_snaphu.py` — SNAPHU unwrapping on 162 pairs → unlock M2/M3/M4 (#12)
2. [ ] Launch `experiments/enhanced/train_film_unet.py` on GPU — first training run (50 epochs)
3. [ ] Run `eval/compute_metrics.py` with FiLMUNet checkpoint → Goldstein vs FiLMUNet comparison table
4. [ ] Ablation studies: 5 model variants (N2N-only → +closure → +temporal → +uncertainty → +FiLM)
5. [ ] Zero-shot transfer to AOI_008 (LA) — re-run `preprocess_pairs.py` + `compute_metrics.py` (#18)
6. [ ] `REPRODUCIBILITY.md` — STAC URL, checksums, seeds, one-command pipeline (#23)
7. [ ] 4-page contest paper draft (#22)

---

*Document Version: 2.2*
*Last Updated: 2026-03-17*
*Project Status: ACTIVE — All model/loss/eval code done; 162 pairs preprocessed; Goldstein baseline M1=1.018 rad, M5=0.050 rad. NEXT: SNAPHU + FiLMUNet training. 20 days to submission.*
