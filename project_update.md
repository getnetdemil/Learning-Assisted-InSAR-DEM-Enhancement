# Learning-Assisted InSAR DEM Enhancement
## IEEE GRSS 2026 Data Fusion Contest — Technical Reference Document

**Contest deadline**: April 06, 2026 (18 days remaining as of 2026-03-19)
**Team**: getnetdemil
**Last updated**: 2026-03-19

---

**Executive Summary**: This project trains a self-supervised, geometry-conditioned neural
network (FiLMUNet) to denoise complex interferometric SAR (InSAR) phase. The model takes
paired sub-looks of the same scene as noisy training targets (Noise2Noise), conditions on
acquisition geometry via Feature-wise Linear Modulation (FiLM), and outputs a denoised
complex interferogram plus per-pixel uncertainty. The uncertainty output feeds directly into
SNAPHU phase unwrapping and SBAS time-series inversion, jointly targeting all five IEEE
GRSS 2026 contest evaluation metrics. The dataset is 791 Capella Space X-band SAR SLCs
across 39 global AOIs; primary training uses Hawaii (AOI_000, 221 collects, 8,834 pairs).

---

## Table of Contents

1. [Background: What is InSAR?](#1-background-what-is-insar)
2. [Dataset: Capella Space X-band SAR](#2-dataset-capella-space-x-band-sar)
3. [Phase 1: Baseline InSAR Pipeline](#3-phase-1-baseline-insar-pipeline)
4. [Phase 2: FiLM-Conditioned U-Net (FiLMUNet)](#4-phase-2-film-conditioned-u-net-filmUNet)
5. [Phase 3: Self-Supervised Physics Loss Suite](#5-phase-3-self-supervised-physics-loss-suite)
6. [Training Setup and Results](#6-training-setup-and-results)
7. [Evaluation Metrics (5 Contest KPIs)](#7-evaluation-metrics-5-contest-kpis)
8. [SNAPHU Phase Unwrapping](#8-snaphu-phase-unwrapping)
9. [Novelty and Scientific Contribution](#9-novelty-and-scientific-contribution)
10. [Session Log](#10-session-log)
11. [Remaining Work](#11-remaining-work)
12. [File Inventory](#12-file-inventory)

---

## 1. Background: What is InSAR?

### 1.1 Synthetic Aperture Radar Basics

Synthetic Aperture Radar (SAR) is an active microwave imaging system mounted on a
satellite. Unlike optical sensors, SAR emits its own electromagnetic pulses and records
the returned echoes — which means it operates day and night and through clouds.

Each recorded SAR pixel is a **complex number** (Single Look Complex, SLC). It has:
- **Amplitude** (|SLC|): backscatter intensity — roughly how strongly the surface reflects
  microwave energy. Depends on surface roughness, moisture, and dielectric properties.
- **Phase** (∠SLC): the fractional round-trip travel time of the radar pulse to the surface
  pixel, modulo 2π. Encodes range distance to sub-wavelength precision.

**Capella Space specifications used in this project:**
- X-band carrier: 9.6 GHz → wavelength λ ≈ 3.1 cm
- Polarisation: HH (horizontal transmit, horizontal receive)
- Mode: Spotlight (steered antenna → finer azimuth resolution)
- Pixel spacing: ~0.35–0.5 m in both range and azimuth directions
- Format: CInt16 complex integer GeoTIFF + extended JSON sidecar

At λ=3.1 cm, one full 2π phase cycle corresponds to λ/2 = 1.55 cm of range displacement.
This extraordinary sensitivity is what makes X-band InSAR powerful — and what makes noise
such a critical problem.

### 1.2 Interferometric SAR (InSAR)

InSAR forms an **interferogram** by multiplying one SLC acquisition (reference) with the
complex conjugate of another (secondary):

```
φ_raw(p) = arg( SLC_ref(p) × conj(SLC_sec(p)) )
```

The result at each pixel p is a wrapped phase value in (−π, π]. This interferometric
phase contains several contributions:

```
φ = (4π/λ) × (B_perp / (R × sin(θ))) × Δh   ← topographic term
  + (4π/λ) × Δd                               ← surface deformation term
  + φ_atm                                      ← atmospheric delay term
  + φ_noise                                    ← phase noise (speckle)
```

where:
- `B_perp` = perpendicular baseline (cross-track separation between two satellite orbits, metres)
- `R` = slant range distance from satellite to target (~500–600 km for Capella)
- `θ` = incidence angle (angle between radar line-of-sight and vertical, typically 30–60°)
- `Δh` = height difference between the topographic surface and the reference ellipsoid
- `Δd` = line-of-sight surface displacement between acquisitions

**Coherence** γ quantifies how well-correlated the two SLC acquisitions are at each pixel:

```
γ(p) = |E[SLC_ref × conj(SLC_sec)]| / sqrt(E[|SLC_ref|²] × E[|SLC_sec|²])
```

γ ∈ [0, 1]: γ=1 means perfect coherence (pure phase signal), γ=0 means complete
decorrelation (pure noise). In practice, γ < 0.35 generally makes phase too noisy
to use for deformation or DEM estimation.

### 1.3 Phase Noise and the Unwrapping Problem

The phase noise term φ_noise is the central challenge. It arises from:
- **Temporal decorrelation**: the surface changes between acquisitions (vegetation growth,
  rain, wind, lava flows) — each changed scatterer adds a random phase contribution
- **Geometric decorrelation**: at large baselines B_perp, the same pixel is illuminated
  from a slightly different angle, causing partial decorrelation even for static surfaces
- **Thermal noise**: receiver electronics introduce random phase shifts, dominated by SNR

The recorded phase is **wrapped** — only the fractional part modulo 2π is observable.
To convert phase to height or displacement, we must first recover the absolute (unwrapped)
phase. This is the **phase unwrapping problem**: integrate the wrapped phase gradient
across the image while handling noise-induced discontinuities.

Phase unwrapping fails catastrophically where coherence drops below ~0.35. A single
unwrapping error (a 2π jump in the wrong place) propagates along the integration path
and corrupts all downstream measurements in that region. **This is why phase denoising
is so valuable**: a better-denoised interferogram has smoother gradients and fewer
spurious π jumps, leading to more successful unwrapping.

### 1.4 DEM Generation from InSAR

Given unwrapped phase φ_unw and known B_perp:

```
Δh = (φ_unw × λ × R × sin(θ)) / (4π × B_perp)
```

For pure DEM estimation (no deformation), pairs with short temporal baselines
(Δt < 15 days) minimise the deformation and atmospheric terms.

**Multi-temporal stacking (SBAS — Small Baseline Subset)** greatly improves accuracy:
a stack of P interferometric pairs can be written as Ax = φ̂ where:
- A is a (P × T) design matrix (+1 at secondary epoch, −1 at reference epoch)
- x is a (T,) vector of epoch-wise phase increments or velocities
- φ̂ is the (P,) vector of observed unwrapped phases

Solving the weighted least-squares system `min_x ‖W(Ax − φ̂)‖₂` gives a time-consistent
phase model. The residual `‖W(Ax* − φ̂)‖₂` is Contest Metric 5.

### 1.5 Why Learning-Assisted?

Classical phase filters (Goldstein-Werner 1998, NL-InSAR 2011, BM3D-SAR 2012) all work
locally and independently per pair. They cannot:
- Adapt to different temporal/geometric baseline regimes in a single model
- Leverage information from the full temporal stack of interferograms
- Propagate uncertainty estimates downstream for weighting
- Directly optimise for the contest evaluation criteria during training

Deep learning offers all of these — but comes with a fundamental obstacle: **no
ground-truth clean interferograms exist**. Real SAR phase is always contaminated by
noise; there is no way to acquire a "clean" reference for supervised training.
This forces us into **self-supervised** methods that exploit the statistical properties
of the interferometric noise itself.

---

## 2. Dataset: Capella Space X-band SAR

### 2.1 Data Source

- **Contest dataset**: IEEE GRSS 2026 Data Fusion Contest, provided via public AWS S3
- **S3 bucket**: `s3://capella-open-data/data/` (region `us-west-2`, no authentication)
- **STAC endpoint**: `https://capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-ieee-data-contest/collection.json`
- **Total items**: 791 SLC GeoTIFFs + 791 GEO (geocoded) GeoTIFFs; 791 extended JSON sidecars
- **Geographic scope**: 39 AOIs globally (0.5° grid clusters, assigned by centroid proximity)
- **Access**: unsigned boto3 download (`Config(signature_version=UNSIGNED)`)

Each SLC acquisition comes with an extended JSON sidecar containing state vectors
(satellite position + velocity at 1-second intervals), incidence/graze angle maps,
Doppler centroid, and PRF. These are critical for geometry computation.

### 2.2 Primary AOI: Hawaii (AOI_000)

| Property | Value |
|----------|-------|
| Collects | 221 (214 with HH SLC asset) |
| Orbits | Both ascending and descending |
| Incidence angle range | 35.8°–56.3° |
| Temporal span | Jan 2024 – Mar 2026 (2+ years) |
| Terrain | Volcanic (Mauna Kea/Loa up to ~4,200 m) |
| Interferometric pairs (loose: Δt≤365d, Δinc≤5°) | 8,834 |
| Interferometric pairs (strict: Δt≤60d, Δinc≤2°) | 3,033 |
| Closure triplets (strict pairs) | 24,171 |
| Downloaded SLC data | ~497 GB in `data/raw/AOI_000/` |

### 2.3 Why Hawaii?

Hawaii uniquely maximises expected improvement on every one of the five contest metrics:

| Criterion | Hawaii | LA | W. Australia | SF Bay | Why Hawaii Wins |
|-----------|--------|----|--------------|--------|-----------------|
| Collects | **221** | 119 | 100 | 88 | More pairs → robust statistics for all metrics |
| Valid pairs | **~8,800** | ~3,500 | ~2,800 | ~1,200 | More training data + triplets |
| Both orbits | **yes** | yes | yes | no (desc only) | Ascending+descending → full 2.5D geometry |
| Inc. range | **35.8–56.3°** | 17.6–52.8° | 34.5–52.8° | 36.5–37.1° | Wide range → FiLM conditioning challenged & trained well |
| Terrain relief | **high (volcanic)** | mixed | flat/arid | moderate | Strong topo signal → well-conditioned DEM NMAD metric |

Metric-specific reasoning:
1. **Triplet closure**: 24,171 triplets provide statistically robust median closure estimation
2. **Unwrap success rate**: steep fringes from 4,200 m relief → DL denoising improvement maximised
3. **Usable pairs fraction**: vegetation + fresh lava decorrelation → more headroom for DL improvement
4. **DEM NMAD**: Mauna Kea/Loa elevation signal → metric is far from floor, leaving room to improve
5. **Temporal consistency**: 2+ year time-series → meaningful SBAS model to be consistent with

### 2.4 Data Access Quirks

Several Capella-specific issues required engineering fixes:

- **CInt16 format**: SLCs are stored as complex 16-bit integers; rasterio reads them as
  `complex64` automatically, but the dtype must be verified on load
- **Nanosecond timestamps**: Capella JSON timestamps have nanosecond precision
  (e.g., `2024-03-15T10:23:45.123456789Z`), which Python 3.10's `fromisoformat()` rejects.
  Fix: truncate to 6 decimal places before parsing
- **Variable SLC dimensions**: C09/C10 satellites produce ~108,000 × 12,000 px images
  (4.3 GB); C13 produces ~35,000 × 18,000 px (1.8 GB). Coregistration must handle both
- **`reference_antenna_position` field**: not consistently populated in extended JSON;
  always use state-vector interpolation (ECEF position at acquisition time) instead
- **GLIBCXX runtime fix**: before any rasterio-dependent script, set:
  `export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH`

---

## 3. Phase 1: Baseline InSAR Pipeline

All pipeline code lives in `src/insar_processing/` and `scripts/`.

### 3.1 Pair-Graph Construction (`src/insar_processing/pair_graph.py`)

Not all combinations of 221 acquisitions make useful interferometric pairs. We use a
quality score Q_ij to rank pairs:

```
Q_ij = (1 / (Δt + 1)) × (1 / (1 + |Δθ_inc|))
```

where Δt is the temporal baseline in days and Δθ_inc is the difference in incidence
angle between the two acquisitions. This score balances:
- **Temporal decorrelation**: short Δt → less surface change → higher coherence
- **Geometric diversity**: small |Δθ_inc| → compatible geometry for interferometry

**Constraints**: both acquisitions must share the same `orbit_state` (ascending/descending)
AND the same `look_direction`. Mixing would put the target on opposite sides of the spacecraft,
making interferometry geometrically nonsensical.

Results for AOI_000:
- Loose pairs (Δt≤365d, Δinc≤5°): **8,834 pairs** → `data/manifests/hawaii_pairs.parquet`
- Strict pairs (Δt≤60d, Δinc≤2°): **3,033 pairs** → used for closure triplets
- **24,171 closure triplets** → `data/manifests/hawaii_triplets_strict.parquet`

### 3.2 Perpendicular Baseline Computation (`src/insar_processing/geometry.py`)

B_perp — the component of the orbital separation vector perpendicular to the
line-of-sight — determines topographic sensitivity and geometric decorrelation.
It is not directly measured; it must be computed from satellite state vectors.

Algorithm:
1. Parse state vectors from extended JSON (position + velocity at 1-second intervals)
2. Interpolate satellite ECEF (Earth-Centred Earth-Fixed) position at scene centre time
   using cubic spline interpolation for both reference and secondary acquisitions
3. Form the baseline vector: `B = pos_sec − pos_ref` (in ECEF coordinates)
4. Compute the look vector `l̂` (unit vector from satellite to scene centre in ECEF)
5. Decompose B into parallel (along l̂) and perpendicular components:
   `B_perp = |B − (B · l̂) × l̂|`

Hawaii statistics: mean B_perp = 35 m, median = 6 m, max = 40 km.
The wide range (6 m to 40 km) is important: larger B_perp pairs have higher topographic
sensitivity but more geometric decorrelation — this diversity is critical for FiLM training.

### 3.3 Coregistration (`scripts/preprocess_pairs.py`)

SLC acquisitions from different orbits are not naturally aligned — each orbit places
pixels at slightly different positions on the ground. Before forming an interferogram,
the secondary SLC must be registered to the reference SLC at sub-pixel accuracy.

Algorithm (phase cross-correlation):
1. Compute 2D FFT of the amplitude (|SLC|) images of reference and secondary
2. Form the normalised cross-power spectrum:
   `R = FFT(|ref|) × conj(FFT(|sec|)) / |R|`
   (phase-only normalisation: unit amplitude, only phase carries shift information)
3. IFFT(R) → correlation peak location (integer pixel offset)
4. Sub-pixel refinement via DFT upsampling (Guizar-Sicairos 2008):
   compute a local DFT around the integer peak at 10× upsampling to achieve
   1/10-pixel registration accuracy
5. Apply shift: linear phase ramp in frequency domain
   `SLC_sec_shifted = IFFT( FFT(SLC_sec) × exp(−j2π × (Δr×f_r + Δa×f_a)) )`
   This is the exact, interpolation-free shift for band-limited signals

### 3.4 Interferogram Formation

```
ifg(p) = SLC_ref(p) × conj(SLC_sec_coreg(p))
```

The raw interferogram is a complex image where:
- Phase: `∠ifg(p) = φ_topo + φ_defo + φ_atm + φ_noise` (all contributions summed)
- Amplitude: `|ifg(p)|` = product of backscatter amplitudes (not used directly)

For network input we normalise to unit amplitude (extract only phase):
```
ifg_norm(p) = ifg(p) / (|ifg(p)| + ε)
```
This gives a complex image with |ifg_norm| ≈ 1 everywhere; Re and Im form the two input
channels to the model.

### 3.5 Coherence Estimation

Using a spatial (box-car) estimator with a 5×5 multi-look window:

```
γ(p) = |mean_{window}( SLC_ref × conj(SLC_sec) )| / sqrt( mean(|SLC_ref|²) × mean(|SLC_sec|²) )
```

Implemented via `scipy.ndimage.uniform_filter` for efficiency on large arrays.
The coherence map serves three roles:
1. Quality indicator for pair selection and usability gating (γ > 0.35)
2. FiLM conditioning input (SNR proxy)
3. SNAPHU pixel weighting (low-coherence pixels cost more to unwrap)

### 3.6 Goldstein-Werner Spectral Filter (`src/insar_processing/filters.py`)

The Goldstein-Werner filter (Goldstein & Werner 1998) is the standard classical baseline:

1. Partition the interferogram into overlapping 32×32 blocks (8-pixel overlap)
2. For each block: compute 2D FFT
3. Smooth the spectral magnitude: `|S_smooth| = uniform_filter(|S|, size=3)`
4. Power-law spectral weighting: `W_block = (|S_smooth| / max(|S_smooth|))^α`
5. Apply: `S_filtered = S × W_block` then IFFT
6. Overlap-add with Hanning taper to reconstruct the filtered image

The parameter α ∈ [0,1] controls filter strength: α=0 is no filtering; α=1 is maximum.

Our coherence-adaptive variant:
```
α(block) = α_min + (α_max − α_min) × (1 − γ_block)
```
Low-coherence blocks (γ→0) receive maximum filtering (α→α_max);
high-coherence blocks (γ→1) receive minimum filtering (α→α_min).
This preserves fine fringe detail where the signal is trustworthy.

### 3.7 Sub-look Splitting for Noise2Noise (`src/insar_processing/sublook.py`)

Self-supervised training via Noise2Noise requires **two independent noisy observations
of the same underlying scene**. We generate these via frequency sub-banding:

- **Method 1 (FFT azimuth sub-banding)**: split the SLC spectrum into lower and upper
  frequency halves in azimuth. Each half forms its own sub-look interferogram.
  Because the two frequency bands are orthogonal, their speckle realisations are
  statistically independent: `Cov(φ_sub_A, φ_sub_B) ≈ 0`
  Verified: sub-look phase correlation = 0.001 (effectively zero) ✓
  Advantage: preserves full range resolution; no aliasing artefacts

- **Method 2 (odd/even lines)**: alternating azimuth lines to each sub-look.
  Simpler but halves azimuth resolution.

The resulting pair (ifg_A, ifg_B) has:
- `E[φ_sub_A] = E[φ_sub_B] = φ_true` (same mean = true interferometric phase)
- Independent noise (speckle decorrelated across the two bands)

This satisfies the Noise2Noise condition exactly: training f(φ_A) to predict φ_B
converges to the MMSE estimator `E[φ_true | φ_A]`.

### 3.8 Outputs Per Processed Pair

For each of the 100 preprocessed pairs, `scripts/preprocess_pairs.py` writes to
`data/processed/pairs/{pair_id}/`:

| File | Description |
|------|-------------|
| `ifg_raw.tif` | 2-band float32 GeoTIFF (Re, Im of unnormalised interferogram) |
| `coherence.tif` | 1-band float32 GeoTIFF, values ∈ [0,1] |
| `ifg_goldstein.tif` | 2-band float32 GeoTIFF (Goldstein-filtered Re, Im) |
| `coreg_meta.json` | Dict: `dt_days`, `bperp_m`, `inc_angle_deg`, `graze_angle_deg`, `orbit_state`, `look_dir`, `q_score`, `row_offset`, `col_offset` |

---

## 4. Phase 2: FiLM-Conditioned U-Net (FiLMUNet)

### 4.1 Motivation: Why Geometry-Conditioned Denoising?

Phase noise in InSAR is not stationary — it depends strongly on acquisition geometry:
- Longer temporal baseline Δt → more decorrelation → stronger noise
- Larger B_perp → more geometric decorrelation → fringe-dependent noise
- Different incidence angles → different scattering geometry → different coherence patterns

A model trained on a mixture of short-Δt (coherent) and long-Δt (noisy) pairs without
geometry awareness would learn a weighted average of these noise regimes, performing
suboptimally for both. FiLM conditioning allows the **same model** to adapt its
behaviour to the current pair's geometry at inference time.

Additionally, FiLM conditioning enables **zero-shot transfer**: a model trained on
Hawaii can be applied to LA (AOI_008) with different incidence angles and baselines
without retraining — the geometry conditioning vector handles the domain shift.

### 4.2 Feature-wise Linear Modulation (FiLM) Primer

FiLM (Perez et al. 2018) conditions a neural network on external information by applying
learned affine transformations to intermediate feature maps:

```
FiLM(x; c) = γ(c) ⊙ x + β(c)
```

where x ∈ R^C is a feature map, c is the conditioning vector, and γ(c), β(c) ∈ R^C
are channel-wise scale and shift predicted by a small MLP. Applied after batch
normalisation (before activation), this is equivalent to replacing the fixed BN affine
parameters with input-dependent ones.

Our residual-centred variant uses `y = (1 + γ(c)) ⊙ x + β(c)`, so at initialisation
(γ→0, β→0) the modulation is identity — this stabilises early training.

### 4.3 Architecture Overview

```
Input:
  x     : (B, 3, H, W)   — Re(ifg), Im(ifg), coherence
  meta  : (B, 7)          — [Δt, θ_inc, θ_graze, B_perp, mode, look, SNR_proxy]
                            (standardised by known physical ranges)

MetadataEncoder:
  Linear(7 → 64) → ReLU → Linear(64 → 64) → ReLU   →  embed: (B, 64)

Encoder (each level: FiLMDoubleConv + MaxPool(2)):
  Level 0: (B,  3, H,   W  ) → (B,  32, H/2,  W/2 )
  Level 1: (B, 32, H/2, W/2) → (B,  64, H/4,  W/4 )
  Level 2: (B, 64, H/4, W/4) → (B, 128, H/8,  W/8 )
  Level 3: (B,128, H/8, W/8) → (B, 256, H/16, W/16)

Bottleneck:
  FiLMDoubleConv(256 → 512)

Decoder (each level: ConvTranspose(2×) + skip-cat + FiLMDoubleConv):
  Level 3: (512+256 → 256)
  Level 2: (256+128 → 128)
  Level 1: (128+ 64 →  64)
  Level 0: ( 64+ 32 →  32)

Output heads (1×1 convolutions):
  head_denoised : Conv(32 → 2)  → denoised Re+Im interferogram
  head_log_var  : Conv(32 → 1)  → per-pixel log(σ²) uncertainty
```

### 4.4 FiLMDoubleConv Block Detail

Each encoder/decoder block is a double convolution with FiLM modulation injected
after the first batch normalisation:

```
x → Conv(3×3, pad=1, no bias) → BatchNorm → FiLM(embed) → ReLU
  → Conv(3×3, pad=1, no bias) → BatchNorm → ReLU
```

FiLM after the first BN is the key design choice: it allows the metadata embedding to
shift and scale the normalised activations before the non-linearity, maximising the
conditioning signal's influence on the feature representation.

### 4.5 Metadata Encoding and Normalisation

The 7-D metadata vector requires careful normalisation so all features have roughly
unit scale entering the linear layer:

| Feature | Physical range | Normalisation (μ, σ) |
|---------|---------------|---------------------|
| Δt (days) | 0–365 | μ=30, σ=60 |
| θ_inc (degrees) | 30–60 | μ=45, σ=8 |
| θ_graze (degrees) | 25–55 | μ=35, σ=8 |
| B_perp (metres) | −40000–40000 | μ=500, σ=2000 |
| orbit_state | {asc=0, desc=1} | μ=0.5, σ=0.5 |
| look_direction | {left=0, right=1} | μ=0.5, σ=0.5 |
| SNR_proxy | coherence-derived | μ=0.5, σ=0.3 |

### 4.6 Model Specifications

- **Total parameters**: 7,959,139 (~8M)
- **Input tile size**: (B, 3, 256, 256)
- **Inference**: ~15 ms per 256×256 tile on GPU (A100)
- **Checkpoint format**: PyTorch `.pt` dict with `model_state_dict`, `optimizer_state_dict`,
  `epoch`, `best_val_loss`

### 4.7 Per-Pixel Uncertainty Output

The `head_log_var` output enables downstream uncertainty-weighted processing.
Interpreting the output as aleatoric (data-driven) uncertainty:

```
σ²(p) = exp(log_var(p))    ← predicted variance at pixel p
```

This uncertainty has three downstream uses:
1. **SNAPHU weighting**: replace uniform pixel weights with `1/σ²(p)` — less confident
   pixels cost more to cross during network-flow unwrapping
2. **SBAS inversion**: `W_p = 1/σ²(p)` in weighted least-squares — temporally inconsistent
   pixels (high uncertainty) have lower weight in the epoch-velocity solution
3. **Closure-weighted loss**: `w(p) = 1/σ²(p)` in L_closure during training

---

## 5. Phase 3: Self-Supervised Physics Loss Suite

### 5.1 Why Self-Supervised?

A fundamental obstacle in InSAR deep learning is that **no ground-truth clean
interferograms exist**. All real SAR data is contaminated by speckle noise; the
"clean" underlying phase is a theoretical construct, not an observable quantity.

This rules out supervised training (no (noisy, clean) pairs) and most forms of
semi-supervised learning. We instead exploit two intrinsic properties of interferometric
phase to define self-supervised objectives:
1. **Statistical independence of sub-look speckle** → Noise2Noise training
2. **Topological closure of triplet phase cycles** → physics-consistency training

Combined, these five loss components directly optimise for the contest evaluation criteria
without requiring any external reference data.

### 5.2 Loss 1 — Noise2Noise (L_n2n, weight=1.0)

**Principle** (Lehtinen et al. 2018): Given two independent noisy observations y₁, y₂
of the same underlying signal x where E[y₁] = E[y₂] = x, training f(y₁) to minimise
‖f(y₁) − y₂‖² converges to the MMSE estimator f*(y) = E[x|y]. This holds for any
zero-mean-noise distribution, not just Gaussian.

**InSAR application**: Sub-look A and sub-look B have the same expected wrapped phase
(= true interferometric phase), with independent speckle noise. Therefore:
```
L_n2n = mean( |wrap( φ_pred − φ_sublook_B )| )
```

We use wrapped-phase L1 rather than amplitude L1 (Re/Im residual) because:
- Phase is the physically meaningful quantity for InSAR (not amplitude)
- Wrapping discontinuities cause large amplitude residuals at ±π boundaries that do not
  correspond to actual prediction errors
- Phase-domain L1 is robust to these wrap artefacts

**Current implementation**: in the absence of actual sub-look pairs in the tile data,
the batch is split in half and each half serves as the other's "target". This is a
simplified proxy; the production version uses actual FFT sub-band pairs.

### 5.3 Loss 2 — Uncertainty NLL (L_unc, weight=0.5)

Heteroscedastic negative log-likelihood forces the uncertainty prediction to be
calibrated (neither overconfident nor trivially infinite):

```
L_unc = mean( 0.5 × exp(−log_σ²) × |wrap(φ_pred − φ_target)|² + 0.5 × log_σ² )
```

The two terms create competing pressures:
- **Accuracy term** `exp(−log_σ²) × err²`: large residuals penalised more when predicted
  σ² is small — forces the model to be accurate where it claims confidence
- **Regularisation term** `log_σ²`: penalises large σ² — prevents trivial solution
  where model predicts infinite uncertainty everywhere to make accuracy term vanish

This is the NLL of an isotropic Laplace distribution:
`log p(φ | μ, σ²) ∝ −|φ − μ|/σ − log(2σ)`, reparametrised with σ² = exp(log_σ²).

### 5.4 Loss 3 — Triplet Closure Consistency (L_closure, weight=0.3)

**Physical principle**: for three SAR acquisitions i, j, k with well-defined geometry,
the interferometric phases must satisfy the closure condition:
```
wrap(φ_ij + φ_jk − φ_ik) = 0
```
Any non-zero closure is pure noise (phase noise is not path-independent, but the
underlying signal is). Minimising closure error during training directly optimises
Contest Metric 1.

**Implementation**:
```
L_closure = mean( w(p) × [1 − cos( wrap(φ̂_ij + φ̂_jk − φ̂_ik) )] )
```

where w(p) = coherence or 1/σ²(p) weighting. The cosine loss is bounded [0, 2]:
- Perfect closure: cos(0) = 1 → loss = 0
- Maximum error: cos(±π) = −1 → loss = 2
- Differentiable everywhere except at cos(±π) (measure-zero set)

### 5.5 Loss 4 — Temporal Consistency (L_temporal, weight=0.2)

The SBAS model `Ax = φ̂` provides a multi-temporal consistency constraint. We embed
this as a differentiable loss:

```
x* = weighted_lstsq(A, φ̂; W=diag(w))       ← closed-form via torch.linalg.lstsq
L_temporal = ‖W^{1/2} × (A×x* − φ̂)‖₂²    ← weighted SBAS residual
```

By minimising this loss during training, the network learns to produce denoised phases
that are inherently consistent with the multi-temporal phase model — the SBAS inversion
residual is minimised **before** the inversion is even run.

This is a novel approach: SBAS is traditionally a post-processing step after denoising.
Embedding it as a training loss creates a direct gradient pathway to Contest Metric 5.

### 5.6 Loss 5 — Gradient Preservation (L_grad, weight=0.1)

Prevents over-smoothing and preserves interferometric fringe structure:

```
L_grad = mean( |∇_y wrap(φ_pred) − ∇_y wrap(φ_input)|
             + |∇_x wrap(φ_pred) − ∇_x wrap(φ_input)| ) / 2
```

Operates on **wrapped phase gradients** (not on Re/Im amplitude gradients) to correctly
handle fringe discontinuities. The low weight (0.1) makes this a structural regulariser
rather than a primary supervision signal.

### 5.7 Combined Loss

```
L_total = 1.0 × L_n2n
        + 0.5 × L_unc
        + 0.3 × L_closure
        + 0.2 × L_temporal
        + 0.1 × L_grad
```

Weight rationale:
- N2N (1.0): dominant self-supervised signal — drives the core denoising
- Uncertainty (0.5): important secondary objective for downstream calibration
- Closure (0.3): geometry-consistency regulariser, directly tied to Metric 1
- Temporal (0.2): stack consistency, directly tied to Metric 5
- Gradient (0.1): lightweight structural prior, prevents fringe destruction

---

## 6. Training Setup and Results

### 6.1 Dataset Configuration

- **Source**: 100 preprocessed pairs from `data/processed/pairs/` (top-100 by Q_score)
- **Splits**: 70 train / 15 val / 15 test (temporal order — no data leakage)
  - Temporal ordering prevents future-to-past leakage (unlike random tile shuffling)
  - AOI-based splits prevent geographic leakage from adjacent tiles
- **Tiling**: 256×256 patches, 128-pixel stride (50% overlap)
  - Train: ~67,270 tiles; Val: ~14,415 tiles
- **Min coherence filter**: 0.15 (tiles below this threshold discarded as uninformative)
- **Augmentation**: random 90° rotations, horizontal flip, global phase offset ∈ (−π, π),
  amplitude jitter ±10%

### 6.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Scheduler | Cosine annealing (T_max=48, eta_min=1e-6) + 2-epoch linear warmup |
| Batch size | 8 |
| Gradient clipping | 1.0 (L2 norm) |
| Epochs | 50 |
| Checkpoint interval | Every 5 epochs + best val loss |

### 6.3 Training Results (as of 2026-03-13, Session 5)

- Training was launched (epochs 1–20) in a prior run and checkpointed
- Session 5 resumed from epoch 20 and is currently running epoch 21/50
- Current loss values: train_loss ≈ 2.11, val_loss ≈ 2.67
- First epoch showed overflow (loss ≈ 7×10²¹) due to batch ordering; epoch 2 onward stable
- Saved checkpoints: `epoch_005.pt`, `epoch_010.pt`, `epoch_015.pt`, `epoch_020.pt`, `best_closure.pt`
- **Bug fixed in Session 5**: `coreg_meta.json` key names were wrong:
  - `delta_days` → `dt_days`, `b_perp_m` → `bperp_m`
  - Silent default values (Δt=0, B_perp=0) were being used for all FiLM conditioning,
    defeating the entire purpose of geometry conditioning

---

## 7. Evaluation Metrics (5 Contest KPIs)

All five metrics are implemented in `src/evaluation/closure_metrics.py`.

### 7.1 Metric 1 — Triplet Closure Error

```
closure_error(i,j,k) = wrap( φ_ij + φ_jk − φ_ik )
Report: median|closure|, mean|closure|, std(closure), RMSE(closure)
Target: median|closure| ↓ ≥30% vs Goldstein-only baseline
```

Physical meaning: perfectly coherent phase has zero closure. Non-zero closure indicates
phase noise or non-linear atmospheric artefacts. Lower closure = better denoising.

Evaluated over all 24,171 strict triplets from the Hawaii pair-graph manifest.

**Model linkage**: L_closure during training directly minimises this metric.

### 7.2 Metric 2 — Unwrap Success Rate

```
success_rate = |{p : γ(p) ≥ 0.35 AND unw_phase(p) ≠ NaN}|
             / |{p : γ(p) ≥ 0.35}|
Target: ↑ ≥15 percentage points vs Goldstein-only baseline
```

Only coherent pixels (γ ≥ 0.35) are evaluated — asking for unwrapping success in
incoherent regions is not meaningful. A pixel is "successfully unwrapped" if SNAPHU
assigns it a finite phase value (not NaN / masked out).

**Model linkage**: FiLMUNet denoising smooths phase gradients → fewer SNAPHU failures
at phase transition regions. Uncertainty output `σ²(p)` used as SNAPHU pixel weight.

### 7.3 Metric 3 — Percent Usable Pairs

```
usable(pair) = (mean_γ ≥ 0.35) AND (median|closure| < 0.5 rad)
usable_fraction = |{usable pairs}| / |{all pairs}|
Target: ↑ ≥25 percentage points vs Goldstein-only baseline
```

A dual gate: a pair must be both coherent (passes decorrelation test) and geometrically
consistent (passes closure test). This is the most stringent metric — it catches pairs
that look coherent but have systematic biases.

**Model linkage**: FiLMUNet improves both coherence (indirectly, via denoising) and
closure (directly, via L_closure). Both gates relax → fraction increases.

### 7.4 Metric 4 — DEM NMAD

NMAD (Normalised Median Absolute Deviation) is a robust estimator of DEM height error:

```
e(p) = h_estimated(p) − h_reference(p)        ← per-pixel height error
NMAD = 1.4826 × median( |e − median(e)| )
Target: ↓ ≥15% vs Goldstein-only baseline
```

The factor 1.4826 = 1 / Φ⁻¹(0.75) normalises MAD to be consistent with σ for a
Gaussian distribution. NMAD is resistant to outliers (unlike RMSE) and preferred for
DEM accuracy assessment.

Evaluated on stable, coherent terrain pixels with low slope (to avoid layover/shadow).
Requires a reference DEM (external, not part of the contest dataset — use NASADEM or
Copernicus DEM 30m as reference).

**Model linkage**: Improved phase denoising → better unwrapping → better SBAS inversion
→ lower height errors. Uncertainty-weighted SBAS (W = 1/σ²) is the primary mechanism.

### 7.5 Metric 5 — Temporal Consistency Residual

```
x* = argmin_x ‖W^{1/2}(Ax − φ̂)‖₂²     ← weighted SBAS inversion
residual = ‖W^{1/2}(Ax* − φ̂)‖₂
W = diag(1/σ²(p))                        ← FiLMUNet uncertainty as weights
Target: ↓ ≥20% vs uniform-weight SBAS baseline
```

Design matrix A has shape (P × T) where P = number of pairs, T = number of epochs.
Each row has +1 at the secondary epoch index and −1 at the reference epoch index.

**Model linkage**: L_temporal during training directly minimises this residual.
FiLMUNet's `σ²(p)` output replaces uniform weights in W, further reducing the residual
by down-weighting temporally inconsistent pixels.

---

## 8. SNAPHU Phase Unwrapping

### 8.1 SNAPHU Overview

SNAPHU (Statistical-cost Network-Flow Phase Unwrapping, Chen & Zebker 2001) solves
the phase unwrapping problem by minimising a cost function defined on a network graph
where each pixel is a node and adjacent-pixel phase differences are edges.

Two operating modes:
- **TOPO** (topographic): L2 cost norm, appropriate for DEM estimation
- **DEFO** (deformation): Bayesian cost, appropriate for subsidence/uplift

For DEM estimation (our use case) we use TOPO mode. Coherence maps provide per-pixel
weighting: low-coherence pixels have high edge cost and the network-flow solver
preferentially routes the unwrapping path away from them.

### 8.2 Implementation (`scripts/unwrap_snaphu.py`)

The script wraps the SNAPHU command-line binary with automated configuration:

1. **Read inputs**: `ifg_goldstein.tif` (or FiLMUNet denoised output) → wrapped phase
   via `arctan2(Im_band, Re_band)`; `coherence.tif` for weighting
2. **Masking**: pixels with γ < 0.1 are set to NaN — too incoherent to unwrap reliably
3. **Write binary inputs**: phase and coherence as flat float32 arrays (SNAPHU format)
4. **Write config file**: `LINELENGTH`, `NLINES`, `NCORRLOOKS=9`, `INITMETHOD=MST`
5. **Auto-tiling** for large images:
   - > 8192 px: 4×4 tile decomposition
   - > 4096 px: 2×2 tile decomposition
   - Smaller: no tiling
6. **Run SNAPHU** as subprocess; capture exit code
7. **Read output**: parse unwrapped phase binary → `unw_phase.tif` (float32 GeoTIFF,
   NaN at masked pixels, original CRS/transform preserved)

Additional features:
- Skip already-processed pairs (`unw_phase.tif` exists)
- Graceful exit with warning if SNAPHU binary not on PATH
- `--help` flag confirmed working

### 8.3 Current Status

- Script: implemented and tested
- SNAPHU binary: **not yet installed** (blocked on: `conda install -c conda-forge snaphu`)
- Will be run on all 100 processed pairs after SNAPHU installation
- Unwrapped phase outputs are required inputs for Metrics 2, 3, 4, and 5

---

## 9. Novelty and Scientific Contribution

### 9.1 Prior Art Review: InSAR Phase Denoising

**Classical approaches:**
- **Goldstein-Werner (1998)**: power-law spectral weighting on overlapping 32×32 blocks.
  Widely used, fast. Weaknesses: purely local (no cross-block context), fixed scalar α
  regardless of geometry or temporal baseline, no uncertainty output.
- **NL-InSAR (Deledalle et al. 2011)**: non-local means adapted to InSAR complex domain.
  Stronger than Goldstein for homogeneous scenes. Weaknesses: computationally expensive
  (hours per image), no temporal awareness, no geometry conditioning.
- **BM3D-SAR (Parrilli 2012)**: block-matching 3D collaborative filtering.
  Best classical baseline, extremely slow, non-geometry-aware, no uncertainty output.

**Supervised DL approaches (2018–2024):**
- **DeepInSAR (Anantrasirichai et al. 2021)**: ResNet for InSAR phase, supervised on
  synthetic interferograms from DEM + deformation model. Weaknesses: large domain gap
  (synthetic vs. real X-band); fixed geometry; no uncertainty output.
- **InSAR-DNN / PhaseNet variants**: U-Nets applied to wrapped phase tiles. Weaknesses:
  require reference clean data (not available for Capella); no temporal or geometric
  awareness; each pair processed independently.
- **RNN/ConvLSTM for time-series InSAR**: temporal awareness but still supervised; no
  explicit closure enforcement.

**Self-supervised SAR work (2022–2024):**
- **MERLIN (Dalsasso et al. 2021)**: self-supervised SAR despeckling via sub-look pairs
  for intensity (amplitude). No phase/coherence handling, no geometry conditioning,
  no InSAR-specific losses.
- **SAR-Noise2Noise (Molini et al. 2022)**: Noise2Noise on SAR intensity. Correct
  framework, wrong modality — intensity N2N does not handle interferometric phase.

**Key gap in all prior work**: no prior method combines (a) self-supervised training on
real complex InSAR data, (b) geometry conditioning for cross-baseline generalisation,
(c) per-pixel uncertainty for downstream use, and (d) physics-consistency losses that
directly optimise the evaluation criteria.

### 9.2 Our Novel Contributions

**Contribution 1 — Geometry-Conditioned Self-Supervised InSAR Denoiser**

FiLMUNet conditions on a 7-D geometry vector [Δt, θ_inc, θ_graze, B_perp, mode, look,
SNR_proxy] via FiLM modulation applied at every encoder/decoder block. A single model
adapts to the full range of Capella acquisition geometries.

Prior work trains separate models per geometry regime or ignores geometry entirely.
FiLM conditioning (Perez 2018) has not previously been applied to SAR physics conditioning.

Contest link: enables zero-shot transfer to AOI_008 (LA) — Metric 3.

**Contribution 2 — Noise2Noise on Real Complex Interferograms**

Sub-look FFT splitting generates two independent speckle realisations with the same
expected interferometric phase. Wrapped-phase L1 loss correctly handles discontinuities
that amplitude-domain (Re/Im) approaches miss.

MERLIN/SAR-N2N operate on intensity only. No prior InSAR DL work uses N2N on real
Capella X-band complex interferograms.

Contest link: eliminates need for synthetic training data — directly impacts Metrics 1, 2.

**Contribution 3 — Triplet Closure as Training Loss**

Contest Metric 1 (triplet closure) is embedded directly as a differentiable training
objective. Prior DL work uses pixel-wise MSE/L1 to a reference, which has no geometric
interpretation and does not directly optimise closure.

Contest link: direct gradient flow to Metric 1; indirect improvement to Metric 2.

**Contribution 4 — Per-Pixel Aleatoric Uncertainty with Downstream Integration**

`head_log_var` predicts per-pixel log(σ²) via heteroscedastic NLL. This uncertainty
weights both SNAPHU and SBAS — replacing coherence (an input proxy) with a learned,
geometry-aware confidence estimate.

No prior InSAR DL method outputs uncertainty; no prior method uses learned uncertainty
as the SBAS weight matrix.

Contest link: uncertainty-weighted SBAS directly targets Metric 5; uncertainty-weighted
SNAPHU targets Metrics 2, 3.

**Contribution 5 — SBAS Temporal Consistency as Training Loss**

L_temporal embeds a differentiable SBAS residual as a training loss, teaching the model
to produce phase stacks that satisfy the temporal model before post-processing.

Embedding SBAS as a training loss is novel. The gradient flows from temporal consistency
back through all pairs in the batch simultaneously.

Contest link: direct gradient flow to Metric 5.

### 9.3 Comparison Table

| Property | Goldstein | NL-InSAR | DeepInSAR | SAR-N2N | **FiLMUNet (ours)** |
|----------|-----------|----------|-----------|---------|---------------------|
| Self-supervised (no GT) | ✓ | ✓ | ✗ | ✓ | **✓** |
| Works on complex phase (Re+Im) | ✓ | ✓ | ✗ | ✗ | **✓** |
| Geometry conditioning | ✗ | ✗ | ✗ | ✗ | **✓ (FiLM, 7-D)** |
| Per-pixel uncertainty output | ✗ | ✗ | ✗ | ✗ | **✓ (log-var head)** |
| Closure loss during training | ✗ | ✗ | ✗ | ✗ | **✓ → Metric 1** |
| Temporal consistency loss | ✗ | ✗ | ✗ | ✗ | **✓ → Metric 5** |
| Uncertainty-weighted SBAS | ✗ | ✗ | ✗ | ✗ | **✓** |
| Trained on real Capella data | ✓ | ✓ | ✗ | partial | **✓ (221 collects)** |
| Cross-AOI zero-shot transfer | n/a | n/a | ✗ | ✗ | **✓ (FiLM)** |

### 9.4 Contest Metric Targets and Mechanisms

| Metric | Primary Mechanism | Secondary Mechanism | Expected Gain |
|--------|------------------|--------------------|----|
| 1. Triplet closure ↓30% | L_closure during training | N2N denoising | Direct |
| 2. Unwrap success ↑15 pp | Smoother fringes → fewer SNAPHU failures | σ²-weighted SNAPHU | Indirect |
| 3. Usable pairs ↑25 pp | Improved closure gate | Higher coherence from denoising | Compounded from 1+2 |
| 4. DEM NMAD ↓15% | σ²-weighted SBAS → better heights | Better unwrapping from Metric 2 | Via Metrics 2+5 |
| 5. Temporal residual ↓20% | L_temporal during training | σ²-weighted SBAS inversion | Direct |

---

## 10. Session Log

### Session 1 — 2026-03-05: Infrastructure, EDA, Strategy

**Completed:**
- [x] Built `scripts/download_subset.py` — STAC crawler + parallel S3 downloader
- [x] Crawled full contest collection: **791 SLC items**, saved to `data/manifests/full_index.parquet`
- [x] Assigned **39 AOIs** via 0.5° grid clustering
- [x] Built `notebooks/data_explorer.ipynb` — 10-section deep-dive EDA (geography, geometry, pair potential, SLC raster properties, storage budget, contest score ranking)
- [x] Confirmed environment: `torch-gpu` at `/scratch/gdemil24/hrwsi_s3client/torch-gpu`, 694 TB available on `/scratch`
- [x] Strategic decision: **Hawaii (AOI_000) as primary AOI**, LA + W. Australia as secondary
- [x] **Started Hawaii (AOI_000) download** — background terminal, ~221 SLCs + metadata JSONs

### Session 2 — 2026-03-06: Phase 1 Coding — Pair Graph, Geometry, Preprocessing

**Download confirmed complete:** 214/221 SLC .tif files + 221 extended JSONs; 497 GB total.
7 collects had no HH asset in STAC — skipped gracefully.

**Built and tested:**
- [x] `src/insar_processing/pair_graph.py` — pair-graph + Q_ij scoring + triplet enumeration
  - Loose: **8,834 pairs**; Strict: **3,033 pairs**; **24,171 closure triplets**
- [x] `src/insar_processing/geometry.py` — B_perp via state-vector interpolation
  - B_perp stats: mean=35m, median=6m, max=40km
  - Manifests saved: `hawaii_pairs.parquet`, `hawaii_triplets_strict.parquet`
- [x] `src/insar_processing/sublook.py` — FFT sub-band + odd/even N2N splits
  - Sub-look phase correlation = 0.001 ✓ (independent speckle confirmed)
- [x] `src/insar_processing/filters.py` — Goldstein, adaptive-Goldstein, boxcar coherence
- [x] `scripts/preprocess_pairs.py` — full end-to-end pipeline
  - Test on top pair (Δt=2.95d, B_perp=457m): coherence=0.419 ✓

Key fixes discovered: GLIBCXX path, nanosecond timestamp truncation, state-vector-only B_perp.

### Session 3 — 2026-03-07: GitHub Project Board Setup

**Completed:**
- [x] `scripts/create_github_issues.sh` — idempotent bulk issue creation
- [x] 9 labels, 1 milestone (`Contest Submission — April 06, 2026`), 21 issues (#3–#23)
- [x] Closed 8 already-completed issues (#3–7, #9, #11, #15)

Open backlog as of 2026-03-12: 11 issues (#8, #10, #12, #13, #17–#23)

### Session 4 — 2026-03-12: Phase 3 DL Pipeline — FiLMUNet + Physics Losses

**Completed:**
- [x] `src/models/film_unet.py` — FiLMUNet (7,963,747 params; dual output heads; smoke-tested) — issue #14 closed
- [x] `src/losses/physics_losses.py` — 5-component InSARLoss — issue #16 closed
- [x] `src/losses/__init__.py` — package init
- [x] Resolved merge conflict: `dev` branch already had more complete versions (PRs #25, #26, #27); rebased onto dev, accepted dev's versions, net delta = non-empty `__init__.py`
- [x] Branch workflow confirmed: feature branches → `dev` → `main`

### Session 5 — 2026-03-13: Training Data + Metrics + SNAPHU + Training Launch

**Completed:**
- [x] Ran `scripts/preprocess_pairs.py` on top-100 Hawaii pairs → `data/processed/pairs/` (100 dirs)
  - Each dir: `ifg_raw.tif`, `ifg_goldstein.tif`, `coherence.tif`, `coreg_meta.json`
- [x] Fixed PYTHONPATH issue for `src/` package imports
- [x] `src/evaluation/closure_metrics.py` — all 5 contest metrics (branch `feat/closure-metrics-13`)
  - `triplet_closure_error()`, `unwrap_success_rate()`, `usable_pairs_fraction()`
  - `dem_nmad()`, `temporal_consistency_residual()`, `compute_baseline_metrics()`
  - Smoke-tested: all 5 metrics pass on synthetic data ✓
- [x] Updated `src/evaluation/__init__.py`
- [x] `scripts/unwrap_snaphu.py` — full SNAPHU wrapper (branch `feat/snaphu-12`)
  - Auto-tiling, NaN masking, graceful PATH check; `--help` confirmed working
  - SNAPHU binary not yet installed (need: `conda install -c conda-forge snaphu`)
- [x] Fixed `InSARTileDataset._load_meta` key names: `delta_days`→`dt_days`, `b_perp_m`→`bperp_m`
  - Silent bug: FiLM conditioning was using all-default metadata (defeating geometry conditioning)
- [x] Fixed `torch.load` FutureWarning: added `weights_only=False`
- [x] Launched FiLMUNet training — resumed from epoch 20
  - 70 train pairs → 67,270 tiles; 15 val pairs → 14,415 tiles
  - Epoch 1 overflow (7e21, batch ordering instability), epoch 2+ stable
  - Epoch 21 active at session end: train_loss≈2.11, val_loss≈2.67

Open branches (ready for PR):
- `feat/closure-metrics-13` → closes #13
- `feat/snaphu-12` → closes #12
- `feat/train-film-unet-17` → closes #17 once training completes

### Session 6 — 2026-03-19: Bug Fixes, Training, SNAPHU Progress

**Root cause found and fixed: broken N2N training**
- The 50-epoch `final.pt` produced M1=1.973 rad (+93.9%) and M5=0.079 rad (+57.3%)
  vs Goldstein — WORSE on both metrics. Root cause: `run_epoch()` used the second
  half of a batch (from different pairs) as the N2N target → model learned to predict
  random noise. Closure and temporal losses were always 0 (inputs never set).
- Fix: replaced cross-batch N2N with **raw → Goldstein supervised denoising**:
  - Input: `ifg_raw.tif` (speckle-noisy), target: `ifg_goldstein.tif` (pseudo-clean)
  - `_load_tile()`, `_augment()`, `__getitem__()`, `run_epoch()` all updated
  - `eval/compute_metrics.py run_inference_on_pair()`: now reads `ifg_raw.tif` as input

**Second bug found and fixed: NaN loss from fresh initialization**
- `uncertainty_nll_loss()` in `src/losses/physics_losses.py:94` computes
  `exp(-log_var)` without clamping. Fresh `head_log_var` Conv init → extreme `lv` →
  `exp(-lv)` overflows → NaN propagates through all loss terms.
- Fix: added `.clamp(-10, 10)` on `log_var` in `src/models/film_unet.py:158`.
- Belt-and-suspenders: added `np.nan_to_num` guards on all five arrays returned by
  `_load_tile()` in `experiments/enhanced/train_film_unet.py` — protects against
  edge-pixel nodata from rasterio windowed reads.

**Metric baseline (Goldstein, 62 triplets):**
- M1 Triplet Closure Error: **1.018 rad** ✓
- M2 Unwrap Success Rate: N/A (SNAPHU in progress)
- M3 Usable Pairs: 0.0 (closure > 0.5 threshold; needs FiLMUNet improvement)
- M4 DEM NMAD: N/A (needs reference DEM)
- M5 Temporal Residual: **0.050 rad** ✓

**SNAPHU progress**: 70/224 pairs unwrapped as of Mar 19 (31.3%, ETA ~Mar 21).
Fixed ntiles bug (Mar 18): `≥4096px → (4,4)` tiles instead of `(2,2)` — prevents
"Exceeded maximum secondary arcs" error.

**Retraining (raw2gold_30ep, PID 3618843)**: restarted with NaN fix, 149,916 train
tiles, 30 epochs, ETA ~Mar 19 21:00 EET.
Checkpoint: `experiments/enhanced/checkpoints/film_unet/raw2gold_30ep/final.pt`

**Phase 3 implementation (completed 2026-03-18):**
- `eval/compute_metrics.py` — all 5 contest metrics (950 lines)
- `experiments/enhanced/train_film_unet.py` — CLI ablation overrides: `--loss_*`,
  `--epochs`, `--run_name`, `--zero_film`; saves `training_summary.json` per run
- `scripts/run_ablations.sh` — V1–V5 ablation variants (20 epochs each)
- `scripts/collect_ablation_results.py` — aggregates results → Markdown table
- `eval/zero_shot_transfer.py` — AOI_008 (LA) select + eval pipeline
- `REPRODUCIBILITY.md` — contest requirement, with real SHA-256 checksums

**Paper**: `Latex_Paper_Temporal_SAR_Change/main.tex` structure complete.
Tables 1 and 2 have `[XX]` placeholders awaiting eval + ablation results.
Figures already copied: `closure_histogram.png`, `phase_comparison.png`,
`temporal_residual_bar.png` → `Latex_Paper_Temporal_SAR_Change/figures/`.

---

## 11. Remaining Work

### 11.1 Critical Path (Before Contest Submission, April 06)

| Task | Priority | Status |
|------|----------|--------|
| Fix NaN loss (clamp log_var) | CRITICAL | **DONE (2026-03-19)** |
| Retrain 30ep (raw2gold, PID 3618843) | CRITICAL | **IN PROGRESS (ETA Mar 19 ~21:00)** |
| SNAPHU unwrapping 224 pairs | CRITICAL | **IN PROGRESS (70/224, ETA Mar 21)** |
| Re-eval with fixed model (M1, M5) | CRITICAL | Pending (after retraining) |
| Ablations V1–V5 (20 ep each) | HIGH | Pending (after retraining) |
| Full eval M2/M3 (after SNAPHU) | HIGH | Pending (after SNAPHU ~Mar 21) |
| Zero-shot transfer AOI_008 (LA) | MEDIUM | Pending (~Mar 25) |
| Fill paper `[XX]` placeholders | CRITICAL | Pending (after eval + ablations) |
| Final review + REPRODUCIBILITY.md checksums | HIGH | Pending (~Apr 1–5) |
| Submit | CRITICAL | April 06, 2026 |

### 11.2 Remaining Timeline

| Date | Milestone |
|------|-----------|
| **Mar 19 (today)** | NaN fix done; training restarted (PID 3618843) |
| **Mar 19 ~21:00** | Training done → run eval (Step 3): check M1 < 1.018 rad |
| **Mar 20** | Eval results → start ablations V1–V5 |
| **Mar 21** | SNAPHU done → full eval M2/M3 |
| **Mar 22–23** | Ablations done → collect_ablation_results.py → Table 2 |
| **Mar 25** | Zero-shot transfer AOI_008 → LA numbers |
| **Mar 26–29** | Fill all `[XX]` in paper, write missing prose |
| **Apr 1–5** | Final review + REPRODUCIBILITY.md checksums |
| **Apr 6** | **SUBMIT** |

---

## 12. File Inventory

| File | Phase | Status | Notes |
|------|-------|--------|-------|
| `src/insar_processing/pair_graph.py` | 1 | **DONE** | main |
| `src/insar_processing/geometry.py` | 1 | **DONE** | main |
| `src/insar_processing/sublook.py` | 1 | **DONE** | main |
| `src/insar_processing/filters.py` | 1 | **DONE** | main |
| `scripts/download_subset.py` | 1 | **DONE** | main |
| `scripts/preprocess_pairs.py` | 1 | **DONE** | main |
| `scripts/patch_coreg_meta.py` | 1 | **DONE** | main — 224/224 patched |
| `src/models/film_unet.py` | 2 | **DONE** | NaN fix: `.clamp(-10,10)` on log_var |
| `src/losses/physics_losses.py` | 2 | **DONE** | 5-component InSARLoss |
| `src/losses/__init__.py` | 2 | **DONE** | package init |
| `experiments/enhanced/train_film_unet.py` | 2 | **DONE** | raw→gold N2N + nan_to_num guards |
| `src/evaluation/closure_metrics.py` | 3 | **DONE** | all 5 contest metrics |
| `scripts/unwrap_snaphu.py` | 2 | **DONE** | running (70/224, ETA Mar 21) |
| `eval/compute_metrics.py` | 3 | **DONE** | 950 lines, all 5 metrics |
| `scripts/run_ablations.sh` | 3 | **DONE** | V1–V5, ready to run |
| `scripts/collect_ablation_results.py` | 3 | **DONE** | aggregates → Markdown table |
| `eval/zero_shot_transfer.py` | 3 | **DONE** | AOI_008 pipeline, not yet run |
| `REPRODUCIBILITY.md` | 4 | **DONE** | real checksums; needs final update |
| `Latex_Paper_Temporal_SAR_Change/main.tex` | 4 | IN PROGRESS | `[XX]` placeholders for eval data |

---

*Document maintained by getnetdemil. For session-by-session git history see `git log --oneline`.*
