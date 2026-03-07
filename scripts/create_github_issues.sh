#!/usr/bin/env bash
# Creates all custom labels, the contest milestone, and all 21 plan.md issues on GitHub.
# Run once: bash scripts/create_github_issues.sh
# Requires: gh CLI authenticated (gh auth status)
# Idempotent: safe to run again — label/milestone creation errors are silenced.

set -euo pipefail
REPO="getnetdemil/Learning-Assisted-InSAR-DEM-Enhancement"

echo "==> Checking gh auth..."
gh auth status --hostname github.com

# ── 1. Labels ──────────────────────────────────────────────────────────────
echo "==> Creating labels..."
gh label create "phase-1" --color "0075ca" --description "Data access and pair graph"      --repo "$REPO" 2>/dev/null || true
gh label create "phase-2" --color "e4e669" --description "Baseline InSAR products"        --repo "$REPO" 2>/dev/null || true
gh label create "phase-3" --color "7057ff" --description "Deep learning method"            --repo "$REPO" 2>/dev/null || true
gh label create "phase-4" --color "d93f0b" --description "Evaluation and paper"            --repo "$REPO" 2>/dev/null || true
gh label create "model"   --color "6f42c1" --description "Neural network code"             --repo "$REPO" 2>/dev/null || true
gh label create "data"    --color "008672" --description "S3, manifests, preprocessing"    --repo "$REPO" 2>/dev/null || true
gh label create "eval"    --color "e99695" --description "Metrics and closure error"       --repo "$REPO" 2>/dev/null || true
gh label create "scripts" --color "f9d0c4" --description "CLI scripts in scripts/"         --repo "$REPO" 2>/dev/null || true
gh label create "blocked" --color "b60205" --description "Cannot proceed"                  --repo "$REPO" 2>/dev/null || true
echo "    Labels done."

# ── 2. Milestone ───────────────────────────────────────────────────────────
MILESTONE_TITLE="Contest Submission — April 06, 2026"
echo "==> Creating milestone..."
gh api "repos/$REPO/milestones" \
  --method POST \
  --field title="$MILESTONE_TITLE" \
  --field due_on="2026-04-06T23:59:00Z" \
  --field description="IEEE GRSS 2026 Data Fusion Contest deadline." \
  --jq '.number' 2>/dev/null || true
echo "    Milestone done."

# ── 3. Helper ──────────────────────────────────────────────────────────────
create_issue() {
  local title="$1"
  local labels="$2"
  local body="$3"
  gh issue create \
    --repo "$REPO" \
    --title "$title" \
    --label "$labels" \
    --milestone "$MILESTONE_TITLE" \
    --body "$body"
  echo "    Created: $title"
}

echo "==> Creating issues..."

# ── Phase 1 ────────────────────────────────────────────────────────────────

create_issue \
  "[Phase 1] Crawl STAC contest collection and build metadata parquet" \
  "phase-1,data,scripts" \
  "## Goal
Build a local parquet index of all 791 SLC items in the contest STAC collection.

## Acceptance criteria
- [ ] \`data/manifests/full_index.parquet\` exists with 791 rows
- [ ] Columns: item_id, datetime, bbox, mode, look_direction, incidence, asset hrefs
- [ ] Script: \`scripts/download_subset.py\`

## Implementation notes
- STAC root: \`https://capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-ieee-data-contest/collection.json\`
- Filter SLC items via \`_SLC_\` in asset href
- Store raw crawl cache as \`data/manifests/_raw_crawl_cache.parquet\`

## Status
COMPLETE — \`data/manifests/full_index.parquet\` (791 rows, 39 AOIs assigned)"

create_issue \
  "[Phase 1] Select AOI subset and document rationale" \
  "phase-1,data" \
  "## Goal
Inspect mode/incidence/AOI distribution and select 3–6 diverse AOIs for training and evaluation.

## Acceptance criteria
- [ ] Primary AOI chosen and justified (land cover, geometry diversity, pair count)
- [ ] At least one secondary AOI for zero-shot transfer demo
- [ ] Selection rationale documented
- [ ] Download manifest with checksums: \`data/manifests/subset_manifest.csv\`

## Status
COMPLETE — Primary: AOI_000 (Hawaii, 221 collects, both orbits, inc 35.8–56.3°, volcanic terrain)
Secondary: AOI_008 (LA) for zero-shot transfer; AOI_024 (W. Australia) for stable baseline"

create_issue \
  "[Phase 1] Implement pair-graph construction and Q_ij edge scoring" \
  "phase-1,data" \
  "## Goal
Build the pair graph: nodes = collects, edges = candidate interferometric pairs scored by geometry quality.

## Acceptance criteria
- [ ] \`src/insar_processing/pair_graph.py\` implemented
- [ ] Edge quality score Q_ij = product of temporal, incidence, grazing, baseline, and compatibility gates
- [ ] \`data/manifests/hawaii_pairs.parquet\` with ≥8,000 pairs and B_perp column
- [ ] \`data/manifests/hawaii_triplets_strict.parquet\` with triplets for closure evaluation

## Score formula
\`\`\`
Q_ij = 1[mode_compat] × 1[look_compat] × 1[flight_compat]
       × exp(−α_t × Δt/τ_t) × exp(−α_inc × |Δθ_inc|/τ_inc)
       × exp(−α_b × |B_perp|/τ_b)
\`\`\`
Default: τ_t=30d, τ_inc=5°, τ_b=300m

## Status
COMPLETE — 8,834 pairs and 24,171 triplets generated for AOI_000 (Hawaii)"

create_issue \
  "[Phase 1] Compute perpendicular baseline from orbit state vectors" \
  "phase-1,data" \
  "## Goal
Compute B_perp for each candidate pair from orbit/state vectors via ISCE3 geometry.

## Acceptance criteria
- [ ] \`src/insar_processing/geometry.py\` implemented
- [ ] B_perp column populated in \`data/manifests/hawaii_pairs.parquet\`
- [ ] Uses ISCE3 state-vector interpolation (not extended JSON sidecar, which is inconsistent)

## Implementation notes
- Capella timestamps have nanosecond precision — truncate to 6 decimal places before \`fromisoformat()\`
- \`reference_antenna_position\` not consistently present in extended JSON; always use state vectors
- ISCE3 orbit object available via \`capella_reader.CapellaSLC.from_file().adapted.isce3_orbit\`

## Status
COMPLETE — \`src/insar_processing/geometry.py\` done"

create_issue \
  "[Phase 1] Pair selection — time-series and DEM-sensitivity subgraphs" \
  "phase-1,data" \
  "## Goal
From the full pair graph, select two task-specific subgraphs within compute budget.

## Acceptance criteria
- [ ] Time-series subgraph: minimum spanning tree + short-Δt edges for temporal consistency
- [ ] DEM-sensitivity subgraph: diverse B_perp distribution with Q_ij ≥ threshold
- [ ] Budget documented in \`configs/data/pair_selection.yaml\`
- [ ] Selected pairs written to \`data/manifests/hawaii_pairs.parquet\`

## Status
COMPLETE — 8,834 pairs selected for AOI_000; see \`data/manifests/hawaii_pairs.parquet\`"

create_issue \
  "[Phase 1] Pair-graph exploration notebook" \
  "phase-1,data" \
  "## Goal
Visualize the pair graph, AOI coverage, and pair-quality distribution for \"temporal storytelling\".

## Acceptance criteria
- [ ] \`notebooks/01_pair_graph_exploration.ipynb\` exists and runs end-to-end
- [ ] Pair graph diagram: nodes = collects, edges colored by Q_ij
- [ ] Distribution plots: Δt, B_perp, incidence angle, coherence proxy
- [ ] AOI map with collect locations
- [ ] Summary statistics table (pairs per AOI, by mode/look)"

# ── Phase 2 ────────────────────────────────────────────────────────────────

create_issue \
  "[Phase 2] Per-pair SLC coregistration pipeline" \
  "phase-2,scripts" \
  "## Goal
Coregister each SLC pair and save misregistration metrics as downstream features.

## Acceptance criteria
- [ ] \`scripts/preprocess_pairs.py\` fully implemented (coreg → ifg → coh → filter)
- [ ] Uses ISCE3 orbit geometry for coarse alignment + correlation offsets for refinement
- [ ] Misregistration metrics saved per pair (become usability predictor features)
- [ ] Output directory structure: \`data/processed/<aoi>/<pair_id>/\`

## Status
COMPLETE — \`scripts/preprocess_pairs.py\` end-to-end pipeline implemented
(read → coreg → ifg → coherence → Goldstein filter)"

create_issue \
  "[Phase 2] Interferogram formation and coherence estimation" \
  "phase-2,scripts" \
  "## Goal
Form complex interferograms and estimate coherence for all selected pairs.

## Acceptance criteria
- [ ] Per-pair outputs in \`data/processed/<aoi>/<pair_id>/\`:
  - \`interferogram.tif\` (complex, stacked Re/Im channels)
  - \`coherence.tif\`
  - \`amplitude_ref.tif\`, \`amplitude_sec.tif\`
- [ ] Coherence estimated using sliding window (configurable size)
- [ ] I/O uses \`src/insar_processing/io.py\` with CInt16 support"

create_issue \
  "[Phase 2] Classical filtering baselines — Goldstein, NL-InSAR, BM3D" \
  "phase-2,scripts" \
  "## Goal
Implement three non-DL baselines for fair comparison with the DL method.

## Acceptance criteria
- [ ] \`src/insar_processing/filters.py\` with:
  - Goldstein adaptive filter
  - NL-InSAR (nonlocal estimator)
  - BM3D-like denoiser (applied to complex channels)
- [ ] All three produce filtered interferogram GeoTIFFs alongside the raw product
- [ ] Coherence-weighted variants documented

## Status
COMPLETE — \`src/insar_processing/filters.py\` done (Goldstein, adaptive, boxcar coherence)"

create_issue \
  "[Phase 2] SNAPHU phase unwrapping script" \
  "phase-2,scripts" \
  "## Goal
Implement reproducible SNAPHU-based phase unwrapping for all pairs.

## Acceptance criteria
- [ ] \`scripts/unwrap_snaphu.py\` implemented
- [ ] Coherence mask threshold consistent and reported (mask coverage % logged)
- [ ] SNAPHU config file templated and checked in
- [ ] Unwrapped phase saved as \`data/processed/<aoi>/<pair_id>/unwrapped_phase.tif\`
- [ ] Connected-component map saved for unwrap success rate metric"

create_issue \
  "[Phase 2] Compute all 5 contest metrics on baseline products" \
  "phase-2,eval" \
  "## Goal
Establish baseline numbers (raw interferograms + classical filters) to benchmark DL improvement.

## Acceptance criteria
- [ ] \`src/evaluation/closure_metrics.py\` with \`compute_all_contest_metrics()\`
- [ ] All 5 metrics implemented:
  1. Triplet closure error (median + 95th pct)
  2. Unwrap success rate (≥90% component coverage + closure gate)
  3. Percent usable pairs (coherence + unwrap + closure gates)
  4. DEM NMAD (\`1.4826 × median(|e − median(e)|)\`)
  5. Temporal consistency residual (\`‖W(Ax − φ)‖₂\`)
- [ ] Baseline numbers documented in \`results/baseline_metrics.json\`"

# ── Phase 3 ────────────────────────────────────────────────────────────────

create_issue \
  "[Phase 3] Implement FiLM-conditioned U-Net" \
  "phase-3,model" \
  "## Goal
Build the primary DL model: U-Net with Feature-wise Linear Modulation for geometry conditioning.

## Acceptance criteria
- [ ] \`src/models/film_unet.py\` with \`FiLMUNet\` class
- [ ] Input: complex interferogram (2 channels: Re, Im); optional coherence channel
- [ ] Conditioning vector: \`[Δt, Δθ_inc, Δθ_graze, B_perp, mode_embed, look_embed, SNR_proxy]\` (dim=7)
- [ ] Output: denoised complex interferogram (2ch) + per-pixel log-variance (1ch)
- [ ] FiLM layers applied at each encoder/decoder level
- [ ] Unit test: forward pass with random input returns correct output shape

## Architecture
\`\`\`python
class FiLMUNet(nn.Module):
    def __init__(self, in_channels=2, metadata_dim=7, features=[32,64,128,256]):
        ...
    def forward(self, x, metadata):
        # returns: denoised (B, 2, H, W), log_var (B, 1, H, W)
\`\`\`"

create_issue \
  "[Phase 3] Implement sub-look splitting for Noise2Noise training" \
  "phase-3,model" \
  "## Goal
Split each SLC into two independent noisy views via sub-look / aperture splitting for N2N self-supervision.

## Acceptance criteria
- [ ] \`src/insar_processing/sublook.py\` implemented
- [ ] FFT-based sub-aperture split (odd/even frequency bins)
- [ ] Two resulting views: \`I^(a)\` and \`I^(b)\` — statistically independent noise realizations
- [ ] Output interferogram pairs per view for N2N loss computation

## Status
COMPLETE — \`src/insar_processing/sublook.py\` done (FFT + odd/even N2N splits)"

create_issue \
  "[Phase 3] Implement physics loss functions (N2N, closure, temporal, gradient)" \
  "phase-3,model" \
  "## Goal
Implement the full self-supervised loss suite in \`src/losses/physics_losses.py\`.

## Acceptance criteria
- [ ] \`src/losses/physics_losses.py\` with all 5 components:
  - \`L_n2n\`: Noise2Noise L1 between sub-look views
  - \`L_unc\`: Heteroscedastic NLL (uncertainty-aware)
  - \`L_closure\`: Triplet closure-consistency weighted by uncertainty/coherence
  - \`L_temporal\`: SBAS-like stack inversion residual
  - \`L_grad\`: Spectral/fringe-preservation gradient loss
- [ ] Combined loss with configurable λ weights (defaults: 1.0, 0.5, 0.3, 0.2, 0.1)
- [ ] Unit tests for each loss component

## Key equations
\`\`\`
L_closure = Σ_p w(p) × (1 − cos(wrap(φ̂_ij + φ̂_jk − φ̂_ik)))
L_temporal = ‖W(Ax* − φ̂)‖²  where x* = argmin ‖W(Ax − φ̂)‖²
\`\`\`"

create_issue \
  "[Phase 3] FiLM U-Net training script with AOI-based splits" \
  "phase-3,model,scripts" \
  "## Goal
Training loop for FiLMUNet with proper AOI-based geographic splits to prevent data leakage.

## Acceptance criteria
- [ ] \`experiments/enhanced/train_film_unet.py\` implemented
- [ ] Dataset/DataLoader using sliding-window tile sampling (not one-by-one)
- [ ] AOI-based splits: train 60–70%, val 10–20%, test 20–30% (held out until final eval)
- [ ] Physically-safe augmentations: 90° rotations, flips, global phase offset, amplitude jitter
- [ ] Checkpointing: best-by-closure-error, best-by-unwrap-success, and final epoch
- [ ] Each checkpoint stores: config.yaml, git hash, dataset manifest
- [ ] WandB or equivalent experiment tracking integrated

## Run command
\`\`\`bash
python experiments/enhanced/train_film_unet.py \\
    --data_config configs/data/capella_aoi_selection.yaml \\
    --model_config configs/model/film_unet.yaml \\
    --train_config configs/train/contest.yaml
\`\`\`"

create_issue \
  "[Phase 3] Uncertainty-weighted SNAPHU and SBAS integration" \
  "phase-3,model" \
  "## Goal
Use the model's predicted per-pixel uncertainty to weight downstream unwrapping and stack inversion.

## Acceptance criteria
- [ ] SNAPHU coherence-like weight map computed as \`1/σ²(p)\` from FiLMUNet output
- [ ] SBAS design matrix weights: \`W = diag(1/σ²)\` integrated in \`eval/compute_metrics.py\`
- [ ] DEM generation uses uncertainty-weighted multi-baseline combination
- [ ] Comparison table: uniform weights vs uncertainty weights on all 5 contest metrics"

create_issue \
  "[Phase 3] Run ablation studies" \
  "phase-3,eval" \
  "## Goal
Systematically ablate each component of the loss function and FiLM conditioning.

## Acceptance criteria
- [ ] Ablation table with 6 rows:
  | Variant | Closure err | Unwrap rate | Usable pairs | Temporal res |
  |---------|------------|-------------|--------------|--------------|
  | N2N-only | | | | |
  | +closure | | | | |
  | +temporal | | | | |
  | +uncertainty | | | | |
  | +FiLM (full) | | | | |
  | −FiLM | | | | |
- [ ] Each variant trained from same random seed with identical data splits
- [ ] Results saved to \`results/ablation_metrics.json\`"

# ── Phase 4 ────────────────────────────────────────────────────────────────

create_issue \
  "[Phase 4] Generate all 5 contest metrics tables and figures" \
  "phase-4,eval,scripts" \
  "## Goal
Produce publication-ready tables and figures for all contest metrics across all methods.

## Acceptance criteria
- [ ] \`eval/compute_metrics.py\` generates:
  - Main metrics table per AOI + aggregated (all 5 metrics × all methods)
  - Pair-selection table (heuristic vs predictor, fraction usable, compute cost)
  - Ablation table
- [ ] All figures exported as PDF/PNG to \`results/figures/\`
- [ ] Numbers match what is reported in the paper

## Methods to compare
1. Raw interferogram (no filter)
2. Goldstein filter
3. NL-InSAR
4. BM3D
5. FiLMUNet (N2N-only)
6. FiLMUNet (full loss + uncertainty)"

create_issue \
  "[Phase 4] Pair-graph visualizations and temporal storytelling" \
  "phase-4,eval" \
  "## Goal
Create compelling visual artifacts that demonstrate why the pair-graph approach and temporal modeling help.

## Acceptance criteria
- [ ] Pair-graph diagram: nodes = collects (colored by date), edges colored by Q_ij
- [ ] Spatial maps: coherence, closure error, unwrap success, per-pixel uncertainty
- [ ] Time-series residual maps before/after DL
- [ ] Side-by-side panels: raw ifg / Goldstein / FiLMUNet / closure phase overlay
- [ ] All visualizations in \`notebooks/02_results_visualization.ipynb\`
- [ ] Key figures exported as camera-ready PDFs"

create_issue \
  "[Phase 4] Write 4-page contest paper" \
  "phase-4" \
  "## Goal
Write the IEEE GRSS 2026 DFC 4-page paper (excluding references).

## Acceptance criteria
- [ ] Submitted by April 06, 2026 (AoE)
- [ ] Internal draft complete by April 20

## Structure
1. **Problem framing**: geometry-diverse temporal InSAR stacks; why pair-graph + closure matters
2. **Method diagram** + key equations (pair scoring, DL loss, uncertainty integration)
3. **Main results table** (5 metrics, all methods) + 2–3 striking visual comparisons
4. **Ablation table** (compact, all loss components)
5. **Scalability discussion** (1–2 sentences)

## Requirements
- [ ] All figures at 300 DPI, IEEE two-column format
- [ ] Method diagram shows end-to-end pipeline
- [ ] Results match \`results/ablation_metrics.json\` exactly
- [ ] Trained model weights uploaded (Hugging Face or GitHub release)"

create_issue \
  "[Phase 4] REPRODUCIBILITY.md and one-command pipeline" \
  "phase-4,scripts" \
  "## Goal
Ensure the contest can be fully reproduced from a single command. Contest explicitly disqualifies entries with missing code or unclear reproducibility.

## Acceptance criteria
- [ ] \`REPRODUCIBILITY.md\` documents:
  - STAC root URL and contest collection ID
  - Exact download manifest + SHA-256 checksums
  - Fixed random seeds and deterministic settings
  - Environment: \`environment.yml\` with pinned versions
- [ ] One-command pipeline:
  \`\`\`bash
  bash run_all.sh  # S3 download → preprocess → train → eval → figures
  \`\`\`
- [ ] Git hash stored in every checkpoint config
- [ ] \`README.md\` updated: one-line purpose, environment setup, reproduction command
- [ ] Trained model weights uploaded to Hugging Face or GitHub release"

echo ""
echo "==> All done! Verify with:"
echo "    gh issue list --repo $REPO --state all"
echo "    gh label list --repo $REPO"
echo "    gh api repos/$REPO/milestones --jq '.[].title'"
