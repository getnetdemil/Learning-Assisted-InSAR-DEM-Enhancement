#!/usr/bin/env bash
# scripts/run_ablations.sh — Run 5 FiLMUNet ablation variants sequentially.
#
# Each variant trains for 20 epochs with a specific loss configuration.
# Results (training_summary.json) are written to:
#   experiments/enhanced/checkpoints/film_unet/<run_name>/
#
# Usage:
#   bash scripts/run_ablations.sh
#   # or with custom python path:
#   PYTHON=/path/to/python bash scripts/run_ablations.sh
#
# After completion, run:
#   python scripts/collect_ablation_results.py

set -euo pipefail

PYTHON="${PYTHON:-conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu python}"
TRAIN="experiments/enhanced/train_film_unet.py"
DATA_CFG="configs/data/capella_aoi_selection.yaml"
MODEL_CFG="configs/model/film_unet.yaml"
TRAIN_CFG="configs/train/contest.yaml"

export LD_LIBRARY_PATH="/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:${LD_LIBRARY_PATH:-}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "FiLMUNet Ablation Suite — 5 variants × 20 epochs"
echo "Started: $(date)"
echo "============================================================"

# ── V1: N2N-only (closure=0, temporal=0, grad=0) ─────────────────────────────
echo ""
echo "[V1] N2N-only (no closure / temporal / gradient losses)"
$PYTHON $TRAIN \
    --data_config $DATA_CFG \
    --model_config $MODEL_CFG \
    --train_config $TRAIN_CFG \
    --loss_closure 0.0 \
    --loss_temporal 0.0 \
    --loss_grad 0.0 \
    --epochs 20 \
    --run_name ablation_v1_n2n_only
echo "[V1] Done: $(date)"

# ── V2: N2N + closure ────────────────────────────────────────────────────────
echo ""
echo "[V2] N2N + closure (no temporal / gradient)"
$PYTHON $TRAIN \
    --data_config $DATA_CFG \
    --model_config $MODEL_CFG \
    --train_config $TRAIN_CFG \
    --loss_closure 0.3 \
    --loss_temporal 0.0 \
    --loss_grad 0.0 \
    --epochs 20 \
    --run_name ablation_v2_plus_closure
echo "[V2] Done: $(date)"

# ── V3: N2N + closure + temporal ─────────────────────────────────────────────
echo ""
echo "[V3] N2N + closure + temporal (no gradient)"
$PYTHON $TRAIN \
    --data_config $DATA_CFG \
    --model_config $MODEL_CFG \
    --train_config $TRAIN_CFG \
    --loss_closure 0.3 \
    --loss_temporal 0.2 \
    --loss_grad 0.0 \
    --epochs 20 \
    --run_name ablation_v3_plus_temporal
echo "[V3] Done: $(date)"

# ── V4: Full model (N2N + closure + temporal + gradient) ──────────────────────
echo ""
echo "[V4] Full model — all losses active (20-epoch retrain)"
$PYTHON $TRAIN \
    --data_config $DATA_CFG \
    --model_config $MODEL_CFG \
    --train_config $TRAIN_CFG \
    --loss_closure 0.3 \
    --loss_temporal 0.2 \
    --loss_grad 0.1 \
    --epochs 20 \
    --run_name ablation_v4_full_model
echo "[V4] Done: $(date)"

# ── V5: Full losses, no FiLM conditioning ────────────────────────────────────
echo ""
echo "[V5] Full losses, no FiLM conditioning (--zero_film)"
$PYTHON $TRAIN \
    --data_config $DATA_CFG \
    --model_config $MODEL_CFG \
    --train_config $TRAIN_CFG \
    --loss_closure 0.3 \
    --loss_temporal 0.2 \
    --loss_grad 0.1 \
    --epochs 20 \
    --zero_film \
    --run_name ablation_v5_no_film
echo "[V5] Done: $(date)"

echo ""
echo "============================================================"
echo "All ablation runs complete: $(date)"
echo "Collect results with:"
echo "  python scripts/collect_ablation_results.py"
echo "============================================================"
