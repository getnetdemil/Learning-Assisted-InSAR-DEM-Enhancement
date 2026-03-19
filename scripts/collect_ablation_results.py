#!/usr/bin/env python3
"""
scripts/collect_ablation_results.py — Aggregate ablation training summaries into a table.

Reads training_summary.json from each ablation variant subdirectory and combines
with eval/compute_metrics.py CSV outputs (if available) to produce:
  - A Markdown ablation table (printed to stdout)
  - experiments/enhanced/outputs/ablation_table.csv

Usage
-----
python scripts/collect_ablation_results.py [--ckpt_base <dir>] [--eval_base <dir>]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Canonical ablation run names in presentation order
ABLATION_VARIANTS = [
    ("ablation_v1_n2n_only",       "V1: N2N-only"),
    ("ablation_v2_plus_closure",   "V2: +Closure"),
    ("ablation_v3_plus_temporal",  "V3: +Temporal"),
    ("ablation_v4_full_model",     "V4: Full model"),
    ("ablation_v5_no_film",        "V5: No FiLM"),
]


def _load_summary(ckpt_dir: Path) -> dict | None:
    summary_path = ckpt_dir / "training_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def _load_eval_metrics(eval_dir: Path) -> dict:
    """Load metrics_comparison.csv from eval output directory."""
    csv_path = eval_dir / "metrics_comparison.csv"
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
        # Pivot: metric × method → value
        result = {}
        for _, row in df.iterrows():
            key = f"{row['metric'].strip()}_{row['method'].strip()}"
            result[key] = row["value"]
        return result
    except Exception:
        return {}


def fmt(v, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect ablation results into a comparison table."
    )
    p.add_argument(
        "--ckpt_base",
        default=str(ROOT / "experiments/enhanced/checkpoints/film_unet"),
        help="Base checkpoint directory containing ablation subdirs.",
    )
    p.add_argument(
        "--eval_base",
        default=str(ROOT / "experiments/enhanced/outputs"),
        help="Base eval output directory containing ablation result CSVs.",
    )
    p.add_argument(
        "--out_csv",
        default=str(ROOT / "experiments/enhanced/outputs/ablation_table.csv"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_base = Path(args.ckpt_base)
    eval_base = Path(args.eval_base)

    rows = []
    for run_name, label in ABLATION_VARIANTS:
        row: dict = {"Variant": label, "run_name": run_name}

        # Training summary
        summary = _load_summary(ckpt_base / run_name)
        if summary is not None:
            row["Epochs"] = summary.get("num_epochs", "—")
            row["zero_film"] = "✓" if summary.get("zero_film") else ""
            lw = summary.get("loss_weights", {})
            row["λ_closure"]  = lw.get("closure",  0.0)
            row["λ_temporal"] = lw.get("temporal", 0.0)
            row["λ_grad"]     = lw.get("grad",     0.0)
            row["Best val closure"] = fmt(summary.get("best_val_closure"))
            final = summary.get("final_val_metrics", {})
            row["Val total loss"] = fmt(final.get("total"))
        else:
            row["Status"] = "PENDING"

        # Eval metrics (if compute_metrics.py was run for this variant)
        eval_dir = eval_base / run_name
        eval_metrics = _load_eval_metrics(eval_dir)
        if eval_metrics:
            film_closure = eval_metrics.get(
                "Triplet Closure Error (rad)_film_unet"
            )
            gold_closure = eval_metrics.get(
                "Triplet Closure Error (rad)_goldstein"
            )
            row["M1 closure (rad)"] = fmt(film_closure)
            if gold_closure and film_closure and not np.isnan(gold_closure) \
                    and not np.isnan(film_closure):
                pct = (film_closure - gold_closure) / (abs(gold_closure) + 1e-12) * 100
                row["M1 vs baseline"] = f"{pct:+.1f}%"
            film_temporal = eval_metrics.get(
                "Temporal Residual (rad)_film_unet"
            )
            row["M5 temporal (rad)"] = fmt(film_temporal)

        rows.append(row)

    df = pd.DataFrame(rows)

    # ── Print Markdown table ──────────────────────────────────────────────────
    print("\n## Ablation Results\n")
    # Core columns to display
    display_cols = [
        "Variant", "λ_closure", "λ_temporal", "λ_grad", "zero_film",
        "Best val closure", "M1 closure (rad)", "M1 vs baseline",
        "M5 temporal (rad)",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].fillna("—").to_markdown(index=False))

    # ── Contest target row ────────────────────────────────────────────────────
    print("\n**Contest targets:** M1 ↓≥30% from baseline (1.018 rad → <0.713 rad),")
    print("M5 ↓≥20% from baseline (0.050 rad → <0.040 rad)\n")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Warn about pending runs
    pending = [r["Variant"] for r in rows if "Status" in r and r["Status"] == "PENDING"]
    if pending:
        print(f"\nPending (no training_summary.json found): {', '.join(pending)}")
        print("Run scripts/run_ablations.sh first.")


if __name__ == "__main__":
    main()
