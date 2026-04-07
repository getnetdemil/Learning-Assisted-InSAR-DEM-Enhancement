#!/usr/bin/env python3
"""
eval/zero_shot_transfer.py — Zero-shot transfer evaluation on AOI_008 (Los Angeles).

Uses the best_closure.pt checkpoint from AOI_000 (Hawaii) training — NO retraining.
Evaluates the 5 contest metrics on AOI_008 pairs to demonstrate generalization.

Workflow
--------
1. Filter full_index.parquet for AOI_008, select top-30 pairs by coherence/recency
2. Run scripts/preprocess_pairs.py on those pairs (if not already done)
3. Run FiLMUNet inference + eval/compute_metrics.py on the AOI_008 pair set
4. Compare against Goldstein baseline on the same pairs

Usage
-----
# Step 1: Select AOI_008 pairs and write a manifest
python eval/zero_shot_transfer.py --phase select \
    --full_index data/manifests/full_index.parquet \
    --out_manifest data/manifests/aoi008_pairs.parquet \
    --n_pairs 30

# Step 2: Preprocess AOI_008 pairs (run preprocess_pairs.py)
#   bash eval/zero_shot_transfer.py --phase preprocess   (generates the command)

# Step 3: Evaluate with zero-shot checkpoint
python eval/zero_shot_transfer.py --phase eval \
    --checkpoint experiments/enhanced/checkpoints/film_unet/best_closure.pt \
    --pairs_dir data/processed/pairs_aoi008 \
    --triplets_manifest data/manifests/aoi008_triplets.parquet \
    --out_dir experiments/enhanced/outputs/zero_shot_aoi008
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Phase 1: Select AOI_008 pairs
# ---------------------------------------------------------------------------

def select_aoi008_pairs(
    full_index: Path,
    out_manifest: Path,
    n_pairs: int = 30,
    aoi: str = "AOI_008",
) -> None:
    """Filter full_index for aoi, rank by collect count, write top-n_pairs."""
    df = pd.read_parquet(full_index)

    # Filter to target AOI
    aoi_df = df[df["aoi"] == aoi].copy()
    if aoi_df.empty:
        print(f"ERROR: No items found for {aoi} in {full_index}")
        sys.exit(1)

    print(f"Found {len(aoi_df)} acquisitions for {aoi}")

    # Sort by collect_time descending (most recent first) and take first n_pairs
    if "collect_time" in aoi_df.columns:
        aoi_df = aoi_df.sort_values("collect_time", ascending=False)
    elif "datetime" in aoi_df.columns:
        aoi_df = aoi_df.sort_values("datetime", ascending=False)

    selected = aoi_df.head(n_pairs)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(out_manifest, index=False)
    print(f"Selected {len(selected)} acquisitions → {out_manifest}")
    print(f"Orbit IDs: {selected['orbit_state'].value_counts().to_dict() if 'orbit_state' in selected.columns else 'N/A'}")
    print(f"\nNext step: preprocess these acquisitions into pairs.")
    print(f"  Use scripts/preprocess_pairs.py with the selected acquisitions.")
    _print_preprocess_command(out_manifest)


def _print_preprocess_command(manifest: Path) -> None:
    print("\n# Suggested preprocess command:")
    print(f"export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH")
    print(f"conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu \\")
    print(f"    python scripts/preprocess_pairs.py \\")
    print(f"    --manifest {manifest} \\")
    print(f"    --pairs_dir data/processed/pairs_aoi008 \\")
    print(f"    --workers 4")


# ---------------------------------------------------------------------------
# Phase 3: Run zero-shot eval
# ---------------------------------------------------------------------------

def run_eval(
    checkpoint: Path,
    pairs_dir: Path,
    triplets_manifest: Path,
    out_dir: Path,
) -> None:
    """Invoke eval/compute_metrics.py on AOI_008 pairs using the Hawaii checkpoint."""
    compute_metrics = ROOT / "eval" / "compute_metrics.py"
    if not compute_metrics.exists():
        print(f"ERROR: {compute_metrics} not found.")
        sys.exit(1)

    cmd = [
        sys.executable, str(compute_metrics),
        "--checkpoint", str(checkpoint),
        "--pairs_dir", str(pairs_dir),
        "--triplets_manifest", str(triplets_manifest),
        "--out_dir", str(out_dir),
    ]

    # If no unw_phase.tif exists yet, skip SNAPHU-dependent metrics
    has_unw = any((pairs_dir / d / "unw_phase.tif").exists()
                  for d in pairs_dir.iterdir() if d.is_dir()) \
              if pairs_dir.exists() else False
    if not has_unw:
        cmd.append("--skip_snaphu_metrics")
        print("Note: unw_phase.tif not found — skipping M2/M3 (run unwrap_snaphu.py first).")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Print comparison vs Hawaii baseline
    csv_path = out_dir / "metrics_comparison.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print("\n=== Zero-Shot AOI_008 Results ===")
        print(df.to_string(index=False))
        print("\nHawaii (AOI_000) reference M1 baseline: 1.018 rad")
        print("Hawaii (AOI_000) reference M5 baseline: 0.050 rad")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Zero-shot transfer evaluation on AOI_008 (Los Angeles).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--phase", choices=["select", "eval"], required=True,
                   help="Workflow phase to run.")

    # Phase: select
    p.add_argument("--full_index",
                   default="data/manifests/full_index.parquet")
    p.add_argument("--out_manifest",
                   default="data/manifests/aoi008_pairs.parquet")
    p.add_argument("--aoi", default="AOI_008")
    p.add_argument("--n_pairs", type=int, default=30)

    # Phase: eval
    p.add_argument("--checkpoint",
                   default="experiments/enhanced/checkpoints/film_unet/best_closure.pt")
    p.add_argument("--pairs_dir",
                   default="data/processed/pairs_aoi008")
    p.add_argument("--triplets_manifest",
                   default="data/manifests/aoi008_triplets.parquet")
    p.add_argument("--out_dir",
                   default="experiments/enhanced/outputs/zero_shot_aoi008")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.phase == "select":
        select_aoi008_pairs(
            full_index=ROOT / args.full_index,
            out_manifest=ROOT / args.out_manifest,
            n_pairs=args.n_pairs,
            aoi=args.aoi,
        )
    elif args.phase == "eval":
        run_eval(
            checkpoint=ROOT / args.checkpoint,
            pairs_dir=ROOT / args.pairs_dir,
            triplets_manifest=ROOT / args.triplets_manifest,
            out_dir=ROOT / args.out_dir,
        )


if __name__ == "__main__":
    main()
