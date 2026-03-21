"""
Retroactive coregistration quality assessment from already-processed pairs.

Reads existing ifg_goldstein.tif + coherence.tif for each processed pair and
computes lightweight quality metrics without re-running coregistration.

Usage
-----
python scripts/assess_coreg_quality.py \
    --pairs_dir data/processed/pairs \
    --out_csv   data/manifests/coreg_quality.csv
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def assess_pair(pair_dir: Path) -> dict | None:
    """
    Compute quality metrics for one processed pair.

    Returns a dict with per-pair metrics, or None if files are missing.
    """
    ifg_path = pair_dir / "ifg_goldstein.tif"
    coh_path = pair_dir / "coherence.tif"
    meta_path = pair_dir / "coreg_meta.json"

    if not ifg_path.exists() or not coh_path.exists():
        return None

    # Load coherence
    with rasterio.open(coh_path) as src:
        coh = src.read(1).astype(np.float32)

    # Load interferogram (Re + Im)
    with rasterio.open(ifg_path) as src:
        re = src.read(1).astype(np.float32)
        im = src.read(2).astype(np.float32)

    ifg_phase = np.arctan2(im, re)  # wrapped phase

    # Coherent pixel mask (coh > 0.2)
    coherent = coh > 0.2

    mean_coh = float(coh.mean())
    coh_p10 = float(np.percentile(coh, 10))

    # Wrapped phase std over coherent pixels (proxy for fringe density vs. noise)
    if coherent.sum() > 100:
        phase_std = float(ifg_phase[coherent].std())
    else:
        phase_std = float("nan")

    # Read existing coreg offsets from metadata (if available)
    row_offset = None
    col_offset = None
    cc_peak_mean = None
    n_coreg_patches = None
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            row_offset = meta.get("row_offset_px")
            col_offset = meta.get("col_offset_px")
            cc_peak_mean = meta.get("cc_peak_mean")
            n_coreg_patches = meta.get("n_coreg_patches")
        except Exception:
            pass

    return {
        "pair_dir":         pair_dir.name,
        "mean_coherence":   mean_coh,
        "coherence_p10":    coh_p10,
        "phase_spatial_std": phase_std,
        "n_coherent_px":    int(coherent.sum()),
        "row_offset_px":    row_offset,
        "col_offset_px":    col_offset,
        "cc_peak_mean":     cc_peak_mean,
        "n_coreg_patches":  n_coreg_patches,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Retroactive coregistration quality assessment.")
    p.add_argument("--pairs_dir", required=True, help="Root dir with per-pair subdirectories")
    p.add_argument("--out_csv", default="data/manifests/coreg_quality.csv",
                   help="Output CSV path")
    p.add_argument("--coh_flag_threshold", type=float, default=0.15,
                   help="Flag pairs with mean_coherence below this value")
    args = p.parse_args()

    pairs_dir = Path(args.pairs_dir)
    pair_dirs = sorted(d for d in pairs_dir.iterdir() if d.is_dir())
    log.info("Found %d pair directories", len(pair_dirs))

    rows = []
    for pd_dir in pair_dirs:
        result = assess_pair(pd_dir)
        if result is None:
            log.debug("Skipping %s (missing files)", pd_dir.name)
            continue
        rows.append(result)

    if not rows:
        log.error("No valid pairs found in %s", pairs_dir)
        return

    df = pd.DataFrame(rows)

    # Flag low-coherence pairs
    df["flag_low_coherence"] = df["mean_coherence"] < args.coh_flag_threshold

    # Summary
    log.info("Assessed %d pairs", len(df))
    log.info("  mean coherence:  %.3f ± %.3f", df["mean_coherence"].mean(), df["mean_coherence"].std())
    log.info("  coherence P10:   %.3f", df["coherence_p10"].mean())
    log.info("  phase_spatial_std: %.3f ± %.3f (rad)",
             df["phase_spatial_std"].mean(), df["phase_spatial_std"].std())
    log.info("  low-coherence flags (< %.2f): %d / %d",
             args.coh_flag_threshold, df["flag_low_coherence"].sum(), len(df))

    if df["cc_peak_mean"].notna().any():
        log.info("  cc_peak_mean:    %.3f ± %.3f",
                 df["cc_peak_mean"].mean(), df["cc_peak_mean"].std())

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    log.info("Saved %s", args.out_csv)


if __name__ == "__main__":
    main()
