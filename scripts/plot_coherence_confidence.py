#!/usr/bin/env python3
"""
Coherence vs FiLMUNet Confidence scatter plot.

For each pair with coherence.tif + log_var.tif, samples pixels and plots
coherence (x) vs confidence = exp(-sqrt(exp(log_var))) (y).

Shows that FiLMUNet uncertainty is physics-grounded: low coherence → low confidence.

Usage
-----
python scripts/plot_coherence_confidence.py \
    --pairs_dir data/processed/AOI024_pairs_full_image/AOI024_selected \
    --out_dir experiments/enhanced/outputs/figures \
    --sample_rate 500
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAIR_COLORS = ["#4878CF", "#E87060", "#2CA02C", "#9467BD",
               "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22"]


def _read(path: Path) -> np.ndarray:
    import rasterio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)


def sample_pair(pair_dir: Path, sample_rate: int = 500) -> tuple[np.ndarray, np.ndarray] | None:
    coh_path = pair_dir / "coherence.tif"
    lv_path  = pair_dir / "log_var.tif"
    if not coh_path.exists() or not lv_path.exists():
        return None

    coh = _read(coh_path)
    lv  = _read(lv_path)

    # Flatten and subsample
    coh_flat = coh.ravel()[::sample_rate]
    lv_flat  = lv.ravel()[::sample_rate]

    # Keep only valid, non-zero-coherence pixels
    valid = np.isfinite(lv_flat) & np.isfinite(coh_flat) & (coh_flat > 0.01)
    coh_s = coh_flat[valid]
    lv_s  = lv_flat[valid]

    # Confidence = exp(-sigma),  sigma = sqrt(exp(log_var))
    sigma      = np.sqrt(np.exp(np.clip(lv_s, -10, 10)))
    confidence = np.exp(-sigma)

    return coh_s, confidence


def plot(pairs_dir: Path, out_dir: Path, sample_rate: int) -> None:
    pair_dirs = sorted(
        p for p in pairs_dir.iterdir()
        if p.is_dir() and (p / "log_var.tif").exists()
    )
    if not pair_dirs:
        print(f"No pairs with log_var.tif in {pairs_dir}")
        return

    print(f"Found {len(pair_dirs)} pairs, sample_rate=1/{sample_rate}")

    # ── Figure layout: density plot (all pairs combined) + per-pair overlay ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "FiLMUNet Confidence vs Interferometric Coherence\n"
        "(higher coherence → lower uncertainty → higher confidence)",
        fontsize=12
    )

    all_coh, all_conf = [], []
    pair_data = []

    for pd_dir in pair_dirs:
        result = sample_pair(pd_dir, sample_rate)
        if result is None:
            print(f"  SKIP {pd_dir.name}: missing files")
            continue
        coh, conf = result
        all_coh.append(coh)
        all_conf.append(conf)
        pair_data.append((pd_dir.name, coh, conf))
        print(f"  {pd_dir.name[:50]}: {len(coh):,} samples  "
              f"median_coh={np.median(coh):.3f}  median_conf={np.median(conf):.3f}")

    if not all_coh:
        print("No data to plot.")
        return

    all_coh  = np.concatenate(all_coh)
    all_conf = np.concatenate(all_conf)

    # Pearson + Spearman correlation
    r_p, _ = pearsonr(all_coh, all_conf)
    r_s, _ = spearmanr(all_coh, all_conf)

    # ── Panel 1: 2-D hexbin density (all pairs combined) ──────────────────────
    ax1 = axes[0]
    hb  = ax1.hexbin(all_coh, all_conf, gridsize=80, cmap="inferno",
                     mincnt=1, bins="log")
    plt.colorbar(hb, ax=ax1, label="log₁₀(count)")

    # Binned median line
    bins   = np.linspace(0, 1, 41)
    centers = 0.5 * (bins[:-1] + bins[1:])
    medians = [np.median(all_conf[(all_coh >= lo) & (all_coh < hi)])
               for lo, hi in zip(bins[:-1], bins[1:])]
    ax1.plot(centers, medians, "w-", lw=2, label="median")
    ax1.plot(centers, medians, "k--", lw=1)

    ax1.set_xlabel("Coherence", fontsize=11)
    ax1.set_ylabel("FiLMUNet Confidence  [exp(−σ̂)]", fontsize=11)
    ax1.set_title(f"All pairs combined  (n={len(all_coh):,})\n"
                  f"Pearson r={r_p:.3f}  |  Spearman ρ={r_s:.3f}", fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=9)

    # ── Panel 2: Per-pair colored scatter + marginal histograms ───────────────
    ax2 = axes[1]
    for i, (name, coh, conf) in enumerate(pair_data):
        color  = PAIR_COLORS[i % len(PAIR_COLORS)]
        ref_d  = name[:8]
        sec_d  = name.split("__")[1][:8] if "__" in name else name[:16]
        label  = f"{ref_d}→{sec_d}  (n={len(coh):,})"

        # Subsample further for scatter visibility
        step = max(1, len(coh) // 8000)
        ax2.scatter(coh[::step], conf[::step], s=1.5, alpha=0.4,
                    color=color, rasterized=True, label=label)

        # Per-pair median line
        medians_i = [np.median(conf[(coh >= lo) & (coh < hi)])
                     if ((coh >= lo) & (coh < hi)).sum() > 5 else np.nan
                     for lo, hi in zip(bins[:-1], bins[1:])]
        ax2.plot(centers, medians_i, "-", color=color, lw=1.8)

    ax2.set_xlabel("Coherence", fontsize=11)
    ax2.set_ylabel("FiLMUNet Confidence  [exp(−σ̂)]", fontsize=11)
    ax2.set_title("Per-pair breakdown", fontsize=10)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=7, markerscale=4, loc="upper left")

    # Reference lines
    for ax in axes:
        ax.axhline(0.5, color="gray", lw=0.8, ls=":")
        ax.axvline(0.35, color="gray", lw=0.8, ls=":",
                   label="coh=0.35 (usable threshold)")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "coherence_vs_confidence.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # ── Summary stats ──────────────────────────────────────────────────────────
    print(f"\nCorrelation (all pairs): Pearson r={r_p:.4f}  Spearman ρ={r_s:.4f}")
    high_coh  = all_conf[all_coh > 0.6]
    low_coh   = all_conf[all_coh < 0.2]
    print(f"Median confidence | coh>0.6: {np.median(high_coh):.4f}"
          f"  | coh<0.2: {np.median(low_coh):.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pairs_dir", required=True)
    p.add_argument("--out_dir",
                   default="experiments/enhanced/outputs/figures")
    p.add_argument("--sample_rate", type=int, default=500,
                   help="Keep 1 in N pixels (larger = faster, fewer points).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plot(Path(args.pairs_dir), ROOT / args.out_dir, args.sample_rate)


if __name__ == "__main__":
    main()
