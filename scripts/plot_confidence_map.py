#!/usr/bin/env python3
"""
Plot FiLMUNet confidence/uncertainty maps for processed InSAR pairs.

For each pair that has log_var.tif + ifg_film_unet.tif, generates a 4-panel figure:
  Panel 1 — Raw wrapped phase
  Panel 2 — Goldstein filtered phase
  Panel 3 — FiLMUNet denoised phase
  Panel 4 — FiLMUNet uncertainty (σ = sqrt(exp(log_var)), shown as confidence = exp(-σ))

Usage
-----
python scripts/plot_confidence_map.py \
    --pairs_dir data/processed/AOI024_pairs_full_image/AOI024_selected \
    --out_dir experiments/enhanced/outputs/figures/confidence_maps \
    [--tile_row 0 --tile_col 0 --tile_size 1024]   # optional spatial crop
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
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_raster(path: Path) -> np.ndarray:
    import rasterio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)


def _read_complex(path: Path) -> np.ndarray:
    import rasterio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32) + 1j * src.read(2).astype(np.float32)


def _crop(arr: np.ndarray, r0: int, c0: int, size: int) -> np.ndarray:
    r1 = min(r0 + size, arr.shape[0])
    c1 = min(c0 + size, arr.shape[1])
    return arr[r0:r1, c0:c1]


# ---------------------------------------------------------------------------
# Per-pair figure
# ---------------------------------------------------------------------------

def plot_pair(pair_dir: Path, out_dir: Path,
              tile_row: int, tile_col: int, tile_size: int) -> None:
    """Generate 4-panel confidence map figure for one pair."""

    raw_path  = pair_dir / "ifg_raw_complex_real_imag.tif"
    gold_path = pair_dir / "ifg_goldstein_complex_real_imag.tif"
    film_path = pair_dir / "ifg_film_unet.tif"
    lv_path   = pair_dir / "log_var.tif"
    coh_path  = pair_dir / "coherence.tif"

    for p in [gold_path, film_path, lv_path]:
        if not p.exists():
            print(f"  SKIP {pair_dir.name}: missing {p.name}")
            return

    ifg_raw  = _read_complex(raw_path) if raw_path.exists() else _read_complex(gold_path)
    ifg_gold = _read_complex(gold_path)
    ifg_film = _read_complex(film_path)
    log_var  = _read_raster(lv_path)
    coh      = _read_raster(coh_path) if coh_path.exists() else None

    # Spatial crop
    def crop(a):
        return _crop(a, tile_row, tile_col, tile_size)

    phase_raw  = crop(np.angle(ifg_raw))
    phase_gold = crop(np.angle(ifg_gold))
    phase_film = crop(np.angle(ifg_film))
    lv         = crop(log_var)
    coh_crop   = crop(coh) if coh is not None else None

    # Confidence = exp(-sigma) where sigma = sqrt(exp(log_var))
    # High confidence (bright) = low uncertainty
    sigma      = np.sqrt(np.exp(np.clip(lv, -10, 10)))
    confidence = np.exp(-sigma)

    # Clip percentiles for display
    lv_valid = lv[np.isfinite(lv)]
    lv_p2, lv_p98 = (float(np.percentile(lv_valid, 2)),
                     float(np.percentile(lv_valid, 98))) if len(lv_valid) else (-5, 0)

    pair_name = pair_dir.name
    ref_date  = pair_name[:8] if len(pair_name) >= 8 else pair_name
    sec_date  = pair_name.split("__")[1][:8] if "__" in pair_name else ""
    title_str = f"{ref_date} → {sec_date}" if sec_date else pair_name

    fig = plt.figure(figsize=(20, 5))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.05)

    panels = [
        (gs[0], phase_raw,  "hsv", -np.pi, np.pi, "Phase (rad)",
         f"Raw Wrapped Phase\n{title_str}"),
        (gs[1], phase_gold, "hsv", -np.pi, np.pi, "Phase (rad)",
         "Goldstein Filtered Phase"),
        (gs[2], phase_film, "hsv", -np.pi, np.pi, "Phase (rad)",
         "FiLMUNet Denoised Phase"),
        (gs[3], confidence, "viridis", 0.0, 1.0, "Confidence",
         "FiLMUNet Confidence Map\n(1 − σ̂,  bright = certain)"),
    ]

    for spec, data, cmap, vmin, vmax, clabel, ptitle in panels:
        ax  = fig.add_subplot(spec)
        im  = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                        interpolation="nearest", aspect="auto")
        ax.set_title(ptitle, fontsize=9, pad=4)
        ax.set_xlabel("Range (pixels)", fontsize=8)
        ax.set_ylabel("Azimuth (pixels)", fontsize=8)
        ax.tick_params(labelsize=7)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="4%", pad=0.05)
        cb  = fig.colorbar(im, cax=cax)
        cb.set_label(clabel, fontsize=7)
        cb.ax.tick_params(labelsize=6)

    # Inset: coherence overlay on confidence panel if available
    if coh_crop is not None:
        ax_last = fig.axes[3]
        ax_ins  = ax_last.inset_axes([0.65, 0.65, 0.33, 0.33])
        ax_ins.imshow(coh_crop, cmap="gray", vmin=0, vmax=1, aspect="auto")
        ax_ins.set_title("Coherence", fontsize=6)
        ax_ins.axis("off")

    fig.suptitle(
        f"FiLMUNet Phase Denoising & Uncertainty  |  {pair_dir.parent.name}",
        fontsize=11, y=1.01
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pair_dir.name[:80]
    out_path = out_dir / f"confidence_map_{stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot FiLMUNet confidence maps for processed InSAR pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pairs_dir", required=True,
                   help="Directory containing processed pair subdirectories.")
    p.add_argument("--out_dir",
                   default="experiments/enhanced/outputs/figures/confidence_maps",
                   help="Output directory for figures.")
    p.add_argument("--tile_row",  type=int, default=0,
                   help="Top-left row for spatial crop (0 = start of image).")
    p.add_argument("--tile_col",  type=int, default=0,
                   help="Top-left col for spatial crop (0 = start of image).")
    p.add_argument("--tile_size", type=int, default=1024,
                   help="Crop size in pixels (square). Use 0 for full image.")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    pairs_dir = Path(args.pairs_dir)
    out_dir   = ROOT / args.out_dir

    pair_dirs = sorted(
        p for p in pairs_dir.iterdir()
        if p.is_dir() and (p / "log_var.tif").exists()
    )
    if not pair_dirs:
        print(f"No pairs with log_var.tif found in {pairs_dir}")
        return

    tile_size = args.tile_size if args.tile_size > 0 else 999_999
    print(f"Found {len(pair_dirs)} pairs with log_var.tif")

    for pd in pair_dirs:
        print(f"Processing: {pd.name[:60]}...")
        plot_pair(pd, out_dir, args.tile_row, args.tile_col, tile_size)

    print(f"\nDone. Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
