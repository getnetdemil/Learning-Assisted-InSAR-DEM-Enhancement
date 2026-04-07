#!/usr/bin/env python3
"""
Weighted multi-baseline DEM inversion from InSAR unwrapped phase stack.

For each pixel, solves:
    h*(x,y) = Σ_p(w_p · s_p · φ_p(x,y)) / Σ_p(w_p · s_p²)

where s_p = 2π / h_amb_p  is the height sensitivity [rad/m],
and weights are:
  goldstein  → coherence(x,y)
  film_unet  → confidence = exp(-sqrt(exp(log_var(x,y))))

This is NOT SBAS displacement inversion — it inverts a single static DEM
(one height unknown per pixel) from a multi-baseline pair stack.

Outputs
-------
  {out_dir}/dem_goldstein.tif    — Goldstein multi-baseline DEM (float32 GeoTIFF)
  {out_dir}/dem_filmunet.tif     — FiLMUNet multi-baseline DEM (float32 GeoTIFF)
  {out_dir}/sbas_dem_comparison.png — 3-panel comparison figure

Usage
-----
python scripts/sbas_dem.py \\
    --pairs_dir data/processed/AOI024_pairs_full_image/AOI024_selected \\
    --out_dir experiments/enhanced/outputs/sbas_dem_aoi024 \\
    --copernicus_dem_dir data/reference/copernicus_dem \\
    --aoi AOI024
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_CAPELLA_ALTITUDE_M = 525_000.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _height_of_ambiguity(bperp_m: float, incidence_deg: float,
                          center_freq_ghz: float) -> float:
    """h_amb [m] — height that causes 2π phase cycle (flat-Earth approx)."""
    if abs(bperp_m) < 10.0:
        return float("nan")
    lam   = 3e8 / (center_freq_ghz * 1e9)
    theta = np.radians(incidence_deg)
    R     = _CAPELLA_ALTITUDE_M / np.cos(theta)
    return lam * R * np.sin(theta) / (2.0 * abs(bperp_m))


def _load_scene_index() -> dict:
    """Load full_index.parquet keyed by scene id."""
    manifest = ROOT / "data" / "manifests" / "full_index.parquet"
    if not manifest.exists():
        print(f"WARNING: {manifest} not found — h_amb cannot be computed.")
        return {}
    df = pd.read_parquet(manifest)
    return {row["id"]: row for _, row in df.iterrows()}


def _detrend_plane(arr: np.ndarray) -> np.ndarray:
    """Subtract least-squares planar fit (removes flat-earth ramp)."""
    rows, cols = np.indices(arr.shape)
    valid = np.isfinite(arr)
    if valid.sum() < 10:
        return arr
    A = np.column_stack([rows[valid].ravel(), cols[valid].ravel(),
                         np.ones(valid.sum())])
    b = arr[valid].ravel()
    try:
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        plane = coeffs[0] * rows + coeffs[1] * cols + coeffs[2]
        return arr - plane
    except Exception:
        return arr


def _nmad(err: np.ndarray) -> float:
    """Normalised Median Absolute Deviation."""
    valid = err[np.isfinite(err)]
    if len(valid) < 10:
        return float("nan")
    return float(1.4826 * np.median(np.abs(valid - np.median(valid))))


def _load_copernicus_median(dem_path: Path, bbox_w: float, bbox_s: float,
                             bbox_e: float, bbox_n: float) -> Optional[float]:
    """Return median Copernicus elevation over a scene bbox, or None."""
    try:
        import rasterio
        from rasterio.windows import from_bounds
        with rasterio.open(dem_path) as src:
            window = from_bounds(bbox_w, bbox_s, bbox_e, bbox_n, src.transform)
            data   = src.read(1, window=window).astype(np.float32)
            nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        val = float(np.nanmedian(data))
        return val if np.isfinite(val) else None
    except Exception as e:
        print(f"  WARNING: Copernicus read error: {e}")
        return None


# ---------------------------------------------------------------------------
# Core inversion
# ---------------------------------------------------------------------------

def _load_pairs_meta(pair_dirs: list[Path], scene_index: dict) -> list[dict]:
    """Extract geometry metadata for each pair dir."""
    result = []
    for pd_dir in pair_dirs:
        meta_path = pd_dir / "coreg_meta.json"
        if not meta_path.exists():
            result.append(None)
            continue
        with open(meta_path) as f:
            m = json.load(f)
        id_ref = m.get("id_ref", "")
        scene  = scene_index.get(id_ref)
        result.append({
            "id_ref":        id_ref,
            "bperp_m":       float(m.get("bperp_m", 0.0)),
            "incidence_deg": float(m.get("incidence_angle_deg", 45.0)),
            "center_freq":   float(scene["center_freq_ghz"]) if scene is not None else 9.6,
            "bbox_w":        float(scene["bbox_w"]) if scene is not None else None,
            "bbox_s":        float(scene["bbox_s"]) if scene is not None else None,
            "bbox_e":        float(scene["bbox_e"]) if scene is not None else None,
            "bbox_n":        float(scene["bbox_n"]) if scene is not None else None,
        })
    return result


def invert_dem(pair_dirs: list[Path], meta_list: list[dict],
               method: str) -> tuple[Optional[np.ndarray], Optional[dict]]:
    """
    Weighted multi-baseline DEM inversion.

    Returns (h_star [H,W] float32, rasterio profile) or (None, None).
    """
    import rasterio

    num: Optional[np.ndarray] = None
    den: Optional[np.ndarray] = None
    profile: Optional[dict]   = None
    n_pairs = 0

    for pd_dir, meta in zip(pair_dirs, meta_list):
        if meta is None:
            continue

        unw_file = "unw_phase.tif" if method == "goldstein" else "unw_phase_film_unet.tif"
        unw_path = pd_dir / unw_file
        if not unw_path.exists():
            print(f"  SKIP {pd_dir.name[:50]}: {unw_file} missing")
            continue

        h_amb = _height_of_ambiguity(meta["bperp_m"], meta["incidence_deg"],
                                      meta["center_freq"])
        if not np.isfinite(h_amb):
            print(f"  SKIP {pd_dir.name[:50]}: h_amb invalid "
                  f"(bperp={meta['bperp_m']:.1f}m)")
            continue

        s_p = 2.0 * np.pi / h_amb  # sensitivity [rad/m]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(unw_path) as src:
                phi = src.read(1).astype(np.float64)
                if profile is None:
                    profile = src.profile.copy()

        # Weight image
        if method == "goldstein":
            coh_path = pd_dir / "coherence.tif"
            if not coh_path.exists():
                w = np.ones_like(phi)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with rasterio.open(coh_path) as src:
                        w = src.read(1).astype(np.float64)
            w = np.clip(w, 0.0, 1.0)
        else:
            lv_path = pd_dir / "log_var.tif"
            if not lv_path.exists():
                w = np.ones_like(phi)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with rasterio.open(lv_path) as src:
                        lv = src.read(1).astype(np.float64)
                sigma = np.sqrt(np.exp(np.clip(lv, -10, 10)))
                w = np.exp(-sigma)

        if num is None:
            num = np.zeros_like(phi)
            den = np.zeros_like(phi)

        valid = np.isfinite(phi) & np.isfinite(w) & (w > 0.01)
        num[valid] += w[valid] * s_p * phi[valid]
        den[valid] += w[valid] * (s_p ** 2)
        n_pairs += 1

        print(f"  [{method}] {pd_dir.name[:55]}")
        print(f"    B_perp={meta['bperp_m']:+.0f}m  h_amb={h_amb:.1f}m  "
              f"s={s_p:.4f} rad/m  valid_px={valid.sum():,}")

    if num is None or n_pairs == 0:
        print(f"  No valid pairs for method={method}.")
        return None, None

    h_star = np.where(den > 1e-10, num / den, np.nan).astype(np.float32)
    h_star = _detrend_plane(h_star)

    print(f"  {method}: {n_pairs} pairs used  "
          f"median_h={np.nanmedian(h_star):.1f}m  "
          f"std_h={np.nanstd(h_star):.1f}m")
    return h_star, profile


# ---------------------------------------------------------------------------
# Save DEM
# ---------------------------------------------------------------------------

def save_dem(h: np.ndarray, profile: dict, out_path: Path) -> None:
    import rasterio
    profile = profile.copy()
    profile.update({"count": 1, "dtype": "float32", "compress": "deflate",
                    "BIGTIFF": "YES"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(h[np.newaxis, :, :])
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# NMAD vs Copernicus
# ---------------------------------------------------------------------------

def compute_nmad_vs_copernicus(h: np.ndarray, meta_list: list[dict],
                                dem_path: Path) -> float:
    """Compute NMAD of (h_insar - h_copernicus_median) over all valid pixels."""
    # Collect reference elevations from all pairs' bboxes
    ref_vals = []
    for meta in meta_list:
        if meta is None or meta["bbox_w"] is None:
            continue
        ref = _load_copernicus_median(dem_path,
                                      meta["bbox_w"], meta["bbox_s"],
                                      meta["bbox_e"], meta["bbox_n"])
        if ref is not None:
            ref_vals.append(ref)
    if not ref_vals:
        return float("nan")
    h_ref = float(np.median(ref_vals))
    err = h[np.isfinite(h)] - h_ref
    return _nmad(err)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_comparison(h_gold: Optional[np.ndarray],
                    h_film: Optional[np.ndarray],
                    nmad_gold: float,
                    nmad_film: float,
                    out_path: Path,
                    pairs_dir_name: str) -> None:
    """3-panel comparison: Goldstein DEM | FiLMUNet DEM | Difference."""

    def _pct(a: np.ndarray, lo=2, hi=98):
        v = a[np.isfinite(a)]
        if len(v) == 0:
            return 0.0, 1.0
        return float(np.percentile(v, lo)), float(np.percentile(v, hi))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Multi-Baseline DEM Inversion — {pairs_dir_name}\n"
        "(height relative to median terrain; flat-Earth ramp removed)",
        fontsize=11
    )

    # Panel 1: Goldstein DEM
    ax1 = axes[0]
    if h_gold is not None:
        vmin, vmax = _pct(h_gold)
        im1 = ax1.imshow(h_gold, cmap="terrain", vmin=vmin, vmax=vmax,
                         aspect="auto", interpolation="nearest")
        nmad_str = f"NMAD={nmad_gold:.1f}m" if np.isfinite(nmad_gold) else "NMAD=N/A"
        ax1.set_title(f"Goldstein DEM\n{nmad_str}", fontsize=10)
        plt.colorbar(im1, ax=ax1, label="Height (m)", fraction=0.046, pad=0.04)
    else:
        ax1.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=ax1.transAxes)
        ax1.set_title("Goldstein DEM\n(no unw_phase.tif)", fontsize=10)

    # Panel 2: FiLMUNet DEM
    ax2 = axes[1]
    if h_film is not None:
        vmin, vmax = _pct(h_film)
        im2 = ax2.imshow(h_film, cmap="terrain", vmin=vmin, vmax=vmax,
                         aspect="auto", interpolation="nearest")
        nmad_str = f"NMAD={nmad_film:.1f}m" if np.isfinite(nmad_film) else "NMAD=N/A"
        ax2.set_title(f"FiLMUNet DEM\n{nmad_str}", fontsize=10)
        plt.colorbar(im2, ax=ax2, label="Height (m)", fraction=0.046, pad=0.04)
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=ax2.transAxes)
        ax2.set_title("FiLMUNet DEM\n(no unw_phase_film_unet.tif)", fontsize=10)

    # Panel 3: Difference FiLMUNet - Goldstein
    ax3 = axes[2]
    if h_gold is not None and h_film is not None:
        # Align shapes if needed (take minimum overlap)
        min_h = min(h_gold.shape[0], h_film.shape[0])
        min_w = min(h_gold.shape[1], h_film.shape[1])
        diff = h_film[:min_h, :min_w] - h_gold[:min_h, :min_w]
        vmax_d = float(np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 95)) if np.any(np.isfinite(diff)) else 50.0
        im3 = ax3.imshow(diff, cmap="RdBu_r", vmin=-vmax_d, vmax=vmax_d,
                         aspect="auto", interpolation="nearest")
        ax3.set_title("Difference\n(FiLMUNet − Goldstein)", fontsize=10)
        plt.colorbar(im3, ax=ax3, label="ΔHeight (m)", fraction=0.046, pad=0.04)
        # Print improvement
        if np.isfinite(nmad_gold) and np.isfinite(nmad_film):
            pct = (nmad_film - nmad_gold) / nmad_gold * 100
            ax3.set_xlabel(f"NMAD improvement: {pct:+.1f}%", fontsize=9)
    else:
        ax3.text(0.5, 0.5, "Need both methods", ha="center", va="center",
                 transform=ax3.transAxes)
        ax3.set_title("Difference (N/A)", fontsize=10)

    for ax in axes:
        ax.set_xlabel("Range (pixels)", fontsize=8)
        ax.set_ylabel("Azimuth (pixels)", fontsize=8)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Weighted multi-baseline DEM inversion from InSAR phase stack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pairs_dir", required=True,
                   help="Directory containing processed pair subdirectories.")
    p.add_argument("--out_dir",
                   default="experiments/enhanced/outputs/sbas_dem",
                   help="Output directory for DEMs and figure.")
    p.add_argument("--method", default="both",
                   choices=["goldstein", "film_unet", "both"],
                   help="Which method(s) to invert.")
    p.add_argument("--copernicus_dem_dir", default=None,
                   help="Directory containing merged Copernicus DEM for NMAD.")
    p.add_argument("--aoi", default=None,
                   help="AOI code (e.g. AOI024) → derives DEM filename aoi024_dem.tif.")
    p.add_argument("--max_pairs", type=int, default=None,
                   help="Limit to N pairs (pairs with both gold+film prioritised).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pairs_dir = Path(args.pairs_dir)
    out_dir   = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover pairs — prefer those that have BOTH files for a fair comparison
    all_dirs = sorted(p for p in pairs_dir.iterdir() if p.is_dir())
    both_dirs = [p for p in all_dirs
                 if (p / "unw_phase.tif").exists() and
                    (p / "unw_phase_film_unet.tif").exists()]
    gold_only = [p for p in all_dirs
                 if (p / "unw_phase.tif").exists() and p not in both_dirs]

    # Use pairs-with-both first, then gold-only to fill up to max_pairs
    if args.max_pairs is not None:
        pair_dirs = (both_dirs + gold_only)[:args.max_pairs]
    else:
        pair_dirs = both_dirs + gold_only

    if not pair_dirs:
        print(f"No pairs with unw_phase.tif found in {pairs_dir}")
        return

    print(f"Using {len(pair_dirs)} pairs "
          f"({len([p for p in pair_dirs if p in both_dirs])} with both gold+film, "
          f"{len([p for p in pair_dirs if p not in both_dirs])} gold-only)")

    print(f"Found {len(pair_dirs)} pairs with unwrapped phase")

    scene_index = _load_scene_index()
    meta_list   = _load_pairs_meta(pair_dirs, scene_index)

    # DEM filename for Copernicus lookup
    if args.aoi and args.aoi.upper() != "AOI000":
        dem_filename = f"{args.aoi.lower()}_dem.tif"
    else:
        dem_filename = "hawaii_dem.tif"
    dem_path = (Path(args.copernicus_dem_dir) / dem_filename
                if args.copernicus_dem_dir else None)

    methods = (["goldstein", "film_unet"] if args.method == "both"
               else [args.method])

    h_gold, h_film = None, None
    nmad_gold, nmad_film = float("nan"), float("nan")

    for method in methods:
        print(f"\n=== Inverting {method} ===")
        h, profile = invert_dem(pair_dirs, meta_list, method)
        if h is None:
            continue

        out_tif = out_dir / f"dem_{method}.tif"
        save_dem(h, profile, out_tif)

        # NMAD vs Copernicus
        if dem_path is not None and dem_path.exists():
            nmad_val = compute_nmad_vs_copernicus(h, meta_list, dem_path)
            print(f"  NMAD vs Copernicus ({dem_filename}): {nmad_val:.2f} m")
        else:
            nmad_val = float("nan")
            if args.copernicus_dem_dir:
                print(f"  WARNING: {dem_path} not found — NMAD skipped")

        if method == "goldstein":
            h_gold, nmad_gold = h, nmad_val
        else:
            h_film, nmad_film = h, nmad_val

    # Figure
    print("\n=== Generating comparison figure ===")
    plot_comparison(
        h_gold, h_film, nmad_gold, nmad_film,
        out_dir / "sbas_dem_comparison.png",
        pairs_dir.name,
    )

    print(f"\nDone. Outputs in: {out_dir}/")
    print(f"  dem_goldstein.tif   NMAD={nmad_gold:.2f}m")
    print(f"  dem_filmunet.tif    NMAD={nmad_film:.2f}m")
    if np.isfinite(nmad_gold) and np.isfinite(nmad_film):
        pct = (nmad_film - nmad_gold) / nmad_gold * 100
        print(f"  NMAD improvement: {pct:+.1f}%")


if __name__ == "__main__":
    main()
