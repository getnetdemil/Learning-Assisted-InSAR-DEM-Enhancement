#!/usr/bin/env python3
"""
DEM NMAD comparison visualization: per-pair NMAD histogram + scatter for
Goldstein vs FiLMUNet.

Output: experiments/enhanced/outputs/figures/dem_nmad_comparison.png

Usage
-----
python scripts/plot_dem_nmad.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAIRS_DIR  = ROOT / "data" / "processed" / "pairs"
DEM_PATH   = ROOT / "data" / "reference" / "copernicus_dem" / "hawaii_dem.tif"
INDEX_PATH = ROOT / "data" / "manifests" / "full_index.parquet"
OUT_PATH   = ROOT / "experiments" / "enhanced" / "outputs" / "figures" / "dem_nmad_comparison.png"

CAPELLA_ALTITUDE_M = 525_000.0

GOLD_COLOR = "#4878CF"
FILM_COLOR = "#E87060"


# ── helpers (mirror compute_metrics.py M4 logic) ──────────────────────────

def _h_amb(bperp_m: float, incidence_deg: float, center_freq_ghz: float) -> float:
    if abs(bperp_m) < 10.0:
        return float("nan")
    wavelength_m = 3e8 / (center_freq_ghz * 1e9)
    theta = np.radians(incidence_deg)
    R = CAPELLA_ALTITUDE_M / np.cos(theta)
    return wavelength_m * R * np.sin(theta) / (2.0 * abs(bperp_m))


def _detrend(arr: np.ndarray) -> np.ndarray:
    rows, cols = np.indices(arr.shape)
    valid = np.isfinite(arr)
    if valid.sum() < 10:
        return arr
    A = np.column_stack([rows[valid].ravel(), cols[valid].ravel(), np.ones(valid.sum())])
    b = arr[valid].ravel()
    try:
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        plane = coeffs[0] * rows + coeffs[1] * cols + coeffs[2]
        return arr - plane
    except Exception:
        return arr


def _copernicus_patch(bbox_w: float, bbox_s: float,
                      bbox_e: float, bbox_n: float) -> np.ndarray | None:
    import rasterio
    from rasterio.windows import from_bounds
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with rasterio.open(DEM_PATH) as src:
                win = from_bounds(bbox_w, bbox_s, bbox_e, bbox_n, src.transform)
                data = src.read(1, window=win).astype(np.float32)
                nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = float("nan")
            return data if data.size > 0 else None
        except Exception:
            return None


def _pair_nmad(pair_dir: Path, unw_file: str, scene_index: dict
               ) -> tuple[float, float]:
    """Return (nmad_m, |bperp|) or (nan, nan) if pair is invalid/skipped."""
    import rasterio
    meta_path = pair_dir / "coreg_meta.json"
    unw_path  = pair_dir / unw_file
    if not meta_path.exists() or not unw_path.exists():
        return float("nan"), float("nan")

    with open(meta_path) as f:
        meta = json.load(f)

    bperp_m       = meta.get("bperp_m", 0.0)
    incidence_deg = meta.get("incidence_angle_deg", 45.0)
    id_ref        = meta.get("id_ref", "")

    scene = scene_index.get(id_ref)
    if scene is None:
        return float("nan"), float("nan")

    h_amb = _h_amb(bperp_m, incidence_deg, float(scene["center_freq_ghz"]))
    if not np.isfinite(h_amb):
        return float("nan"), float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with rasterio.open(unw_path) as src:
                unw = src.read(1).astype(np.float32)
        except Exception:
            return float("nan"), float("nan")

    h_insar = unw * h_amb / (2.0 * np.pi)
    h_insar[~np.isfinite(h_insar)] = float("nan")
    h_insar = _detrend(h_insar)

    h_ref = _copernicus_patch(
        float(scene["bbox_w"]), float(scene["bbox_s"]),
        float(scene["bbox_e"]), float(scene["bbox_n"]),
    )
    if h_ref is None or not np.any(np.isfinite(h_ref)):
        return float("nan"), float("nan")

    h_ref_median = float(np.nanmedian(h_ref))
    valid = np.isfinite(h_insar)
    if valid.sum() < 100:
        return float("nan"), float("nan")

    e = h_insar[valid] - h_ref_median
    nmad_val = float(1.4826 * np.median(np.abs(e - np.median(e))))
    return nmad_val, abs(bperp_m)


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading scene index ...", flush=True)
    df = pd.read_parquet(INDEX_PATH)
    scene_index = {row["id"]: row for _, row in df.iterrows()}

    pair_dirs = sorted(p for p in PAIRS_DIR.iterdir() if p.is_dir())
    print(f"Processing {len(pair_dirs)} pairs ...", flush=True)

    gold_nmads: list[float] = []
    film_nmads: list[float] = []
    bperps:     list[float] = []

    for i, pd_dir in enumerate(pair_dirs):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(pair_dirs)}]", flush=True)
        g, bp = _pair_nmad(pd_dir, "unw_phase.tif", scene_index)
        f, _  = _pair_nmad(pd_dir, "unw_phase_film_unet.tif", scene_index)
        if np.isfinite(g) and np.isfinite(f):
            gold_nmads.append(g)
            film_nmads.append(f)
            bperps.append(bp)

    gold = np.array(gold_nmads)
    film = np.array(film_nmads)
    bp   = np.array(bperps)

    n_valid = len(gold)
    print(f"Valid pairs: {n_valid}", flush=True)
    print(f"Goldstein  mean NMAD = {gold.mean():.3f} m", flush=True)
    print(f"FiLMUNet   mean NMAD = {film.mean():.3f} m", flush=True)
    print(f"Improvement = {(film.mean()/gold.mean() - 1)*100:+.1f}%", flush=True)
    frac_better = (film < gold).mean()
    print(f"FiLMUNet better in {frac_better*100:.0f}% of pairs", flush=True)

    # ── Figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.patch.set_facecolor("white")

    # ── Panel 1: Histogram ───────────────────────────────────────────────
    ax = axes[0]
    vmax = np.percentile(np.concatenate([gold, film]), 97)
    bins = np.linspace(0, vmax * 1.05, 30)

    ax.hist(gold, bins=bins, alpha=0.6, color=GOLD_COLOR,
            label=f"Goldstein (mean\u202f=\u202f{gold.mean():.1f}\u202fm)", density=True)
    ax.hist(film, bins=bins, alpha=0.6, color=FILM_COLOR,
            label=f"FiLMUNet  (mean\u202f=\u202f{film.mean():.1f}\u202fm)", density=True)
    ax.axvline(gold.mean(), color=GOLD_COLOR, linestyle="--", linewidth=1.5)
    ax.axvline(film.mean(), color=FILM_COLOR, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Per-pair DEM NMAD (m)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Distribution across pairs", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel 2: Scatter Goldstein vs FiLMUNet ───────────────────────────
    ax = axes[1]
    sc = ax.scatter(gold, film, c=bp, cmap="viridis",
                    alpha=0.65, s=18, edgecolors="none",
                    vmin=bp.min(), vmax=np.percentile(bp, 95))
    lim = vmax * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.6, label="y = x (no change)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Goldstein NMAD (m)", fontsize=11)
    ax.set_ylabel("FiLMUNet NMAD (m)", fontsize=11)
    ax.set_title("Per-pair comparison", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")

    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$|B_\perp|$ (m)", fontsize=10)

    ax.text(0.97, 0.05,
            f"FiLMUNet better: {frac_better*100:.0f}% of pairs\n"
            f"Mean NMAD: {gold.mean():.1f}\u2192{film.mean():.1f}\u202fm ({(film.mean()/gold.mean()-1)*100:+.1f}%)",
            transform=ax.transAxes, fontsize=8.5, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#cccccc"))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "DEM NMAD: Goldstein vs. FiLMUNet \u2014 AOI_000 Hawaii (224 pairs)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved → {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
