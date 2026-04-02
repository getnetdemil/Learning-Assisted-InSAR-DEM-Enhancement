#!/usr/bin/env python3
"""
Copernicus DEM vs InSAR-derived height comparison figure.

Layout (2 rows × 3 cols):
  Row 0 — Geographic reference (Copernicus GLO-30)
    (0,0) Hawaii overview + scene footprints
    (0,1) Copernicus DEM zoom: best pair bbox  (~10×10 km)
    (0,2) Elevation histogram: Copernicus vs Goldstein vs FiLMUNet (zero-mean)

  Row 1 — InSAR products for best pair (SAR slant-range geometry)
    (1,0) Goldstein unwrapped phase
    (1,1) Goldstein InSAR height  h = unw × h_amb / (2π),  after plane detrend
    (1,2) FiLMUNet InSAR height   (same formula)

NOTE: Row 1 is in SAR slant-range geometry; Row 0 is geographic.
      The elevation histogram in (0,2) is the valid cross-system comparison.

Best pair: |B_perp| = 780 m  →  h_amb ≈ 12 m  (highest DEM sensitivity in the dataset)

Output: experiments/enhanced/outputs/figures/dem_comparison.png
"""

from __future__ import annotations
import json, sys, warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.insar_processing.geometry import load_extended_meta, geocode_patch_corners

PAIRS_DIR  = ROOT / "data" / "processed" / "pairs"
DEM_PATH   = ROOT / "data" / "reference" / "copernicus_dem" / "hawaii_dem.tif"
INDEX_PATH = ROOT / "data" / "manifests" / "full_index.parquet"
OUT_PATH   = ROOT / "experiments" / "enhanced" / "outputs" / "figures" / "dem_comparison.png"

CAPELLA_ALTITUDE_M = 525_000.0
GOLD_COLOR = "#4878CF"
FILM_COLOR = "#E87060"
COPER_COLOR = "#2CA02C"


# ── helpers ──────────────────────────────────────────────────────────────

def _h_amb(bperp_m, incidence_deg, center_freq_ghz):
    lam   = 3e8 / (center_freq_ghz * 1e9)
    theta = np.radians(incidence_deg)
    R     = CAPELLA_ALTITUDE_M / np.cos(theta)
    return lam * R * np.sin(theta) / (2.0 * abs(bperp_m))


def _detrend(arr):
    rows, cols = np.indices(arr.shape)
    valid = np.isfinite(arr)
    if valid.sum() < 10:
        return arr
    A = np.column_stack([rows[valid].ravel(), cols[valid].ravel(), np.ones(valid.sum())])
    try:
        c, *_ = np.linalg.lstsq(A, arr[valid].ravel(), rcond=None)
        return arr - (c[0]*rows + c[1]*cols + c[2])
    except Exception:
        return arr


def _read_raster(path):
    import rasterio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)


def _copernicus_patch(bbox_w, bbox_s, bbox_e, bbox_n):
    import rasterio
    from rasterio.windows import from_bounds
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(DEM_PATH) as src:
            win  = from_bounds(bbox_w, bbox_s, bbox_e, bbox_n, src.transform)
            data = src.read(1, window=win).astype(np.float32)
            nd   = src.nodata
            ext  = [bbox_w, bbox_e, bbox_s, bbox_n]   # for imshow extent
        if nd is not None:
            data[data == nd] = float("nan")
        return data, ext


def _copernicus_overview(scale=20):
    """Return downsampled hawaii_dem array + geographic extent."""
    import rasterio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(DEM_PATH) as src:
            data = src.read(1, out_shape=(src.count,
                                          src.height // scale,
                                          src.width  // scale)).astype(np.float32)
            ext  = [src.bounds.left, src.bounds.right,
                    src.bounds.bottom, src.bounds.top]
    data[data < -500] = float("nan")
    return data, ext


# ── main ──────────────────────────────────────────────────────────────────

def main():
    # ── identify best pair ───────────────────────────────────────────────
    print("Finding best pair (max |B_perp|) ...", flush=True)
    df = pd.read_parquet(INDEX_PATH)
    scene_idx = {r["id"]: r for _, r in df.iterrows()}

    pair_dirs = sorted(p for p in PAIRS_DIR.iterdir() if p.is_dir())
    best_pd, best_bp = None, 0.0
    for pd_dir in pair_dirs:
        meta_p = pd_dir / "coreg_meta.json"
        if not meta_p.exists():
            continue
        m  = json.load(open(meta_p))
        bp = abs(m.get("bperp_m", 0.0))
        if bp > best_bp:
            # check both unw files exist
            if (pd_dir/"unw_phase.tif").exists() and \
               (pd_dir/"unw_phase_film_unet.tif").exists():
                best_bp = bp
                best_pd = pd_dir
                best_meta = m

    assert best_pd is not None, "No valid pair found"
    bperp_m       = best_meta["bperp_m"]
    incidence_deg = best_meta["incidence_angle_deg"]
    id_ref        = best_meta["id_ref"]
    scene         = scene_idx[id_ref]
    freq_ghz      = float(scene["center_freq_ghz"])
    bbox_w, bbox_s = float(scene["bbox_w"]), float(scene["bbox_s"])
    bbox_e, bbox_n = float(scene["bbox_e"]), float(scene["bbox_n"])
    ha = _h_amb(bperp_m, incidence_deg, freq_ghz)

    print(f"Best pair: {best_pd.name[:60]}", flush=True)
    print(f"  |B_perp|={best_bp:.1f} m  h_amb={ha:.1f} m  inc={incidence_deg}°", flush=True)
    print(f"  bbox: ({bbox_w:.3f}, {bbox_s:.3f}) → ({bbox_e:.3f}, {bbox_n:.3f})", flush=True)

    # ── load data ────────────────────────────────────────────────────────
    print("Loading rasters ...", flush=True)
    unw_gold = _read_raster(best_pd / "unw_phase.tif")
    unw_film = _read_raster(best_pd / "unw_phase_film_unet.tif")

    h_gold = _detrend(unw_gold * ha / (2.0 * np.pi))
    h_film = _detrend(unw_film * ha / (2.0 * np.pi))
    h_gold[~np.isfinite(h_gold)] = float("nan")
    h_film[~np.isfinite(h_film)] = float("nan")

    cop_patch, cop_ext = _copernicus_patch(bbox_w, bbox_s, bbox_e, bbox_n)
    cop_overview, ov_ext = _copernicus_overview(scale=20)

    # ── Geocode SAR patch corners for Row 1 geographic display ───────────
    _ext_json = ROOT / "data" / "raw" / "AOI_000" / id_ref / f"{id_ref}_extended.json"
    _sar_ext  = None
    if _ext_json.exists():
        try:
            _ext_meta  = load_extended_meta(_ext_json)
            _h_terrain = float(np.nanmedian(cop_patch[np.isfinite(cop_patch)])) \
                         if np.any(np.isfinite(cop_patch)) else 300.0
            _corners   = geocode_patch_corners(
                _ext_meta,
                patch_row        = int(best_meta["patch_row_ref"]),
                patch_col        = int(best_meta["patch_col_ref"]),
                patch_size       = int(best_meta.get("patch_size", 4096)),
                terrain_height_m = _h_terrain,
            )
            _sar_ext = [float(_corners[:, 1].min()), float(_corners[:, 1].max()),
                        float(_corners[:, 0].min()), float(_corners[:, 0].max())]
            print(f"  SAR geocoded extent: lon [{_sar_ext[0]:.4f}, {_sar_ext[1]:.4f}]  "
                  f"lat [{_sar_ext[2]:.4f}, {_sar_ext[3]:.4f}]", flush=True)
        except Exception as _e:
            print(f"  WARNING: geocoding failed ({_e}); falling back to pixel coords",
                  flush=True)

    # ── all pair footprints for overview ─────────────────────────────────
    footprints = []
    seen_bboxes = set()
    for pd_dir in pair_dirs:
        meta_p = pd_dir / "coreg_meta.json"
        if not meta_p.exists():
            continue
        idr = json.load(open(meta_p)).get("id_ref", "")
        sc  = scene_idx.get(idr)
        if sc is None:
            continue
        key = (round(float(sc["bbox_w"]), 2), round(float(sc["bbox_s"]), 2))
        if key not in seen_bboxes:
            seen_bboxes.add(key)
            footprints.append((float(sc["bbox_w"]), float(sc["bbox_s"]),
                               float(sc["bbox_e"]), float(sc["bbox_n"])))

    # ── figure ────────────────────────────────────────────────────────────
    print("Building figure ...", flush=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.patch.set_facecolor("white")

    # ── (0,0) Hawaii DEM overview + footprints ───────────────────────────
    ax = axes[0, 0]
    vm = np.nanpercentile(cop_overview, 98)
    im = ax.imshow(cop_overview, extent=ov_ext, origin="upper",
                   cmap="terrain", vmin=0, vmax=vm, aspect="auto")
    plt.colorbar(im, ax=ax, label="Elev. (m)", fraction=0.035, pad=0.02)
    for (w, s, e, n) in footprints:
        rect = mpatches.Rectangle((w, s), e-w, n-s,
                                   lw=0.4, edgecolor="#FF4444",
                                   facecolor="none", alpha=0.5)
        ax.add_patch(rect)
    # highlight best pair
    rect = mpatches.Rectangle((bbox_w, bbox_s), bbox_e-bbox_w, bbox_n-bbox_s,
                               lw=1.5, edgecolor="yellow",
                               facecolor="yellow", alpha=0.25)
    ax.add_patch(rect)
    rect2 = mpatches.Rectangle((bbox_w, bbox_s), bbox_e-bbox_w, bbox_n-bbox_s,
                                lw=1.5, edgecolor="yellow", facecolor="none")
    ax.add_patch(rect2)
    ax.set_xlabel("Longitude (°)", fontsize=9)
    ax.set_ylabel("Latitude (°)", fontsize=9)
    ax.set_title("Copernicus GLO-30 — Hawaii\n(red = scene footprints, yellow = best pair)",
                 fontsize=9)

    # ── (0,1) Copernicus zoom: best pair bbox ────────────────────────────
    ax = axes[0, 1]
    vm2 = np.nanpercentile(cop_patch[np.isfinite(cop_patch)], 98) if np.any(np.isfinite(cop_patch)) else 1000
    im2 = ax.imshow(cop_patch, extent=cop_ext, origin="upper",
                    cmap="terrain", vmin=0, vmax=vm2, aspect="auto")
    plt.colorbar(im2, ax=ax, label="Elev. (m)", fraction=0.05, pad=0.02)
    ax.set_xlabel("Longitude (°)", fontsize=9)
    ax.set_ylabel("Latitude (°)", fontsize=9)
    ax.set_title(f"Copernicus DEM — scene bbox\n"
                 f"({bbox_w:.3f}°–{bbox_e:.3f}°W, {bbox_s:.3f}°–{bbox_n:.3f}°N)",
                 fontsize=9)

    # ── (0,2) Elevation histogram: Copernicus vs InSAR ───────────────────
    ax = axes[0, 2]
    cop_vals  = cop_patch[np.isfinite(cop_patch)].ravel()
    gold_vals = h_gold[np.isfinite(h_gold)].ravel()
    film_vals = h_film[np.isfinite(h_film)].ravel()

    # zero-mean each so shapes compare (removes absolute offset)
    cop_z  = cop_vals  - np.nanmean(cop_vals)
    gold_z = gold_vals - np.nanmean(gold_vals)
    film_z = film_vals - np.nanmean(film_vals)

    lim = max(np.percentile(np.abs(cop_z), 99),
              np.percentile(np.abs(gold_z), 99),
              np.percentile(np.abs(film_z), 99))
    bins = np.linspace(-lim, lim, 60)

    ax.hist(cop_z,  bins=bins, density=True, alpha=0.55, color=COPER_COLOR,
            label=f"Copernicus (n={len(cop_vals)//1000:.0f}k)")
    ax.hist(gold_z, bins=bins, density=True, alpha=0.50, color=GOLD_COLOR,
            label=f"Goldstein InSAR (n={len(gold_vals)//1000:.0f}k px)")
    ax.hist(film_z, bins=bins, density=True, alpha=0.50, color=FILM_COLOR,
            label=f"FiLMUNet InSAR (n={len(film_vals)//1000:.0f}k px)")

    ax.set_xlabel("Elevation − mean (m)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Elevation distribution (zero-mean)\nCopernicus vs InSAR-derived heights",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.97, 0.97, f"|B\u22a5| = {best_bp:.0f} m\nh\u2090\u2098\u2095 = {ha:.1f} m",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="#ccc"))

    # ── (1,0) Goldstein unwrapped phase ──────────────────────────────────
    ax = axes[1, 0]
    unw_disp = unw_gold.copy()
    unw_disp[~np.isfinite(unw_disp)] = np.nan
    vp = np.nanpercentile(np.abs(unw_disp[np.isfinite(unw_disp)]), 95)
    _geo_kw = {"extent": _sar_ext, "origin": "upper"} if _sar_ext else {}
    im3 = ax.imshow(unw_disp, cmap="RdBu_r", vmin=-vp, vmax=vp, aspect="auto", **_geo_kw)
    plt.colorbar(im3, ax=ax, label="Phase (rad)", fraction=0.05, pad=0.02)
    _geom_note = "geocoded, ~100 m" if _sar_ext else "SAR slant-range geometry"
    ax.set_title(f"Goldstein: unwrapped phase\n({_geom_note})", fontsize=9)
    ax.set_xlabel("Longitude (°)" if _sar_ext else "Range pixels", fontsize=9)
    ax.set_ylabel("Latitude (°)"  if _sar_ext else "Azimuth pixels", fontsize=9)
    nan_pct = 100 * np.isnan(unw_disp).mean()
    ax.text(0.02, 0.02, f"NaN: {nan_pct:.0f}%", transform=ax.transAxes,
            fontsize=8, color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))

    # ── (1,1) Goldstein InSAR height ─────────────────────────────────────
    ax = axes[1, 1]
    vhg = np.nanpercentile(np.abs(h_gold[np.isfinite(h_gold)]), 95)
    im4 = ax.imshow(h_gold, cmap="terrain_r", vmin=-vhg, vmax=vhg, aspect="auto", **_geo_kw)
    plt.colorbar(im4, ax=ax, label="Height (m)", fraction=0.05, pad=0.02)
    ax.set_title(f"Goldstein InSAR height\n(detrended, h\u2090\u2098\u2095\u202f=\u202f{ha:.1f}\u202fm, {_geom_note})",
                 fontsize=9)
    ax.set_xlabel("Longitude (°)" if _sar_ext else "Range pixels", fontsize=9)
    ax.set_ylabel("Latitude (°)"  if _sar_ext else "Azimuth pixels", fontsize=9)

    # ── (1,2) FiLMUNet InSAR height ──────────────────────────────────────
    ax = axes[1, 2]
    vhf = np.nanpercentile(np.abs(h_film[np.isfinite(h_film)]), 95)
    vhmax = max(vhg, vhf)
    im5 = ax.imshow(h_film, cmap="terrain_r", vmin=-vhmax, vmax=vhmax, aspect="auto", **_geo_kw)
    plt.colorbar(im5, ax=ax, label="Height (m)", fraction=0.05, pad=0.02)
    ax.set_title(f"FiLMUNet InSAR height\n(detrended, h\u2090\u2098\u2095\u202f=\u202f{ha:.1f}\u202fm, {_geom_note})",
                 fontsize=9)
    ax.set_xlabel("Longitude (°)" if _sar_ext else "Range pixels", fontsize=9)
    ax.set_ylabel("Latitude (°)"  if _sar_ext else "Azimuth pixels", fontsize=9)

    # NMAD annotation on both InSAR height panels
    for ax_h, vals, label, col in [
        (axes[1,1], gold_vals, f"NMAD={1.4826*np.median(np.abs(gold_vals-np.median(gold_vals))):.1f} m", GOLD_COLOR),
        (axes[1,2], film_vals, f"NMAD={1.4826*np.median(np.abs(film_vals-np.median(film_vals))):.1f} m", FILM_COLOR),
    ]:
        ax_h.text(0.98, 0.02, label, transform=ax_h.transAxes,
                  fontsize=8, ha="right", va="bottom", color="white",
                  bbox=dict(boxstyle="round,pad=0.25", fc=col, alpha=0.85))

    # ── suptitle + note ───────────────────────────────────────────────────
    fig.suptitle(
        "Copernicus GLO-30 Reference vs InSAR-derived Heights — AOI_000 Hawaii\n"
        r"Best pair: $|B_\perp|$ = 780 m  ($h_{\rm amb}$ = 12 m)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    _note = (
        "Row 1 geocoded by Range-Doppler sphere intersection (~100 m, flat-Earth DEM correction). "
        "Panel (0,2) is the valid cross-system elevation comparison."
    ) if _sar_ext else (
        "Row 1 panels are in SAR slant-range geometry (not geocoded). "
        "Panel (0,2) provides the valid cross-system elevation comparison."
    )
    fig.text(0.5, -0.01, _note, ha="center", fontsize=8, style="italic", color="#555555")

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved → {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
