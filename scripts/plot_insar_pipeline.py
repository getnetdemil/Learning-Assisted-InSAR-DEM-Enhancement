#!/usr/bin/env python3
"""
InSAR Pipeline Stage Visualizations — AOI_000 Hawaii

6 publication-ready figures, one per processing stage.
Uses the best pair (max |B_perp| = 780 m) as the representative example.

Figures
-------
  pipeline_01_coreg_insar.png      SAR amplitude + raw phase + coherence
  pipeline_02_phase_filtering.png  Raw / Goldstein / FiLMUNet wrapped phase + uncertainty
  pipeline_03_unwrapping.png       Wrapped → unwrapped (Goldstein vs FiLMUNet)
  pipeline_04_phase_to_elevation.png  Phase → DEM + transect + height distribution
  pipeline_05_geocoding.png        SAR pixel coords → geocoded lat/lon
  pipeline_06_google_earth.png     Products overlaid on Copernicus terrain basemap

Usage (one-liner)
-----------------
TASK_NAME="plot_pipeline" && DATE=$(date +%%Y%%m%%d_%%H%%M%%S) && LOG="logs/${TASK_NAME}_${DATE}.log" && export LD_LIBRARY_PATH=/scratch/gdemil24/hrwsi_s3client/torch-gpu/lib:$LD_LIBRARY_PATH && conda run --prefix /scratch/gdemil24/hrwsi_s3client/torch-gpu --no-capture-output env PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement python scripts/plot_insar_pipeline.py | tee "$LOG"
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
import matplotlib.patches as mpatches
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.insar_processing.geometry import load_extended_meta, geocode_patch_corners

PAIRS_DIR  = ROOT / "data" / "processed" / "pairs"
DEM_PATH   = ROOT / "data" / "reference" / "copernicus_dem" / "hawaii_dem.tif"
INDEX_PATH = ROOT / "data" / "manifests" / "full_index.parquet"
OUT_DIR    = ROOT / "experiments" / "enhanced" / "outputs" / "figures" / "pipeline"

CAPELLA_ALTITUDE_M = 525_000.0
GOLD_COLOR  = "#4878CF"
FILM_COLOR  = "#E87060"
COPER_COLOR = "#2CA02C"


# ── helpers ───────────────────────────────────────────────────────────────────

def _h_amb(bperp_m: float, incidence_deg: float, center_freq_ghz: float = 9.65) -> float:
    lam   = 3e8 / (center_freq_ghz * 1e9)
    theta = np.radians(incidence_deg)
    R     = CAPELLA_ALTITUDE_M / np.cos(theta)
    return lam * R * np.sin(theta) / (2.0 * abs(bperp_m))


def _detrend(arr: np.ndarray) -> np.ndarray:
    rows, cols = np.indices(arr.shape)
    valid = np.isfinite(arr)
    if valid.sum() < 10:
        return arr
    A = np.column_stack([rows[valid].ravel(), cols[valid].ravel(), np.ones(valid.sum())])
    try:
        c, *_ = np.linalg.lstsq(A, arr[valid].ravel(), rcond=None)
        return arr - (c[0] * rows + c[1] * cols + c[2])
    except Exception:
        return arr


def _read_raster(path: Path) -> np.ndarray:
    import rasterio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)


def _read_complex(path: Path) -> np.ndarray:
    """Read 2-band float32 TIF (band1=real, band2=imag) → complex64."""
    import rasterio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(path) as src:
            real = src.read(1).astype(np.float32)
            imag = src.read(2).astype(np.float32)
    return real + 1j * imag


def _copernicus_patch(bbox_w: float, bbox_s: float,
                      bbox_e: float, bbox_n: float):
    """Return (data, [left, right, bottom, top]) for the Copernicus DEM window."""
    import rasterio
    from rasterio.windows import from_bounds
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(DEM_PATH) as src:
            win  = from_bounds(bbox_w, bbox_s, bbox_e, bbox_n, src.transform)
            data = src.read(1, window=win).astype(np.float32)
            nd   = src.nodata
    if nd is not None:
        data[data == nd] = float("nan")
    return data, [bbox_w, bbox_e, bbox_s, bbox_n]


def _copernicus_overview(scale: int = 20):
    """Return downsampled DEM array + geographic extent [left, right, bottom, top]."""
    import rasterio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(DEM_PATH) as src:
            data = src.read(1, out_shape=(src.count, src.height // scale,
                                          src.width  // scale)).astype(np.float32)
            ext  = [src.bounds.left, src.bounds.right,
                    src.bounds.bottom, src.bounds.top]
    data[data < -500] = float("nan")
    return data, ext


def _nmad(arr: np.ndarray) -> float:
    v = arr[np.isfinite(arr)]
    if len(v) == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def _find_best_pair():
    """Return (pair_dir, meta_dict) for pair with max |B_perp|."""
    best_pd, best_bp, best_m = None, 0.0, None
    for d in sorted(PAIRS_DIR.iterdir()):
        mp = d / "coreg_meta.json"
        if not mp.exists():
            continue
        m  = json.load(open(mp))
        bp = abs(m.get("bperp_m", 0.0))
        if bp > best_bp \
                and (d / "unw_phase.tif").exists() \
                and (d / "unw_phase_film_unet.tif").exists():
            best_bp = bp
            best_pd = d
            best_m  = m
    assert best_pd is not None, "No valid pair found in PAIRS_DIR"
    return best_pd, best_m


def _geocode_sar_extent(meta: dict, cop_patch: np.ndarray):
    """
    Use geocode_patch_corners to get approximate lat/lon extent for imshow.
    Returns [lon_W, lon_E, lat_S, lat_N] or None if unavailable.
    """
    ext_json = (ROOT / "data" / "raw" / "AOI_000"
                / meta["id_ref"] / f"{meta['id_ref']}_extended.json")
    if not ext_json.exists():
        print(f"  WARNING: extended JSON not found: {ext_json.name}", flush=True)
        return None
    try:
        ext_meta  = load_extended_meta(ext_json)
        h_terrain = float(np.nanmedian(cop_patch[np.isfinite(cop_patch)])) \
                    if np.any(np.isfinite(cop_patch)) else 300.0
        corners   = geocode_patch_corners(
            ext_meta,
            patch_row        = int(meta["patch_row_ref"]),
            patch_col        = int(meta["patch_col_ref"]),
            patch_size       = int(meta.get("patch_size", 4096)),
            terrain_height_m = h_terrain,
        )
        # corners shape (4, 2): TL, TR, BL, BR → [lat, lon]
        return [float(corners[:, 1].min()), float(corners[:, 1].max()),
                float(corners[:, 0].min()), float(corners[:, 0].max())]
    except Exception as e:
        print(f"  WARNING: geocoding failed ({e})", flush=True)
        return None


def _save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {p.relative_to(ROOT)}", flush=True)


# ── Figure 1: Co-registration & InSAR Formation ──────────────────────────────

def fig1_coreg_insar(pair_dir: Path, meta: dict, info: pd.Series) -> None:
    print("Fig 1: Co-registration & InSAR Formation ...", flush=True)
    ifg_raw = _read_complex(pair_dir / "ifg_raw_complex_real_imag.tif")
    coh     = _read_raster(pair_dir / "coherence.tif")

    amp       = np.log1p(np.abs(ifg_raw))
    phase_raw = np.angle(ifg_raw)

    bperp = meta["bperp_m"]
    dt    = meta["dt_days"]
    inc   = meta["incidence_angle_deg"]
    ha    = _h_amb(bperp, inc)
    coh_med = float(np.nanmedian(coh[np.isfinite(coh)]))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Stage 1 — Co-registration & InSAR Formation\n"
        f"Δt = {dt:.1f} d  |  B⊥ = {bperp:.0f} m  |  θ_inc = {inc:.1f}°  |  h_amb = {ha:.1f} m",
        fontsize=11, fontweight="bold",
    )

    # (0,0) Interferometric amplitude
    ax = axes[0]
    v1, v99 = np.nanpercentile(amp, 1), np.nanpercentile(amp, 99)
    im = ax.imshow(amp, cmap="gray", vmin=v1, vmax=v99, origin="upper")
    fig.colorbar(im, ax=ax, label="log(1 + |ifg|)", fraction=0.046, pad=0.04)
    ax.set_title("SAR Cross-Amplitude\n(log-scaled interferometric intensity)", fontsize=9)
    ax.set_xlabel("Range (pixels)")
    ax.set_ylabel("Azimuth (pixels)")

    # (0,1) Raw wrapped phase
    ax = axes[1]
    im = ax.imshow(phase_raw, cmap="hsv", vmin=-np.pi, vmax=np.pi, origin="upper")
    fig.colorbar(im, ax=ax, label="Phase (rad)", fraction=0.046, pad=0.04)
    ax.set_title("Raw Wrapped Interferogram Phase\n(unfiltered, −π to π)", fontsize=9)
    ax.set_xlabel("Range (pixels)")
    ax.set_ylabel("Azimuth (pixels)")

    # (0,2) Coherence
    ax = axes[2]
    im = ax.imshow(coh, cmap="inferno", vmin=0, vmax=1, origin="upper")
    fig.colorbar(im, ax=ax, label="Coherence [0, 1]", fraction=0.046, pad=0.04)
    ax.set_title(f"Spatial Coherence\n(median = {coh_med:.3f})", fontsize=9)
    ax.set_xlabel("Range (pixels)")
    ax.set_ylabel("Azimuth (pixels)")

    plt.tight_layout()
    _save(fig, "pipeline_01_coreg_insar.png")


# ── Figure 2: Phase Filtering / Denoising ────────────────────────────────────

def fig2_phase_filtering(pair_dir: Path, meta: dict) -> None:
    print("Fig 2: Phase Filtering / Denoising ...", flush=True)
    ifg_raw  = _read_complex(pair_dir / "ifg_raw_complex_real_imag.tif")
    ifg_gold = _read_complex(pair_dir / "ifg_goldstein_complex_real_imag.tif")
    ifg_film = _read_complex(pair_dir / "ifg_film_unet.tif")
    log_var  = _read_raster(pair_dir / "log_var.tif")

    phase_raw  = np.angle(ifg_raw)
    phase_gold = np.angle(ifg_gold)
    phase_film = np.angle(ifg_film)

    lv_valid = log_var[np.isfinite(log_var)]
    lv_v1, lv_v99 = (float(np.percentile(lv_valid, 2)),
                     float(np.percentile(lv_valid, 98))) if len(lv_valid) else (-5, 0)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Stage 2 — Phase Filtering: Raw → Goldstein → FiLMUNet",
        fontsize=11, fontweight="bold",
    )

    panels = [
        (axes[0], phase_raw,  "hsv",    -np.pi, np.pi, "Phase (rad)",
         "Raw Wrapped Phase\n(no filtering)"),
        (axes[1], phase_gold, "hsv",    -np.pi, np.pi, "Phase (rad)",
         "Goldstein Filtered Phase\n(adaptive frequency-domain filter)"),
        (axes[2], phase_film, "hsv",    -np.pi, np.pi, "Phase (rad)",
         "FiLMUNet Filtered Phase\n(geometry-conditioned learned denoising)"),
        (axes[3], log_var,    "plasma", lv_v1,  lv_v99, "log σ²",
         "FiLMUNet Uncertainty\n(per-pixel log-variance σ²)"),
    ]
    for ax, data, cmap, vmin, vmax, clabel, title in panels:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        fig.colorbar(im, ax=ax, label=clabel, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Range (pixels)")
        ax.set_ylabel("Azimuth (pixels)")

    plt.tight_layout()
    _save(fig, "pipeline_02_phase_filtering.png")


# ── Figure 3: Phase Unwrapping ────────────────────────────────────────────────

def _valid_bbox(arr: np.ndarray, pad: int = 150):
    """(r0, r1, c0, c1) tight bounding box around finite pixels + padding."""
    rows, cols = np.where(np.isfinite(arr))
    if len(rows) == 0:
        return 0, arr.shape[0] - 1, 0, arr.shape[1] - 1
    H, W = arr.shape
    return (max(0, int(rows.min()) - pad),
            min(H - 1, int(rows.max()) + pad),
            max(0, int(cols.min()) - pad),
            min(W - 1, int(cols.max()) + pad))


def _add_zoom_box(ax, bbox, color="red", lw=1.5):
    """Draw a rectangle on ax showing the zoom region (r0,r1,c0,c1)."""
    r0, r1, c0, c1 = bbox
    ax.add_patch(mpatches.Rectangle(
        (c0, r0), c1 - c0, r1 - r0,
        lw=lw, edgecolor=color, facecolor="none", linestyle="--",
    ))


def fig3_unwrapping(pair_dir: Path, meta: dict) -> None:
    print("Fig 3: Phase Unwrapping ...", flush=True)
    ifg_gold = _read_complex(pair_dir / "ifg_goldstein_complex_real_imag.tif")
    ifg_film = _read_complex(pair_dir / "ifg_film_unet.tif")
    unw_gold = _read_raster(pair_dir / "unw_phase.tif")
    unw_film = _read_raster(pair_dir / "unw_phase_film_unet.tif")

    wrap_gold = np.angle(ifg_gold)
    wrap_film = np.angle(ifg_film)
    mask_gold = np.isfinite(unw_gold).astype(np.float32)
    cov_gold  = mask_gold.mean() * 100
    cov_film  = np.isfinite(unw_film).mean() * 100

    diff = unw_gold - unw_film
    diff[~(np.isfinite(unw_gold) & np.isfinite(unw_film))] = float("nan")
    dv = float(np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 95)) \
         if np.any(np.isfinite(diff)) else 30.0

    # ── bounding boxes for valid regions ─────────────────────────────────────
    bg = _valid_bbox(unw_gold, pad=150)   # (r0, r1, c0, c1) Goldstein
    bf = _valid_bbox(unw_film, pad=150)   # FiLMUNet
    # union bbox for the diff panel (covers both valid sets)
    bd = (min(bg[0], bf[0]), max(bg[1], bf[1]),
          min(bg[2], bf[2]), max(bg[3], bf[3]))

    # cropped arrays for column 1 and 2
    ug_crop   = unw_gold[bg[0]:bg[1]+1, bg[2]:bg[3]+1]
    uf_crop   = unw_film[bf[0]:bf[1]+1, bf[2]:bf[3]+1]
    diff_crop = diff[bd[0]:bd[1]+1, bd[2]:bd[3]+1]

    # color ranges on the cropped data
    ug_valid = ug_crop[np.isfinite(ug_crop)]
    uf_valid = uf_crop[np.isfinite(uf_crop)]
    ug_v1  = float(np.percentile(ug_valid, 2))  if len(ug_valid) else -100
    ug_v99 = float(np.percentile(ug_valid, 98)) if len(ug_valid) else  100
    uf_v1  = float(np.percentile(uf_valid, 2))  if len(uf_valid) else -100
    uf_v99 = float(np.percentile(uf_valid, 98)) if len(uf_valid) else  100

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor("white")
    fig.suptitle("Stage 3 — Phase Unwrapping (SNAPHU)\n"
                 "Column 1: full scene  |  Column 2: zoomed to valid region (red box)  "
                 "|  Column 3: valid mask / phase difference",
                 fontsize=10, fontweight="bold")

    # ── Row 0: Goldstein ─────────────────────────────────────────────────────
    # (0,0) full wrapped phase + red zoom box
    ax = axes[0, 0]
    im = ax.imshow(wrap_gold, cmap="hsv", vmin=-np.pi, vmax=np.pi, origin="upper")
    fig.colorbar(im, ax=ax, label="Phase (rad)", fraction=0.046, pad=0.04)
    _add_zoom_box(ax, bg)
    ax.set_title("Goldstein: Wrapped Phase\n(red box = zoom region in col 2)", fontsize=9)
    ax.set_xlabel("Range (pixels)"); ax.set_ylabel("Azimuth (pixels)")

    # (0,1) zoomed unwrapped phase
    ax = axes[0, 1]
    im = ax.imshow(ug_crop, cmap="RdYlBu_r", vmin=ug_v1, vmax=ug_v99, origin="upper")
    fig.colorbar(im, ax=ax, label="Phase (rad)", fraction=0.046, pad=0.04)
    gh, gw = ug_crop.shape
    ax.set_title(f"Goldstein: Unwrapped Phase — ZOOMED\n"
                 f"coverage = {cov_gold:.1f}%  |  valid region: {gh}×{gw} px", fontsize=9)
    ax.set_xlabel("Range (pixels, zoomed)"); ax.set_ylabel("Azimuth (pixels, zoomed)")

    # (0,2) full valid-pixel mask (context: shows how sparse coverage is)
    ax = axes[0, 2]
    im = ax.imshow(mask_gold, cmap="gray", vmin=0, vmax=1, origin="upper")
    fig.colorbar(im, ax=ax, label="Valid pixel", fraction=0.046, pad=0.04)
    _add_zoom_box(ax, bg)
    ax.set_title(f"Goldstein: Valid-pixel Mask (full scene)\n"
                 f"{cov_gold:.1f}% of 4096×4096 unwrapped", fontsize=9)
    ax.set_xlabel("Range (pixels)"); ax.set_ylabel("Azimuth (pixels)")

    # ── Row 1: FiLMUNet ──────────────────────────────────────────────────────
    # (1,0) full wrapped phase + red zoom box
    ax = axes[1, 0]
    im = ax.imshow(wrap_film, cmap="hsv", vmin=-np.pi, vmax=np.pi, origin="upper")
    fig.colorbar(im, ax=ax, label="Phase (rad)", fraction=0.046, pad=0.04)
    _add_zoom_box(ax, bf)
    ax.set_title("FiLMUNet: Wrapped Phase\n(red box = zoom region in col 2)", fontsize=9)
    ax.set_xlabel("Range (pixels)"); ax.set_ylabel("Azimuth (pixels)")

    # (1,1) zoomed unwrapped phase
    ax = axes[1, 1]
    im = ax.imshow(uf_crop, cmap="RdYlBu_r", vmin=uf_v1, vmax=uf_v99, origin="upper")
    fig.colorbar(im, ax=ax, label="Phase (rad)", fraction=0.046, pad=0.04)
    fh, fw = uf_crop.shape
    ax.set_title(f"FiLMUNet: Unwrapped Phase — ZOOMED\n"
                 f"coverage = {cov_film:.1f}%  |  valid region: {fh}×{fw} px", fontsize=9)
    ax.set_xlabel("Range (pixels, zoomed)"); ax.set_ylabel("Azimuth (pixels, zoomed)")

    # (1,2) zoomed diff
    ax = axes[1, 2]
    im = ax.imshow(diff_crop, cmap="seismic", vmin=-dv, vmax=dv, origin="upper")
    fig.colorbar(im, ax=ax, label="Δ Phase (rad)", fraction=0.046, pad=0.04)
    dh, dw = diff_crop.shape
    ax.set_title(f"Goldstein − FiLMUNet — ZOOMED\n"
                 f"phase difference, clipped ± {dv:.0f} rad  |  {dh}×{dw} px", fontsize=9)
    ax.set_xlabel("Range (pixels, zoomed)"); ax.set_ylabel("Azimuth (pixels, zoomed)")

    plt.tight_layout()
    _save(fig, "pipeline_03_unwrapping.png")


# ── shared geo-zoom helper ────────────────────────────────────────────────────

def _pixel_bbox_to_geo(bbox, arr_shape, sar_ext):
    """Map pixel bbox (r0,r1,c0,c1) to (lon_min,lon_max,lat_min,lat_max)."""
    r0, r1, c0, c1 = bbox
    H, W = arr_shape
    lon_W, lon_E, lat_S, lat_N = sar_ext
    dlon = lon_E - lon_W
    dlat = lat_N - lat_S
    return (lon_W + c0 / W * dlon,
            lon_W + (c1 + 1) / W * dlon,
            lat_N - (r1 + 1) / H * dlat,
            lat_N - r0 / H * dlat)   # (lon_min, lon_max, lat_min, lat_max)


# ── Figure 4: Phase to Elevation ─────────────────────────────────────────────

def fig4_phase_to_elevation(pair_dir: Path, meta: dict, info: pd.Series) -> None:
    print("Fig 4: Phase → Elevation ...", flush=True)
    bperp = meta["bperp_m"]
    inc   = meta["incidence_angle_deg"]
    ha    = _h_amb(bperp, inc)

    unw_gold = _read_raster(pair_dir / "unw_phase.tif")
    unw_film = _read_raster(pair_dir / "unw_phase_film_unet.tif")

    h_gold = _detrend(unw_gold * ha / (2.0 * np.pi))
    h_film = _detrend(unw_film * ha / (2.0 * np.pi))
    h_gold[~np.isfinite(h_gold)] = float("nan")
    h_film[~np.isfinite(h_film)] = float("nan")

    nmad_gold = _nmad(h_gold)
    nmad_film = _nmad(h_film)

    bbox_w = float(info["bbox_w"]); bbox_s = float(info["bbox_s"])
    bbox_e = float(info["bbox_e"]); bbox_n = float(info["bbox_n"])
    cop_patch, cop_ext = _copernicus_patch(bbox_w, bbox_s, bbox_e, bbox_n)

    cop_valid = cop_patch[np.isfinite(cop_patch)]
    vm_cop = float(np.percentile(cop_valid, 98)) if len(cop_valid) else 1000.0

    # ── zoom: crop InSAR height arrays to their valid bboxes ─────────────────
    bg   = _valid_bbox(h_gold, pad=150)
    bf   = _valid_bbox(h_film, pad=150)
    hg_crop  = h_gold[bg[0]:bg[1]+1, bg[2]:bg[3]+1]
    hf_crop  = h_film[bf[0]:bf[1]+1, bf[2]:bf[3]+1]

    diff = h_gold - h_film
    diff[~(np.isfinite(h_gold) & np.isfinite(h_film))] = float("nan")
    bd       = _valid_bbox(diff, pad=150)
    diff_crop = diff[bd[0]:bd[1]+1, bd[2]:bd[3]+1]

    # Shared color range from the cropped valid data
    all_h = np.concatenate([hg_crop[np.isfinite(hg_crop)], hf_crop[np.isfinite(hf_crop)]])
    h_vmin = float(np.percentile(all_h, 2))  if len(all_h) else -200
    h_vmax = float(np.percentile(all_h, 98)) if len(all_h) else  200
    dv = float(np.nanpercentile(np.abs(diff_crop[np.isfinite(diff_crop)]), 95)) \
         if np.any(np.isfinite(diff_crop)) else 20.0

    # Transect: use row with most valid pixels (blind middle row is all NaN)
    best_rg = int(np.argmax(np.sum(np.isfinite(h_gold), axis=1)))
    best_rf = int(np.argmax(np.sum(np.isfinite(h_film), axis=1)))
    mid_cop = cop_patch.shape[0] // 2
    g_row   = h_gold[best_rg, :]
    f_row   = h_film[best_rf, :]
    c_row   = cop_patch[mid_cop, :]
    xg = np.linspace(0, 1, len(g_row))
    xf = np.linspace(0, 1, len(f_row))
    xc = np.linspace(0, 1, len(c_row))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Stage 4 — Phase → Elevation  (h_amb = {ha:.1f} m, plane-ramp detrended)\n"
        f"NMAD: Goldstein = {nmad_gold:.1f} m  |  FiLMUNet = {nmad_film:.1f} m  "
        f"|  col 1-2 zoomed to valid region (red box on col 0)",
        fontsize=10, fontweight="bold",
    )

    # (0,0) Copernicus reference — full geographic extent
    ax = axes[0, 0]
    im = ax.imshow(cop_patch, extent=cop_ext, origin="upper",
                   cmap="terrain", vmin=0, vmax=vm_cop, aspect="auto")
    fig.colorbar(im, ax=ax, label="Height (m)", fraction=0.046, pad=0.04)
    ax.set_title("Copernicus GLO-30 Reference\n(full scene, geographic coordinates)", fontsize=9)
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")

    # (0,1) Goldstein InSAR height — ZOOMED to valid bbox
    ax = axes[0, 1]
    im = ax.imshow(hg_crop, cmap="terrain", vmin=h_vmin, vmax=h_vmax, origin="upper")
    fig.colorbar(im, ax=ax, label="Height (m)", fraction=0.046, pad=0.04)
    gh, gw = hg_crop.shape
    ax.set_title(f"Goldstein InSAR Height — ZOOMED\n"
                 f"NMAD = {nmad_gold:.1f} m  |  valid region: {gh}×{gw} px", fontsize=9)
    ax.set_xlabel("Range (pixels, zoomed)"); ax.set_ylabel("Azimuth (pixels, zoomed)")

    # (0,2) FiLMUNet InSAR height — ZOOMED to valid bbox
    ax = axes[0, 2]
    im = ax.imshow(hf_crop, cmap="terrain", vmin=h_vmin, vmax=h_vmax, origin="upper")
    fig.colorbar(im, ax=ax, label="Height (m)", fraction=0.046, pad=0.04)
    fh, fw = hf_crop.shape
    ax.set_title(f"FiLMUNet InSAR Height — ZOOMED\n"
                 f"NMAD = {nmad_film:.1f} m  |  valid region: {fh}×{fw} px", fontsize=9)
    ax.set_xlabel("Range (pixels, zoomed)"); ax.set_ylabel("Azimuth (pixels, zoomed)")

    # (1,0) Height distribution (plot — no zoom needed)
    ax = axes[1, 0]
    cop_z  = cop_valid - float(np.nanmean(cop_valid))
    gold_z = hg_crop[np.isfinite(hg_crop)] - float(np.nanmean(hg_crop[np.isfinite(hg_crop)]))
    film_z = hf_crop[np.isfinite(hf_crop)] - float(np.nanmean(hf_crop[np.isfinite(hf_crop)]))
    lim    = max(np.percentile(np.abs(cop_z), 99),
                 np.percentile(np.abs(gold_z), 99),
                 np.percentile(np.abs(film_z), 99))
    bins   = np.linspace(-lim, lim, 60)
    ax.hist(cop_z,  bins=bins, density=True, alpha=0.55, color=COPER_COLOR,
            label=f"Copernicus (n={len(cop_z)//1000:.0f}k)")
    ax.hist(gold_z, bins=bins, density=True, alpha=0.50, color=GOLD_COLOR,
            label=f"Goldstein  (n={len(gold_z)})")
    ax.hist(film_z, bins=bins, density=True, alpha=0.50, color=FILM_COLOR,
            label=f"FiLMUNet   (n={len(film_z)})")
    ax.set_xlabel("Height − mean (m)"); ax.set_ylabel("Density")
    ax.set_title("Height Distribution\n(zero-mean, shape comparison)", fontsize=9)
    ax.legend(fontsize=7)

    # (1,1) Height difference — ZOOMED
    ax = axes[1, 1]
    im = ax.imshow(diff_crop, cmap="seismic", vmin=-dv, vmax=dv, origin="upper")
    fig.colorbar(im, ax=ax, label="Δ Height (m)", fraction=0.046, pad=0.04)
    dh, dw = diff_crop.shape
    ax.set_title(f"Goldstein − FiLMUNet Height — ZOOMED\n"
                 f"clipped ± {dv:.0f} m  |  {dh}×{dw} px", fontsize=9)
    ax.set_xlabel("Range (pixels, zoomed)"); ax.set_ylabel("Azimuth (pixels, zoomed)")

    # (1,2) Elevation transect through the row with most valid InSAR pixels
    ax = axes[1, 2]
    ax.plot(xc, c_row, color=COPER_COLOR, lw=1.5, label="Copernicus GLO-30", alpha=0.85)
    ax.plot(xg, g_row, color=GOLD_COLOR,  lw=1.0,
            label=f"Goldstein InSAR (row {best_rg})", alpha=0.75)
    ax.plot(xf, f_row, color=FILM_COLOR,  lw=1.0,
            label=f"FiLMUNet InSAR  (row {best_rf})", alpha=0.75)
    ax.set_xlabel("Normalised scene position (0 = near-range, 1 = far-range)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Elevation Transect (row with most valid InSAR pixels)\n"
                 "InSAR vs Copernicus reference", fontsize=9)
    ax.legend(fontsize=7)

    plt.tight_layout()
    _save(fig, "pipeline_04_phase_to_elevation.png")


# ── Figure 5: Range-Doppler Terrain Correction (Geocoding) ───────────────────

def fig5_geocoding(pair_dir: Path, meta: dict,
                   cop_patch: np.ndarray, sar_ext) -> None:
    print("Fig 5: Range-Doppler Terrain Correction ...", flush=True)
    coh      = _read_raster(pair_dir / "coherence.tif")
    unw_film = _read_raster(pair_dir / "unw_phase_film_unet.tif")
    ha       = _h_amb(meta["bperp_m"], meta["incidence_angle_deg"])
    h_film   = _detrend(unw_film * ha / (2.0 * np.pi))
    h_film[~np.isfinite(h_film)] = float("nan")

    hf_valid = h_film[np.isfinite(h_film)]
    hf_v1  = float(np.percentile(hf_valid, 2))  if len(hf_valid) else -100
    hf_v99 = float(np.percentile(hf_valid, 98)) if len(hf_valid) else  100

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Stage 5 — Range-Doppler Terrain Correction (Geocoding)",
                 fontsize=11, fontweight="bold")

    # (0,0) SAR pixel coordinates
    ax = axes[0]
    im = ax.imshow(coh, cmap="inferno", vmin=0, vmax=1, origin="upper")
    fig.colorbar(im, ax=ax, label="Coherence", fraction=0.046, pad=0.04)
    ax.set_title("Coherence: SAR Slant-Range Geometry\n"
                 "(azimuth × range pixel coordinates)", fontsize=9)
    ax.set_xlabel("Range (pixels)"); ax.set_ylabel("Azimuth (pixels)")

    # (0,1) Geocoded coherence
    ax = axes[1]
    if sar_ext is not None:
        im = ax.imshow(coh, cmap="inferno", vmin=0, vmax=1, origin="upper",
                       extent=sar_ext, aspect="auto")
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
        ax.set_title("Coherence: Geocoded\n"
                     "(Range-Doppler sphere intersection, ~100 m)", fontsize=9)
    else:
        im = ax.imshow(np.zeros((4, 4)), cmap="gray", vmin=0, vmax=1)
        ax.text(0.5, 0.5, "Extended JSON not found\n(geocoding unavailable)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9,
                color="red")
        ax.set_title("Coherence: Geocoded (unavailable)", fontsize=9)
    fig.colorbar(im, ax=ax, label="Coherence", fraction=0.046, pad=0.04)

    # (0,2) Geocoded FiLMUNet height — zoom to valid region
    ax = axes[2]
    if sar_ext is not None:
        im = ax.imshow(h_film, cmap="terrain", vmin=hf_v1, vmax=hf_v99, origin="upper",
                       extent=sar_ext, aspect="auto")
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
        # zoom viewport to the valid bbox in geographic space
        if np.any(np.isfinite(h_film)):
            bh = _valid_bbox(h_film, pad=200)
            lmin, lmax, latmin, latmax = _pixel_bbox_to_geo(bh, h_film.shape, sar_ext)
            ax.set_xlim(lmin, lmax)
            ax.set_ylim(latmin, latmax)
            fh2, fw2 = bh[1]-bh[0], bh[3]-bh[2]
            ax.set_title(f"FiLMUNet Height: Geocoded — ZOOMED\n"
                         f"detrended  |  valid region: ~{fh2}×{fw2} px", fontsize=9)
        else:
            ax.set_title("FiLMUNet InSAR Height: Geocoded\n(no valid pixels)", fontsize=9)
    else:
        im = ax.imshow(h_film, cmap="terrain", vmin=hf_v1, vmax=hf_v99, origin="upper")
        ax.set_title("FiLMUNet InSAR Height\n(SAR geometry — geocoding unavailable)", fontsize=9)
        ax.set_xlabel("Range (pixels)"); ax.set_ylabel("Azimuth (pixels)")
    fig.colorbar(im, ax=ax, label="Height (m)", fraction=0.046, pad=0.04)

    plt.tight_layout()
    _save(fig, "pipeline_05_geocoding.png")


# ── Figure 6: Geographic Overlay (terrain basemap) ───────────────────────────

def fig6_geographic_overlay(pair_dir: Path, meta: dict, info: pd.Series,
                             cop_patch: np.ndarray, cop_ext: list,
                             sar_ext) -> None:
    print("Fig 6: Geographic Overlay (Copernicus terrain basemap) ...", flush=True)
    coh      = _read_raster(pair_dir / "coherence.tif")
    ifg_film = _read_complex(pair_dir / "ifg_film_unet.tif")
    unw_film = _read_raster(pair_dir / "unw_phase_film_unet.tif")
    ha       = _h_amb(meta["bperp_m"], meta["incidence_angle_deg"])
    h_film   = _detrend(unw_film * ha / (2.0 * np.pi))
    h_film[~np.isfinite(h_film)] = float("nan")
    phase_film = np.angle(ifg_film)

    cop_overview, ov_ext = _copernicus_overview(scale=20)
    bbox_w = float(info["bbox_w"]); bbox_s = float(info["bbox_s"])
    bbox_e = float(info["bbox_e"]); bbox_n = float(info["bbox_n"])

    cop_valid = cop_patch[np.isfinite(cop_patch)]
    vm_cop = float(np.percentile(cop_valid, 98)) if len(cop_valid) else 1000.0
    ov_vm  = float(np.nanpercentile(cop_overview[np.isfinite(cop_overview)], 98))

    hf_valid = h_film[np.isfinite(h_film)]
    hf_v1  = float(np.percentile(hf_valid, 2))  if len(hf_valid) else -100
    hf_v99 = float(np.percentile(hf_valid, 98)) if len(hf_valid) else  100

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Stage 6 — Geographic Overlay on Terrain Basemap (Copernicus GLO-30)\n"
        "InSAR products geocoded and overlaid on reference elevation",
        fontsize=11, fontweight="bold",
    )

    # (0,0) Terrain basemap + scene footprint
    ax = axes[0]
    ax.imshow(cop_overview, extent=ov_ext, origin="upper",
              cmap="terrain", vmin=0, vmax=ov_vm, aspect="auto")
    for (lw, ec, fc, al) in [(2.0, "yellow", "yellow", 0.20),
                              (2.0, "yellow", "none",   1.00)]:
        ax.add_patch(mpatches.Rectangle(
            (bbox_w, bbox_s), bbox_e - bbox_w, bbox_n - bbox_s,
            lw=lw, edgecolor=ec, facecolor=fc, alpha=al,
        ))
    ax.set_xlim(bbox_w - 0.6, bbox_e + 0.6)
    ax.set_ylim(bbox_s - 0.4, bbox_n + 0.4)
    ax.set_title("Terrain Basemap (Copernicus GLO-30)\nScene footprint (yellow)", fontsize=9)
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")

    # (0,1) Coherence overlaid on terrain
    ax = axes[1]
    ax.imshow(cop_patch, extent=cop_ext, origin="upper",
              cmap="terrain", vmin=0, vmax=vm_cop, aspect="auto")
    if sar_ext is not None:
        coh_norm = np.clip(coh, 0, 1)
        coh_rgba = plt.cm.inferno(coh_norm)
        coh_rgba[..., 3] = 0.55
        ax.imshow(coh_rgba, extent=sar_ext, origin="upper", aspect="auto")
    else:
        ax.imshow(coh, cmap="inferno", vmin=0, vmax=1, origin="upper",
                  extent=cop_ext, aspect="auto", alpha=0.55)
    ax.set_title("Coherence Overlay\non Terrain Basemap (α = 0.55)", fontsize=9)
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")

    # (0,2) FiLMUNet height overlaid on terrain — zoom to valid region
    ax = axes[2]
    if sar_ext is not None:
        h_norm = np.where(np.isfinite(h_film),
                          np.clip((h_film - hf_v1) / max(hf_v99 - hf_v1, 1e-6), 0, 1),
                          np.nan)
        h_rgba = plt.cm.terrain(np.nan_to_num(h_norm, nan=0.0))
        h_rgba[..., 3] = np.where(np.isfinite(h_film), 0.60, 0.0)
        # zoom to valid bbox + a generous margin so terrain context is visible
        if np.any(np.isfinite(h_film)):
            bh  = _valid_bbox(h_film, pad=400)
            lmin, lmax, latmin, latmax = _pixel_bbox_to_geo(bh, h_film.shape, sar_ext)
            margin_lon = (lmax - lmin) * 0.5
            margin_lat = (latmax - latmin) * 0.5
            ax.imshow(cop_patch, extent=cop_ext, origin="upper",
                      cmap="terrain", vmin=0, vmax=vm_cop, aspect="auto")
            ax.imshow(h_rgba, extent=sar_ext, origin="upper", aspect="auto")
            ax.set_xlim(lmin - margin_lon, lmax + margin_lon)
            ax.set_ylim(latmin - margin_lat, latmax + margin_lat)
        else:
            ax.imshow(cop_patch, extent=cop_ext, origin="upper",
                      cmap="terrain", vmin=0, vmax=vm_cop, aspect="auto")
            ax.imshow(h_rgba, extent=sar_ext, origin="upper", aspect="auto")
    else:
        ax.imshow(cop_patch, extent=cop_ext, origin="upper",
                  cmap="terrain", vmin=0, vmax=vm_cop, aspect="auto")
        ax.imshow(h_film, cmap="terrain", vmin=hf_v1, vmax=hf_v99, origin="upper",
                  extent=cop_ext, aspect="auto", alpha=0.60)
    ax.set_title("FiLMUNet InSAR Height Overlay — ZOOMED\n"
                 "on Terrain Basemap (α = 0.60, with context margin)", fontsize=9)
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")

    plt.tight_layout()
    _save(fig, "pipeline_06_google_earth.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60, flush=True)
    print("InSAR Pipeline Stage Visualizations — AOI_000 Hawaii", flush=True)
    print("=" * 60, flush=True)

    print("\nFinding best pair (max |B_perp|) ...", flush=True)
    pair_dir, meta = _find_best_pair()
    print(f"  Dir : {pair_dir.name[:72]}", flush=True)
    print(f"  B⊥  : {abs(meta['bperp_m']):.1f} m  |  Δt : {meta['dt_days']:.1f} d  |  "
          f"θ_inc : {meta['incidence_angle_deg']:.1f}°", flush=True)

    # scene info for bbox / freq
    df   = pd.read_parquet(INDEX_PATH)
    rows = df[df["id"] == meta["id_ref"]]
    assert len(rows) > 0, f"id_ref {meta['id_ref']!r} not found in index"
    info = rows.iloc[0]

    bbox_w = float(info["bbox_w"]); bbox_s = float(info["bbox_s"])
    bbox_e = float(info["bbox_e"]); bbox_n = float(info["bbox_n"])

    print("\nLoading Copernicus DEM patch ...", flush=True)
    cop_patch, cop_ext = _copernicus_patch(bbox_w, bbox_s, bbox_e, bbox_n)

    print("Computing SAR geocode extent ...", flush=True)
    sar_ext = _geocode_sar_extent(meta, cop_patch)
    if sar_ext:
        print(f"  lon [{sar_ext[0]:.4f}, {sar_ext[1]:.4f}]  "
              f"lat [{sar_ext[2]:.4f}, {sar_ext[3]:.4f}]", flush=True)

    print(flush=True)
    fig1_coreg_insar(pair_dir, meta, info)
    fig2_phase_filtering(pair_dir, meta)
    fig3_unwrapping(pair_dir, meta)
    fig4_phase_to_elevation(pair_dir, meta, info)
    fig5_geocoding(pair_dir, meta, cop_patch, sar_ext)
    fig6_geographic_overlay(pair_dir, meta, info, cop_patch, cop_ext, sar_ext)

    print(f"\nDone. All 6 figures in {OUT_DIR.relative_to(ROOT)}/", flush=True)


if __name__ == "__main__":
    main()
