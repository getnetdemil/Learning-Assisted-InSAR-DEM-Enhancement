#!/usr/bin/env python3
"""
Extended DEM NMAD visualizations — 4-panel figure:

  (0,0) Copernicus DEM overview with pair scene footprints
  (0,1) |B_perp| vs per-pair NMAD — physics explanation of large NMAD values
  (1,0) Per-pair NMAD improvement bar (Goldstein − FiLMUNet, sorted)
  (1,1) Height error CDF (sampled pixels): |h_insar − h_ref_median|

Output: experiments/enhanced/outputs/figures/dem_nmad_extended.png
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

PAIRS_DIR  = ROOT / "data" / "processed" / "pairs"
DEM_PATH   = ROOT / "data" / "reference" / "copernicus_dem" / "hawaii_dem.tif"
INDEX_PATH = ROOT / "data" / "manifests" / "full_index.parquet"
OUT_PATH   = ROOT / "experiments" / "enhanced" / "outputs" / "figures" / "dem_nmad_extended.png"

CAPELLA_ALTITUDE_M = 525_000.0
GOLD_COLOR = "#4878CF"
FILM_COLOR = "#E87060"
MAX_CDF_PIXELS_PER_PAIR = 500   # limit memory for CDF panel
RNG = np.random.default_rng(42)


# ── physics helpers (mirror compute_metrics.py M4) ──────────────────────

def _h_amb(bperp_m, incidence_deg, center_freq_ghz):
    if abs(bperp_m) < 10.0:
        return float("nan")
    lam = 3e8 / (center_freq_ghz * 1e9)
    theta = np.radians(incidence_deg)
    R = CAPELLA_ALTITUDE_M / np.cos(theta)
    return lam * R * np.sin(theta) / (2.0 * abs(bperp_m))


def _detrend(arr):
    rows, cols = np.indices(arr.shape)
    valid = np.isfinite(arr)
    if valid.sum() < 10:
        return arr
    A = np.column_stack([rows[valid].ravel(), cols[valid].ravel(), np.ones(valid.sum())])
    try:
        coeffs, *_ = np.linalg.lstsq(A, arr[valid].ravel(), rcond=None)
        return arr - (coeffs[0] * rows + coeffs[1] * cols + coeffs[2])
    except Exception:
        return arr


def _copernicus_patch(bbox_w, bbox_s, bbox_e, bbox_n):
    import rasterio
    from rasterio.windows import from_bounds
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with rasterio.open(DEM_PATH) as src:
                win  = from_bounds(bbox_w, bbox_s, bbox_e, bbox_n, src.transform)
                data = src.read(1, window=win).astype(np.float32)
                nd   = src.nodata
            if nd is not None:
                data[data == nd] = float("nan")
            return data if data.size > 0 else None
        except Exception:
            return None


def _load_unw(pair_dir, fname):
    import rasterio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with rasterio.open(pair_dir / fname) as src:
                return src.read(1).astype(np.float32)
        except Exception:
            return None


# ── per-pair data collection ─────────────────────────────────────────────

def collect_pairs(scene_index):
    """
    Returns dict with per-pair data for all valid pairs.
    Each entry: {gold_nmad, film_nmad, bperp, h_amb,
                 gold_errors_sample, film_errors_sample}
    """
    import rasterio  # noqa – ensure available

    pair_dirs = sorted(p for p in PAIRS_DIR.iterdir() if p.is_dir())
    records = []

    for i, pd_dir in enumerate(pair_dirs):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(pair_dirs)}]", flush=True)

        meta_p = pd_dir / "coreg_meta.json"
        if not meta_p.exists():
            continue
        with open(meta_p) as f:
            meta = json.load(f)

        bperp_m       = meta.get("bperp_m", 0.0)
        incidence_deg = meta.get("incidence_angle_deg", 45.0)
        id_ref        = meta.get("id_ref", "")

        scene = scene_index.get(id_ref)
        if scene is None:
            continue

        ha = _h_amb(bperp_m, incidence_deg, float(scene["center_freq_ghz"]))
        if not np.isfinite(ha):
            continue

        h_ref_patch = _copernicus_patch(
            float(scene["bbox_w"]), float(scene["bbox_s"]),
            float(scene["bbox_e"]), float(scene["bbox_n"]),
        )
        if h_ref_patch is None or not np.any(np.isfinite(h_ref_patch)):
            continue
        h_ref_median = float(np.nanmedian(h_ref_patch))

        def _pair_stats(fname):
            unw = _load_unw(pd_dir, fname)
            if unw is None:
                return float("nan"), None
            h = _detrend(unw * ha / (2.0 * np.pi))
            h[~np.isfinite(h)] = float("nan")
            valid = np.isfinite(h)
            if valid.sum() < 100:
                return float("nan"), None
            e = h[valid] - h_ref_median
            nmad = float(1.4826 * np.median(np.abs(e - np.median(e))))
            # sample for CDF
            idx = RNG.choice(len(e), size=min(MAX_CDF_PIXELS_PER_PAIR, len(e)),
                             replace=False)
            return nmad, e[idx]

        g_nmad, g_err = _pair_stats("unw_phase.tif")
        f_nmad, f_err = _pair_stats("unw_phase_film_unet.tif")

        if np.isfinite(g_nmad) and np.isfinite(f_nmad):
            records.append({
                "gold_nmad": g_nmad,
                "film_nmad": f_nmad,
                "bperp":     abs(bperp_m),
                "h_amb":     ha,
                "bbox_w":    float(scene["bbox_w"]),
                "bbox_s":    float(scene["bbox_s"]),
                "bbox_e":    float(scene["bbox_e"]),
                "bbox_n":    float(scene["bbox_n"]),
                "gold_err":  g_err,
                "film_err":  f_err,
            })

    return records


# ── panel helpers ─────────────────────────────────────────────────────────

def panel_dem_overview(ax, records):
    """Copernicus DEM overview (downsampled) + pair footprints."""
    import rasterio
    print("  Loading DEM for overview ...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(DEM_PATH) as src:
            # downsample 20x → ~720×720
            scale = 20
            data = src.read(
                1,
                out_shape=(src.count,
                           src.height // scale,
                           src.width  // scale),
            ).astype(np.float32)
            extent = [src.bounds.left, src.bounds.right,
                      src.bounds.bottom, src.bounds.top]

    data[data < -500] = float("nan")
    vm = np.nanpercentile(data, 98)

    im = ax.imshow(data, extent=extent, origin="upper",
                   cmap="terrain", vmin=0, vmax=vm, aspect="auto")
    plt.colorbar(im, ax=ax, label="Elevation (m)", fraction=0.035, pad=0.03)

    # unique footprints
    seen = set()
    for r in records:
        key = (round(r["bbox_w"], 2), round(r["bbox_s"], 2))
        if key in seen:
            continue
        seen.add(key)
        w, s, e, n = r["bbox_w"], r["bbox_s"], r["bbox_e"], r["bbox_n"]
        rect = mpatches.Rectangle(
            (w, s), e - w, n - s,
            linewidth=0.5, edgecolor="#FF4444", facecolor="none", alpha=0.6,
        )
        ax.add_patch(rect)

    ax.set_xlabel("Longitude (°)", fontsize=10)
    ax.set_ylabel("Latitude (°)", fontsize=10)
    ax.set_title("Copernicus GLO-30 DEM — Hawaii\nwith processed scene footprints", fontsize=10)


def panel_bperp_vs_nmad(ax, records):
    """|B_perp| vs per-pair NMAD with power-law fit."""
    bp    = np.array([r["bperp"]     for r in records])
    g_n   = np.array([r["gold_nmad"] for r in records])
    f_n   = np.array([r["film_nmad"] for r in records])
    h_amb = np.array([r["h_amb"]     for r in records])

    ax.scatter(bp, g_n, s=18, alpha=0.55, color=GOLD_COLOR,
               label="Goldstein", edgecolors="none", zorder=3)
    ax.scatter(bp, f_n, s=18, alpha=0.55, color=FILM_COLOR,
               label="FiLMUNet",  edgecolors="none", zorder=3)

    # theory curve: NMAD ∝ h_amb ∝ 1/|B_perp|  (for constant phase noise)
    bp_sort = np.sort(bp)
    # fit h_amb vs b_perp for a representative pair; scale by median(NMAD / h_amb)
    scale = np.nanmedian(g_n / h_amb)
    # h_amb is already computed per pair; plot median h_amb * scale vs bp
    # use the midpoint pair's reference
    ref_h_amb  = h_amb[np.argmin(np.abs(bp - np.median(bp)))]
    ref_bp     = bp[np.argmin(np.abs(bp - np.median(bp)))]
    theory_y   = g_n[np.argmin(np.abs(bp - np.median(bp)))] * ref_bp / bp_sort
    ax.plot(bp_sort, theory_y, "k--", linewidth=1.2, alpha=0.7,
            label=r"NMAD $\propto 1/|B_\perp|$")

    ax.set_xlabel(r"$|B_\perp|$ (m)", fontsize=10)
    ax.set_ylabel("Per-pair NMAD (m)", fontsize=10)
    ax.set_title(r"Phase-noise amplification: large $h_{\rm amb}$ at small $|B_\perp|$",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # annotation
    ax.text(0.97, 0.97,
            r"NMAD $= h_{\rm amb} \cdot \sigma_\varphi\,/\,(2\pi)$"
            "\n" r"$h_{\rm amb} = \lambda R \sin\theta\,/\,(2|B_\perp|)$",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85,
                      ec="#cccccc"))


def panel_improvement_bar(ax, records):
    """Sorted bar chart of per-pair improvement (Goldstein NMAD − FiLMUNet NMAD)."""
    diff = np.array([r["gold_nmad"] - r["film_nmad"] for r in records])
    diff_sorted = np.sort(diff)
    n = len(diff_sorted)
    colors = [FILM_COLOR if d >= 0 else GOLD_COLOR for d in diff_sorted]

    ax.bar(np.arange(n), diff_sorted, color=colors, width=1.0, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(diff.mean(), color="purple", linewidth=1.2, linestyle="--",
               label=f"Mean gain = {diff.mean():.1f} m")

    frac_pos = (diff >= 0).mean()
    ax.text(0.97, 0.97,
            f"FiLMUNet better: {frac_pos*100:.0f}% of pairs",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85, ec="#cccccc"))

    ax.set_xlabel("Pairs (sorted by improvement)", fontsize=10)
    ax.set_ylabel(r"$\Delta$NMAD = Goldstein $-$ FiLMUNet (m)", fontsize=10)
    ax.set_title("Per-pair NMAD improvement\n(positive = FiLMUNet better)", fontsize=10)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        mpatches.Patch(facecolor=FILM_COLOR, label="FiLMUNet better"),
        mpatches.Patch(facecolor=GOLD_COLOR, label="Goldstein better"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper left")


def panel_cdf(ax, records):
    """CDF of |height error| = |h_insar − h_ref_median|, sampled pixels."""
    g_errors = np.concatenate([r["gold_err"] for r in records
                                if r["gold_err"] is not None])
    f_errors = np.concatenate([r["film_err"] for r in records
                                if r["film_err"] is not None])

    for errors, color, label in [
        (g_errors, GOLD_COLOR, f"Goldstein  (n={len(g_errors)//1000:.0f}k px)"),
        (f_errors, FILM_COLOR, f"FiLMUNet   (n={len(f_errors)//1000:.0f}k px)"),
    ]:
        abs_e = np.abs(errors)
        sorted_e = np.sort(abs_e)
        cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        # clip display to 99th percentile
        p99 = np.percentile(sorted_e, 99)
        mask = sorted_e <= p99
        ax.plot(sorted_e[mask], cdf[mask], color=color, linewidth=1.8, label=label)

    # mark 50th percentile
    for errors, color in [(g_errors, GOLD_COLOR), (f_errors, FILM_COLOR)]:
        p50 = np.percentile(np.abs(errors), 50)
        ax.axvline(p50, color=color, linestyle=":", linewidth=1.2, alpha=0.7)

    ax.set_xlabel(r"$|h_{\rm InSAR} - h_{\rm ref}|$ (m)", fontsize=10)
    ax.set_ylabel("Cumulative fraction", fontsize=10)
    ax.set_title("Height error CDF (sampled valid pixels,\nall pairs aggregated)", fontsize=10)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # annotation: median error
    g_med = np.median(np.abs(g_errors))
    f_med = np.median(np.abs(f_errors))
    ax.text(0.97, 0.06,
            f"Median |error|: Goldstein {g_med:.1f} m, FiLMUNet {f_med:.1f} m",
            transform=ax.transAxes, fontsize=8.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85, ec="#cccccc"))


# ── main ──────────────────────────────────────────────────────────────────

def main():
    print("Loading scene index ...", flush=True)
    df = pd.read_parquet(INDEX_PATH)
    scene_index = {row["id"]: row for _, row in df.iterrows()}

    print("Collecting per-pair data ...", flush=True)
    records = collect_pairs(scene_index)
    print(f"Valid pairs: {len(records)}", flush=True)

    # ── build figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.patch.set_facecolor("white")

    print("Panel 1: DEM overview ...", flush=True)
    panel_dem_overview(axes[0, 0], records)

    print("Panel 2: B_perp vs NMAD ...", flush=True)
    panel_bperp_vs_nmad(axes[0, 1], records)

    print("Panel 3: Improvement bar ...", flush=True)
    panel_improvement_bar(axes[1, 0], records)

    print("Panel 4: Height error CDF ...", flush=True)
    panel_cdf(axes[1, 1], records)

    fig.suptitle(
        "DEM NMAD Analysis: FiLMUNet vs. Goldstein — AOI_000 Hawaii",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved \u2192 {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
