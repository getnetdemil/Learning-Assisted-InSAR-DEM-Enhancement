"""
Coregistration, interferogram formation, coherence estimation, and Goldstein
filtering for Capella SLC pairs.

Reads raw SLC COG GeoTIFFs (CInt16), estimates sub-pixel coregistration offsets
via phase cross-correlation, forms the complex interferogram, estimates coherence,
and saves results as GeoTIFF.

Since ISCE3 / capella-reader are not available, this uses rasterio for I/O and
a NumPy/SciPy cross-correlation coregistration.

Usage
-----
# Process top-100 pairs by Q_ij for Hawaii:
python scripts/preprocess_pairs.py \\
    --pairs_manifest data/manifests/hawaii_pairs.parquet \\
    --raw_dir data/raw/AOI_000 \\
    --out_dir data/processed/AOI_000 \\
    --max_pairs 100 \\
    --patch_size 4096 \\
    --n_workers 4

# Process a single pair (for debugging):
python scripts/preprocess_pairs.py \\
    --pairs_manifest data/manifests/hawaii_pairs.parquet \\
    --raw_dir data/raw/AOI_000 \\
    --out_dir data/processed/AOI_000 \\
    --max_pairs 1
"""

import argparse
import concurrent.futures
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from src.insar_processing.filters import (
    adaptive_goldstein,
    boxcar_coherence,
    coherence_from_ifg,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def find_slc(raw_dir: Path, collect_id: str) -> Path | None:
    """Return the .tif path for a collect ID, or None if not found."""
    d = raw_dir / collect_id
    if not d.exists():
        return None
    tifs = list(d.glob("*.tif"))
    return tifs[0] if tifs else None


def read_slc_patch(
    tif_path: Path,
    row_off: int,
    col_off: int,
    rows: int,
    cols: int,
) -> np.ndarray:
    """
    Read a complex patch from a Capella SLC GeoTIFF.

    Returns complex64 array shape (rows, cols), or raises on error.
    rasterio reads CInt16 as complex64 automatically.
    """
    with rasterio.open(tif_path) as src:
        row_off = max(0, min(row_off, src.height - rows))
        col_off = max(0, min(col_off, src.width - cols))
        window = Window(col_off, row_off, cols, rows)
        data = src.read(1, window=window)
    return data.astype(np.complex64)


def get_slc_shape(tif_path: Path) -> tuple[int, int]:
    """Return (height, width) of an SLC without reading data."""
    with rasterio.open(tif_path) as src:
        return src.height, src.width


# ---------------------------------------------------------------------------
# Coregistration
# ---------------------------------------------------------------------------

def estimate_offset_cc(
    ref: np.ndarray,
    sec: np.ndarray,
    upsample_factor: int = 10,
) -> tuple[float, float, float]:
    """
    Estimate (row_offset, col_offset) to align `sec` to `ref` via
    phase cross-correlation in the frequency domain.

    Uses a sub-set of the image (centre region) for speed.

    Parameters
    ----------
    ref, sec : np.ndarray   complex or real 2-D arrays (same shape)
    upsample_factor : int   sub-pixel precision: 1/upsample_factor pixels

    Returns
    -------
    (row_offset, col_offset, cc_peak_score) : floats
        shift to apply to sec to align to ref; cc_peak_score ∈ [0,1]
    """
    # Use intensity images for cross-correlation (more stable than phase)
    ref_int = np.abs(ref).astype(np.float32)
    sec_int = np.abs(sec).astype(np.float32)

    # Subtract mean to suppress DC component
    ref_int -= ref_int.mean()
    sec_int -= sec_int.mean()

    # Cross-correlation via FFT
    R = np.fft.fft2(ref_int) * np.conj(np.fft.fft2(sec_int))
    R /= (np.abs(R) + 1e-10)           # normalised cross-correlation (phase only)
    cc = np.fft.ifft2(R).real

    # Find peak
    rows, cols = cc.shape
    peak_flat = np.argmax(cc)
    r_peak, c_peak = np.unravel_index(peak_flat, (rows, cols))

    # CC quality score: peak / max → always [0, 1]
    cc_max = float(cc.max())
    cc_peak_score = float(cc[r_peak, c_peak] / cc_max) if cc_max > 0 else 0.0

    # Wrap offsets to (-N/2, N/2)
    r_off = r_peak if r_peak < rows // 2 else r_peak - rows
    c_off = c_peak if c_peak < cols // 2 else c_peak - cols

    if upsample_factor <= 1:
        return float(r_off), float(c_off), cc_peak_score

    # Sub-pixel refinement via DFT upsampling in a small neighbourhood
    r_off_sub, c_off_sub = _subpixel_offset(R, r_off, c_off, upsample_factor)
    return r_off_sub, c_off_sub, cc_peak_score


def _subpixel_offset(
    R_norm: np.ndarray,
    r_int: int,
    c_int: int,
    upsample: int,
) -> tuple[float, float]:
    """
    Sub-pixel offset refinement via partial DFT upsampling (Guizar-Sicairos 2008).
    Evaluates the cross-correlation in a (1.5 × 1.5)-pixel region around (r_int, c_int)
    at upsample × resolution.
    """
    rows, cols = R_norm.shape
    region = upsample * 1.5

    # Frequency indices
    freq_r = np.fft.ifftshift(np.arange(-np.fix(rows / 2), np.ceil(rows / 2))) / rows
    freq_c = np.fft.ifftshift(np.arange(-np.fix(cols / 2), np.ceil(cols / 2))) / cols

    up_rows = int(np.ceil(region * upsample))
    up_cols = int(np.ceil(region * upsample))

    row_shift = np.arange(-np.fix(up_rows / 2), np.ceil(up_rows / 2)) / upsample + r_int
    col_shift = np.arange(-np.fix(up_cols / 2), np.ceil(up_cols / 2)) / upsample + c_int

    # Partial DFT: CC(row_shift, col_shift) = sum_k sum_l R(k,l) exp(2πi(k*dr + l*dc))
    kern_r = np.exp(2j * np.pi * np.outer(row_shift, freq_r))
    kern_c = np.exp(2j * np.pi * np.outer(freq_c, col_shift))
    cc_up = (kern_r @ R_norm @ kern_c).real

    peak_flat = np.argmax(cc_up)
    rp, cp = np.unravel_index(peak_flat, cc_up.shape)
    return float(row_shift[rp]), float(col_shift[cp])


def apply_shift(
    slc: np.ndarray,
    row_off: float,
    col_off: float,
) -> np.ndarray:
    """
    Shift a complex SLC by (row_off, col_off) pixels via FFT phase ramp.

    Fractional shifts are handled by applying a linear phase ramp in the
    frequency domain — exact for band-limited signals.
    """
    rows, cols = slc.shape
    fr = np.fft.fftfreq(rows)[:, np.newaxis]
    fc = np.fft.fftfreq(cols)[np.newaxis, :]
    phase_ramp = np.exp(-2j * np.pi * (row_off * fr + col_off * fc))
    shifted = np.fft.ifft2(np.fft.fft2(slc) * phase_ramp)
    return shifted.astype(slc.dtype)


def estimate_offset_grid(
    ref: np.ndarray,
    sec: np.ndarray,
    n_grid: int = 3,
    patch_frac: float = 0.15,
    upsample_factor: int = 10,
    min_cc_score: float = 0.05,
) -> dict:
    """
    Estimate coregistration offset from a n_grid × n_grid regular patch grid.

    Patches are placed at evenly-spaced positions across the image (avoiding
    the first and last 20% of each axis).  Each patch contributes an
    independent (row_off, col_off, cc_score) estimate.  The returned offset
    is the CC-score-weighted mean over accepted patches.

    Parameters
    ----------
    ref, sec        : 2-D complex arrays (same shape)
    n_grid          : grid dimension (default 3 → 9 patches)
    patch_frac      : patch side = max(H,W) * patch_frac (default 0.15 → ~614 px for 4096)
    min_cc_score    : patches below this CC score are excluded from the mean

    Returns
    -------
    dict with keys:
        row_offset_px           float  weighted-mean row shift
        col_offset_px           float  weighted-mean col shift
        cc_peak_mean            float  mean CC score of accepted patches
        cc_peak_min             float  min  CC score of accepted patches
        n_patches_ok            int    number of patches above min_cc_score
        offset_row_std          float  std of row offsets across patches (px)
        offset_col_std          float  std of col offsets across patches (px)
        estimated_rotation_mrad float  rotation from affine fit (informational only)
    """
    H, W = ref.shape
    patch_size = max(64, int(max(H, W) * patch_frac))

    # Grid centres at evenly-spaced positions, avoiding the outer 20% of each axis
    row_frac = np.linspace(0.2, 0.8, n_grid)
    col_frac = np.linspace(0.2, 0.8, n_grid)

    offsets_r, offsets_c, scores = [], [], []
    cx_list, cy_list = [], []  # patch centre positions (for affine rotation fit)

    for rf in row_frac:
        for cf in col_frac:
            r0 = int(rf * H) - patch_size // 2
            c0 = int(cf * W) - patch_size // 2
            r0 = max(0, min(r0, H - patch_size))
            c0 = max(0, min(c0, W - patch_size))
            r1, c1 = r0 + patch_size, c0 + patch_size

            patch_r = ref[r0:r1, c0:c1]
            patch_s = sec[r0:r1, c0:c1]

            try:
                dr, dc, score = estimate_offset_cc(patch_r, patch_s, upsample_factor)
            except Exception:
                continue

            offsets_r.append(dr)
            offsets_c.append(dc)
            scores.append(score)
            cx_list.append(cf * W)
            cy_list.append(rf * H)

    if not offsets_r:
        # Fallback to centre patch
        dr, dc, score = estimate_offset_cc(ref, sec, upsample_factor)
        return {
            "row_offset_px": dr, "col_offset_px": dc,
            "cc_peak_mean": score, "cc_peak_min": score,
            "n_patches_ok": 1, "offset_row_std": 0.0, "offset_col_std": 0.0,
            "estimated_rotation_mrad": 0.0,
        }

    offsets_r = np.array(offsets_r)
    offsets_c = np.array(offsets_c)
    scores    = np.array(scores)

    # Filter by min CC score
    ok = scores >= min_cc_score
    if ok.sum() == 0:
        ok = np.ones(len(scores), dtype=bool)  # accept all if none pass threshold

    offsets_r_ok = offsets_r[ok]
    offsets_c_ok = offsets_c[ok]
    scores_ok    = scores[ok]
    cx_ok        = np.array(cx_list)[ok]
    cy_ok        = np.array(cy_list)[ok]

    # CC-score-weighted mean offset
    w = scores_ok / scores_ok.sum()
    mean_dr = float((w * offsets_r_ok).sum())
    mean_dc = float((w * offsets_c_ok).sum())

    # Consistency: std of offsets (near 0 for pure translation / flat terrain)
    std_r = float(offsets_r_ok.std()) if len(offsets_r_ok) > 1 else 0.0
    std_c = float(offsets_c_ok.std()) if len(offsets_c_ok) > 1 else 0.0

    # Affine rotation estimate (informational only — NOT applied):
    # Fit dr = a0 + a1*cx + a2*cy; dc = b0 + b1*cx + b2*cy
    # rotation ≈ (a2 - b1) / 2 [rad/px] → mrad/1000px
    rotation_mrad = 0.0
    if len(offsets_r_ok) >= 3:
        try:
            A_fit = np.column_stack([np.ones(len(cx_ok)), cx_ok, cy_ok])
            coef_r, _, _, _ = np.linalg.lstsq(A_fit, offsets_r_ok, rcond=None)
            coef_c, _, _, _ = np.linalg.lstsq(A_fit, offsets_c_ok, rcond=None)
            rotation_mrad = float((coef_r[2] - coef_c[1]) / 2 * 1000)
        except Exception:
            pass

    return {
        "row_offset_px":            mean_dr,
        "col_offset_px":            mean_dc,
        "cc_peak_mean":             float(scores_ok.mean()),
        "cc_peak_min":              float(scores_ok.min()),
        "n_patches_ok":             int(ok.sum()),
        "offset_row_std":           std_r,
        "offset_col_std":           std_c,
        "estimated_rotation_mrad":  rotation_mrad,
    }


# ---------------------------------------------------------------------------
# Interferogram + coherence
# ---------------------------------------------------------------------------

def form_interferogram(
    slc_ref: np.ndarray,
    slc_sec: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Form complex interferogram: ifg = slc_ref × conj(slc_sec).

    If normalize=True, each pixel is normalised to unit amplitude, isolating
    the interferometric phase (useful for training data).
    """
    ifg = slc_ref * np.conj(slc_sec)
    if normalize:
        amp = np.abs(ifg)
        ifg = ifg / (amp + 1e-10)
    return ifg.astype(np.complex64)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_complex_tif(arr: np.ndarray, path: Path, crs=None, transform=None) -> None:
    """Save a complex float32 array as a 2-band GeoTIFF (Re, Im)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 2,
        "height": arr.shape[0],
        "width": arr.shape[1],
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if crs:
        profile["crs"] = crs
    if transform:
        profile["transform"] = transform

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.real.astype(np.float32), 1)
        dst.write(arr.imag.astype(np.float32), 2)


def save_float_tif(arr: np.ndarray, path: Path, crs=None, transform=None) -> None:
    """Save a real float32 array as a single-band GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": arr.shape[0],
        "width": arr.shape[1],
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if crs:
        profile["crs"] = crs
    if transform:
        profile["transform"] = transform

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


# ---------------------------------------------------------------------------
# Single-pair processing
# ---------------------------------------------------------------------------

def process_pair(
    row: pd.Series,
    raw_dir: Path,
    out_dir: Path,
    patch_size: int,
    looks_range: int,
    looks_azimuth: int,
    goldstein_alpha: float,
    use_adaptive: bool,
    coreg_n_grid: int = 3,
) -> dict:
    """
    Process one interferometric pair end-to-end.

    Steps
    -----
    1. Find SLC .tif files
    2. Determine overlapping patch region (centre of both images)
    3. Read patches
    4. Estimate coregistration offset
    5. Shift secondary SLC
    6. Form interferogram
    7. Estimate coherence
    8. Apply Goldstein filter
    9. Save: raw ifg, coherence, filtered ifg

    Returns
    -------
    dict with "ok": bool and optional "error": str
    """
    id_ref, id_sec = row["id_ref"], row["id_sec"]

    # --- Locate SLC files ---
    tif_ref = find_slc(raw_dir, id_ref)
    tif_sec = find_slc(raw_dir, id_sec)
    if tif_ref is None or tif_sec is None:
        return {"ok": False, "error": f"Missing SLC: ref={tif_ref} sec={tif_sec}"}

    # --- Determine centre patch (use smaller of the two images) ---
    h_ref, w_ref = get_slc_shape(tif_ref)
    h_sec, w_sec = get_slc_shape(tif_sec)
    ps = patch_size
    row_r = max(0, h_ref // 2 - ps // 2)
    col_r = max(0, w_ref // 2 - ps // 2)
    row_s = max(0, h_sec // 2 - ps // 2)
    col_s = max(0, w_sec // 2 - ps // 2)

    try:
        slc_ref = read_slc_patch(tif_ref, row_r, col_r, ps, ps)
        slc_sec = read_slc_patch(tif_sec, row_s, col_s, ps, ps)
    except Exception as e:
        return {"ok": False, "error": f"Read error: {e}"}

    # --- Coregistration ---
    coreg_stats: dict = {}
    dr, dc = 0.0, 0.0
    try:
        result = estimate_offset_grid(slc_ref, slc_sec, n_grid=coreg_n_grid, upsample_factor=10)
        dr = result["row_offset_px"]
        dc = result["col_offset_px"]
        coreg_stats = result
        log.debug(
            "  Coreg offset: dr=%.3f dc=%.3f px | cc_mean=%.3f cc_min=%.3f n=%d "
            "std_r=%.3f std_c=%.3f rot=%.2f mrad",
            dr, dc,
            result["cc_peak_mean"], result["cc_peak_min"], result["n_patches_ok"],
            result["offset_row_std"], result["offset_col_std"],
            result["estimated_rotation_mrad"],
        )
        slc_sec_coreg = apply_shift(slc_sec, dr, dc)
    except Exception as e:
        log.warning("Coregistration failed for %s/%s: %s", id_ref[:30], id_sec[:30], e)
        slc_sec_coreg = slc_sec

    # --- Interferogram ---
    ifg = form_interferogram(slc_ref, slc_sec_coreg, normalize=False)

    # --- Coherence ---
    ifg_raw = form_interferogram(slc_ref, slc_sec_coreg, normalize=False)
    coherence = boxcar_coherence(
        ifg_raw, slc_ref, slc_sec_coreg,
        looks_range=looks_range, looks_azimuth=looks_azimuth,
    )

    # --- Goldstein filter ---
    try:
        if use_adaptive:
            ifg_filtered = adaptive_goldstein(
                ifg_raw, coherence,
                block_size=32, overlap=8,
            )
        else:
            from src.insar_processing.filters import goldstein
            ifg_filtered = goldstein(ifg_raw, alpha=goldstein_alpha, block_size=32, overlap=8)
    except Exception as e:
        log.warning("Goldstein filter failed: %s", e)
        ifg_filtered = ifg_raw

    # --- Save ---
    pair_id = f"{id_ref}__{id_sec}"
    pair_dir = out_dir / pair_id  # full ID (104 chars max, well under 255 limit)
    pair_dir.mkdir(parents=True, exist_ok=True)

    save_complex_tif(ifg_raw.astype(np.complex64), pair_dir / "ifg_raw.tif")
    save_float_tif(coherence, pair_dir / "coherence.tif")
    save_complex_tif(ifg_filtered.astype(np.complex64), pair_dir / "ifg_goldstein.tif")

    # Save coregistration metadata
    import json
    with open(pair_dir / "coreg_meta.json", "w") as f:
        json.dump({
            "id_ref":              id_ref,
            "id_sec":              id_sec,
            "dt_days":             float(row.get("dt_days", 0)),
            "dinc_deg":            float(row.get("dinc_deg", 0)),
            "q_score":             float(row.get("q_score", 0)),
            "bperp_m":             float(row.get("bperp_m", 0)) if "bperp_m" in row.index else None,
            "row_offset_px":       dr,
            "col_offset_px":       dc,
            "patch_size":          ps,
            "patch_row_ref":       row_r,
            "patch_col_ref":       col_r,
            # Coregistration quality metrics (multi-patch grid)
            "cc_peak_mean":        coreg_stats.get("cc_peak_mean",  None),
            "cc_peak_min":         coreg_stats.get("cc_peak_min",   None),
            "n_coreg_patches":     coreg_stats.get("n_patches_ok",  None),
            "offset_row_std_px":   coreg_stats.get("offset_row_std", None),
            "offset_col_std_px":   coreg_stats.get("offset_col_std", None),
        }, f, indent=2)

    return {"ok": True, "pair_dir": str(pair_dir), "mean_coherence": float(coherence.mean())}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess Capella SLC pairs → interferograms.")
    p.add_argument("--pairs_manifest", required=True, help="Path to hawaii_pairs.parquet")
    p.add_argument("--raw_dir", required=True, help="Root dir with per-collect subdirs")
    p.add_argument("--out_dir", required=True, help="Output directory for processed pairs")
    p.add_argument("--max_pairs", type=int, default=None, help="Max pairs to process")
    p.add_argument("--dt_max", type=float, default=7.0, help="Max Δt filter (days)")
    p.add_argument("--coreg_n_grid", type=int, default=3,
                   help="Grid dimension for multi-patch coregistration (default 3 = 3×3=9 patches). "
                        "Set 1 to use single centre patch (legacy).")
    p.add_argument("--dinc_max", type=float, default=2.0, help="Max Δinc filter (deg)")
    p.add_argument("--patch_size", type=int, default=4096, help="SLC patch size (pixels)")
    p.add_argument("--looks_range", type=int, default=5, help="Range looks for coherence")
    p.add_argument("--looks_azimuth", type=int, default=5, help="Azimuth looks for coherence")
    p.add_argument("--goldstein_alpha", type=float, default=0.5, help="Goldstein α [0-1]")
    p.add_argument("--adaptive", action="store_true", help="Use coherence-adaptive Goldstein")
    p.add_argument("--n_workers", type=int, default=4, help="Parallel workers")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pairs = pd.read_parquet(args.pairs_manifest)
    pairs["datetime_ref"] = pd.to_datetime(pairs["datetime_ref"], utc=True)
    pairs["datetime_sec"] = pd.to_datetime(pairs["datetime_sec"], utc=True)

    # Filter to manageable subset
    subset = pairs[
        (pairs["dt_days"] <= args.dt_max) &
        (pairs["dinc_deg"] <= args.dinc_max)
    ].copy()
    log.info("Pairs after filter (dt≤%dd, Δinc≤%.1f°): %d", args.dt_max, args.dinc_max, len(subset))

    if args.max_pairs:
        subset = subset.head(args.max_pairs)
        log.info("Capped at %d pairs", len(subset))

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok_count = fail_count = 0
    coh_values = []

    def _process(row):
        return process_pair(
            row, raw_dir, out_dir,
            patch_size=args.patch_size,
            looks_range=args.looks_range,
            looks_azimuth=args.looks_azimuth,
            goldstein_alpha=args.goldstein_alpha,
            use_adaptive=args.adaptive,
            coreg_n_grid=args.coreg_n_grid,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {
            ex.submit(_process, row): row["id_ref"][:30]
            for _, row in subset.iterrows()
        }
        for fut in concurrent.futures.as_completed(futures):
            pair_id = futures[fut]
            result = fut.result()
            if result["ok"]:
                ok_count += 1
                coh_values.append(result.get("mean_coherence", 0))
                log.info("OK  %s  coh=%.3f", pair_id, result.get("mean_coherence", 0))
            else:
                fail_count += 1
                log.warning("FAIL %s: %s", pair_id, result.get("error", ""))

    log.info("Done. %d OK, %d failed.", ok_count, fail_count)
    if coh_values:
        log.info("Mean coherence across pairs: %.3f", np.mean(coh_values))


if __name__ == "__main__":
    main()
