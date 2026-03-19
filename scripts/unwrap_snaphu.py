"""
SNAPHU phase unwrapping for processed Capella InSAR pairs.

Reads ifg_goldstein.tif (2-band Re/Im) and coherence.tif from each pair
directory and saves the unwrapped phase as unw_phase.tif (single-band
float32 GeoTIFF).

Unwrapping backend
------------------
Uses the snaphu-py Python package (pip install snaphu), which bundles the
SNAPHU algorithm as a compiled extension — no separate CLI binary needed.

Install:    pip install snaphu
            # or: conda install -c conda-forge snaphu-py

Usage
-----
# Unwrap all processed pairs:
python scripts/unwrap_snaphu.py --pairs_dir data/processed/pairs

# Limit to 50 pairs with 2 workers:
python scripts/unwrap_snaphu.py \\
    --pairs_dir data/processed/pairs \\
    --max_pairs 50 \\
    --mode DEFO \\
    --workers 2
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# snaphu-py availability check
# ---------------------------------------------------------------------------

def check_snaphu_py() -> bool:
    """Return True if the snaphu Python package is importable."""
    try:
        import snaphu  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Python-API unwrapping
# ---------------------------------------------------------------------------

def unwrap_with_snaphu_py(
    wrapped_phase: np.ndarray,
    coherence: np.ndarray,
    mode: str = "smooth",
    nlooks: float = 9.0,
    nproc: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unwrap phase using the snaphu Python package.

    Parameters
    ----------
    wrapped_phase : (H, W) float32 wrapped phase in radians.
    coherence     : (H, W) float32 coherence in [0, 1].
    mode          : SNAPHU cost model — 'smooth' (default) or 'defo'.
    nlooks        : Equivalent number of independent looks (default 9).
    nproc         : Number of threads for tiled processing.

    Returns
    -------
    unw      : (H, W) float32 unwrapped phase.
    conncomp : (H, W) uint32 connected-component labels (0 = unassigned).
    """
    import snaphu

    # snaphu.unwrap expects a complex interferogram
    igram = (np.cos(wrapped_phase) + 1j * np.sin(wrapped_phase)).astype(np.complex64)

    # Determine tile count for large images
    # 4096×4096 images need ntiles=(4,4) to keep each tile ≤1024×1024;
    # larger tiles (2048×2048) hit SNAPHU's "Exceeded maximum secondary arcs" limit.
    H, W = wrapped_phase.shape
    ntiles = (1, 1)
    if max(H, W) > 8192:
        ntiles = (8, 8)
    elif max(H, W) >= 4096:
        ntiles = (4, 4)
    elif max(H, W) > 2048:
        ntiles = (2, 2)

    # tile_overlap: ≥128 px and ≥10% of tile dimension to satisfy SNAPHU constraints
    tile_h = H // ntiles[0] if ntiles[0] > 1 else H
    tile_w = W // ntiles[1] if ntiles[1] > 1 else W
    tile_overlap = max(128, tile_h // 10, tile_w // 10)

    unw, conncomp = snaphu.unwrap(
        igram,
        coherence.astype(np.float32),
        nlooks=nlooks,
        cost=mode,
        init="mcf",
        ntiles=ntiles,
        tile_overlap=tile_overlap,
        nproc=nproc,
    )
    return np.asarray(unw, dtype=np.float32), np.asarray(conncomp, dtype=np.uint32)


# ---------------------------------------------------------------------------
# End-to-end per-pair processing
# ---------------------------------------------------------------------------

def process_pair(
    pair_dir: Path,
    out_dir: Optional[Path] = None,
    mode: str = "DEFO",
    coh_mask_threshold: float = 0.1,
    nlooks: float = 9.0,
    nproc: int = 1,
) -> Optional[Path]:
    """
    Unwrap one pair directory end-to-end.

    Steps
    -----
    1. Load ifg_goldstein.tif (2-band Re/Im) → wrapped phase via arctan2
    2. Load coherence.tif
    3. Mask low-coherence pixels (set to 0) before unwrapping
    4. Run SNAPHU via snaphu-py Python API
    5. Set NaN where coherence < coh_mask_threshold or conncomp == 0
    6. Save unw_phase.tif preserving CRS/transform from ifg_goldstein.tif

    Parameters
    ----------
    pair_dir : Path
        Processed pair directory containing ifg_goldstein.tif, coherence.tif.
    out_dir : Path, optional
        Where to save unw_phase.tif. Defaults to pair_dir.
    mode : str
        SNAPHU cost mode — 'DEFO' maps to 'smooth', 'TOPO' maps to 'smooth'
        (the snaphu-py API uses 'smooth'/'defo' keywords).
    coh_mask_threshold : float
        Pixels below this coherence are masked out (set to NaN in output).
    nlooks : float
        Equivalent number of independent looks.
    nproc : int
        Number of threads for tiled SNAPHU processing.

    Returns
    -------
    Path to unw_phase.tif on success, None on failure.
    """
    try:
        import rasterio
    except ImportError:
        log.error("rasterio not available.")
        return None

    ifg_path = pair_dir / "ifg_goldstein.tif"
    coh_path = pair_dir / "coherence.tif"

    if not ifg_path.exists() or not coh_path.exists():
        log.warning("Missing files in %s — skipping.", pair_dir.name)
        return None

    # --- Load interferogram ---
    try:
        with rasterio.open(ifg_path) as src:
            re_band = src.read(1).astype(np.float32)
            im_band = src.read(2).astype(np.float32)
            profile = src.profile.copy()
        with rasterio.open(coh_path) as src:
            coherence = src.read(1).astype(np.float32)
    except Exception as e:
        log.warning("Read error in %s: %s", pair_dir.name, e)
        return None

    wrapped_phase = np.arctan2(im_band, re_band)

    # --- Coherence mask: zero out unreliable pixels ---
    low_coh_mask = coherence < coh_mask_threshold
    wrapped_masked = wrapped_phase.copy()
    coh_masked = coherence.copy()
    wrapped_masked[low_coh_mask] = 0.0
    coh_masked[low_coh_mask] = 0.0

    # Map mode string to snaphu-py cost keyword
    cost = "defo" if mode.upper() == "DEFO" else "smooth"

    # --- Unwrap ---
    try:
        unw, conncomp = unwrap_with_snaphu_py(
            wrapped_masked, coh_masked, mode=cost, nlooks=nlooks, nproc=nproc
        )
    except Exception as e:
        log.error("SNAPHU failed for %s: %s", pair_dir.name, e)
        return None

    # --- Post-process: NaN where low coherence or unconnected ---
    unw[low_coh_mask] = float("nan")
    unw[conncomp == 0] = float("nan")

    # --- Save unw_phase.tif ---
    dest_dir = out_dir if out_dir is not None else pair_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / "unw_phase.tif"

    out_profile = {
        "driver":     "GTiff",
        "dtype":      "float32",
        "count":      1,
        "height":     unw.shape[0],
        "width":      unw.shape[1],
        "compress":   "deflate",
        "tiled":      True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if profile.get("crs"):
        out_profile["crs"] = profile["crs"]
    if profile.get("transform"):
        out_profile["transform"] = profile["transform"]

    try:
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(unw, 1)
    except Exception as e:
        log.error("Failed to save %s: %s", out_path, e)
        return None

    log.info("Saved %s (nan_frac=%.2f%%)",
             out_path, 100 * float(np.isnan(unw).mean()))
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch SNAPHU phase unwrapping for processed Capella pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pairs_dir", required=True,
                   help="Root directory containing per-pair subdirectories.")
    p.add_argument("--out_dir", default=None,
                   help="Output directory (default: same as each pair_dir).")
    p.add_argument("--max_pairs", type=int, default=None,
                   help="Maximum number of pairs to process.")
    p.add_argument("--mode", choices=["DEFO", "TOPO"], default="DEFO",
                   help="SNAPHU cost mode (DEFO=deformation, TOPO=topographic).")
    p.add_argument("--coh_threshold", type=float, default=0.1,
                   help="Coherence below which pixels are masked (set to NaN in output).")
    p.add_argument("--nlooks", type=float, default=9.0,
                   help="Equivalent number of independent looks.")
    p.add_argument("--workers", type=int, default=2,
                   help="Number of parallel worker threads.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not check_snaphu_py():
        log.error(
            "snaphu Python package not found.\n"
            "Install with: pip install snaphu\n"
            "  or: conda install -c conda-forge snaphu-py"
        )
        raise SystemExit(1)

    pairs_dir = Path(args.pairs_dir)
    if not pairs_dir.is_dir():
        log.error("pairs_dir does not exist: %s", pairs_dir)
        raise SystemExit(1)

    pair_dirs = sorted(p for p in pairs_dir.iterdir() if p.is_dir())
    # Skip already unwrapped pairs
    pair_dirs = [d for d in pair_dirs if not (d / "unw_phase.tif").exists()]
    log.info("Found %d pair directories to unwrap", len(pair_dirs))

    if args.max_pairs:
        pair_dirs = pair_dirs[: args.max_pairs]
        log.info("Capped at %d pairs", len(pair_dirs))

    out_dir = Path(args.out_dir) if args.out_dir else None

    ok_count = fail_count = 0

    def _unwrap(pd_dir: Path) -> bool:
        result = process_pair(
            pd_dir,
            out_dir=out_dir,
            mode=args.mode,
            coh_mask_threshold=args.coh_threshold,
            nlooks=args.nlooks,
        )
        return result is not None

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_unwrap, d): d.name for d in pair_dirs}
        for fut in concurrent.futures.as_completed(futures):
            name = futures[fut]
            try:
                success = fut.result()
            except Exception as e:
                log.error("Unexpected error for %s: %s", name, e)
                success = False
            if success:
                ok_count += 1
                log.info("OK  %s", name)
            else:
                fail_count += 1
                log.warning("FAIL %s", name)

    log.info("Done. %d succeeded, %d failed.", ok_count, fail_count)


if __name__ == "__main__":
    main()
