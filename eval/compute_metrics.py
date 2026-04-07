#!/usr/bin/env python3
"""
eval/compute_metrics.py — Contest metric evaluation pipeline.

Computes all 5 IEEE GRSS 2026 Data Fusion Contest metrics for two methods:
  • goldstein  — Goldstein-filtered interferogram (baseline)
  • film_unet  — FiLMUNet denoised interferogram (model)

Outputs
-------
  {out_dir}/metrics_comparison.csv
  {out_dir}/figures/closure_histogram.png
  {out_dir}/figures/phase_comparison.png
  {out_dir}/figures/temporal_residual_bar.png

Usage
-----
python eval/compute_metrics.py \\
    --checkpoint experiments/enhanced/checkpoints/film_unet/best_closure.pt \\
    --pairs_dir data/processed/pairs \\
    --triplets_manifest data/manifests/hawaii_triplets_strict.parquet \\
    --out_dir experiments/enhanced/outputs \\
    [--skip_inference] \\
    [--skip_snaphu_metrics] \\
    [--test_only]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.film_unet import FiLMUNet
from evaluation.closure_metrics import (
    triplet_closure_error,
    unwrap_success_rate,
    usable_pairs_fraction,
    temporal_consistency_residual,
)
from evaluation.dem_metrics import nmad as _nmad

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# Metadata normalisation — must match InSARTileDataset._load_meta
_META_MEAN = np.array([30.0, 45.0, 35.0, 500.0, 0.5, 0.5, 0.5], dtype=np.float32)
_META_STD  = np.array([60.0,  8.0,  8.0, 2000.0, 0.5, 0.5, 0.3], dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_meta_normalised(pair_dir: Path) -> np.ndarray:
    """Load and z-score normalise metadata from coreg_meta.json."""
    meta_path = pair_dir / "coreg_meta.json"
    try:
        with open(meta_path) as f:
            m = json.load(f)
        dt    = float(m.get("dt_days", 30.0))
        inc   = float(m.get("incidence_angle_deg", 45.0))
        graze = 90.0 - inc
        bperp = float(m.get("bperp_m", 500.0))
        mode  = 1.0 if str(m.get("mode", "")).upper() == "SL" else 0.0
        look  = 1.0 if str(m.get("look_direction", "")).upper() == "RIGHT" else 0.0
        snr   = float(m.get("snr_proxy", 0.5))
        raw   = np.array([dt, inc, graze, bperp, mode, look, snr], dtype=np.float32)
    except Exception:
        raw = _META_MEAN.copy()
    return (raw - _META_MEAN) / (_META_STD + 1e-8)


def _pair_date(pair_dir: Path) -> str:
    """Extract YYYYMMDD reference date from pair directory name or coreg_meta."""
    meta_path = pair_dir / "coreg_meta.json"
    try:
        with open(meta_path) as f:
            m = json.load(f)
        id_ref = m.get("id_ref", pair_dir.name)
        parts = id_ref.split("_")
        # Pattern: CAPELLA_C13_SP_SLC_HH_YYYYMMDDHHMMSS_...
        for part in parts:
            if len(part) == 14 and part.isdigit():
                return part[:8]
    except Exception:
        pass
    # Fallback: first 8 chars of dir name
    return pair_dir.name[:8]


def _discover_pairs(pairs_dir: Path) -> list[Path]:
    """Return sorted pair dirs that contain ifg_goldstein_complex_real_imag.tif."""
    return sorted(
        p for p in pairs_dir.iterdir()
        if p.is_dir() and (p / "ifg_goldstein_complex_real_imag.tif").exists()
    )


def _temporal_split_test(pair_dirs: list[Path], test_frac: float = 0.15) -> list[Path]:
    """Return the last test_frac fraction of pairs sorted by reference date."""
    sorted_dirs = sorted(pair_dirs, key=_pair_date)
    n_test = max(1, int(len(sorted_dirs) * test_frac))
    return sorted_dirs[-n_test:]


def _load_model(checkpoint: str, device: torch.device) -> FiLMUNet:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("configs", {}).get("model", {})
    # Infer in_channels from first conv weight shape
    state = ckpt["model_state"]
    first_key = next(iter(state))
    in_ch = cfg.get("in_channels", 3)
    for k, v in state.items():
        if "downs.0.conv1.weight" in k:
            in_ch = v.shape[1]
            break
    model = FiLMUNet(
        in_channels=in_ch,
        metadata_dim=cfg.get("metadata_dim", 7),
        features=cfg.get("features", [32, 64, 128, 256]),
        embed_dim=cfg.get("embed_dim", 64),
    )
    model.load_state_dict(state)
    model.eval()
    log.info("Loaded FiLMUNet (in_channels=%d) from %s", in_ch, checkpoint)
    return model.to(device)


# ---------------------------------------------------------------------------
# Sliding-window inference
# ---------------------------------------------------------------------------

def run_inference_on_pair(
    model: FiLMUNet,
    pair_dir: Path,
    device: torch.device,
    tile_size: int = 256,
    stride: int = 128,
    batch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tile → forward pass → stitch with Hanning overlap-add.

    Parameters
    ----------
    batch_size : int
        Number of tiles to stack into a single GPU forward pass.
        Higher values use more VRAM but run faster (e.g. 16–32 for 24 GB).

    Returns
    -------
    denoised_re_im : (H, W, 2) float32
    log_var        : (H, W)    float32
    """
    import rasterio

    raw_path = pair_dir / "ifg_raw_complex_real_imag.tif"
    ifg_src = raw_path if raw_path.exists() else pair_dir / "ifg_goldstein_complex_real_imag.tif"
    with rasterio.open(ifg_src) as src:
        re  = src.read(1).astype(np.float32)
        im  = src.read(2).astype(np.float32)
    with rasterio.open(pair_dir / "coherence.tif") as src:
        coh = src.read(1).astype(np.float32)

    H, W = re.shape
    meta_vec = torch.from_numpy(_load_meta_normalised(pair_dir)).float().to(device)  # (7,)

    # Accumulation buffers
    out_re = np.zeros((H, W), dtype=np.float32)
    out_im = np.zeros((H, W), dtype=np.float32)
    out_lv = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    win = np.outer(np.hanning(tile_size), np.hanning(tile_size)).astype(np.float32)

    # Tile origin rows/cols
    rows = list(range(0, max(1, H - tile_size + 1), stride))
    cols = list(range(0, max(1, W - tile_size + 1), stride))
    if not rows or rows[-1] + tile_size < H:
        rows.append(max(0, H - tile_size))
    if not cols or cols[-1] + tile_size < W:
        cols.append(max(0, W - tile_size))

    # Flatten all tile positions
    positions = [(r, c) for r in rows for c in cols]

    def _process_batch(batch_tiles, batch_pos):
        """Run one batched forward pass and scatter results."""
        x = torch.stack(batch_tiles, dim=0).to(device)          # (B, 3, T, T)
        meta_b = meta_vec.unsqueeze(0).expand(len(batch_tiles), -1)  # (B, 7)
        pred, lv = model(x, meta_b)                              # (B,2,T,T), (B,1,T,T)
        p_np = pred.cpu().numpy()                                # (B, 2, T, T)
        l_np = lv[:, 0].cpu().numpy()                           # (B, T, T)
        for i, (r, c) in enumerate(batch_pos):
            r2 = min(r + tile_size, H)
            c2 = min(c + tile_size, W)
            th, tw = r2 - r, c2 - c
            w = win[:th, :tw]
            out_re[r:r2, c:c2] += p_np[i, 0, :th, :tw] * w
            out_im[r:r2, c:c2] += p_np[i, 1, :th, :tw] * w
            out_lv[r:r2, c:c2] += l_np[i,    :th, :tw] * w
            weight[r:r2, c:c2] += w

    batch_tiles: list = []
    batch_pos:   list = []

    with torch.no_grad():
        for r, c in positions:
            r2 = min(r + tile_size, H)
            c2 = min(c + tile_size, W)
            th, tw = r2 - r, c2 - c

            re_t  = re[r:r2, c:c2]
            im_t  = im[r:r2, c:c2]
            coh_t = coh[r:r2, c:c2]

            if th < tile_size or tw < tile_size:
                def _pad(a):
                    return np.pad(a, ((0, tile_size - th), (0, tile_size - tw)))
                re_t, im_t, coh_t = _pad(re_t), _pad(im_t), _pad(coh_t)

            batch_tiles.append(torch.from_numpy(np.stack([re_t, im_t, coh_t], axis=0)).float())
            batch_pos.append((r, c))

            if len(batch_tiles) == batch_size:
                _process_batch(batch_tiles, batch_pos)
                batch_tiles, batch_pos = [], []

        if batch_tiles:  # flush remaining tiles
            _process_batch(batch_tiles, batch_pos)

    eps = 1e-8
    out_re /= (weight + eps)
    out_im /= (weight + eps)
    out_lv /= (weight + eps)

    return np.stack([out_re, out_im], axis=-1), out_lv


def save_inference_outputs(
    pair_dir: Path,
    denoised_re_im: np.ndarray,
    log_var: np.ndarray,
) -> None:
    """Save ifg_film_unet.tif (2-band) and log_var.tif preserving georeferencing."""
    import rasterio

    with rasterio.open(pair_dir / "ifg_goldstein_complex_real_imag.tif") as src:
        profile = src.profile.copy()

    H, W = log_var.shape

    # 2-band interferogram
    ifg_profile = {**profile, "count": 2, "dtype": "float32", "compress": "deflate", "BIGTIFF": "YES"}
    with rasterio.open(pair_dir / "ifg_film_unet.tif", "w", **ifg_profile) as dst:
        dst.write(denoised_re_im[:, :, 0], 1)
        dst.write(denoised_re_im[:, :, 1], 2)

    # 1-band log-variance
    lv_profile = {**profile, "count": 1, "dtype": "float32", "compress": "deflate", "BIGTIFF": "YES"}
    with rasterio.open(pair_dir / "log_var.tif", "w", **lv_profile) as dst:
        dst.write(log_var, 1)


# ---------------------------------------------------------------------------
# Phase loading helpers
# ---------------------------------------------------------------------------

def _load_phase(pair_dir: Path, method: str) -> Optional[np.ndarray]:
    """Load wrapped phase (H, W) for the given method.

    For large full-image rasters (>10M pixels) the array is spatially
    subsampled to ~1M pixels so that the phase lookup dict used by M1
    closure computation does not exhaust system RAM.
    """
    import rasterio
    fname = "ifg_goldstein_complex_real_imag.tif" if method == "goldstein" else "ifg_film_unet.tif"
    path = pair_dir / fname
    if not path.exists():
        return None
    try:
        with rasterio.open(path) as src:
            re = src.read(1).astype(np.float32)
            im = src.read(2).astype(np.float32)
        phi = np.arctan2(im, re)
        # Subsample large images to keep the phase-lookup dict in RAM
        if phi.size > 10_000_000:
            step = max(1, int(np.sqrt(phi.size / 1_000_000)))
            phi = np.ascontiguousarray(phi[::step, ::step])
        return phi
    except Exception as e:
        log.warning("Could not load %s: %s", path, e)
        return None


def _load_complex_mean(pair_dir: Path, method: str) -> Optional[tuple]:
    """Return (mean_Re, mean_Im) for the interferogram — used for vector-mean phase."""
    import rasterio
    fname = "ifg_goldstein_complex_real_imag.tif" if method == "goldstein" else "ifg_film_unet.tif"
    path = pair_dir / fname
    if not path.exists():
        return None
    try:
        with rasterio.open(path) as src:
            re = src.read(1).astype(np.float32)
            im = src.read(2).astype(np.float32)
        return float(np.nanmean(re)), float(np.nanmean(im))
    except Exception as e:
        log.warning("Could not load %s: %s", path, e)
        return None


def _load_coherence(pair_dir: Path) -> Optional[np.ndarray]:
    import rasterio
    try:
        with rasterio.open(pair_dir / "coherence.tif") as src:
            return src.read(1).astype(np.float32)
    except Exception:
        return None


def _load_unw(pair_dir: Path, filename: str = "unw_phase.tif") -> Optional[np.ndarray]:
    import rasterio
    path = pair_dir / filename
    if not path.exists():
        return None  # avoid GDAL file-not-found noise
    try:
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Triplet closure — over test pairs
# ---------------------------------------------------------------------------

def _build_phase_lookup(
    pair_dirs: list[Path],
    method: str,
    fallback_method: Optional[str] = None,
) -> dict[tuple[str, str], np.ndarray]:
    """
    Build {(id_ref, id_sec): phase_array} from coreg_meta.json.

    Triplet manifest uses individual acquisition IDs (id_a, id_b, id_c),
    not pair directory names, so we need this metadata-driven lookup.

    Parameters
    ----------
    fallback_method : str or None
        If the primary method's file is absent, fall back to this method
        (e.g., use 'goldstein' for pairs where 'film_unet' hasn't been run).
    """
    lookup: dict[tuple[str, str], np.ndarray] = {}
    for pd_dir in pair_dirs:
        phi = _load_phase(pd_dir, method)
        if phi is None and fallback_method is not None:
            phi = _load_phase(pd_dir, fallback_method)
        if phi is None:
            continue
        try:
            with open(pd_dir / "coreg_meta.json") as f:
                m = json.load(f)
            id_ref = m.get("id_ref", "")
            id_sec = m.get("id_sec", "")
            if id_ref and id_sec:
                lookup[(id_ref, id_sec)] = phi
        except Exception:
            pass
    return lookup


def _find_pair_phase(
    lookup: dict[tuple[str, str], np.ndarray],
    id_a: str,
    id_b: str,
) -> Optional[np.ndarray]:
    """Look up phase for pair (a→b); negate if only (b→a) exists."""
    if (id_a, id_b) in lookup:
        return lookup[(id_a, id_b)]
    if (id_b, id_a) in lookup:
        return -lookup[(id_b, id_a)]
    return None


def _iter_triplet_errors(
    lookup: dict[tuple[str, str], np.ndarray],
    triplets_df: pd.DataFrame,
) -> list[float]:
    """
    Iterate over triplets (id_a, id_b, id_c) and return per-triplet median
    closure errors in radians.

    Closure = wrap(phi_ab + phi_bc − phi_ac)
    """
    if triplets_df.empty:
        return []
    errors: list[float] = []
    for _, row in triplets_df.iterrows():
        id_a = str(row.get("id_a", ""))
        id_b = str(row.get("id_b", ""))
        id_c = str(row.get("id_c", ""))
        phi_ab = _find_pair_phase(lookup, id_a, id_b)
        phi_bc = _find_pair_phase(lookup, id_b, id_c)
        phi_ac = _find_pair_phase(lookup, id_a, id_c)
        if phi_ab is None or phi_bc is None or phi_ac is None:
            continue
        if phi_ab.shape != phi_bc.shape or phi_ab.shape != phi_ac.shape:
            continue
        err = triplet_closure_error(phi_ab, phi_bc, phi_ac)
        errors.append(err["median_rad"])
    return errors


def compute_closure_metrics(
    pair_dirs: list[Path],
    triplets_df: pd.DataFrame,
    method: str,
    fallback_method: Optional[str] = None,
) -> dict:
    """
    Compute Metric 1 (triplet closure) over available triplets.

    Uses `all_pairs` for the phase lookup so that triplets can be matched
    even when only a subset was processed by the model.  For pairs where
    the model output is absent, `fallback_method` (typically 'goldstein')
    is used instead — this means the FiLMUNet metric slightly underestimates
    the true improvement, but avoids zero-triplet results.

    Returns dict with keys: n_triplets, median_rad, mean_rad
    """
    lookup = _build_phase_lookup(pair_dirs, method, fallback_method=fallback_method)
    errors = _iter_triplet_errors(lookup, triplets_df)

    if not errors:
        return {"n_triplets": 0, "median_rad": float("nan"), "mean_rad": float("nan")}
    return {
        "n_triplets": len(errors),
        "median_rad": float(np.median(errors)),
        "mean_rad":   float(np.mean(errors)),
    }


# ---------------------------------------------------------------------------
# Per-pair stats collection
# ---------------------------------------------------------------------------

def collect_pair_stats(
    pair_dirs: list[Path],
    method: str,
    skip_snaphu: bool,
    unw_filename: str = "unw_phase.tif",
) -> list[dict]:
    """Collect per-pair metrics needed for metrics 2 and 3."""
    results = []
    for pd_dir in pair_dirs:
        coh = _load_coherence(pd_dir)
        if coh is None:
            continue
        mean_coh = float(np.nanmean(coh))
        r: dict = {"pair_dir": pd_dir.name, "mean_coherence": mean_coh}

        # Closure will be filled in by compute_closure_metrics
        # Use NaN as placeholder — filled in aggregate step
        r["median_closure_rad"] = float("nan")

        # Metric 2: unwrap success rate
        if not skip_snaphu:
            unw = _load_unw(pd_dir, filename=unw_filename)
            if unw is not None:
                r["unwrap_success_rate"] = unwrap_success_rate(unw, coh)

        results.append(r)
    return results


# ---------------------------------------------------------------------------
# SBAS design matrix and Metric 5
# ---------------------------------------------------------------------------

def _extract_collect_date(id_str: str) -> str:
    """Return YYYYMMDDHHMMSS from a collect ID string."""
    for part in id_str.split("_"):
        if len(part) == 14 and part.isdigit():
            return part
    return id_str


def build_sbas_design_matrix(
    pair_dirs: list[Path],
) -> tuple[np.ndarray, list[str]]:
    """
    Build the SBAS design matrix A (P, T) from the loaded pair directories.

    Convention: phi_ij = phi_j - phi_i → A[p, ref_idx] = -1, A[p, sec_idx] = +1
    where ref_idx is the earlier (reference) acquisition and sec_idx is the later.

    Returns
    -------
    A      : (P, T) float32 design matrix
    epochs : list of T epoch labels (YYYYMMDDHHMMSS strings, sorted)
    """
    pairs_meta: list[dict] = []
    for pd_dir in pair_dirs:
        meta_path = pd_dir / "coreg_meta.json"
        try:
            with open(meta_path) as f:
                m = json.load(f)
            id_ref = m.get("id_ref", "")
            id_sec = m.get("id_sec", "")
            pairs_meta.append({
                "dir": pd_dir,
                "t_ref": _extract_collect_date(id_ref),
                "t_sec": _extract_collect_date(id_sec),
            })
        except Exception:
            continue

    # Unique sorted epochs
    all_epochs = sorted(set(
        t for pm in pairs_meta for t in (pm["t_ref"], pm["t_sec"])
    ))
    epoch_idx = {t: i for i, t in enumerate(all_epochs)}
    T = len(all_epochs)
    P = len(pairs_meta)

    A = np.zeros((P, T), dtype=np.float32)
    for i, pm in enumerate(pairs_meta):
        A[i, epoch_idx[pm["t_ref"]]] = -1.0
        A[i, epoch_idx[pm["t_sec"]]] = +1.0

    return A, all_epochs


def compute_temporal_residual(
    pair_dirs: list[Path],
    method: str,
    fallback_method: Optional[str] = "goldstein",
) -> float:
    """
    Compute Metric 5 (temporal consistency residual).

    Uses per-pair mean phase as the SBAS observation vector.
    Weights = 1 / (1 + median_log_var) for film_unet, uniform for goldstein.
    For pairs without model output, falls back to `fallback_method`.

    Returns NaN if the SBAS system is underdetermined (more epochs than pairs),
    since lstsq gives a trivially zero residual in that case.
    """
    obs_list: list[float] = []
    valid_dirs: list[Path] = []
    weights_list: list[float] = []

    for pd_dir in pair_dirs:
        cplx = _load_complex_mean(pd_dir, method)
        if cplx is None and fallback_method is not None:
            cplx = _load_complex_mean(pd_dir, fallback_method)
        if cplx is None:
            continue
        mean_re, mean_im = cplx
        obs_list.append(float(np.arctan2(mean_im, mean_re)))  # vector mean phase
        valid_dirs.append(pd_dir)

        if method == "film_unet":
            lv_path = pd_dir / "log_var.tif"
            if lv_path.exists():  # check first to avoid GDAL noise
                try:
                    import rasterio
                    with rasterio.open(lv_path) as src:
                        lv = src.read(1).astype(np.float32)
                    med_var = float(np.exp(np.nanmedian(lv)))
                    weights_list.append(1.0 / max(med_var, 1e-4))
                except Exception:
                    weights_list.append(1.0)
            else:
                weights_list.append(1.0)
        else:
            weights_list.append(1.0)

    P = len(valid_dirs)
    if P < 3:
        return float("nan")

    A, epochs = build_sbas_design_matrix(valid_dirs)
    T = len(epochs)

    if P <= T:
        log.warning(
            "SBAS system is underdetermined (P=%d pairs, T=%d epochs): "
            "temporal residual is meaningless — need more pairs than epochs.",
            P, T,
        )
        return float("nan")

    phi_stack = np.array(obs_list, dtype=np.float32).reshape(-1, 1)
    weights   = np.array(weights_list, dtype=np.float32)
    weights   = weights / weights.mean()   # prevent scale inflation

    try:
        return temporal_consistency_residual(phi_stack, A, weights)
    except Exception as e:
        log.warning("Temporal residual computation failed: %s", e)
        return float("nan")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save_closure_histogram(
    closure_errors_gold: list[float],
    closure_errors_model: list[float],
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping closure histogram.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, np.pi, 60)
    ax.hist(closure_errors_gold,  bins=bins, alpha=0.6, label="Goldstein", color="steelblue")
    ax.hist(closure_errors_model, bins=bins, alpha=0.6, label="FiLMUNet",  color="coral")
    ax.set_xlabel("Triplet closure error (rad)")
    ax.set_ylabel("Count")
    ax.set_title("Triplet Phase Closure Error Distribution")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out_path)


def _collect_triplet_errors_list(
    pair_dirs: list[Path],
    triplets_df: pd.DataFrame,
    method: str,
    fallback_method: Optional[str] = None,
) -> list[float]:
    """Return per-triplet median closure error as a flat list (reuses lookup helpers)."""
    lookup = _build_phase_lookup(pair_dirs, method, fallback_method=fallback_method)
    return _iter_triplet_errors(lookup, triplets_df)


def _save_phase_comparison(pair_dirs: list[Path], out_path: Path) -> None:
    """Side-by-side phase images for up to 3 pairs: raw / Goldstein / FiLMUNet."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import rasterio
    except ImportError:
        log.warning("matplotlib/rasterio not available — skipping phase comparison.")
        return

    # Pick pairs that have both goldstein and film_unet
    selected = []
    for pd_dir in pair_dirs:
        if (pd_dir / "ifg_film_unet.tif").exists():
            selected.append(pd_dir)
        if len(selected) == 3:
            break

    if not selected:
        log.warning("No film_unet outputs found — skipping phase comparison figure.")
        return

    n = len(selected)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    titles = ["Raw phase", "Goldstein", "FiLMUNet"]
    fnames = ["ifg_raw_complex_real_imag.tif", "ifg_goldstein_complex_real_imag.tif", "ifg_film_unet.tif"]

    for row_i, pd_dir in enumerate(selected):
        for col_j, (fname, title) in enumerate(zip(fnames, titles)):
            ax = axes[row_i, col_j]
            fpath = pd_dir / fname
            if fpath.exists():
                try:
                    with rasterio.open(fpath) as src:
                        re = src.read(1).astype(np.float32)
                        im = src.read(2).astype(np.float32)
                    phi = np.arctan2(im, re)
                    ax.imshow(phi, cmap="hsv", vmin=-np.pi, vmax=np.pi, aspect="auto")
                except Exception:
                    ax.text(0.5, 0.5, "load error", ha="center", va="center",
                            transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes)
            ax.set_title(f"{title}\n{pd_dir.name[:30]}…" if len(pd_dir.name) > 30
                         else f"{title}\n{pd_dir.name}", fontsize=7)
            ax.axis("off")

    fig.suptitle("Phase Comparison (hue = phase angle)", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out_path)


def _save_temporal_residual_bar(
    gold_residual: float,
    model_residual: float,
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping temporal residual bar.")
        return

    labels = ["Goldstein", "FiLMUNet"]
    values = [gold_residual, model_residual]
    colors = ["steelblue", "coral"]
    valid = [(l, v, c) for l, v, c in zip(labels, values, colors)
             if not np.isnan(v)]
    if not valid:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    labs, vals, cols = zip(*valid)
    ax.bar(labs, vals, color=cols, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Temporal consistency residual (rad)")
    ax.set_title("Metric 5: Temporal Consistency Residual")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Metric 4 helpers — DEM NMAD
# ---------------------------------------------------------------------------

_CAPELLA_ALTITUDE_M  = 525_000.0   # approximate orbital altitude
_MANIFEST_CACHE: dict | None = None


def _load_scene_index() -> dict:
    """Load full_index.parquet keyed by scene id (cached)."""
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is not None:
        return _MANIFEST_CACHE
    manifest = ROOT / "data" / "manifests" / "full_index.parquet"
    if not manifest.exists():
        log.warning("full_index.parquet not found — M4 will be NaN.")
        _MANIFEST_CACHE = {}
        return _MANIFEST_CACHE
    df = pd.read_parquet(manifest)
    _MANIFEST_CACHE = {row["id"]: row for _, row in df.iterrows()}
    return _MANIFEST_CACHE


def _height_of_ambiguity(bperp_m: float, incidence_deg: float,
                         center_freq_ghz: float) -> float:
    """Height-of-ambiguity in metres (flat-Earth approximation)."""
    if abs(bperp_m) < 10.0:
        return float("nan")
    wavelength_m = 3e8 / (center_freq_ghz * 1e9)
    theta = np.radians(incidence_deg)
    R     = _CAPELLA_ALTITUDE_M / np.cos(theta)
    return wavelength_m * R * np.sin(theta) / (2.0 * abs(bperp_m))


def _load_copernicus_patch(dem_path: Path, bbox_w: float, bbox_s: float,
                           bbox_e: float, bbox_n: float) -> Optional[np.ndarray]:
    """Return Copernicus DEM heights (float32) cropped to bbox, or None."""
    try:
        import rasterio
        from rasterio.windows import from_bounds
        with rasterio.open(dem_path) as src:
            window = from_bounds(bbox_w, bbox_s, bbox_e, bbox_n, src.transform)
            data   = src.read(1, window=window).astype(np.float32)
            nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = float("nan")
        return data if data.size > 0 else None
    except Exception as e:
        log.debug("Copernicus read error: %s", e)
        return None


def _detrend_plane(arr: np.ndarray) -> np.ndarray:
    """Subtract least-squares planar fit from a 2-D array (removes flat-earth ramp)."""
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


def _compute_m4_for_method(pair_dirs: list[Path], unw_filename: str,
                            scene_index: dict, dem_path: Path) -> float:
    """Compute mean per-pair DEM NMAD for one method."""
    import rasterio
    nmad_list: list[float] = []

    for pd_dir in pair_dirs:
        meta_path = pd_dir / "coreg_meta.json"
        unw_path  = pd_dir / unw_filename
        if not meta_path.exists() or not unw_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        bperp_m       = meta.get("bperp_m", 0.0)
        incidence_deg = meta.get("incidence_angle_deg", 45.0)
        id_ref        = meta.get("id_ref", "")

        scene = scene_index.get(id_ref)
        if scene is None:
            continue

        h_amb = _height_of_ambiguity(bperp_m, incidence_deg,
                                     float(scene["center_freq_ghz"]))
        if not np.isfinite(h_amb):
            continue

        # Load unwrapped phase
        try:
            with rasterio.open(unw_path) as src:
                unw = src.read(1).astype(np.float32)
        except Exception:
            continue

        # Convert phase → height
        h_insar = unw * h_amb / (2.0 * np.pi)
        h_insar[~np.isfinite(h_insar)] = float("nan")

        # Remove flat-earth ramp
        h_insar = _detrend_plane(h_insar)

        # Load reference DEM for scene bbox
        h_ref = _load_copernicus_patch(
            dem_path,
            float(scene["bbox_w"]), float(scene["bbox_s"]),
            float(scene["bbox_e"]), float(scene["bbox_n"]),
        )
        if h_ref is None or not np.any(np.isfinite(h_ref)):
            continue

        # Reference: median terrain height over scene bbox (scalar)
        h_ref_median = float(np.nanmedian(h_ref))

        # NMAD: scatter of (h_insar − h_ref_median) over valid pixels
        valid = np.isfinite(h_insar)
        if valid.sum() < 100:
            continue
        e = h_insar[valid] - h_ref_median
        pair_nmad = float(1.4826 * np.median(np.abs(e - np.median(e))))
        nmad_list.append(pair_nmad)

    if not nmad_list:
        return float("nan")
    return float(np.mean(nmad_list))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _fmt(v: float, unit: str = "") -> str:
    if np.isnan(v):
        return "  N/A  "
    return f"{v:7.3f}{unit}"


def _improvement(gold: float, model: float, higher_is_better: bool = False) -> str:
    if np.isnan(gold) or np.isnan(model):
        return "   N/A  "
    if higher_is_better:
        diff_pp = (model - gold) * 100
        return f"+{diff_pp:.1f} pp" if diff_pp >= 0 else f"{diff_pp:.1f} pp"
    else:
        pct = (model - gold) / (abs(gold) + 1e-12) * 100
        return f"{pct:+.1f}%"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute all 5 contest metrics for Goldstein vs FiLMUNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to FiLMUNet checkpoint (.pt).")
    p.add_argument("--pairs_dir", required=True,
                   help="Root directory of processed pair subdirectories.")
    p.add_argument("--triplets_manifest", required=True,
                   help="Parquet file with triplet definitions.")
    p.add_argument("--out_dir", default="experiments/enhanced/outputs",
                   help="Output directory for CSV and figures.")
    p.add_argument("--tile_size",   type=int, default=256)
    p.add_argument("--stride",      type=int, default=128)
    p.add_argument("--batch_size",  type=int, default=16,
                   help="Tiles per GPU forward pass (higher = more VRAM used, faster).")
    p.add_argument("--test_frac", type=float, default=0.15,
                   help="Fraction of pairs to use as the held-out test set.")
    p.add_argument("--skip_inference", action="store_true",
                   help="Skip model inference (re-use existing ifg_film_unet.tif).")
    p.add_argument("--force_inference", action="store_true",
                   help="Force overwrite existing ifg_film_unet.tif (re-run inference).")
    p.add_argument("--skip_snaphu_metrics", action="store_true",
                   help="Skip metrics 2/3/4 that require unw_phase.tif.")
    p.add_argument("--snaphu_only", action="store_true",
                   help="Compute only M2/M3/M4 (skip M1 triplet closure and M5 temporal residual).")
    p.add_argument("--copernicus_dem_dir", default=None,
                   help="Dir containing the merged Copernicus GLO-30 DEM for M4.")
    p.add_argument("--aoi", default=None,
                   help="AOI code (e.g. AOI000, AOI008, AOI024). Derives DEM filename: "
                        "AOI000→hawaii_dem.tif, others→aoi008_dem.tif / aoi024_dem.tif etc.")
    p.add_argument("--test_only", action="store_true",
                   help="Evaluate on test split only (last --test_frac pairs by date).")
    p.add_argument("--max_pairs", type=int, default=None,
                   help="Limit evaluation to first N pairs (for quick tests).")
    p.add_argument("--device", default=None,
                   help="PyTorch device string (default: cuda if available).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Timestamp-based eval tag ────────────────────────────────────────────
    eval_ts    = datetime.now().strftime('%Y%m%d_%H%M')
    chkpt_stem = Path(args.checkpoint).stem   # e.g. "raw2gold_20260319_2139_final"
    eval_tag   = f"eval_{chkpt_stem}_{eval_ts}"

    log_path = Path("logs") / f"{eval_tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)
    log.info("Eval tag: %s | Log: %s", eval_tag, log_path)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Using device: %s", device)

    # ── Discover pairs ──────────────────────────────────────────────────────
    pairs_dir = Path(args.pairs_dir)
    all_pairs = _discover_pairs(pairs_dir)
    if not all_pairs:
        log.error("No pairs with ifg_goldstein_complex_real_imag.tif found in %s", pairs_dir)
        sys.exit(1)

    eval_pairs = _temporal_split_test(all_pairs, args.test_frac) if args.test_only \
                 else all_pairs
    if args.max_pairs is not None:
        eval_pairs = eval_pairs[:args.max_pairs]
    log.info("Evaluating on %d pairs (%s)", len(eval_pairs),
             "test split" if args.test_only else "all pairs")

    # ── Load triplets manifest ──────────────────────────────────────────────
    triplets_path = Path(args.triplets_manifest)
    if triplets_path.exists():
        triplets_df = pd.read_parquet(triplets_path)
        log.info("Loaded %d triplets from %s", len(triplets_df), triplets_path)
    else:
        log.warning("Triplets manifest not found: %s — metric 1 will be N/A", triplets_path)
        triplets_df = pd.DataFrame()

    # ── FiLMUNet inference ──────────────────────────────────────────────────
    if not args.skip_inference:
        model = _load_model(args.checkpoint, device)
        n_infer = 0
        for i, pd_dir in enumerate(eval_pairs):
            if (pd_dir / "ifg_film_unet.tif").exists() and not args.force_inference:
                log.info("[%d/%d] Skipping (ifg_film_unet.tif exists): %s",
                         i + 1, len(eval_pairs), pd_dir.name)
                continue
            log.info("[%d/%d] Inference: %s", i + 1, len(eval_pairs), pd_dir.name)
            try:
                denoised, log_var = run_inference_on_pair(
                    model, pd_dir, device, args.tile_size, args.stride, args.batch_size
                )
                save_inference_outputs(pd_dir, denoised, log_var)
                n_infer += 1
            except Exception as e:
                log.error("Inference failed for %s: %s", pd_dir.name, e)
        log.info("Inference complete: %d new outputs written.", n_infer)

    # ── Metric 1: Triplet closure ───────────────────────────────────────────
    if not args.snaphu_only:
        # Use ALL processed pairs so triplets can be matched.
        # FiLMUNet uses Goldstein as fallback for pairs without inference output.
        gold_closure  = compute_closure_metrics(
            all_pairs, triplets_df, "goldstein")
        model_closure = compute_closure_metrics(
            all_pairs, triplets_df, "film_unet", fallback_method="goldstein")
        gold_errors_list  = _collect_triplet_errors_list(
            all_pairs, triplets_df, "goldstein")
        model_errors_list = _collect_triplet_errors_list(
            all_pairs, triplets_df, "film_unet", fallback_method="goldstein")
    else:
        _nan_closure = {"median_rad": float("nan"), "mean_rad": float("nan"), "n_triplets": 0}
        gold_closure  = _nan_closure
        model_closure = _nan_closure
        gold_errors_list  = []
        model_errors_list = []

    # ── Metrics 2 & 3 ──────────────────────────────────────────────────────
    # Per-pair coherence stats — separate for each method's unwrapped phase
    pair_stats_gold  = collect_pair_stats(eval_pairs, "goldstein",  args.skip_snaphu_metrics,
                                          unw_filename="unw_phase.tif")
    pair_stats_model = collect_pair_stats(eval_pairs, "film_unet",  args.skip_snaphu_metrics,
                                          unw_filename="unw_phase_film_unet.tif")
    # Alias for downstream code that still references pair_stats
    pair_stats = pair_stats_gold

    if not args.skip_snaphu_metrics:
        # Metric 2: mean unwrap success rate across pairs
        uwr_list_gold  = [r["unwrap_success_rate"] for r in pair_stats_gold
                          if "unwrap_success_rate" in r]
        uwr_list_model = [r["unwrap_success_rate"] for r in pair_stats_model
                          if "unwrap_success_rate" in r]
        gold_uwr  = float(np.mean(uwr_list_gold))  if uwr_list_gold  else float("nan")
        model_uwr = float(np.mean(uwr_list_model)) if uwr_list_model else float("nan")
    else:
        gold_uwr  = float("nan")
        model_uwr = float("nan")

    # Metric 3: usable pairs fraction (back-fill closure from aggregate)
    if gold_errors_list:
        med_gold_closure  = float(np.median(gold_errors_list))
        med_model_closure = float(np.median(model_errors_list)) if model_errors_list \
                            else float("nan")
        for r in pair_stats:
            r.setdefault("median_closure_rad", med_gold_closure)
    else:
        med_gold_closure  = float("nan")
        med_model_closure = float("nan")

    # Use coherence-only gate when no closure data is available
    closure_thresh = 0.5 if not np.isnan(med_gold_closure) else float("inf")
    gold_usable = usable_pairs_fraction(pair_stats, closure_threshold_rad=closure_thresh)

    model_pair_stats = [dict(r) for r in pair_stats]
    if not np.isnan(med_model_closure):
        for r in model_pair_stats:
            r["median_closure_rad"] = med_model_closure
    m_closure_thresh = 0.5 if not np.isnan(med_model_closure) else float("inf")
    model_usable = usable_pairs_fraction(model_pair_stats, closure_threshold_rad=m_closure_thresh)

    # ── Metric 4: DEM NMAD ─────────────────────────────────────────────────
    # Derive DEM filename from --aoi: AOI000→hawaii_dem.tif, else aoi024_dem.tif etc.
    if args.aoi and args.aoi.upper() != "AOI000":
        dem_filename = f"{args.aoi.lower()}_dem.tif"
    else:
        dem_filename = "hawaii_dem.tif"

    if args.copernicus_dem_dir and not args.skip_snaphu_metrics:
        dem_path = Path(args.copernicus_dem_dir) / dem_filename
        if dem_path.exists():
            scene_idx  = _load_scene_index()
            gold_nmad  = _compute_m4_for_method(eval_pairs, "unw_phase.tif",
                                                 scene_idx, dem_path)
            model_nmad = _compute_m4_for_method(eval_pairs, "unw_phase_film_unet.tif",
                                                 scene_idx, dem_path)
            log.info("M4 Goldstein NMAD=%.3f m  FiLMUNet NMAD=%.3f m",
                     gold_nmad, model_nmad)
        else:
            log.warning("%s not found in %s — run download_copernicus_dem.py --merged_name %s first.",
                        dem_filename, args.copernicus_dem_dir, dem_filename)
            gold_nmad  = float("nan")
            model_nmad = float("nan")
    else:
        gold_nmad  = float("nan")
        model_nmad = float("nan")

    # ── Metric 5: Temporal consistency ─────────────────────────────────────
    if not args.snaphu_only:
        # Use all pairs for an overconstrained SBAS system; FiLMUNet falls back
        # to Goldstein for pairs without model outputs.
        gold_temporal  = compute_temporal_residual(all_pairs, "goldstein")
        model_temporal = compute_temporal_residual(all_pairs, "film_unet")
    else:
        gold_temporal  = float("nan")
        model_temporal = float("nan")

    # ── Assemble results ────────────────────────────────────────────────────
    metrics = {
        "triplet_closure_rad":    (gold_closure["median_rad"],  model_closure["median_rad"]),
        "unwrap_success_rate":    (gold_uwr,                    model_uwr),
        "usable_pairs_fraction":  (gold_usable,                 model_usable),
        "dem_nmad_m":             (gold_nmad,                   model_nmad),
        "temporal_residual_rad":  (gold_temporal,               model_temporal),
    }
    higher_is_better = {
        "triplet_closure_rad":    False,
        "unwrap_success_rate":    True,
        "usable_pairs_fraction":  True,
        "dem_nmad_m":             False,
        "temporal_residual_rad":  False,
    }
    labels = {
        "triplet_closure_rad":    "Triplet Closure Error (rad)",
        "unwrap_success_rate":    "Unwrap Success Rate",
        "usable_pairs_fraction":  "Usable Pairs Fraction",
        "dem_nmad_m":             "DEM NMAD (m)",
        "temporal_residual_rad":  "Temporal Residual (rad)",
    }

    # ── Print summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("=== Contest Metric Comparison ===")
    print("=" * 62)
    print(f"{'Metric':<34} {'Goldstein':>10} {'FiLMUNet':>10} {'Improvement':>12}")
    print("-" * 62)
    for key, (g, m) in metrics.items():
        hib = higher_is_better[key]
        note = ""
        if np.isnan(g) and np.isnan(m):
            note = "  ← N/A (unw_phase.tif not found)" if key in (
                "unwrap_success_rate", "dem_nmad_m") else ""
        print(f"{labels[key]:<34} {_fmt(g):>10} {_fmt(m):>10} {_improvement(g, m, hib):>12}{note}")
    print("=" * 62)

    if args.skip_snaphu_metrics:
        print("Note: Metrics 2/3/4 require unw_phase.tif — run scripts/unwrap_snaphu.py first.")
    print(f"Triplets matched: {gold_closure['n_triplets']} (goldstein), "
          f"{model_closure['n_triplets']} (film_unet)")
    if gold_closure["n_triplets"] == 0:
        print("Warning: 0 complete triplets found in processed pairs.")
        print("  The 100 processed pairs do not form any complete triplet (a→b, b→c, a→c).")
        print("  To compute Metric 1, run preprocess_pairs.py on additional pairs that")
        print("  complete triplets — or run eval without --test_only to use all pairs.")
    print()

    # ── Save CSV ────────────────────────────────────────────────────────────
    rows = []
    for key, (g, m) in metrics.items():
        hib = higher_is_better[key]
        rows.append({"metric": labels[key], "method": "goldstein", "value": g})
        rows.append({"metric": labels[key], "method": "film_unet",  "value": m})
        impv = float("nan")
        if not (np.isnan(g) or np.isnan(m)):
            impv = (m - g) / (abs(g) + 1e-12) * 100 if not hib \
                   else (m - g) * 100
        rows.append({"metric": labels[key], "method": "improvement_pct", "value": impv})

    csv_path = out_dir / f"metrics_{eval_tag}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log.info("Saved %s", csv_path)

    # ── Figures ─────────────────────────────────────────────────────────────
    _save_closure_histogram(
        gold_errors_list,
        model_errors_list,
        fig_dir / f"{eval_tag}_closure_histogram.png",
    )
    _save_phase_comparison(eval_pairs, fig_dir / f"{eval_tag}_phase_comparison.png")
    _save_temporal_residual_bar(
        gold_temporal,
        model_temporal,
        fig_dir / f"{eval_tag}_temporal_residual_bar.png",
    )

    print(f"Outputs written to: {out_dir}/")
    print(f"  metrics_{eval_tag}.csv")
    print(f"  figures/{eval_tag}_closure_histogram.png")
    print(f"  figures/{eval_tag}_phase_comparison.png")
    print(f"  figures/{eval_tag}_temporal_residual_bar.png")
    print(f"  logs/{eval_tag}.log")


if __name__ == "__main__":
    main()
