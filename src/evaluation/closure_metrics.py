"""
Contest evaluation metrics for InSAR phase quality assessment.

Implements all 5 official IEEE GRSS 2026 Data Fusion Contest metrics:

1. triplet_closure_error     — median/mean/std/rmse of wrap(φ_ij + φ_jk − φ_ik)
2. unwrap_success_rate       — fraction of coherent pixels with valid unwrapped phase
3. usable_pairs_fraction     — fraction of pairs passing coherence + closure gates
4. dem_nmad                  — Normalized Median Absolute Deviation (1.4826×MAD)
5. temporal_consistency_residual — SBAS ‖W(Ax* − φ̂)‖₂

Also provides compute_baseline_metrics() to aggregate over a processed-pairs directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric 1: Triplet closure error
# ---------------------------------------------------------------------------

def triplet_closure_error(
    phi_ij: np.ndarray,
    phi_jk: np.ndarray,
    phi_ik: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Contest metric 1: Triplet phase closure error.

    closure = wrap(phi_ij + phi_jk - phi_ik)

    Parameters
    ----------
    phi_ij, phi_jk, phi_ik : np.ndarray
        (H, W) wrapped phase arrays for pairs ij, jk, ik.
    mask : np.ndarray, optional
        Boolean (H, W) mask; True = valid pixel.

    Returns
    -------
    dict with keys: median_rad, mean_rad, std_rad, rmse_rad
    """
    closure = np.angle(np.exp(1j * (phi_ij + phi_jk - phi_ik)))  # wrap to [-π, π]
    if mask is not None:
        pixels = closure[mask]
    else:
        pixels = closure.ravel()
    abs_pixels = np.abs(pixels)
    return {
        "median_rad": float(np.median(abs_pixels)),
        "mean_rad":   float(np.mean(abs_pixels)),
        "std_rad":    float(np.std(pixels)),
        "rmse_rad":   float(np.sqrt(np.mean(pixels ** 2))),
    }


# ---------------------------------------------------------------------------
# Metric 2: Unwrap success rate
# ---------------------------------------------------------------------------

def unwrap_success_rate(
    unw_phase: np.ndarray,
    coherence: np.ndarray,
    coh_threshold: float = 0.35,
) -> float:
    """
    Contest metric 2: Fraction of coherent pixels with successful unwrapping.

    Parameters
    ----------
    unw_phase : np.ndarray
        (H, W) unwrapped phase; NaN indicates failed/masked pixels.
    coherence : np.ndarray
        (H, W) coherence in [0, 1].
    coh_threshold : float
        Minimum coherence to count as 'coherent'.

    Returns
    -------
    float in [0, 1]
    """
    coherent = coherence >= coh_threshold
    success  = coherent & ~np.isnan(unw_phase)
    if not coherent.any():
        return 0.0
    return float(success.sum() / coherent.sum())


# ---------------------------------------------------------------------------
# Metric 3: Usable pairs fraction
# ---------------------------------------------------------------------------

def usable_pairs_fraction(
    pair_results: list[dict],
    coh_threshold: float = 0.35,
    closure_threshold_rad: float = 0.5,
) -> float:
    """
    Contest metric 3: Fraction of pairs classified as 'usable'.

    A pair is usable if:
      - mean_coherence >= coh_threshold
      - median_closure_rad < closure_threshold_rad  (≈28°)

    Parameters
    ----------
    pair_results : list[dict]
        Each dict must have 'mean_coherence'. Optionally 'median_closure_rad'.
    coh_threshold : float
        Coherence gate.
    closure_threshold_rad : float
        Closure-error gate (radians).

    Returns
    -------
    float in [0, 1]
    """
    if not pair_results:
        return 0.0
    usable = [
        p for p in pair_results
        if p["mean_coherence"] >= coh_threshold
        and p.get("median_closure_rad", float("inf")) < closure_threshold_rad
    ]
    return len(usable) / len(pair_results)


# ---------------------------------------------------------------------------
# Metric 4: DEM NMAD
# ---------------------------------------------------------------------------

def dem_nmad(
    pred_dem: np.ndarray,
    ref_dem: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Contest metric 4: Normalized Median Absolute Deviation.

    NMAD = 1.4826 × median(|err − median(err)|)

    Parameters
    ----------
    pred_dem, ref_dem : np.ndarray
        (H, W) elevation arrays (same units).
    mask : np.ndarray, optional
        Boolean (H, W); True = valid pixel.

    Returns
    -------
    float (same units as input DEMs)
    """
    err = (pred_dem - ref_dem).ravel()
    if mask is not None:
        err = err[mask.ravel()]
    if err.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(err - np.median(err))))


# ---------------------------------------------------------------------------
# Metric 5: Temporal consistency residual
# ---------------------------------------------------------------------------

def temporal_consistency_residual(
    phi_stack: np.ndarray,
    A: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Contest metric 5: SBAS temporal consistency ‖W(Ax* − φ̂)‖₂.

    Solves the weighted least-squares system  min_x ‖W(Ax − φ̂)‖₂
    and returns the normalised RMS residual.

    Parameters
    ----------
    phi_stack : np.ndarray
        (P, N) array of P pair observations over N pixels (or mean values).
    A : np.ndarray
        (P, T) SBAS design matrix (integers: +1 reference, -1 secondary).
    weights : np.ndarray, optional
        (P,) per-pair weights. Defaults to uniform weights.

    Returns
    -------
    float — RMS residual ‖W(Ax* − φ̂)‖₂ / √(P·N)
    """
    if weights is None:
        weights = np.ones(phi_stack.shape[0])
    W = np.diag(weights)
    Aw  = W @ A
    phw = W @ phi_stack
    x_star, *_ = np.linalg.lstsq(Aw, phw, rcond=None)
    residual = phw - Aw @ x_star
    return float(np.sqrt(np.mean(residual ** 2)))


# ---------------------------------------------------------------------------
# Aggregate: compute all available metrics over a processed-pairs directory
# ---------------------------------------------------------------------------

def compute_baseline_metrics(
    pairs_dir: str,
    triplets_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
) -> dict:
    """
    Compute all available metrics over a processed-pairs directory.

    Scans `pairs_dir` for subdirectories containing:
      - coherence.tif     (single-band float32)
      - ifg_goldstein.tif (2-band Re/Im float32 = wrapped phase after filtering)
      - coreg_meta.json   (metadata including mean_coherence if available)

    If `triplets_df` is provided (DataFrame with columns id_ref_ij, id_sec_ij,
    id_ref_jk, id_sec_jk, id_ref_ik, id_sec_ik), closure errors are computed
    for matched triplets.

    Parameters
    ----------
    pairs_dir : str
        Path to directory of processed pair subdirectories.
    triplets_df : pd.DataFrame, optional
        Triplet definitions for closure computation.
    output_path : str, optional
        If provided, results are saved as JSON.

    Returns
    -------
    dict with metric values (NaN for unavailable metrics).
    """
    try:
        import rasterio
    except ImportError:
        log.error("rasterio not available — cannot load raster data.")
        return {}

    pairs_dir = Path(pairs_dir)
    pair_dirs = sorted(p for p in pairs_dir.iterdir() if p.is_dir())
    log.info("Found %d pair directories in %s", len(pair_dirs), pairs_dir)

    # Collect per-pair stats
    pair_results = []
    loaded_phases: dict[str, np.ndarray] = {}  # pair_dir_name → wrapped phase array

    for pd_dir in pair_dirs:
        meta_path = pd_dir / "coreg_meta.json"
        coh_path  = pd_dir / "coherence.tif"
        ifg_path  = pd_dir / "ifg_goldstein.tif"

        if not coh_path.exists() or not ifg_path.exists():
            log.debug("Skipping incomplete pair dir: %s", pd_dir.name)
            continue

        try:
            with rasterio.open(coh_path) as src:
                coh = src.read(1).astype(np.float32)
            with rasterio.open(ifg_path) as src:
                re = src.read(1).astype(np.float32)
                im = src.read(2).astype(np.float32)
            phase = np.arctan2(im, re)
        except Exception as e:
            log.warning("Failed to read %s: %s", pd_dir.name, e)
            continue

        mean_coh = float(np.nanmean(coh))
        result = {"pair_dir": pd_dir.name, "mean_coherence": mean_coh}

        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                result.update({k: v for k, v in meta.items()
                                if k in ("dt_days", "dinc_deg", "q_score", "bperp_m")})
            except Exception:
                pass

        pair_results.append(result)
        loaded_phases[pd_dir.name] = phase

    log.info("Loaded %d valid pairs", len(pair_results))

    # Metric 2 requires unw_phase.tif (produced by SNAPHU) — skip here.
    # It is computed per-pair in the eval/compute_metrics.py script.

    # --- Metric 3: usable pairs (closure not yet computed — use coherence gate only) ---
    # Closure gate will be applied if triplets_df is provided
    metric3_coh_only = usable_pairs_fraction(
        pair_results,
        closure_threshold_rad=float("inf"),  # no closure gate yet
    )

    # --- Metric 1: closure errors (requires triplets) ---
    closure_errors: list[dict] = []
    if triplets_df is not None and len(triplets_df) > 0:
        # Build name→phase lookup
        dir_names = {r["pair_dir"] for r in pair_results}

        for _, row in triplets_df.iterrows():
            # Try to match triplet to loaded pairs by pair directory name
            # Convention: pair dir names are "{id_ref}__{id_sec}"[:80]
            def _find_pair(id_r: str, id_s: str) -> Optional[np.ndarray]:
                key = f"{id_r}__{id_s}"[:80]
                if key in loaded_phases:
                    return loaded_phases[key]
                # Try reversed
                key2 = f"{id_s}__{id_r}"[:80]
                if key2 in loaded_phases:
                    return -loaded_phases[key2]  # conjugate = negate phase
                return None

            phi_ij = _find_pair(row.get("id_ref_ij", ""), row.get("id_sec_ij", ""))
            phi_jk = _find_pair(row.get("id_ref_jk", ""), row.get("id_sec_jk", ""))
            phi_ik = _find_pair(row.get("id_ref_ik", ""), row.get("id_sec_ik", ""))

            if phi_ij is None or phi_jk is None or phi_ik is None:
                continue
            if phi_ij.shape != phi_jk.shape or phi_ij.shape != phi_ik.shape:
                continue

            err = triplet_closure_error(phi_ij, phi_jk, phi_ik)
            closure_errors.append(err)

        log.info("Computed closure errors for %d triplets", len(closure_errors))

        # Back-fill closure into pair_results for metric 3
        # (approximate: assign median closure from all matched triplets)
        if closure_errors:
            med_closure = float(np.median([e["median_rad"] for e in closure_errors]))
            for r in pair_results:
                r.setdefault("median_closure_rad", med_closure)

    # --- Metric 3 with closure gate ---
    metric3 = usable_pairs_fraction(pair_results)

    # --- Assemble results ---
    results: dict = {
        "n_pairs": len(pair_results),
        "mean_coherence_across_pairs": float(np.mean([r["mean_coherence"] for r in pair_results]))
        if pair_results else float("nan"),
        "metric1_triplet_closure": {
            "n_triplets": len(closure_errors),
            "median_rad": float(np.median([e["median_rad"] for e in closure_errors]))
            if closure_errors else float("nan"),
            "mean_rad": float(np.mean([e["mean_rad"] for e in closure_errors]))
            if closure_errors else float("nan"),
        },
        "metric2_unwrap_success_rate": float("nan"),   # requires unw_phase.tif
        "metric3_usable_pairs_fraction": metric3,
        "metric3_coh_gate_only": metric3_coh_only,
        "metric4_dem_nmad": float("nan"),              # requires reference DEM
        "metric5_temporal_residual": float("nan"),     # requires SBAS stack
        "per_pair": pair_results,
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # per_pair may contain numpy types — convert
        safe = json.loads(json.dumps(results, default=lambda x: float(x) if hasattr(x, '__float__') else str(x)))
        with open(out, "w") as f:
            json.dump(safe, f, indent=2)
        log.info("Saved metrics to %s", output_path)

    return results
