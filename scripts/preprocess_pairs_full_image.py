#!/usr/bin/env python3
"""
Capella Spotlight PFA SLC co-registration pipeline for InSAR.

V5.8 notes
----------
- Exact-output-preserving speedups for remap, coherence denominator reuse, and grid caching.
- Optional aggressive flags for skipping pass-2 and using OpenCV coherence.


V5 changes relative to V4
-------------------------
- Composed single-pass warp: if pass-2 wins, combine pass-1 and pass-2 offset
  fields and resample the original slave once.
- Better interpolation option: cubic spline or Lanczos4 via OpenCV remap.
- Practical common-band filtering using a shared 2D FFT taper.
- Candidate testing:
    pass1
    pass2 sequential
    pass2 composed single-pass
    pass2 composed + common-band
  and automatic selection by coherence.
- Clearer residual vector plot.
- Saves TIFF outputs for coherence and interferogram products for QGIS.

Notes
-----
- Common-band filtering here is a practical image-domain approximation applied
  identically to master and candidate slave images. It is not equivalent to a
  full mission-grade raw-data common-band processor.
"""

from __future__ import annotations
import argparse, csv, json, math, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, map_coordinates, uniform_filter
from scipy.signal import fftconvolve
from skimage.registration import phase_cross_correlation

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cupy_ndimage
    _HAS_CUPY = True
except Exception:
    cp = None
    cupy_ndimage = None
    _HAS_CUPY = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    pd = None
    _HAS_PANDAS = False


def ensure_dir(path: os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def check_gpu_support() -> Dict[str, bool]:
    return {
        "opencv": bool(_HAS_CV2),
        "cupy": bool(_HAS_CUPY),
        "coherence_commonband_gpu_feasible": bool(_HAS_CUPY),
        "remap_gpu_feasible": False,
        "default_gpu_enabled": False,
    }


@lru_cache(maxsize=16)
def _cached_index_grids(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    H, W = shape
    rr, cc = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")
    return rr, cc


@lru_cache(maxsize=32)
def _cached_commonband_mask(shape: Tuple[int, int], frac: float, smooth_frac: float) -> np.ndarray:
    H, W = shape
    fy = np.fft.fftfreq(H)
    fx = np.fft.fftfreq(W)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    ry = np.abs(FY) / (np.max(np.abs(fy)) + 1e-12)
    rx = np.abs(FX) / (np.max(np.abs(fx)) + 1e-12)

    a = frac
    b = min(1.0, frac + smooth_frac)

    def soft(r: np.ndarray) -> np.ndarray:
        out = np.ones_like(r, dtype=np.float32)
        mid = (r > a) & (r < b)
        out[r >= b] = 0.0
        out[mid] = 0.5 * (1 + np.cos(np.pi * (r[mid] - a) / (b - a)))
        return out

    return soft(ry) * soft(rx)


def _valid_tie_point_arrays(tie_points: Sequence["TiePoint"]) -> Dict[str, np.ndarray]:
    idx = np.asarray([i for i, tp in enumerate(tie_points) if tp.valid and np.isfinite(tp.drow) and np.isfinite(tp.dcol)], dtype=np.int64)
    if idx.size == 0:
        empty = np.asarray([], dtype=np.float64)
        return {"idx": idx, "row": empty, "col": empty, "drow": empty, "dcol": empty, "peak": empty, "peak_ratio": empty, "phase_error": empty}
    row = np.asarray([tie_points[i].row for i in idx], dtype=np.float64)
    col = np.asarray([tie_points[i].col for i in idx], dtype=np.float64)
    drow = np.asarray([tie_points[i].drow for i in idx], dtype=np.float64)
    dcol = np.asarray([tie_points[i].dcol for i in idx], dtype=np.float64)
    peak = np.asarray([tie_points[i].peak for i in idx], dtype=np.float64)
    peak_ratio = np.asarray([tie_points[i].peak_ratio for i in idx], dtype=np.float64)
    phase_error = np.asarray([tie_points[i].phase_error for i in idx], dtype=np.float64)
    return {
        "idx": idx,
        "row": row,
        "col": col,
        "drow": drow,
        "dcol": dcol,
        "peak": peak,
        "peak_ratio": peak_ratio,
        "phase_error": phase_error,
    }


def _stats_from_array(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.9)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def make_fringe_cmap() -> LinearSegmentedColormap:
    colors = [
        '#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6',
        '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226', '#2b2d42'
    ]
    return LinearSegmentedColormap.from_list('fringe_contrast', colors, N=512)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class CapellaMeta:
    json_path: str
    collect_id: str
    start_timestamp: str
    stop_timestamp: str
    center_time: str
    platform: str
    mode: str
    product_type: str
    data_type: str
    algorithm: str
    rows: int
    cols: int
    pixel_spacing_row: float
    pixel_spacing_col: float
    row_sample_spacing: float
    col_sample_spacing: float
    scene_ref_row: float
    scene_ref_col: float
    scene_ref_ecef: np.ndarray
    center_target_ecef: np.ndarray
    incidence_angle_deg: float
    look_angle_deg: float
    squint_angle_deg: float
    row_direction: np.ndarray
    col_direction: np.ndarray
    slant_plane_normal: np.ndarray
    center_of_aperture_time: float
    center_of_aperture_pos: np.ndarray
    center_of_aperture_vel: np.ndarray
    center_frequency_hz: float
    sampling_frequency_hz: float
    pointing: str
    tx_pol: str
    rx_pol: str
    orbit_direction: str
    state_vector_times: np.ndarray
    state_vector_positions: np.ndarray
    state_vector_velocities: np.ndarray
    doppler_poly: np.ndarray
    nesz_peak_db: float
    prf_samples: List[Tuple[str, float]]

    @property
    def wavelength_m(self) -> float:
        return 299792458.0 / self.center_frequency_hz


def parse_capella_extended_json(json_path: os.PathLike) -> CapellaMeta:
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    collect = d["collect"]
    img = collect["image"]
    geom = img["image_geometry"]
    center = img["center_pixel"]
    radar = collect["radar"]
    state = collect["state"]
    sv_times, sv_pos, sv_vel = [], [], []
    for sv in state["state_vectors"]:
        sv_times.append(sv["time"])
        sv_pos.append(sv["position"])
        sv_vel.append(sv["velocity"])
    prf_samples = []
    for item in radar.get("prf", []):
        for ts in item["start_timestamps"]:
            prf_samples.append((ts, item["prf"]))
    return CapellaMeta(
        json_path=str(json_path),
        collect_id=collect["collect_id"],
        start_timestamp=collect["start_timestamp"],
        stop_timestamp=collect["stop_timestamp"],
        center_time=center["center_time"],
        platform=collect["platform"],
        mode=collect["mode"],
        product_type=d["product_type"],
        data_type=img["data_type"],
        algorithm=img["algorithm"],
        rows=int(img["rows"]),
        cols=int(img["columns"]),
        pixel_spacing_row=float(img["pixel_spacing_row"]),
        pixel_spacing_col=float(img["pixel_spacing_column"]),
        row_sample_spacing=float(geom["row_sample_spacing"]),
        col_sample_spacing=float(geom["col_sample_spacing"]),
        scene_ref_row=float(geom["scene_reference_point_row_col"][0]),
        scene_ref_col=float(geom["scene_reference_point_row_col"][1]),
        scene_ref_ecef=np.asarray(geom["scene_reference_point_ecef"], dtype=np.float64),
        center_target_ecef=np.asarray(center["target_position"], dtype=np.float64),
        incidence_angle_deg=float(center["incidence_angle"]),
        look_angle_deg=float(center["look_angle"]),
        squint_angle_deg=float(center["squint_angle"]),
        row_direction=np.asarray(geom["row_direction"], dtype=np.float64),
        col_direction=np.asarray(geom["col_direction"], dtype=np.float64),
        slant_plane_normal=np.asarray(geom["slant_plane_normal"], dtype=np.float64),
        center_of_aperture_time=float(geom["center_of_aperture"]["time"]),
        center_of_aperture_pos=np.asarray(geom["center_of_aperture"]["antenna_reference_point"], dtype=np.float64),
        center_of_aperture_vel=np.asarray(geom["center_of_aperture"]["velocity_antenna_reference_point"], dtype=np.float64),
        center_frequency_hz=float(radar["center_frequency"]),
        sampling_frequency_hz=float(radar["sampling_frequency"]),
        pointing=str(radar["pointing"]),
        tx_pol=str(radar["transmit_polarization"]),
        rx_pol=str(radar["receive_polarization"]),
        orbit_direction=str(state["direction"]),
        state_vector_times=np.asarray(sv_times),
        state_vector_positions=np.asarray(sv_pos, dtype=np.float64),
        state_vector_velocities=np.asarray(sv_vel, dtype=np.float64),
        doppler_poly=np.asarray(img["frequency_doppler_centroid_polynomial"]["coefficients"], dtype=np.float64),
        nesz_peak_db=float(img.get("nesz_peak", np.nan)),
        prf_samples=prf_samples,
    )


def pair_compatibility_report(master: CapellaMeta, slave: CapellaMeta) -> Dict[str, object]:
    def _to_utc64(ts: str) -> np.datetime64:
        # Avoid numpy timezone warning; timestamps are both UTC with trailing Z.
        return np.datetime64(ts.replace('Z', ''))
    center_dt_sec = abs(_to_utc64(master.center_time) - _to_utc64(slave.center_time)) / np.timedelta64(1, 's')
    scene_delta_m = float(np.linalg.norm(master.scene_ref_ecef - slave.scene_ref_ecef))
    center_target_delta_m = float(np.linalg.norm(master.center_target_ecef - slave.center_target_ecef))
    row_dir_angle = math.degrees(math.acos(np.clip(np.dot(master.row_direction, slave.row_direction), -1.0, 1.0)))
    col_dir_angle = math.degrees(math.acos(np.clip(np.dot(master.col_direction, slave.col_direction), -1.0, 1.0)))
    baseline_vec = slave.center_of_aperture_pos - master.center_of_aperture_pos
    los = master.center_target_ecef - master.center_of_aperture_pos
    los /= np.linalg.norm(los)
    baseline_parallel = float(np.dot(baseline_vec, los))
    baseline_perp = float(np.linalg.norm(baseline_vec - baseline_parallel * los))
    return {
        "same_platform": master.platform == slave.platform,
        "same_mode": master.mode == slave.mode,
        "same_product_type": master.product_type == slave.product_type,
        "same_algorithm": master.algorithm == slave.algorithm,
        "same_pointing": master.pointing == slave.pointing,
        "same_orbit_direction": master.orbit_direction == slave.orbit_direction,
        "same_polarization": (master.tx_pol == slave.tx_pol) and (master.rx_pol == slave.rx_pol),
        "center_time_separation_sec": float(center_dt_sec),
        "scene_ref_delta_m": scene_delta_m,
        "center_target_delta_m": center_target_delta_m,
        "row_direction_angle_deg": row_dir_angle,
        "col_direction_angle_deg": col_dir_angle,
        "baseline_total_m": float(np.linalg.norm(baseline_vec)),
        "baseline_parallel_m": baseline_parallel,
        "baseline_perpendicular_m": baseline_perp,
        "master_wavelength_m": master.wavelength_m,
        "slave_wavelength_m": slave.wavelength_m,
        "seed_row_shift_px": float(slave.scene_ref_row - master.scene_ref_row),
        "seed_col_shift_px": float(slave.scene_ref_col - master.scene_ref_col),
        "rows_diff": int(slave.rows - master.rows),
        "cols_diff": int(slave.cols - master.cols),
    }


def _to_complex(arr: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(arr):
        return arr.astype(np.complex64, copy=False)
    if arr.ndim >= 3 and arr.shape[-1] == 2:
        return arr[..., 0].astype(np.float32) + 1j * arr[..., 1].astype(np.float32)
    if arr.dtype.fields is not None:
        fields = list(arr.dtype.fields.keys())
        if len(fields) >= 2:
            return arr[fields[0]].astype(np.float32) + 1j * arr[fields[1]].astype(np.float32)
    raise ValueError(f"Unsupported SLC TIFF dtype/shape: {arr.dtype}, {arr.shape}")


def read_complex_slc(tif_path: os.PathLike, expected_rows=None, expected_cols=None) -> np.ndarray:
    arr = tifffile.imread(tif_path)
    z = _to_complex(arr)
    if z.ndim != 2:
        raise ValueError(f"Expected 2D complex SLC, got {z.shape}")
    if expected_rows is not None and expected_cols is not None:
        if z.shape == (expected_rows, expected_cols):
            return z
        if z.shape == (expected_cols, expected_rows):
            return z.T.copy()
    return z


def robust_amplitude(z: np.ndarray, log_scale: bool = True, sigma: float = 0.0) -> np.ndarray:
    a = np.abs(z).astype(np.float32)
    if sigma > 0:
        a = gaussian_filter(a, sigma=sigma)
    if log_scale:
        a = np.log1p(a)
    return a


def multilook_mean(img: np.ndarray, looks_row: int, looks_col: int) -> np.ndarray:
    r = (img.shape[0] // looks_row) * looks_row
    c = (img.shape[1] // looks_col) * looks_col
    x = img[:r, :c]
    return x.reshape(r // looks_row, looks_row, c // looks_col, looks_col).mean(axis=(1, 3))


def estimate_global_shift_thumbnail(master_amp: np.ndarray, slave_amp: np.ndarray, upsample_factor: int = 50):
    shift_rc, error, phasediff = phase_cross_correlation(master_amp, slave_amp, upsample_factor=upsample_factor)
    return float(shift_rc[0]), float(shift_rc[1]), {"error": float(error), "phasediff": float(phasediff)}



def _box_sum_valid(img: np.ndarray, win_h: int, win_w: int) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32, order="C")
    ii = np.pad(x, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0, dtype=np.float64).cumsum(axis=1, dtype=np.float64)
    out = ii[win_h:, win_w:] - ii[:-win_h, win_w:] - ii[win_h:, :-win_w] + ii[:-win_h, :-win_w]
    return out.astype(np.float32, copy=False)


def ncc_search_with_ratio(master_patch: np.ndarray, slave_search: np.ndarray):
    A = master_patch.astype(np.float32, copy=False)
    B = slave_search.astype(np.float32, copy=False)
    A0 = A - A.mean()
    A0 /= (A0.std() + 1e-6)
    kernel = np.flipud(np.fliplr(A0))
    numer = fftconvolve(B, kernel, mode="valid")
    h, w = A0.shape
    sum_B = _box_sum_valid(B, h, w)
    sum_B2 = _box_sum_valid(B * B, h, w)
    n = float(A0.size)
    mean_B = sum_B / n
    var_B = np.maximum(sum_B2 / n - mean_B * mean_B, 1e-8)
    denom = np.sqrt(var_B) * math.sqrt(n)
    ncc = numer / (denom + 1e-6)
    flat = ncc.ravel()
    idx = int(np.argmax(flat))
    peak = float(flat[idx])
    peak_idx = np.unravel_index(idx, ncc.shape)
    if flat.size > 1:
        top2_idx = np.argpartition(flat, -2)[-2:]
        top2_vals = flat[top2_idx]
        if top2_idx[0] == idx:
            second = float(top2_vals[1])
        elif top2_idx[1] == idx:
            second = float(top2_vals[0])
        else:
            second = float(np.max(top2_vals))
    else:
        second = 1e-6
    ratio = peak / max(second, 1e-6)
    return int(peak_idx[0]), int(peak_idx[1]), peak, float(ratio)


def build_design(x, y, order: str = "quadratic"):
    if order == "linear":
        return np.column_stack([np.ones_like(x), x, y, x * y])
    return np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])


def fit_surface(x, y, z, weights=None, order="quadratic"):
    G = build_design(x, y, order=order)
    if weights is not None:
        w = np.sqrt(np.clip(weights, 1e-8, None))[:, None]
        G = G * w
        z = z * w[:, 0]
    coeffs, *_ = np.linalg.lstsq(G, z, rcond=None)
    return coeffs


def eval_surface(coeffs, x, y, order="quadratic"):
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if order == "linear":
        out = coeffs[0] + coeffs[1] * x_arr + coeffs[2] * y_arr + coeffs[3] * x_arr * y_arr
    else:
        out = (
            coeffs[0]
            + coeffs[1] * x_arr
            + coeffs[2] * y_arr
            + coeffs[3] * x_arr * x_arr
            + coeffs[4] * x_arr * y_arr
            + coeffs[5] * y_arr * y_arr
        )
    return out


@dataclass
class TiePoint:
    row: float
    col: float
    drow: float
    dcol: float
    peak: float
    peak_ratio: float
    phase_error: float
    valid: bool
    reject_reason: str = ""
    residual_drow: float = np.nan
    residual_dcol: float = np.nan
    residual_mag: float = np.nan



def _resolve_tp_workers(max_workers: int, n_tasks: int) -> int:
    if n_tasks <= 0:
        return 1
    if max_workers is None or max_workers <= 0:
        cpu = os.cpu_count() or 1
        return max(1, min(n_tasks, min(8, cpu)))
    return max(1, min(int(max_workers), n_tasks))


def _estimate_single_local_offset(master_amp: np.ndarray, slave_amp: np.ndarray,
                                  r0: int, c0: int,
                                  drow0: float, dcol0: float,
                                  win: int, search: int, upsample_factor: int,
                                  peak_threshold: float, peak_ratio_threshold: float,
                                  texture_threshold: float) -> TiePoint:
    H, W = master_amp.shape
    rm0, rm1 = r0 - win // 2, r0 - win // 2 + win
    cm0, cm1 = c0 - win // 2, c0 - win // 2 + win
    mp = master_amp[rm0:rm1, cm0:cm1]
    if mp.shape != (win, win):
        return TiePoint(r0, c0, np.nan, np.nan, 0.0, 0.0, np.nan, False, "shape")

    texture = float(mp.std() / (mp.mean() + 1e-6))
    if texture < texture_threshold:
        return TiePoint(r0, c0, np.nan, np.nan, 0.0, 0.0, np.nan, False, "texture")

    rs_center = int(round(r0 + drow0))
    cs_center = int(round(c0 + dcol0))
    rs0 = rs_center - win // 2 - search
    rs1 = rs_center + win // 2 + search
    cs0 = cs_center - win // 2 - search
    cs1 = cs_center + win // 2 + search
    if rs0 < 0 or cs0 < 0 or rs1 > H or cs1 > W:
        return TiePoint(r0, c0, np.nan, np.nan, 0.0, 0.0, np.nan, False, "border")

    ss = slave_amp[rs0:rs1, cs0:cs1]
    dr_int, dc_int, peak, peak_ratio = ncc_search_with_ratio(mp, ss)
    if peak < peak_threshold:
        return TiePoint(r0, c0, np.nan, np.nan, peak, peak_ratio, np.nan, False, "peak")
    if peak_ratio < peak_ratio_threshold:
        return TiePoint(r0, c0, np.nan, np.nan, peak, peak_ratio, np.nan, False, "peak_ratio")

    sr0, sc0 = rs0 + dr_int, cs0 + dc_int
    sp = slave_amp[sr0:sr0+win, sc0:sc0+win]
    shift_rc, error, _ = phase_cross_correlation(mp, sp, upsample_factor=upsample_factor)
    dr = (sr0 - rm0) + float(shift_rc[0])
    dc = (sc0 - cm0) + float(shift_rc[1])
    return TiePoint(r0, c0, dr, dc, peak, peak_ratio, float(error), True)


def estimate_local_offsets(master_amp, slave_amp, seed_shift_rc, grid_rows=9, grid_cols=9,
                           win=256, search=48, upsample_factor=20,
                           peak_threshold=0.08, peak_ratio_threshold=1.15,
                           texture_threshold=0.03, border=256, max_workers=1):
    H, W = master_amp.shape
    drow0, dcol0 = seed_shift_rc
    rows = np.linspace(border + win // 2, H - border - win // 2, grid_rows)
    cols = np.linspace(border + win // 2, W - border - win // 2, grid_cols)
    centers = [(int(round(r0)), int(round(c0))) for r0 in rows for c0 in cols]
    workers = _resolve_tp_workers(max_workers, len(centers))

    if workers <= 1 or len(centers) <= 1:
        return [
            _estimate_single_local_offset(
                master_amp, slave_amp, r0, c0, drow0, dcol0, win, search, upsample_factor,
                peak_threshold, peak_ratio_threshold, texture_threshold
            )
            for (r0, c0) in centers
        ]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(
            lambda rc: _estimate_single_local_offset(
                master_amp, slave_amp, rc[0], rc[1], drow0, dcol0, win, search, upsample_factor,
                peak_threshold, peak_ratio_threshold, texture_threshold
            ),
            centers
        ))


def fit_offset_models(tie_points: Sequence[TiePoint], min_points=12, order="quadratic"):
    arrs = _valid_tie_point_arrays(tie_points)
    n_good = int(arrs["idx"].size)
    if n_good < min_points:
        raise RuntimeError(f"Not enough valid tie points: {n_good} < {min_points}")
    r = arrs["row"]
    c = arrs["col"]
    dr = arrs["drow"]
    dc = arrs["dcol"]
    phase_error = np.where(np.isfinite(arrs["phase_error"]), arrs["phase_error"], 0.0)
    w = np.maximum(arrs["peak"], 1e-3) * np.maximum(arrs["peak_ratio"], 1.0) / (1.0 + phase_error)
    r_mean, r_std = float(r.mean()), float(r.std() + 1e-6)
    c_mean, c_std = float(c.mean()), float(c.std() + 1e-6)
    rn = (r - r_mean) / r_std
    cn = (c - c_mean) / c_std
    coeffs_dr = fit_surface(rn, cn, dr, weights=w, order=order)
    coeffs_dc = fit_surface(rn, cn, dc, weights=w, order=order)
    return {
        "coeffs_drow": coeffs_dr, "coeffs_dcol": coeffs_dc,
        "r_mean": np.array([r_mean]), "r_std": np.array([r_std]),
        "c_mean": np.array([c_mean]), "c_std": np.array([c_std]),
        "n_valid": np.array([n_good]), "order": np.array([order]),
    }


def evaluate_offset_models(models: Dict[str, np.ndarray], rows, cols):
    r_mean = float(models["r_mean"][0]); r_std = float(models["r_std"][0])
    c_mean = float(models["c_mean"][0]); c_std = float(models["c_std"][0])
    order = str(models.get("order", np.array(["quadratic"]))[0])
    rn = (rows - r_mean) / r_std
    cn = (cols - c_mean) / c_std
    drow = eval_surface(models["coeffs_drow"], rn, cn, order=order)
    dcol = eval_surface(models["coeffs_dcol"], rn, cn, order=order)
    return drow, dcol


def annotate_tie_point_residuals(tie_points, models):
    arrs = _valid_tie_point_arrays(tie_points)
    if arrs["idx"].size == 0:
        return list(tie_points)
    pred_r, pred_c = evaluate_offset_models(models, arrs["row"], arrs["col"])
    rr = arrs["drow"] - pred_r
    cc = arrs["dcol"] - pred_c
    mags = np.hypot(rr, cc)
    for i, tp_idx in enumerate(arrs["idx"]):
        tp = tie_points[int(tp_idx)]
        tp.residual_drow = float(rr[i])
        tp.residual_dcol = float(cc[i])
        tp.residual_mag = float(mags[i])
    return list(tie_points)


def robust_filter_tie_points(tie_points, threshold_px=None, mad_scale=4.5, min_keep=12):
    valid_idx = np.asarray([i for i, tp in enumerate(tie_points) if tp.valid and np.isfinite(tp.residual_mag)], dtype=np.int64)
    n_before = int(valid_idx.size)
    if n_before == 0:
        return list(tie_points), {"n_before": 0, "n_after": 0, "residual_threshold_px": float("nan")}
    mags = np.asarray([tie_points[i].residual_mag for i in valid_idx], dtype=np.float64)
    med = float(np.median(mags))
    mad = float(np.median(np.abs(mags - med)) + 1e-6)
    thr = threshold_px if threshold_px is not None else max(2.0, med + mad_scale * 1.4826 * mad)
    reject_mask = mags > thr
    reject_idx = valid_idx[reject_mask]
    for tp_idx in reject_idx:
        tie_points[int(tp_idx)].valid = False
        tie_points[int(tp_idx)].reject_reason = "mad_outlier"
    n_after = int(np.count_nonzero(~reject_mask))
    if n_after < min_keep:
        for tp_idx in reject_idx:
            tie_points[int(tp_idx)].valid = True
            tie_points[int(tp_idx)].reject_reason = ""
        n_after = n_before
    return list(tie_points), {"n_before": n_before, "n_after": n_after, "residual_threshold_px": float(thr)}


def summarize_tie_points(tie_points):
    arrs = _valid_tie_point_arrays(tie_points)
    valid_idx = arrs["idx"]
    peaks = arrs["peak"]
    ratios = arrs["peak_ratio"]
    errs = arrs["phase_error"][np.isfinite(arrs["phase_error"])]
    out = {
        "n_total": int(len(tie_points)),
        "n_valid": int(valid_idx.size),
        "peak_mean": float(np.mean(peaks)) if peaks.size else float("nan"),
        "peak_median": float(np.median(peaks)) if peaks.size else float("nan"),
        "peak_ratio_mean": float(np.mean(ratios)) if ratios.size else float("nan"),
        "peak_ratio_median": float(np.median(ratios)) if ratios.size else float("nan"),
        "phase_error_mean": float(np.mean(errs)) if errs.size else float("nan"),
        "n_rejected_peak": int(sum(tp.reject_reason == "peak" for tp in tie_points)),
        "n_rejected_peak_ratio": int(sum(tp.reject_reason == "peak_ratio" for tp in tie_points)),
        "n_rejected_mad_outlier": int(sum(tp.reject_reason == "mad_outlier" for tp in tie_points)),
    }
    if valid_idx.size:
        residuals = np.asarray([tie_points[i].residual_mag for i in valid_idx if np.isfinite(tie_points[i].residual_mag)], dtype=np.float64)
        rr = np.asarray([tie_points[i].residual_drow for i in valid_idx if np.isfinite(tie_points[i].residual_drow)], dtype=np.float64)
        cc = np.asarray([tie_points[i].residual_dcol for i in valid_idx if np.isfinite(tie_points[i].residual_dcol)], dtype=np.float64)
        if residuals.size:
            out.update({
                "residual_mag_mean": float(np.mean(residuals)),
                "residual_mag_median": float(np.median(residuals)),
                "residual_mag_p90": float(np.quantile(residuals, 0.9)),
                "residual_row_rmse": float(np.sqrt(np.mean(rr**2))),
                "residual_col_rmse": float(np.sqrt(np.mean(cc**2))),
                "residual_row_medabs": float(np.median(np.abs(rr))),
                "residual_col_medabs": float(np.median(np.abs(cc))),
            })
    return out


def build_offset_grids(models, shape, rr: Optional[np.ndarray] = None, cc: Optional[np.ndarray] = None):
    if rr is None or cc is None:
        rr, cc = _cached_index_grids(tuple(shape))
    drow, dcol = evaluate_offset_models(models, rr, cc)
    return rr, cc, drow, dcol


def remap_complex_cv2(slave_slc: np.ndarray, sample_r: np.ndarray, sample_c: np.ndarray, interp: str = "lanczos", tile: int = 4096) -> np.ndarray:
    import cv2

    if interp == "lanczos":
        interp_flag = cv2.INTER_LANCZOS4
        pad = 4
    else:
        interp_flag = cv2.INTER_CUBIC
        pad = 2

    H, W = slave_slc.shape
    out_ri = np.zeros((H, W, 2), dtype=np.float32)
    src_ri = np.empty((H, W, 2), dtype=np.float32)
    src_ri[..., 0] = np.real(slave_slc).astype(np.float32, copy=False)
    src_ri[..., 1] = np.imag(slave_slc).astype(np.float32, copy=False)

    for r0 in range(0, H, tile):
        r1 = min(H, r0 + tile)
        for c0 in range(0, W, tile):
            c1 = min(W, c0 + tile)
            mapy_full = sample_r[r0:r1, c0:c1].astype(np.float32)
            mapx_full = sample_c[r0:r1, c0:c1].astype(np.float32)

            finite = np.isfinite(mapy_full) & np.isfinite(mapx_full)
            if not np.any(finite):
                continue

            ymin = int(np.floor(np.min(mapy_full[finite]))) - pad
            ymax = int(np.ceil(np.max(mapy_full[finite]))) + pad
            xmin = int(np.floor(np.min(mapx_full[finite]))) - pad
            xmax = int(np.ceil(np.max(mapx_full[finite]))) + pad

            ymin_clip = max(0, ymin)
            xmin_clip = max(0, xmin)
            ymax_clip = min(H, ymax + 1)
            xmax_clip = min(W, xmax + 1)

            src_tile = src_ri[ymin_clip:ymax_clip, xmin_clip:xmax_clip]

            mapy = (mapy_full - ymin_clip).astype(np.float32)
            mapx = (mapx_full - xmin_clip).astype(np.float32)

            bad = (~finite) | (mapy < 0) | (mapx < 0) | (mapy > (ymax_clip - ymin_clip - 1)) | (mapx > (xmax_clip - xmin_clip - 1))
            if np.any(bad):
                mapy = mapy.copy()
                mapx = mapx.copy()
                mapy[bad] = -1e6
                mapx[bad] = -1e6

            out_ri[r0:r1, c0:c1] = cv2.remap(
                src_tile, mapx, mapy,
                interpolation=interp_flag,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
    return out_ri[..., 0] + 1j * out_ri[..., 1]


def remap_complex_scipy(slave_slc: np.ndarray, sample_r: np.ndarray, sample_c: np.ndarray, order: int = 3) -> np.ndarray:
    real = map_coordinates(np.real(slave_slc), [sample_r, sample_c], order=order, mode="constant", cval=0.0, prefilter=True)
    imag = map_coordinates(np.imag(slave_slc), [sample_r, sample_c], order=order, mode="constant", cval=0.0, prefilter=True)
    return real.astype(np.float32) + 1j * imag.astype(np.float32)


def resample_slave_complex(slave_slc, sample_r, sample_c, interp="cubic", tile=4096):
    if interp == "lanczos" and _HAS_CV2:
        return remap_complex_cv2(slave_slc, sample_r, sample_c, interp="lanczos", tile=tile)
    return remap_complex_scipy(slave_slc, sample_r, sample_c, order=3)


def compose_sample_maps(shape, models1, models2=None, rr: Optional[np.ndarray] = None, cc: Optional[np.ndarray] = None):
    rr, cc, drow1, dcol1 = build_offset_grids(models1, shape, rr=rr, cc=cc)
    if models2 is None:
        return rr + drow1, cc + dcol1
    drow2, dcol2 = evaluate_offset_models(models2, rr, cc)
    return rr + drow1 + drow2, cc + dcol1 + dcol2


def interferogram(master_slc, slave_coreg):
    return master_slc * np.conj(slave_coreg)


def _precompute_master_coherence_denominator(master_slc: np.ndarray, win: int = 9) -> np.ndarray:
    return uniform_filter(np.abs(master_slc) ** 2, size=win, mode="nearest")


def _coherence_scipy(master_slc, slave_coreg, win=9, master_den1: Optional[np.ndarray] = None):
    num = uniform_filter(master_slc * np.conj(slave_coreg), size=win, mode="nearest")
    den1 = master_den1 if master_den1 is not None else uniform_filter(np.abs(master_slc) ** 2, size=win, mode="nearest")
    den2 = uniform_filter(np.abs(slave_coreg) ** 2, size=win, mode="nearest")
    coh = np.abs(num) / np.sqrt(np.maximum(den1 * den2, 1e-12))
    return np.clip(coh.astype(np.float32), 0.0, 1.0)


def _coherence_opencv(master_slc, slave_coreg, win=9, master_den1: Optional[np.ndarray] = None):
    if not _HAS_CV2:
        raise RuntimeError("OpenCV coherence backend requested but cv2 is unavailable")
    ksize = (int(win), int(win))
    prod = master_slc * np.conj(slave_coreg)
    num_real = cv2.boxFilter(np.real(prod).astype(np.float32), ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REPLICATE, normalize=True)
    num_imag = cv2.boxFilter(np.imag(prod).astype(np.float32), ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REPLICATE, normalize=True)
    den1 = master_den1 if master_den1 is not None else cv2.boxFilter((np.abs(master_slc) ** 2).astype(np.float32), ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REPLICATE, normalize=True)
    den2 = cv2.boxFilter((np.abs(slave_coreg) ** 2).astype(np.float32), ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REPLICATE, normalize=True)
    coh = np.hypot(num_real, num_imag) / np.sqrt(np.maximum(den1 * den2, 1e-12))
    return np.clip(coh.astype(np.float32), 0.0, 1.0)


def coherence(master_slc, slave_coreg, win=9, master_den1: Optional[np.ndarray] = None, backend: str = "scipy"):
    if backend == "opencv":
        return _coherence_opencv(master_slc, slave_coreg, win=win, master_den1=master_den1)
    return _coherence_scipy(master_slc, slave_coreg, win=win, master_den1=master_den1)


def make_commonband_mask(shape, frac=0.92, smooth_frac=0.06):
    return _cached_commonband_mask(tuple(shape), float(frac), float(smooth_frac))


def apply_commonband_filter_pair(master_slc, slave_slc, frac=0.92, smooth_frac=0.06):
    mask = make_commonband_mask(master_slc.shape, frac=frac, smooth_frac=smooth_frac)
    M = np.fft.fft2(master_slc) * mask
    S = np.fft.fft2(slave_slc) * mask
    return np.fft.ifft2(M).astype(np.complex64), np.fft.ifft2(S).astype(np.complex64), mask


def coherence_stats(coh):
    return _stats_from_array(np.asarray(coh, dtype=np.float32))


def save_tiff(path: Path, arr: np.ndarray):
    tifffile.imwrite(str(path), arr)


def save_quicklooks(out_dir: Path, master_amp, slave_amp, slave_coreg_amp, ifg, coh, tie_points):
    fringe_cmap = make_fringe_cmap()

    def save_img(path, img, cmap="gray", vmin=None, vmax=None):
        plt.figure(figsize=(10, 8))
        plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    save_img(out_dir / "master_amplitude.png", master_amp, cmap="gray")
    save_img(out_dir / "slave_amplitude.png", slave_amp, cmap="gray")
    save_img(out_dir / "slave_coreg_amplitude.png", slave_coreg_amp, cmap="gray")
    save_img(out_dir / "interferogram_phase.png", np.angle(ifg), cmap=fringe_cmap, vmin=-np.pi, vmax=np.pi)
    save_img(out_dir / "coherence.png", coh, cmap="gray", vmin=0.0, vmax=1.0)

    good = np.asarray([(tp.col, tp.row) for tp in tie_points if tp.valid], dtype=np.float64)
    bad = np.asarray([(tp.col, tp.row) for tp in tie_points if not tp.valid], dtype=np.float64)

    plt.figure(figsize=(10, 8))
    plt.imshow(master_amp, cmap="gray")
    if good.size:
        plt.scatter(good[:, 0], good[:, 1], s=28, c="cyan", edgecolors="black", linewidths=0.5, label="valid")
    if bad.size:
        plt.scatter(bad[:, 0], bad[:, 1], s=28, c="red", edgecolors="black", linewidths=0.5, label="rejected")
    if good.size or bad.size:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "tie_points.png", dpi=150)
    plt.close()

    good_res = np.asarray([(tp.col, tp.row, tp.residual_dcol, tp.residual_drow, tp.residual_mag)
                           for tp in tie_points if tp.valid and np.isfinite(tp.residual_mag)], dtype=np.float64)
    if good_res.size:
        plt.figure(figsize=(10, 8))
        plt.imshow(master_amp, cmap="gray")
        qx = good_res[:, 0]
        qy = good_res[:, 1]
        qu = good_res[:, 2]
        qv = good_res[:, 3]
        plt.quiver(qx, qy, qu, qv, color="cyan", angles="xy", scale_units="xy",
                   scale=0.25, width=0.004, headwidth=4.5, headlength=6.0,
                   headaxislength=4.5, alpha=0.95)
        plt.scatter(qx, qy, s=10, c="yellow", edgecolors="black", linewidths=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "tie_point_residual_vectors.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.hist(good_res[:, 4], bins=20, color="dimgray", edgecolor="white")
        plt.xlabel("Residual magnitude [px]")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / "tie_point_residual_hist.png", dpi=150)
        plt.close()


def phase_to_rgb_tiff(phase: np.ndarray, cmap_name: str = "hsv") -> np.ndarray:
    phase_norm = (np.asarray(phase, dtype=np.float32) + np.pi) / (2.0 * np.pi)
    phase_norm = np.mod(phase_norm, 1.0)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(phase_norm)
    return np.round(255.0 * rgba[..., :3]).astype(np.uint8)


def goldstein_filter_interferogram(ifg: np.ndarray, alpha: float = 0.5, block_size: int = 32, step: Optional[int] = None) -> np.ndarray:
    z = np.asarray(ifg, dtype=np.complex64)
    H, W = z.shape
    block_size = int(block_size)
    if block_size < 4:
        raise ValueError("goldstein block_size must be >= 4")
    if step is None:
        step = max(1, block_size // 2)
    step = int(step)
    if step < 1 or step > block_size:
        raise ValueError("goldstein step must be in [1, block_size]")

    win1d = np.hanning(block_size).astype(np.float32)
    if not np.any(win1d):
        win1d = np.ones(block_size, dtype=np.float32)
    win2d = np.outer(win1d, win1d).astype(np.float32)

    out = np.zeros((H, W), dtype=np.complex64)
    wsum = np.zeros((H, W), dtype=np.float32)

    row_starts = list(range(0, max(H - block_size + 1, 1), step))
    col_starts = list(range(0, max(W - block_size + 1, 1), step))
    last_r = max(H - block_size, 0)
    last_c = max(W - block_size, 0)
    if row_starts[-1] != last_r:
        row_starts.append(last_r)
    if col_starts[-1] != last_c:
        col_starts.append(last_c)

    eps = 1e-6
    alpha = float(alpha)

    for r0 in row_starts:
        r1 = r0 + block_size
        for c0 in col_starts:
            c1 = c0 + block_size
            patch = z[r0:r1, c0:c1]
            if patch.shape != (block_size, block_size):
                pad = np.zeros((block_size, block_size), dtype=np.complex64)
                pad[:patch.shape[0], :patch.shape[1]] = patch
                patch = pad
            F = np.fft.fft2(patch)
            mag = np.abs(F).astype(np.float32)
            Hf = np.power(np.maximum(mag, eps), alpha).astype(np.float32)
            Hf /= (np.max(Hf) + eps)
            patch_f = np.fft.ifft2(F * Hf).astype(np.complex64)
            h_eff = min(block_size, H - r0)
            w_eff = min(block_size, W - c0)
            w = win2d[:h_eff, :w_eff]
            out[r0:r0+h_eff, c0:c0+w_eff] += patch_f[:h_eff, :w_eff] * w
            wsum[r0:r0+h_eff, c0:c0+w_eff] += w

    mask = wsum > 0
    out[mask] /= wsum[mask]
    out[~mask] = z[~mask]
    return out


def save_qgis_products(out_dir: Path, ifg: np.ndarray, coh: np.ndarray,
                       goldstein_alpha: float = 0.5,
                       goldstein_block_size: int = 32,
                       goldstein_step: Optional[int] = None,
                       phase_cmap: str = "hsv") -> Dict[str, float]:
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    raw_complex = np.stack(
        [np.real(ifg).astype(np.float32), np.imag(ifg).astype(np.float32)],
        axis=-1
    )
    save_tiff(out_dir / "ifg_raw_complex_real_imag.tif", raw_complex)
    timings["ifg_raw_complex_tiff"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    raw_phase = np.angle(ifg).astype(np.float32)
    save_tiff(out_dir / "ifg_raw.tif", raw_phase)
    timings["ifg_raw_tiff"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ifg_gold = goldstein_filter_interferogram(
        ifg,
        alpha=goldstein_alpha,
        block_size=goldstein_block_size,
        step=goldstein_step,
    )
    timings["goldstein_filter"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    gold_complex = np.stack(
        [np.real(ifg_gold).astype(np.float32), np.imag(ifg_gold).astype(np.float32)],
        axis=-1
    )
    save_tiff(out_dir / "ifg_goldstein_complex_real_imag.tif", gold_complex)
    timings["ifg_goldstein_complex_tiff"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    gold_phase = np.angle(ifg_gold).astype(np.float32)
    save_tiff(out_dir / "ifg_goldstein.tif", gold_phase)
    timings["ifg_goldstein_tiff"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    save_tiff(out_dir / "coherence.tif", coh.astype(np.float32))
    timings["coherence_tiff"] = time.perf_counter() - t0

    return timings


def build_initial_shift_seed(master_meta, slave_meta):
    return float(slave_meta.scene_ref_row - master_meta.scene_ref_row), float(slave_meta.scene_ref_col - master_meta.scene_ref_col)




def write_coreg_meta_json(out_dir: Path,
                          master_meta: CapellaMeta,
                          slave_meta: CapellaMeta,
                          compatibility: Dict[str, object],
                          diagnostics: Dict[str, object],
                          tps_final: Sequence[TiePoint],
                          patch_win: int,
                          H: int,
                          W: int) -> None:
    arrs = _valid_tie_point_arrays(tps_final)
    row_vals = arrs["drow"]
    col_vals = arrs["dcol"]
    peak_vals = arrs["peak"]
    row_res = np.asarray([tps_final[i].residual_drow for i in arrs["idx"] if np.isfinite(tps_final[i].residual_drow)], dtype=np.float64)
    col_res = np.asarray([tps_final[i].residual_dcol for i in arrs["idx"] if np.isfinite(tps_final[i].residual_dcol)], dtype=np.float64)
    rms_vals = np.asarray([tps_final[i].residual_mag for i in arrs["idx"] if np.isfinite(tps_final[i].residual_mag)], dtype=np.float64)

    dinc_deg = abs(master_meta.incidence_angle_deg - slave_meta.incidence_angle_deg)
    coh_mean = float(diagnostics["coherence"]["mean"])
    rms_mean = float(np.mean(rms_vals)) if rms_vals.size else float("nan")
    q_score = float(np.clip(coh_mean * np.exp(-0.5 * (0.0 if not np.isfinite(rms_mean) else rms_mean)), 0.0, 1.0))

    meta = {
        "id_ref": Path(master_meta.json_path).stem.replace("_extended", ""),
        "id_sec": Path(slave_meta.json_path).stem.replace("_extended", ""),
        "dt_days": float(compatibility["center_time_separation_sec"]) / 86400.0,
        "dinc_deg": float(dinc_deg),
        "q_score": q_score,
        "bperp_m": float(compatibility["baseline_perpendicular_m"]),
        "row_offset_px": float(np.mean(row_vals)) if row_vals.size else float("nan"),
        "col_offset_px": float(np.mean(col_vals)) if col_vals.size else float("nan"),
        "patch_size": int(patch_win),
        "patch_row_ref": int(H // 2),
        "patch_col_ref": int(W // 2),
        "cc_peak_mean": float(np.mean(peak_vals)) if peak_vals.size else float("nan"),
        "cc_peak_min": float(np.min(peak_vals)) if peak_vals.size else float("nan"),
        "n_coreg_patches": int(arrs["idx"].size),
        "n_coreg_patches_total": int(len(tps_final)),
        "offset_row_std_px": float(np.std(row_vals)) if row_vals.size else float("nan"),
        "offset_col_std_px": float(np.std(col_vals)) if col_vals.size else float("nan"),
        "estimated_rotation_mrad": 0.0,
        "rms_mean_px": float(np.mean(rms_vals)) if rms_vals.size else float("nan"),
        "rms_std_px": float(np.std(rms_vals)) if rms_vals.size else float("nan"),
        "row_residual_mean_px": float(np.mean(row_res)) if row_res.size else float("nan"),
        "row_residual_std_px": float(np.std(row_res)) if row_res.size else float("nan"),
        "col_residual_mean_px": float(np.mean(col_res)) if col_res.size else float("nan"),
        "col_residual_std_px": float(np.std(col_res)) if col_res.size else float("nan"),
        "coherence_mean": float(diagnostics["coherence"]["mean"]),
        "coherence_median": float(diagnostics["coherence"]["median"]),
        "coherence_p90": float(diagnostics["coherence"]["p90"]),
        "coherence_p95": float(diagnostics["coherence"]["p95"]),
        "selected_candidate": diagnostics.get("selected_candidate", ""),
        "coreg_stage": "pass2" if "pass2" in diagnostics.get("selected_candidate", "") else "pass1",
    }
    with open(out_dir / "coreg_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, cls=NpEncoder)


def write_coreg_residuals_txt(out_dir: Path, tps_final: Sequence[TiePoint]) -> None:
    arrs = _valid_tie_point_arrays(tps_final)
    lines = []
    header = f'{"GCP":>12s}{"ref_x":>14s}{"ref_y":>14s}{"sec_x":>14s}{"sec_y":>14s}{"rms":>14s}'
    lines.append(header)

    for i, tp_idx in enumerate(arrs["idx"]):
        tp = tps_final[int(tp_idx)]
        ref_x = tp.col
        ref_y = tp.row
        sec_x = tp.col + tp.dcol
        sec_y = tp.row + tp.drow
        rms = tp.residual_mag if np.isfinite(tp.residual_mag) else float("nan")
        lines.append(f'{("GCP"+str(i)):>12s}{ref_x:14.4f}{ref_y:14.4f}{sec_x:14.4f}{sec_y:14.4f}{rms:14.4f}')

    row_res = np.asarray([tps_final[i].residual_drow for i in arrs["idx"] if np.isfinite(tps_final[i].residual_drow)], dtype=np.float64)
    col_res = np.asarray([tps_final[i].residual_dcol for i in arrs["idx"] if np.isfinite(tps_final[i].residual_dcol)], dtype=np.float64)
    rms_vals = np.asarray([tps_final[i].residual_mag for i in arrs["idx"] if np.isfinite(tps_final[i].residual_mag)], dtype=np.float64)

    lines.append("")
    lines.append(f'{"rmsStd":<18s}{(float(np.std(rms_vals)) if rms_vals.size else float("nan")):0.4f}')
    lines.append(f'{"rmsMean":<18s}{(float(np.mean(rms_vals)) if rms_vals.size else float("nan")):0.4f}')
    lines.append(f'{"rowResidualStd":<18s}{(float(np.std(row_res)) if row_res.size else float("nan")):0.4f}')
    lines.append(f'{"rowResidualMean":<18s}{(float(np.mean(row_res)) if row_res.size else float("nan")):0.4f}')
    lines.append(f'{"colResidualStd":<18s}{(float(np.std(col_res)) if col_res.size else float("nan")):0.4f}')
    lines.append(f'{"colResidualMean":<18s}{(float(np.mean(col_res)) if col_res.size else float("nan")):0.4f}')
    lines.append("")
    with open(out_dir / "coreg_residuals.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_pipeline(master_tif, slave_tif, master_json, slave_json, out_dir,
                 thumb_looks_row=16, thumb_looks_col=16, grid_rows=9, grid_cols=9,
                 patch_win=256, search_radius=48, local_upsample=20,
                 pass2_search_radius=8, residual_pass=True,
                 interp="lanczos", commonband_frac=0.92, commonband_smooth_frac=0.06,
                 save_candidate_products=False, remap_tile=4096, allow_commonband_selection=False,
                 pass1_peak_threshold=0.08, pass1_peak_ratio_threshold=1.08,
                 pass2_peak_threshold=0.05, pass2_peak_ratio_threshold=1.12,
                 tp_workers=1, save_png_quicklooks=False, save_npy_outputs=False,
                 skip_pass2=False, coherence_backend="scipy",
                 goldstein_alpha=0.5, goldstein_block_size=32,
                 goldstein_step=None, phase_cmap="hsv",
                 min_coherence_mean: Optional[float] = None):
    out_dir = ensure_dir(out_dir)
    timings: Dict[str, float] = {}
    coherence_win = 9

    def _tic() -> float:
        return time.perf_counter()

    def _toc(name: str, start: float) -> None:
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - start)

    t0 = _tic()
    master_meta = parse_capella_extended_json(master_json)
    slave_meta = parse_capella_extended_json(slave_json)
    compatibility = pair_compatibility_report(master_meta, slave_meta)
    _toc("metadata", t0)

    t0 = _tic()
    master_slc = read_complex_slc(master_tif, expected_rows=master_meta.rows, expected_cols=master_meta.cols)
    slave_slc = read_complex_slc(slave_tif, expected_rows=slave_meta.rows, expected_cols=slave_meta.cols)
    H, W = min(master_slc.shape[0], slave_slc.shape[0]), min(master_slc.shape[1], slave_slc.shape[1])
    master_slc = master_slc[:H, :W]
    slave_slc = slave_slc[:H, :W]
    rr_full, cc_full = _cached_index_grids((H, W))
    _toc("read_crop", t0)

    t0 = _tic()
    master_amp = robust_amplitude(master_slc, log_scale=True, sigma=0.5)
    slave_amp = robust_amplitude(slave_slc, log_scale=True, sigma=0.5)
    _toc("amplitude", t0)

    t0 = _tic()
    seed_shift = build_initial_shift_seed(master_meta, slave_meta)
    master_thumb = multilook_mean(master_amp, thumb_looks_row, thumb_looks_col)
    slave_thumb = multilook_mean(slave_amp, thumb_looks_row, thumb_looks_col)
    drow_thumb, dcol_thumb, thumb_info = estimate_global_shift_thumbnail(master_thumb, slave_thumb)
    coarse_shift = (seed_shift[0] + drow_thumb * thumb_looks_row, seed_shift[1] + dcol_thumb * thumb_looks_col)
    _toc("thumbnail_shift", t0)

    diagnostics = {
        "compatibility": compatibility,
        "seed_shift_rc": seed_shift,
        "thumbnail_shift_rc": (drow_thumb, dcol_thumb),
        "thumbnail_info": thumb_info,
        "coarse_shift_rc": coarse_shift,
        "interpolation": interp if (interp != "lanczos" or _HAS_CV2) else "cubic_fallback",
        "commonband_auto_selection_enabled": bool(allow_commonband_selection),
        "commonband_branch_executed": False,
        "gpu_support": check_gpu_support(),
        "output_options": {
            "save_png_quicklooks": bool(save_png_quicklooks),
            "save_npy_outputs": bool(save_npy_outputs),
            "save_candidate_products": bool(save_candidate_products),
        },
        "skip_pass2": bool(skip_pass2),
        "coherence_backend": str(coherence_backend),
        "tie_point_workers": int(_resolve_tp_workers(tp_workers, grid_rows * grid_cols)),
        "thresholds": {
            "pass1_peak_threshold": float(pass1_peak_threshold),
            "pass1_peak_ratio_threshold": float(pass1_peak_ratio_threshold),
            "pass2_peak_threshold": float(pass2_peak_threshold),
            "pass2_peak_ratio_threshold": float(pass2_peak_ratio_threshold),
        },
        "min_coherence_mean": None if min_coherence_mean is None else float(min_coherence_mean),
    }

    t0 = _tic()
    master_den1 = _precompute_master_coherence_denominator(master_slc, win=coherence_win)
    _toc("master_coherence_denominator", t0)

    t0 = _tic()
    tps1 = estimate_local_offsets(master_amp, slave_amp, coarse_shift, grid_rows, grid_cols, patch_win,
                                  search_radius, local_upsample,
                                  peak_threshold=pass1_peak_threshold,
                                  peak_ratio_threshold=pass1_peak_ratio_threshold,
                                  max_workers=tp_workers)
    _toc("pass1_tie_points", t0)

    t0 = _tic()
    models1 = fit_offset_models(tps1, order="quadratic")
    tps1 = annotate_tie_point_residuals(tps1, models1)
    tps1, filt1 = robust_filter_tie_points(tps1, threshold_px=None, mad_scale=4.5)
    models1 = fit_offset_models(tps1, order="quadratic")
    tps1 = annotate_tie_point_residuals(tps1, models1)
    diagnostics["tie_points_pass1"] = summarize_tie_points(tps1)
    diagnostics["pass1_model_order"] = "quadratic"
    diagnostics["pass1_robust_filter"] = filt1
    _toc("pass1_model_fit_filter", t0)

    t0 = _tic()
    sr1, sc1 = compose_sample_maps((H, W), models1, None, rr=rr_full, cc=cc_full)
    slave_coreg1 = resample_slave_complex(slave_slc, sr1, sc1, interp=interp, tile=remap_tile)
    _toc("pass1_remap", t0)

    t0 = _tic()
    coh1 = coherence(master_slc, slave_coreg1, win=coherence_win, master_den1=master_den1, backend=coherence_backend)
    coh1_stats = coherence_stats(coh1)
    diagnostics["candidate_pass1_coherence"] = coh1_stats
    _toc("pass1_coherence", t0)

    candidates = {
        "pass1": {
            "slave": slave_coreg1,
            "coh": coh1,
            "coh_stats": coh1_stats,
            "tie_points": tps1,
            "model_order": "quadratic",
        }
    }

    models2 = None
    if residual_pass and skip_pass2:
        diagnostics["pass2_skipped"] = True
    if residual_pass and not skip_pass2:
        t0 = _tic()
        slave_coreg1_amp = robust_amplitude(slave_coreg1, log_scale=True, sigma=0.5)
        _toc("pass2_amplitude", t0)

        t0 = _tic()
        pass2_grid_rows = max(grid_rows + 2, grid_rows)
        pass2_grid_cols = max(grid_cols + 2, grid_cols)
        tps2 = estimate_local_offsets(master_amp, slave_coreg1_amp, (0.0, 0.0),
                                      pass2_grid_rows, pass2_grid_cols,
                                      patch_win, pass2_search_radius, local_upsample,
                                      peak_threshold=pass2_peak_threshold,
                                      peak_ratio_threshold=pass2_peak_ratio_threshold,
                                      max_workers=tp_workers)
        diagnostics["tie_point_workers_pass2"] = int(_resolve_tp_workers(tp_workers, pass2_grid_rows * pass2_grid_cols))
        _toc("pass2_tie_points", t0)

        t0 = _tic()
        models2 = fit_offset_models(tps2, order="linear")
        tps2 = annotate_tie_point_residuals(tps2, models2)
        tps2, filt2 = robust_filter_tie_points(tps2, threshold_px=2.0, mad_scale=3.5)
        models2 = fit_offset_models(tps2, order="linear")
        tps2 = annotate_tie_point_residuals(tps2, models2)
        diagnostics["tie_points_pass2"] = summarize_tie_points(tps2)
        diagnostics["pass2_model_order"] = "linear"
        diagnostics["pass2_robust_filter"] = filt2
        _toc("pass2_model_fit_filter", t0)

        t0 = _tic()
        d2r, d2c = evaluate_offset_models(models2, rr_full, cc_full)
        slave_coreg2_seq = resample_slave_complex(slave_coreg1, rr_full + d2r, cc_full + d2c, interp=interp, tile=remap_tile)
        _toc("pass2_sequential_remap", t0)

        t0 = _tic()
        coh2_seq = coherence(master_slc, slave_coreg2_seq, win=coherence_win, master_den1=master_den1, backend=coherence_backend)
        coh2_seq_stats = coherence_stats(coh2_seq)
        diagnostics["candidate_pass2_sequential_coherence"] = coh2_seq_stats
        _toc("pass2_sequential_coherence", t0)
        candidates["pass2_sequential"] = {
            "slave": slave_coreg2_seq,
            "coh": coh2_seq,
            "coh_stats": coh2_seq_stats,
            "tie_points": tps2,
            "model_order": "linear",
        }

        t0 = _tic()
        sr_comp, sc_comp = compose_sample_maps((H, W), models1, models2, rr=rr_full, cc=cc_full)
        slave_coreg2_comp = resample_slave_complex(slave_slc, sr_comp, sc_comp, interp=interp, tile=remap_tile)
        _toc("pass2_composed_remap", t0)

        t0 = _tic()
        coh2_comp = coherence(master_slc, slave_coreg2_comp, win=coherence_win, master_den1=master_den1, backend=coherence_backend)
        coh2_comp_stats = coherence_stats(coh2_comp)
        diagnostics["candidate_pass2_composed_coherence"] = coh2_comp_stats
        _toc("pass2_composed_coherence", t0)
        candidates["pass2_composed"] = {
            "slave": slave_coreg2_comp,
            "coh": coh2_comp,
            "coh_stats": coh2_comp_stats,
            "tie_points": tps2,
            "model_order": "quadratic+linear_composed",
        }


    t0 = _tic()
    def score(item):
        st = item["coh_stats"]
        return (st["mean"], st["median"], st["p90"], st["p95"])

    selected_name = max(candidates.keys(), key=lambda k: score(candidates[k]))
    selected = candidates[selected_name]
    diagnostics["selected_candidate"] = selected_name
    final_master = selected.get("master", master_slc)
    final_slave = selected["slave"]
    final_coh = selected["coh"]
    final_tps = selected["tie_points"]
    diagnostics["coherence"] = selected["coh_stats"]
    diagnostics["final_tie_point_residuals"] = summarize_tie_points(final_tps)
    diagnostics["final_model_order"] = selected["model_order"]
    _toc("candidate_selection", t0)

    t0 = _tic()
    ifg = interferogram(final_master, final_slave)
    _toc("final_ifg", t0)

    coherence_mean = float(diagnostics["coherence"]["mean"])
    if min_coherence_mean is not None and coherence_mean < float(min_coherence_mean):
        diagnostics["pair_rejected"] = True
        diagnostics["rejection_reason"] = "coherence_below_threshold"
        diagnostics["rejection_threshold_mean"] = float(min_coherence_mean)
        diagnostics["accepted_for_output"] = False

        t0 = _tic()
        diagnostics["timings_sec"] = {k: float(v) for k, v in sorted(timings.items(), key=lambda kv: kv[0])}
        total_runtime_sec = float(sum(diagnostics["timings_sec"].values()))
        diagnostics["timings_pct"] = {k: float((v / total_runtime_sec) * 100.0) if total_runtime_sec > 0 else 0.0 for k, v in diagnostics["timings_sec"].items()}
        diagnostics["runtime_total_sec"] = total_runtime_sec
        diagnostics["runtime_total_min"] = total_runtime_sec / 60.0
        write_coreg_meta_json(out_dir, master_meta, slave_meta, compatibility, diagnostics, final_tps, patch_win, H, W)
        write_coreg_residuals_txt(out_dir, final_tps)
        _toc("write_metadata", t0)

        diagnostics["timings_sec"] = {k: float(v) for k, v in sorted(timings.items(), key=lambda kv: kv[0])}
        total_runtime_sec = float(sum(diagnostics["timings_sec"].values()))
        diagnostics["timings_pct"] = {k: float((v / total_runtime_sec) * 100.0) if total_runtime_sec > 0 else 0.0 for k, v in diagnostics["timings_sec"].items()}
        diagnostics["runtime_total_sec"] = total_runtime_sec
        diagnostics["runtime_total_min"] = total_runtime_sec / 60.0
        return diagnostics
    else:
        diagnostics["pair_rejected"] = False
        diagnostics["accepted_for_output"] = True

    if save_png_quicklooks:
        t0 = _tic()
        final_slave_amp = robust_amplitude(final_slave, log_scale=True, sigma=0.5)
        _toc("final_slave_amplitude", t0)

        t0 = _tic()
        save_quicklooks(out_dir, master_amp, slave_amp, final_slave_amp, ifg, final_coh, final_tps)
        _toc("write_png_quicklooks", t0)

    t0 = _tic()
    tiff_timings = save_qgis_products(
        out_dir, ifg, final_coh,
        goldstein_alpha=goldstein_alpha,
        goldstein_block_size=goldstein_block_size,
        goldstein_step=goldstein_step,
        phase_cmap=phase_cmap,
    )
    if save_candidate_products:
        for name, item in candidates.items():
            sub = ensure_dir(out_dir / f"candidate_{name}")
            ifg_c = interferogram(item.get("master", master_slc), item["slave"])
            save_qgis_products(
                sub, ifg_c, item["coh"],
                goldstein_alpha=goldstein_alpha,
                goldstein_block_size=goldstein_block_size,
                goldstein_step=goldstein_step,
                phase_cmap=phase_cmap,
            )
    _toc("write_tiff_outputs", t0)
    diagnostics["tiff_write_breakdown_sec"] = {k: float(v) for k, v in tiff_timings.items()}

    if save_npy_outputs:
        t0 = _tic()
        np.save(out_dir / "master_slc_cropped.npy", final_master)
        np.save(out_dir / "slave_slc_cropped.npy", slave_slc)
        np.save(out_dir / "slave_coreg.npy", final_slave)
        np.save(out_dir / "interferogram.npy", ifg)
        np.save(out_dir / "coherence.npy", final_coh)
        np.savez(out_dir / "offset_model_pass1.npz", **models1)
        if models2 is not None:
            np.savez(out_dir / "offset_model_pass2.npz", **models2)
        _toc("write_numpy_outputs", t0)

    t0 = _tic()
    diagnostics["timings_sec"] = {k: float(v) for k, v in sorted(timings.items(), key=lambda kv: kv[0])}
    total_runtime_sec = float(sum(diagnostics["timings_sec"].values()))
    diagnostics["timings_pct"] = {k: float((v / total_runtime_sec) * 100.0) if total_runtime_sec > 0 else 0.0 for k, v in diagnostics["timings_sec"].items()}
    diagnostics["runtime_total_sec"] = total_runtime_sec
    diagnostics["runtime_total_min"] = total_runtime_sec / 60.0
    write_coreg_meta_json(out_dir, master_meta, slave_meta, compatibility, diagnostics, final_tps, patch_win, H, W)
    write_coreg_residuals_txt(out_dir, final_tps)
    with open(out_dir / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, cls=NpEncoder)
    _toc("write_metadata", t0)

    diagnostics["timings_sec"] = {k: float(v) for k, v in sorted(timings.items(), key=lambda kv: kv[0])}
    total_runtime_sec = float(sum(diagnostics["timings_sec"].values()))
    diagnostics["timings_pct"] = {k: float((v / total_runtime_sec) * 100.0) if total_runtime_sec > 0 else 0.0 for k, v in diagnostics["timings_sec"].items()}
    diagnostics["runtime_total_sec"] = total_runtime_sec
    diagnostics["runtime_total_min"] = total_runtime_sec / 60.0
    with open(out_dir / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, cls=NpEncoder)
    return diagnostics


def _pipeline_kwargs_from_args(args) -> Dict[str, object]:
    return {
        "thumb_looks_row": args.thumb_looks_row,
        "thumb_looks_col": args.thumb_looks_col,
        "grid_rows": args.grid_rows,
        "grid_cols": args.grid_cols,
        "patch_win": args.patch_win,
        "search_radius": args.search_radius,
        "pass2_search_radius": args.pass2_search_radius,
        "local_upsample": args.local_upsample,
        "residual_pass": not args.no_residual_pass,
        "interp": args.interp,
        "commonband_frac": args.commonband_frac,
        "commonband_smooth_frac": args.commonband_smooth_frac,
        "save_candidate_products": args.save_candidate_products,
        "remap_tile": args.remap_tile,
        "allow_commonband_selection": args.allow_commonband_selection,
        "pass1_peak_threshold": args.pass1_peak_threshold,
        "pass1_peak_ratio_threshold": args.pass1_peak_ratio_threshold,
        "pass2_peak_threshold": args.pass2_peak_threshold,
        "pass2_peak_ratio_threshold": args.pass2_peak_ratio_threshold,
        "tp_workers": args.tp_workers,
        "save_png_quicklooks": args.save_png_quicklooks,
        "save_npy_outputs": args.save_npy_outputs,
        "skip_pass2": args.skip_pass2,
        "coherence_backend": args.coherence_backend,
        "goldstein_alpha": args.goldstein_alpha,
        "goldstein_block_size": args.goldstein_block_size,
        "goldstein_step": args.goldstein_step,
        "phase_cmap": args.phase_cmap,
        "min_coherence_mean": args.min_coherence_mean,
    }


def _normalize_pair_id(value: object) -> str:
    return str(value).strip()


def _candidate_file_paths(raw_dir: Path, pair_id: str, suffixes: Sequence[str]) -> List[Path]:
    return [raw_dir / f"{pair_id}{suffix}" for suffix in suffixes]


def _resolve_pair_file(raw_dir: Path, pair_id: str, kind: str, index: Dict[Tuple[str, str], str]) -> str:
    pair_id = _normalize_pair_id(pair_id)
    if kind == "tif":
        suffixes = [".tif", ".tiff"]
    elif kind == "json":
        suffixes = ["_extended.json", ".json"]
    else:
        raise ValueError(f"Unsupported kind: {kind}")

    for cand in _candidate_file_paths(raw_dir, pair_id, suffixes):
        if cand.exists():
            return str(cand)

    indexed = index.get((kind, pair_id))
    if indexed is not None:
        return indexed

    raise FileNotFoundError(f"Could not resolve {kind} for pair id '{pair_id}' under {raw_dir}")


def _build_raw_dir_index(raw_dir: Path) -> Dict[Tuple[str, str], str]:
    index: Dict[Tuple[str, str], str] = {}
    for path in raw_dir.rglob('*'):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if lower.endswith(('.tif', '.tiff')):
            index.setdefault(("tif", path.stem), str(path))
        elif lower.endswith('.json'):
            stem = path.stem
            if stem.endswith('_extended'):
                index.setdefault(("json", stem[:-9]), str(path))
            index.setdefault(("json", stem), str(path))
    return index


def _pick_first_existing_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        actual = cols_lower.get(cand.lower())
        if actual is not None:
            return actual
    return None


def _read_pairs_parquet_manifest(manifest_path: os.PathLike,
                                 raw_dir: os.PathLike,
                                 batch_out_dir: os.PathLike) -> List[Dict[str, str]]:
    if not _HAS_PANDAS:
        raise ImportError("pandas is required to read --pairs_manifest parquet files")

    manifest_path = Path(manifest_path)
    raw_dir = Path(raw_dir)
    batch_out_dir = Path(batch_out_dir)
    df = pd.read_parquet(manifest_path)
    if df.empty:
        return []

    columns = list(df.columns)
    master_id_col = _pick_first_existing_column(columns, [
        'master_id', 'id_ref', 'ref_id', 'reference_id', 'master', 'ref', 'reference', 'primary_id'
    ])
    slave_id_col = _pick_first_existing_column(columns, [
        'slave_id', 'id_sec', 'sec_id', 'secondary_id', 'slave', 'sec', 'secondary'
    ])

    master_tif_col = _pick_first_existing_column(columns, ['master_tif', 'ref_tif', 'reference_tif'])
    slave_tif_col = _pick_first_existing_column(columns, ['slave_tif', 'sec_tif', 'secondary_tif'])
    master_json_col = _pick_first_existing_column(columns, ['master_json', 'ref_json', 'reference_json'])
    slave_json_col = _pick_first_existing_column(columns, ['slave_json', 'sec_json', 'secondary_json'])
    pair_id_col = _pick_first_existing_column(columns, ['pair_id', 'pair_name', 'name'])
    out_dir_col = _pick_first_existing_column(columns, ['out_dir'])

    if (master_tif_col is None or slave_tif_col is None or master_json_col is None or slave_json_col is None) and (master_id_col is None or slave_id_col is None):
        raise ValueError(
            "Parquet manifest must contain either full path columns "
            "(master_tif, slave_tif, master_json, slave_json) or id columns like "
            "(master_id/id_ref, slave_id/id_sec)."
        )

    raw_index = _build_raw_dir_index(raw_dir)
    jobs: List[Dict[str, str]] = []
    for i, row in df.iterrows():
        row_dict = row.to_dict()
        if pair_id_col is not None and pd.notna(row_dict.get(pair_id_col)):
            pair_id = _normalize_pair_id(row_dict[pair_id_col])
        elif master_id_col is not None and slave_id_col is not None:
            pair_id = f"{_normalize_pair_id(row_dict[master_id_col])}__{_normalize_pair_id(row_dict[slave_id_col])}"
        else:
            pair_id = f"pair_{i:05d}"

        job: Dict[str, str] = {"pair_id": pair_id}

        if master_tif_col is not None and pd.notna(row_dict.get(master_tif_col)):
            job['master_tif'] = str(row_dict[master_tif_col])
        else:
            job['master_tif'] = _resolve_pair_file(raw_dir, row_dict[master_id_col], 'tif', raw_index)

        if slave_tif_col is not None and pd.notna(row_dict.get(slave_tif_col)):
            job['slave_tif'] = str(row_dict[slave_tif_col])
        else:
            job['slave_tif'] = _resolve_pair_file(raw_dir, row_dict[slave_id_col], 'tif', raw_index)

        if master_json_col is not None and pd.notna(row_dict.get(master_json_col)):
            job['master_json'] = str(row_dict[master_json_col])
        else:
            job['master_json'] = _resolve_pair_file(raw_dir, row_dict[master_id_col], 'json', raw_index)

        if slave_json_col is not None and pd.notna(row_dict.get(slave_json_col)):
            job['slave_json'] = str(row_dict[slave_json_col])
        else:
            job['slave_json'] = _resolve_pair_file(raw_dir, row_dict[slave_id_col], 'json', raw_index)

        if out_dir_col is not None and pd.notna(row_dict.get(out_dir_col)):
            job['out_dir'] = str(row_dict[out_dir_col])
        else:
            job['out_dir'] = str(batch_out_dir / pair_id)

        jobs.append(job)
    return jobs


def _read_batch_manifest(manifest_path: os.PathLike, batch_out_dir: Optional[os.PathLike]) -> List[Dict[str, str]]:
    manifest_path = Path(manifest_path)
    rows: List[Dict[str, str]] = []
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"master_tif", "slave_tif", "master_json", "slave_json"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Batch manifest is missing columns: {sorted(missing)}")
        for i, row in enumerate(reader):
            item = {k: row[k] for k in required}
            pair_id = row.get("pair_id") or f"pair_{i:05d}"
            if row.get("out_dir"):
                item["out_dir"] = row["out_dir"]
            elif batch_out_dir is not None:
                item["out_dir"] = str(Path(batch_out_dir) / pair_id)
            else:
                raise ValueError("Each batch row needs out_dir, or provide --out-dir as a batch root.")
            item["pair_id"] = pair_id
            rows.append(item)
    return rows


def run_pipeline_batch(manifest_path: os.PathLike, batch_out_dir: Optional[os.PathLike] = None,
                       max_workers: int = 1, manifest_kind: str = "csv",
                       raw_dir: Optional[os.PathLike] = None, max_pairs: Optional[int] = None,
                       **pipeline_kwargs) -> List[Dict[str, object]]:
    if manifest_kind == "parquet":
        if raw_dir is None:
            raise ValueError("--raw_dir is required when using --pairs_manifest parquet mode")
        if batch_out_dir is None:
            raise ValueError("--out-dir is required as a batch root when using --pairs_manifest parquet mode")
        jobs = _read_pairs_parquet_manifest(manifest_path, raw_dir, batch_out_dir)
    else:
        jobs = _read_batch_manifest(manifest_path, batch_out_dir)
    if max_pairs is not None:
        jobs = jobs[:max(0, int(max_pairs))]
    results: List[Dict[str, object]] = []

    def _run_one(job: Dict[str, str]) -> Dict[str, object]:
        try:
            diagnostics = run_pipeline(
                master_tif=job["master_tif"],
                slave_tif=job["slave_tif"],
                master_json=job["master_json"],
                slave_json=job["slave_json"],
                out_dir=job["out_dir"],
                **pipeline_kwargs,
            )
            status = "rejected" if diagnostics.get("pair_rejected", False) else "ok"
            result = {
                "pair_id": job["pair_id"],
                "out_dir": job["out_dir"],
                "status": status,
                "selected_candidate": diagnostics.get("selected_candidate", ""),
                "coherence_mean": diagnostics.get("coherence", {}).get("mean", float("nan")),
                "coherence_median": diagnostics.get("coherence", {}).get("median", float("nan")),
            }
            if status == "rejected":
                result["rejection_reason"] = diagnostics.get("rejection_reason", "")
                result["rejection_threshold_mean"] = diagnostics.get("rejection_threshold_mean", None)
            return result
        except Exception as exc:
            return {
                "pair_id": job["pair_id"],
                "out_dir": job["out_dir"],
                "status": "error",
                "error": repr(exc),
            }

    if max_workers <= 1:
        for job in jobs:
            results.append(_run_one(job))
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_one, job): job for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda x: x["pair_id"])
    return results


def build_argparser():
    p = argparse.ArgumentParser(description="Capella Spotlight PFA SLC co-registration pipeline for InSAR (v5 optimized + batch)")
    p.add_argument("--master-tif")
    p.add_argument("--slave-tif")
    p.add_argument("--master-json")
    p.add_argument("--slave-json")
    p.add_argument("--out-dir")
    p.add_argument("--batch-manifest")
    p.add_argument("--pairs_manifest")
    p.add_argument("--raw_dir")
    p.add_argument("--batch-workers", type=int, default=1)
    p.add_argument("--max-pairs", type=int, default=None,
                   help="Process only the first N pairs from the batch/parquet manifest.")
    p.add_argument("--thumb-looks-row", type=int, default=16)
    p.add_argument("--thumb-looks-col", type=int, default=16)
    p.add_argument("--grid-rows", type=int, default=9)
    p.add_argument("--grid-cols", type=int, default=9)
    p.add_argument("--patch-win", type=int, default=256)
    p.add_argument("--search-radius", type=int, default=48)
    p.add_argument("--pass2-search-radius", type=int, default=8)
    p.add_argument("--local-upsample", type=int, default=20)
    p.add_argument("--tp-workers", type=int, default=1,
                   help="Tie-point worker threads. Use 0 for auto.")
    p.add_argument("--interp", choices=["cubic", "lanczos"], default="lanczos")
    p.add_argument("--commonband-frac", type=float, default=0.92)
    p.add_argument("--commonband-smooth-frac", type=float, default=0.06)
    p.add_argument("--no-residual-pass", action="store_true")
    p.add_argument("--save-candidate-products", action="store_true")
    p.add_argument("--save-png-quicklooks", action="store_true",
                   help="Save PNG quicklooks and tie-point plots. Default is off for speed.")
    p.add_argument("--save-npy-outputs", action="store_true",
                   help="Save NPY/NPZ intermediate arrays. Default is off for speed.")
    p.add_argument("--skip-pass2", action="store_true",
                   help="Skip all pass-2 candidate generation. Faster, but changes candidate-selection logic.")
    p.add_argument("--coherence-backend", choices=["scipy", "opencv"], default="scipy",
                   help="Coherence backend. 'scipy' is the strict/default path; 'opencv' may be faster but is not guaranteed bit-identical.")
    p.add_argument("--goldstein-alpha", type=float, default=0.5)
    p.add_argument("--goldstein-block-size", type=int, default=32)
    p.add_argument("--goldstein-step", type=int, default=None)
    p.add_argument("--phase-cmap", type=str, default="hsv")
    p.add_argument("--min-coherence-mean", type=float, default=None,
                   help="Reject a pair and write no pair outputs if final coherence mean is below this threshold.")
    p.add_argument("--remap-tile", type=int, default=4096)
    p.add_argument("--allow-commonband-selection", action="store_true",
                   help="Retained for CLI compatibility, but the pass-2 common-band branch is skipped in this fast build.")
    p.add_argument("--pass1-peak-threshold", type=float, default=0.08)
    p.add_argument("--pass1-peak-ratio-threshold", type=float, default=1.08)
    p.add_argument("--pass2-peak-threshold", type=float, default=0.05)
    p.add_argument("--pass2-peak-ratio-threshold", type=float, default=1.12)
    return p


def main():
    args = build_argparser().parse_args()
    pipeline_kwargs = _pipeline_kwargs_from_args(args)

    manifest_path = args.pairs_manifest or args.batch_manifest
    manifest_kind = "parquet" if args.pairs_manifest else "csv"

    if manifest_path:
        if args.batch_workers < 1:
            raise ValueError("--batch-workers must be >= 1")
        results = run_pipeline_batch(
            manifest_path=manifest_path,
            batch_out_dir=args.out_dir,
            max_workers=args.batch_workers,
            manifest_kind=manifest_kind,
            raw_dir=args.raw_dir,
            max_pairs=args.max_pairs,
            **pipeline_kwargs,
        )
        summary = {
            "n_pairs": len(results),
            "n_ok": int(sum(r["status"] == "ok" for r in results)),
            "n_rejected": int(sum(r["status"] == "rejected" for r in results)),
            "n_error": int(sum(r["status"] == "error" for r in results)),
            "results": results,
        }
        if args.out_dir:
            ensure_dir(args.out_dir)
            with open(Path(args.out_dir) / "batch_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, cls=NpEncoder)
        print(json.dumps(summary, indent=2, cls=NpEncoder))
        return

    required_single = [args.master_tif, args.slave_tif, args.master_json, args.slave_json, args.out_dir]
    if not all(required_single):
        raise ValueError("Single-run mode requires --master-tif, --slave-tif, --master-json, --slave-json, and --out-dir.")

    diagnostics = run_pipeline(
        master_tif=args.master_tif, slave_tif=args.slave_tif,
        master_json=args.master_json, slave_json=args.slave_json,
        out_dir=args.out_dir,
        **pipeline_kwargs,
    )
    print(json.dumps(diagnostics, indent=2, cls=NpEncoder))


if __name__ == "__main__":
    main()
