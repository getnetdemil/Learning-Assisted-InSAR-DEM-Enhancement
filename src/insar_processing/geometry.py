"""
InSAR geometry utilities: perpendicular baseline, ECEF/geodetic conversions,
and state-vector interpolation from Capella extended JSON metadata.

All position/velocity vectors are ECEF (metres / metres-per-second).
Angles are in radians unless a function name contains '_deg'.

Usage
-----
from src.insar_processing.geometry import load_extended_meta, compute_bperp

meta_ref = load_extended_meta("data/raw/AOI_000/CAPELLA_.../..._extended.json")
meta_sec = load_extended_meta("data/raw/AOI_000/CAPELLA_.../..._extended.json")
bperp = compute_bperp(meta_ref, meta_sec)          # metres, signed
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_extended_meta(json_path: str | Path) -> dict:
    """Load a Capella extended JSON sidecar. Returns the parsed dict."""
    with open(json_path) as f:
        return json.load(f)


def find_extended_json(slc_tif_path: str | Path) -> Path:
    """
    Given a path to a Capella SLC .tif, return the companion *_extended.json.
    Assumes the naming convention: <stem>.tif → <stem>_extended.json
    """
    p = Path(slc_tif_path)
    candidate = p.parent / (p.stem + "_extended.json")
    if candidate.exists():
        return candidate
    # Fallback: search the same directory
    matches = list(p.parent.glob("*_extended.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No extended JSON found for {slc_tif_path}")


# ---------------------------------------------------------------------------
# State-vector interpolation
# ---------------------------------------------------------------------------

def _parse_iso(ts: str) -> datetime:
    """Parse ISO-8601 UTC timestamp (with or without trailing Z)."""
    return datetime.fromisoformat(ts.rstrip("Z")).replace(tzinfo=timezone.utc)


def interpolate_state_vector(
    state_vectors: list[dict],
    target_time: datetime,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolate ECEF position and velocity at target_time.

    Parameters
    ----------
    state_vectors : list[dict]
        Each entry: {"time": ISO str, "position": [x,y,z], "velocity": [vx,vy,vz]}.
    target_time : datetime
        UTC datetime to interpolate to.

    Returns
    -------
    pos : np.ndarray shape (3,)   ECEF position (metres)
    vel : np.ndarray shape (3,)   ECEF velocity (m/s)
    """
    times = np.array([(_parse_iso(sv["time"]) - target_time).total_seconds() for sv in state_vectors])
    positions = np.array([sv["position"] for sv in state_vectors])
    velocities = np.array([sv["velocity"] for sv in state_vectors])

    # Find bracketing indices
    idx = np.searchsorted(times, 0.0)
    idx = int(np.clip(idx, 1, len(times) - 1))
    i0, i1 = idx - 1, idx

    t0, t1 = times[i0], times[i1]
    alpha = -t0 / (t1 - t0)  # fraction in [0,1]

    pos = positions[i0] + alpha * (positions[i1] - positions[i0])
    vel = velocities[i0] + alpha * (velocities[i1] - velocities[i0])
    return pos, vel


# ---------------------------------------------------------------------------
# Perpendicular baseline
# ---------------------------------------------------------------------------

def compute_bperp(meta_ref: dict, meta_sec: dict) -> float:
    """
    Compute the signed perpendicular baseline between two Capella SLC collects.

    Uses the pre-computed `reference_antenna_position` (ECEF, metres) from each
    extended JSON sidecar — no ISCE3 dependency required for a first-order estimate.
    For sub-metre accuracy, use `compute_bperp_interp` which interpolates state vectors.

    Sign convention: positive when the secondary satellite is on the far-range side
    of the reference (i.e., larger slant range).

    Parameters
    ----------
    meta_ref : dict   Reference collect extended JSON (from load_extended_meta).
    meta_sec : dict   Secondary collect extended JSON.

    Returns
    -------
    float   Perpendicular baseline in metres (signed).
    """
    img_ref = meta_ref["collect"]["image"]
    img_sec = meta_sec["collect"]["image"]

    P1 = np.array(img_ref["reference_antenna_position"])   # satellite position, collect 1
    P2 = np.array(img_sec["reference_antenna_position"])   # satellite position, collect 2
    T  = np.array(img_ref["center_pixel"]["target_position"])  # scene centre (ECEF)

    return _bperp_from_positions(P1, P2, T, meta_ref)


def compute_bperp_interp(meta_ref: dict, meta_sec: dict) -> float:
    """
    Compute signed perpendicular baseline using state-vector interpolation.

    More accurate than compute_bperp when the reference_antenna_position field
    is unavailable or when sub-metre precision is needed.
    """
    img_ref = meta_ref["collect"]["image"]
    svs_ref = meta_ref["collect"]["state"]["state_vectors"]
    svs_sec = meta_sec["collect"]["state"]["state_vectors"]

    center_time_ref = _parse_iso(img_ref["center_pixel"]["center_time"])
    center_time_sec = _parse_iso(
        meta_sec["collect"]["image"]["center_pixel"]["center_time"]
    )

    P1, _ = interpolate_state_vector(svs_ref, center_time_ref)
    P2, _ = interpolate_state_vector(svs_sec, center_time_sec)
    T = np.array(img_ref["center_pixel"]["target_position"])

    return _bperp_from_positions(P1, P2, T, meta_ref)


def _bperp_from_positions(
    P1: np.ndarray,
    P2: np.ndarray,
    T: np.ndarray,
    meta_ref: dict,
) -> float:
    """
    Internal: compute signed B_perp given satellite positions P1, P2 and target T.

    Steps
    -----
    1.  B        = P2 - P1                  (baseline vector)
    2.  r1       = (T - P1) / |T - P1|      (unit look vector from P1 to T)
    3.  h1       = V1 / |V1|                (unit along-track from state vectors)
    4.  B_slant  = B - (B·h1)h1             (remove along-track component)
    5.  B_perp   = B_slant · (r1 × h1)     (signed perpendicular baseline)
    """
    # Along-track direction from state vectors at mid-collect time
    svs = meta_ref["collect"]["state"]["state_vectors"]
    center_time = _parse_iso(meta_ref["collect"]["image"]["center_pixel"]["center_time"])
    _, V1 = interpolate_state_vector(svs, center_time)
    h1 = V1 / np.linalg.norm(V1)

    B = P2 - P1
    R1 = T - P1
    r1 = R1 / np.linalg.norm(R1)

    # Remove along-track component to get slant-plane baseline
    B_slant = B - np.dot(B, h1) * h1

    # Perpendicular direction in slant plane (r1 × h1, normalised)
    perp = np.cross(r1, h1)
    norm_perp = np.linalg.norm(perp)
    if norm_perp < 1e-10:
        return 0.0
    perp = perp / norm_perp

    return float(np.dot(B_slant, perp))


# ---------------------------------------------------------------------------
# ECEF / geodetic conversions
# ---------------------------------------------------------------------------

# WGS-84 constants
_WGS84_A = 6_378_137.0          # semi-major axis (m)
_WGS84_F = 1.0 / 298.257223563  # flattening
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)
_WGS84_E2 = 2 * _WGS84_F - _WGS84_F ** 2  # first eccentricity squared


def ecef_to_geodetic(xyz: np.ndarray) -> tuple[float, float, float]:
    """
    Convert ECEF Cartesian to geodetic (lat, lon, height).

    Returns
    -------
    lat_deg : float   geodetic latitude (degrees, +N)
    lon_deg : float   longitude (degrees, +E)
    h_m     : float   ellipsoidal height (metres)
    """
    x, y, z = xyz
    lon = np.degrees(np.arctan2(y, x))
    p = np.sqrt(x ** 2 + y ** 2)

    # Bowring iterative method
    lat = np.arctan2(z, p * (1.0 - _WGS84_E2))
    for _ in range(5):
        sin_lat = np.sin(lat)
        N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)
        lat = np.arctan2(z + _WGS84_E2 * N * sin_lat, p)

    sin_lat = np.sin(lat)
    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)
    h = p / np.cos(lat) - N if abs(np.cos(lat)) > 1e-10 else abs(z) / sin_lat - N * (1.0 - _WGS84_E2)

    return float(np.degrees(lat)), float(lon), float(h)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float = 0.0) -> np.ndarray:
    """Convert geodetic (lat, lon, height) to ECEF Cartesian (metres)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)
    return np.array([
        (N + h_m) * cos_lat * np.cos(lon),
        (N + h_m) * cos_lat * np.sin(lon),
        (N * (1.0 - _WGS84_E2) + h_m) * sin_lat,
    ])


# ---------------------------------------------------------------------------
# Geometry metadata extractor (for pair_graph enrichment)
# ---------------------------------------------------------------------------

def extract_geometry(meta: dict) -> dict:
    """
    Extract key geometry scalars from an extended JSON dict.

    Returns a flat dict suitable for merging into a pair-edge DataFrame:
        antenna_pos_ecef  : [x, y, z]  satellite position at collect centre
        target_pos_ecef   : [x, y, z]  scene centre on ground (ECEF)
        center_time       : str         ISO timestamp
        incidence_deg     : float       centre-pixel incidence angle
        look_angle_deg    : float       centre-pixel look angle
        slant_range_m     : float       slant range to scene centre
    """
    img = meta["collect"]["image"]
    cp = img["center_pixel"]
    P = np.array(img["reference_antenna_position"])
    T = np.array(cp["target_position"])

    return {
        "antenna_pos_ecef": P.tolist(),
        "target_pos_ecef": T.tolist(),
        "center_time": cp["center_time"],
        "incidence_deg": cp["incidence_angle"],
        "look_angle_deg": cp["look_angle"],
        "slant_range_m": float(np.linalg.norm(T - P)),
    }
