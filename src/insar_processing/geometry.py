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
    """
    Parse ISO-8601 UTC timestamp (with or without trailing Z).

    Capella timestamps use nanosecond precision (9 decimal places).
    Python 3.10 fromisoformat only handles up to microseconds (6 places),
    so we truncate the fractional seconds to 6 digits before parsing.
    """
    ts = ts.rstrip("Z")
    # Truncate sub-microsecond digits if present (e.g. .778268633 → .778268)
    if "." in ts:
        base, frac = ts.split(".", 1)
        ts = base + "." + frac[:6]
    return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)


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

    Uses `reference_antenna_position` when available (newer product versions);
    falls back to state-vector interpolation at center_time otherwise.

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

    # Prefer pre-computed antenna position; fall back to state-vector interpolation
    if "reference_antenna_position" in img_ref and "reference_antenna_position" in img_sec:
        P1 = np.array(img_ref["reference_antenna_position"])
        P2 = np.array(img_sec["reference_antenna_position"])
    else:
        svs_ref = meta_ref["collect"]["state"]["state_vectors"]
        svs_sec = meta_sec["collect"]["state"]["state_vectors"]
        ct_ref = _parse_iso(img_ref["center_pixel"]["center_time"])
        ct_sec = _parse_iso(img_sec["center_pixel"]["center_time"])
        P1, _ = interpolate_state_vector(svs_ref, ct_ref)
        P2, _ = interpolate_state_vector(svs_sec, ct_sec)

    T = np.array(img_ref["center_pixel"]["target_position"])
    return _bperp_from_positions(P1, P2, T, meta_ref)


def compute_bperp_interp(meta_ref: dict, meta_sec: dict) -> float:
    """
    Compute signed perpendicular baseline using state-vector interpolation (always).

    Useful for validation or when reference_antenna_position is absent.
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
# Range-Doppler geocoding
# ---------------------------------------------------------------------------

def _rd_sphere_intersect(
    sat_pos: np.ndarray,
    sat_vel: np.ndarray,
    slant_range_m: float,
    terrain_height_m: float = 0.0,
    look_direction: str = "right",
) -> "np.ndarray | None":
    """
    Spherical-Earth Range-Doppler sphere intersection (zero-Doppler assumption).
    Accurate to ~100 m for a 10 km patch.

    Decompose target P = alpha*u_n + beta*u_cross where:
      u_n     = S/|S|  (Earth-centre → satellite unit vector)
      u_cross = cross-track right unit vector
      alpha   = (|S|² + Re² - R²) / (2|S|)
      beta    = sqrt(Re² - alpha²)   (positive = right-looking)
    """
    S, V, R = sat_pos, sat_vel, slant_range_m
    Re = _WGS84_A + terrain_height_m

    S_mag = np.linalg.norm(S)
    u_n     = S / S_mag
    u_nadir = -u_n
    u_along = V / np.linalg.norm(V)
    u_cross = np.cross(u_nadir, u_along)
    norm_c  = np.linalg.norm(u_cross)
    if norm_c < 1e-10:
        return None
    u_cross /= norm_c
    if look_direction == "left":
        u_cross = -u_cross

    alpha   = (S_mag**2 + Re**2 - R**2) / (2.0 * S_mag)
    beta_sq = Re**2 - alpha**2
    if beta_sq < 0:
        return None
    return alpha * u_n + np.sqrt(beta_sq) * u_cross


def geocode_patch_corners(
    extended_meta: dict,
    patch_row: int,
    patch_col: int,
    patch_size: int = 4096,
    terrain_height_m: float = 0.0,
) -> np.ndarray:
    """
    Return (lat, lon) for the 4 corners of a patch: TL, TR, BL, BR.
    Shape (4, 2).  Uses slant_plane timing when available; falls back to
    affine pixel-spacing approx for PFA geometry (null timing fields).

    Parameters
    ----------
    extended_meta    : dict   Parsed Capella extended JSON (from load_extended_meta).
    patch_row        : int    Top-left row of patch in the full SLC image.
    patch_col        : int    Top-left column of patch in the full SLC image.
    patch_size       : int    Patch width/height in pixels (square).
    terrain_height_m : float  Median terrain height above WGS-84 for flat-Earth correction.
    """
    import datetime as _dt

    img  = extended_meta["collect"]["image"]
    geom = img["image_geometry"]
    cp   = img["center_pixel"]
    svs  = extended_meta["collect"]["state"]["state_vectors"]

    corners_rc = [
        (patch_row,              patch_col),
        (patch_row,              patch_col + patch_size - 1),
        (patch_row + patch_size - 1, patch_col),
        (patch_row + patch_size - 1, patch_col + patch_size - 1),
    ]

    # ── slant_plane: use first_line_time + delta_line_time ────────────────
    if geom["type"] == "slant_plane" and geom.get("first_line_time") is not None:
        t0 = _parse_iso(geom["first_line_time"])
        dt = geom["delta_line_time"]           # seconds per row
        R0 = geom["range_to_first_sample"]     # metres
        dR = geom["delta_range_sample"]        # metres per column

        # Auto-detect look sign from center_pixel.target_position so we work
        # for both right-looking and left-looking collects.
        rows_total = img["rows"]
        cols_total = img["columns"]
        t_ctr = t0 + _dt.timedelta(seconds=rows_total / 2.0 * dt)
        R_ctr = R0 + cols_total / 2.0 * dR
        pos_ctr, vel_ctr = interpolate_state_vector(svs, t_ctr)
        S_mag = np.linalg.norm(pos_ctr)
        u_n_c = pos_ctr / S_mag
        u_cross_c = np.cross(-u_n_c, vel_ctr / np.linalg.norm(vel_ctr))
        norm_c = np.linalg.norm(u_cross_c)
        if norm_c > 1e-10:
            u_cross_c /= norm_c
        Re_c  = _WGS84_A + terrain_height_m
        alpha_c = (S_mag**2 + Re_c**2 - R_ctr**2) / (2.0 * S_mag)
        beta_c  = np.sqrt(max(0.0, Re_c**2 - alpha_c**2))
        T_known = np.array(cp["target_position"])
        P_plus  = alpha_c * u_n_c + beta_c  * u_cross_c
        P_minus = alpha_c * u_n_c - beta_c  * u_cross_c
        look_sign = 1.0 if (np.linalg.norm(P_plus - T_known) <=
                            np.linalg.norm(P_minus - T_known)) else -1.0

        out = []
        for (r, c) in corners_rc:
            t_az = t0 + _dt.timedelta(seconds=float(r) * dt)
            R    = R0 + float(c) * dR
            pos, vel = interpolate_state_vector(svs, t_az)
            S_mag2  = np.linalg.norm(pos)
            u_n2    = pos / S_mag2
            u_cross2 = np.cross(-u_n2, vel / np.linalg.norm(vel))
            norm2   = np.linalg.norm(u_cross2)
            if norm2 < 1e-10:
                u_cross2 = u_cross_c
            else:
                u_cross2 /= norm2
            Re2    = _WGS84_A + terrain_height_m
            alpha2 = (S_mag2**2 + Re2**2 - R**2) / (2.0 * S_mag2)
            beta2_sq = Re2**2 - alpha2**2
            if beta2_sq < 0:
                T = T_known
            else:
                P = alpha2 * u_n2 + look_sign * np.sqrt(beta2_sq) * u_cross2
                T = P
            lat, lon, _ = ecef_to_geodetic(T)
            out.append([lat, lon])
        return np.array(out)

    # ── PFA fallback: affine approx from center_pixel ─────────────────────
    T = np.array(cp["target_position"])
    ctr_lat, ctr_lon, _ = ecef_to_geodetic(T)
    rows_total = img["rows"]
    cols_total = img["columns"]
    px_az = img["pixel_spacing_row"]
    px_rg = img["pixel_spacing_column"]
    inc   = np.radians(cp["incidence_angle"])
    px_gr = px_rg / np.sin(inc)   # slant-range → ground-range pixel size
    Re    = _WGS84_A

    out = []
    for (r, c) in corners_rc:
        dr   = r - rows_total / 2.0
        dc   = c - cols_total / 2.0
        dlat = np.degrees(dr * px_az / Re)
        dlon = np.degrees(dc * px_gr / (Re * np.cos(np.radians(ctr_lat))))
        out.append([ctr_lat + dlat, ctr_lon + dlon])
    return np.array(out)


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
    T = np.array(cp["target_position"])

    # Satellite position: prefer pre-computed field, else interpolate
    if "reference_antenna_position" in img:
        P = np.array(img["reference_antenna_position"])
    else:
        svs = meta["collect"]["state"]["state_vectors"]
        ct = _parse_iso(cp["center_time"])
        P, _ = interpolate_state_vector(svs, ct)

    return {
        "antenna_pos_ecef": P.tolist(),
        "target_pos_ecef": T.tolist(),
        "center_time": cp["center_time"],
        "incidence_deg": cp["incidence_angle"],
        "look_angle_deg": cp["look_angle"],
        "slant_range_m": float(np.linalg.norm(T - P)),
    }
