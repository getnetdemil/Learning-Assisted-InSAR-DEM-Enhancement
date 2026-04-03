
"""
Pair-graph construction and quality scoring for Capella InSAR stack processing.

This module turns a manifest DataFrame into a filtered list of candidate acquisition
pairs. The original version only required matching orbit_state and look_direction,
then ranked pairs with a simple score based on temporal and incidence-angle
differences. That is useful for exploratory pairing, but it is too permissive for a
classical repeat-pass InSAR pipeline.

This updated version keeps the simple interface while making the pairing logic much
stricter and more explicit:

1. Normalise column names from either the project's internal manifest style or the
   raw Capella contest CSV.
2. Require consistent acquisition geometry by default:
   - same AOI label (when available)
   - same orbit_state
   - same look_direction / observation_direction
   - same orbital plane
   - same platform
   - same instrument mode
3. Optionally require same collection type / product type / polarisation when those
   fields exist in the manifest. These gates are disabled by default because the
   project's `full_index.parquet` does not currently carry those fields.
4. Reject pairs whose incidence / look / squint geometry differs too much.
5. Optionally reject pairs whose range / azimuth sampling differs too much.
6. Keep triplet enumeration, because closure triplets are still derived from the
   accepted pair graph.

The goal is not to build a full physical coherence model here. The goal is to reject
obviously poor or weakly compatible pairs before downstream classical processing.

# Updated pair selection logic for stricter InSAR compatibility.
# Main improvements over the earlier version:
# 1. Added manifest-aware column aliasing so the code works directly with
#    full_index.parquet fields such as look_angle_deg, squint_angle_deg,
#    px_spacing_rg_m, and px_spacing_az_m.
# 2. Tightened hard compatibility checks to prefer physically consistent pairs,
#    including same orbit state, look direction, orbital plane, platform,
#    instrument mode, and AOI.
# 3. Added optional geometry-consistency filters based on incidence-angle,
#    look-angle, squint-angle, and range/azimuth pixel-spacing differences.
# 4. Improved pair ranking so quality is not based only on temporal baseline
#    and incidence mismatch, but also penalizes additional geometry mismatch.
# 5. Preserved triplet enumeration for closure-network construction while
#    producing a cleaner pair graph for classical InSAR preprocessing.
#
# Practical effect:
# - The old graph was more permissive and produced many more candidate pairs,
#   but included weaker or less consistent geometries.
# - The updated graph is stricter and produces fewer, cleaner pairs that are
#   better suited for pairwise InSAR and DEM-oriented processing.
# - In practice, this stricter logic improves pair quality, but may reduce
#   graph connectivity, so separate "strict" and "stack" pairing modes may be
#   useful for different downstream tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Column normalisation helpers
# -----------------------------------------------------------------------------
# The project uses more than one manifest style:
# - internal manifests often use names such as:
#       id, look_direction, incidence_angle_deg
# - the Capella contest CSV uses names such as:
#       collect_id, observation_direction, incidence_angle
#
# To keep pair_graph.py robust, we map both styles into one canonical schema.
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "collect_id", "stac_id"),
    "datetime": ("datetime", "start_datetime"),
    "orbit_state": ("orbit_state",),
    "look_direction": ("look_direction", "observation_direction"),
    "incidence_angle_deg": ("incidence_angle_deg", "incidence_angle"),
    "orbital_plane": ("orbital_plane",),
    "platform": ("platform",),
    "instrument_mode": ("instrument_mode",),
    "collection_type": ("collection_type",),
    "product_type": ("product_type",),
    "polarization": ("polarization", "polarizations"),
    "look_angle": ("look_angle", "look_angle_deg"),
    "squint_angle": ("squint_angle", "squint_angle_deg"),
    "resolution_range": ("resolution_range", "px_spacing_rg_m"),
    "resolution_azimuth": ("resolution_azimuth", "px_spacing_az_m"),
    "aoi": ("aoi",),
    "bbox_w": ("bbox_w",),
    "bbox_s": ("bbox_s",),
    "bbox_e": ("bbox_e",),
    "bbox_n": ("bbox_n",),
}


def _first_present(columns: Iterable[str], aliases: tuple[str, ...]) -> str | None:
    """Return the first alias that exists in *columns*, otherwise None."""
    for name in aliases:
        if name in columns:
            return name
    return None


def _normalise_string_series(s: pd.Series) -> pd.Series:
    """
    Normalise string-like metadata fields.

    We strip whitespace and lower-case values so that comparisons such as
    'Descending' vs 'descending' do not create false incompatibilities.
    """
    out = s.astype("string").str.strip().str.lower()
    return out.replace({"<na>": pd.NA, "nan": pd.NA, "none": pd.NA})


def _same_required(a, b) -> bool:
    """
    Strict equality test for required metadata fields.

    If either side is missing, return False. This makes the pairing conservative:
    incomplete metadata does not pass a required compatibility gate.
    """
    if pd.isna(a) or pd.isna(b):
        return False
    return a == b


def _allowed_value(v, allowed: set[str]) -> bool:
    """Return True only when a non-missing value belongs to the allowed set."""
    if pd.isna(v):
        return False
    return str(v).lower() in allowed


def _prepare_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the manifest with a canonical set of columns.

    The output DataFrame always provides these canonical names when available:
    id, datetime, orbit_state, look_direction, incidence_angle_deg, orbital_plane,
    platform, instrument_mode, collection_type, product_type, polarization,
    look_angle, squint_angle, resolution_range, resolution_azimuth, aoi, bbox_*

    Missing optional fields are created and filled with NaN/NA. Missing required
    fields raise a clear ValueError.
    """
    out = df.copy()

    rename_map: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        src = _first_present(out.columns, aliases)
        if src is not None and src != canonical:
            rename_map[src] = canonical

    out = out.rename(columns=rename_map)

    required = ["id", "datetime", "orbit_state", "look_direction", "incidence_angle_deg"]
    missing_required = [c for c in required if c not in out.columns]
    if missing_required:
        raise ValueError(
            "Manifest is missing required columns after normalisation: "
            f"{missing_required}. Available columns: {list(df.columns)}"
        )

    # Create optional columns if absent so later code can use the same field names
    # without constantly checking for column existence.
    optional_cols = [
        "orbital_plane",
        "platform",
        "instrument_mode",
        "collection_type",
        "product_type",
        "polarization",
        "look_angle",
        "squint_angle",
        "resolution_range",
        "resolution_azimuth",
        "aoi",
        "bbox_w",
        "bbox_s",
        "bbox_e",
        "bbox_n",
    ]
    for c in optional_cols:
        if c not in out.columns:
            out[c] = pd.NA

    # Datetime parsing.
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
    if out["datetime"].isna().any():
        bad = out[out["datetime"].isna()]["id"].tolist()[:5]
        raise ValueError(
            "Some datetime values could not be parsed after normalisation. "
            f"Example ids: {bad}"
        )

    # String-normalise categorical fields used for compatibility checks.
    for c in [
        "orbit_state",
        "look_direction",
        "platform",
        "instrument_mode",
        "collection_type",
        "product_type",
        "polarization",
        "aoi",
    ]:
        out[c] = _normalise_string_series(out[c])

    # Numeric coercion for geometric fields.
    for c in [
        "incidence_angle_deg",
        "look_angle",
        "squint_angle",
        "resolution_range",
        "resolution_azimuth",
        "orbital_plane",
        "bbox_w",
        "bbox_s",
        "bbox_e",
        "bbox_n",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def _relative_diff(a: float | int | None, b: float | int | None) -> float | None:
    """
    Return the relative difference |a-b| / max(|a|, |b|), or None if unavailable.
    """
    if pd.isna(a) or pd.isna(b):
        return None
    denom = max(abs(float(a)), abs(float(b)), 1e-12)
    return abs(float(a) - float(b)) / denom


def _bbox_overlap_fraction(r: dict, s: dict) -> float | None:
    """
    Return a simple footprint-overlap proxy based on axis-aligned lon/lat boxes.

    The result is intersection_area / min(area_ref, area_sec), which is a practical
    screening metric for "do these two products mostly observe the same area?".
    This is not geodetically exact, but it is useful as a lightweight compatibility
    gate before heavier downstream processing.
    """
    needed = ["bbox_w", "bbox_s", "bbox_e", "bbox_n"]
    if any(pd.isna(r.get(k)) or pd.isna(s.get(k)) for k in needed):
        return None

    iw = max(float(r["bbox_w"]), float(s["bbox_w"]))
    ie = min(float(r["bbox_e"]), float(s["bbox_e"]))
    is_ = max(float(r["bbox_s"]), float(s["bbox_s"]))
    in_ = min(float(r["bbox_n"]), float(s["bbox_n"]))

    if ie <= iw or in_ <= is_:
        return 0.0

    inter = (ie - iw) * (in_ - is_)
    area_r = max((float(r["bbox_e"]) - float(r["bbox_w"])) * (float(r["bbox_n"]) - float(r["bbox_s"])), 0.0)
    area_s = max((float(s["bbox_e"]) - float(s["bbox_w"])) * (float(s["bbox_n"]) - float(s["bbox_s"])), 0.0)

    if area_r <= 0.0 or area_s <= 0.0:
        return None

    return inter / min(area_r, area_s)


def _score_pair(
    dt_days: float,
    dinc_deg: float,
    dlook_deg: float | None,
    dsquint_deg: float | None,
    rg_res_rel: float | None,
    az_res_rel: float | None,
    overlap_frac: float | None,
) -> float:
    """
    Compute a soft pair quality score.

    The score is still heuristic, but it now penalises more than just temporal
    baseline and incidence angle:
        - temporal separation
        - incidence mismatch
        - look-angle mismatch (if available)
        - squint mismatch (if available)
        - resolution mismatch (if available)
        - low footprint overlap (if available)

    Lower mismatch -> higher score.
    """
    q = (1.0 / (dt_days + 1.0)) * (1.0 / (1.0 + dinc_deg))

    if dlook_deg is not None:
        q *= 1.0 / (1.0 + 0.5 * dlook_deg)
    if dsquint_deg is not None:
        q *= 1.0 / (1.0 + 0.25 * dsquint_deg)
    if rg_res_rel is not None:
        q *= 1.0 / (1.0 + 2.0 * rg_res_rel)
    if az_res_rel is not None:
        q *= 1.0 / (1.0 + 2.0 * az_res_rel)
    if overlap_frac is not None:
        # reward higher overlap while keeping the score bounded in [0, q]
        q *= max(0.0, min(1.0, float(overlap_frac)))

    return float(q)


@dataclass
class PairGraphConfig:
    # ------------------------------------------------------------------
    # Time / geometry filters
    # ------------------------------------------------------------------
    dt_max_days: float = 365.0            # maximum temporal baseline (days)
    dt_min_days: float = 0.0              # optional minimum temporal baseline (days)
    dinc_max_deg: float = 5.0             # maximum incidence-angle difference (deg)
    dlook_max_deg: float | None = 2.5     # maximum look-angle difference (deg)
    dsquint_max_deg: float | None = 5.0   # maximum squint-angle difference (deg)

    # ------------------------------------------------------------------
    # Strict compatibility switches
    # ------------------------------------------------------------------
    require_same_aoi: bool = True               # do not pair across AOI labels
    require_same_orbit: bool = True             # ascending/descending must match
    require_same_look: bool = True              # left/right must match
    require_same_orbital_plane: bool = True     # track family / orbital plane must match
    require_same_platform: bool = True          # same Capella satellite preferred
    require_same_mode: bool = True              # same instrument_mode
    require_same_collection_type: bool = False  # disabled by default: absent in full_index.parquet
    require_same_product_type: bool = False     # disabled by default: absent in full_index.parquet
    require_same_polarization: bool = False     # disabled by default: absent in full_index.parquet

    # ------------------------------------------------------------------
    # Optional product-type gate
    # ------------------------------------------------------------------
    # Disabled by default because the project's full_index.parquet does not carry
    # product_type. If a richer manifest is used later, set for example ("slc",).
    allowed_product_types: tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Optional resolution / overlap consistency gates
    # ------------------------------------------------------------------
    max_range_res_rel_diff: float | None = 0.15
    max_azimuth_res_rel_diff: float | None = 0.15
    min_bbox_overlap_frac: float | None = 0.50  # require at least 50% overlap of the smaller footprint

    # ------------------------------------------------------------------
    # Score threshold
    # ------------------------------------------------------------------
    min_q_score: float = 0.0


def build_pair_graph(df: pd.DataFrame, cfg: PairGraphConfig | None = None) -> pd.DataFrame:
    """
    Build all valid interferometric pairs from a manifest DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Manifest DataFrame. The function accepts either the project's internal
        manifest style or the raw Capella CSV style. Common fields are normalised
        automatically.
    cfg : PairGraphConfig
        Pairing thresholds and compatibility switches. Uses stricter defaults if None.

    Returns
    -------
    pd.DataFrame
        Edge list sorted by q_score descending. The output includes both the pair
        identifiers and the metadata that explains why the pair passed.
    """
    if cfg is None:
        cfg = PairGraphConfig()

    df = _prepare_manifest(df).reset_index(drop=True)

    # Convert datetime to "days since epoch" for simple and fast temporal differences.
    df["_t_days"] = df["datetime"].astype("int64") / 1e9 / 86400.0

    records = []
    rows = df.to_dict("records")
    n = len(rows)
    allowed_types = {x.lower() for x in cfg.allowed_product_types}

    for i in range(n):
        r = rows[i]
        for j in range(i + 1, n):
            s = rows[j]

            # --------------------------------------------------------------
            # Hard compatibility filters.
            # These are intentionally conservative. It is better to reject a
            # dubious pair here than to waste preprocessing time on a pair that
            # is geometrically weak for interferometry.
            # --------------------------------------------------------------
            if cfg.require_same_aoi and not _same_required(r["aoi"], s["aoi"]):
                continue

            if cfg.require_same_product_type and not _same_required(r["product_type"], s["product_type"]):
                continue

            if allowed_types:
                if not _allowed_value(r["product_type"], allowed_types):
                    continue
                if not _allowed_value(s["product_type"], allowed_types):
                    continue

            if cfg.require_same_orbit and not _same_required(r["orbit_state"], s["orbit_state"]):
                continue
            if cfg.require_same_look and not _same_required(r["look_direction"], s["look_direction"]):
                continue
            if cfg.require_same_orbital_plane and not _same_required(r["orbital_plane"], s["orbital_plane"]):
                continue
            if cfg.require_same_platform and not _same_required(r["platform"], s["platform"]):
                continue
            if cfg.require_same_mode and not _same_required(r["instrument_mode"], s["instrument_mode"]):
                continue
            if cfg.require_same_collection_type and not _same_required(r["collection_type"], s["collection_type"]):
                continue
            if cfg.require_same_polarization and not _same_required(r["polarization"], s["polarization"]):
                continue

            # --------------------------------------------------------------
            # Temporal baseline filter.
            # --------------------------------------------------------------
            dt = abs(r["_t_days"] - s["_t_days"])
            if dt < cfg.dt_min_days or dt > cfg.dt_max_days:
                continue

            # --------------------------------------------------------------
            # Geometry consistency filters.
            # --------------------------------------------------------------
            dinc = abs(r["incidence_angle_deg"] - s["incidence_angle_deg"])
            if dinc > cfg.dinc_max_deg:
                continue

            dlook = None
            if not pd.isna(r["look_angle"]) and not pd.isna(s["look_angle"]):
                dlook = abs(float(r["look_angle"]) - float(s["look_angle"]))
                if cfg.dlook_max_deg is not None and dlook > cfg.dlook_max_deg:
                    continue

            dsquint = None
            if not pd.isna(r["squint_angle"]) and not pd.isna(s["squint_angle"]):
                dsquint = abs(float(r["squint_angle"]) - float(s["squint_angle"]))
                if cfg.dsquint_max_deg is not None and dsquint > cfg.dsquint_max_deg:
                    continue

            # --------------------------------------------------------------
            # Resolution consistency filters.
            # Use resolution_range (aliased from px_spacing_rg_m for the current
            # manifest) rather than the non-existent resolution_ground_range field.
            # --------------------------------------------------------------
            rg_res_rel = _relative_diff(r["resolution_range"], s["resolution_range"])
            if (
                cfg.max_range_res_rel_diff is not None
                and rg_res_rel is not None
                and rg_res_rel > cfg.max_range_res_rel_diff
            ):
                continue

            az_res_rel = _relative_diff(r["resolution_azimuth"], s["resolution_azimuth"])
            if (
                cfg.max_azimuth_res_rel_diff is not None
                and az_res_rel is not None
                and az_res_rel > cfg.max_azimuth_res_rel_diff
            ):
                continue

            # --------------------------------------------------------------
            # Lightweight footprint-overlap gate from manifest bounding boxes.
            # This helps reject same-AOI products whose actual footprints overlap
            # poorly or not at all.
            # --------------------------------------------------------------
            overlap_frac = _bbox_overlap_fraction(r, s)
            if (
                cfg.min_bbox_overlap_frac is not None
                and overlap_frac is not None
                and overlap_frac < cfg.min_bbox_overlap_frac
            ):
                continue

            # --------------------------------------------------------------
            # Soft ranking score.
            # --------------------------------------------------------------
            q = _score_pair(
                dt_days=dt,
                dinc_deg=dinc,
                dlook_deg=dlook,
                dsquint_deg=dsquint,
                rg_res_rel=rg_res_rel,
                az_res_rel=az_res_rel,
                overlap_frac=overlap_frac,
            )
            if q < cfg.min_q_score:
                continue

            records.append({
                "id_ref": r["id"],
                "id_sec": s["id"],
                "datetime_ref": r["datetime"],
                "datetime_sec": s["datetime"],
                "dt_days": round(dt, 6),
                "dinc_deg": round(dinc, 6),
                "dlook_deg": None if dlook is None else round(dlook, 6),
                "dsquint_deg": None if dsquint is None else round(dsquint, 6),
                "range_res_rel_diff": None if rg_res_rel is None else round(rg_res_rel, 6),
                "azimuth_res_rel_diff": None if az_res_rel is None else round(az_res_rel, 6),
                "bbox_overlap_frac": None if overlap_frac is None else round(overlap_frac, 6),
                "orbit_state": r["orbit_state"],
                "look_direction": r["look_direction"],
                "orbital_plane_ref": r.get("orbital_plane"),
                "orbital_plane_sec": s.get("orbital_plane"),
                "platform_ref": r.get("platform"),
                "platform_sec": s.get("platform"),
                "instrument_mode_ref": r.get("instrument_mode"),
                "instrument_mode_sec": s.get("instrument_mode"),
                "collection_type_ref": r.get("collection_type"),
                "collection_type_sec": s.get("collection_type"),
                "product_type_ref": r.get("product_type"),
                "product_type_sec": s.get("product_type"),
                "polarization_ref": r.get("polarization"),
                "polarization_sec": s.get("polarization"),
                "incidence_ref": r["incidence_angle_deg"],
                "incidence_sec": s["incidence_angle_deg"],
                "look_angle_ref": r.get("look_angle"),
                "look_angle_sec": s.get("look_angle"),
                "squint_angle_ref": r.get("squint_angle"),
                "squint_angle_sec": s.get("squint_angle"),
                "q_score": round(q, 8),
                "aoi": r.get("aoi"),
            })

    if not records:
        return pd.DataFrame(columns=[
            "id_ref", "id_sec", "datetime_ref", "datetime_sec", "dt_days",
            "dinc_deg", "dlook_deg", "dsquint_deg", "range_res_rel_diff",
            "azimuth_res_rel_diff", "bbox_overlap_frac", "orbit_state", "look_direction",
            "orbital_plane_ref", "orbital_plane_sec",
            "platform_ref", "platform_sec",
            "instrument_mode_ref", "instrument_mode_sec",
            "collection_type_ref", "collection_type_sec",
            "product_type_ref", "product_type_sec",
            "polarization_ref", "polarization_sec",
            "incidence_ref", "incidence_sec",
            "look_angle_ref", "look_angle_sec",
            "squint_angle_ref", "squint_angle_sec",
            "q_score", "aoi",
        ])

    edges = pd.DataFrame(records).sort_values(
        ["q_score", "dt_days", "dinc_deg"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return edges


def select_top_pairs(
    edges: pd.DataFrame,
    max_pairs: int,
    strategy: str = "q_score",
) -> pd.DataFrame:
    """
    Select a subset of pairs for processing.

    Parameters
    ----------
    edges : pd.DataFrame
        Output of build_pair_graph.
    max_pairs : int
        Maximum number of pairs to return.
    strategy : str
        "q_score"           — top-N by quality score (default)
        "temporal_coverage" — greedy coverage of unique acquisitions

    Returns
    -------
    pd.DataFrame
        Filtered subset of edges.
    """
    if strategy == "q_score":
        return edges.head(max_pairs).reset_index(drop=True)

    if strategy == "temporal_coverage":
        selected = []
        covered = set()

        # Greedy coverage heuristic:
        # prefer edges that introduce new collects, while still respecting the
        # original q_score ordering.
        for _, row in edges.iterrows():
            if len(selected) >= max_pairs:
                break
            gain = int(row["id_ref"] not in covered) + int(row["id_sec"] not in covered)
            if gain > 0 or len(selected) < max_pairs:
                selected.append(row)
                covered.add(row["id_ref"])
                covered.add(row["id_sec"])

        return pd.DataFrame(selected).reset_index(drop=True)

    raise ValueError(f"Unknown strategy: {strategy!r}. Use 'q_score' or 'temporal_coverage'.")


def find_triplets(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Enumerate all closed triplets (A, B, C) from the accepted edge list.

    A triplet is valid when all three pairwise edges exist:
        A-B, B-C, and A-C

    Important:
    This function does not compute the closure phase error itself. It only
    computes the triplet topology needed later by the closure-metric code.
    """
    if edges.empty:
        return pd.DataFrame(columns=["id_a", "id_b", "id_c"])

    # Build adjacency set for O(1) lookup.
    edge_set = set(zip(edges["id_ref"], edges["id_sec"]))
    # The stored graph is undirected for pairing purposes, so also add the reverse
    # orientation to simplify neighbour queries.
    edge_set |= {(s, r) for r, s in edge_set}

    nodes = list(set(edges["id_ref"]) | set(edges["id_sec"]))
    node_idx = {n: i for i, n in enumerate(nodes)}

    neighbours: dict[str, list[str]] = {n: [] for n in nodes}
    for _, row in edges.iterrows():
        neighbours[row["id_ref"]].append(row["id_sec"])
        neighbours[row["id_sec"]].append(row["id_ref"])

    triplets = []
    visited = set()

    for a in nodes:
        for b in neighbours[a]:
            if node_idx[b] <= node_idx[a]:
                continue
            for c in neighbours[b]:
                if node_idx[c] <= node_idx[b]:
                    continue
                if (a, c) in edge_set:
                    key = tuple(sorted([a, b, c]))
                    if key not in visited:
                        visited.add(key)
                        triplets.append({"id_a": key[0], "id_b": key[1], "id_c": key[2]})

    return pd.DataFrame(triplets)


def summarize_graph(edges: pd.DataFrame) -> dict:
    """Return summary statistics for the pair graph."""
    if edges.empty:
        return {"n_pairs": 0, "n_unique_collects": 0}

    summary = {
        "n_pairs": len(edges),
        "n_unique_collects": len(set(edges["id_ref"]) | set(edges["id_sec"])),
        "dt_min_days": float(edges["dt_days"].min()),
        "dt_median_days": float(edges["dt_days"].median()),
        "dt_max_days": float(edges["dt_days"].max()),
        "dinc_median_deg": float(edges["dinc_deg"].median()),
        "q_min": float(edges["q_score"].min()),
        "q_median": float(edges["q_score"].median()),
        "q_max": float(edges["q_score"].max()),
        "orbits": sorted(edges["orbit_state"].dropna().unique().tolist()),
        "look_dirs": sorted(edges["look_direction"].dropna().unique().tolist()),
    }

    for c in [
        "dlook_deg",
        "dsquint_deg",
        "range_res_rel_diff",
        "azimuth_res_rel_diff",
        "bbox_overlap_frac",
    ]:
        if c in edges.columns and edges[c].notna().any():
            summary[f"{c}_median"] = float(edges[c].dropna().median())

    return summary
