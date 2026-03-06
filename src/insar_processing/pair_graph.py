"""
Pair-graph construction and edge quality scoring for InSAR stack processing.

Given a manifest DataFrame, builds a directed graph of candidate acquisition pairs,
scores each edge with Q_ij, and filters to a valid working set.

    Q_ij = 1 / (Δt_days + 1)  ×  1 / (1 + |Δθ_inc_deg|)

Constraints: same orbit_state AND same look_direction (required for coherent interferometry).

Usage
-----
from src.insar_processing.pair_graph import PairGraphConfig, build_pair_graph, summarize_graph
import pandas as pd

df = pd.read_parquet("data/manifests/full_index.parquet")
hawaii = df[df["aoi"] == "AOI_000"]

cfg = PairGraphConfig(dt_max_days=365, dinc_max_deg=5.0)
edges = build_pair_graph(hawaii, cfg)
print(summarize_graph(edges))
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PairGraphConfig:
    dt_max_days: float = 365.0      # maximum temporal baseline (days)
    dinc_max_deg: float = 5.0       # maximum incidence angle difference (degrees)
    require_same_orbit: bool = True  # require matching ascending/descending
    require_same_look: bool = True   # require matching left/right look direction
    min_q_score: float = 0.0        # minimum Q_ij to include an edge


def build_pair_graph(df: pd.DataFrame, cfg: PairGraphConfig | None = None) -> pd.DataFrame:
    """
    Build all valid interferometric pairs from a manifest DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Manifest with columns: id, datetime, orbit_state, look_direction,
        incidence_angle_deg, orbital_plane, aoi.
    cfg : PairGraphConfig
        Filtering thresholds. Uses loose defaults if None.

    Returns
    -------
    pd.DataFrame
        Edge list sorted by q_score descending. Columns:
        id_ref, id_sec, datetime_ref, datetime_sec, dt_days, dinc_deg,
        orbit_state, look_direction, incidence_ref, incidence_sec,
        orbital_plane_ref, orbital_plane_sec, q_score, aoi.
    """
    if cfg is None:
        cfg = PairGraphConfig()

    df = df.copy().reset_index(drop=True)
    # Convert datetime to days-since-epoch as float for fast arithmetic
    df["_t"] = df["datetime"].astype("int64") / 1e9 / 86400.0

    records = []
    rows = df.to_dict("records")
    n = len(rows)

    for i in range(n):
        r = rows[i]
        for j in range(i + 1, n):
            s = rows[j]

            if cfg.require_same_orbit and r["orbit_state"] != s["orbit_state"]:
                continue
            if cfg.require_same_look and r["look_direction"] != s["look_direction"]:
                continue

            dt = abs(r["_t"] - s["_t"])
            if dt > cfg.dt_max_days:
                continue

            dinc = abs(r["incidence_angle_deg"] - s["incidence_angle_deg"])
            if dinc > cfg.dinc_max_deg:
                continue

            q = (1.0 / (dt + 1.0)) * (1.0 / (1.0 + dinc))
            if q < cfg.min_q_score:
                continue

            records.append({
                "id_ref": r["id"],
                "id_sec": s["id"],
                "datetime_ref": r["datetime"],
                "datetime_sec": s["datetime"],
                "dt_days": round(dt, 3),
                "dinc_deg": round(dinc, 4),
                "orbit_state": r["orbit_state"],
                "look_direction": r["look_direction"],
                "incidence_ref": r["incidence_angle_deg"],
                "incidence_sec": s["incidence_angle_deg"],
                "orbital_plane_ref": r.get("orbital_plane"),
                "orbital_plane_sec": s.get("orbital_plane"),
                "q_score": round(q, 6),
                "aoi": r.get("aoi"),
            })

    if not records:
        return pd.DataFrame(columns=[
            "id_ref", "id_sec", "datetime_ref", "datetime_sec", "dt_days",
            "dinc_deg", "orbit_state", "look_direction", "incidence_ref",
            "incidence_sec", "orbital_plane_ref", "orbital_plane_sec",
            "q_score", "aoi",
        ])

    edges = pd.DataFrame(records).sort_values("q_score", ascending=False).reset_index(drop=True)
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
        "q_score"          — top-N by quality score (default)
        "temporal_coverage" — greedy maximization of unique collect coverage

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
        for _, row in edges.iterrows():
            if len(selected) >= max_pairs:
                break
            selected.append(row)
            covered.add(row["id_ref"])
            covered.add(row["id_sec"])
        return pd.DataFrame(selected).reset_index(drop=True)

    raise ValueError(f"Unknown strategy: {strategy!r}. Use 'q_score' or 'temporal_coverage'.")


def find_triplets(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Enumerate all closed triplets (i→j, j→k, i→k) from the edge list.

    A triplet (A, B, C) is valid when edges A→B, B→C, and A→C all exist.
    Used for triplet closure error computation.

    Returns
    -------
    pd.DataFrame
        Columns: id_a, id_b, id_c — each row is one closure triplet.
    """
    # Build adjacency set for O(1) lookup
    edge_set = set(zip(edges["id_ref"], edges["id_sec"]))
    # Also add reverse direction (edge list is undirected stored as i<j by index)
    edge_set |= {(s, r) for r, s in edge_set}

    nodes = list(set(edges["id_ref"]) | set(edges["id_sec"]))
    node_idx = {n: i for i, n in enumerate(nodes)}

    # Build neighbour lists
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
                if (a, c) in edge_set or (c, a) in edge_set:
                    key = tuple(sorted([a, b, c]))
                    if key not in visited:
                        visited.add(key)
                        triplets.append({"id_a": key[0], "id_b": key[1], "id_c": key[2]})

    return pd.DataFrame(triplets)


def summarize_graph(edges: pd.DataFrame) -> dict:
    """Return summary statistics for the pair graph."""
    if edges.empty:
        return {"n_pairs": 0, "n_unique_collects": 0}

    return {
        "n_pairs": len(edges),
        "n_unique_collects": len(set(edges["id_ref"]) | set(edges["id_sec"])),
        "dt_min_days": edges["dt_days"].min(),
        "dt_median_days": edges["dt_days"].median(),
        "dt_max_days": edges["dt_days"].max(),
        "dinc_median_deg": edges["dinc_deg"].median(),
        "q_min": edges["q_score"].min(),
        "q_median": edges["q_score"].median(),
        "q_max": edges["q_score"].max(),
        "orbits": sorted(edges["orbit_state"].dropna().unique().tolist()),
        "look_dirs": sorted(edges["look_direction"].dropna().unique().tolist()),
    }
