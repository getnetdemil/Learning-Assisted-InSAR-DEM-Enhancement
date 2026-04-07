"""
Build interferometric pairs manifest using the strict full-image pair graph.

Reads full_index.parquet, filters to one or more AOIs, constructs the
pair graph with src.insar_processing.pair_graph_full_image, and saves
the resulting edge list as a parquet file.

Also optionally enumerates triplets for closure-error computation.

Usage
-----
# Build pairs for Hawaii (AOI_000) with default strict config:
PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement \\
python scripts/build_pairs_manifest.py \\
    --manifest data/manifests/full_index.parquet \\
    --aoi AOI_000 \\
    --out data/manifests/full_index_full_image.parquet

# Relax platform requirement (allow cross-satellite pairs):
PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement \\
python scripts/build_pairs_manifest.py \\
    --manifest data/manifests/full_index.parquet \\
    --aoi AOI_000 \\
    --no-require-same-platform \\
    --out data/manifests/full_index_full_image.parquet

# All AOIs, no platform restriction, save triplets too:
PYTHONPATH=/scratch/gdemil24/Learning-Assisted-InSAR-DEM-Enhancement \\
python scripts/build_pairs_manifest.py \\
    --manifest data/manifests/full_index.parquet \\
    --no-require-same-platform \\
    --out data/manifests/full_index_full_image.parquet \\
    --triplets data/manifests/full_index_triplets_full_image.parquet
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.insar_processing.pair_graph_full_image import (
    PairGraphConfig,
    build_pair_graph,
    find_triplets,
    summarize_graph,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build strict interferometric pair graph from full_index.parquet."
    )
    p.add_argument("--manifest", default="data/manifests/full_index.parquet",
                   help="Input manifest parquet (default: data/manifests/full_index.parquet)")
    p.add_argument("--aoi", default=None,
                   help="Filter to a single AOI label (e.g. AOI_000). Omit for all AOIs.")
    p.add_argument("--out", required=True,
                   help="Output parquet path for the pairs edge list")
    p.add_argument("--triplets", default=None,
                   help="Optional output parquet path for closure triplets")

    # Temporal / geometry thresholds
    p.add_argument("--dt-max", type=float, default=365.0,
                   help="Maximum temporal baseline in days (default: 365)")
    p.add_argument("--dt-min", type=float, default=0.0,
                   help="Minimum temporal baseline in days (default: 0)")
    p.add_argument("--dinc-max", type=float, default=5.0,
                   help="Maximum incidence-angle difference in degrees (default: 5)")
    p.add_argument("--dlook-max", type=float, default=2.5,
                   help="Maximum look-angle difference in degrees (default: 2.5). "
                        "Pass -1 to disable.")
    p.add_argument("--dsquint-max", type=float, default=5.0,
                   help="Maximum squint-angle difference in degrees (default: 5). "
                        "Pass -1 to disable.")
    p.add_argument("--min-bbox-overlap", type=float, default=0.50,
                   help="Minimum footprint overlap fraction (default: 0.50). "
                        "Pass -1 to disable.")
    p.add_argument("--max-range-res-rel", type=float, default=0.15,
                   help="Maximum relative range-resolution difference (default: 0.15). "
                        "Pass -1 to disable.")
    p.add_argument("--max-az-res-rel", type=float, default=0.15,
                   help="Maximum relative azimuth-resolution difference (default: 0.15). "
                        "Pass -1 to disable.")
    p.add_argument("--min-q", type=float, default=0.0,
                   help="Minimum quality score to include a pair (default: 0)")

    # Hard compatibility switches
    p.add_argument("--no-require-same-aoi", dest="require_same_aoi",
                   action="store_false", default=True,
                   help="Allow pairs across different AOI labels")
    p.add_argument("--no-require-same-orbit", dest="require_same_orbit",
                   action="store_false", default=True,
                   help="Allow ascending/descending mixing")
    p.add_argument("--no-require-same-look", dest="require_same_look",
                   action="store_false", default=True,
                   help="Allow left/right look mixing")
    p.add_argument("--no-require-same-orbital-plane", dest="require_same_orbital_plane",
                   action="store_false", default=True,
                   help="Allow different orbital planes (track numbers)")
    p.add_argument("--no-require-same-platform", dest="require_same_platform",
                   action="store_false", default=True,
                   help="Allow cross-satellite pairs (different Capella satellites)")
    p.add_argument("--no-require-same-mode", dest="require_same_mode",
                   action="store_false", default=True,
                   help="Allow different instrument modes (e.g. spotlight + stripmap)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        raise SystemExit(1)

    log.info("Loading manifest: %s", manifest_path)
    df = pd.read_parquet(manifest_path)
    log.info("Total rows: %d, columns: %s", len(df), list(df.columns))

    if args.aoi:
        before = len(df)
        df = df[df["aoi"] == args.aoi].copy()
        log.info("Filtered to AOI=%s: %d → %d rows", args.aoi, before, len(df))
        if df.empty:
            log.error("No rows after AOI filter. Available AOIs: %s",
                      sorted(pd.read_parquet(manifest_path)["aoi"].dropna().unique().tolist()))
            raise SystemExit(1)

    # Build config from CLI args
    cfg = PairGraphConfig(
        dt_max_days=args.dt_max,
        dt_min_days=args.dt_min,
        dinc_max_deg=args.dinc_max,
        dlook_max_deg=None if args.dlook_max < 0 else args.dlook_max,
        dsquint_max_deg=None if args.dsquint_max < 0 else args.dsquint_max,
        min_bbox_overlap_frac=None if args.min_bbox_overlap < 0 else args.min_bbox_overlap,
        max_range_res_rel_diff=None if args.max_range_res_rel < 0 else args.max_range_res_rel,
        max_azimuth_res_rel_diff=None if args.max_az_res_rel < 0 else args.max_az_res_rel,
        min_q_score=args.min_q,
        require_same_aoi=args.require_same_aoi,
        require_same_orbit=args.require_same_orbit,
        require_same_look=args.require_same_look,
        require_same_orbital_plane=args.require_same_orbital_plane,
        require_same_platform=args.require_same_platform,
        require_same_mode=args.require_same_mode,
    )

    log.info("Building pair graph (dt_max=%.0fd, dinc_max=%.1f°, dlook_max=%s, "
             "same_platform=%s, same_orbital_plane=%s) ...",
             cfg.dt_max_days, cfg.dinc_max_deg,
             f"{cfg.dlook_max_deg:.1f}°" if cfg.dlook_max_deg is not None else "off",
             cfg.require_same_platform,
             cfg.require_same_orbital_plane)

    edges = build_pair_graph(df, cfg)
    summary = summarize_graph(edges)
    log.info("Pair graph summary: %s", summary)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(out_path, index=False)
    log.info("Saved %d pairs → %s", len(edges), out_path)

    if args.triplets:
        log.info("Enumerating closure triplets ...")
        triplets = find_triplets(edges)
        triplet_path = Path(args.triplets)
        triplet_path.parent.mkdir(parents=True, exist_ok=True)
        triplets.to_parquet(triplet_path, index=False)
        log.info("Saved %d triplets → %s", len(triplets), triplet_path)


if __name__ == "__main__":
    main()
