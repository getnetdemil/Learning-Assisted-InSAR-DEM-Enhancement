#!/usr/bin/env python3
"""
Download Copernicus GLO-30 DEM tiles for a bounding box and merge into a single GeoTIFF.

Source: s3://copernicus-dem-30m/ (public, no authentication required)

Usage
-----
python scripts/download_copernicus_dem.py \
    --bbox_w -158 --bbox_s 18 --bbox_e -154 --bbox_n 22 \
    --out_dir data/reference/copernicus_dem

Output: {out_dir}/hawaii_dem.tif  (merged, ~30m resolution GeoTIFF)
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BUCKET      = "copernicus-dem-30m"
REGION      = "eu-central-1"
# Tile key pattern: Copernicus_DSM_COG_10_N{lat:02d}_00_{ew}{lon:03d}_00_DEM/
#                   Copernicus_DSM_COG_10_N{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif
TILE_PREFIX = "Copernicus_DSM_COG_10_N{lat:02d}_00_{ew}{lon:03d}_00_DEM"


def _tile_key(lat: int, lon: int) -> str:
    """Return the S3 key for a Copernicus GLO-30 tile."""
    ew  = "W" if lon < 0 else "E"
    abs_lon = abs(lon)
    name = TILE_PREFIX.format(lat=lat, ew=ew, lon=abs_lon)
    return f"{name}/{name}.tif"


def download_tiles(bbox_w: float, bbox_s: float, bbox_e: float, bbox_n: float,
                   out_dir: Path) -> list[Path]:
    """Download all GLO-30 tiles intersecting the bbox. Returns list of local paths."""
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", region_name=REGION, config=Config(signature_version=UNSIGNED))

    # Integer tile grid: tile at (lat, lon) covers [lat, lat+1) × [lon, lon+1)
    lat_min = math.floor(bbox_s)
    lat_max = math.floor(bbox_n)
    lon_min = math.floor(bbox_w)
    lon_max = math.floor(bbox_e)

    local_paths: list[Path] = []
    for lat in range(lat_min, lat_max + 1):
        for lon in range(lon_min, lon_max + 1):
            key        = _tile_key(lat, lon)
            local_path = tiles_dir / Path(key).name
            if local_path.exists():
                log.info("Already exists: %s", local_path.name)
                local_paths.append(local_path)
                continue
            try:
                log.info("Downloading s3://%s/%s → %s", BUCKET, key, local_path.name)
                s3.download_file(BUCKET, key, str(local_path))
                local_paths.append(local_path)
            except s3.exceptions.ClientError as e:
                if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                    log.warning("Tile not found (ocean/no-data): %s", key)
                else:
                    log.error("S3 error for %s: %s", key, e)

    return local_paths


def merge_tiles(tile_paths: list[Path], out_path: Path) -> None:
    """Merge multiple GeoTIFF tiles into a single file using rasterio."""
    import rasterio
    from rasterio.merge import merge

    if not tile_paths:
        log.error("No tiles to merge.")
        return

    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, transform = merge(datasets)
        profile = datasets[0].profile.copy()
        profile.update({
            "height":    mosaic.shape[1],
            "width":     mosaic.shape[2],
            "transform": transform,
            "compress":  "deflate",
            "tiled":     True,
            "blockxsize": 512,
            "blockysize": 512,
        })
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic)
        log.info("Saved merged DEM → %s  (shape %s)", out_path, mosaic.shape[1:])
    finally:
        for ds in datasets:
            ds.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and merge Copernicus GLO-30 DEM tiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bbox_w", type=float, required=True, help="West longitude (degrees)")
    p.add_argument("--bbox_s", type=float, required=True, help="South latitude (degrees)")
    p.add_argument("--bbox_e", type=float, required=True, help="East longitude (degrees)")
    p.add_argument("--bbox_n", type=float, required=True, help="North latitude (degrees)")
    p.add_argument("--out_dir", default="data/reference/copernicus_dem",
                   help="Output directory for tiles and merged DEM.")
    p.add_argument("--merged_name", default="hawaii_dem.tif",
                   help="Filename for the merged output GeoTIFF.")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_path = out_dir / args.merged_name
    if merged_path.exists():
        log.info("Merged DEM already exists: %s — delete to re-download.", merged_path)
        return

    tile_paths = download_tiles(args.bbox_w, args.bbox_s, args.bbox_e, args.bbox_n, out_dir)
    log.info("Downloaded %d tiles.", len(tile_paths))

    if tile_paths:
        merge_tiles(tile_paths, merged_path)
    else:
        log.error("No tiles downloaded — check bbox and S3 access.")
        sys.exit(1)


if __name__ == "__main__":
    main()
