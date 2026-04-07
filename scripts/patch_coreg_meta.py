"""
Patch coreg_meta.json files with FiLM conditioning fields missing from
the original preprocess_pairs.py output.

Adds four fields to each coreg_meta.json:
  incidence_angle_deg  — mean of ref+sec incidence angles
  mode                 — always "SL" (Capella Spotlight)
  look_direction       — "RIGHT" or "LEFT" from manifest
  snr_proxy            — q_score from manifest (quality proxy in [0,1])

Source of truth: data/manifests/hawaii_pairs.parquet

Usage
-----
python scripts/patch_coreg_meta.py \\
    --pairs_dir data/processed/pairs \\
    --manifest  data/manifests/hawaii_pairs.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Patch coreg_meta.json with FiLM conditioning fields.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pairs_dir", required=True,
                   help="Root directory containing per-pair subdirectories.")
    p.add_argument("--manifest", required=True,
                   help="Path to hawaii_pairs.parquet manifest.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pairs_dir = Path(args.pairs_dir)
    if not pairs_dir.is_dir():
        log.error("pairs_dir does not exist: %s", pairs_dir)
        raise SystemExit(1)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        raise SystemExit(1)

    # Load manifest and build (id_ref, id_sec) → row lookup
    df = pd.read_parquet(manifest_path)
    required = {"id_ref", "id_sec", "incidence_ref", "incidence_sec",
                "look_direction", "q_score"}
    missing = required - set(df.columns)
    if missing:
        log.error("Manifest missing columns: %s", missing)
        raise SystemExit(1)

    lookup: dict[tuple[str, str], object] = {
        (row.id_ref, row.id_sec): row
        for row in df.itertuples(index=False)
    }
    log.info("Loaded manifest: %d pairs", len(lookup))

    pair_dirs = sorted(p for p in pairs_dir.iterdir() if p.is_dir())
    n_total = 0
    n_patched = 0
    n_missing = 0

    for pair_dir in pair_dirs:
        meta_path = pair_dir / "coreg_meta.json"
        if not meta_path.exists():
            continue
        n_total += 1

        with open(meta_path) as f:
            meta = json.load(f)

        key = (meta.get("id_ref", ""), meta.get("id_sec", ""))
        row = lookup.get(key)
        if row is None:
            log.warning("Pair not found in manifest: %s / %s", key[0], key[1])
            n_missing += 1
            continue

        meta["incidence_angle_deg"] = (row.incidence_ref + row.incidence_sec) / 2.0
        meta["mode"] = "SL"  # all Capella Spotlight
        meta["look_direction"] = str(row.look_direction)
        meta["snr_proxy"] = float(row.q_score)

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        n_patched += 1

    log.info("Patched %d / %d coreg_meta.json files (%d not in manifest)",
             n_patched, n_total, n_missing)


if __name__ == "__main__":
    main()
