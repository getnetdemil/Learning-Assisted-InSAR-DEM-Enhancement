"""
Select the minimal set of pairs needed to complete 2-leg triplets in the processed set.

After preprocessing the top-100 pairs by q_score, every triplet in the manifest has
at most 2 of its 3 legs processed.  This script identifies the unique missing 3rd legs,
verifies they exist in hawaii_pairs.parquet, and saves a parquet/CSV subset for
follow-on preprocessing.

Usage
-----
python scripts/select_triplet_completing_pairs.py \
    --pairs_manifest    data/manifests/hawaii_pairs.parquet \
    --triplets_manifest data/manifests/hawaii_triplets_strict.parquet \
    --processed_dir     data/processed/pairs \
    --out_parquet       data/manifests/triplet_completing_pairs.parquet \
    --out_csv           data/manifests/triplet_completing_pairs.csv
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_proc_set(processed_dir: Path, pair_lookup: set) -> set:
    """
    Read coreg_meta.json in each processed pair directory and return the set of
    canonical (id_ref, id_sec) keys that have been processed.

    Directory names are truncated (filesystem limit) — always use the JSON.
    """
    proc = set()
    for d in sorted(processed_dir.iterdir()):
        meta_path = d / "coreg_meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            m = json.load(f)
        key_fwd = (m["id_ref"], m["id_sec"])
        key_rev = (m["id_sec"], m["id_ref"])
        if key_fwd in pair_lookup:
            proc.add(key_fwd)
        elif key_rev in pair_lookup:
            proc.add(key_rev)
    return proc


def canonical(a: str, b: str, pair_lookup: set):
    """Return the canonical (id_ref, id_sec) for edge (a, b), or None."""
    if (a, b) in pair_lookup:
        return (a, b)
    if (b, a) in pair_lookup:
        return (b, a)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    pairs_df = pd.read_parquet(args.pairs_manifest)
    triplets_df = pd.read_parquet(args.triplets_manifest)
    processed_dir = Path(args.processed_dir)

    # Build lookup set from manifest (canonical direction only)
    pair_lookup: set = set(zip(pairs_df["id_ref"], pairs_df["id_sec"]))

    # Build processed set from coreg_meta.json files
    proc_set = build_proc_set(processed_dir, pair_lookup)
    print(f"Processed pairs found:  {len(proc_set)}")

    # Count unique epochs in processed set
    epochs_proc: set = set()
    for id_ref, id_sec in proc_set:
        epochs_proc.add(id_ref)
        epochs_proc.add(id_sec)

    # Find 2-leg triplets and collect missing 3rd legs
    # missing maps (id_ref, id_sec) -> number of triplets it would complete
    missing: dict = {}
    two_leg_count = 0

    for row in triplets_df.itertuples(index=False):
        a, b, c = row.id_a, row.id_b, row.id_c
        legs = [
            canonical(a, b, pair_lookup),
            canonical(b, c, pair_lookup),
            canonical(a, c, pair_lookup),
        ]
        if any(l is None for l in legs):
            continue  # triplet not in pair manifest at all
        in_proc = [l for l in legs if l in proc_set]
        not_proc = [l for l in legs if l not in proc_set]
        if len(in_proc) == 2 and len(not_proc) == 1:
            two_leg_count += 1
            key = not_proc[0]
            missing[key] = missing.get(key, 0) + 1

    print(f"2-leg triplets found:    {two_leg_count}")
    print(f"Unique missing pairs:    {len(missing)}")

    # Pull rows from pairs_df using boolean mask (NOT groupby with tuple keys)
    missing_set = set(missing.keys())
    mask = pairs_df.apply(
        lambda r: (r["id_ref"], r["id_sec"]) in missing_set, axis=1
    )
    subset = pairs_df[mask].copy()

    # Verify all missing pairs were found
    found = set(zip(subset["id_ref"], subset["id_sec"]))
    not_found = missing_set - found
    if not_found:
        print(f"WARNING: {len(not_found)} missing pairs NOT in hawaii_pairs.parquet:")
        for p in sorted(not_found):
            print(f"  {p}")
    else:
        print(f"All {len(subset)} pairs verified in hawaii_pairs.parquet ✓")

    # Compute post-fix network statistics
    epochs_new: set = set()
    for id_ref, id_sec in missing_set:
        epochs_new.add(id_ref)
        epochs_new.add(id_sec)
    all_epochs = epochs_proc | epochs_new
    P_after = len(proc_set) + len(subset)
    T_after = len(all_epochs)
    overdetermined = P_after - T_after
    status = f"OVERCONSTRAINED (+{overdetermined})" if overdetermined > 0 else f"UNDERDETERMINED ({overdetermined})"
    print(f"After processing: P={P_after}, T={T_after} → {status}")

    # Save outputs
    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(out_parquet, index=False)
    print(f"Saved {len(subset)} rows → {out_parquet}")

    if args.out_csv:
        out_csv = Path(args.out_csv)
        subset.to_csv(out_csv, index=False)
        print(f"Saved {len(subset)} rows → {out_csv}")

    if overdetermined <= 0:
        print("ERROR: Network still underdetermined after adding missing pairs.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs_manifest",
        default="data/manifests/hawaii_pairs.parquet",
        help="Path to hawaii_pairs.parquet (full pair manifest)",
    )
    parser.add_argument(
        "--triplets_manifest",
        default="data/manifests/hawaii_triplets_strict.parquet",
        help="Path to hawaii_triplets_strict.parquet",
    )
    parser.add_argument(
        "--processed_dir",
        default="data/processed/pairs",
        help="Directory containing processed pair subdirectories",
    )
    parser.add_argument(
        "--out_parquet",
        default="data/manifests/triplet_completing_pairs.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--out_csv",
        default="data/manifests/triplet_completing_pairs.csv",
        help="Output CSV path (optional, pass empty string to skip)",
    )
    args = parser.parse_args()
    main(args)
