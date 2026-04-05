"""
One-time patch: re-save ifg_goldstein_complex_real_imag.tif files as proper 2-band rasterio TIFFs.

tifffile saves (H,W,2) arrays as multi-page TIFFs that rasterio reads as 1 band.
This script reads with tifffile (correct), then re-saves with rasterio (2-band compatible).
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
import tifffile

pairs_dir = Path("data/processed/pairs_full_image")
fixed = 0
skipped_missing = 0
skipped_ok = 0

for pair_dir in sorted(d for d in pairs_dir.iterdir() if d.is_dir()):
    path = pair_dir / "ifg_goldstein_complex_real_imag.tif"
    if not path.exists():
        skipped_missing += 1
        continue

    # Check if already fixed
    with rasterio.open(str(path)) as src:
        if src.count == 2:
            skipped_ok += 1
            continue

    arr = tifffile.imread(str(path))
    if arr.ndim != 3 or arr.shape[-1] != 2:
        print(f"SKIP unexpected shape {arr.shape}: {pair_dir.name}", flush=True)
        continue

    re = arr[..., 0].astype(np.float32)
    im = arr[..., 1].astype(np.float32)
    H, W = re.shape
    tmp = str(path) + ".tmp"
    with rasterio.open(tmp, "w", driver="GTiff", count=2,
                       dtype="float32", height=H, width=W) as dst:
        dst.write(re, 1)
        dst.write(im, 2)
    shutil.move(tmp, str(path))
    fixed += 1
    print(f"[{fixed}] Fixed {pair_dir.name}", flush=True)

print(f"\nDone: {fixed} fixed, {skipped_ok} already ok, {skipped_missing} missing file.")
