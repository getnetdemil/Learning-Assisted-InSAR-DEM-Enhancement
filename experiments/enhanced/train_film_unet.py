"""
FiLMUNet training script — self-supervised InSAR interferogram enhancement.

Usage
-----
python experiments/enhanced/train_film_unet.py \\
    --data_config configs/data/capella_aoi_selection.yaml \\
    --model_config configs/model/film_unet.yaml \\
    --train_config configs/train/contest.yaml

Three checkpoints are written automatically:
  <output_dir>/best_closure.pt   — best validation closure error
  <output_dir>/best_unwrap.pt    — (placeholder) best unwrap success rate
  <output_dir>/final.pt          — end of last epoch

Each checkpoint stores: model state, optimizer state, epoch, config dicts, git hash.
"""

import argparse
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

# Suppress rasterio georeference warning — training tiles are pixel-only (no CRS needed)
try:
    import rasterio.errors
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
except ImportError:
    pass


def _worker_init(worker_id: int) -> None:
    """Suppress rasterio warnings in DataLoader worker processes."""
    try:
        import rasterio.errors
        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
    except Exception:
        pass

# ── repo-relative import fix ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from models.film_unet import FiLMUNet
from losses.physics_losses import InSARLoss, LossWeights, PhysicsLossInputs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ─── utilities ────────────────────────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


# ─── dataset ──────────────────────────────────────────────────────────────────

class InSARTileDataset(Dataset):
    """
    Loads preprocessed interferogram tiles for FiLMUNet training.

    Each item in the manifest points to a directory containing:
        ifg_goldstein.tif   — complex interferogram (CInt16 / float32 Re+Im)
        coherence.tif       — coherence [0,1]
        coreg_meta.json     — pair metadata (dates, B_perp, inc_angle, …)

    Returns a dict with:
        x        (3, H, W) float32 — Re, Im, coherence
        metadata (7,)      float32 — FiLM conditioning vector
        phi      (H, W)    float32 — wrapped phase (for closure loss)
        pair_id  str
    """

    # Metadata normalisation constants (approximate physical ranges)
    _META_MEAN = np.array([30.0, 45.0, 35.0, 500.0, 0.5, 0.5, 0.5], dtype=np.float32)
    _META_STD  = np.array([60.0,  8.0,  8.0, 2000.0, 0.5, 0.5, 0.3], dtype=np.float32)

    def __init__(
        self,
        pair_dirs: List[Path],
        tile_size: int = 256,
        stride: int = 128,
        min_coherence: float = 0.15,
        augment: bool = False,
        in_channels: int = 3,
    ):
        self.tile_size = tile_size
        self.stride = stride
        self.min_coherence = min_coherence
        self.augment = augment
        self.in_channels = in_channels

        self.tiles: List[Tuple[Path, int, int]] = []  # (pair_dir, row_off, col_off)
        for pd_path in pair_dirs:
            self._index_pair(pd_path)

    def _index_pair(self, pair_dir: Path) -> None:
        ifg_path = pair_dir / "ifg_goldstein.tif"
        coh_path = pair_dir / "coherence.tif"
        if not ifg_path.exists() or not coh_path.exists():
            return
        try:
            import rasterio
            with rasterio.open(coh_path) as src:
                H, W = src.height, src.width
        except Exception:
            return
        T = self.tile_size
        S = self.stride
        for r in range(0, H - T + 1, S):
            for c in range(0, W - T + 1, S):
                self.tiles.append((pair_dir, r, c))

    def _load_tile(self, pair_dir: Path, r: int, c: int):
        import rasterio
        T = self.tile_size
        window = rasterio.windows.Window(c, r, T, T)

        # Goldstein (pseudo-clean target)
        with rasterio.open(pair_dir / "ifg_goldstein.tif") as src:
            data = src.read(window=window)
            if data.shape[0] == 1:
                cplx = data[0].astype(np.complex64)
                re_gold, im_gold = cplx.real, cplx.imag
            else:
                re_gold, im_gold = data[0].astype(np.float32), data[1].astype(np.float32)

        # Raw (noisy input) — fallback to Goldstein if missing
        raw_path = pair_dir / "ifg_raw.tif"
        if raw_path.exists():
            with rasterio.open(raw_path) as src:
                data = src.read(window=window)
                if data.shape[0] == 1:
                    cplx = data[0].astype(np.complex64)
                    re_raw, im_raw = cplx.real, cplx.imag
                else:
                    re_raw, im_raw = data[0].astype(np.float32), data[1].astype(np.float32)
        else:
            re_raw, im_raw = re_gold, im_gold

        with rasterio.open(pair_dir / "coherence.tif") as src:
            coh = src.read(1, window=window).astype(np.float32)

        # Guard against NaN/Inf from edge pixels or masked nodata
        re_raw  = np.nan_to_num(re_raw,  nan=0.0, posinf=0.0, neginf=0.0)
        im_raw  = np.nan_to_num(im_raw,  nan=0.0, posinf=0.0, neginf=0.0)
        re_gold = np.nan_to_num(re_gold, nan=0.0, posinf=0.0, neginf=0.0)
        im_gold = np.nan_to_num(im_gold, nan=0.0, posinf=0.0, neginf=0.0)
        coh     = np.nan_to_num(coh,     nan=0.0, posinf=1.0, neginf=0.0)

        return re_raw, im_raw, re_gold, im_gold, coh

    def _load_meta(self, pair_dir: Path) -> np.ndarray:
        meta_path = pair_dir / "coreg_meta.json"
        try:
            with open(meta_path) as f:
                m = json.load(f)
            dt = float(m.get("dt_days", 30.0))          # key from preprocess_pairs.py
            inc = float(m.get("incidence_angle_deg", 45.0))  # not in coreg_meta; use default
            graze = 90.0 - inc
            bperp = float(m.get("bperp_m", 500.0))      # key from preprocess_pairs.py
            mode = 1.0 if str(m.get("mode", "")).upper() == "SL" else 0.0
            look = 1.0 if str(m.get("look_direction", "")).upper() == "RIGHT" else 0.0
            snr = float(m.get("snr_proxy", 0.5))
            raw = np.array([dt, inc, graze, bperp, mode, look, snr], dtype=np.float32)
        except Exception:
            raw = self._META_MEAN.copy()
        return (raw - self._META_MEAN) / (self._META_STD + 1e-8)

    @staticmethod
    def _load_meta_static(pair_dir: Path) -> np.ndarray:
        """Static version of _load_meta — used by TripletTileDataset."""
        _META_MEAN = np.array([30.0, 45.0, 35.0, 500.0, 0.5, 0.5, 0.5], dtype=np.float32)
        _META_STD  = np.array([60.0,  8.0,  8.0, 2000.0, 0.5, 0.5, 0.3], dtype=np.float32)
        meta_path = pair_dir / "coreg_meta.json"
        try:
            with open(meta_path) as f:
                m = json.load(f)
            dt    = float(m.get("dt_days", 30.0))
            inc   = float(m.get("incidence_angle_deg", 45.0))
            graze = 90.0 - inc
            bperp = float(m.get("bperp_m", 500.0))
            mode  = 1.0 if str(m.get("mode", "")).upper() == "SL" else 0.0
            look  = 1.0 if str(m.get("look_direction", "")).upper() == "RIGHT" else 0.0
            snr   = float(m.get("snr_proxy", 0.5))
            raw   = np.array([dt, inc, graze, bperp, mode, look, snr], dtype=np.float32)
        except Exception:
            raw = _META_MEAN.copy()
        return (raw - _META_MEAN) / (_META_STD + 1e-8)

    @staticmethod
    def _augment(re_raw, im_raw, re_gold, im_gold, coh):
        """Physically-safe augmentations: rotations, flips, global phase offset."""
        k = random.randint(0, 3)
        re_raw  = np.rot90(re_raw,  k).copy()
        im_raw  = np.rot90(im_raw,  k).copy()
        re_gold = np.rot90(re_gold, k).copy()
        im_gold = np.rot90(im_gold, k).copy()
        coh     = np.rot90(coh,     k).copy()
        if random.random() > 0.5:
            re_raw  = np.fliplr(re_raw).copy()
            im_raw  = np.fliplr(im_raw).copy()
            re_gold = np.fliplr(re_gold).copy()
            im_gold = np.fliplr(im_gold).copy()
            coh     = np.fliplr(coh).copy()
        # Global phase offset applied consistently to both raw and gold
        phi_offset = random.uniform(-np.pi, np.pi)
        cos_o, sin_o = np.cos(phi_offset), np.sin(phi_offset)
        re_raw_new  = re_raw  * cos_o - im_raw  * sin_o
        im_raw_new  = re_raw  * sin_o + im_raw  * cos_o
        re_gold_new = re_gold * cos_o - im_gold * sin_o
        im_gold_new = re_gold * sin_o + im_gold * cos_o
        # Amplitude jitter ±10%
        scale = random.uniform(0.9, 1.1)
        return re_raw_new * scale, im_raw_new * scale, re_gold_new * scale, im_gold_new * scale, coh

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Optional[dict]:
        pair_dir, r, c = self.tiles[idx]
        re_raw, im_raw, re_gold, im_gold, coh = self._load_tile(pair_dir, r, c)

        if coh.mean() < self.min_coherence:
            # Return a random other tile to avoid None in DataLoader
            alt = random.randint(0, len(self.tiles) - 1)
            return self[alt]

        if self.augment:
            re_raw, im_raw, re_gold, im_gold, coh = self._augment(
                re_raw, im_raw, re_gold, im_gold, coh)

        meta = self._load_meta(pair_dir)
        phi = np.arctan2(im_gold, re_gold).astype(np.float32)  # Goldstein phase as reference

        if self.in_channels == 3:
            x = np.stack([re_raw, im_raw, coh], axis=0)    # noisy input
        else:
            x = np.stack([re_raw, im_raw], axis=0)

        gold = np.stack([re_gold, im_gold], axis=0)         # pseudo-clean target

        return {
            "x":        torch.from_numpy(x),
            "gold":     torch.from_numpy(gold),
            "metadata": torch.from_numpy(meta),
            "phi":      torch.from_numpy(phi),
            "pair_id":  str(pair_dir.name),
        }


def _pair_key(row, edge: str) -> str:
    """Build pair dir name (id_ref__id_sec) from triplet row for edge 'ij', 'jk', or 'ik'.

    Triplet manifest columns: id_a, id_b, id_c.
    Edge mapping: ij=a→b, jk=b→c, ik=a→c  (double-underscore separator, matches dir names).
    """
    _edge_map = {"ij": ("id_a", "id_b"), "jk": ("id_b", "id_c"), "ik": ("id_a", "id_c")}
    col_ref, col_sec = _edge_map[edge]
    return f"{row[col_ref]}__{row[col_sec]}"


class TripletTileDataset(Dataset):
    """
    Each item is a spatially-aligned tile from all 3 pairs of a closure triplet.
    Enables the closure loss in run_epoch().

    triplets_df : DataFrame with columns id_ref_ij, id_sec_ij,
                  id_ref_jk, id_sec_jk, id_ref_ik, id_sec_ik
    pair_dir_map: {pair_id -> Path} built from discover_pair_dirs()
    """

    def __init__(
        self,
        triplets_df: pd.DataFrame,
        pair_dir_map: Dict[str, Path],
        tile_size: int = 256,
        stride: int = 128,
        min_coherence: float = 0.15,
    ):
        self.tile_size = tile_size
        self.min_coherence = min_coherence
        self.tiles: List[Tuple] = []  # (dir_ij, dir_jk, dir_ik, r, c)

        for _, row in triplets_df.iterrows():
            # Try canonical order first; fall back to reversed (ref/sec swapped)
            _edge_rev = {"ij": ("id_b", "id_a"), "jk": ("id_c", "id_b"), "ik": ("id_c", "id_a")}
            def _lookup(edge):
                key = _pair_key(row, edge)
                if key in pair_dir_map:
                    return pair_dir_map[key]
                cols = _edge_rev[edge]
                rev = f"{row[cols[0]]}__{row[cols[1]]}"
                return pair_dir_map.get(rev)
            dir_ij = _lookup("ij")
            dir_jk = _lookup("jk")
            dir_ik = _lookup("ik")
            if None in (dir_ij, dir_jk, dir_ik):
                continue
            try:
                import rasterio
                with rasterio.open(dir_ij / "ifg_goldstein.tif") as src:
                    H, W = src.height, src.width
            except Exception:
                continue
            T, S = tile_size, stride
            for r in range(0, H - T + 1, S):
                for c in range(0, W - T + 1, S):
                    self.tiles.append((dir_ij, dir_jk, dir_ik, r, c))
            # Cover bottom-right corner
            if H > T:
                for c in range(0, W - T + 1, S):
                    self.tiles.append((dir_ij, dir_jk, dir_ik, H - T, c))
            if W > T:
                for r in range(0, H - T + 1, S):
                    self.tiles.append((dir_ij, dir_jk, dir_ik, r, W - T))

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> dict:
        dir_ij, dir_jk, dir_ik, r, c = self.tiles[idx]
        tile_ij = self._load(dir_ij, r, c)
        tile_jk = self._load(dir_jk, r, c)
        tile_ik = self._load(dir_ik, r, c)
        if any(t is None for t in (tile_ij, tile_jk, tile_ik)):
            alt = random.randint(0, len(self.tiles) - 1)
            return self[alt]
        x_ij, gold_ij, phi_ij, coh_ij = tile_ij
        x_jk, gold_jk, phi_jk, _      = tile_jk
        x_ik, gold_ik, phi_ik, _      = tile_ik
        return {
            "x_ij":    torch.from_numpy(x_ij),
            "gold_ij": torch.from_numpy(gold_ij),
            "phi_ij":  torch.from_numpy(phi_ij),
            "x_jk":    torch.from_numpy(x_jk),
            "gold_jk": torch.from_numpy(gold_jk),
            "phi_jk":  torch.from_numpy(phi_jk),
            "x_ik":    torch.from_numpy(x_ik),
            "gold_ik": torch.from_numpy(gold_ik),
            "phi_ik":  torch.from_numpy(phi_ik),
            "coh_ij":  torch.from_numpy(coh_ij),
            "meta_ij": torch.from_numpy(InSARTileDataset._load_meta_static(dir_ij)),
            "meta_jk": torch.from_numpy(InSARTileDataset._load_meta_static(dir_jk)),
            "meta_ik": torch.from_numpy(InSARTileDataset._load_meta_static(dir_ik)),
        }

    def _load(
        self, pair_dir: Path, r: int, c: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Returns (x, gold, phi, coh) arrays or None if low coherence."""
        import rasterio
        T = self.tile_size
        window = rasterio.windows.Window(c, r, T, T)
        try:
            with rasterio.open(pair_dir / "ifg_goldstein.tif") as src:
                data = src.read(window=window)
            if data.shape[0] == 1:
                cplx = data[0].astype(np.complex64)
                re_gold, im_gold = cplx.real, cplx.imag
            else:
                re_gold = data[0].astype(np.float32)
                im_gold = data[1].astype(np.float32)
            re_gold = np.nan_to_num(re_gold, nan=0.0)
            im_gold = np.nan_to_num(im_gold, nan=0.0)

            raw_path = pair_dir / "ifg_raw.tif"
            if raw_path.exists():
                with rasterio.open(raw_path) as src:
                    raw = src.read(window=window)
                if raw.shape[0] == 1:
                    cplx = raw[0].astype(np.complex64)
                    re_raw, im_raw = cplx.real, cplx.imag
                else:
                    re_raw = raw[0].astype(np.float32)
                    im_raw = raw[1].astype(np.float32)
                re_raw = np.nan_to_num(re_raw, nan=0.0)
                im_raw = np.nan_to_num(im_raw, nan=0.0)
            else:
                re_raw, im_raw = re_gold.copy(), im_gold.copy()

            with rasterio.open(pair_dir / "coherence.tif") as src:
                coh = np.nan_to_num(
                    src.read(1, window=window).astype(np.float32), nan=0.0
                )
        except Exception:
            return None

        if coh.mean() < self.min_coherence:
            return None

        x    = np.stack([re_raw, im_raw, coh], axis=0)          # (3, H, W)
        gold = np.stack([re_gold, im_gold], axis=0)              # (2, H, W)
        phi  = np.arctan2(im_gold, re_gold)[np.newaxis].astype(np.float32)  # (1, H, W)
        return x, gold, phi, coh[np.newaxis]


# ─── data loading helpers ─────────────────────────────────────────────────────

def discover_pair_dirs(pairs_dir: Path) -> List[Path]:
    """Return all subdirectories of pairs_dir that contain ifg_goldstein.tif."""
    return sorted(
        p for p in pairs_dir.iterdir()
        if p.is_dir() and (p / "ifg_goldstein.tif").exists()
    )


def temporal_split(pair_dirs: List[Path], train_frac: float, val_frac: float):
    """Split pair dirs by date order (encoded in dir name YYYYMMDD_YYYYMMDD)."""
    def _date_key(p: Path) -> str:
        return p.name[:8] if len(p.name) >= 8 else p.name

    sorted_dirs = sorted(pair_dirs, key=_date_key)
    n = len(sorted_dirs)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return (
        sorted_dirs[:n_train],
        sorted_dirs[n_train:n_train + n_val],
        sorted_dirs[n_train + n_val:],
    )


# ─── checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    configs: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "configs": configs,
            "git_hash": git_hash(),
        },
        path,
    )


# ─── training / validation loops ─────────────────────────────────────────────

def run_epoch(
    model: FiLMUNet,
    loader: DataLoader,
    criterion: InSARLoss,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_clip: float = 1.0,
    zero_metadata: bool = False,
    triplet_loader: Optional[DataLoader] = None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = {}
    n = 0

    # Prepare triplet iterator (cycles over triplet batches in lock-step with main loader)
    _triplet_iter = iter(triplet_loader) if (triplet_loader is not None and training) else None

    with torch.set_grad_enabled(training):
        for batch in loader:
            x    = batch["x"].float().to(device)          # (B, C, H, W) — raw (noisy)
            gold = batch["gold"].float().to(device)        # (B, 2, H, W) — Goldstein (pseudo-clean)
            meta = batch["metadata"].float().to(device)
            if zero_metadata:
                meta = torch.zeros_like(meta)              # ablation V5: no FiLM conditioning

            denoised, log_var = model(x, meta)

            inp = PhysicsLossInputs(
                pred_a=denoised,      # model prediction on raw input
                sublook_b=gold,       # Goldstein as N2N target
                log_var=log_var,
                full_look=gold,       # gradient loss: model output vs Goldstein
            )
            loss, breakdown = criterion(inp)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            for k, v in breakdown.items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1

            # ── Closure loss from triplet batch ──────────────────────────────
            if _triplet_iter is not None:
                try:
                    tri = next(_triplet_iter)
                except StopIteration:
                    _triplet_iter = iter(triplet_loader)
                    tri = next(_triplet_iter)

                x_ij = tri["x_ij"].float().to(device)
                x_jk = tri["x_jk"].float().to(device)
                x_ik = tri["x_ik"].float().to(device)
                if zero_metadata:
                    m_ij = torch.zeros(x_ij.shape[0], 7, device=device)
                    m_jk = torch.zeros_like(m_ij)
                    m_ik = torch.zeros_like(m_ij)
                else:
                    m_ij = tri["meta_ij"].float().to(device)
                    m_jk = tri["meta_jk"].float().to(device)
                    m_ik = tri["meta_ik"].float().to(device)

                d_ij, lv_ij = model(x_ij, m_ij)
                d_jk, _     = model(x_jk, m_jk)
                d_ik, _     = model(x_ik, m_ik)

                phi_ij = torch.atan2(d_ij[:, 1], d_ij[:, 0])  # (B, H, W)
                phi_jk = torch.atan2(d_jk[:, 1], d_jk[:, 0])
                phi_ik = torch.atan2(d_ik[:, 1], d_ik[:, 0])

                clos_inp = PhysicsLossInputs(
                    pred_a=d_ij,
                    sublook_b=tri["gold_ij"].float().to(device),
                    log_var=lv_ij,
                    full_look=tri["gold_ij"].float().to(device),
                    phi_ij=phi_ij,
                    phi_jk=phi_jk,
                    phi_ik=phi_ik,
                    closure_weight=tri["coh_ij"].float().to(device).squeeze(1),
                )
                clos_loss, clos_breakdown = criterion(clos_inp)

                optimizer.zero_grad()
                clos_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                for k, v in clos_breakdown.items():
                    totals[f"tri_{k}"] = totals.get(f"tri_{k}", 0.0) + float(v)

    return {k: v / max(n, 1) for k, v in totals.items()}


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FiLMUNet for InSAR denoising.")
    p.add_argument("--data_config", required=True)
    p.add_argument("--model_config", required=True)
    p.add_argument("--train_config", required=True)
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from.")

    # Ablation overrides — override the YAML loss weights at the CLI level
    p.add_argument("--loss_n2n",     type=float, default=None, help="Override loss_weights.n2n")
    p.add_argument("--loss_unc",     type=float, default=None, help="Override loss_weights.unc")
    p.add_argument("--loss_closure", type=float, default=None, help="Override loss_weights.closure")
    p.add_argument("--loss_temporal",type=float, default=None, help="Override loss_weights.temporal")
    p.add_argument("--loss_grad",    type=float, default=None, help="Override loss_weights.grad")
    p.add_argument("--epochs",       type=int,   default=None, help="Override num_epochs")
    # run_name creates a subdirectory under output_dir for isolated ablation checkpoints
    p.add_argument("--run_name",     default=None,
                   help="Subdirectory name under output_dir for this ablation run.")
    # V5 ablation: pass zeros as metadata (disables FiLM geometric conditioning)
    p.add_argument("--zero_film",    action="store_true",
                   help="Zero out metadata vector — tests without FiLM conditioning (V5).")
    p.add_argument("--triplets_manifest", type=str, default=None,
                   help="Parquet of triplets; enables closure loss via TripletTileDataset.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)
    all_configs = {"data": data_cfg, "model": model_cfg, "train": train_cfg}

    # ── Apply CLI ablation overrides ──
    lw = train_cfg.setdefault("loss_weights", {})
    for key, val in [
        ("n2n",     args.loss_n2n),
        ("unc",     args.loss_unc),
        ("closure", args.loss_closure),
        ("temporal",args.loss_temporal),
        ("grad",    args.loss_grad),
    ]:
        if val is not None:
            lw[key] = val
    if args.epochs is not None:
        train_cfg["num_epochs"] = args.epochs

    run_name = args.run_name or "filmUNet"
    run_ts   = datetime.now().strftime('%Y%m%d_%H%M')
    run_tag  = f"{run_name}_{run_ts}"

    set_seed(train_cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── data ──
    pairs_dir = Path(data_cfg["pairs_dir"])
    pair_dirs = discover_pair_dirs(pairs_dir)
    if not pair_dirs:
        print(f"No preprocessed pairs found in {pairs_dir}. "
              "Run scripts/preprocess_pairs.py first.")
        sys.exit(1)

    sp = data_cfg.get("temporal_split", {})
    train_dirs, val_dirs, _ = temporal_split(
        pair_dirs,
        train_frac=sp.get("train_frac", 0.70),
        val_frac=sp.get("val_frac", 0.15),
    )
    print(f"Pairs: {len(train_dirs)} train / {len(val_dirs)} val / "
          f"{len(pair_dirs) - len(train_dirs) - len(val_dirs)} test (held out)")

    tile_size = data_cfg.get("tile_size", 256)
    stride = data_cfg.get("stride", 128)
    min_coh = data_cfg.get("min_coherence", 0.15)
    in_ch = data_cfg.get("in_channels", 3)

    train_ds = InSARTileDataset(train_dirs, tile_size, stride, min_coh,
                                augment=True, in_channels=in_ch)
    val_ds = InSARTileDataset(val_dirs, tile_size, stride, min_coh,
                              augment=False, in_channels=in_ch)
    print(f"Tiles: {len(train_ds)} train / {len(val_ds)} val")

    num_workers = train_cfg.get("num_workers", 4)
    pin_memory = train_cfg.get("pin_memory", True) and device.type == "cuda"
    batch_size = train_cfg.get("batch_size", 8)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              drop_last=True, worker_init_fn=_worker_init)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory,
                            drop_last=False, worker_init_fn=_worker_init)

    # ── triplet loader (closure loss) ──
    triplet_loader = None
    if args.triplets_manifest and Path(args.triplets_manifest).exists():
        triplets_df = pd.read_parquet(args.triplets_manifest)
        pair_dir_map: Dict[str, Path] = {}
        for pd_dir in train_dirs + val_dirs:
            # Key = dir name = "{id_ref}__{id_sec}" (matches _pair_key output)
            pair_dir_map[pd_dir.name] = pd_dir
            # Also index by id_ref__id_sec from coreg_meta (same thing, but verify)
            meta_path = pd_dir / "coreg_meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        m = json.load(f)
                    id_ref = m.get("id_ref", "")
                    id_sec = m.get("id_sec", "")
                    if id_ref and id_sec:
                        pair_dir_map[f"{id_ref}__{id_sec}"] = pd_dir
                except Exception:
                    pass
        tri_ds = TripletTileDataset(
            triplets_df, pair_dir_map,
            tile_size=tile_size,
            stride=stride,
            min_coherence=min_coh,
        )
        log.info("TripletTileDataset: %d triplet-tiles from %d triplets",
                 len(tri_ds), len(triplets_df))
        triplet_loader = DataLoader(
            tri_ds, batch_size=batch_size,
            shuffle=True, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=True,
            worker_init_fn=_worker_init,
        )
    elif args.triplets_manifest:
        log.warning("--triplets_manifest specified but not found: %s", args.triplets_manifest)

    # ── model ──
    model = FiLMUNet(
        in_channels=model_cfg.get("in_channels", in_ch),
        metadata_dim=model_cfg.get("metadata_dim", 7),
        features=model_cfg.get("features", [32, 64, 128, 256]),
        embed_dim=model_cfg.get("embed_dim", 64),
    ).to(device)
    print(f"FiLMUNet params: {sum(p.numel() for p in model.parameters()):,}")

    # ── loss ──
    lw_cfg = train_cfg.get("loss_weights", {})
    criterion = InSARLoss(LossWeights(
        n2n=lw_cfg.get("n2n", 1.0),
        unc=lw_cfg.get("unc", 0.5),
        closure=lw_cfg.get("closure", 0.3),
        temporal=lw_cfg.get("temporal", 0.2),
        grad=lw_cfg.get("grad", 0.1),
    ))

    # ── optimiser + scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    num_epochs = train_cfg.get("num_epochs", 50)
    warmup = train_cfg.get("warmup_epochs", 2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup, eta_min=1e-6
    )

    # ── optional resume ──
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']} ({args.resume})")

    # ── optional wandb ──
    use_wandb = train_cfg.get("use_wandb", False)
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=train_cfg.get("wandb_project", "insar-dem-enhancement"),
                config=all_configs,
            )
        except ImportError:
            print("wandb not installed — skipping experiment tracking.")
            use_wandb = False

    base_out = Path(train_cfg.get("output_dir", "experiments/enhanced/checkpoints/film_unet"))
    out_dir  = base_out / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path("logs") / f"{run_tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)

    print(f"Run tag : {run_tag}")
    print(f"Ckpt dir: {out_dir}")
    print(f"Log file: {log_path}")
    log.info("Run tag: %s | Log: %s", run_tag, log_path)

    grad_clip = train_cfg.get("grad_clip", 1.0)
    save_every = train_cfg.get("save_every_n_epochs", 5)

    best_closure = float("inf")
    best_unwrap = 0.0

    # ── training loop ──
    for epoch in range(start_epoch, num_epochs):
        # Warmup LR
        if epoch < warmup:
            lr_scale = (epoch + 1) / warmup
            for pg in optimizer.param_groups:
                pg["lr"] = train_cfg.get("learning_rate", 1e-4) * lr_scale

        train_metrics = run_epoch(model, train_loader, criterion, optimizer,
                                  device, grad_clip, zero_metadata=args.zero_film,
                                  triplet_loader=triplet_loader)
        val_metrics = run_epoch(model, val_loader, criterion, None, device,
                                zero_metadata=args.zero_film)

        if epoch >= warmup:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:>3}/{num_epochs} | "
            f"train_loss={train_metrics.get('total', 0):.4f} "
            f"val_loss={val_metrics.get('total', 0):.4f} "
            f"train_closure={train_metrics.get('closure', 0):.4f} "
            f"val_closure={val_metrics.get('closure', 0):.4f} "
            f"lr={current_lr:.2e}"
        )

        if use_wandb:
            import wandb
            wandb.log({"epoch": epoch + 1, "lr": current_lr,
                       **{f"train/{k}": v for k, v in train_metrics.items()},
                       **{f"val/{k}": v for k, v in val_metrics.items()}})

        # Best-by-closure checkpoint (use train_closure: val never has triplets)
        train_closure = train_metrics.get("closure", float("inf"))
        if train_closure < best_closure:
            best_closure = train_closure
            save_checkpoint(
                out_dir / f"{run_tag}_best_closure.pt", model, optimizer, epoch,
                {"train_closure": train_closure}, all_configs,
            )
            print(f"  -> {run_tag}_best_closure.pt (closure={train_closure:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                out_dir / f"{run_tag}_epoch_{epoch+1:03d}.pt", model, optimizer, epoch,
                val_metrics, all_configs,
            )

    # Final checkpoint
    save_checkpoint(
        out_dir / f"{run_tag}_final.pt", model, optimizer, num_epochs - 1,
        val_metrics, all_configs,
    )
    print(f"Training complete. Checkpoints in {out_dir}/")

    # Save training summary for ablation result collection
    summary = {
        "run_tag": run_tag,
        "run_name": run_name,
        "num_epochs": num_epochs,
        "zero_film": args.zero_film,
        "loss_weights": all_configs["train"]["loss_weights"],
        "best_train_closure": best_closure,
        "final_val_metrics": val_metrics,
        "git_hash": git_hash(),
    }
    summary_path = out_dir / f"{run_tag}_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary: {summary_path}")

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
