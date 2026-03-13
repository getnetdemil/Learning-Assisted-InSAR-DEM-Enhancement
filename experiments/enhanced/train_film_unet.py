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
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

# ── repo-relative import fix ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from models.film_unet import FiLMUNet
from losses.physics_losses import InSARLoss, LossWeights, PhysicsLossInputs


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
        ifg_path = pair_dir / "ifg_goldstein.tif"
        coh_path = pair_dir / "coherence.tif"

        with rasterio.open(ifg_path) as src:
            data = src.read(window=window)  # (2, T, T) Re+Im or (1, T, T) complex
            if data.shape[0] == 1:
                # Complex int — convert
                cplx = data[0].astype(np.complex64)
                re = cplx.real
                im = cplx.imag
            else:
                re, im = data[0].astype(np.float32), data[1].astype(np.float32)

        with rasterio.open(coh_path) as src:
            coh = src.read(1, window=window).astype(np.float32)

        return re, im, coh

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
    def _augment(re, im, coh):
        """Physically-safe augmentations: rotations, flips, global phase offset."""
        k = random.randint(0, 3)
        re = np.rot90(re, k).copy()
        im = np.rot90(im, k).copy()
        coh = np.rot90(coh, k).copy()
        if random.random() > 0.5:
            re = np.fliplr(re).copy()
            im = np.fliplr(im).copy()
            coh = np.fliplr(coh).copy()
        # Global phase offset (does not affect closure or N2N residuals)
        phi_offset = random.uniform(-np.pi, np.pi)
        cos_o, sin_o = np.cos(phi_offset), np.sin(phi_offset)
        re_new = re * cos_o - im * sin_o
        im_new = re * sin_o + im * cos_o
        # Amplitude jitter ±10%
        scale = random.uniform(0.9, 1.1)
        return re_new * scale, im_new * scale, coh

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Optional[dict]:
        pair_dir, r, c = self.tiles[idx]
        re, im, coh = self._load_tile(pair_dir, r, c)

        if coh.mean() < self.min_coherence:
            # Return a random other tile to avoid None in DataLoader
            alt = random.randint(0, len(self.tiles) - 1)
            return self[alt]

        if self.augment:
            re, im, coh = self._augment(re, im, coh)

        meta = self._load_meta(pair_dir)
        phi = np.arctan2(im, re).astype(np.float32)

        if self.in_channels == 3:
            x = np.stack([re, im, coh], axis=0)
        else:
            x = np.stack([re, im], axis=0)

        return {
            "x": torch.from_numpy(x),
            "metadata": torch.from_numpy(meta),
            "phi": torch.from_numpy(phi),
            "pair_id": str(pair_dir.name),
        }


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
) -> dict:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = {}
    n = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            x = batch["x"].float().to(device)          # (B, C, H, W)
            meta = batch["metadata"].float().to(device) # (B, 7)

            denoised, log_var = model(x, meta)

            # For N2N: split the batch in half as proxy sub-look pair
            # (during actual training, pass real sub-look A/B pairs)
            half = x.shape[0] // 2
            if half < 1:
                continue

            # Use first half's prediction vs second half's input as N2N target
            inp = PhysicsLossInputs(
                pred_a=denoised[:half, :2],
                sublook_b=x[half:half * 2, :2],
                log_var=log_var[:half],
                full_look=x[:half, :2],
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

    return {k: v / max(n, 1) for k, v in totals.items()}


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FiLMUNet for InSAR denoising.")
    p.add_argument("--data_config", required=True)
    p.add_argument("--model_config", required=True)
    p.add_argument("--train_config", required=True)
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)
    all_configs = {"data": data_cfg, "model": model_cfg, "train": train_cfg}

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
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory,
                            drop_last=False)

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
        ckpt = torch.load(args.resume, map_location=device)
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

    out_dir = Path(train_cfg.get("output_dir", "experiments/enhanced/checkpoints/film_unet"))
    out_dir.mkdir(parents=True, exist_ok=True)

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
                                  device, grad_clip)
        val_metrics = run_epoch(model, val_loader, criterion, None, device)

        if epoch >= warmup:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:>3}/{num_epochs} | "
            f"train_loss={train_metrics.get('total', 0):.4f} "
            f"val_loss={val_metrics.get('total', 0):.4f} "
            f"val_closure={val_metrics.get('closure', 0):.4f} "
            f"lr={current_lr:.2e}"
        )

        if use_wandb:
            import wandb
            wandb.log({"epoch": epoch + 1, "lr": current_lr,
                       **{f"train/{k}": v for k, v in train_metrics.items()},
                       **{f"val/{k}": v for k, v in val_metrics.items()}})

        # Best-by-closure checkpoint
        val_closure = val_metrics.get("closure", float("inf"))
        if val_closure < best_closure:
            best_closure = val_closure
            save_checkpoint(
                out_dir / "best_closure.pt", model, optimizer, epoch,
                {"val_closure": val_closure}, all_configs,
            )
            print(f"  -> best_closure.pt (closure={val_closure:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                out_dir / f"epoch_{epoch+1:03d}.pt", model, optimizer, epoch,
                val_metrics, all_configs,
            )

    # Final checkpoint
    save_checkpoint(
        out_dir / "final.pt", model, optimizer, num_epochs - 1,
        val_metrics, all_configs,
    )
    print(f"Training complete. Checkpoints in {out_dir}/")

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
