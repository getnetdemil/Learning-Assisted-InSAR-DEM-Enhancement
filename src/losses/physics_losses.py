"""
Self-supervised physics-informed loss functions for FiLMUNet training.

Five components, all operating on wrapped complex interferograms:

1. L_n2n      — Noise2Noise L1 between two independent sub-look views
2. L_unc      — Heteroscedastic NLL (encourages calibrated uncertainty)
3. L_closure  — Triplet closure-consistency weighted by coherence/uncertainty
4. L_temporal — SBAS-like stack inversion residual
5. L_grad     — Fringe-preservation gradient loss

Combined:
    L = λ_n2n * L_n2n
      + λ_unc * L_unc
      + λ_closure * L_closure
      + λ_temporal * L_temporal
      + λ_grad * L_grad

Phase tensors throughout are 2-channel (Re, Im).
The wrapped phase of a 2ch tensor x is atan2(x[:,1], x[:,0]).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F


# ─── helpers ──────────────────────────────────────────────────────────────────

def _phase(x: torch.Tensor) -> torch.Tensor:
    """Wrapped phase from 2ch Re/Im tensor. Returns (B, H, W)."""
    return torch.atan2(x[:, 1], x[:, 0])


def _wrap(phi: torch.Tensor) -> torch.Tensor:
    """Wrap phase to [-π, π]."""
    return torch.atan2(torch.sin(phi), torch.cos(phi))


def _complex_mul_conj(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply a by conjugate(b), both 2ch Re/Im.  Returns 2ch Re/Im."""
    re = a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]
    im = a[:, 1] * b[:, 0] - a[:, 0] * b[:, 1]
    return torch.stack([re, im], dim=1)


# ─── individual losses ────────────────────────────────────────────────────────

def noise2noise_loss(
    pred_a: torch.Tensor,
    target_b: torch.Tensor,
) -> torch.Tensor:
    """
    Noise2Noise L1 loss between two independent sub-look predictions.

    In training, the model sees sub-look A and predicts the full-look phase;
    the loss is computed against the noisy sub-look B estimate (also 2ch Re/Im).

    Args:
        pred_a:   (B, 2, H, W) — model output for sub-look A input
        target_b: (B, 2, H, W) — sub-look B interferogram (noisy target)
    Returns:
        Scalar loss.
    """
    # Phase-domain L1 (more robust than Re/Im L1 for wrapped quantities)
    phi_pred = _phase(pred_a)
    phi_tgt = _phase(target_b)
    return _wrap(phi_pred - phi_tgt).abs().mean()


def uncertainty_nll_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    log_var: torch.Tensor,
) -> torch.Tensor:
    """
    Heteroscedastic negative log-likelihood for calibrated uncertainty.

    L_unc = 0.5 * exp(-log_var) * |φ_pred − φ_target|² + 0.5 * log_var

    Args:
        pred:    (B, 2, H, W) — denoised Re/Im from model
        target:  (B, 2, H, W) — noisy target Re/Im
        log_var: (B, 1, H, W) — predicted log-variance from model
    Returns:
        Scalar loss.
    """
    phi_pred = _phase(pred)
    phi_tgt = _phase(target)
    residual = _wrap(phi_pred - phi_tgt)          # (B, H, W)
    lv = log_var.squeeze(1)                        # (B, H, W)
    return (0.5 * torch.exp(-lv) * residual.pow(2) + 0.5 * lv).mean()


def closure_loss(
    phi_ij: torch.Tensor,
    phi_jk: torch.Tensor,
    phi_ik: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Triplet closure-consistency loss.

    L_closure = mean_p [ w(p) * (1 − cos(wrap(φ̂_ij + φ̂_jk − φ̂_ik))) ]

    For a consistent triplet, wrap(φ_ij + φ_jk − φ_ik) = 0 and cos → 1.

    Args:
        phi_ij: (B, H, W) — wrapped phase for pair (i,j)
        phi_jk: (B, H, W) — wrapped phase for pair (j,k)
        phi_ik: (B, H, W) — wrapped phase for pair (i,k)
        weight: (B, H, W) optional coherence/uncertainty weight (default: ones)
    Returns:
        Scalar loss in [0, 2].
    """
    closure = _wrap(phi_ij + phi_jk - phi_ik)     # (B, H, W)
    term = 1.0 - torch.cos(closure)               # 0 when perfect
    if weight is not None:
        return (weight * term).mean()
    return term.mean()


def temporal_consistency_loss(
    phi_hat: torch.Tensor,
    A: torch.Tensor,
    coh_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    SBAS-like stack inversion residual.

    Minimises ‖W(Ax* − φ̂)‖² where x* is the LSQR solution of W·A·x = W·φ̂.

    In practice during training we approximate this with the closure-phase
    residual over the mini-batch pairs:
        L_temporal = ‖W * (A @ lstsq(A, phi_hat) − phi_hat)‖²

    Args:
        phi_hat:      (P, N) — stack of P unwrapped-like phase maps, N pixels
        A:            (P, T) — SBAS design matrix (pairs × epochs)
        coh_weights:  (P, N) optional coherence weights (default: ones)
    Returns:
        Scalar loss.
    """
    # Weighted least-squares: x* = (AᵀWA)⁻¹ AᵀW φ̂
    if coh_weights is None:
        coh_weights = torch.ones_like(phi_hat)

    # Column-wise least-squares (solve per pixel)
    # A: (P, T), W: diagonal from coh_weights mean over pixels (P,)
    W_diag = coh_weights.mean(dim=1)               # (P,) — one weight per pair
    W_sqrt = W_diag.sqrt().unsqueeze(1)            # (P, 1)
    Aw = A * W_sqrt                                # (P, T) — weighted design matrix
    phi_w = phi_hat * W_sqrt                       # (P, N)

    # Solve: x_star = lstsq(Aw, phi_w)
    x_star = torch.linalg.lstsq(Aw, phi_w).solution  # (T, N)

    # Residual
    residual = phi_w - Aw @ x_star                 # (P, N)
    return (residual ** 2).mean()


def gradient_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Fringe-preservation gradient loss on the phase image.

    L_grad = mean(|∇φ_pred − ∇φ_target|)

    Encourages the model to preserve fine fringe structure rather than
    over-smoothing.

    Args:
        pred:   (B, 2, H, W) — denoised Re/Im
        target: (B, 2, H, W) — reference Re/Im (e.g., full-look noisy)
    Returns:
        Scalar loss.
    """
    phi_pred = _phase(pred).unsqueeze(1)     # (B, 1, H, W)
    phi_tgt = _phase(target).unsqueeze(1)

    # Sobel-like gradient via finite differences
    dy_pred = phi_pred[:, :, 1:, :] - phi_pred[:, :, :-1, :]
    dx_pred = phi_pred[:, :, :, 1:] - phi_pred[:, :, :, :-1]
    dy_tgt = phi_tgt[:, :, 1:, :] - phi_tgt[:, :, :-1, :]
    dx_tgt = phi_tgt[:, :, :, 1:] - phi_tgt[:, :, :, :-1]

    return (_wrap(dy_pred - dy_tgt).abs().mean() +
            _wrap(dx_pred - dx_tgt).abs().mean()) / 2.0


# ─── combined loss ────────────────────────────────────────────────────────────

@dataclass
class LossWeights:
    n2n: float = 1.0
    unc: float = 0.5
    closure: float = 0.3
    temporal: float = 0.2
    grad: float = 0.1


@dataclass
class PhysicsLossInputs:
    """Container for all inputs needed by InSARLoss.forward()."""
    # Sub-look views for N2N (both required)
    pred_a: torch.Tensor           # (B, 2, H, W) model output on sub-look A
    sublook_b: torch.Tensor        # (B, 2, H, W) sub-look B noisy target
    log_var: torch.Tensor          # (B, 1, H, W) predicted log-variance
    # Full-look (or sub-look A) for gradient loss
    full_look: torch.Tensor        # (B, 2, H, W) full-look interferogram
    # Triplet closure (optional — skip if not available in batch)
    phi_ij: Optional[torch.Tensor] = None   # (B, H, W)
    phi_jk: Optional[torch.Tensor] = None   # (B, H, W)
    phi_ik: Optional[torch.Tensor] = None   # (B, H, W)
    closure_weight: Optional[torch.Tensor] = None  # (B, H, W)
    # SBAS temporal (optional)
    phi_stack: Optional[torch.Tensor] = None   # (P, N) stack phases
    sbas_A: Optional[torch.Tensor] = None      # (P, T) design matrix
    coh_stack: Optional[torch.Tensor] = None   # (P, N) coherence weights


class InSARLoss(torch.nn.Module):
    """
    Combined self-supervised InSAR loss.

    Usage:
        criterion = InSARLoss(weights=LossWeights(n2n=1.0, unc=0.5, closure=0.3))
        loss, breakdown = criterion(inputs)
    """

    def __init__(self, weights: Optional[LossWeights] = None):
        super().__init__()
        self.w = weights or LossWeights()

    def forward(self, inp: PhysicsLossInputs) -> tuple[torch.Tensor, dict]:
        breakdown: dict = {}

        # 1. N2N
        l_n2n = noise2noise_loss(inp.pred_a, inp.sublook_b)
        breakdown["n2n"] = l_n2n.item()

        # 2. Uncertainty NLL
        l_unc = uncertainty_nll_loss(inp.pred_a, inp.sublook_b, inp.log_var)
        breakdown["unc"] = l_unc.item()

        # 3. Closure (optional)
        l_closure = torch.tensor(0.0, device=inp.pred_a.device)
        if inp.phi_ij is not None:
            l_closure = closure_loss(inp.phi_ij, inp.phi_jk, inp.phi_ik,
                                     inp.closure_weight)
        breakdown["closure"] = l_closure.item()

        # 4. Temporal (optional)
        l_temporal = torch.tensor(0.0, device=inp.pred_a.device)
        if inp.phi_stack is not None and inp.sbas_A is not None:
            l_temporal = temporal_consistency_loss(inp.phi_stack, inp.sbas_A,
                                                   inp.coh_stack)
        breakdown["temporal"] = l_temporal.item()

        # 5. Gradient
        l_grad = gradient_loss(inp.pred_a, inp.full_look)
        breakdown["grad"] = l_grad.item()

        total = (self.w.n2n * l_n2n
                 + self.w.unc * l_unc
                 + self.w.closure * l_closure
                 + self.w.temporal * l_temporal
                 + self.w.grad * l_grad)

        breakdown["total"] = total.item()
        return total, breakdown
