"""
DEM quality metrics for evaluating baseline and learning-assisted methods.
"""

import numpy as np


def rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    diff = pred - target
    return float(np.sqrt(np.mean(diff ** 2)))


def mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    diff = np.abs(pred - target)
    return float(np.mean(diff))


def bias(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    diff = pred - target
    return float(np.mean(diff))


def nmad(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Normalised Median Absolute Deviation: 1.4826 × median(|e − median(e)|)."""
    if mask is not None:
        pred, target = pred[mask], target[mask]
    e = pred.astype(np.float64) - target.astype(np.float64)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(e - np.median(e))))

