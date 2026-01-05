"""
Calibration metrics for binary classifiers (checklist 4.5).

- ECE: Expected Calibration Error (weighted average of |acc(b) - conf(b)| over bins).
- MCE: Maximum Calibration Error (max over bins).
- Reliability curve: per-bin accuracy and mean confidence for plotting.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class ReliabilityCurve:
    """Per-bin stats for reliability diagram."""
    bin_edges: np.ndarray       # (n_bins+1,) edges
    bin_acc: np.ndarray         # (n_bins,) accuracy in each bin
    bin_conf: np.ndarray        # (n_bins,) mean predicted prob in each bin
    bin_counts: np.ndarray     # (n_bins,) sample count per bin


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    equal_mass: bool = True,
) -> float:
    """
    Expected Calibration Error.

    ECE = sum_b (n_b / n) * |acc(b) - conf(b)|.
    Uses equal-frequency bins if equal_mass else equal-width.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    if equal_mass:
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.percentile(y_prob, quantiles * 100)
        bin_edges[0], bin_edges[-1] = 0.0, 1.0
    else:
        bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if b < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        n_b = mask.sum()
        if n_b == 0:
            continue
        acc_b = y_true[mask].mean()
        conf_b = y_prob[mask].mean()
        ece += (n_b / n) * abs(acc_b - conf_b)
    return float(ece)


def compute_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    equal_mass: bool = True,
) -> float:
    """Maximum Calibration Error: max over bins of |acc(b) - conf(b)|."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    if equal_mass:
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.percentile(y_prob, quantiles * 100)
        bin_edges[0], bin_edges[-1] = 0.0, 1.0
    else:
        bin_edges = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if b < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        n_b = mask.sum()
        if n_b == 0:
            continue
        acc_b = y_true[mask].mean()
        conf_b = y_prob[mask].mean()
        mce = max(mce, abs(acc_b - conf_b))
    return float(mce)


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    equal_mass: bool = True,
) -> ReliabilityCurve:
    """Per-bin accuracy and mean confidence for reliability diagram."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    if equal_mass:
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.percentile(y_prob, quantiles * 100)
        bin_edges[0], bin_edges[-1] = 0.0, 1.0
    else:
        bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_acc = np.full(n_bins, np.nan)
    bin_conf = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if b < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        n_b = mask.sum()
        bin_counts[b] = n_b
        if n_b > 0:
            bin_acc[b] = y_true[mask].mean()
            bin_conf[b] = y_prob[mask].mean()
    return ReliabilityCurve(
        bin_edges=bin_edges,
        bin_acc=bin_acc,
        bin_conf=bin_conf,
        bin_counts=bin_counts,
    )
