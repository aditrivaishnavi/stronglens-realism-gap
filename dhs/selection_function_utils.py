"""Utilities for selection-function grid execution and aggregation."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import beta as beta_dist


def m5_from_psfdepth(psfdepth_nmgy_invvar: np.ndarray) -> np.ndarray:
    """Approximate 5-sigma point-source depth in AB mag from psfdepth.

    In Legacy Surveys sweeps/tractor catalogs, PSFDEPTH_* fields are in
    (nanomaggies^-2) and represent inverse variance of PSF flux.

    sigma_flux = 1/sqrt(psfdepth)
    flux_5sig = 5*sigma_flux
    mag_5sig = 22.5 - 2.5*log10(flux_5sig)
    """
    psfdepth = np.asarray(psfdepth_nmgy_invvar, dtype=np.float64)
    out = np.full_like(psfdepth, np.nan, dtype=np.float64)
    ok = np.isfinite(psfdepth) & (psfdepth > 0)
    sigma = np.empty_like(psfdepth, dtype=np.float64)
    sigma[ok] = 1.0 / np.sqrt(psfdepth[ok])
    flux5 = 5.0 * sigma[ok]
    out[ok] = 22.5 - 2.5 * np.log10(flux5)
    return out


def bayes_binomial_interval(k: int, n: int, level: float = 0.68, prior: str = "jeffreys") -> Tuple[float, float]:
    """Bayesian binomial credible interval using scipy Beta distribution.

    prior:
      - "jeffreys": Beta(0.5, 0.5)
      - "uniform":  Beta(1.0, 1.0)
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    if prior == "jeffreys":
        a0, b0 = 0.5, 0.5
    elif prior == "uniform":
        a0, b0 = 1.0, 1.0
    else:
        raise ValueError(f"Unknown prior: {prior}")

    a = a0 + k
    b = b0 + (n - k)
    alpha = (1.0 - level) / 2.0
    lo = float(beta_dist.ppf(alpha, a, b))
    hi = float(beta_dist.ppf(1.0 - alpha, a, b))
    return lo, hi


def build_edges(start: float, stop: float, step: float) -> np.ndarray:
    vals = np.arange(start, stop + 1e-9, step, dtype=np.float64)
    # interpret as bin centers; edges are half-steps
    half = step / 2.0
    edges = np.concatenate([[vals[0] - half], vals + half])
    return edges


def digitize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return bin index in [0, nbins-1], or -1 if out of range."""
    idx = np.digitize(values, edges) - 1
    nb = len(edges) - 1
    idx[(idx < 0) | (idx >= nb)] = -1
    return idx
