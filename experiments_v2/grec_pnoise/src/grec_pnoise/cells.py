from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .contracts import ContractError


def nearest_node_lower_midpoint(
    values: np.ndarray, nodes: np.ndarray
) -> np.ndarray:
    """Nearest-node assignment with exact midpoint ties sent to lower node."""
    values = np.asarray(values, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 1 or nodes.size < 2 or not np.all(np.diff(nodes) > 0):
        raise ContractError("condition nodes must be finite and strictly increasing")
    if not np.isfinite(values).all() or not np.isfinite(nodes).all():
        raise ContractError("condition values and nodes must be finite")
    boundaries = (nodes[:-1] + nodes[1:]) / 2.0
    return nodes[np.searchsorted(boundaries, values, side="left")]


def depth_mag_from_psfdepth(psfdepth_r: np.ndarray) -> np.ndarray:
    """DR10 Tractor PSF inverse variance (nmgy^-2) to 5-sigma AB depth."""
    inverse_variance = np.asarray(psfdepth_r, dtype=np.float64)
    if not np.isfinite(inverse_variance).all() or (inverse_variance <= 0).any():
        raise ContractError("psfdepth_r must be finite positive inverse nmgy^2")
    return 22.5 - 2.5 * np.log10(5.0 / np.sqrt(inverse_variance))


def derive_condition_cells(
    psfsize_r: np.ndarray,
    psfdepth_r: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cells = config["cells"]
    psf = np.asarray(psfsize_r, dtype=np.float64)
    if not np.isfinite(psf).all():
        raise ContractError("psfsize_r must be finite")
    if (
        (psf < float(cells["psfsize_r_valid_min_arcsec"])).any()
        or (psf > float(cells["psfsize_r_valid_max_arcsec"])).any()
    ):
        raise ContractError("psfsize_r lies outside preregistered valid range")
    depth = depth_mag_from_psfdepth(psfdepth_r)
    if (
        (depth < float(cells["depth_mag_valid_min"])).any()
        or (depth > float(cells["depth_mag_valid_max"])).any()
    ):
        raise ContractError("derived depth lies outside preregistered valid range")
    psf_nodes = np.asarray(cells["psf_nodes_arcsec"], dtype=np.float64)
    depth_nodes = np.asarray(cells["depth_nodes_mag"], dtype=np.float64)
    return (
        nearest_node_lower_midpoint(psf, psf_nodes),
        nearest_node_lower_midpoint(depth, depth_nodes),
        depth,
    )
