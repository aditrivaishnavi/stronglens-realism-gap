"""Shared score-blind G_rec condition-cell selection."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

ROLE_SEED = "stronglens-latest-detector-evaluation-v1"
HOSTS_PER_RETAINED_CONDITION_CELL = 500
MIN_SPATIAL_GROUPS_PER_CONDITION_CELL = 100
MAX_HOSTS_PER_SPATIAL_GROUP_PER_CELL = 5
PSF_NODES = np.array([0.90, 1.05, 1.20, 1.35, 1.50, 1.65, 1.80])
DEPTH_NODES = np.array([22.5, 23.0, 23.5, 24.0, 24.5])

REQUIRED_COLUMNS = {
    "galaxy_id",
    "evaluation_spatial_group_id",
    "psfsize_r",
    "psfdepth_r",
}


def stable_hash(value: str) -> str:
    return hashlib.sha256(f"{ROLE_SEED}:{value}".encode()).hexdigest()


def nearest(values: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    boundaries = (nodes[:-1] + nodes[1:]) / 2.0
    return nodes[np.searchsorted(boundaries, values, side="left")]


def assign_condition_cells(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the frozen PSF/depth condition cells from raw catalogue values."""
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"G_rec candidate frame lacks columns: {missing}")
    if frame["galaxy_id"].astype(str).duplicated().any():
        raise ValueError("G_rec candidate galaxy IDs are not unique")
    psfsize = frame["psfsize_r"].to_numpy(dtype=float)
    psfdepth = frame["psfdepth_r"].to_numpy(dtype=float)
    if (
        not np.isfinite(psfsize).all()
        or not np.isfinite(psfdepth).all()
        or (psfsize <= 0).any()
        or (psfdepth <= 0).any()
    ):
        raise ValueError("G_rec candidate observing metadata are invalid")
    result = frame.copy()
    result["depth_mag"] = 22.5 - 2.5 * np.log10(5.0 / np.sqrt(psfdepth))
    result["psf_bin"] = nearest(psfsize, PSF_NODES)
    result["depth_bin"] = nearest(
        result["depth_mag"].to_numpy(dtype=float), DEPTH_NODES
    )
    return result


def select_grec_hosts(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Apply the frozen support, group-cap, and hash-order selection."""
    candidates = assign_condition_cells(frame)
    support = (
        candidates.groupby(["psf_bin", "depth_bin"])
        .agg(
            candidate_hosts=("galaxy_id", "size"),
            spatial_groups=("evaluation_spatial_group_id", "nunique"),
        )
        .reset_index()
    )
    capacity = (
        candidates.groupby(
            ["psf_bin", "depth_bin", "evaluation_spatial_group_id"]
        )
        .size()
        .clip(upper=MAX_HOSTS_PER_SPATIAL_GROUP_PER_CELL)
        .groupby(level=[0, 1])
        .sum()
        .rename("capped_host_capacity")
        .reset_index()
    )
    support = support.merge(capacity, on=["psf_bin", "depth_bin"], how="left")
    retained_cells = support[
        support["candidate_hosts"].ge(HOSTS_PER_RETAINED_CONDITION_CELL)
        & support["spatial_groups"].ge(MIN_SPATIAL_GROUPS_PER_CONDITION_CELL)
        & support["capped_host_capacity"].ge(
            HOSTS_PER_RETAINED_CONDITION_CELL
        )
    ][["psf_bin", "depth_bin"]]
    candidates = candidates.merge(
        retained_cells.assign(retained_condition_cell=True),
        on=["psf_bin", "depth_bin"],
        how="left",
    )
    candidates["retained_condition_cell"] = candidates[
        "retained_condition_cell"
    ].fillna(False)
    candidates["host_order"] = candidates.apply(
        lambda row: stable_hash(
            f"G_rec:{row['psf_bin']}:{row['depth_bin']}:{row['galaxy_id']}"
        ),
        axis=1,
    )
    selectable = candidates[candidates["retained_condition_cell"]].sort_values(
        [
            "psf_bin",
            "depth_bin",
            "evaluation_spatial_group_id",
            "host_order",
        ]
    )
    selectable["within_group_rank"] = selectable.groupby(
        ["psf_bin", "depth_bin", "evaluation_spatial_group_id"]
    ).cumcount()
    selectable = selectable[
        selectable["within_group_rank"].lt(
            MAX_HOSTS_PER_SPATIAL_GROUP_PER_CELL
        )
    ]
    selected = (
        selectable.sort_values(
            ["psf_bin", "depth_bin", "within_group_rank", "host_order"]
        )
        .groupby(["psf_bin", "depth_bin"], group_keys=False)
        .head(HOSTS_PER_RETAINED_CONDITION_CELL)
        .copy()
    )
    counts = selected.groupby(["psf_bin", "depth_bin"]).size()
    if not counts.eq(HOSTS_PER_RETAINED_CONDITION_CELL).all():
        raise AssertionError("retained G_rec cell does not have exactly 500 hosts")
    selected_group_counts = selected.groupby(["psf_bin", "depth_bin"])[
        "evaluation_spatial_group_id"
    ].nunique()
    if not selected_group_counts.ge(
        MIN_SPATIAL_GROUPS_PER_CONDITION_CELL
    ).all():
        raise AssertionError("retained G_rec cell has inadequate spatial groups")
    maximum_group = (
        selected.groupby(
            ["psf_bin", "depth_bin", "evaluation_spatial_group_id"]
        )
        .size()
        .max()
    )
    if maximum_group > MAX_HOSTS_PER_SPATIAL_GROUP_PER_CELL:
        raise AssertionError("one spatial group contributes too many cell hosts")
    return selected, support, selected_group_counts
