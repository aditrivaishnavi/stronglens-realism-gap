from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .contracts import ContractError, seed_from_key


def _validate_binary_scores(
    manifest: pd.DataFrame,
    scores: pd.DataFrame,
    identity: str,
) -> pd.DataFrame:
    required_manifest = {
        identity,
        "scene_id",
        "host_galaxy_id",
        "host_spatial_group_id",
        "source_parameter_id",
        "lens_parameter_id",
        "render_config_id",
        "theta_e_arcsec",
        "psf_bin",
        "depth_bin",
        "host_scientific_weight",
        "scene_within_host_weight",
    }
    required_scores = {identity, "detected_at_frozen_threshold"}
    if required_manifest - set(manifest) or required_scores - set(scores):
        raise ContractError("inference input lacks registered identity/weight columns")
    if manifest[identity].astype(str).duplicated().any() or scores[
        identity
    ].astype(str).duplicated().any():
        raise ContractError("inference identities must be unique")
    if set(manifest[identity].astype(str)) != set(scores[identity].astype(str)):
        raise ContractError("scores do not exactly cover the frozen manifest")
    joined = manifest.merge(scores, on=identity, validate="one_to_one", sort=False)
    decisions = joined["detected_at_frozen_threshold"]
    if not decisions.isin([True, False, 0, 1]).all():
        raise ContractError("threshold decisions are not binary")
    joined["decision"] = decisions.astype(np.float64)
    return joined


def _global_spatial_multiplier_bootstrap(
    host_radius: pd.DataFrame,
    *,
    draws: int,
    seed_namespace: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    radii = np.sort(host_radius["theta_e_arcsec"].unique())
    cells = host_radius[["psf_bin", "depth_bin"]].drop_duplicates().sort_values(
        ["psf_bin", "depth_bin"]
    )
    rng = np.random.Generator(
        np.random.PCG64DXSM(seed_from_key(seed_namespace, draws))
    )
    groups = np.sort(host_radius["host_spatial_group_id"].astype(str).unique())
    group_index = {group_id: index for index, group_id in enumerate(groups)}
    host_frames: list[pd.DataFrame] = []
    point_cells: list[np.ndarray] = []
    for cell in cells.itertuples(index=False):
        frame = host_radius[
            host_radius["psf_bin"].eq(cell.psf_bin)
            & host_radius["depth_bin"].eq(cell.depth_bin)
        ]
        if frame.groupby("host_galaxy_id")["host_spatial_group_id"].nunique().max() != 1:
            raise ContractError("one host crosses spatial groups")
        matrix = frame.pivot(
            index=[
                "host_galaxy_id",
                "host_spatial_group_id",
            ],
            columns="theta_e_arcsec",
            values="value",
        )
        if list(matrix.columns) != list(radii) or matrix.isna().any().any():
            raise ContractError(
                "each host must have one value at every registered radius"
            )
        host_frames.append(matrix)
        point_cells.append(matrix.to_numpy().mean(axis=0))
    point_cell_array = np.stack(point_cells)
    raw_cell_draws = np.empty(
        (draws, len(host_frames), radii.size), dtype=np.float64
    )
    for draw in range(draws):
        # One strictly-positive global draw per spatial group. The same multiplier
        # follows that group into every cell/radius where it occurs.
        multipliers = rng.exponential(scale=1.0, size=len(groups))
        for cell_index, matrix in enumerate(host_frames):
            host_groups = matrix.index.get_level_values(
                "host_spatial_group_id"
            ).astype(str)
            weights = np.array(
                [multipliers[group_index[group_id]] for group_id in host_groups]
            )
            raw_cell_draws[draw, cell_index] = np.average(
                matrix.to_numpy(), axis=0, weights=weights
            )
    # Ratio multiplier estimates have small finite-sample bias when group sizes
    # differ. Recenter every cell/radius to the exact equal-host point estimator;
    # this leaves the complete cross-cell/radius covariance unchanged.
    cell_draws = (
        raw_cell_draws
        - raw_cell_draws.mean(axis=0, keepdims=True)
        + point_cell_array[None, :, :]
    )
    by_radius = cell_draws.mean(axis=1)
    overall = by_radius.mean(axis=1)
    shared_group_cells = (
        host_radius[["host_spatial_group_id", "psf_bin", "depth_bin"]]
        .drop_duplicates()
        .groupby("host_spatial_group_id")
        .size()
    )
    return overall, by_radius, cell_draws, {
        "primary_sampling_unit": "host_spatial_group",
        "multiplier_distribution": "iid_exponential_mean_one",
        "one_global_multiplier_per_group_per_replicate": True,
        "same_multiplier_applied_across_cells_and_radii": True,
        "estimator_weighting": "equal_host_within_cell_equal_cell_overall",
        "draws_recentered_to_equal_host_point_estimator": True,
        "maximum_absolute_recentering_error": float(
            np.max(np.abs(cell_draws.mean(axis=0) - point_cell_array))
        ),
        "spatial_groups_crossing_cells": int((shared_group_cells > 1).sum()),
        "cell_host_counts_preserved_in_point_estimator": True,
        "cell_weights_preserved": True,
        "radius_rows_kept_together_within_host": True,
    }


def _interval(values: np.ndarray, confidence: float) -> list[float]:
    tail = (1.0 - confidence) / 2.0
    return [float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))]


def analyse_grec(
    parent_manifest: pd.DataFrame,
    parent_scores: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    bootstrap_draws: int | None = None,
) -> dict[str, Any]:
    joined = _validate_binary_scores(parent_manifest, parent_scores, "grec_row_id")
    per_host_radius = (
        joined.groupby(
            [
                "host_galaxy_id",
                "host_spatial_group_id",
                "psf_bin",
                "depth_bin",
                "theta_e_arcsec",
            ],
            as_index=False,
        )["decision"]
        .mean()
        .rename(columns={"decision": "value"})
    )
    if not joined.groupby("host_galaxy_id")["grec_row_id"].size().eq(
        len(config["grec"]["radii_arcsec"])
    ).all():
        raise ContractError("G_rec host repetition differs from the x11 contract")
    cell_radius = (
        per_host_radius.groupby(
            ["psf_bin", "depth_bin", "theta_e_arcsec"], as_index=False
        )["value"]
        .mean()
        .rename(columns={"value": "recovery"})
    )
    radius = (
        cell_radius.groupby("theta_e_arcsec", as_index=False)["recovery"].mean()
    )
    estimate = float(radius["recovery"].mean())
    draws = int(bootstrap_draws or config["inference"]["bootstrap_draws"])
    overall_draws, radius_draws, _, bootstrap_design = (
        _global_spatial_multiplier_bootstrap(
        per_host_radius,
        draws=draws,
        seed_namespace=str(config["inference"]["bootstrap_seed_namespace"])
        + ":grec",
        )
    )
    confidence = float(config["inference"]["confidence_level"])
    radius_values = radius["recovery"].to_numpy()
    max_deviation = np.max(np.abs(radius_draws - radius_values[None, :]), axis=1)
    simultaneous_half_width = float(np.quantile(max_deviation, confidence))
    return {
        "estimand": "equal_retained_cell_then_equal_host_recovery",
        "estimate": estimate,
        "confidence_interval": _interval(overall_draws, confidence),
        "bootstrap_draws": draws,
        "bootstrap_design": bootstrap_design,
        "distinct_hosts": int(joined["host_galaxy_id"].nunique()),
        "distinct_spatial_groups": int(
            joined["host_spatial_group_id"].nunique()
        ),
        "radius": [
            {
                "theta_e_arcsec": float(theta),
                "estimate": float(value),
                "simultaneous_interval": [
                    float(value - simultaneous_half_width),
                    float(value + simultaneous_half_width),
                ],
            }
            for theta, value in zip(
                radius["theta_e_arcsec"], radius_values, strict=True
            )
        ],
        "cell_radius_records": cell_radius.to_dict(orient="records"),
    }


def analyse_pnoise(
    parent_manifest: pd.DataFrame,
    parent_scores: pd.DataFrame,
    child_manifest: pd.DataFrame,
    child_scores: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    expected_draw_count: int,
    bootstrap_draws: int | None = None,
) -> dict[str, Any]:
    parent = _validate_binary_scores(
        parent_manifest, parent_scores, "grec_row_id"
    )
    child = _validate_binary_scores(
        child_manifest, child_scores, "pnoise_row_id"
    )
    required_child = {"parent_grec_row_id", "draw_index"}
    if required_child - set(child):
        raise ContractError("P_noise child manifest lacks parent/draw identities")
    parent_ids = set(parent["grec_row_id"].astype(str))
    child_parent_ids = set(child["parent_grec_row_id"].astype(str))
    if child_parent_ids != parent_ids:
        raise ContractError(
            "P_noise parent-child coverage is not exact in both directions"
        )
    if child.duplicated(["parent_grec_row_id", "draw_index"]).any():
        raise ContractError("duplicate parent/draw rows in P_noise")
    expected_draw_indices = set(range(int(expected_draw_count)))
    draw_index_sets = child.groupby("parent_grec_row_id")["draw_index"].apply(
        lambda values: set(map(int, values))
    )
    if not draw_index_sets.map(lambda value: value == expected_draw_indices).all():
        raise ContractError("P_noise draw indices do not match frozen cardinality")
    draw_counts = child.groupby("parent_grec_row_id").size()
    if not draw_counts.eq(int(expected_draw_count)).all():
        raise ContractError("P_noise child count does not match frozen draw count")
    identity_check = child.merge(
        parent[
            [
                "grec_row_id",
                "scene_id",
                "host_galaxy_id",
                "host_spatial_group_id",
                "source_parameter_id",
                "lens_parameter_id",
                "render_config_id",
                "theta_e_arcsec",
                "psf_bin",
                "depth_bin",
            ]
        ],
        left_on="parent_grec_row_id",
        right_on="grec_row_id",
        suffixes=("_child", "_parent"),
        validate="many_to_one",
    )
    for field in (
        "scene_id",
        "host_galaxy_id",
        "host_spatial_group_id",
        "source_parameter_id",
        "lens_parameter_id",
        "render_config_id",
        "theta_e_arcsec",
        "psf_bin",
        "depth_bin",
    ):
        if not identity_check[f"{field}_child"].eq(
            identity_check[f"{field}_parent"]
        ).all():
            raise ContractError(f"P_noise child changed parent identity {field}")
    physical_scene = (
        child.groupby("parent_grec_row_id")["decision"]
        .agg(["mean", "var", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "physical_probability",
                "var": "physical_draw_variance",
                "count": "draw_count",
            }
        )
    )
    scene = parent.merge(
        physical_scene,
        left_on="grec_row_id",
        right_on="parent_grec_row_id",
        validate="one_to_one",
    )
    scene["value"] = scene["physical_probability"] - scene["decision"]
    host_radius = (
        scene.groupby(
            [
                "host_galaxy_id",
                "host_spatial_group_id",
                "psf_bin",
                "depth_bin",
                "theta_e_arcsec",
            ],
            as_index=False,
        )["value"]
        .mean()
    )
    estimate = float(
        host_radius.groupby("host_galaxy_id")["value"].mean().mean()
    )
    draws = int(bootstrap_draws or config["inference"]["bootstrap_draws"])
    overall_draws, radius_draws, _, bootstrap_design = (
        _global_spatial_multiplier_bootstrap(
        host_radius,
        draws=draws,
        seed_namespace=str(config["inference"]["bootstrap_seed_namespace"])
        + ":pnoise",
        )
    )
    radius = host_radius.groupby("theta_e_arcsec", as_index=False)["value"].mean()
    radius_values = radius["value"].to_numpy()
    confidence = float(config["inference"]["confidence_level"])
    max_deviation = np.max(np.abs(radius_draws - radius_values[None, :]), axis=1)
    simultaneous_half_width = float(np.quantile(max_deviation, confidence))

    host_counts = scene.groupby("host_galaxy_id")["grec_row_id"].transform("size")
    weights = 1.0 / (scene["host_galaxy_id"].nunique() * host_counts)
    if expected_draw_count < 2:
        finite_draw_mcse: float | None = None
        mcse_status = "unavailable_infinite_with_one_draw"
        mcse_precision_gate_passed = False
    else:
        finite_draw_mcse = float(
            np.sqrt(
                np.sum(
                    np.square(weights.to_numpy())
                    * scene["physical_draw_variance"].to_numpy()
                    / scene["draw_count"].to_numpy()
                )
            )
        )
        mcse_status = "estimated_from_within_scene_draw_variance"
        mcse_precision_gate_passed = bool(np.isfinite(finite_draw_mcse))
    crossings = child.merge(
        parent[["grec_row_id", "decision"]].rename(
            columns={"grec_row_id": "parent_grec_row_id", "decision": "parent_decision"}
        ),
        on="parent_grec_row_id",
        validate="many_to_one",
    )
    return {
        "estimand": "equal_host_expected_physical_minus_expectation_detection_risk",
        "estimate": estimate,
        "confidence_interval_sampling": _interval(overall_draws, confidence),
        "finite_photon_draw_mcse": finite_draw_mcse,
        "finite_photon_draw_mcse_if_unavailable": (
            "infinite" if finite_draw_mcse is None else None
        ),
        "finite_photon_draw_mcse_status": mcse_status,
        "finite_photon_draw_precision_gate_passed": mcse_precision_gate_passed,
        "bootstrap_draws": draws,
        "bootstrap_design": bootstrap_design,
        "draw_count": int(expected_draw_count),
        "distinct_hosts": int(scene["host_galaxy_id"].nunique()),
        "distinct_scenes": len(scene),
        "gained_crossing_probability": float(
            ((crossings["decision"] == 1) & (crossings["parent_decision"] == 0)).mean()
        ),
        "lost_crossing_probability": float(
            ((crossings["decision"] == 0) & (crossings["parent_decision"] == 1)).mean()
        ),
        "radius": [
            {
                "theta_e_arcsec": float(theta),
                "estimate": float(value),
                "simultaneous_interval": [
                    float(value - simultaneous_half_width),
                    float(value + simultaneous_half_width),
                ],
            }
            for theta, value in zip(
                radius["theta_e_arcsec"], radius_values, strict=True
            )
        ],
    }
