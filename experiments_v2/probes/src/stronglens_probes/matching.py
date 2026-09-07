"""Deterministic score-blind G_match assignment and rendered-pair validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .authorities import AuthorityContext
from .contracts import (
    ContractError,
    assert_score_blind,
    canonical_frame_sha256,
    canonical_json_sha256,
    canonical_row_sha256,
    derive_catalogue_covariates,
    require_columns,
    require_nonempty_columns,
    require_sha256,
    require_sha256_column,
    sha256_file,
    stable_digest,
)
from .parents import RAW_TRACTOR_COLUMNS, BoundTractorRows, GRecArtifacts


@dataclass(frozen=True)
class MatchResult:
    matches: pd.DataFrame
    reference_disposition: pd.DataFrame
    host_disposition: pd.DataFrame
    receipt: dict[str, Any]


@dataclass(frozen=True)
class FrozenMatchArtifact:
    frame: pd.DataFrame
    receipt: dict[str, Any]
    manifest_sha256: str
    manifest_path: Path
    receipt_path: Path


@dataclass(frozen=True)
class FrozenPairPlanArtifact:
    frame: pd.DataFrame
    receipt: dict[str, Any]
    manifest_sha256: str
    manifest_path: Path
    receipt_path: Path


@dataclass(frozen=True)
class RenderedPairArtifact:
    frame: pd.DataFrame
    path: Path
    receipt_path: Path
    manifest_sha256: str
    receipt: dict[str, Any]


def assert_frozen_match_artifact(
    artifact: FrozenMatchArtifact,
    *,
    config: Mapping[str, Any],
    authority: AuthorityContext,
) -> None:
    authority.assert_config(config)
    if (
        sha256_file(artifact.manifest_path) != artifact.manifest_sha256
        or json.loads(artifact.receipt_path.read_text(encoding="utf-8"))
        != artifact.receipt
    ):
        raise ContractError("frozen match artifact changed after validation")
    if (
        artifact.receipt.get("authority_digest") != authority.authority_digest
        or artifact.receipt.get("matching_contract_sha256")
        != matching_contract_sha256(config)
        or artifact.receipt.get("matches_manifest_sha256") != artifact.manifest_sha256
    ):
        raise ContractError("frozen match artifact has invalid authorities")
    try:
        pd.testing.assert_frame_equal(
            artifact.frame.reset_index(drop=True),
            pd.read_parquet(artifact.manifest_path).reset_index(drop=True),
            check_exact=True,
        )
    except AssertionError as error:
        raise ContractError("frozen match in-memory rows mutated") from error
    require_sha256_column(artifact.frame, "match_row_sha256", context="frozen matches")
    if len(artifact.frame) < 10:
        raise ContractError("frozen match artifact has fewer than 10 references")


def load_frozen_pair_plan(
    *,
    pair_plan_path: Path,
    receipt_path: Path,
    frozen_matches: FrozenMatchArtifact,
    grec: GRecArtifacts,
    config: Mapping[str, Any],
    authority: AuthorityContext,
) -> FrozenPairPlanArtifact:
    """Authenticate pair-plan bytes, row hashes, authorities, and match parent."""
    assert_frozen_match_artifact(
        frozen_matches,
        config=config,
        authority=authority,
    )
    grec.revalidate(config=dict(config), authority=authority)
    manifest_sha256 = sha256_file(pair_plan_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "complete_frozen_G_match_pair_plan"
        or receipt.get("pair_manifest_sha256") != manifest_sha256
        or receipt.get("matches_manifest_sha256") != frozen_matches.manifest_sha256
        or receipt.get("authority_digest") != authority.authority_digest
        or receipt.get("G_rec_scene_manifest_sha256") != grec.scene_sha256
        or receipt.get("G_rec_host_manifest_sha256") != grec.host_sha256
        or receipt.get("G_rec_tractor_manifest_sha256") != grec.tractor.sha256
    ):
        raise ContractError("pair-plan receipt has invalid parent authorities")
    frame = pd.read_parquet(pair_plan_path)
    require_columns(frame, _PAIR_PLAN_REQUIRED, context="frozen pair plan")
    hash_columns = [
        column for column in frame.columns if column != "pair_plan_row_sha256"
    ]
    expected_hashes = [
        canonical_row_sha256(row, hash_columns)
        for row in frame.to_dict(orient="records")
    ]
    if frame["pair_plan_row_sha256"].astype(str).tolist() != expected_hashes:
        raise ContractError("frozen pair plan has invalid row hashes")
    if set(frame["pair_id"].astype(str)) != set(
        frozen_matches.frame["pair_id"].astype(str)
    ):
        raise ContractError("frozen pair plan differs from frozen match membership")
    scene_lookup = grec.scenes.set_index(grec.scenes["grec_row_id"].astype(str))
    tractor_lookup = grec.tractor.frame.set_index(
        grec.tractor.frame["galaxy_id"].astype(str)
    )
    scene_columns = {
        "scene_id": "scene_id",
        "target_galaxy_id": "host_galaxy_id",
        "host_parent_id": "host_parent_id",
        "base_pixel_sha256": "host_pixel_sha256",
        "host_physical_group_id": "host_physical_group_id",
        "host_spatial_group_id": "host_spatial_group_id",
        "source_parameter_id": "source_parameter_id",
        "lens_parameter_id": "lens_parameter_id",
        "psf_id": "psf_id",
        "rendering_id": "rendering_id",
        "render_parameter_id": "render_parameter_id",
        "random_draw_id": "random_draw_id",
        "intrinsic_flux_g_nmgy": "intrinsic_flux_g_nmgy",
        "intrinsic_flux_r_nmgy": "intrinsic_flux_r_nmgy",
        "intrinsic_flux_z_nmgy": "intrinsic_flux_z_nmgy",
        "source_flux_parameter_sha256": "source_flux_parameter_sha256",
    }
    for plan in frame.itertuples(index=False):
        grec_row_id = str(plan.grec_row_id)
        if grec_row_id not in scene_lookup.index:
            raise ContractError("pair plan has a non-authority G_rec scene")
        scene = scene_lookup.loc[grec_row_id]
        for plan_column, scene_column in scene_columns.items():
            if str(getattr(plan, plan_column)) != str(scene[scene_column]):
                raise ContractError(
                    f"pair plan changed G_rec scene field {plan_column}"
                )
        tractor = tractor_lookup.loc[str(plan.target_galaxy_id)]
        expected_tractor_hash = canonical_row_sha256(
            tractor.to_dict(),
            [*RAW_TRACTOR_COLUMNS, "payload_sha256"],
        )
        if str(plan.tractor_row_sha256) != expected_tractor_hash:
            raise ContractError("pair plan changed raw Tractor parent")
    return FrozenPairPlanArtifact(
        frame=frame,
        receipt=receipt,
        manifest_sha256=manifest_sha256,
        manifest_path=pair_plan_path.resolve(),
        receipt_path=receipt_path.resolve(),
    )


def assert_frozen_pair_plan_artifact(
    artifact: FrozenPairPlanArtifact,
    *,
    frozen_matches: FrozenMatchArtifact,
    config: Mapping[str, Any],
    authority: AuthorityContext,
) -> None:
    assert_frozen_match_artifact(
        frozen_matches,
        config=config,
        authority=authority,
    )
    if (
        sha256_file(artifact.manifest_path) != artifact.manifest_sha256
        or json.loads(artifact.receipt_path.read_text(encoding="utf-8"))
        != artifact.receipt
        or artifact.receipt.get("pair_manifest_sha256") != artifact.manifest_sha256
        or artifact.receipt.get("matches_manifest_sha256")
        != frozen_matches.manifest_sha256
        or artifact.receipt.get("authority_digest") != authority.authority_digest
    ):
        raise ContractError("frozen pair-plan artifact changed or has wrong authority")
    try:
        pd.testing.assert_frame_equal(
            artifact.frame.reset_index(drop=True),
            pd.read_parquet(artifact.manifest_path).reset_index(drop=True),
            check_exact=True,
        )
    except AssertionError as error:
        raise ContractError("frozen pair-plan in-memory rows mutated") from error


def matching_contract_sha256(config: Mapping[str, Any]) -> str:
    specification = config["G_match"]
    contract = {
        "covariates": specification["covariates"],
        "scaling": specification["scaling"],
        "matching": specification["matching"],
        "role_column": specification["host_universe_role_column"],
        "role_value": specification["host_universe_role_value"],
        "score_blind": specification["score_blind"],
    }
    return canonical_json_sha256(contract)


def _validate_matching_specification(config: Mapping[str, Any]) -> None:
    specification = config["G_match"]
    matching = specification["matching"]
    if specification["scaling"] != {
        "fit_population": "exact_G_rec_hosts_only",
        "centre": "median",
        "scale": "interquartile_range",
    }:
        raise ContractError("G_match scaling contract has drifted")
    expected = {
        "distance": "robust_diagonal_euclidean",
        "component_caliper_standardised": 2.0,
        "total_distance_caliper": 3.0,
        "targets_per_reference": 1,
        "replacement": False,
        "assignment": "maximum_cardinality_then_minimum_distance",
        "tie_rule": "sha256_sorted_identifiers",
        "unmatched_references": "retain_with_disposition",
        "minimum_matched_references_for_five_fold_analysis": 10,
        "contract_version": "score_blind_robust_iqr_optimal_v1",
    }
    if matching != expected:
        raise ContractError("G_match matching contract has drifted")


def _with_covariates(frame: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    derived = derive_catalogue_covariates(frame)
    for column in covariates:
        if column in frame and not np.allclose(
            pd.to_numeric(frame[column], errors="coerce"),
            derived[column],
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ContractError(
                f"stored matching covariate {column} differs from raw Tractor"
            )
    return derived


def _stable_order(values: pd.Series, *, namespace: str) -> np.ndarray:
    keys = [
        (stable_digest(namespace, value), str(value), index)
        for index, value in enumerate(values.astype(str))
    ]
    return np.asarray([entry[2] for entry in sorted(keys)], dtype=np.int64)


def match_targets(
    reference_tractor: BoundTractorRows,
    host_tractor: BoundTractorRows,
    grec: GRecArtifacts,
    *,
    config: Mapping[str, Any],
    authority: AuthorityContext,
) -> MatchResult:
    """Assign at most one unique target per reference under frozen covariate rules."""
    authority.assert_config(config)
    _validate_matching_specification(config)
    if not authority.host_physicals:
        raise ContractError("final host physical-group authority is unavailable")
    reference_tractor.revalidate(dict(config))
    host_tractor.revalidate(dict(config))
    grec.revalidate(config=dict(config), authority=authority)
    references = reference_tractor.frame
    grec_host_ids = set(grec.hosts["galaxy_id"].astype(str))
    experiment_hosts = host_tractor.frame[
        host_tractor.frame["galaxy_id"].astype(str).isin(grec_host_ids)
    ].copy()
    if set(experiment_hosts["galaxy_id"].astype(str)) != grec_host_ids:
        raise ContractError("G_match candidates do not equal the G_rec host bank")
    specification = config["G_match"]
    covariates = list(specification["covariates"])
    matching = specification["matching"]
    role_column = specification["host_universe_role_column"]
    role_value = specification["host_universe_role_value"]
    require_columns(references, {"galaxy_id"}, context="G_match references")
    require_columns(
        experiment_hosts,
        {"galaxy_id", role_column},
        context="experiment host universe",
    )
    assert_score_blind(references, context="G_match reference matching table")
    assert_score_blind(experiment_hosts, context="G_match host matching universe")
    authority.assert_reference_payloads(
        references,
        id_column="galaxy_id",
        payload_column="payload_sha256",
        exact=True,
        context="G_match reference matching table",
    )
    for row in experiment_hosts[["galaxy_id", "payload_sha256"]].itertuples(
        index=False
    ):
        if authority.host_payloads.get(str(row.galaxy_id)) != str(row.payload_sha256):
            raise ContractError("G_match candidate has non-authority host pixels")
    if not experiment_hosts[role_column].eq(role_value).all():
        raise ContractError("G_match candidates must all come from experiment_host")
    if experiment_hosts["galaxy_id"].astype(str).duplicated().any():
        raise ContractError("experiment host universe contains duplicate IDs")
    if set(references["galaxy_id"].astype(str)) & set(
        experiment_hosts["galaxy_id"].astype(str)
    ):
        raise ContractError("a reference system appears in the target host universe")
    if matching["targets_per_reference"] != 1 or matching["replacement"] is not False:
        raise ContractError("only one unique target per reference is implemented")

    reference = _with_covariates(references, covariates)
    hosts = _with_covariates(experiment_hosts, covariates)
    reference["galaxy_id"] = reference["galaxy_id"].astype(str)
    hosts["galaxy_id"] = hosts["galaxy_id"].astype(str)
    reference_complete = np.isfinite(reference[covariates].to_numpy(float)).all(axis=1)
    host_complete = np.isfinite(hosts[covariates].to_numpy(float)).all(axis=1)
    reference_disposition = reference[["galaxy_id"]].copy()
    reference_disposition["covariates_complete"] = reference_complete
    reference_disposition["matched"] = False
    reference_disposition["target_galaxy_id"] = pd.NA
    reference_disposition["disposition"] = np.where(
        reference_complete, "eligible_for_matching", "unmatched_missing_covariate"
    )
    host_disposition = hosts[["galaxy_id"]].copy()
    host_disposition["covariates_complete"] = host_complete
    host_disposition["selected"] = False
    host_disposition["disposition"] = np.where(
        host_complete, "eligible_not_selected", "excluded_missing_covariate"
    )
    eligible_references = reference[reference_complete].copy()
    eligible_hosts = hosts[host_complete].copy()
    if eligible_references.empty:
        raise ContractError("no reference has complete matching covariates")
    if eligible_hosts.empty:
        raise ContractError("no experiment host has complete matching covariates")

    host_matrix = eligible_hosts[covariates].to_numpy(float)
    centre = np.median(host_matrix, axis=0)
    q25, q75 = np.quantile(host_matrix, [0.25, 0.75], axis=0, method="linear")
    scale = q75 - q25
    if not np.isfinite(scale).all() or (scale <= 0.0).any():
        bad = [covariates[index] for index in np.flatnonzero(scale <= 0.0)]
        raise ContractError(f"host-universe covariate IQR is degenerate: {bad}")

    reference_order = _stable_order(
        eligible_references["galaxy_id"], namespace="G_match:reference"
    )
    host_order = _stable_order(eligible_hosts["galaxy_id"], namespace="G_match:host")
    eligible_references = eligible_references.iloc[reference_order].reset_index(
        drop=True
    )
    eligible_hosts = eligible_hosts.iloc[host_order].reset_index(drop=True)
    reference_matrix = eligible_references[covariates].to_numpy(float)
    host_matrix = eligible_hosts[covariates].to_numpy(float)
    differences = (reference_matrix[:, None, :] - host_matrix[None, :, :]) / scale
    distances = np.sqrt(np.square(differences).sum(axis=2))
    component_caliper = float(matching["component_caliper_standardised"])
    distance_caliper = float(matching["total_distance_caliper"])
    admissible = (np.abs(differences) <= component_caliper).all(axis=2) & (
        distances <= distance_caliper
    )

    n_reference, n_host = distances.shape
    unmatched_penalty = (n_reference + 1) * (distance_caliper + 1.0)
    cost = np.full((n_reference, n_host + n_reference), np.inf, dtype=float)
    cost[:, :n_host] = np.where(admissible, distances, np.inf)
    for index in range(n_reference):
        cost[index, n_host + index] = unmatched_penalty
    rows, columns = linear_sum_assignment(cost)
    if len(rows) != n_reference:
        raise ContractError(
            "assignment did not return a disposition for each reference"
        )

    match_rows: list[dict[str, Any]] = []
    for ref_index, selected_column in zip(rows, columns, strict=True):
        reference_id = str(eligible_references.iloc[ref_index]["galaxy_id"])
        disposition_index = reference_disposition.index[
            reference_disposition["galaxy_id"].astype(str).eq(reference_id)
        ][0]
        if selected_column >= n_host:
            reference_disposition.loc[disposition_index, "disposition"] = (
                "unmatched_no_common_support"
            )
            continue
        target_id = str(eligible_hosts.iloc[selected_column]["galaxy_id"])
        pair_id = f"gmatch-{stable_digest(reference_id, target_id)[:24]}"
        row: dict[str, Any] = {
            "pair_id": pair_id,
            "reference_id": reference_id,
            "target_galaxy_id": target_id,
            "distance": float(distances[ref_index, selected_column]),
        }
        for covariate_index, covariate in enumerate(covariates):
            row[f"delta_standardised__{covariate}"] = float(
                differences[ref_index, selected_column, covariate_index]
            )
            row[f"reference__{covariate}"] = float(
                reference_matrix[ref_index, covariate_index]
            )
            row[f"target__{covariate}"] = float(
                host_matrix[selected_column, covariate_index]
            )
        match_rows.append(row)
        reference_disposition.loc[disposition_index, "matched"] = True
        reference_disposition.loc[disposition_index, "target_galaxy_id"] = target_id
        reference_disposition.loc[disposition_index, "disposition"] = "matched"
        host_index = host_disposition.index[
            host_disposition["galaxy_id"].astype(str).eq(target_id)
        ][0]
        host_disposition.loc[host_index, "selected"] = True
        host_disposition.loc[host_index, "disposition"] = "selected"

    matches = pd.DataFrame(match_rows)
    if not matches.empty:
        matches = matches.sort_values("reference_id", ignore_index=True)
        if (
            matches["reference_id"].duplicated().any()
            or matches["target_galaxy_id"].duplicated().any()
            or (matches["distance"] > distance_caliper).any()
        ):
            raise ContractError("assignment violates uniqueness or caliper")
    minimum = int(matching["minimum_matched_references_for_five_fold_analysis"])
    if len(matches) < minimum:
        raise ContractError(
            f"G_match has {len(matches)} matches, below required minimum {minimum}"
        )
    match_columns = list(matches.columns)
    matches["match_row_sha256"] = [
        canonical_row_sha256(row, match_columns)
        for row in matches.to_dict(orient="records")
    ]
    matches_content_sha256 = canonical_frame_sha256(
        matches,
        columns=matches.columns,
        sort_by=["reference_id"],
    )
    receipt = {
        "schema_version": 2,
        "method": "maximum_cardinality_then_minimum_robust_distance",
        "score_blind": True,
        "host_universe_rows": len(experiment_hosts),
        "candidate_population": "exact_G_rec_hosts",
        "G_rec_host_manifest_sha256": grec.host_sha256,
        "host_universe_complete_covariates": int(host_complete.sum()),
        "references": 43,
        "references_complete_covariates": int(reference_complete.sum()),
        "matched_references": len(matches),
        "unmatched_references": 43 - len(matches),
        "covariates": covariates,
        "scaling_population": "exact_G_rec_hosts_only",
        "G_rec_host_median": dict(zip(covariates, centre.tolist(), strict=True)),
        "G_rec_host_iqr": dict(zip(covariates, scale.tolist(), strict=True)),
        "component_caliper_standardised": component_caliper,
        "total_distance_caliper": distance_caliper,
        "replacement": False,
        "targets_per_reference": 1,
        "tie_rule": "sha256_sorted_identifiers",
        "host_total_flux_floor": None,
        "matching_contract_sha256": matching_contract_sha256(config),
        "authority_digest": authority.authority_digest,
        "reference_manifest_sha256": authority.reference_manifest_sha256,
        "reference_payload_manifest_sha256": (
            authority.reference_payload_manifest_sha256
        ),
        "host_universe_manifest_sha256": (authority.experiment_host_universe_sha256),
        "reference_tractor_manifest_sha256": reference_tractor.sha256,
        "host_tractor_manifest_sha256": host_tractor.sha256,
        "reference_catalogue_content_sha256": canonical_frame_sha256(
            reference,
            columns=["galaxy_id", "payload_sha256", *covariates],
            sort_by=["galaxy_id"],
        ),
        "host_catalogue_content_sha256": canonical_frame_sha256(
            hosts,
            columns=[
                "galaxy_id",
                "payload_sha256",
                role_column,
                *covariates,
            ],
            sort_by=["galaxy_id"],
        ),
        "matches_content_sha256": matches_content_sha256,
    }
    return MatchResult(
        matches=matches,
        reference_disposition=reference_disposition.sort_values(
            "galaxy_id", ignore_index=True
        ),
        host_disposition=host_disposition.sort_values("galaxy_id", ignore_index=True),
        receipt=receipt,
    )


def load_frozen_matches(
    *,
    matches_path: Path,
    receipt_path: Path,
    reference_tractor: BoundTractorRows,
    host_tractor: BoundTractorRows,
    grec: GRecArtifacts,
    config: Mapping[str, Any],
    authority: AuthorityContext,
) -> FrozenMatchArtifact:
    """Authenticate a persisted match file by receipt and exact recomputation."""
    authority.assert_config(config)
    manifest_sha256 = sha256_file(matches_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require_sha256(
        receipt.get("matches_manifest_sha256"), context="match receipt manifest"
    )
    if receipt["matches_manifest_sha256"] != manifest_sha256:
        raise ContractError("frozen match manifest differs from its receipt")
    if receipt.get("authority_digest") != authority.authority_digest:
        raise ContractError("frozen match receipt has the wrong authority")
    if receipt.get("matching_contract_sha256") != matching_contract_sha256(config):
        raise ContractError("frozen match receipt has the wrong matching contract")
    frame = pd.read_parquet(matches_path)
    recomputed = match_targets(
        reference_tractor,
        host_tractor,
        grec,
        config=config,
        authority=authority,
    )
    if list(frame.columns) != list(recomputed.matches.columns):
        raise ContractError("frozen match manifest schema differs from recomputation")
    try:
        pd.testing.assert_frame_equal(
            frame.reset_index(drop=True),
            recomputed.matches.reset_index(drop=True),
            check_exact=True,
        )
    except AssertionError as error:
        raise ContractError("frozen matches differ from exact recomputation") from error
    if (
        receipt.get("matches_content_sha256")
        != recomputed.receipt["matches_content_sha256"]
    ):
        raise ContractError("frozen match content digest differs from recomputation")
    for field in (
        "reference_catalogue_content_sha256",
        "host_catalogue_content_sha256",
    ):
        if receipt.get(field) != recomputed.receipt[field]:
            raise ContractError(f"frozen match receipt has wrong {field}")
    return FrozenMatchArtifact(
        frame=frame,
        receipt=receipt,
        manifest_sha256=manifest_sha256,
        manifest_path=matches_path.resolve(),
        receipt_path=receipt_path.resolve(),
    )


_PAIR_REQUIRED = {
    "pair_id",
    "reference_id",
    "target_galaxy_id",
    "arm",
    "grec_row_id",
    "scene_id",
    "base_pixel_sha256",
    "host_parent_id",
    "host_physical_group_id",
    "host_spatial_group_id",
    "final_image_sha256",
    "source_parameter_id",
    "lens_parameter_id",
    "psf_id",
    "rendering_id",
    "render_parameter_id",
    "random_draw_id",
    "source_flux_policy",
    "intrinsic_flux_g_nmgy",
    "intrinsic_flux_r_nmgy",
    "intrinsic_flux_z_nmgy",
    "source_flux_parameter_sha256",
    "pair_scene_link_manifest_sha256",
    "grec_scene_manifest_sha256",
    "grec_host_manifest_sha256",
    "tractor_manifest_sha256",
    "tractor_row_sha256",
    "match_row_sha256",
    "pair_plan_row_sha256",
    "reference_manifest_sha256",
    "reference_payload_manifest_sha256",
    "host_universe_manifest_sha256",
    "matches_manifest_sha256",
    "pair_manifest_sha256",
}
_PAIR_PLAN_REQUIRED = _PAIR_REQUIRED - {
    "final_image_sha256",
    "pair_manifest_sha256",
}


def validate_rendered_pairs(
    pairs: pd.DataFrame,
    *,
    frozen_matches: FrozenMatchArtifact,
    frozen_pair_plan: FrozenPairPlanArtifact,
    config: Mapping[str, Any],
    authority: AuthorityContext,
) -> None:
    """Validate exactly one bare/injected pair per selected target."""
    authority.assert_config(config)
    assert_frozen_pair_plan_artifact(
        frozen_pair_plan,
        frozen_matches=frozen_matches,
        config=config,
        authority=authority,
    )
    pair_plan = frozen_pair_plan.frame
    pair_manifest_sha256 = frozen_pair_plan.manifest_sha256
    assert_frozen_match_artifact(
        frozen_matches,
        config=config,
        authority=authority,
    )
    matches = frozen_matches.frame
    require_columns(pairs, _PAIR_REQUIRED, context="G_match rendered pairs")
    require_columns(pair_plan, _PAIR_PLAN_REQUIRED, context="pair plan")
    require_nonempty_columns(
        pairs,
        {
            "pair_id",
            "reference_id",
            "target_galaxy_id",
            "arm",
            "host_parent_id",
            "host_physical_group_id",
            "host_spatial_group_id",
            "source_parameter_id",
            "lens_parameter_id",
            "psf_id",
            "rendering_id",
            "render_parameter_id",
            "random_draw_id",
        },
        context="G_match rendered pairs",
    )
    for column in (
        "base_pixel_sha256",
        "final_image_sha256",
        "source_flux_parameter_sha256",
        "pair_scene_link_manifest_sha256",
        "grec_scene_manifest_sha256",
        "grec_host_manifest_sha256",
        "tractor_manifest_sha256",
        "tractor_row_sha256",
        "match_row_sha256",
        "pair_plan_row_sha256",
        "reference_manifest_sha256",
        "reference_payload_manifest_sha256",
        "host_universe_manifest_sha256",
        "matches_manifest_sha256",
        "pair_manifest_sha256",
    ):
        require_sha256_column(pairs, column, context="G_match rendered pairs")
    prohibited_floor_columns = [
        column
        for column in pairs.columns
        if "floor" in column.lower()
        and any(token in column.lower() for token in ("flux", "brightness", "host"))
    ]
    if prohibited_floor_columns:
        raise ContractError(
            f"host-total flux floor fields are forbidden: {prohibited_floor_columns}"
        )
    if not pairs["source_flux_policy"].eq("fixed_intrinsic").all():
        raise ContractError("G_match primary must use fixed intrinsic source flux")
    for column in (
        "intrinsic_flux_g_nmgy",
        "intrinsic_flux_r_nmgy",
        "intrinsic_flux_z_nmgy",
    ):
        values = pd.to_numeric(pairs[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ContractError(f"G_match has invalid fixed-intrinsic {column}")
    authority_columns = {
        "reference_manifest_sha256": authority.reference_manifest_sha256,
        "reference_payload_manifest_sha256": (
            authority.reference_payload_manifest_sha256
        ),
        "host_universe_manifest_sha256": (authority.experiment_host_universe_sha256),
        "matches_manifest_sha256": frozen_matches.manifest_sha256,
        "pair_manifest_sha256": pair_manifest_sha256,
    }
    for column, expected in authority_columns.items():
        if not pairs[column].eq(expected).all():
            raise ContractError(f"G_match rendered pairs have wrong {column}")
    expected_pairs = set(matches["pair_id"].astype(str))
    if set(pairs["pair_id"].astype(str)) != expected_pairs:
        raise ContractError("rendered pairs do not cover exactly the frozen matches")
    if pairs["target_galaxy_id"].astype(str).nunique() != len(matches):
        raise ContractError("a target is repeated across G_match pairs")
    expected_match = matches.set_index("pair_id")
    plan = pair_plan.set_index(["pair_id", "arm"])
    if len(plan) != len(pair_plan) or not plan.index.is_unique:
        raise ContractError("G_match pair plan has duplicate pair/arm rows")
    shared_columns = [
        "reference_id",
        "target_galaxy_id",
        "grec_row_id",
        "scene_id",
        "base_pixel_sha256",
        "host_parent_id",
        "host_physical_group_id",
        "host_spatial_group_id",
        "source_parameter_id",
        "lens_parameter_id",
        "psf_id",
        "rendering_id",
        "render_parameter_id",
        "random_draw_id",
        "source_flux_policy",
        "intrinsic_flux_g_nmgy",
        "intrinsic_flux_r_nmgy",
        "intrinsic_flux_z_nmgy",
        "source_flux_parameter_sha256",
        "pair_scene_link_manifest_sha256",
        "grec_scene_manifest_sha256",
        "grec_host_manifest_sha256",
        "tractor_manifest_sha256",
        "tractor_row_sha256",
        "match_row_sha256",
    ]
    for pair_id, frame in pairs.groupby("pair_id", sort=False):
        if len(frame) != 2 or set(frame["arm"]) != {"bare", "injected"}:
            raise ContractError(f"{pair_id} is not one bare/injected pair")
        for column in shared_columns:
            if frame[column].nunique(dropna=False) != 1:
                raise ContractError(f"{pair_id} does not share {column} across arms")
        match = expected_match.loc[str(pair_id)]
        if (
            str(frame["reference_id"].iloc[0]) != str(match["reference_id"])
            or str(frame["target_galaxy_id"].iloc[0]) != str(match["target_galaxy_id"])
            or str(frame["match_row_sha256"].iloc[0]) != str(match["match_row_sha256"])
        ):
            raise ContractError(f"{pair_id} differs from its frozen match")
        target_id = str(match["target_galaxy_id"])
        if (
            frame["base_pixel_sha256"].iloc[0] != authority.host_payloads[target_id]
            or str(frame["host_parent_id"].iloc[0]) != authority.host_parents[target_id]
        ):
            raise ContractError(f"{pair_id} has wrong bare-host ancestry")
        for rendered in frame.itertuples(index=False):
            expected_plan = plan.loc[(str(pair_id), str(rendered.arm))]
            for column in _PAIR_REQUIRED - {
                "pair_id",
                "arm",
                "final_image_sha256",
                "pair_manifest_sha256",
            }:
                if str(getattr(rendered, column)) != str(expected_plan[column]):
                    raise ContractError(
                        f"{pair_id}/{rendered.arm} differs from frozen "
                        f"pair plan {column}"
                    )
        if frame["final_image_sha256"].isna().any():
            raise ContractError(f"{pair_id} has a missing final image hash")
        if frame["final_image_sha256"].nunique() != 2:
            raise ContractError(f"{pair_id} bare and injected pixels are identical")
        bare_hash = frame.loc[frame["arm"].eq("bare"), "final_image_sha256"].iloc[0]
        if bare_hash != frame["base_pixel_sha256"].iloc[0]:
            raise ContractError(f"{pair_id} bare pixels differ from host payload")


def load_rendered_pairs(
    *,
    path: Path,
    receipt_path: Path,
    frozen_matches: FrozenMatchArtifact,
    frozen_pair_plan: FrozenPairPlanArtifact,
    config: Mapping[str, Any],
    authority: AuthorityContext,
) -> RenderedPairArtifact:
    """Reload rendered rows from a receipt-bound file and revalidate all parents."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest_sha256 = sha256_file(path)
    if (
        receipt.get("status") != "complete_G_match_rendered_pair_validation"
        or receipt.get("rendered_pairs_sha256") != manifest_sha256
        or receipt.get("pair_manifest_sha256") != frozen_pair_plan.manifest_sha256
        or receipt.get("matches_manifest_sha256") != frozen_matches.manifest_sha256
        or receipt.get("authority_digest") != authority.authority_digest
    ):
        raise ContractError("rendered-pair receipt has invalid parent authorities")
    frame = pd.read_parquet(path)
    validate_rendered_pairs(
        frame,
        frozen_matches=frozen_matches,
        frozen_pair_plan=frozen_pair_plan,
        config=config,
        authority=authority,
    )
    return RenderedPairArtifact(
        frame=frame,
        path=path.resolve(),
        receipt_path=receipt_path.resolve(),
        manifest_sha256=manifest_sha256,
        receipt=receipt,
    )
