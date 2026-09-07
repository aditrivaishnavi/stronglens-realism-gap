"""Shared fail-closed contracts for pre-score manifests and representations."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


class ContractError(RuntimeError):
    """Raised when an input does not satisfy the registered scientific contract."""


_OUTCOME_TOKENS = {
    "auc",
    "decision",
    "detection",
    "detectoroutput",
    "logit",
    "logits",
    "oof",
    "prediction",
    "predictions",
    "probability",
    "probabilities",
    "score",
    "scores",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_digest(*parts: object) -> str:
    encoded = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def require_sha256(value: object, *, context: str) -> str:
    if not is_sha256(value):
        raise ContractError(f"{context} must be a lowercase SHA-256")
    return str(value)


def require_sha256_column(
    frame: pd.DataFrame,
    column: str,
    *,
    context: str,
    unique: bool = False,
) -> None:
    require_columns(frame, {column}, context=context)
    if not frame[column].map(is_sha256).all():
        raise ContractError(f"{context}.{column} contains an invalid SHA-256")
    if unique and frame[column].duplicated().any():
        raise ContractError(f"{context}.{column} must be unique")


def require_nonempty_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    context: str,
) -> None:
    require_columns(frame, columns, context=context)
    for column in columns:
        values = frame[column]
        normalized = values.astype(str).str.strip()
        if (
            values.isna().any()
            or normalized.eq("").any()
            or normalized.str.lower().isin({"nan", "none", "null"}).any()
        ):
            raise ContractError(f"{context}.{column} contains a null or empty value")


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_protocol_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash immutable protocol fields while excluding future execution state."""
    normalized = copy.deepcopy(dict(config))
    execution = normalized["execution"]
    execution["extraction_authorization_receipt_sha256"] = None
    execution["analysis_authorization_receipt_sha256"] = None
    execution["final_feature_suite_authority_sha256"] = {
        "G_domain": None,
        "G_match": None,
    }
    execution["feature_extraction_authorized"] = False
    execution["final_scoring_authorized"] = False
    execution["final_analysis_authorized"] = False
    return canonical_json_sha256(normalized)


def ordered_mapping_sha256(values: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted((str(key), str(value)) for key, value in values.items()):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_row_sha256(row: Mapping[str, Any], columns: Iterable[str]) -> str:
    payload: dict[str, Any] = {}
    for column in sorted(columns):
        value = row[column]
        if pd.isna(value):
            payload[column] = None
        elif isinstance(value, (np.integer, int)):
            payload[column] = int(value)
        elif isinstance(value, (np.floating, float)):
            payload[column] = float(value).hex()
        elif isinstance(value, (np.bool_, bool)):
            payload[column] = bool(value)
        else:
            payload[column] = str(value)
    return canonical_json_sha256(payload)


def canonical_frame_sha256(
    frame: pd.DataFrame,
    *,
    columns: Iterable[str],
    sort_by: Iterable[str],
) -> str:
    selected_columns = list(columns)
    order = list(sort_by)
    require_columns(frame, set(selected_columns) | set(order), context="hashed frame")
    selected = frame.sort_values(order, kind="mergesort").reset_index(drop=True)
    digest = hashlib.sha256()
    for row in selected[selected_columns].to_dict(orient="records"):
        digest.update(canonical_row_sha256(row, selected_columns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    *,
    context: str,
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ContractError(f"{context} lacks required columns: {missing}")


def _column_tokens(column: str) -> set[str]:
    lowered = column.lower()
    pieces = set(re.split(r"[^a-z0-9]+", lowered))
    pieces.add(re.sub(r"[^a-z0-9]+", "", lowered))
    return pieces


def assert_score_blind(
    frame: pd.DataFrame,
    *,
    context: str,
    allowed_columns: Iterable[str] = (),
) -> None:
    allowed = set(allowed_columns)
    prohibited = sorted(
        column
        for column in frame.columns
        if column not in allowed and _column_tokens(str(column)) & _OUTCOME_TOKENS
    )
    if prohibited:
        raise ContractError(f"{context} contains outcome-bearing columns: {prohibited}")


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError("probe config must be a mapping")
    validate_config(payload)
    return payload


def validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "study_id",
        "status",
        "authorities",
        "shared_references",
        "representations",
        "probe",
        "G_domain",
        "G_match",
        "inference",
        "execution",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ContractError(f"config lacks required sections: {missing}")
    if config["schema_version"] != 2:
        raise ContractError("unsupported config schema")
    references = config["shared_references"]
    if (
        references.get("expected_count") != 43
        or references.get("used_by") != ["G_domain", "G_match"]
        or references.get("independent_replications") is not False
    ):
        raise ContractError("the exact shared 43-reference contract has drifted")
    predicate = config["G_domain"]["predicate"]
    if predicate != {
        "observed_zmag_max_inclusive": 19.5,
        "sersic_min_inclusive": 4.0,
        "axis_ratio_q_min_inclusive": 0.6,
    }:
        raise ContractError("G_domain predicate must remain exactly <=19.5/>=4/>=0.6")
    if (
        config["G_domain"].get("parent") != "G_rec"
        or config["G_domain"].get("independently_rendered") is not False
    ):
        raise ContractError("G_domain must be an unchanged-pixel G_rec child")
    authorities = config["authorities"]
    for field in (
        "frozen_detector_checkpoint_sha256",
        "reference_manifest_sha256",
        "reference_payload_manifest_sha256",
        "experiment_host_universe_sha256",
        "reference_payload_adapter_manifest_sha256",
        "experiment_host_adapter_manifest_sha256",
        "reference_tractor_manifest_sha256",
        "experiment_host_tractor_source_manifest_sha256",
        "experiment_host_tractor_manifest_sha256",
        "final_grec_selection_manifest_sha256",
        "grec_host_source_manifest_sha256",
        "grec_scene_source_manifest_sha256",
        "grec_parent_source_manifest_sha256",
        "grec_host_manifest_sha256",
        "grec_scene_manifest_sha256",
        "schema_adapter_receipt_sha256",
    ):
        require_sha256(authorities.get(field), context=f"authorities.{field}")
    for field in (
        "experiment_host_parent_id_column",
        "experiment_host_payload_hash_column",
        "experiment_host_physical_group_column",
        "experiment_host_spatial_group_column",
        "reference_psf_id_column",
        "reference_render_parameter_id_column",
    ):
        if not isinstance(authorities.get(field), str) or not authorities[field]:
            raise ContractError(f"authorities.{field} must be a column name")
    representation = config["representations"]
    primary = representation["primary"]
    diagnostics = representation["required_diagnostics"]
    if (
        primary.get("id") != "stein_ssl_resnet50"
        or primary.get("feature_dimension") != 2048
        or primary.get("pretraining_image_overlap_status") != "unproven"
        or "primary" not in primary.get("result_label", "")
        or "unproven" not in primary.get("result_label", "")
    ):
        raise ContractError("Stein must remain the overlap-qualified G_domain primary")
    expected_diagnostics = {
        "imagenet_effnet_v2_s": 1280,
        "pixel_statistics": 18,
    }
    if {entry.get("id"): entry.get("feature_dimension") for entry in diagnostics} != (
        expected_diagnostics
    ):
        raise ContractError("ImageNet-1280 and pixel-18 diagnostics are mandatory")
    for entry in [primary, *diagnostics]:
        require_sha256(
            entry.get("preprocessing_contract_sha256"),
            context=f"{entry.get('id')}.preprocessing_contract_sha256",
        )
        require_sha256(
            entry.get("extraction_contract_sha256"),
            context=f"{entry.get('id')}.extraction_contract_sha256",
        )
        for field in ("extractor_code_path", "preprocessing_code_path"):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise ContractError(f"{entry.get('id')}.{field} is required")
        for field in ("extractor_code_sha256", "preprocessing_code_sha256"):
            require_sha256(
                entry.get(field),
                context=f"{entry.get('id')}.{field}",
            )
        if entry.get("checkpoint_sha256") is not None:
            require_sha256(
                entry.get("checkpoint_sha256"),
                context=f"{entry.get('id')}.checkpoint_sha256",
            )
    matching = config["G_match"]["matching"]
    rendering = config["G_match"]["rendering"]
    axis = config["G_match"]["reference_axis"]
    expected_dependencies = [
        "physical_group_id",
        "geographic_spatial_key",
        "spatial_group_id",
        "reference_id",
        "host_galaxy_id",
        "host_parent_id",
        "source_parameter_id",
        "lens_parameter_id",
        "psf_id",
        "rendering_id",
        "render_parameter_id",
        "random_draw_id",
        "base_pixel_sha256",
        "pixel_sha256",
    ]
    if (
        config["G_match"].get("score_blind") is not True
        or config["G_match"].get("candidate_population") != "exact_G_rec_hosts"
        or config["G_match"].get("scene_selection")
        != {
            "theta_e_arcsec": 1.5,
            "rule": "exact_radius_then_lexicographic_grec_row_id",
        }
        or config["G_match"].get("host_universe_role_column") != "proposed_role"
        or config["G_match"].get("host_universe_role_value") != "experiment_host"
        or config["G_match"].get("covariates")
        != [
            "observed_zmag",
            "sersic_n",
            "axis_ratio_q",
            "psf_fwhm_r_arcsec",
            "depth5_r_mag",
        ]
        or matching.get("targets_per_reference") != 1
        or matching.get("replacement") is not False
        or rendering.get("arms") != ["bare", "injected"]
        or rendering.get("source_flux_policy") != "fixed_intrinsic"
        or rendering.get("host_total_flux_floor") != "forbidden"
        or axis.get("fit_rows") != ["observed_reference", "bare"]
        or axis.get("injected_rows_fit_axis") is not False
        or matching.get("minimum_matched_references_for_five_fold_analysis") != 10
        or matching.get("distance") != "robust_diagonal_euclidean"
        or matching.get("assignment") != "maximum_cardinality_then_minimum_distance"
        or matching.get("tie_rule") != "sha256_sorted_identifiers"
        or config["probe"].get("dependency_columns") != expected_dependencies
    ):
        raise ContractError(
            "G_match pairing, score-blindness, or axis contract drifted"
        )
    inference = config["inference"]
    if (
        inference.get("multiplier_scheme") != "stratified_dependency_bootstrap"
        or int(inference.get("bootstrap_draws", 0)) != 10_000
        or inference.get("strata")
        != "joint_reference_fold_tuple_and_domain_simulation_fold"
        or inference.get("interval_status")
        != "disabled_until_actual_estimator_coverage_at_least_0.90"
        or inference.get("shared_reference_resamples_between_probes") is not True
        or inference.get("resample_dependency_components") is not True
        or inference.get("independent_replication_claim") != "forbidden"
    ):
        raise ContractError("shared-reference covariance contract has drifted")
    execution = config["execution"]
    grec_contract = config["G_domain"].get("G_rec_contract", {})
    if (
        grec_contract.get("hosts") != 4500
        or grec_contract.get("scenes") != 49_500
        or grec_contract.get("condition_cells") != 9
        or grec_contract.get("hosts_per_cell") != 500
        or grec_contract.get("theta_e_arcsec")
        != [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    ):
        raise ContractError("G_rec 4500x11 nine-cell contract has drifted")
    for field in ("source_tree_sha256", "dependency_lock_sha256"):
        require_sha256(execution.get(field), context=f"execution.{field}")
    if not 1 <= int(execution.get("maximum_parallel_jobs", 0)) <= 32:
        raise ContractError("maximum_parallel_jobs must be between 1 and 32")
    if int(execution.get("support_canary_draws", 0)) != 10_000:
        raise ContractError("support canary must exercise all 10,000 draws")
    for field in (
        "feature_extraction_authorized",
        "final_scoring_authorized",
        "final_analysis_authorized",
    ):
        if not isinstance(execution.get(field), bool):
            raise ContractError(f"execution.{field} must be boolean")
    if execution["final_analysis_authorized"]:
        require_sha256(
            execution.get("analysis_authorization_receipt_sha256"),
            context="analysis authorization receipt",
        )


def validate_reference_roster(
    references: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    development_ids: Iterable[str] = (),
) -> pd.DataFrame:
    specification = config["shared_references"]
    id_column = specification["id_column"]
    required = {
        id_column,
        "endpoint_role",
        "label",
        "lensing_status",
        "final_lensing_status",
        "split",
        "physical_group_id",
        "spatial_component_id",
    }
    require_columns(references, required, context="reference roster")
    assert_score_blind(references, context="reference roster")
    result = references.copy()
    require_nonempty_columns(
        result,
        {id_column, "physical_group_id", "spatial_component_id"},
        context="reference roster",
    )
    result[id_column] = result[id_column].astype(str)
    if len(result) != 43 or result[id_column].nunique() != 43:
        raise ContractError("reference roster must contain exactly 43 unique systems")
    checks = {
        "endpoint_role": specification["endpoint_role"],
        "label": 1,
        "lensing_status": specification["lensing_status"],
        "final_lensing_status": specification["lensing_status"],
        "split": specification["fixed_detector_split"],
    }
    for column, expected in checks.items():
        if not result[column].eq(expected).all():
            raise ContractError(f"reference roster has unexpected {column}")
    overlap = set(result[id_column]) & {str(value) for value in development_ids}
    if overlap:
        raise ContractError(
            f"reference roster overlaps detector development: {sorted(overlap)[:5]}"
        )
    return result.sort_values(id_column).reset_index(drop=True)


def validate_reference_payloads(
    references: pd.DataFrame,
    payloads: pd.DataFrame,
    *,
    id_column: str = "galaxy_id",
) -> pd.DataFrame:
    require_columns(
        payloads,
        {id_column, "payload_sha256"},
        context="reference payload manifest",
    )
    assert_score_blind(payloads, context="reference payload manifest")
    require_nonempty_columns(
        payloads, {id_column}, context="reference payload manifest"
    )
    selected = payloads[payloads[id_column].astype(str).isin(references[id_column])]
    if len(selected) != 43 or selected[id_column].astype(str).nunique() != 43:
        raise ContractError("payload manifest does not cover the exact 43 references")
    if "valid" in selected and not selected["valid"].eq(1).all():
        raise ContractError("at least one fixed reference failed payload validation")
    if "bucket" in selected and not selected["bucket"].eq("positive").all():
        raise ContractError("fixed references do not all use positive-prefix payloads")
    if (
        selected["payload_sha256"].isna().any()
        or selected["payload_sha256"].duplicated().any()
    ):
        raise ContractError("reference payload hashes are missing or duplicated")
    require_sha256_column(
        selected,
        "payload_sha256",
        context="reference payload manifest",
        unique=True,
    )
    return selected.sort_values(id_column).reset_index(drop=True)


def assert_same_reference_ids(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    id_column: str = "galaxy_id",
) -> None:
    require_columns(left, {id_column}, context="left reference table")
    require_columns(right, {id_column}, context="right reference table")
    left_ids = set(left[id_column].astype(str))
    right_ids = set(right[id_column].astype(str))
    if left_ids != right_ids or len(left_ids) != 43:
        raise ContractError("G_domain and G_match do not use the same exact 43 IDs")


def derive_catalogue_covariates(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the registered observed host quantities without clipping."""
    require_columns(
        frame,
        {
            "flux_z",
            "sersic",
            "shape_e1",
            "shape_e2",
            "psfsize_r",
            "psfdepth_r",
        },
        context="Tractor catalogue",
    )
    result = frame.copy()
    flux_z = pd.to_numeric(result["flux_z"], errors="coerce").to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result["observed_zmag"] = np.where(
            flux_z > 0.0,
            22.5 - 2.5 * np.log10(flux_z),
            np.nan,
        )
    result["sersic_n"] = pd.to_numeric(result["sersic"], errors="coerce")
    e1 = pd.to_numeric(result["shape_e1"], errors="coerce").to_numpy(float)
    e2 = pd.to_numeric(result["shape_e2"], errors="coerce").to_numpy(float)
    ellipticity = np.hypot(e1, e2)
    axis_ratio = (1.0 - ellipticity) / (1.0 + ellipticity)
    axis_ratio[~np.isfinite(axis_ratio) | (axis_ratio < 0.0) | (axis_ratio > 1.0)] = (
        np.nan
    )
    result["axis_ratio_q"] = axis_ratio
    result["psf_fwhm_r_arcsec"] = pd.to_numeric(result["psfsize_r"], errors="coerce")
    psfdepth = pd.to_numeric(result["psfdepth_r"], errors="coerce").to_numpy(float)
    result["depth5_r_mag"] = np.where(
        psfdepth > 0.0,
        22.5 - 2.5 * np.log10(5.0 / np.sqrt(psfdepth)),
        np.nan,
    )
    return result


def representation_label(
    config: Mapping[str, Any],
    representation_id: str,
    *,
    requested_role: str,
) -> str:
    representations = config["representations"]
    entries = {
        entry["id"]: entry
        for entry in [
            representations["primary"],
            *representations["required_diagnostics"],
        ]
    }
    if representation_id not in entries:
        raise ContractError(f"unregistered representation: {representation_id}")
    entry = entries[representation_id]
    primary_id = representations["primary"]["id"]
    diagnostic_ids = {entry["id"] for entry in representations["required_diagnostics"]}
    if requested_role == "primary" and representation_id != primary_id:
        raise ContractError(f"{representation_id} is not registered as primary")
    if requested_role == "diagnostic" and representation_id not in diagnostic_ids:
        raise ContractError(f"{representation_id} is not a registered diagnostic")
    if requested_role not in {"primary", "diagnostic"}:
        raise ContractError(f"invalid representation role: {requested_role}")
    return str(entry["result_label"])


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".parquet",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".json",
        dir=path.parent,
        delete=False,
    ) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
