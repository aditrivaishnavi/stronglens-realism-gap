"""Prepare, authorize, and execute the final G_match point-estimate analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from stronglens_probes.contracts import assert_score_blind
from stronglens_probes.dependencies import (
    assert_dependency_closed_folds,
    assign_dependency_folds,
    build_dependency_components,
)
from stronglens_probes.probes import crossfit_gmatch_axis

from .contracts import (
    ContractError,
    atomic_write_json,
    atomic_write_parquet,
    canonical_frame_sha256,
    canonical_json_sha256,
    load_config,
    package_source_tree_sha256,
    read_json,
    resolve_bound_path,
    runtime_environment,
    sequence_sha256,
    sha256_file,
    validate_config_seal,
    validate_runtime,
    verify_file_binding,
    write_config_seal,
)

REPRESENTATIONS = (
    "stein_ssl_resnet50",
    "imagenet_effnet_v2_s",
    "pixel_statistics",
)
PREDICTION_COLUMNS = (
    "row_id",
    "pair_id",
    "population",
    "reference_id",
    "target_galaxy_id",
    "dependency_component_id",
    "fold",
    "axis_score",
    "representation",
    "representation_role",
)


def _paths(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    result = root / "results_v2/gmatch/analysis"
    outputs = config["outputs"]
    return {
        "result_dir": result,
        "config": (root / "experiments_v2/gmatch_analysis_v2/config/production.json"),
        "seal": root / "experiments_v2/gmatch_analysis_v2/config/CONFIG_SEAL.json",
        "lock": root / "experiments_v2/gmatch_analysis_v2/uv.lock",
        "fold_map": result / outputs["fold_map"],
        "fold_receipt": result / outputs["fold_receipt"],
        "result_contract": result / outputs["result_contract"],
        "authorization": result / outputs["analysis_authorization"],
        "result_receipt": result / outputs["result_receipt"],
        "metric_recomputation": result / outputs["metric_recomputation"],
    }


def _representation_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = [
        config["representations"]["primary"],
        *config["representations"]["required_diagnostics"],
    ]
    result = {str(entry["id"]): dict(entry) for entry in entries}
    if tuple(result) != REPRESENTATIONS:
        raise ContractError("representation order or mandatory suite differs")
    return result


def _verify_upstream_sources(
    root: Path,
    config: dict[str, Any],
    *,
    require_imported: bool,
) -> dict[str, dict[str, str]]:
    expected = config["upstream_source_closure"]
    if set(expected) != {
        "stronglens_probes",
        "stronglens_probes.authorities",
        "stronglens_probes.authorization",
        "stronglens_probes.contracts",
        "stronglens_probes.dependencies",
        "stronglens_probes.features",
        "stronglens_probes.probes",
    }:
        raise ContractError("executed upstream source closure is incomplete")
    observed: dict[str, dict[str, str]] = {}
    for module_name, binding in sorted(expected.items()):
        path = verify_file_binding(
            root,
            binding,
            context=f"upstream source {module_name}",
        )
        observed[module_name] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
        }
        if require_imported:
            module = sys.modules.get(module_name)
            imported = Path(str(getattr(module, "__file__", ""))).resolve()
            if module is None or imported != path.resolve():
                raise ContractError(
                    f"imported source for {module_name} is not the sealed file"
                )
    if observed != expected:
        raise ContractError("upstream source closure differs from config")
    return observed


def _validate_pre_feature_authorities(
    root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    paths: dict[str, Path] = {}
    for name, binding in config["inputs"].items():
        if name == "old_gmatch_result":
            continue
        paths[name] = verify_file_binding(root, binding, context=name)

    suite = read_json(paths["feature_suite_authority"])
    independent = read_json(paths["feature_independent_validation"])
    stage = read_json(paths["feature_stage_receipt"])
    preparation = read_json(paths["preparation_receipt"])
    preparation_independent = read_json(paths["preparation_independent_validation"])
    preparation_authority = read_json(paths["preparation_authority"])
    feature_authorization = read_json(paths["feature_authorization"])
    environment = read_json(paths["feature_environment_receipt"])
    expected_suite_sha = config["inputs"]["feature_suite_authority"]["sha256"]
    expected_independent_sha = config["inputs"]["feature_independent_validation"][
        "sha256"
    ]
    if (
        suite.get("status") != "authenticated_final_G_match_feature_suite_authority"
        or suite.get("analysis") != "G_match"
        or suite.get("data_role") != "final"
        or suite.get("rows", 129) != 129
        or suite.get("pairs") != 43
        or suite.get("analysis_manifest_sha256")
        != config["inputs"]["analysis_manifest"]["sha256"]
        or suite.get("producer_attribution_uuid") != config["feature_producer_uuid"]
    ):
        raise ContractError(
            "feature-suite authority is not the exact final G_match suite"
        )
    if (
        independent.get("status") != "PASS"
        or independent.get("feature_suite_authority_sha256") != expected_suite_sha
        or independent.get("rows") != 129
        or independent.get("pairs") != 43
        or independent.get("fold_creation_performed") is not False
        or independent.get("probe_fitting_performed") is not False
        or independent.get("G_rec_scores_accessed") is not False
        or independent.get("producer_attribution_uuid")
        != config["feature_producer_uuid"]
        or independent.get("independent_review_uuid") != config["feature_verifier_uuid"]
        or sha256_file(paths["feature_independent_validation"])
        != expected_independent_sha
    ):
        raise ContractError("independent feature validation is not the bound PASS")
    if (
        stage.get("status") != "PASS"
        or stage.get("feature_suite_authority_sha256") != expected_suite_sha
        or stage.get("rows") != 129
        or stage.get("pairs") != 43
        or stage.get("probe_fitting_performed") is not False
        or stage.get("G_rec_scores_accessed") is not False
    ):
        raise ContractError("feature stage receipt is not a clean final PASS")
    if (
        preparation.get("verdict") != "PASS"
        or preparation.get("analysis_rows") != 129
        or preparation.get("matched_references") != 43
        or preparation.get("probe_fitting_performed") is not False
        or preparation.get("detector_scores_used") is not False
        or preparation.get("detector_outcomes_used") is not False
    ):
        raise ContractError("preparation receipt is not the eligible PASS")
    if (
        preparation_independent.get("verdict") != "PASS"
        or preparation_independent.get("analysis_rows") != 129
        or preparation_independent.get("matched_references") != 43
    ):
        raise ContractError("independent preparation validation is not PASS")
    if (
        preparation_authority.get("status")
        != "gmatch_v2_direct_authority_adapters_complete"
        or preparation_authority.get("authority_digest")
        != config["manifest_contract"]["authority_digest"]
    ):
        raise ContractError("preparation authority is not the bound direct authority")
    if (
        feature_authorization.get("status")
        != "authorized_final_G_match_feature_extraction"
        or feature_authorization.get("probe_fitting_authorized") is not False
        or feature_authorization.get("G_rec_score_access_authorized") is not False
        or feature_authorization.get("producer_attribution_uuid")
        != config["feature_producer_uuid"]
    ):
        raise ContractError("upstream feature authorization is inconsistent")
    if (
        environment.get("status") != "exact_locked_feature_environment_verified"
        or environment.get("dependency_lock_sha256")
        != config["inputs"]["feature_lock"]["sha256"]
        or environment.get("producer_attribution_uuid")
        != config["feature_producer_uuid"]
    ):
        raise ContractError("feature environment receipt is inconsistent")

    parent_bindings = suite.get("parent_bindings")
    if not isinstance(parent_bindings, dict) or not parent_bindings:
        raise ContractError("feature-suite authority lacks preparation parent bindings")
    for name, binding in parent_bindings.items():
        verify_file_binding(root, binding, context=f"suite parent {name}")
    feature_source_closure = suite.get("transitive_source_closure")
    if (
        not isinstance(feature_source_closure, dict)
        or canonical_json_sha256(feature_source_closure)
        != suite.get("transitive_source_closure_sha256")
        or suite.get("transitive_source_closure_sha256")
        != config["feature_extraction_source_closure_sha256"]
    ):
        raise ContractError("feature extraction source closure digest differs")
    for name, binding in feature_source_closure.items():
        verify_file_binding(
            root,
            binding,
            context=f"feature extraction source {name}",
        )

    configured = _representation_specs(config)
    suite_representations = suite.get("representations")
    if not isinstance(suite_representations, dict):
        raise ContractError("feature-suite authority lacks representations")
    if set(suite_representations) != set(REPRESENTATIONS):
        raise ContractError("feature-suite representation set differs")
    for representation, specification in configured.items():
        authority_entry = suite_representations[representation]
        expected = {
            "features_sha256": specification["features_sha256"],
            "extraction_receipt_sha256": specification["receipt_sha256"],
            "checkpoint_sha256": specification["checkpoint_sha256"],
            "dimension": specification["dimension"],
        }
        for field, value in expected.items():
            if authority_entry.get(field) != value:
                raise ContractError(
                    f"{representation} suite authority has wrong {field}"
                )
        if (
            authority_entry.get("row_ids_sha256")
            != config["manifest_contract"]["row_ids_sha256"]
            or authority_entry.get("pixel_hashes_sha256")
            != config["manifest_contract"]["pixel_hashes_sha256"]
            or authority_entry.get("transitive_source_closure_sha256")
            != suite.get("transitive_source_closure_sha256")
        ):
            raise ContractError(f"{representation} authority ancestry differs")

    _verify_upstream_sources(root, config, require_imported=True)
    return {
        "paths": paths,
        "suite": suite,
        "independent": independent,
        "stage": stage,
        "feature_authorization": feature_authorization,
        "environment": environment,
    }


def _validate_manifest(
    root: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    path = verify_file_binding(
        root,
        config["inputs"]["analysis_manifest"],
        context="analysis manifest",
    )
    frame = pd.read_parquet(path)
    required = {
        "row_id",
        "population",
        "data_role",
        "pair_id",
        "reference_id",
        "target_galaxy_id",
        "authority_digest",
        "pixel_sha256",
        *config["probe"]["dependency_columns"],
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ContractError(f"analysis manifest lacks columns: {missing}")
    assert_score_blind(frame, context="G_match fold-allocation manifest")
    if (
        len(frame) != 129
        or frame["row_id"].astype(str).nunique() != 129
        or frame["pair_id"].astype(str).nunique() != 43
        or not frame["data_role"].eq("final").all()
        or not frame["authority_digest"]
        .eq(config["manifest_contract"]["authority_digest"])
        .all()
        or frame["population"].value_counts().sort_index().to_dict()
        != {"bare": 43, "injected": 43, "observed_reference": 43}
    ):
        raise ContractError("analysis manifest support or authority differs")
    for pair_id, triplet in frame.groupby("pair_id", sort=False):
        if (
            len(triplet) != 3
            or set(triplet["population"]) != {"observed_reference", "bare", "injected"}
            or triplet["reference_id"].astype(str).nunique() != 1
            or triplet["target_galaxy_id"].astype(str).nunique() != 1
        ):
            raise ContractError(f"{pair_id} is not one exact bound triplet")
    population_references = {
        population: set(part["reference_id"].astype(str))
        for population, part in frame.groupby("population")
    }
    if (
        len(next(iter(population_references.values()))) != 43
        or len({frozenset(value) for value in population_references.values()}) != 1
    ):
        raise ContractError("three populations do not share the exact 43 references")
    checks = {
        "content_sha256": canonical_frame_sha256(
            frame,
            columns=frame.columns.tolist(),
            sort_by=["row_id"],
        ),
        "row_ids_sha256": sequence_sha256(frame["row_id"]),
        "pixel_hashes_sha256": sequence_sha256(frame["pixel_sha256"]),
        "reference_ids_sha256": sequence_sha256(
            sorted(set(frame["reference_id"].astype(str)))
        ),
        "pair_ids_sha256": sequence_sha256(sorted(set(frame["pair_id"].astype(str)))),
    }
    for field, value in checks.items():
        if value != config["manifest_contract"][field]:
            raise ContractError(f"analysis manifest {field} differs")
    return frame


def _componentize_and_allocate(
    manifest: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dependency_columns = list(config["probe"]["dependency_columns"])
    componentized = build_dependency_components(
        manifest,
        row_id_column="row_id",
        dependency_columns=dependency_columns,
    )
    if (
        componentized["pair_id"].nunique() != 43
        or not componentized.groupby("pair_id")["dependency_component_id"]
        .nunique()
        .eq(1)
        .all()
    ):
        raise ContractError(
            "the 43 scientific triplets are not dependency-component closed"
        )
    fit = componentized[
        componentized["population"].isin(["observed_reference", "bare"])
    ].copy()
    fit["fold_label"] = fit["population"].eq("observed_reference").astype(np.int8)
    folds = assign_dependency_folds(
        fit,
        component_column="dependency_component_id",
        label_column="fold_label",
        n_splits=int(config["probe"]["folds"]),
        seed=int(config["probe"]["fold_seed"]),
    )
    assert_dependency_closed_folds(
        componentized,
        folds,
        component_column="dependency_component_id",
        dependency_columns=dependency_columns,
    )
    attached = componentized.merge(
        folds,
        on="dependency_component_id",
        how="left",
        validate="many_to_one",
    )
    if attached.groupby("pair_id")["fold"].nunique().max() != 1:
        raise ContractError("an observed/bare/injected triplet crosses folds")
    return componentized, folds


def _fold_support(
    componentized: pd.DataFrame,
    folds: pd.DataFrame,
) -> list[dict[str, int]]:
    attached = componentized.merge(
        folds,
        on="dependency_component_id",
        how="left",
        validate="many_to_one",
    )
    support = []
    for fold, part in attached.groupby("fold", sort=True):
        counts = part["population"].value_counts().to_dict()
        triplets = int(part["pair_id"].nunique())
        support.append(
            {
                "fold": int(fold),
                "rows": len(part),
                "triplets": triplets,
                "dependency_components": int(part["dependency_component_id"].nunique()),
                "observed_reference_rows": int(counts.get("observed_reference", 0)),
                "bare_rows": int(counts.get("bare", 0)),
                "injected_rows": int(counts.get("injected", 0)),
                "axis_training_rows": 2 * (43 - triplets),
                "axis_training_injected_rows": 0,
            }
        )
    return support


def prepare(root: Path) -> dict[str, Any]:
    """Allocate folds and write contracts before any feature matrix is opened."""
    config, config_path = load_config(root)
    paths = _paths(root, config)
    validate_config_seal(root, config)
    validate_runtime(root, config)
    authorities = _validate_pre_feature_authorities(root, config)
    manifest = _validate_manifest(root, config)
    componentized, folds = _componentize_and_allocate(manifest, config)
    for name in ("fold_map", "fold_receipt", "result_contract", "authorization"):
        if paths[name].exists():
            raise FileExistsError(f"prepare refuses to overwrite {paths[name]}")

    atomic_write_parquet(folds, paths["fold_map"])
    support = _fold_support(componentized, folds)
    fold_receipt = {
        "schema_version": 1,
        "status": "final_G_match_dependency_reference_folds_sealed",
        "analysis": "G_match",
        "data_role": "final",
        "analysis_manifest_sha256": sha256_file(
            authorities["paths"]["analysis_manifest"]
        ),
        "analysis_manifest_content_sha256": config["manifest_contract"][
            "content_sha256"
        ],
        "feature_suite_authority_sha256": config["inputs"]["feature_suite_authority"][
            "sha256"
        ],
        "feature_independent_validation_sha256": config["inputs"][
            "feature_independent_validation"
        ]["sha256"],
        "fold_map_sha256": sha256_file(paths["fold_map"]),
        "fold_map_content_sha256": canonical_frame_sha256(
            folds,
            columns=folds.columns.tolist(),
            sort_by=["dependency_component_id"],
        ),
        "folds": 5,
        "fold_seed": 42,
        "rows": 129,
        "triplets": 43,
        "dependency_components": int(
            componentized["dependency_component_id"].nunique()
        ),
        "fold_support": support,
        "dependency_columns": config["probe"]["dependency_columns"],
        "allocation_fit_populations": ["observed_reference", "bare"],
        "allocation_fit_rows": 86,
        "allocation_injected_rows": 0,
        "allocation_used_feature_bytes": False,
        "allocation_used_scientific_outcomes": False,
        "allocation_class_field": "score_blind population provenance",
        "all_triplets_fold_closed": True,
        "all_registered_dependencies_fold_closed": True,
        "analysis_producer_uuid": config["analysis_producer_uuid"],
    }
    atomic_write_json(fold_receipt, paths["fold_receipt"])

    seal_sha = sha256_file(paths["seal"])
    source_tree_sha = package_source_tree_sha256(
        root / "experiments_v2/gmatch_analysis_v2"
    )
    contract = {
        "schema_version": 1,
        "status": "sealed_final_G_match_result_contract",
        "analysis": "G_match",
        "scope": "final_analysis_only",
        "data_role": "final",
        "config_sha256": sha256_file(config_path),
        "config_seal_sha256": seal_sha,
        "dependency_lock_sha256": sha256_file(paths["lock"]),
        "package_source_tree_sha256": source_tree_sha,
        "upstream_source_closure": config["upstream_source_closure"],
        "feature_extraction_source_closure": authorities["suite"][
            "transitive_source_closure"
        ],
        "feature_extraction_source_closure_sha256": authorities["suite"][
            "transitive_source_closure_sha256"
        ],
        "registered_implementation": config["registered_implementation"],
        "input_bindings": {
            name: dict(binding) for name, binding in config["inputs"].items()
        },
        "representation_bindings": {
            name: {
                "features_path": spec["features_path"],
                "features_sha256": spec["features_sha256"],
                "receipt_path": spec["receipt_path"],
                "receipt_sha256": spec["receipt_sha256"],
                "checkpoint_path": spec["checkpoint_path"],
                "checkpoint_sha256": spec["checkpoint_sha256"],
                "dimension": spec["dimension"],
                "role": spec["role"],
            }
            for name, spec in _representation_specs(config).items()
        },
        "manifest_binding": {
            "rows": 129,
            "triplets": 43,
            **config["manifest_contract"],
        },
        "fold_binding": {
            "fold_map_path": paths["fold_map"].name,
            "fold_map_sha256": sha256_file(paths["fold_map"]),
            "fold_map_content_sha256": fold_receipt["fold_map_content_sha256"],
            "fold_receipt_sha256": sha256_file(paths["fold_receipt"]),
            "folds": 5,
            "fold_seed": 42,
            "triplet_closed": True,
            "reference_closed": True,
            "dependency_closed": True,
        },
        "estimator": {
            "classifier": "l2_logistic_regression",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 5000,
            "estimator_seed": 42,
            "standardisation": "fold_local_standard_scaler",
            "axis_fit_rows": ["observed_reference", "bare"],
            "axis_fit_injected_rows": False,
            "axis_orientation": "toward_observed_reference",
            "movement_scale": "training_bare_mean_and_sd",
        },
        "metrics": {
            "primary_representation": "stein_ssl_resnet50",
            "primary": "equal_target_mean_injected_minus_bare_axis_movement",
            "mandatory_diagnostic_representations": [
                "imagenet_effnet_v2_s",
                "pixel_statistics",
            ],
            "diagnostics": [
                "pooled_observed_vs_bare_auc",
                "pooled_observed_vs_injected_auc",
                "pooled_injected_vs_bare_auc",
                "paired_injected_above_bare_concordance",
            ],
            "pooled_AUCs_are_paired_effects": False,
            "scope": "finite_cohort_model_conditional_point_estimates_only",
            "formal_confidence_interval": None,
            "significance_test": None,
        },
        "outputs": config["outputs"],
        "analysis_producer_uuid": config["analysis_producer_uuid"],
        "metric_verifier_uuid": config["metric_verifier_uuid"],
        "feature_producer_uuid": config["feature_producer_uuid"],
        "feature_verifier_uuid": config["feature_verifier_uuid"],
        "independent_feature_pass_uuid": config["independent_feature_pass_uuid"],
        "contract_creation_used_feature_bytes": False,
        "contract_creation_used_scientific_outcomes": False,
        "bootstrap_permitted": False,
        "G_rec_score_access_permitted": False,
        "manuscript_edit_permitted": False,
        "independent_replication_claim_permitted": False,
    }
    atomic_write_json(contract, paths["result_contract"])

    authorization = {
        "schema_version": 1,
        "status": "authorized_final_G_match_analysis",
        "authorization_kind": "analysis_only",
        "authorization_basis": (
            "The user explicitly authorized implementation and execution of the "
            "narrow final G_match analysis after feature PASS "
            f"{config['independent_feature_pass_uuid']} on 2026-09-16."
        ),
        "analysis": "G_match",
        "data_role": "final",
        "allowed_operations": [
            "read_authenticated_feature_matrices",
            "fit_registered_observed_vs_bare_axes",
            "score_held_out_triplets",
            "write_OOF_predictions",
            "write_final_result_receipt",
            "recompute_metrics_from_predictions",
        ],
        "result_contract_sha256": sha256_file(paths["result_contract"]),
        "config_sha256": sha256_file(config_path),
        "config_seal_sha256": seal_sha,
        "fold_map_sha256": sha256_file(paths["fold_map"]),
        "fold_receipt_sha256": sha256_file(paths["fold_receipt"]),
        "feature_suite_authority_sha256": config["inputs"]["feature_suite_authority"][
            "sha256"
        ],
        "feature_independent_validation_sha256": config["inputs"][
            "feature_independent_validation"
        ]["sha256"],
        "analysis_manifest_sha256": config["inputs"]["analysis_manifest"]["sha256"],
        "preparation_authority_sha256": config["inputs"]["preparation_authority"][
            "sha256"
        ],
        "representations": contract["representation_bindings"],
        "upstream_source_closure": config["upstream_source_closure"],
        "feature_extraction_source_closure": contract[
            "feature_extraction_source_closure"
        ],
        "feature_extraction_source_closure_sha256": contract[
            "feature_extraction_source_closure_sha256"
        ],
        "analysis_dependency_lock_sha256": sha256_file(paths["lock"]),
        "feature_dependency_lock_sha256": config["inputs"]["feature_lock"]["sha256"],
        "feature_access_authorized": True,
        "probe_fitting_authorized": True,
        "fold_creation_authorized": True,
        "bootstrap_authorized": False,
        "detector_score_access_authorized": False,
        "detector_outcome_access_authorized": False,
        "G_rec_score_access_authorized": False,
        "manuscript_edit_authorized": False,
        "feature_bytes_opened_before_authorization": False,
        "scientific_outcomes_accessed_before_authorization": False,
        "analysis_producer_uuid": config["analysis_producer_uuid"],
        "metric_verifier_uuid": config["metric_verifier_uuid"],
        "feature_producer_uuid": config["feature_producer_uuid"],
        "feature_verifier_uuid": config["feature_verifier_uuid"],
        "independent_feature_pass_uuid": config["independent_feature_pass_uuid"],
    }
    atomic_write_json(authorization, paths["authorization"])
    return {
        "status": "prepared_and_authorized",
        "feature_bytes_opened": False,
        "fold_map_sha256": sha256_file(paths["fold_map"]),
        "fold_receipt_sha256": sha256_file(paths["fold_receipt"]),
        "result_contract_sha256": sha256_file(paths["result_contract"]),
        "analysis_authorization_sha256": sha256_file(paths["authorization"]),
        "fold_support": support,
    }


def _validate_contract_and_authorization(
    root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = read_json(paths["result_contract"])
    authorization = read_json(paths["authorization"])
    seal = validate_config_seal(root, config)
    expected_common = {
        "config_sha256": seal["config_sha256"],
        "config_seal_sha256": sha256_file(paths["seal"]),
    }
    if (
        contract.get("status") != "sealed_final_G_match_result_contract"
        or contract.get("analysis") != "G_match"
        or contract.get("scope") != "final_analysis_only"
        or contract.get("registered_implementation")
        != config["registered_implementation"]
        or contract.get("feature_extraction_source_closure_sha256")
        != config["feature_extraction_source_closure_sha256"]
        or contract.get("input_bindings") != config["inputs"]
        or contract.get("package_source_tree_sha256")
        != package_source_tree_sha256(root / "experiments_v2/gmatch_analysis_v2")
        or contract.get("dependency_lock_sha256") != sha256_file(paths["lock"])
        or contract.get("bootstrap_permitted") is not False
        or contract.get("G_rec_score_access_permitted") is not False
        or contract.get("manuscript_edit_permitted") is not False
        or any(contract.get(key) != value for key, value in expected_common.items())
    ):
        raise ContractError("final result contract is stale or invalid")
    _validate_authorization_payload(
        authorization,
        config=config,
        expected_contract_sha256=sha256_file(paths["result_contract"]),
        expected_config_sha256=expected_common["config_sha256"],
        expected_config_seal_sha256=expected_common["config_seal_sha256"],
    )
    return contract, authorization


def _validate_authorization_payload(
    authorization: dict[str, Any],
    *,
    config: dict[str, Any],
    expected_contract_sha256: str,
    expected_config_sha256: str,
    expected_config_seal_sha256: str,
) -> None:
    """Validate exact stage authority without opening a feature matrix."""
    expected_operations = [
        "read_authenticated_feature_matrices",
        "fit_registered_observed_vs_bare_axes",
        "score_held_out_triplets",
        "write_OOF_predictions",
        "write_final_result_receipt",
        "recompute_metrics_from_predictions",
    ]
    if (
        authorization.get("status") != "authorized_final_G_match_analysis"
        or authorization.get("authorization_kind") != "analysis_only"
        or authorization.get("allowed_operations") != expected_operations
        or authorization.get("result_contract_sha256") != expected_contract_sha256
        or authorization.get("feature_suite_authority_sha256")
        != config["inputs"]["feature_suite_authority"]["sha256"]
        or authorization.get("feature_independent_validation_sha256")
        != config["inputs"]["feature_independent_validation"]["sha256"]
        or authorization.get("analysis_manifest_sha256")
        != config["inputs"]["analysis_manifest"]["sha256"]
        or authorization.get("feature_access_authorized") is not True
        or authorization.get("probe_fitting_authorized") is not True
        or authorization.get("fold_creation_authorized") is not True
        or authorization.get("bootstrap_authorized") is not False
        or authorization.get("detector_score_access_authorized") is not False
        or authorization.get("detector_outcome_access_authorized") is not False
        or authorization.get("G_rec_score_access_authorized") is not False
        or authorization.get("manuscript_edit_authorized") is not False
        or authorization.get("feature_bytes_opened_before_authorization") is not False
        or authorization.get("analysis_producer_uuid")
        != config["analysis_producer_uuid"]
        or authorization.get("metric_verifier_uuid") != config["metric_verifier_uuid"]
        or authorization.get("feature_producer_uuid") != config["feature_producer_uuid"]
        or authorization.get("feature_verifier_uuid") != config["feature_verifier_uuid"]
        or authorization.get("independent_feature_pass_uuid")
        != config["independent_feature_pass_uuid"]
        or authorization.get("upstream_source_closure")
        != config["upstream_source_closure"]
        or authorization.get("feature_extraction_source_closure_sha256")
        != config["feature_extraction_source_closure_sha256"]
        or authorization.get("feature_dependency_lock_sha256")
        != config["inputs"]["feature_lock"]["sha256"]
        or authorization.get("preparation_authority_sha256")
        != config["inputs"]["preparation_authority"]["sha256"]
        or authorization.get("config_sha256") != expected_config_sha256
        or authorization.get("config_seal_sha256") != expected_config_seal_sha256
    ):
        raise ContractError("final analysis authorization is not exact and fail-closed")


def _validate_fold_artifacts(
    config: dict[str, Any],
    paths: dict[str, Path],
    manifest: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    componentized, expected = _componentize_and_allocate(manifest, config)
    observed = pd.read_parquet(paths["fold_map"])
    try:
        pd.testing.assert_frame_equal(observed, expected, check_exact=True)
    except AssertionError as error:
        raise ContractError("fold map differs from deterministic allocation") from error
    receipt = read_json(paths["fold_receipt"])
    checks = {
        "fold_map_sha256": sha256_file(paths["fold_map"]),
        "fold_map_content_sha256": canonical_frame_sha256(
            observed,
            columns=observed.columns.tolist(),
            sort_by=["dependency_component_id"],
        ),
        "fold_receipt_sha256": sha256_file(paths["fold_receipt"]),
    }
    if (
        receipt.get("status") != "final_G_match_dependency_reference_folds_sealed"
        or receipt.get("fold_support") != _fold_support(componentized, observed)
        or receipt.get("allocation_used_feature_bytes") is not False
        or receipt.get("allocation_injected_rows") != 0
        or any(
            contract["fold_binding"].get(key) != value for key, value in checks.items()
        )
    ):
        raise ContractError("fold authority is stale or inconsistent")
    return componentized, observed


def _validate_feature_inputs(
    root: Path,
    config: dict[str, Any],
    authorities: dict[str, Any],
) -> dict[str, SimpleNamespace]:
    suite = authorities["suite"]
    manifest_sha = config["inputs"]["analysis_manifest"]["sha256"]
    artifacts: dict[str, SimpleNamespace] = {}
    for representation, spec in _representation_specs(config).items():
        features_path = resolve_bound_path(root, spec["features_path"])
        receipt_path = resolve_bound_path(root, spec["receipt_path"])
        if sha256_file(features_path) != spec["features_sha256"]:
            raise ContractError(f"{representation} feature matrix hash differs")
        if sha256_file(receipt_path) != spec["receipt_sha256"]:
            raise ContractError(f"{representation} feature receipt hash differs")
        receipt = read_json(receipt_path)
        authority_entry = suite["representations"][representation]
        if (
            receipt.get("status") != "complete"
            or receipt.get("schema_version") != 2
            or receipt.get("representation_id") != representation
            or receipt.get("analysis_manifest_sha256") != manifest_sha
            or receipt.get("features_file_sha256") != spec["features_sha256"]
            or receipt.get("feature_dimension") != spec["dimension"]
            or receipt.get("dtype") != "float32"
            or receipt.get("rows") != 129
            or receipt.get("row_ids_sha256")
            != config["manifest_contract"]["row_ids_sha256"]
            or receipt.get("pixel_hashes_sha256")
            != config["manifest_contract"]["pixel_hashes_sha256"]
            or receipt.get("producer_attribution_uuid")
            != config["feature_producer_uuid"]
            or receipt.get("checkpoint_sha256") != spec["checkpoint_sha256"]
            or authority_entry.get("features_sha256") != spec["features_sha256"]
            or authority_entry.get("extraction_receipt_sha256")
            != spec["receipt_sha256"]
        ):
            raise ContractError(f"{representation} feature receipt is inconsistent")
        checkpoint_path = spec["checkpoint_path"]
        if checkpoint_path is None:
            if spec["checkpoint_sha256"] is not None:
                raise ContractError("pixel diagnostic has an unexpected checkpoint")
        else:
            checkpoint = Path(checkpoint_path)
            if (
                not checkpoint.is_file()
                or sha256_file(checkpoint) != spec["checkpoint_sha256"]
            ):
                raise ContractError(f"{representation} checkpoint bytes differ")
        values = np.load(features_path, mmap_mode="r", allow_pickle=False)
        if values.shape != (129, int(spec["dimension"])) or values.dtype != np.float32:
            raise ContractError(f"{representation} feature shape or dtype differs")
        for start in range(0, len(values), 32):
            if not np.isfinite(values[start : start + 32]).all():
                raise ContractError(f"{representation} feature matrix is non-finite")
        artifacts[representation] = SimpleNamespace(
            representation_id=representation,
            manifest_path=resolve_bound_path(
                root, config["inputs"]["analysis_manifest"]["path"]
            ),
            features_path=features_path,
            receipt_sha256=spec["receipt_sha256"],
            authority_digest=config["manifest_contract"]["authority_digest"],
        )
    return artifacts


class _AnalysisAuthority:
    """Analysis-local view of already authenticated reference metadata."""

    def __init__(self, manifest: pd.DataFrame, authority_digest: str) -> None:
        observed = manifest[manifest["population"].eq("observed_reference")]
        if len(observed) != 43 or observed["reference_id"].nunique() != 43:
            raise ContractError("local reference authority is not exactly 43 rows")
        self.authority_digest = authority_digest
        self._payload = dict(
            zip(
                observed["reference_id"].astype(str),
                observed["pixel_sha256"].astype(str),
                strict=True,
            )
        )
        self._physical = dict(
            zip(
                observed["reference_id"].astype(str),
                observed["physical_group_id"].astype(str),
                strict=True,
            )
        )
        self._spatial = dict(
            zip(
                observed["reference_id"].astype(str),
                observed["spatial_group_id"].astype(str),
                strict=True,
            )
        )
        self._psf = dict(
            zip(
                observed["reference_id"].astype(str),
                observed["psf_id"].astype(str),
                strict=True,
            )
        )
        self._render = dict(
            zip(
                observed["reference_id"].astype(str),
                observed["render_parameter_id"].astype(str),
                strict=True,
            )
        )

    @staticmethod
    def _assert_mapping(
        frame: pd.DataFrame,
        id_column: str,
        value_column: str,
        expected: dict[str, str],
        context: str,
    ) -> None:
        observed = dict(
            zip(
                frame[id_column].astype(str),
                frame[value_column].astype(str),
                strict=True,
            )
        )
        if len(frame) != 43 or observed != expected:
            raise ContractError(f"{context} differs from authenticated references")

    def assert_reference_payloads(
        self,
        frame: pd.DataFrame,
        *,
        id_column: str,
        payload_column: str,
        exact: bool,
        context: str,
    ) -> None:
        del exact
        self._assert_mapping(frame, id_column, payload_column, self._payload, context)

    def assert_reference_metadata(
        self,
        frame: pd.DataFrame,
        *,
        id_column: str,
        physical_column: str,
        spatial_column: str,
        exact: bool,
        context: str,
    ) -> None:
        del exact
        self._assert_mapping(frame, id_column, physical_column, self._physical, context)
        self._assert_mapping(frame, id_column, spatial_column, self._spatial, context)

    def assert_reference_render_metadata(
        self,
        frame: pd.DataFrame,
        *,
        id_column: str,
        psf_column: str,
        render_column: str,
        exact: bool,
        context: str,
    ) -> None:
        del exact
        self._assert_mapping(frame, id_column, psf_column, self._psf, context)
        self._assert_mapping(frame, id_column, render_column, self._render, context)


def _temporary_component_manifest(
    componentized: pd.DataFrame,
    directory: Path,
) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=".gmatch-componentized.",
        suffix=".parquet",
        dir=directory,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    componentized.to_parquet(temporary, index=False)
    return temporary


def _pooled_auc(wide: pd.DataFrame, positive: str, negative: str) -> float:
    labels = np.r_[np.ones(len(wide)), np.zeros(len(wide))]
    scores = np.r_[wide[positive].to_numpy(), wide[negative].to_numpy()]
    return float(roc_auc_score(labels, scores))


def _metrics_from_predictions(frame: pd.DataFrame) -> dict[str, Any]:
    wide = frame.pivot(
        index=["pair_id", "reference_id", "target_galaxy_id"],
        columns="population",
        values="axis_score",
    ).reset_index()
    wide.columns.name = None
    if len(wide) != 43 or wide.isna().any().any():
        raise ContractError("OOF predictions do not form 43 complete triplets")
    movement = wide["injected"].to_numpy() - wide["bare"].to_numpy()
    above = int(np.sum(movement > 0.0))
    ties = int(np.sum(movement == 0.0))
    below = int(np.sum(movement < 0.0))
    return {
        "primary_within_target_movement": float(np.mean(movement)),
        "diagnostics": {
            "pooled_observed_vs_bare_auc": _pooled_auc(
                wide, "observed_reference", "bare"
            ),
            "pooled_observed_vs_injected_auc": _pooled_auc(
                wide, "observed_reference", "injected"
            ),
            "pooled_injected_vs_bare_auc": _pooled_auc(wide, "injected", "bare"),
            "paired_injected_above_bare_concordance": (above + 0.5 * ties) / 43.0,
            "paired_counts": {
                "injected_above_bare": above,
                "ties": ties,
                "injected_below_bare": below,
                "total": 43,
            },
        },
        "movement_distribution": {
            "minimum": float(np.min(movement)),
            "median": float(np.median(movement)),
            "maximum": float(np.max(movement)),
        },
    }


def _assert_result_metrics(
    registered: dict[str, Any],
    recomputed: dict[str, Any],
) -> None:
    pairs = [
        (
            float(registered["primary_within_target_movement"]),
            float(recomputed["primary_within_target_movement"]),
            "primary movement",
        )
    ]
    for name in (
        "pooled_observed_vs_bare_auc",
        "pooled_observed_vs_injected_auc",
        "pooled_injected_vs_bare_auc",
        "paired_injected_above_bare_concordance",
    ):
        pairs.append(
            (
                float(registered["diagnostics"][name]),
                float(recomputed["diagnostics"][name]),
                name,
            )
        )
    for observed, expected, context in pairs:
        if not np.isclose(observed, expected, rtol=0.0, atol=2e-12):
            raise ContractError(
                f"{context} differs between registered output and predictions"
            )


def _old_result_comparison(
    root: Path,
    config: dict[str, Any],
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    old_path = verify_file_binding(
        root,
        config["inputs"]["old_gmatch_result"],
        context="old G_match result",
    )
    old = read_json(old_path)
    comparisons: dict[str, Any] = {}
    for representation in REPRESENTATIONS:
        entry = old["results"][representation]
        comparisons[representation] = {
            "old_separately_fitted_observed_vs_bare_auc": entry["real_vs_bare_target"][
                "pooled_oof_auc"
            ],
            "old_separately_fitted_observed_vs_injected_auc": entry[
                "real_vs_injection"
            ]["pooled_oof_auc"],
            "old_separately_fitted_injected_vs_bare_auc": entry[
                "injection_vs_own_bare_target"
            ]["pooled_oof_auc"],
            "new_shared_axis_pooled_observed_vs_bare_auc": results[representation][
                "diagnostics"
            ]["pooled_observed_vs_bare_auc"],
            "new_shared_axis_pooled_observed_vs_injected_auc": results[representation][
                "diagnostics"
            ]["pooled_observed_vs_injected_auc"],
            "new_shared_axis_pooled_injected_vs_bare_auc": results[representation][
                "diagnostics"
            ]["pooled_injected_vs_bare_auc"],
            "old_primary_movement_analogue": None,
        }
    return {
        "old_result_path": config["inputs"]["old_gmatch_result"]["path"],
        "old_result_sha256": sha256_file(old_path),
        "old_cohort": {"references": 64, "matched_targets": 448},
        "new_cohort": {"references": 43, "matched_targets": 43},
        "representations": comparisons,
        "comparability": (
            "Context only: the old exploratory AUCs used separate comparison-specific "
            "axes and a 64-reference/448-target cohort; the new diagnostics use one "
            "observed-vs-bare axis and the exact terminal 43 triplets. There is no "
            "old analogue for the registered within-target movement."
        ),
    }


def _commit_predictions_and_receipt(
    predictions: dict[str, pd.DataFrame],
    receipt: dict[str, Any],
    prediction_paths: dict[str, Path],
    receipt_path: Path,
) -> None:
    temporary: dict[str, Path] = {}
    moved: list[Path] = []
    try:
        for representation, frame in predictions.items():
            destination = prediction_paths[representation]
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination.name}.",
                suffix=".parquet",
                dir=destination.parent,
                delete=False,
            ) as handle:
                path = Path(handle.name)
            frame.to_parquet(path, index=False)
            temporary[representation] = path
        receipt["outputs"] = {
            representation: {
                "path": prediction_paths[representation].name,
                "sha256": sha256_file(temporary[representation]),
                "content_sha256": canonical_frame_sha256(
                    predictions[representation],
                    columns=predictions[representation].columns.tolist(),
                    sort_by=["row_id"],
                ),
                "rows": len(predictions[representation]),
            }
            for representation in REPRESENTATIONS
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{receipt_path.name}.",
            suffix=".json",
            dir=receipt_path.parent,
            delete=False,
        ) as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            temporary["receipt"] = Path(handle.name)
        for representation in REPRESENTATIONS:
            destination = prediction_paths[representation]
            os.replace(temporary[representation], destination)
            moved.append(destination)
        os.replace(temporary["receipt"], receipt_path)
        moved.append(receipt_path)
    except Exception:
        for path in moved:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def run(root: Path) -> dict[str, Any]:
    """Authenticate all bytes, run all representations, and commit final outputs."""
    config, _ = load_config(root)
    paths = _paths(root, config)
    validate_runtime(root, config)
    contract, authorization = _validate_contract_and_authorization(root, config, paths)
    authorities = _validate_pre_feature_authorities(root, config)
    manifest = _validate_manifest(root, config)
    componentized, folds = _validate_fold_artifacts(config, paths, manifest, contract)
    artifacts = _validate_feature_inputs(root, config, authorities)

    prediction_paths = {
        representation: paths["result_dir"]
        / config["outputs"]["prediction_files"][representation]
        for representation in REPRESENTATIONS
    }
    for output in [*prediction_paths.values(), paths["result_receipt"]]:
        if output.exists():
            raise FileExistsError(f"run refuses to overwrite {output}")

    local_authority = _AnalysisAuthority(
        componentized,
        config["manifest_contract"]["authority_digest"],
    )
    temporary_manifest = _temporary_component_manifest(
        componentized,
        paths["result_dir"],
    )
    results: dict[str, dict[str, Any]] = {}
    predictions: dict[str, pd.DataFrame] = {}
    try:
        for representation, artifact in artifacts.items():
            artifact.manifest_path = temporary_manifest
            result, frame, _ = crossfit_gmatch_axis(
                artifact,
                folds,
                config=config,
                authority=local_authority,
                _prevalidated=True,
            )
            role = _representation_specs(config)[representation]["role"]
            frame["representation"] = representation
            frame["representation_role"] = role
            frame = frame[list(PREDICTION_COLUMNS)]
            recomputed = _metrics_from_predictions(frame)
            _assert_result_metrics(result, recomputed)
            result["diagnostics"]["paired_counts"] = recomputed["diagnostics"][
                "paired_counts"
            ]
            result["axis_orientation"] = "toward_observed_reference"
            result["primary_is_paired_within_target_movement"] = True
            result["pooled_AUC_diagnostics_are_nonpaired"] = True
            results[representation] = result
            predictions[representation] = frame
    finally:
        temporary_manifest.unlink(missing_ok=True)

    identity = [
        "row_id",
        "pair_id",
        "population",
        "reference_id",
        "target_galaxy_id",
        "dependency_component_id",
        "fold",
    ]
    baseline = predictions[REPRESENTATIONS[0]][identity]
    for representation in REPRESENTATIONS[1:]:
        try:
            pd.testing.assert_frame_equal(
                baseline,
                predictions[representation][identity],
                check_exact=True,
            )
        except AssertionError as error:
            raise ContractError(
                "representations do not share exact rows, pairs, and folds"
            ) from error

    output_bindings: dict[str, dict[str, Any]] = {}
    temp_bindings: dict[str, Path] = {}
    try:
        for representation, frame in predictions.items():
            destination = prediction_paths[representation]
            with tempfile.NamedTemporaryFile(
                prefix=f".binding-{destination.name}.",
                suffix=".parquet",
                dir=destination.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            frame.to_parquet(temporary, index=False)
            temp_bindings[representation] = temporary
            output_bindings[representation] = {
                "path": destination.name,
                "sha256": sha256_file(temporary),
                "content_sha256": canonical_frame_sha256(
                    frame,
                    columns=frame.columns.tolist(),
                    sort_by=["row_id"],
                ),
                "rows": len(frame),
            }
        old_comparison = _old_result_comparison(root, config, results)
        receipt = {
            "schema_version": 1,
            "status": "complete_final_G_match_linear_axis_analysis",
            "analysis": "G_match",
            "scope": "final_analysis_only",
            "data_role": "final",
            "analysis_producer_uuid": config["analysis_producer_uuid"],
            "metric_verifier_uuid": config["metric_verifier_uuid"],
            "feature_producer_uuid": config["feature_producer_uuid"],
            "feature_verifier_uuid": config["feature_verifier_uuid"],
            "independent_feature_pass_uuid": config["independent_feature_pass_uuid"],
            "config_sha256": sha256_file(paths["config"]),
            "config_seal_sha256": sha256_file(paths["seal"]),
            "result_contract_sha256": sha256_file(paths["result_contract"]),
            "analysis_authorization_sha256": sha256_file(paths["authorization"]),
            "analysis_manifest_sha256": config["inputs"]["analysis_manifest"]["sha256"],
            "feature_suite_authority_sha256": config["inputs"][
                "feature_suite_authority"
            ]["sha256"],
            "feature_independent_validation_sha256": config["inputs"][
                "feature_independent_validation"
            ]["sha256"],
            "preparation_authority_sha256": config["inputs"]["preparation_authority"][
                "sha256"
            ],
            "analysis_dependency_lock_sha256": sha256_file(paths["lock"]),
            "feature_dependency_lock_sha256": config["inputs"]["feature_lock"][
                "sha256"
            ],
            "package_source_tree_sha256": package_source_tree_sha256(
                root / "experiments_v2/gmatch_analysis_v2"
            ),
            "upstream_source_closure": config["upstream_source_closure"],
            "feature_extraction_source_closure": authorities["suite"][
                "transitive_source_closure"
            ],
            "feature_extraction_source_closure_sha256": authorities["suite"][
                "transitive_source_closure_sha256"
            ],
            "runtime_environment": runtime_environment(),
            "registered_function": config["registered_implementation"]["function"],
            "registered_function_sha256": config["registered_implementation"]["sha256"],
            "fold_map_sha256": sha256_file(paths["fold_map"]),
            "fold_receipt_sha256": sha256_file(paths["fold_receipt"]),
            "fold_support": _fold_support(componentized, folds),
            "outputs": output_bindings,
            "results": results,
            "primary": {
                "representation": "stein_ssl_resnet50",
                "metric": "equal_target_mean_injected_minus_bare_axis_movement",
                "point_estimate": results["stein_ssl_resnet50"][
                    "primary_within_target_movement"
                ],
                "targets": 43,
            },
            "old_result_comparison": old_comparison,
            "all_metrics_recomputed_from_OOF_predictions_before_commit": True,
            "mandatory_diagnostics_complete": True,
            "feature_bytes_opened_only_after_authorization_validation": True,
            "injected_rows_used_to_fit_axis_or_scale": False,
            "axis_oriented_toward_observed_reference": True,
            "pooled_AUCs_are_nonpaired_diagnostics": True,
            "uncertainty_scope": (
                "finite_cohort_model_conditional_point_estimates_only"
            ),
            "formal_confidence_interval": None,
            "significance_test": None,
            "bootstrap_run": False,
            "causal_attribution": False,
            "realism_percentage_reported": False,
            "independent_replication_claimed": False,
            "G_rec_scores_accessed": False,
            "detector_scores_accessed": False,
            "detector_outcomes_accessed": False,
            "manuscript_edited": False,
            "ready_for_separate_final_verification": True,
            "authorization_digest": canonical_json_sha256(authorization),
        }
    finally:
        for temporary in temp_bindings.values():
            temporary.unlink(missing_ok=True)

    _commit_predictions_and_receipt(
        predictions,
        receipt,
        prediction_paths,
        paths["result_receipt"],
    )
    return {
        "status": "complete",
        "result_receipt_sha256": sha256_file(paths["result_receipt"]),
        "primary_movement": receipt["primary"]["point_estimate"],
        "results": {
            representation: {
                "movement": result["primary_within_target_movement"],
                **result["diagnostics"],
            }
            for representation, result in results.items()
        },
        "fold_support": receipt["fold_support"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "prepare", "run"))
    parser.add_argument("--repository-root", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.repository_root.resolve()
    if arguments.command == "seal":
        result = write_config_seal(root)
    elif arguments.command == "prepare":
        result = prepare(root)
    else:
        result = run(root)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
