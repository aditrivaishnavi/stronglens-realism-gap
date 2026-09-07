"""Seal, authorize, and run the final hash-bound G_domain linear probe.

The ``prepare`` and ``authorize`` commands do not open feature matrices or any
scientific outcome.  The ``run`` command is deliberately separate and refuses
to start until the resulting contract and final authorization are config-pinned.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn
import yaml

from stronglens_probes.authorities import load_authority_context
from stronglens_probes.authorization import (
    load_execution_authorization,
    protocol_config_sha256,
)
from stronglens_probes.contracts import (
    ContractError,
    assert_score_blind,
    atomic_write_json,
    atomic_write_parquet,
    canonical_frame_sha256,
    require_columns,
    sha256_file,
)
from stronglens_probes.dependencies import (
    assert_dependency_closed_folds,
    assign_dependency_folds,
    build_dependency_components,
)
from stronglens_probes.execution import _source_tree_sha256
from stronglens_probes.features import (
    domain_parent_bindings,
    load_feature_artifact,
    load_feature_suite_authority,
    recompute_domain_manifest,
)
from stronglens_probes.gdomain import load_gdomain_subset
from stronglens_probes.parents import load_grec_artifacts
from stronglens_probes.probes import crossfit_domain

REPRESENTATIONS = (
    "stein_ssl_resnet50",
    "imagenet_effnet_v2_s",
    "pixel_statistics",
)

PREDICTION_COLUMNS = (
    "row_id",
    "population",
    "reference_id",
    "host_galaxy_id",
    "dependency_component_id",
    "fold",
    "label",
    "analysis_weight",
    "oof_score",
    "representation",
    "representation_role",
)


def _paths(root: Path) -> dict[str, Path]:
    result = root / "results_v2/probes/gdomain_v2"
    authorities = root / "results_v2/probes/authorities"
    evaluation = root / "evaluation_dataset_prep"
    return {
        "config": root / "experiments_v2/probes/configs/probes.yaml",
        "lock": root / "experiments_v2/probes/uv.lock",
        "runner": Path(__file__).resolve(),
        "independent_verifier": (
            Path(__file__).resolve().parent
            / "independent_verify_final_gdomain.py"
        ),
        "references": evaluation / "reference_cohort/confirmed_43.parquet",
        "reference_payload_source": (
            evaluation
            / "reference_cohort/confirmed_43_payload_validated.parquet"
        ),
        "host_universe_source": evaluation / "role_proposal/proposed_roles.parquet",
        "reference_adapter": authorities / "reference_payload_adapter.parquet",
        "host_adapter": authorities / "host_authority_adapter.parquet",
        "tractor_adapter": authorities / "host_tractor_adapter.parquet",
        "parent_adapter": authorities / "grec_parent_adapter.parquet",
        "adapter_receipt": authorities / "FINAL_SCHEMA_ADAPTER_RECEIPT.json",
        "manifest": result / "gdomain_analysis_manifest.parquet",
        "subset": result / "gdomain_subset.parquet",
        "subset_receipt": result / "GDOMAIN_SUBSET_RECEIPT.json",
        "preparation_receipt": result / "GDOMAIN_PREPARATION_RECEIPT.json",
        "extraction_authorization": (
            result / "GDOMAIN_EXTRACTION_AUTHORIZATION.json"
        ),
        "checkpoint_verification": (
            result / "ENCODER_CHECKPOINT_VERIFICATION.json"
        ),
        "suite_authority": result / "GDOMAIN_FEATURE_SUITE_AUTHORITY.json",
        "stein_receipt": result / "STEIN_SSL_RESNET50_FEATURE_RECEIPT.json",
        "imagenet_receipt": (
            result / "IMAGENET_EFFNET_V2_S_FEATURE_RECEIPT.json"
        ),
        "pixel_receipt": result / "PIXEL_STATISTICS_FEATURE_RECEIPT.json",
        "stein_features": result / "stein_ssl_resnet50_features.npy",
        "imagenet_features": result / "imagenet_effnet_v2_s_features.npy",
        "pixel_features": result / "pixel_statistics_features.npy",
        "fold_map": result / "gdomain_dependency_folds.parquet",
        "fold_receipt": result / "GDOMAIN_FOLD_RECEIPT.json",
        "result_contract": result / "GDOMAIN_FINAL_RESULT_CONTRACT.json",
        "analysis_authorization": (
            result / "GDOMAIN_FINAL_ANALYSIS_AUTHORIZATION.json"
        ),
        "result_receipt": result / "GDOMAIN_FINAL_RESULT_RECEIPT.json",
        "independent_verification": (
            result / "GDOMAIN_FINAL_INDEPENDENT_VERIFICATION.json"
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return payload


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError("probe config must be a mapping")
    return payload


def _componentized_manifest(
    manifest: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    assert_score_blind(manifest, context="G_domain fold-allocation manifest")
    dependency_columns = list(config["probe"]["dependency_columns"])
    require_columns(
        manifest,
        {"row_id", "population", *dependency_columns},
        context="G_domain fold-allocation manifest",
    )
    if set(manifest["population"]) != {"observed_reference", "simulated"}:
        raise ContractError("G_domain manifest has unexpected populations")
    componentized = build_dependency_components(
        manifest,
        row_id_column="row_id",
        dependency_columns=dependency_columns,
    )
    componentized["fold_label"] = (
        componentized["population"].eq("observed_reference").astype(np.int8)
    )
    return componentized


def _expected_fold_map(
    componentized: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    folds = assign_dependency_folds(
        componentized,
        component_column="dependency_component_id",
        label_column="fold_label",
        n_splits=int(config["probe"]["folds"]),
        seed=int(config["probe"]["fold_seed"]),
    )
    assert_dependency_closed_folds(
        componentized,
        folds,
        component_column="dependency_component_id",
        dependency_columns=config["probe"]["dependency_columns"],
    )
    return folds


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
    support: list[dict[str, int]] = []
    for fold, frame in attached.groupby("fold", sort=True):
        observed = frame["fold_label"].eq(1)
        simulated = ~observed
        support.append(
            {
                "fold": int(fold),
                "rows": len(frame),
                "observed_rows": int(observed.sum()),
                "simulated_rows": int(simulated.sum()),
                "dependency_components": int(
                    frame["dependency_component_id"].nunique()
                ),
                "observed_dependency_components": int(
                    frame.loc[observed, "dependency_component_id"].nunique()
                ),
                "simulated_dependency_components": int(
                    frame.loc[simulated, "dependency_component_id"].nunique()
                ),
            }
        )
    return support


def _receipt_bindings(paths: dict[str, Path]) -> dict[str, str]:
    return {
        "adapter_receipt_sha256": sha256_file(paths["adapter_receipt"]),
        "checkpoint_verification_receipt_sha256": sha256_file(
            paths["checkpoint_verification"]
        ),
        "extraction_authorization_receipt_sha256": sha256_file(
            paths["extraction_authorization"]
        ),
        "preparation_receipt_sha256": sha256_file(paths["preparation_receipt"]),
        "subset_receipt_sha256": sha256_file(paths["subset_receipt"]),
    }


def prepare(root: Path) -> dict[str, Any]:
    """Create folds and result contract without opening features or outcomes."""
    paths = _paths(root)
    for output in ("fold_map", "fold_receipt", "result_contract"):
        if paths[output].exists():
            raise FileExistsError(f"prepare refuses to overwrite {paths[output]}")

    config = _load_config(paths["config"])
    protocol_sha256 = protocol_config_sha256(config)
    manifest_sha256 = sha256_file(paths["manifest"])
    suite = _load_json(paths["suite_authority"])
    preparation = _load_json(paths["preparation_receipt"])
    receipts = {
        representation: _load_json(paths[f"{representation.split('_')[0]}_receipt"])
        for representation in REPRESENTATIONS
    }
    if suite.get("analysis") != "G_domain" or suite.get("data_role") != "final":
        raise ContractError("feature-suite authority is not final G_domain")
    if suite.get("analysis_manifest_sha256") != manifest_sha256:
        raise ContractError("suite authority has the wrong analysis manifest")
    if preparation.get("analysis_manifest_sha256") != manifest_sha256:
        raise ContractError("preparation receipt has the wrong analysis manifest")
    for representation, receipt in receipts.items():
        if (
            receipt.get("representation_id") != representation
            or receipt.get("analysis_manifest_sha256") != manifest_sha256
            or receipt.get("status") != "complete"
        ):
            raise ContractError(f"{representation} receipt is not complete and bound")

    manifest = pd.read_parquet(paths["manifest"])
    componentized = _componentized_manifest(manifest, config)
    folds = _expected_fold_map(componentized, config)
    atomic_write_parquet(folds, paths["fold_map"])
    fold_receipt = {
        "schema_version": 1,
        "status": "final_G_domain_dependency_folds_sealed",
        "analysis": "G_domain",
        "data_role": "final",
        "analysis_manifest_sha256": manifest_sha256,
        "analysis_manifest_content_sha256": canonical_frame_sha256(
            manifest,
            columns=manifest.columns,
            sort_by=["row_id"],
        ),
        "protocol_config_sha256": protocol_sha256,
        "authority_digest": suite["authority_digest"],
        "fold_map_sha256": sha256_file(paths["fold_map"]),
        "fold_map_content_sha256": canonical_frame_sha256(
            folds,
            columns=folds.columns,
            sort_by=["dependency_component_id"],
        ),
        "fold_seed": int(config["probe"]["fold_seed"]),
        "folds": int(config["probe"]["folds"]),
        "dependency_columns": list(config["probe"]["dependency_columns"]),
        "rows": len(componentized),
        "dependency_components": int(
            componentized["dependency_component_id"].nunique()
        ),
        "observed_rows": int(componentized["fold_label"].sum()),
        "simulated_rows": int(componentized["fold_label"].eq(0).sum()),
        "fold_support": _fold_support(componentized, folds),
        "allocation_used_feature_bytes": False,
        "allocation_used_scientific_outcomes": False,
        "allocation_class_field": "score_blind population provenance",
    }
    atomic_write_json(fold_receipt, paths["fold_receipt"])

    suite_sha256 = sha256_file(paths["suite_authority"])
    contract = {
        "schema_version": 1,
        "status": "sealed_final_G_domain_result_contract",
        "analysis": "G_domain",
        "data_role": "final",
        "allowed_stage": "probe_analysis",
        "protocol_config_sha256": protocol_sha256,
        "authority_digest": suite["authority_digest"],
        "implementation": {
            "standalone_runner_path": str(paths["runner"].relative_to(root)),
            "standalone_runner_sha256": sha256_file(paths["runner"]),
            "independent_verifier_path": str(
                paths["independent_verifier"].relative_to(root)
            ),
            "independent_verifier_sha256": sha256_file(
                paths["independent_verifier"]
            ),
            "package_source_tree_sha256": _source_tree_sha256(),
            "dependency_lock_sha256": sha256_file(paths["lock"]),
            "registered_function": "stronglens_probes.probes.crossfit_domain",
        },
        "input_bindings": {
            "analysis_manifest_sha256": manifest_sha256,
            "analysis_manifest_content_sha256": fold_receipt[
                "analysis_manifest_content_sha256"
            ],
            "feature_suite_authority_sha256": suite_sha256,
            "subset_manifest_sha256": sha256_file(paths["subset"]),
            **_receipt_bindings(paths),
            "reference_manifest_sha256": config["authorities"][
                "reference_manifest_sha256"
            ],
            "reference_payload_manifest_sha256": config["authorities"][
                "reference_payload_manifest_sha256"
            ],
            "experiment_host_universe_sha256": config["authorities"][
                "experiment_host_universe_sha256"
            ],
            "reference_payload_adapter_manifest_sha256": config["authorities"][
                "reference_payload_adapter_manifest_sha256"
            ],
            "experiment_host_adapter_manifest_sha256": config["authorities"][
                "experiment_host_adapter_manifest_sha256"
            ],
            "experiment_host_tractor_manifest_sha256": config["authorities"][
                "experiment_host_tractor_manifest_sha256"
            ],
            "G_rec_scene_manifest_sha256": config["authorities"][
                "grec_scene_manifest_sha256"
            ],
            "representations": suite["representations"],
        },
        "fold_binding": {
            "fold_map_sha256": fold_receipt["fold_map_sha256"],
            "fold_map_content_sha256": fold_receipt[
                "fold_map_content_sha256"
            ],
            "fold_receipt_sha256": sha256_file(paths["fold_receipt"]),
            "folds": int(config["probe"]["folds"]),
            "fold_seed": int(config["probe"]["fold_seed"]),
        },
        "estimator": {
            "classifier": config["probe"]["classifier"],
            "C": float(config["probe"]["C"]),
            "solver": config["probe"]["solver"],
            "max_iter": int(config["probe"]["max_iter"]),
            "estimator_seed": int(config["probe"]["estimator_seed"]),
            "standardisation": config["probe"]["standardisation"],
            "weighting": config["probe"]["weighting"],
        },
        "representations": {
            "primary": "stein_ssl_resnet50",
            "mandatory_diagnostics": [
                "imagenet_effnet_v2_s",
                "pixel_statistics",
            ],
        },
        "metrics": {
            "primary": "dependency_component_equal_oof_roc_auc",
            "diagnostic": "row_weighted_oof_roc_auc",
            "fold_diagnostic": "dependency_component_equal_oof_roc_auc",
            "scope": "finite_cohort_model_conditional_point_estimates",
            "significance_test": None,
            "population_confidence_interval": None,
        },
        "outputs": {
            "prediction_files": {
                representation: (
                    f"gdomain_oof_{representation}.parquet"
                )
                for representation in REPRESENTATIONS
            },
            "prediction_columns": list(PREDICTION_COLUMNS),
            "result_receipt": paths["result_receipt"].name,
            "independent_verification": paths["independent_verification"].name,
        },
        "bootstrap_permitted": False,
        "G_match_permitted": False,
        "contract_creation_used_feature_bytes": False,
        "contract_creation_used_scientific_outcomes": False,
    }
    atomic_write_json(contract, paths["result_contract"])
    return {
        "status": "prepared",
        "fold_map_sha256": sha256_file(paths["fold_map"]),
        "fold_receipt_sha256": sha256_file(paths["fold_receipt"]),
        "result_contract_sha256": sha256_file(paths["result_contract"]),
        "rows": len(componentized),
        "dependency_components": int(
            componentized["dependency_component_id"].nunique()
        ),
        "fold_support": fold_receipt["fold_support"],
    }


def authorize(root: Path) -> dict[str, Any]:
    """Write a final analysis authorization; config pinning is a later step."""
    paths = _paths(root)
    if paths["analysis_authorization"].exists():
        raise FileExistsError(
            "authorize refuses to overwrite final analysis authorization"
        )
    config = _load_config(paths["config"])
    contract = _load_json(paths["result_contract"])
    suite = _load_json(paths["suite_authority"])
    contract_sha256 = sha256_file(paths["result_contract"])
    suite_sha256 = sha256_file(paths["suite_authority"])
    manifest_sha256 = sha256_file(paths["manifest"])
    if contract["input_bindings"]["feature_suite_authority_sha256"] != suite_sha256:
        raise ContractError("result contract has the wrong feature suite")
    if contract["input_bindings"]["analysis_manifest_sha256"] != manifest_sha256:
        raise ContractError("result contract has the wrong analysis manifest")
    if contract["implementation"]["standalone_runner_sha256"] != sha256_file(
        paths["runner"]
    ):
        raise ContractError("result contract has the wrong runner")

    authorization = {
        "schema_version": 1,
        "status": "authorized_final_analysis",
        "authorization_kind": "analysis",
        "data_role": "final",
        "authorization_basis": (
            "User explicitly authorized implementation and execution of the final "
            "G_domain linear-probe stage on 2026-09-16."
        ),
        "allowed_stages": ["probe_analysis"],
        "manifest_sha256s": [manifest_sha256],
        "protocol_config_sha256": protocol_config_sha256(config),
        "reference_manifest_sha256": config["authorities"][
            "reference_manifest_sha256"
        ],
        "reference_payload_manifest_sha256": config["authorities"][
            "reference_payload_manifest_sha256"
        ],
        "host_universe_manifest_sha256": config["authorities"][
            "experiment_host_universe_sha256"
        ],
        "reference_payload_adapter_manifest_sha256": config["authorities"][
            "reference_payload_adapter_manifest_sha256"
        ],
        "host_adapter_manifest_sha256": config["authorities"][
            "experiment_host_adapter_manifest_sha256"
        ],
        "host_tractor_adapter_manifest_sha256": config["authorities"][
            "experiment_host_tractor_manifest_sha256"
        ],
        "G_rec_scene_manifest_sha256": config["authorities"][
            "grec_scene_manifest_sha256"
        ],
        "feature_suite_authority_sha256s": [suite_sha256],
        "result_contract_sha256s": [contract_sha256],
        "extraction_authorization_receipt_sha256": sha256_file(
            paths["extraction_authorization"]
        ),
        "checkpoint_verification_receipt_sha256": sha256_file(
            paths["checkpoint_verification"]
        ),
        "feature_bindings": suite["representations"],
        "parent_bindings": suite["parent_bindings"],
        "bootstrap_authorized": False,
        "G_match_authorized": False,
        "scientific_outcomes_accessed_before_authorization": False,
        "feature_bytes_opened_before_authorization": False,
    }
    atomic_write_json(authorization, paths["analysis_authorization"])
    return {
        "status": "authorization_written_config_patch_required",
        "analysis_authorization_receipt_sha256": sha256_file(
            paths["analysis_authorization"]
        ),
        "final_feature_suite_authority_sha256": suite_sha256,
        "result_contract_sha256": contract_sha256,
    }


def _validate_contract(
    root: Path,
    paths: dict[str, Path],
    config: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    contract = _load_json(paths["result_contract"])
    contract_sha256 = sha256_file(paths["result_contract"])
    if (
        contract.get("status") != "sealed_final_G_domain_result_contract"
        or contract.get("analysis") != "G_domain"
        or contract.get("allowed_stage") != "probe_analysis"
        or contract.get("bootstrap_permitted") is not False
        or contract.get("G_match_permitted") is not False
        or contract.get("protocol_config_sha256")
        != protocol_config_sha256(config)
    ):
        raise ContractError("final G_domain result contract is invalid")
    implementation = contract["implementation"]
    checks = {
        "standalone_runner_sha256": sha256_file(paths["runner"]),
        "independent_verifier_sha256": sha256_file(
            paths["independent_verifier"]
        ),
        "package_source_tree_sha256": _source_tree_sha256(),
        "dependency_lock_sha256": sha256_file(paths["lock"]),
    }
    for field, observed in checks.items():
        if implementation.get(field) != observed:
            raise ContractError(f"result contract has wrong {field}")
    if implementation["package_source_tree_sha256"] != config["execution"][
        "source_tree_sha256"
    ]:
        raise ContractError("registered package source tree has drifted")
    if implementation["dependency_lock_sha256"] != config["execution"][
        "dependency_lock_sha256"
    ]:
        raise ContractError("registered dependency lock has drifted")
    return contract, contract_sha256


def _load_authorities(
    paths: dict[str, Path],
    config: dict[str, Any],
):
    authority = load_authority_context(
        config=config,
        references_path=paths["references"],
        reference_payloads_path=paths["reference_payload_source"],
        host_universe_path=paths["host_universe_source"],
        reference_payload_adapter_path=paths["reference_adapter"],
        host_adapter_path=paths["host_adapter"],
    )
    grec = load_grec_artifacts(
        scene_path=paths["parent_adapter"],
        host_path=paths["host_adapter"],
        host_tractor_path=paths["tractor_adapter"],
        config=config,
        authority=authority,
    )
    subset = load_gdomain_subset(
        path=paths["subset"],
        receipt_path=paths["subset_receipt"],
        grec=grec,
        config=config,
        authority=authority,
    )
    recomputed = recompute_domain_manifest(
        grec=grec,
        subset=subset,
        config=config,
        authority=authority,
    )
    return authority, grec, subset, recomputed


def _checkpoint_paths(paths: dict[str, Path]) -> dict[str, Path | None]:
    verification = _load_json(paths["checkpoint_verification"])
    if verification.get("status") != (
        "encoder_checkpoint_bytes_verified_and_strict_loaded"
    ):
        raise ContractError("encoder checkpoint verification is incomplete")
    checkpoints: dict[str, Path | None] = {"pixel_statistics": None}
    for representation in REPRESENTATIONS[:2]:
        entry = verification["checkpoints"][representation]
        checkpoint = Path(entry["path"]).resolve()
        if (
            sha256_file(checkpoint) != entry["sha256"]
            or entry.get("configured_sha256_match") is not True
            or not str(entry.get("strict_architecture_load", "")).startswith(
                "passed"
            )
        ):
            raise ContractError(f"{representation} checkpoint verification failed")
        checkpoints[representation] = checkpoint
    return checkpoints


def _feature_paths(
    paths: dict[str, Path],
    representation: str,
) -> tuple[Path, Path]:
    prefix = representation.split("_")[0]
    return paths[f"{prefix}_features"], paths[f"{prefix}_receipt"]


def _validate_fold_artifacts(
    paths: dict[str, Path],
    config: dict[str, Any],
    contract: dict[str, Any],
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_receipt = _load_json(paths["fold_receipt"])
    folds = pd.read_parquet(paths["fold_map"])
    componentized = _componentized_manifest(manifest, config)
    expected = _expected_fold_map(componentized, config)
    try:
        pd.testing.assert_frame_equal(folds, expected, check_exact=True)
    except AssertionError as error:
        raise ContractError("fold map differs from deterministic allocation") from error
    observed = {
        "fold_map_sha256": sha256_file(paths["fold_map"]),
        "fold_map_content_sha256": canonical_frame_sha256(
            folds,
            columns=folds.columns,
            sort_by=["dependency_component_id"],
        ),
        "fold_receipt_sha256": sha256_file(paths["fold_receipt"]),
    }
    for field, value in observed.items():
        if contract["fold_binding"].get(field) != value:
            raise ContractError(f"result contract has wrong {field}")
    if (
        fold_receipt.get("fold_map_sha256") != observed["fold_map_sha256"]
        or fold_receipt.get("fold_map_content_sha256")
        != observed["fold_map_content_sha256"]
        or fold_receipt.get("fold_support")
        != _fold_support(componentized, folds)
    ):
        raise ContractError("fold receipt is inconsistent")
    return componentized.drop(columns="fold_label"), folds


def _runtime_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _result_support(predictions: pd.DataFrame) -> dict[str, Any]:
    by_fold = []
    for fold, frame in predictions.groupby("fold", sort=True):
        observed = frame["label"].eq(1)
        by_fold.append(
            {
                "fold": int(fold),
                "rows": len(frame),
                "observed_rows": int(observed.sum()),
                "simulated_rows": int((~observed).sum()),
                "dependency_components": int(
                    frame["dependency_component_id"].nunique()
                ),
                "observed_dependency_components": int(
                    frame.loc[observed, "dependency_component_id"].nunique()
                ),
                "simulated_dependency_components": int(
                    frame.loc[~observed, "dependency_component_id"].nunique()
                ),
            }
        )
    observed = predictions["label"].eq(1)
    return {
        "rows": len(predictions),
        "observed_rows": int(observed.sum()),
        "simulated_rows": int((~observed).sum()),
        "dependency_components": int(
            predictions["dependency_component_id"].nunique()
        ),
        "observed_dependency_components": int(
            predictions.loc[observed, "dependency_component_id"].nunique()
        ),
        "simulated_dependency_components": int(
            predictions.loc[~observed, "dependency_component_id"].nunique()
        ),
        "folds": by_fold,
    }


def _temporary_component_manifest(
    componentized: pd.DataFrame,
    directory: Path,
) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=".gdomain-componentized.",
        suffix=".parquet",
        dir=directory,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    componentized.to_parquet(temporary, index=False)
    return temporary


def run(root: Path) -> dict[str, Any]:
    """Revalidate all parents, execute all three probes, and commit results."""
    paths = _paths(root)
    config = _load_config(paths["config"])
    contract, contract_sha256 = _validate_contract(root, paths, config)
    suite_sha256 = sha256_file(paths["suite_authority"])
    authorization_sha256 = sha256_file(paths["analysis_authorization"])
    execution = config["execution"]
    if (
        execution.get("final_analysis_authorized") is not True
        or execution.get("analysis_authorization_receipt_sha256")
        != authorization_sha256
        or execution["final_feature_suite_authority_sha256"].get("G_domain")
        != suite_sha256
    ):
        raise ContractError("final G_domain analysis is not config-pinned")

    prediction_paths = {
        representation: paths["result_receipt"].parent
        / contract["outputs"]["prediction_files"][representation]
        for representation in REPRESENTATIONS
    }
    for output in [*prediction_paths.values(), paths["result_receipt"]]:
        if output.exists():
            raise FileExistsError(f"run refuses to overwrite {output}")

    authority, grec, subset, recomputed = _load_authorities(paths, config)
    if authority.authority_digest != contract["authority_digest"]:
        raise ContractError("result contract has the wrong authority digest")
    manifest = pd.read_parquet(paths["manifest"])
    try:
        pd.testing.assert_frame_equal(manifest, recomputed, check_exact=True)
    except AssertionError as error:
        raise ContractError("analysis manifest differs from recomputation") from error

    extraction_authorization = load_execution_authorization(
        path=paths["extraction_authorization"],
        config=config,
        authority=authority,
    )
    checkpoints = _checkpoint_paths(paths)
    artifacts = {}
    for representation in REPRESENTATIONS:
        features_path, receipt_path = _feature_paths(paths, representation)
        artifacts[representation] = load_feature_artifact(
            features_path=features_path,
            receipt_path=receipt_path,
            manifest_path=paths["manifest"],
            config=config,
            authority=authority,
            authorization=extraction_authorization,
            checkpoint_path=checkpoints[representation],
            expected_data_role="final",
        )

    analysis_authorization = load_execution_authorization(
        path=paths["analysis_authorization"],
        config=config,
        authority=authority,
    )
    if (
        list(analysis_authorization.allowed_stages) != ["probe_analysis"]
        or analysis_authorization.receipt.get("result_contract_sha256s")
        != [contract_sha256]
        or analysis_authorization.receipt.get(
            "feature_suite_authority_sha256s"
        )
        != [suite_sha256]
    ):
        raise ContractError("analysis authorization is not exact and stage-limited")
    suite = load_feature_suite_authority(
        path=paths["suite_authority"],
        artifacts=artifacts,
        analysis_manifest_path=paths["manifest"],
        recomputed_manifest=recomputed,
        parent_bindings=domain_parent_bindings(grec, subset),
        analysis_authorization=analysis_authorization,
        parent_artifacts={"grec": grec, "subset": subset},
        config=config,
        authority=authority,
    )
    componentized, folds = _validate_fold_artifacts(
        paths,
        config,
        contract,
        manifest,
    )
    observed = componentized["population"].eq("observed_reference")
    crossfit_authority = replace(
        authority,
        reference_payload_items=tuple(
            sorted(
                zip(
                    componentized.loc[observed, "reference_id"].astype(str),
                    componentized.loc[observed, "pixel_sha256"].astype(str),
                    strict=True,
                )
            )
        ),
    )

    temporary_manifest = _temporary_component_manifest(
        componentized,
        paths["result_receipt"].parent,
    )
    results: dict[str, dict[str, Any]] = {}
    predictions: dict[str, pd.DataFrame] = {}
    try:
        # crossfit_domain currently reads its manifest path rather than the already
        # authenticated in-memory manifest.  The temporary file is the exact,
        # row-order-preserving authenticated manifest plus its deterministic
        # dependency-component column.  The external suite is fully revalidated
        # immediately above before this private prevalidated entry point is used.
        # The package context stores payload-file hashes, whereas crossfit_domain
        # validates the manifest's array-byte ``pixel_sha256`` column against that
        # map.  ``crossfit_authority`` is an analysis-local view of the exact
        # content hashes from the revalidated manifest; its authority digest and
        # all physical authority paths remain unchanged.
        for representation in REPRESENTATIONS:
            suite.revalidate(
                config=config,
                authority=authority,
                stage="probe_analysis",
            )
            artifact = replace(
                artifacts[representation],
                manifest_path=temporary_manifest,
            )
            result, frame = crossfit_domain(
                artifact,
                folds,
                config=config,
                authority=crossfit_authority,
                _prevalidated=True,
            )
            role = (
                "primary"
                if representation == "stein_ssl_resnet50"
                else "diagnostic"
            )
            frame["representation"] = representation
            frame["representation_role"] = role
            frame = frame[list(PREDICTION_COLUMNS)]
            results[representation] = result
            predictions[representation] = frame
    finally:
        temporary_manifest.unlink(missing_ok=True)

    temporary_outputs: dict[str, Path] = {}
    try:
        output_bindings: dict[str, dict[str, Any]] = {}
        for representation, frame in predictions.items():
            destination = prediction_paths[representation]
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination.name}.",
                suffix=".parquet",
                dir=destination.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            frame.to_parquet(temporary, index=False)
            temporary_outputs[representation] = temporary
            output_bindings[representation] = {
                "path": destination.name,
                "sha256": sha256_file(temporary),
                "content_sha256": canonical_frame_sha256(
                    frame,
                    columns=frame.columns,
                    sort_by=["row_id"],
                ),
                "rows": len(frame),
            }

        receipt = {
            "schema_version": 1,
            "status": "complete_final_G_domain_linear_probe",
            "analysis": "G_domain",
            "data_role": "final",
            "protocol_config_sha256": protocol_config_sha256(config),
            "authority_digest": authority.authority_digest,
            "result_contract_sha256": contract_sha256,
            "analysis_authorization_receipt_sha256": authorization_sha256,
            "extraction_authorization_receipt_sha256": (
                extraction_authorization.receipt_sha256
            ),
            "feature_suite_authority_sha256": suite_sha256,
            "analysis_manifest_sha256": sha256_file(paths["manifest"]),
            "subset_manifest_sha256": subset.manifest_sha256,
            "fold_map_sha256": sha256_file(paths["fold_map"]),
            "fold_receipt_sha256": sha256_file(paths["fold_receipt"]),
            "runtime_environment": _runtime_environment(),
            "source_tree_sha256": _source_tree_sha256(),
            "runner_sha256": sha256_file(paths["runner"]),
            "outputs": output_bindings,
            "results": results,
            "support": _result_support(
                predictions["stein_ssl_resnet50"]
            ),
            "uncertainty_scope": (
                "finite_cohort_model_conditional_point_estimates"
            ),
            "significance_test": None,
            "population_confidence_interval": None,
            "bootstrap_run": False,
            "G_match_run": False,
            "mandatory_diagnostics_complete": True,
            "package_crossfit_domain_executed": True,
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{paths['result_receipt'].name}.",
            suffix=".json",
            dir=paths["result_receipt"].parent,
            delete=False,
        ) as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            temporary_receipt = Path(handle.name)
        temporary_outputs["result_receipt"] = temporary_receipt

        moved: list[Path] = []
        try:
            for representation in REPRESENTATIONS:
                destination = prediction_paths[representation]
                os.replace(temporary_outputs[representation], destination)
                moved.append(destination)
            os.replace(temporary_receipt, paths["result_receipt"])
            moved.append(paths["result_receipt"])
        except Exception:
            for path in moved:
                path.unlink(missing_ok=True)
            raise
    finally:
        for temporary in temporary_outputs.values():
            temporary.unlink(missing_ok=True)

    return {
        "status": "complete",
        "result_receipt_sha256": sha256_file(paths["result_receipt"]),
        "AUCs": {
            representation: {
                "dependency_component_equal": result["primary_oof_auc"],
                "row_weighted_diagnostic": (
                    result["row_weighted_oof_auc_diagnostic"]
                ),
            }
            for representation, result in results.items()
        },
        "support": _result_support(predictions["stein_ssl_resnet50"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "authorize", "run"))
    parser.add_argument("--repository-root", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.repository_root.resolve()
    handlers = {
        "prepare": prepare,
        "authorize": authorize,
        "run": run,
    }
    result = handlers[arguments.command](root)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
