from __future__ import annotations

import importlib
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .contracts import (
    PilotContractError,
    assert_config_integrity,
    authority_path,
    load_json,
    sha256_file,
)


def validate_inherited_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate predecessor lineage without claiming successor-era creation."""
    assert_config_integrity(config)
    configured = config["authorities"]
    predecessor_path = authority_path(config, "predecessor_config")
    with predecessor_path.open("rb") as handle:
        predecessor = tomllib.load(handle)
    if predecessor.get("study", {}).get("status") != configured[
        "predecessor_config"
    ]["status"]:
        raise PilotContractError("predecessor config lifecycle status changed")
    predecessor_sha = configured["predecessor_config"]["sha256"]

    receipts = {
        name: load_json(authority_path(config, name))
        for name in (
            "grec_host_receipt",
            "final_grec_selection_receipt",
            "host_flux_calibration",
            "grec_scene_receipt",
            "grec_parent_receipt",
            "grec_resume_audit",
            "renderer_validation_receipt",
            "quadrature_amendment",
            "threshold_receipt",
            "governance_amendment",
            "negative_role_receipt",
            "historical_staging_receipt",
        )
    }
    for name, receipt in receipts.items():
        expected_status = configured[name].get("status")
        if expected_status is not None and receipt.get("status") != expected_status:
            raise PilotContractError(
                f"{name} status {receipt.get('status')!r} != {expected_status!r}"
            )

    artifact_bindings = {
        "grec_host_receipt": (
            "host_manifest_sha256",
            configured["grec_host_manifest"]["sha256"],
        ),
        "final_grec_selection_receipt": (
            "selection_manifest_sha256",
            configured["final_grec_selection"]["sha256"],
        ),
        "grec_scene_receipt": (
            "scene_table_sha256",
            configured["grec_scene_manifest"]["sha256"],
        ),
        "grec_parent_receipt": (
            "parent_manifest_sha256",
            configured["grec_parent_manifest"]["sha256"],
        ),
    }
    for receipt_name, (field, expected) in artifact_bindings.items():
        if receipts[receipt_name].get(field) != expected:
            raise PilotContractError(f"{receipt_name} does not bind {field}")
    if receipts["final_grec_selection_receipt"].get("evidence_hashes", {}).get(
        "test_manifest"
    ) != configured["test_manifest"]["sha256"]:
        raise PilotContractError("final G_rec selection does not bind test identities")

    parent = receipts["grec_parent_receipt"]
    if (
        parent.get("scene_table_sha256")
        != configured["grec_scene_manifest"]["sha256"]
        or parent.get("scene_table_receipt_sha256")
        != configured["grec_scene_receipt"]["sha256"]
        or parent.get("renderer_validation_receipt_sha256")
        != configured["renderer_validation_receipt"]["sha256"]
    ):
        raise PilotContractError("G_rec parent authority graph is incomplete")
    for name in ("grec_host_receipt", "grec_scene_receipt", "grec_parent_receipt"):
        provenance = receipts[name].get("config", {})
        if provenance.get("config_file_sha256") != predecessor_sha:
            raise PilotContractError(f"{name} does not identify predecessor config")

    renderer = receipts["renderer_validation_receipt"]
    if (
        renderer.get("final_outcomes_used_false") is not True
        or renderer.get("scorer_closure_sha256")
        != config["scoring"]["authoritative_scorer_closure_sha256"]
    ):
        raise PilotContractError("renderer validation/scorer closure is not authoritative")
    amendment = receipts["quadrature_amendment"]
    if (
        amendment.get("production_config_sha256") != predecessor_sha
        or amendment.get("scene_table_sha256")
        != configured["grec_scene_manifest"]["sha256"]
        or amendment.get("renderer_validation_receipt_sha256")
        != configured["renderer_validation_receipt"]["sha256"]
        or amendment.get("resumable_runner_sha256")
        != config["inherited_code"][
            "experiments_v2/grec_pnoise/scripts/render_grec_resumable.py"
        ]
        or amendment.get("outcome_access", {}).get("cnn_scores_accessed") is not False
        or amendment.get("supplemental_orders") != [48, 64, 96]
    ):
        raise PilotContractError("score-blind quadrature amendment is invalid")

    threshold = receipts["threshold_receipt"]
    if (
        threshold.get("checkpoint_sha256") != config["scoring"]["checkpoint_sha256"]
        or threshold.get("logit_threshold_float32_bits")
        != config["scoring"]["threshold_float32_bits"]
        or threshold.get("scorer_closure_sha256")
        != config["scoring"]["authoritative_scorer_closure_sha256"]
        or threshold.get("backend_profile", {}).get("profile_sha256")
        != config["scoring"]["authoritative_backend_profile_sha256"]
        or threshold.get("comparison") != config["scoring"]["comparison"]
    ):
        raise PilotContractError("threshold/scorer/backend authority mismatch")

    governance = receipts["governance_amendment"]
    bindings = governance.get("artifact_bindings", {})
    if (
        bindings.get("production_config", {}).get("sha256") != predecessor_sha
        or bindings.get("common_threshold_receipt", {}).get("sha256")
        != configured["threshold_receipt"]["sha256"]
        or governance.get("execution_config_lifecycle_correction", {}).get(
            "retroactive_unqualified_governance_pass_claimed"
        )
        is not False
    ):
        raise PilotContractError("governance amendment does not close predecessor lineage")

    negative = receipts["negative_role_receipt"]
    for role_name, authority_name in (
        ("threshold_calibration", "threshold_calibration_role"),
        ("achieved_fpr", "achieved_fpr_role"),
    ):
        if negative.get("roles", {}).get(role_name, {}).get("sha256") != configured[
            authority_name
        ]["sha256"]:
            raise PilotContractError(f"negative-role receipt does not bind {role_name}")

    staging = receipts["historical_staging_receipt"]
    historical_bindings = {
        "cutouts_tar": "historical_cutout_tar",
        "cutout_metadata": "historical_metadata",
        "scoring_manifest": "historical_scoring_manifest",
    }
    for input_name, authority_name in historical_bindings.items():
        if staging.get("inputs", {}).get(input_name, {}).get("sha256") != configured[
            authority_name
        ]["sha256"]:
            raise PilotContractError(f"historical staging does not bind {input_name}")
    if staging.get("config", {}).get("source_sha256") != configured[
        "historical_pnoise_config"
    ]["sha256"]:
        raise PilotContractError("historical staging config binding changed")

    return {
        "status": "successor_inherited_authority_verified",
        "predecessor_config_sha256": predecessor_sha,
        "predecessor_lifecycle_status": predecessor["study"]["status"],
        "successor_does_not_relabel_predecessor_creation": True,
        "authority_sha256": {
            name: record["sha256"] for name, record in sorted(configured.items())
        },
        "threshold_float32_bits": threshold["logit_threshold_float32_bits"],
        "scorer_closure_sha256": threshold["scorer_closure_sha256"],
        "backend_profile_sha256": threshold["backend_profile"]["profile_sha256"],
    }


def import_predecessor_modules(config: Mapping[str, Any]) -> dict[str, Any]:
    """Import only the hash-bound predecessor renderer/scorer implementation."""
    source_root = Path(config["paths"]["predecessor_package_src"]).resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    modules = {
        name: importlib.import_module(f"grec_pnoise.{name}")
        for name in ("contracts", "pnoise", "rendering", "scene_table", "scoring")
    }
    repository_root = Path(config["paths"]["repository_root"]).resolve()
    for name, module in modules.items():
        path = Path(module.__file__).resolve()
        expected_path = (
            repository_root
            / f"experiments_v2/grec_pnoise/src/grec_pnoise/{name}.py"
        )
        if path != expected_path:
            raise PilotContractError(
                f"predecessor module {name} imported from {path}, not {expected_path}"
            )
        expected = config["inherited_code"][
            expected_path.relative_to(repository_root).as_posix()
        ]
        if sha256_file(path) != expected:
            raise PilotContractError(f"predecessor module {name} changed")
    return modules


def load_predecessor_config(
    config: Mapping[str, Any], modules: Mapping[str, Any]
) -> dict[str, Any]:
    predecessor = modules["contracts"].load_config(
        authority_path(config, "predecessor_config")
    )
    if predecessor["__provenance__"]["config_file_sha256"] != config["authorities"][
        "predecessor_config"
    ]["sha256"]:
        raise PilotContractError("loaded predecessor config digest mismatch")
    return predecessor
