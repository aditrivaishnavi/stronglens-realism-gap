from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .contracts import (
    ContractError,
    assert_config_integrity,
    atomic_write_json,
    config_provenance,
    load_json,
    sha256_file,
    source_closure,
    validate_scorer_receipt,
    verify_parent_grec_authority,
    verify_source_closure,
)
from .design import validate_source_receipt
from .draw_count import validate_draw_count_receipt
from .threshold import validate_threshold_receipt


def _binding(
    path: str | Path,
    *,
    base: Path,
    receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "relative_path": os.path.relpath(Path(path).resolve(), base),
        "sha256": sha256_file(path),
    }
    if receipt is not None:
        result["status"] = receipt.get("status")
    return result


def resolve_binding(
    execution_lock_path: str | Path,
    binding: Mapping[str, Any],
) -> Path:
    relative = Path(str(binding.get("relative_path", "")))
    if relative.is_absolute():
        raise ContractError("execution lock binding must be portable and relative")
    return (Path(execution_lock_path).resolve().parent / relative).resolve()


def build_execution_lock(
    *,
    expected_config_sha256: str,
    config: dict[str, Any],
    final_host_manifest_path: str | Path,
    final_host_receipt_path: str | Path,
    scene_table_path: str | Path,
    scene_receipt_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    source_manifest_path: str | Path,
    source_receipt_path: str | Path,
    design_path: str | Path,
    design_receipt_path: str | Path,
    scorer_receipt_path: str | Path,
    threshold_receipt_path: str | Path,
    draw_count_receipt_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze every upstream authority before final P_src rendering."""
    assert_config_integrity(config)
    provenance = config_provenance(config)
    if config["status"] != "frozen_pre_scoring":
        raise ContractError("execution lock requires status=frozen_pre_scoring")
    if expected_config_sha256 != provenance["config_file_sha256"]:
        raise ContractError("explicit expected config SHA-256 does not match")
    _, parent_authority = verify_parent_grec_authority(
        final_host_manifest_path=final_host_manifest_path,
        final_host_receipt_path=final_host_receipt_path,
        scene_table_path=scene_table_path,
        scene_receipt_path=scene_receipt_path,
        parent_manifest_path=parent_manifest_path,
        parent_receipt_path=parent_receipt_path,
        config=config,
    )
    _, source_receipt = validate_source_receipt(
        source_manifest_path,
        source_receipt_path,
        config,
        require_final=True,
        expected_role="experiment",
    )
    design_receipt = load_json(design_receipt_path)
    if (
        design_receipt.get("status")
        != config["execution"]["required_design_receipt_status"]
        or design_receipt.get("pair_design_sha256") != sha256_file(design_path)
        or design_receipt.get("config") != provenance
        or design_receipt.get("parent_authority") != parent_authority
        or design_receipt.get("source_manifest_sha256")
        != sha256_file(source_manifest_path)
        or design_receipt.get("source_receipt_sha256")
        != sha256_file(source_receipt_path)
    ):
        raise ContractError("parent-child design receipt is not final or bound")
    verify_source_closure(design_receipt.get("source_closure", {}), config)
    scorer_receipt = validate_scorer_receipt(
        scorer_receipt_path, config, require_final=True
    )
    threshold_receipt, _ = validate_threshold_receipt(
        threshold_receipt_path, config, require_final=True
    )
    validate_draw_count_receipt(
        draw_count_receipt_path,
        config,
        expected_final_design_path=design_path,
        expected_final_design_receipt_path=design_receipt_path,
        expected_scorer_receipt_path=scorer_receipt_path,
        expected_threshold_receipt_path=threshold_receipt_path,
        production=True,
    )
    closure = source_closure(config)
    binding_base = Path(output_path).resolve().parent
    lock = {
        "schema_version": 2,
        "status": config["execution"]["required_lock_status"],
        "config": provenance,
        "expected_config_sha256": expected_config_sha256,
        "source_closure": closure,
        "parent_authority": parent_authority,
        "bindings": {
            "final_host_manifest": _binding(
                final_host_manifest_path, base=binding_base
            ),
            "final_host_receipt": _binding(
                final_host_receipt_path,
                base=binding_base,
                receipt=load_json(final_host_receipt_path),
            ),
            "grec_scene_table": _binding(scene_table_path, base=binding_base),
            "grec_scene_receipt": _binding(
                scene_receipt_path,
                base=binding_base,
                receipt=load_json(scene_receipt_path),
            ),
            "grec_parent_manifest": _binding(parent_manifest_path, base=binding_base),
            "grec_parent_receipt": _binding(
                parent_receipt_path,
                base=binding_base,
                receipt=load_json(parent_receipt_path),
            ),
            "source_manifest": _binding(source_manifest_path, base=binding_base),
            "source_receipt": _binding(
                source_receipt_path, base=binding_base, receipt=source_receipt
            ),
            "pair_design": _binding(design_path, base=binding_base),
            "pair_design_receipt": _binding(
                design_receipt_path, base=binding_base, receipt=design_receipt
            ),
            "scorer_receipt": _binding(
                scorer_receipt_path, base=binding_base, receipt=scorer_receipt
            ),
            "threshold_receipt": _binding(
                threshold_receipt_path,
                base=binding_base,
                receipt=threshold_receipt,
            ),
            "draw_count_receipt": _binding(
                draw_count_receipt_path,
                base=binding_base,
                receipt=load_json(draw_count_receipt_path),
            ),
        },
        "final_outcomes_used": False,
    }
    atomic_write_json(output_path, lock)
    return lock


def validate_execution_lock(
    path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    assert_config_integrity(config)
    lock = load_json(path)
    if (
        lock.get("status") != config["execution"]["required_lock_status"]
        or lock.get("config") != config_provenance(config)
        or lock.get("expected_config_sha256")
        != config_provenance(config)["config_file_sha256"]
    ):
        raise ContractError("execution lock status or exact config hash drift")
    verify_source_closure(lock.get("source_closure", {}), config)
    for name, binding in lock.get("bindings", {}).items():
        artifact = resolve_binding(path, binding)
        if not artifact.is_file() or sha256_file(artifact) != binding.get("sha256"):
            raise ContractError(f"execution binding drift: {name}")
        if (
            "status" in binding
            and load_json(artifact).get("status") != binding["status"]
        ):
            raise ContractError(f"execution receipt status drift: {name}")
    bindings = lock["bindings"]
    _, parent_authority = verify_parent_grec_authority(
        final_host_manifest_path=resolve_binding(path, bindings["final_host_manifest"]),
        final_host_receipt_path=resolve_binding(path, bindings["final_host_receipt"]),
        scene_table_path=resolve_binding(path, bindings["grec_scene_table"]),
        scene_receipt_path=resolve_binding(path, bindings["grec_scene_receipt"]),
        parent_manifest_path=resolve_binding(path, bindings["grec_parent_manifest"]),
        parent_receipt_path=resolve_binding(path, bindings["grec_parent_receipt"]),
        config=config,
    )
    if parent_authority != lock.get("parent_authority"):
        raise ContractError("G_rec parent authority drift after execution freeze")
    validate_source_receipt(
        resolve_binding(path, bindings["source_manifest"]),
        resolve_binding(path, bindings["source_receipt"]),
        config,
        require_final=True,
        expected_role="experiment",
    )
    validate_scorer_receipt(
        resolve_binding(path, bindings["scorer_receipt"]),
        config,
        require_final=True,
    )
    validate_threshold_receipt(
        resolve_binding(path, bindings["threshold_receipt"]),
        config,
        require_final=True,
    )
    validate_draw_count_receipt(
        resolve_binding(path, bindings["draw_count_receipt"]),
        config,
        expected_final_design_path=resolve_binding(path, bindings["pair_design"]),
        expected_final_design_receipt_path=resolve_binding(
            path, bindings["pair_design_receipt"]
        ),
        expected_scorer_receipt_path=resolve_binding(path, bindings["scorer_receipt"]),
        expected_threshold_receipt_path=resolve_binding(
            path, bindings["threshold_receipt"]
        ),
        production=True,
    )
    design_receipt = load_json(resolve_binding(path, bindings["pair_design_receipt"]))
    if (
        design_receipt.get("pair_design_sha256") != bindings["pair_design"]["sha256"]
        or design_receipt.get("parent_authority") != parent_authority
    ):
        raise ContractError("pair design authority drift")
    return lock
