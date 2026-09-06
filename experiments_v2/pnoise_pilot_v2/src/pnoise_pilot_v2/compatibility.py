from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .authority import import_predecessor_modules, load_predecessor_config
from .contracts import (
    PilotContractError,
    assert_config_integrity,
    atomic_write_json,
    config_provenance,
    sha256_json,
)


def production_draw_receipt(
    config: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    modules: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    assert_config_integrity(config)
    threshold_sha256 = config["authorities"]["threshold_receipt"]["sha256"]
    selected = int(selection["selected_draw_count"])
    if selected not in {int(value) for value in config["pilot"]["candidate_draw_counts"]}:
        raise PilotContractError("selected draw count is outside successor candidates")
    return {
        "schema_version": 1,
        "status": predecessor["pnoise"]["required_draw_receipt_status"],
        "design_only": True,
        "final_outcomes_used": False,
        "metric": config["handoff"]["production_expected_metric"],
        "selected_draw_count": selected,
        "absolute_mcse_gate_passed": bool(
            selection["absolute_mcse_gate_passed"]
        ),
        "relative_mcse_gate_passed": bool(
            selection["relative_mcse_gate_passed"]
        ),
        "threshold_receipt_sha256": threshold_sha256,
        "pnoise_config_sha256": modules["contracts"].scientific_section_sha256(
            predecessor,
            "pnoise",
            "render",
            "host",
        ),
        "photon_stream_namespace": predecessor["pnoise"][
            "photon_stream_namespace"
        ],
        # This field intentionally remains predecessor provenance because the
        # existing production validator requires the rendering configuration
        # under which its parent/scientific contract was created.
        "config": modules["contracts"].config_provenance(predecessor),
        "successor_config": config_provenance(config),
        "successor_selection_metric": config["inference"]["metric"],
        "predecessor_bindings": {
            name: config["authorities"][name]["sha256"]
            for name in (
                "predecessor_config",
                "grec_parent_manifest",
                "grec_parent_receipt",
                "grec_scene_manifest",
                "grec_scene_receipt",
                "renderer_validation_receipt",
                "quadrature_amendment",
                "threshold_receipt",
                "governance_amendment",
            )
        },
        "scorer_bindings": {
            "checkpoint_sha256": config["scoring"]["checkpoint_sha256"],
            "scorer_closure_sha256": config["scoring"][
                "authoritative_scorer_closure_sha256"
            ],
            "backend_profile_sha256": config["scoring"][
                "authoritative_backend_profile_sha256"
            ],
        },
        **{
            key: value
            for key, value in selection.items()
            if key
            not in {
                "selected_draw_count",
                "absolute_mcse_gate_passed",
                "relative_mcse_gate_passed",
            }
        },
    }


def validate_with_actual_production_validator(
    receipt_path: str | Path,
    config: Mapping[str, Any],
    predecessor: Mapping[str, Any] | None = None,
    modules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    assert_config_integrity(config)
    loaded_modules = (
        dict(modules) if modules is not None else import_predecessor_modules(config)
    )
    loaded_predecessor = (
        dict(predecessor)
        if predecessor is not None
        else load_predecessor_config(config, loaded_modules)
    )
    return loaded_modules["pnoise"].validate_draw_count_receipt(
        receipt_path,
        loaded_predecessor,
        config["authorities"]["threshold_receipt"]["path"],
    )


def build_prelaunch_handoff_compatibility(
    config: Mapping[str, Any],
    output: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert_config_integrity(config)
    output_path = Path(output)
    modules = import_predecessor_modules(config)
    predecessor = load_predecessor_config(config, modules)
    probe_selection = {
        "selected_draw_count": max(
            int(value) for value in config["pilot"]["candidate_draw_counts"]
        ),
        "absolute_mcse_gate_passed": True,
        "relative_mcse_gate_passed": True,
        "compatibility_probe_only": True,
    }
    probe = production_draw_receipt(
        config,
        predecessor,
        modules,
        probe_selection,
    )
    probe_path = output_path / ".production_draw_handoff_probe.json"
    atomic_write_json(probe_path, probe)
    try:
        validated = validate_with_actual_production_validator(
            probe_path,
            config,
            predecessor,
            modules,
        )
    finally:
        probe_path.unlink(missing_ok=True)
    if validated.get("selected_draw_count") != probe["selected_draw_count"]:
        raise PilotContractError("actual production validator changed probe count")
    compatibility = {
        "schema_version": 1,
        "status": "pnoise_production_handoff_compatibility_verified",
        "actual_production_validator_passed": True,
        "validator": (
            "grec_pnoise.pnoise.validate_draw_count_receipt"
        ),
        "validator_source_sha256": config["inherited_code"][
            "experiments_v2/grec_pnoise/src/grec_pnoise/pnoise.py"
        ],
        "probe_receipt_canonical_sha256": sha256_json(probe),
        "production_expected_metric": config["handoff"][
            "production_expected_metric"
        ],
        "production_expected_status": predecessor["pnoise"][
            "required_draw_receipt_status"
        ],
        "pnoise_config_sha256": probe["pnoise_config_sha256"],
        "photon_stream_namespace": probe["photon_stream_namespace"],
        "predecessor_config": probe["config"],
        "predecessor_bindings": probe["predecessor_bindings"],
        "scorer_bindings": probe["scorer_bindings"],
        "threshold_receipt_sha256": probe["threshold_receipt_sha256"],
        "successor_config": config_provenance(config),
    }
    return compatibility, probe


def validate_handoff_compatibility_receipt(
    config: Mapping[str, Any],
    path: str | Path,
) -> dict[str, Any]:
    assert_config_integrity(config)
    from .contracts import load_json

    receipt = load_json(path)
    expected = {
        "status": "pnoise_production_handoff_compatibility_verified",
        "actual_production_validator_passed": True,
        "validator": "grec_pnoise.pnoise.validate_draw_count_receipt",
        "validator_source_sha256": config["inherited_code"][
            "experiments_v2/grec_pnoise/src/grec_pnoise/pnoise.py"
        ],
        "production_expected_metric": config["handoff"][
            "production_expected_metric"
        ],
        "threshold_receipt_sha256": config["authorities"]["threshold_receipt"][
            "sha256"
        ],
        "successor_config": config_provenance(config),
    }
    failures = {
        key: (receipt.get(key), value)
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if failures:
        raise PilotContractError(f"production handoff receipt is stale: {failures}")
    if len(str(receipt.get("pnoise_config_sha256", ""))) != 64:
        raise PilotContractError("production handoff lacks pnoise config digest")
    return receipt
