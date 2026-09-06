from __future__ import annotations

import copy
import shutil
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pnoise_pilot_v2.authority import (
    import_predecessor_modules,
    load_predecessor_config,
)
from pnoise_pilot_v2.compatibility import (
    production_draw_receipt,
    validate_with_actual_production_validator,
)
from pnoise_pilot_v2.contracts import (
    PilotContractError,
    _validate_config_seal,
    array_bytes_sha256,
    assert_config_integrity,
    atomic_write_json,
    ensure_no_final_outcome_path,
    expected_junit_properties,
    load_config,
    load_expected_nodeids,
    parse_passing_junit,
    sha256_file,
    successor_code_closure,
    validate_test_receipt,
)
from pnoise_pilot_v2.deep_validation import (
    _reopen_comparison_pixel,
    _reopen_design_host,
)
from pnoise_pilot_v2.prepare import (
    DEPTH_NODES,
    PSF_NODES,
    _overlap_record,
    deterministic_balanced_selection,
    nearest_lower_midpoint,
    prepare_design,
    validate_overlap_evidence,
)
from pnoise_pilot_v2.selector import (
    _expected_photon_key_bytes,
    evaluate_draw_panel,
    select_draw_count,
    validate_frozen_pilot_scores,
    validate_pilot_score_payload,
)
from pnoise_pilot_v2.streaming import (
    run_streaming_pilot,
    validate_design_freeze,
    write_streamed_pilot_scores,
)

PACKAGE = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE / "config/production.toml"


@pytest.fixture(scope="session")
def sealed_config() -> dict:
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="session", autouse=True)
def bind_junit_to_code_and_config(
    record_testsuite_property,
    sealed_config: dict,
) -> None:
    for name, value in expected_junit_properties(sealed_config).items():
        record_testsuite_property(name, value)


def _valid_overlap(config: dict) -> dict:
    gates = [
        {
            "role": role,
            "id_overlap": 0,
            "payload_file_sha256_overlap": 0,
            "raw_array_sha256_overlap": 0,
            "physical_group_overlap": 0,
            "canonical_spatial_component_overlap": 0,
            "healpix_128_overlap": 0,
            "within_38_arcsec_hosts": 0,
            "minimum_angular_separation_arcsec": 39.0,
        }
        for role in config["overlap"]["comparison_roles"]
    ]
    return {
        "status": "zero_current_role_overlap_verified",
        "gates": gates,
        "final_id_overlap": 0,
        "final_pixel_array_sha256_overlap": 0,
        "final_physical_group_overlap": 0,
        "final_canonical_spatial_component_overlap": 0,
        "final_healpix_128_overlap": 0,
        "final_within_38_arcsec_hosts": 0,
        "threshold_calibration_pixel_array_sha256_overlap": 0,
        "achieved_fpr_pixel_array_sha256_overlap": 0,
    }


def _write_junit(
    output_path: Path,
    config: dict,
    *,
    suite_name: str | None = None,
    tests: int | None = None,
    failures: int = 0,
    errors: int = 0,
    skipped: int = 0,
    actual_testcases: int | None = None,
    testcase_nodeids: list[str] | None = None,
    property_overrides: dict[str, str] | None = None,
) -> None:
    declared_tests = (
        tests
        if tests is not None
        else config["validation"]["expected_test_cardinality"]
    )
    suite = ET.Element(
        "testsuite",
        {
            "name": suite_name or config["validation"]["test_suite_id"],
            "tests": str(declared_tests),
            "failures": str(failures),
            "errors": str(errors),
            "skipped": str(skipped),
        },
    )
    properties = ET.SubElement(suite, "properties")
    property_values = expected_junit_properties(config)
    property_values.update(property_overrides or {})
    for property_name, value in property_values.items():
        ET.SubElement(
            properties,
            "property",
            {"name": property_name, "value": value},
        )
    if testcase_nodeids is None:
        expected_nodeids = load_expected_nodeids(config)
        count = declared_tests if actual_testcases is None else actual_testcases
        testcase_nodeids = expected_nodeids[:count]
    for nodeid in testcase_nodeids:
        testcase_path, testcase_name = nodeid.split("::", 1)
        ET.SubElement(
            suite,
            "testcase",
            {
                "classname": testcase_path.removesuffix(".py").replace("/", "."),
                "name": testcase_name,
            },
        )
    ET.ElementTree(suite).write(
        output_path,
        encoding="utf-8",
        xml_declaration=True,
    )


def test_successor_config_freezes_counts_and_excludes_one_draw(
    sealed_config: dict,
) -> None:
    assert sealed_config["study"]["status"] == "adopted_execution_authority_pre_pilot"
    assert (
        sealed_config["study"]["predecessor_artifacts_created_under_successor"]
        is False
    )
    assert sealed_config["design"]["expected_hosts"] == 288
    assert sealed_config["design"]["expected_scenes"] == 3_168
    assert sealed_config["pilot"]["candidate_draw_counts"] == [2, 4, 8, 16, 32, 64]
    assert sealed_config["pilot"]["diagnostic_nonselectable_draw_counts"] == [1]
    assert sealed_config["pilot"]["pixel_persistence"] == "none_stream_batches_only"
    assert_config_integrity(sealed_config)


def test_config_seal_rejects_changed_scientific_bytes(tmp_path: Path) -> None:
    config_copy = tmp_path / "production.toml"
    seal_copy = tmp_path / "CONFIG_SEAL.json"
    shutil.copy2(CONFIG_PATH, config_copy)
    shutil.copy2(CONFIG_PATH.with_name("CONFIG_SEAL.json"), seal_copy)
    with config_copy.open("rb") as handle:
        parsed = tomllib.load(handle)
    _validate_config_seal(config_copy, parsed)
    changed = config_copy.read_text().replace(
        "candidate_draw_counts = [2, 4, 8, 16, 32, 64]",
        "candidate_draw_counts = [2, 4]",
    )
    config_copy.write_text(changed)
    with config_copy.open("rb") as handle:
        changed_parsed = tomllib.load(handle)
    with pytest.raises(PilotContractError, match="seal mismatch"):
        _validate_config_seal(config_copy, changed_parsed)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("pilot", "candidate_draw_counts", [2, 4]),
        ("inference", "mcse_absolute_max", 0.5),
        ("inference", "mcse_fraction_of_sampling_se_max", 0.9),
        ("pnoise", "gain_electron_per_nanomaggy_r", 151.0),
    ],
)
def test_in_memory_scientific_mutation_fails_integrity(
    sealed_config: dict,
    section: str,
    field: str,
    value,
) -> None:
    mutated = copy.deepcopy(sealed_config)
    mutated[section][field] = value
    with pytest.raises(PilotContractError, match="in-memory"):
        assert_config_integrity(mutated)


@pytest.mark.parametrize("entry", ["prepare", "freeze", "streaming", "selector"])
def test_every_execution_entry_rejects_mutated_config(
    sealed_config: dict,
    tmp_path: Path,
    entry: str,
) -> None:
    mutated = copy.deepcopy(sealed_config)
    mutated["pilot"]["candidate_draw_counts"] = [2, 4]
    calls = {
        "prepare": lambda: prepare_design(
            mutated,
            junit_path=tmp_path / "pytest.xml",
            aws_profile="",
            workers=1,
        ),
        "freeze": lambda: validate_design_freeze(mutated),
        "streaming": lambda: run_streaming_pilot(mutated),
        "selector": lambda: select_draw_count(mutated),
    }
    with pytest.raises(PilotContractError, match="in-memory"):
        calls[entry]()


def test_midpoint_ties_go_to_lower_node() -> None:
    psf_midpoints = (PSF_NODES[:-1] + PSF_NODES[1:]) / 2
    depth_midpoints = (DEPTH_NODES[:-1] + DEPTH_NODES[1:]) / 2
    np.testing.assert_array_equal(
        nearest_lower_midpoint(psf_midpoints, PSF_NODES), PSF_NODES[:-1]
    )
    np.testing.assert_array_equal(
        nearest_lower_midpoint(depth_midpoints, DEPTH_NODES), DEPTH_NODES[:-1]
    )


def test_balanced_selection_is_deterministic_and_uses_distinct_units() -> None:
    rows = [
        {
            "galaxy_id": f"{cell}-{index}",
            "pilot_cell": cell,
            "physical_group_id": f"physical-{cell}-{index}",
            "spatial_component_id": f"spatial-{cell}-{index}",
            "cutout_path": f"/stage/stronglens_calibration/{cell}-{index}.npz",
        }
        for cell in ("1.05|24.5", "1.20|24.0")
        for index in range(8)
    ]
    frame = pd.DataFrame(rows)
    kwargs = {
        "cells": ["1.05|24.5", "1.20|24.0"],
        "hosts_per_cell": 4,
        "namespace": "test",
    }
    first = deterministic_balanced_selection(frame, **kwargs)
    second = deterministic_balanced_selection(
        frame.sample(frac=1, random_state=2),
        **kwargs,
    )
    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("pilot_cell").size().eq(4).all()
    assert not first["physical_group_id"].duplicated().any()
    assert not first["spatial_component_id"].duplicated().any()


def test_balanced_selection_fails_instead_of_reusing_spatial_units() -> None:
    frame = pd.DataFrame(
        {
            "galaxy_id": ["a", "b"],
            "pilot_cell": ["1.05|24.5", "1.05|24.5"],
            "physical_group_id": ["p1", "p2"],
            "spatial_component_id": ["shared", "shared"],
            "cutout_path": ["/stage/a", "/stage/b"],
        }
    )
    with pytest.raises(PilotContractError, match="insufficient distinct"):
        deterministic_balanced_selection(
            frame,
            cells=["1.05|24.5"],
            hosts_per_cell=2,
            namespace="test",
        )


def test_raw_array_overlap_is_computed_independently_of_npz_hash() -> None:
    hosts = pd.DataFrame(
        {
            "host_galaxy_id": ["design"],
            "physical_group_id": ["design-physical"],
            "historical_spatial_component_id": ["design-spatial"],
            "historical_healpix_128": [1],
            "host_payload_sha256": ["different-container"],
            "host_pixel_sha256": ["same-array"],
            "host_ra_deg": [0.0],
            "host_dec_deg": [0.0],
        }
    )
    other = pd.DataFrame(
        {
            "galaxy_id": ["other"],
            "physical_group_id": ["other-physical"],
            "spatial_component_id": ["other-spatial"],
            "healpix_128": [2],
            "payload_sha256": ["other-container"],
            "raw_array_sha256": ["same-array"],
            "ra": [1.0],
            "dec": [0.0],
        }
    )
    record = _overlap_record(
        hosts,
        other,
        role="threshold_calibration",
        id_column="galaxy_id",
        pixel_column="payload_sha256",
        raw_array_column="raw_array_sha256",
        ra_column="ra",
        dec_column="dec",
        minimum_separation_arcsec=38.0,
    )
    assert record["payload_file_sha256_overlap"] == 0
    assert record["raw_array_sha256_overlap"] == 1


@pytest.mark.parametrize(
    "field",
    [
        "id_overlap",
        "payload_file_sha256_overlap",
        "raw_array_sha256_overlap",
        "physical_group_overlap",
        "canonical_spatial_component_overlap",
        "healpix_128_overlap",
        "within_38_arcsec_hosts",
    ],
)
def test_freeze_rejects_every_nonzero_numeric_overlap(
    sealed_config: dict,
    field: str,
) -> None:
    evidence = _valid_overlap(sealed_config)
    evidence["gates"][0][field] = 1
    with pytest.raises(PilotContractError, match="numeric overlap gate"):
        validate_overlap_evidence(evidence, sealed_config)


@pytest.mark.parametrize("role_index", [1, 2])
def test_negative_role_raw_array_overlap_cannot_hide_behind_status(
    sealed_config: dict,
    role_index: int,
) -> None:
    evidence = _valid_overlap(sealed_config)
    evidence["gates"][role_index]["raw_array_sha256_overlap"] = 1
    with pytest.raises(PilotContractError, match="numeric overlap gate"):
        validate_overlap_evidence(evidence, sealed_config)


def test_freeze_rejects_angular_separation_at_bound(sealed_config: dict) -> None:
    evidence = _valid_overlap(sealed_config)
    evidence["gates"][0]["minimum_angular_separation_arcsec"] = 38.0
    with pytest.raises(PilotContractError, match="angular overlap gate"):
        validate_overlap_evidence(evidence, sealed_config)


def test_valid_numeric_overlap_evidence_passes(sealed_config: dict) -> None:
    validate_overlap_evidence(_valid_overlap(sealed_config), sealed_config)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"suite_name": "unrelated"}, "suite identity"),
        ({"tests": 1}, "cardinality"),
        ({"actual_testcases": 0}, "cardinality"),
        ({"skipped": 1}, "cardinality"),
        (
            {"property_overrides": {"test_source_sha256": "0" * 64}},
            "code/config bindings",
        ),
    ],
)
def test_junit_identity_cardinality_skips_and_bindings_fail_closed(
    sealed_config: dict,
    tmp_path: Path,
    mutation: dict,
    match: str,
) -> None:
    report = tmp_path / "pytest.xml"
    _write_junit(report, sealed_config, **mutation)
    with pytest.raises(PilotContractError, match=match):
        parse_passing_junit(report, sealed_config)


def test_copied_properties_with_unrelated_testcases_fail(
    sealed_config: dict,
    tmp_path: Path,
) -> None:
    report = tmp_path / "pytest.xml"
    unrelated = [
        f"tests/test_pipeline.py::test_unrelated_{index}"
        for index in range(sealed_config["validation"]["expected_test_cardinality"])
    ]
    _write_junit(report, sealed_config, testcase_nodeids=unrelated)
    with pytest.raises(PilotContractError, match="node-ID set"):
        parse_passing_junit(report, sealed_config)


def test_reordered_expected_testcases_fail(
    sealed_config: dict,
    tmp_path: Path,
) -> None:
    report = tmp_path / "pytest.xml"
    nodeids = load_expected_nodeids(sealed_config)
    nodeids[0], nodeids[1] = nodeids[1], nodeids[0]
    _write_junit(report, sealed_config, testcase_nodeids=nodeids)
    with pytest.raises(PilotContractError, match="node-ID set"):
        parse_passing_junit(report, sealed_config)


def test_test_receipt_requires_exact_bound_junit(
    sealed_config: dict,
    tmp_path: Path,
) -> None:
    report = tmp_path / "pytest.xml"
    _write_junit(report, sealed_config)
    parsed = parse_passing_junit(report, sealed_config)
    receipt = {
        **parsed,
        "junit_path": str(report),
        "successor_code_closure": successor_code_closure(sealed_config),
        "config": sealed_config["__provenance__"],
    }
    receipt_path = atomic_write_json(tmp_path / "DESIGN_TEST_RECEIPT.json", receipt)
    validate_test_receipt(
        sealed_config,
        tmp_path,
        expected_receipt_sha256=sha256_file(receipt_path),
    )
    report.write_text(report.read_text() + " ")
    with pytest.raises(PilotContractError, match="stale"):
        validate_test_receipt(
            sealed_config,
            tmp_path,
            expected_receipt_sha256=sha256_file(receipt_path),
        )


def test_test_receipt_is_mandatory(sealed_config: dict, tmp_path: Path) -> None:
    with pytest.raises(PilotContractError, match="required"):
        validate_test_receipt(sealed_config, tmp_path)


def test_actual_production_draw_validator_accepts_successor_handoff(
    sealed_config: dict,
    tmp_path: Path,
) -> None:
    modules = import_predecessor_modules(sealed_config)
    predecessor = load_predecessor_config(sealed_config, modules)
    receipt = production_draw_receipt(
        sealed_config,
        predecessor,
        modules,
        {
            "selected_draw_count": 64,
            "absolute_mcse_gate_passed": True,
            "relative_mcse_gate_passed": True,
        },
    )
    path = atomic_write_json(tmp_path / "draw_receipt.json", receipt)
    validated = validate_with_actual_production_validator(
        path,
        sealed_config,
        predecessor,
        modules,
    )
    assert validated["metric"] == "host_equal_paired_detection_risk_difference"
    assert validated["config"] == modules["contracts"].config_provenance(predecessor)
    assert validated["successor_config"] == sealed_config["__provenance__"]


def _bounded_serialized_score_fixture(
    tmp_path: Path,
    config: dict,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    original = pd.DataFrame(
        {
            "scene_id": [
                "scene_0123456789abcdef0123456789abcdef01234567",
                "scene_89abcdef0123456789abcdef0123456789abcdef",
                "scene_1111111111111111111111111111111111111111",
                "scene_2222222222222222222222222222222222222222",
            ],
            "host_galaxy_id": [
                "0016p142_1511",
                "0016p142_1511",
                "longer_representative_host_identifier",
                "longer_representative_host_identifier",
            ],
            "psf_bin": [1.05, 1.05, 1.20, 1.20],
            "depth_bin": [24.5, 24.5, 24.0, 24.0],
            "radius_index": [0, 1, 0, 1],
            "theta_e_arcsec": [0.5, 0.75, 0.5, 0.75],
        }
    )
    parquet_path = tmp_path / "representative_scenes.parquet"
    original.to_parquet(parquet_path, index=False)
    scenes = pd.read_parquet(parquet_path)
    assert pd.__version__ == "3.0.5"
    assert scenes["scene_id"].to_numpy().dtype == np.dtype(object)
    ledger = scenes[["scene_id", "host_galaxy_id"]].copy()
    ledger["global_role"] = config["design"]["global_role"]
    draw_count = 2
    expectation_logits = np.zeros(len(scenes), dtype=np.float32)
    physical_logits = np.zeros((len(scenes), draw_count), dtype=np.float32)
    scene_ids = scenes["scene_id"].astype(str).tolist()
    score_path = tmp_path / "bounded_pilot_scores.npz"
    write_streamed_pilot_scores(
        score_path,
        scenes,
        draw_count=draw_count,
        expectation_logits=expectation_logits,
        physical_logits=physical_logits,
        expectation_decisions=np.zeros(len(scenes), dtype=np.bool_),
        physical_decisions=np.zeros((len(scenes), draw_count), dtype=np.bool_),
        photon_draw_key_bytes=_expected_photon_key_bytes(
            scene_ids,
            draw_count,
            config["pnoise"]["photon_stream_namespace"],
        ),
        threshold_float32_bits=config["scoring"]["threshold_float32_bits"],
    )
    with np.load(score_path, allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }
    return arrays, scenes, ledger


def test_parquet_ids_roundtrip_actual_producer_npz_without_pickle(
    sealed_config: dict,
    tmp_path: Path,
) -> None:
    arrays, scenes, ledger = _bounded_serialized_score_fixture(
        tmp_path,
        sealed_config,
    )
    validated = validate_pilot_score_payload(
        arrays,
        scenes,
        ledger,
        scene_count=4,
        draw_count=2,
        threshold_float32_bits=sealed_config["scoring"][
            "threshold_float32_bits"
        ],
        photon_stream_namespace=sealed_config["pnoise"][
            "photon_stream_namespace"
        ],
        global_role=sealed_config["design"]["global_role"],
    )
    assert validated["scene_ids"].dtype.kind == "U"
    assert validated["host_ids"].dtype.kind == "U"
    assert validated["scene_ids"].tolist() == scenes["scene_id"].tolist()
    assert validated["host_ids"].tolist() == scenes["host_galaxy_id"].tolist()


def test_object_dtype_identity_archive_is_rejected(
    sealed_config: dict,
    tmp_path: Path,
) -> None:
    arrays, scenes, ledger = _bounded_serialized_score_fixture(
        tmp_path,
        sealed_config,
    )
    arrays["scene_ids"] = arrays["scene_ids"].astype(object)
    with pytest.raises(PilotContractError, match="fixed-width Unicode"):
        validate_pilot_score_payload(
            arrays,
            scenes,
            ledger,
            scene_count=4,
            draw_count=2,
            threshold_float32_bits=sealed_config["scoring"][
                "threshold_float32_bits"
            ],
            photon_stream_namespace=sealed_config["pnoise"][
                "photon_stream_namespace"
            ],
            global_role=sealed_config["design"]["global_role"],
        )


def test_truncated_fixed_width_identity_archive_is_rejected(
    sealed_config: dict,
    tmp_path: Path,
) -> None:
    arrays, scenes, ledger = _bounded_serialized_score_fixture(
        tmp_path,
        sealed_config,
    )
    arrays["host_ids"] = arrays["host_ids"].astype("<U8")
    with pytest.raises(PilotContractError, match="fixed-width Unicode"):
        validate_pilot_score_payload(
            arrays,
            scenes,
            ledger,
            scene_count=4,
            draw_count=2,
            threshold_float32_bits=sealed_config["scoring"][
                "threshold_float32_bits"
            ],
            photon_stream_namespace=sealed_config["pnoise"][
                "photon_stream_namespace"
            ],
            global_role=sealed_config["design"]["global_role"],
        )


def _valid_frozen_score_inputs(
    config: dict,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    radii = np.asarray(config["design"]["radii_arcsec"], dtype=np.float32)
    scene_ids: list[str] = []
    host_ids: list[str] = []
    cell_ids: list[str] = []
    radius_indices: list[int] = []
    theta: list[float] = []
    for cell_index, cell in enumerate(config["design"]["supported_cells"]):
        for host_index in range(config["design"]["hosts_per_cell"]):
            host_id = f"host_{cell_index:02d}_{host_index:02d}"
            for radius_index, radius in enumerate(radii):
                scene_ids.append(f"scene_{len(scene_ids):04d}")
                host_ids.append(host_id)
                cell_ids.append(cell)
                radius_indices.append(radius_index)
                theta.append(float(radius))
    scenes = pd.DataFrame(
        {
            "scene_id": scene_ids,
            "host_galaxy_id": host_ids,
            "psf_bin": [float(value.split("|")[0]) for value in cell_ids],
            "depth_bin": [float(value.split("|")[1]) for value in cell_ids],
            "radius_index": radius_indices,
            "theta_e_arcsec": theta,
        }
    )
    ledger = scenes[["scene_id", "host_galaxy_id"]].copy()
    ledger["global_role"] = config["design"]["global_role"]
    draw_count = config["pilot"]["maximum_draw_count"]
    arrays = {
        "scene_ids": np.asarray(scene_ids),
        "host_ids": np.asarray(host_ids),
        "cell_ids": np.asarray(cell_ids),
        "radius_index": np.asarray(radius_indices, dtype=np.int16),
        "theta_e_arcsec": np.asarray(theta, dtype=np.float32),
        "draw_indices": np.arange(draw_count, dtype=np.int16),
        "expectation_logits": np.zeros(len(scenes), dtype=np.float32),
        "physical_logits": np.zeros((len(scenes), draw_count), dtype=np.float32),
        "expectation_decisions": np.zeros(len(scenes), dtype=np.bool_),
        "physical_decisions": np.zeros(
            (len(scenes), draw_count),
            dtype=np.bool_,
        ),
        "photon_draw_key_bytes": _expected_photon_key_bytes(
            scene_ids,
            draw_count,
            config["pnoise"]["photon_stream_namespace"],
        ),
        "threshold_float32_bits": np.asarray(
            [config["scoring"]["threshold_float32_bits"]],
            dtype=np.uint32,
        ),
    }
    return arrays, scenes, ledger


def test_forged_small_pilot_archive_is_rejected(sealed_config: dict) -> None:
    arrays, scenes, ledger = _valid_frozen_score_inputs(sealed_config)
    forged_arrays = {
        name: value[:8] if value.ndim else value for name, value in arrays.items()
    }
    forged_arrays["draw_indices"] = arrays["draw_indices"]
    forged_arrays["threshold_float32_bits"] = arrays["threshold_float32_bits"]
    with pytest.raises(PilotContractError, match="cardinality"):
        validate_frozen_pilot_scores(
            forged_arrays,
            scenes.iloc[:8],
            ledger.iloc[:8],
            sealed_config,
        )


def test_bit_flipped_decision_is_recomputed_and_rejected(
    sealed_config: dict,
) -> None:
    arrays, scenes, ledger = _valid_frozen_score_inputs(sealed_config)
    arrays["physical_decisions"][0, 0] = True
    with pytest.raises(PilotContractError, match="recompute"):
        validate_frozen_pilot_scores(arrays, scenes, ledger, sealed_config)


def test_reordered_scene_identity_is_rejected(sealed_config: dict) -> None:
    arrays, scenes, ledger = _valid_frozen_score_inputs(sealed_config)
    arrays["scene_ids"][[0, 1]] = arrays["scene_ids"][[1, 0]]
    with pytest.raises(PilotContractError, match="ordered"):
        validate_frozen_pilot_scores(arrays, scenes, ledger, sealed_config)


def test_postfreeze_design_host_payload_substitution_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "host.npz"
    original = np.zeros((101, 101, 3), dtype=np.float32)
    np.savez(path, cutout=original)
    row = {
        "host_pixel_path": str(path),
        "host_payload_sha256": sha256_file(path),
        "host_pixel_sha256": array_bytes_sha256(original),
        "pilot_cell": "1.05|24.5",
    }
    _reopen_design_host(row)
    substituted = original.copy()
    substituted[0, 0, 0] = 1
    np.savez(path, cutout=substituted)
    with pytest.raises(PilotContractError, match="payload changed"):
        _reopen_design_host(row)


def test_postfreeze_comparison_payload_substitution_is_rejected(
    tmp_path: Path,
) -> None:
    payload_root = tmp_path / "payloads"
    payload_root.mkdir()
    path = payload_root / "row.npz"
    original = np.zeros((101, 101, 3), dtype=np.float32)
    np.savez(path, cutout=original)
    row = {
        "source_payload_path": str(path),
        "source_payload_sha256": sha256_file(path),
        "raw_array_key": "cutout",
        "raw_array_sha256": array_bytes_sha256(original),
        "global_role": "threshold_calibration",
    }
    _reopen_comparison_pixel(row, payload_root=payload_root)
    substituted = original.copy()
    substituted[0, 0, 0] = 1
    np.savez(path, cutout=substituted)
    with pytest.raises(PilotContractError, match="payload changed"):
        _reopen_comparison_pixel(row, payload_root=payload_root)


def test_one_draw_is_diagnostic_and_never_selectable() -> None:
    hosts = np.repeat(["a", "b", "c", "d"], 2)
    cells = np.repeat(["cell1", "cell1", "cell2", "cell2"], 2)
    physical = np.zeros((8, 64), dtype=np.uint8)
    physical[2:4] = 1
    physical[4:6, ::2] = 1
    physical[6:8, :16] = 1
    result = evaluate_draw_panel(
        physical,
        np.zeros(8, dtype=np.uint8),
        hosts,
        cells,
        [2, 4, 8, 16, 32, 64],
        absolute_mcse_max=1.0,
        fraction_of_sampling_se_max=100.0,
    )
    diagnostic = result["candidate_results"][0]
    assert diagnostic["draw_count"] == 1
    assert diagnostic["selectable"] is False
    assert diagnostic["estimated_mcse"] is None
    assert result["selected_draw_count"] == 2


def test_selector_rejects_one_as_selectable_candidate() -> None:
    with pytest.raises(PilotContractError, match="within"):
        evaluate_draw_panel(
            np.zeros((8, 64), dtype=np.uint8),
            np.zeros(8, dtype=np.uint8),
            np.repeat(["a", "b", "c", "d"], 2),
            np.repeat(["cell1", "cell1", "cell2", "cell2"], 2),
            [1, 2],
            absolute_mcse_max=1.0,
            fraction_of_sampling_se_max=1.0,
        )


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/repo/results_v2/grec/scores/grec_scores.parquet",
        "/tmp/repo/results_v2/grec/GREC_ANALYSIS.json",
        "/tmp/repo/results_v2/grec/GREC_INDEPENDENT_VERIFICATION.json",
        "/tmp/repo/results_v2/grec/score.log",
        "/tmp/repo/results_v2/pnoise/final/pixels.npz",
    ],
)
def test_final_outcome_paths_are_rejected(path: str) -> None:
    with pytest.raises(PilotContractError, match="forbidden"):
        ensure_no_final_outcome_path(path)
