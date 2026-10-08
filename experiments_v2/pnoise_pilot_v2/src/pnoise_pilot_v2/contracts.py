from __future__ import annotations

import hashlib
import json
import os
import tomllib
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class PilotContractError(RuntimeError):
    """A scientific, provenance, or execution gate failed closed."""


FORBIDDEN_FINAL_OUTCOME_FRAGMENTS = (
    "/results_v2/grec/scores/",
    "/results_v2/grec/grec_analysis.json",
    "/results_v2/grec/grec_independent_verification.json",
    "/results_v2/grec/score.log",
    "/results_v2/pnoise/final",
)
CONFIG_SEAL_NAME = "CONFIG_SEAL.json"
SCIENTIFIC_CONFIG_SECTIONS = (
    "study",
    "design",
    "overlap",
    "historical_source",
    "pilot",
    "pnoise",
    "inference",
    "handoff",
    "scoring",
    "execution",
    "validation",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        default=_json_default,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_bytes_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def fixed_width_unicode(values: Any, *, field: str) -> np.ndarray:
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or raw.size == 0:
        raise PilotContractError(f"{field} must be a nonempty one-dimensional sequence")
    if any(value is None or not isinstance(value, str) for value in raw):
        raise PilotContractError(f"{field} contains a null or non-string identity")
    strings = [str(value) for value in raw]
    width = max(len(value) for value in strings)
    if width < 1:
        raise PilotContractError(f"{field} contains an empty identity")
    result = np.asarray(strings, dtype=f"<U{width}")
    if (
        result.dtype.kind != "U"
        or result.dtype.itemsize // np.dtype("U1").itemsize != width
        or result.tolist() != strings
    ):
        raise PilotContractError(f"{field} fixed-width Unicode conversion truncated data")
    return result


def stable_id(prefix: str, value: Any, *, length: int = 40) -> str:
    return f"{prefix}_{sha256_json(value)[:length]}"


def ensure_no_final_outcome_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    lowered = resolved.as_posix().lower()
    for fragment in FORBIDDEN_FINAL_OUTCOME_FRAGMENTS:
        if fragment in lowered:
            raise PilotContractError(
                f"final G_rec/P_noise outcome access is forbidden: {resolved}"
            )
    return resolved


def load_json(path: str | Path) -> dict[str, Any]:
    safe = ensure_no_final_outcome_path(path)
    value = json.loads(safe.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PilotContractError(f"{safe} must contain one JSON object")
    return value


def normalized_scientific_config(config: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in SCIENTIFIC_CONFIG_SECTIONS if name not in config]
    if missing:
        raise PilotContractError(f"scientific config sections are missing: {missing}")
    return {name: config[name] for name in SCIENTIFIC_CONFIG_SECTIONS}


def load_expected_nodeids(config: Mapping[str, Any]) -> list[str]:
    validation = config["validation"]
    path = ensure_no_final_outcome_path(validation["expected_nodeids_path"])
    if sha256_file(path) != validation["expected_nodeids_file_sha256"]:
        raise PilotContractError("expected pytest node-ID manifest digest mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") != "pnoise_pilot_expected_nodeids_frozen"
        or not isinstance(manifest.get("nodeids"), list)
    ):
        raise PilotContractError("expected pytest node-ID manifest is invalid")
    nodeids = [str(value) for value in manifest["nodeids"]]
    ordered_sha256 = sha256_json(nodeids)
    if (
        len(nodeids) != int(validation["expected_test_cardinality"])
        or len(set(nodeids)) != len(nodeids)
        or ordered_sha256 != validation["expected_ordered_nodeids_sha256"]
        or ordered_sha256 != manifest.get("ordered_nodeids_sha256")
    ):
        raise PilotContractError("expected pytest node-ID set/order is invalid")
    return nodeids


def _validate_config_seal(
    config_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    seal_path = config_path.with_name(CONFIG_SEAL_NAME)
    if not seal_path.is_file():
        raise PilotContractError(f"configuration seal is missing: {seal_path}")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not isinstance(seal, dict):
        raise PilotContractError("configuration seal must be one JSON object")
    expected = {
        "status": "pnoise_pilot_v2_configuration_sealed",
        "config_filename": config_path.name,
        "config_bytes_sha256": sha256_file(config_path),
        "normalized_full_config_sha256": sha256_json(config),
        "normalized_scientific_config_sha256": sha256_json(
            normalized_scientific_config(config)
        ),
        "scientific_sections": list(SCIENTIFIC_CONFIG_SECTIONS),
    }
    failures = {
        key: (seal.get(key), value)
        for key, value in expected.items()
        if seal.get(key) != value
    }
    if failures:
        raise PilotContractError(f"configuration seal mismatch: {failures}")
    return {
        **seal,
        "path": str(seal_path.resolve()),
        "sha256": sha256_file(seal_path),
    }


def load_config(
    path: str | Path,
    *,
    verify_large_files: bool = True,
    verify_external_artifacts: bool = True,
) -> dict[str, Any]:
    config_path = ensure_no_final_outcome_path(path)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("study", {}).get("status") != "adopted_execution_authority_pre_pilot":
        raise PilotContractError("successor configuration is not adopted")
    if config.get("study", {}).get("predecessor_artifacts_created_under_successor") is not False:
        raise PilotContractError("successor must not relabel predecessor artifacts")
    seal = _validate_config_seal(config_path, config)
    load_expected_nodeids(config)
    authorities = config.get("authorities")
    if not isinstance(authorities, dict) or not authorities:
        raise PilotContractError("successor configuration has no inherited authorities")
    if verify_external_artifacts:
        for name, record in authorities.items():
            authority_path = ensure_no_final_outcome_path(record["path"])
            if not authority_path.is_file():
                raise PilotContractError(f"missing authority {name}: {authority_path}")
            if not verify_large_files and name == "historical_cutout_tar":
                continue
            observed = sha256_file(authority_path)
            if observed != record["sha256"]:
                raise PilotContractError(
                    f"inherited authority drift at {name}: "
                    f"{observed} != {record['sha256']}"
                )
    root = Path(config["paths"]["repository_root"]).resolve()
    if verify_external_artifacts:
        for relative, expected in config.get("inherited_code", {}).items():
            source = ensure_no_final_outcome_path(root / relative)
            observed = sha256_file(source)
            if observed != expected:
                raise PilotContractError(
                    f"inherited code drift at {relative}: {observed} != {expected}"
                )
    if verify_external_artifacts:
        checkpoint = ensure_no_final_outcome_path(config["scoring"]["checkpoint"])
        if sha256_file(checkpoint) != config["scoring"]["checkpoint_sha256"]:
            raise PilotContractError("authoritative checkpoint digest mismatch")
    config["__provenance__"] = {
        "path": str(config_path),
        "sha256": sha256_file(config_path),
        "contract_sha256": sha256_json(config),
        "normalized_scientific_config_sha256": seal[
            "normalized_scientific_config_sha256"
        ],
        "config_seal_path": seal["path"],
        "config_seal_sha256": seal["sha256"],
    }
    return config


def assert_config_integrity(config: Mapping[str, Any]) -> None:
    provenance = config.get("__provenance__")
    if not isinstance(provenance, Mapping):
        raise PilotContractError("configuration was not loaded through load_config")
    body = {key: value for key, value in config.items() if key != "__provenance__"}
    if sha256_json(body) != provenance["contract_sha256"]:
        raise PilotContractError("in-memory successor configuration drift")
    if sha256_file(provenance["path"]) != provenance["sha256"]:
        raise PilotContractError("successor configuration file changed")
    seal = _validate_config_seal(Path(str(provenance["path"])), body)
    if (
        seal["sha256"] != provenance["config_seal_sha256"]
        or seal["normalized_scientific_config_sha256"]
        != provenance["normalized_scientific_config_sha256"]
    ):
        raise PilotContractError("configuration seal changed after load")
    load_expected_nodeids(body)


def authority_path(config: Mapping[str, Any], name: str) -> Path:
    try:
        return ensure_no_final_outcome_path(config["authorities"][name]["path"])
    except KeyError as error:
        raise PilotContractError(f"missing configured authority {name}") from error


def config_provenance(config: Mapping[str, Any]) -> dict[str, str]:
    provenance = config.get("__provenance__")
    if not isinstance(provenance, Mapping):
        raise PilotContractError("configuration provenance unavailable")
    return dict(provenance)


def successor_code_closure(config: Mapping[str, Any]) -> dict[str, str]:
    root = Path(config["paths"]["repository_root"]).resolve()
    package = root / "experiments_v2/pnoise_pilot_v2"
    paths = [
        package / "pyproject.toml",
        package / "config/production.toml",
        package / f"config/{CONFIG_SEAL_NAME}",
        package / "tests/EXPECTED_NODEIDS.json",
        *sorted((package / "src/pnoise_pilot_v2").glob("*.py")),
        *sorted((package / "tests").glob("test_*.py")),
    ]
    if not paths or any(not path.is_file() for path in paths):
        raise PilotContractError("successor code closure is incomplete")
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in paths
    }


def inherited_code_closure(config: Mapping[str, Any]) -> dict[str, str]:
    root = Path(config["paths"]["repository_root"]).resolve()
    closure = dict(config["inherited_code"])
    for relative, expected in closure.items():
        if sha256_file(root / relative) != expected:
            raise PilotContractError(f"inherited code closure drift at {relative}")
    return closure


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return destination


def atomic_write_parquet(path: str | Path, frame: pd.DataFrame) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".partial.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(destination)
    return destination


def atomic_write_npz(path: str | Path, **arrays: np.ndarray) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return destination


def require_absent(*paths: str | Path) -> None:
    existing = [str(Path(path)) for path in paths if Path(path).exists()]
    if existing:
        raise PilotContractError(f"refusing to overwrite frozen artifacts: {existing}")


def expected_junit_properties(config: Mapping[str, Any]) -> dict[str, str]:
    assert_config_integrity(config)
    closure = successor_code_closure(config)
    test_path = "experiments_v2/pnoise_pilot_v2/tests/test_pipeline.py"
    return {
        "contract_suite_id": str(config["validation"]["test_suite_id"]),
        "config_file_sha256": str(config["__provenance__"]["sha256"]),
        "config_seal_sha256": str(config["__provenance__"]["config_seal_sha256"]),
        "normalized_scientific_config_sha256": str(
            config["__provenance__"]["normalized_scientific_config_sha256"]
        ),
        "test_source_sha256": closure[test_path],
        "successor_code_closure_sha256": sha256_json(closure),
        "expected_nodeids_file_sha256": str(
            config["validation"]["expected_nodeids_file_sha256"]
        ),
        "ordered_nodeids_sha256": str(
            config["validation"]["expected_ordered_nodeids_sha256"]
        ),
    }


def parse_passing_junit(
    path: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    assert_config_integrity(config)
    junit = ensure_no_final_outcome_path(path)
    root = ET.parse(junit).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if len(suites) != 1:
        raise PilotContractError("JUnit report must contain exactly one test suite")
    suite = suites[0]
    totals = {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    expected_count = int(config["validation"]["expected_test_cardinality"])
    testcases = suite.findall("testcase")
    testcase_result_counts = {
        "failures": sum(case.find("failure") is not None for case in testcases),
        "errors": sum(case.find("error") is not None for case in testcases),
        "skipped": sum(case.find("skipped") is not None for case in testcases),
    }
    if (
        totals["tests"] != expected_count
        or len(testcases) != expected_count
        or totals["failures"]
        or totals["errors"]
        or totals["skipped"]
        or any(testcase_result_counts.values())
    ):
        raise PilotContractError(
            "test report cardinality/result mismatch: "
            f"declared={totals}, actual_tests={len(testcases)}, "
            f"actual_results={testcase_result_counts}"
        )
    expected_nodeids = load_expected_nodeids(config)
    observed_nodeids: list[str] = []
    for case in testcases:
        classname = str(case.attrib.get("classname", ""))
        name = str(case.attrib.get("name", ""))
        if classname != "tests.test_pipeline" or not name:
            raise PilotContractError("JUnit testcase identity is outside frozen suite")
        observed_nodeids.append(f"tests/test_pipeline.py::{name}")
    if observed_nodeids != expected_nodeids:
        raise PilotContractError("JUnit ordered testcase node-ID set mismatch")
    expected_suite = str(config["validation"]["test_suite_id"])
    if suite.attrib.get("name") != expected_suite:
        raise PilotContractError("JUnit suite identity mismatch")
    properties_element = suite.find("properties")
    properties: dict[str, str] = {}
    if properties_element is not None:
        for property_element in properties_element.findall("property"):
            name = str(property_element.attrib.get("name", ""))
            if not name or name in properties:
                raise PilotContractError("JUnit properties are missing or duplicated")
            properties[name] = str(property_element.attrib.get("value", ""))
    expected_properties = expected_junit_properties(config)
    mismatches = {
        name: (properties.get(name), value)
        for name, value in expected_properties.items()
        if properties.get(name) != value
    }
    if mismatches:
        raise PilotContractError(f"JUnit code/config bindings mismatch: {mismatches}")
    return {
        "status": "pnoise_design_pilot_tests_passed",
        **totals,
        "suite_name": expected_suite,
        "properties": expected_properties,
        "ordered_nodeids": expected_nodeids,
        "ordered_nodeids_sha256": sha256_json(expected_nodeids),
        "junit_sha256": sha256_file(junit),
    }


def validate_test_receipt(
    config: Mapping[str, Any],
    output: str | Path,
    *,
    expected_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    assert_config_integrity(config)
    output_path = Path(output).resolve()
    receipt_path = output_path / "DESIGN_TEST_RECEIPT.json"
    if not receipt_path.is_file():
        raise PilotContractError("DESIGN_TEST_RECEIPT.json is required")
    if (
        expected_receipt_sha256 is not None
        and sha256_file(receipt_path) != expected_receipt_sha256
    ):
        raise PilotContractError("design test receipt digest differs from freeze")
    receipt = load_json(receipt_path)
    junit_path = output_path / "pytest.xml"
    if Path(str(receipt.get("junit_path", ""))).resolve() != junit_path:
        raise PilotContractError("test receipt must bind the canonical JUnit path")
    if not junit_path.is_file():
        raise PilotContractError("bound JUnit report is missing")
    observed = parse_passing_junit(junit_path, config)
    expected = {
        **observed,
        "junit_path": str(junit_path),
        "successor_code_closure": successor_code_closure(config),
        "config": config_provenance(config),
    }
    failures = {
        key: (receipt.get(key), value)
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if failures:
        raise PilotContractError(f"design test receipt is stale: {failures}")
    return receipt


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")
