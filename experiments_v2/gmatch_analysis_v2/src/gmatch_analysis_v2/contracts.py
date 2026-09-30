"""Small fail-closed contracts for the isolated G_match analysis."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd


class ContractError(RuntimeError):
    """Raised when an analysis authority or result contract does not hold."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ContractError(f"{context} must be a lowercase SHA-256")
    return value


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return payload


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
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


def _canonical_scalar(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value).hex()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


def canonical_frame_sha256(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    sort_by: Sequence[str],
) -> str:
    selected = frame.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
    digest = hashlib.sha256()
    for source in selected[list(columns)].to_dict(orient="records"):
        row = {column: _canonical_scalar(source[column]) for column in sorted(columns)}
        digest.update(canonical_json_sha256(row).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def sequence_sha256(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def resolve_bound_path(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ContractError(f"unsafe repository-relative path: {relative}")
    path = root.joinpath(*value.parts)
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ContractError(f"path escapes repository root: {relative}") from error
    return path


def verify_file_binding(
    root: Path,
    binding: Mapping[str, Any],
    *,
    context: str,
) -> Path:
    path = resolve_bound_path(root, str(binding.get("path")))
    if not path.is_file():
        raise ContractError(f"bound {context} is missing: {path}")
    expected = require_sha256(binding.get("sha256"), context=f"{context}.sha256")
    observed = sha256_file(path)
    if observed != expected:
        raise ContractError(f"bound {context} hash differs: {observed} != {expected}")
    return path


def package_source_bindings(package_root: Path) -> dict[str, str]:
    fixed = [".gitignore", "README.md", "pyproject.toml"]
    paths = [package_root / name for name in fixed]
    paths.extend(sorted((package_root / "src").rglob("*.py")))
    paths.extend(sorted((package_root / "tests").rglob("*.py")))
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise ContractError(f"package source file missing: {missing}")
    return {
        str(path.relative_to(package_root)): sha256_file(path) for path in sorted(paths)
    }


def package_source_tree_sha256(package_root: Path) -> str:
    return canonical_json_sha256(package_source_bindings(package_root))


def load_config(root: Path) -> tuple[dict[str, Any], Path]:
    path = root / "experiments_v2/gmatch_analysis_v2/config/production.json"
    config = read_json(path)
    if (
        config.get("schema_version") != 1
        or config.get("status") != "adopted_final_G_match_analysis_only"
        or config.get("analysis") != "G_match"
        or config.get("scope") != "final_analysis_only"
    ):
        raise ContractError("analysis config is not the adopted final G_match contract")
    return config, path


def validate_config_seal(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    package_root = root / "experiments_v2/gmatch_analysis_v2"
    config_path = package_root / "config/production.json"
    seal_path = package_root / "config/CONFIG_SEAL.json"
    seal = read_json(seal_path)
    expected = {
        "schema_version": 1,
        "status": "adopted_final_G_match_analysis_config_sealed",
        "scope": "final_analysis_only",
        "analysis": "G_match",
        "config_path": str(config_path.relative_to(root)),
        "config_sha256": sha256_file(config_path),
        "dependency_lock_path": str((package_root / "uv.lock").relative_to(root)),
        "dependency_lock_sha256": sha256_file(package_root / "uv.lock"),
        "locked_python": "==3.12.13",
        "package_source_files": package_source_bindings(package_root),
        "package_source_tree_sha256": package_source_tree_sha256(package_root),
        "registered_implementation": config["registered_implementation"],
        "upstream_source_closure": config["upstream_source_closure"],
        "analysis_producer_uuid": config["analysis_producer_uuid"],
        "metric_verifier_uuid": config["metric_verifier_uuid"],
        "feature_producer_uuid": config["feature_producer_uuid"],
        "feature_verifier_uuid": config["feature_verifier_uuid"],
        "independent_feature_pass_uuid": config["independent_feature_pass_uuid"],
        "feature_access_authorized": True,
        "probe_fitting_authorized": True,
        "bootstrap_authorized": False,
        "G_rec_score_access_authorized": False,
        "manuscript_edit_authorized": False,
    }
    if seal != expected:
        raise ContractError("analysis config seal is missing, stale, or inconsistent")
    return seal


def write_config_seal(root: Path) -> dict[str, Any]:
    config, _ = load_config(root)
    package_root = root / "experiments_v2/gmatch_analysis_v2"
    seal_path = package_root / "config/CONFIG_SEAL.json"
    if seal_path.exists():
        raise FileExistsError(f"seal refuses to overwrite {seal_path}")
    payload = {
        "schema_version": 1,
        "status": "adopted_final_G_match_analysis_config_sealed",
        "scope": "final_analysis_only",
        "analysis": "G_match",
        "config_path": "experiments_v2/gmatch_analysis_v2/config/production.json",
        "config_sha256": sha256_file(package_root / "config/production.json"),
        "dependency_lock_path": "experiments_v2/gmatch_analysis_v2/uv.lock",
        "dependency_lock_sha256": sha256_file(package_root / "uv.lock"),
        "locked_python": "==3.12.13",
        "package_source_files": package_source_bindings(package_root),
        "package_source_tree_sha256": package_source_tree_sha256(package_root),
        "registered_implementation": config["registered_implementation"],
        "upstream_source_closure": config["upstream_source_closure"],
        "analysis_producer_uuid": config["analysis_producer_uuid"],
        "metric_verifier_uuid": config["metric_verifier_uuid"],
        "feature_producer_uuid": config["feature_producer_uuid"],
        "feature_verifier_uuid": config["feature_verifier_uuid"],
        "independent_feature_pass_uuid": config["independent_feature_pass_uuid"],
        "feature_access_authorized": True,
        "probe_fitting_authorized": True,
        "bootstrap_authorized": False,
        "G_rec_score_access_authorized": False,
        "manuscript_edit_authorized": False,
    }
    atomic_write_json(payload, seal_path)
    return payload


def runtime_environment() -> dict[str, str]:
    names = (
        "gmatch-analysis-v2",
        "joblib",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "scipy",
        "stronglens-probes",
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **{name: importlib.metadata.version(name) for name in names},
    }


def validate_runtime(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    expected_prefix = (root / "experiments_v2/gmatch_analysis_v2/.venv").resolve()
    if (
        platform.python_implementation() != "CPython"
        or platform.python_version() != "3.12.13"
        or Path(sys.prefix).resolve() != expected_prefix
    ):
        raise ContractError("analysis is not running in its locked CPython environment")
    observed = runtime_environment()
    for name, version in config["runtime"]["direct_packages"].items():
        if observed.get(name) != version:
            raise ContractError(
                f"runtime package {name} differs: {observed.get(name)} != {version}"
            )
    return observed
