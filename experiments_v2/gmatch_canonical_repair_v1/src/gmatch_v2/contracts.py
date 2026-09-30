"""Small fail-closed primitives for G_match v2 preparation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SHA256_LENGTH = 64


class ContractError(RuntimeError):
    """Raised whenever an immutable scientific contract is not satisfied."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _normalise(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return {"__float_hex__": value.hex()}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    return str(value)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        _normalise(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_row_sha256(row: Mapping[str, Any], columns: Sequence[str]) -> str:
    return canonical_json_sha256([(column, row.get(column)) for column in columns])


def frame_content_sha256(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    sort_by: Sequence[str],
) -> str:
    ordered = frame.sort_values(list(sort_by), kind="mergesort", ignore_index=True)
    hashes = [
        canonical_row_sha256(row, columns)
        for row in ordered[list(columns)].to_dict(orient="records")
    ]
    return canonical_json_sha256(hashes)


def stable_digest(*parts: object) -> str:
    return canonical_json_sha256([str(part) for part in parts])


def geographic_spatial_key(healpix_128: object) -> str:
    """Return one typed NSIDE=128 nested HEALPix dependency namespace."""
    if pd.isna(healpix_128):
        raise ContractError("geographic spatial key cannot be null")
    try:
        numeric = float(healpix_128)
        integer = int(numeric)
    except (TypeError, ValueError, OverflowError) as error:
        raise ContractError("geographic spatial key must be an integer") from error
    if numeric != integer or not 0 <= integer < 12 * 128**2:
        raise ContractError("geographic spatial key is outside NSIDE=128")
    return f"healpix128:{integer}"


def require_columns(
    frame: pd.DataFrame, columns: Iterable[str], *, context: str
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ContractError(f"{context} lacks required columns: {missing}")


def require_nonempty(
    frame: pd.DataFrame, columns: Iterable[str], *, context: str
) -> None:
    require_columns(frame, columns, context=context)
    for column in columns:
        values = frame[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ContractError(f"{context} has empty {column}")


def require_sha256(value: object, *, context: str) -> str:
    text = str(value)
    if len(text) != SHA256_LENGTH:
        raise ContractError(f"{context} is not a SHA-256 digest")
    try:
        int(text, 16)
    except ValueError as error:
        raise ContractError(f"{context} is not a SHA-256 digest") from error
    return text


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ContractError(f"path escapes repository root: {path}") from error


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


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


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    expected_study = {
        "status": "adopted_execution_authority_pre_feature",
        "scope": "score_blind_final_preparation_only",
        "data_role": "final",
    }
    for field, expected in expected_study.items():
        if config.get("study", {}).get(field) != expected:
            raise ContractError(f"configuration has wrong study.{field}")
    execution = config.get("execution", {})
    required_true = (
        "detector_scores_forbidden",
        "detector_outcomes_forbidden",
        "feature_extraction_forbidden",
        "probe_fitting_forbidden",
        "bootstrap_forbidden",
        "stop_before_feature_authorization",
    )
    if not all(execution.get(field) is True for field in required_true):
        raise ContractError("configuration does not fail closed before features")
    matching = config.get("matching", {})
    expected_matching = {
        "candidate_population": "exact_final_grec_hosts",
        "centre": "median",
        "scale": "interquartile_range",
        "scaling_population": "exact_final_grec_hosts_only",
        "scaler_value_policy": (
            "component_wise_all_finite_values_exact_authorized_roster"
        ),
        "scaler_authority_rows": 4500,
        "candidate_eligibility": "all_five_covariates_finite",
        "missing_covariate_policy": (
            "retain_in_scaler_authority_accounting_mark_ineligible_no_imputation"
        ),
        "imputation": False,
        "expected_complete_matching_candidates": 4500,
        "expected_nonfinite_observed_zmag_hosts": 0,
        "prior_scaler_comparator": "complete_case_4265_hosts_non_governing",
        "distance": "robust_diagonal_euclidean",
        "component_caliper_standardised": 2.0,
        "total_distance_caliper": 3.0,
        "targets_per_reference": 1,
        "replacement": False,
        "assignment": "maximum_cardinality_then_minimum_distance",
        "tie_rule": "sha256_sorted_identifiers",
        "host_total_flux_floor": "forbidden",
    }
    for field, expected in expected_matching.items():
        if matching.get(field) != expected:
            raise ContractError(f"configuration has drifted matching.{field}")
    expected_scaler = {
        "observed_zmag": {
            "finite": 4500,
            "missing": 0,
            "median": 19.39710237176541,
            "iqr": 1.0567950650698492,
        },
        "sersic_n": {
            "finite": 4500,
            "missing": 0,
            "median": 2.5997047424316406,
            "iqr": 3.0,
        },
        "axis_ratio_q": {
            "finite": 4500,
            "missing": 0,
            "median": 0.6087695572740404,
            "iqr": 0.34962079133933555,
        },
        "psf_fwhm_r_arcsec": {
            "finite": 4500,
            "missing": 0,
            "median": 1.3872898653320314,
            "iqr": 0.31376147270202637,
        },
        "depth5_r_mag": {
            "finite": 4500,
            "missing": 0,
            "median": 24.295993050549264,
            "iqr": 0.41633649881539725,
        },
    }
    if matching.get("expected_scaler") != expected_scaler:
        raise ContractError("configuration has drifted matching.expected_scaler")
    from .schemas import FORBIDDEN_ALIASES

    schema_policy = config.get("schema_policy", {})
    if (
        schema_policy.get("mode") != "strict_context_specific_allowlists"
        or schema_policy.get("unknown_columns") != "reject"
        or schema_policy.get("missing_columns") != "reject"
        or schema_policy.get("duplicate_columns") != "reject"
        or set(schema_policy.get("forbidden_aliases", [])) != FORBIDDEN_ALIASES
    ):
        raise ContractError("configuration has drifted strict schema policy")
    if config.get("pairing", {}).get("theta_e_arcsec") != 1.5:
        raise ContractError("configuration has wrong G_match radius")
    return config


def resolve_path(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def authority_paths(config: Mapping[str, Any], root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name, entry in config["authorities"].items():
        path = resolve_path(root, str(entry["path"]))
        if not path.is_file():
            raise ContractError(f"authority is missing: {name}")
        expected = require_sha256(entry["sha256"], context=f"{name} configured hash")
        if sha256_file(path) != expected:
            raise ContractError(f"authority bytes changed: {name}")
        result[name] = path
    return result


def source_tree_sha256(package_root: Path) -> tuple[str, dict[str, str]]:
    selected = [
        package_root / "pyproject.toml",
        package_root / "README.md",
        *sorted((package_root / "src").rglob("*.py")),
        *sorted((package_root / "scripts").glob("*.py")),
        *sorted((package_root / "tests").glob("*.py")),
    ]
    hashes = {
        path.relative_to(package_root).as_posix(): sha256_file(path)
        for path in selected
    }
    return canonical_json_sha256(hashes), hashes
