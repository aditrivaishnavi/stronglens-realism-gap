from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


class ContractError(RuntimeError):
    """Raised instead of weakening a frozen scientific or provenance contract."""


SCORE_TOKENS = ("score", "logit", "probability", "detected", "threshold")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_bytes_sha256(array: np.ndarray) -> str:
    """Hash exact C-order element bytes; dtype/shape are recorded separately."""
    value = np.asarray(array)
    if not value.flags.c_contiguous:
        value = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


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


def stable_id(prefix: str, value: Any, length: int = 32) -> str:
    digest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return f"{prefix}_{digest[:length]}"


def seed_from_key(namespace: str, *parts: object) -> int:
    material = canonical_json_bytes([namespace, *map(str, parts)])
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("study", {}).get("status") != (
        "implementation_candidate_not_execution_authority"
    ):
        raise ContractError("configuration status is not the reviewed candidate state")
    documents = config.get("governance", {}).get("documents", {})
    if not documents:
        raise ContractError("configuration does not bind governing documents")
    for document, expected in documents.items():
        observed = sha256_file(document)
        if observed != expected:
            raise ContractError(
                f"governing document drift at {document}: {observed} != {expected}"
            )
    config["__provenance__"] = {
        "config_path": str(config_path),
        "config_file_sha256": sha256_file(config_path),
        "config_contract_sha256": sha256_json(config),
        "governing_document_sha256": dict(documents),
    }
    return config


def config_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    provenance = config.get("__provenance__")
    if not isinstance(provenance, dict):
        raise ContractError("configuration was not loaded through load_config")
    return dict(provenance)


def assert_config_integrity(config: Mapping[str, Any]) -> None:
    provenance = config_provenance(config)
    current = {
        key: value for key, value in config.items() if key != "__provenance__"
    }
    observed = sha256_json(current)
    if observed != provenance["config_contract_sha256"]:
        raise ContractError(
            "in-memory configuration drifted after load; reload a reviewed config"
        )
    path = Path(str(provenance["config_path"]))
    if sha256_file(path) != provenance["config_file_sha256"]:
        raise ContractError("configuration file changed after load")


def scientific_section_sha256(
    config: Mapping[str, Any], *section_names: str
) -> str:
    missing = [name for name in section_names if name not in config]
    if missing:
        raise ContractError(f"configuration lacks scientific sections: {missing}")
    return sha256_json({name: config[name] for name in section_names})


def package_source_sha256(*module_filenames: str) -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        name: sha256_file(root / name)
        for name in sorted(module_filenames)
    }


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain one JSON object")
    return value


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
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


def require_receipt(
    receipt_path: str | Path,
    *,
    status: str,
    artifact_path: str | Path | None = None,
    artifact_digest_field: str | None = None,
) -> dict[str, Any]:
    receipt = load_json(receipt_path)
    if receipt.get("status") != status:
        raise ContractError(
            f"receipt status {receipt.get('status')!r} is not required {status!r}"
        )
    if artifact_path is not None:
        if not artifact_digest_field:
            raise ValueError("artifact_digest_field is required with artifact_path")
        observed = sha256_file(artifact_path)
        if receipt.get(artifact_digest_field) != observed:
            raise ContractError(
                f"{artifact_digest_field} does not bind {artifact_path}: {observed}"
            )
    return receipt


def verify_final_host_receipt(
    manifest_path: str | Path,
    receipt_path: str | Path,
    required_status: str,
) -> dict[str, Any]:
    receipt = require_receipt(
        receipt_path,
        status=required_status,
        artifact_path=manifest_path,
        artifact_digest_field="host_manifest_sha256",
    )
    required = {
        "detector_scores_used": False,
        "full_tractor_eligibility": "passed",
        "physical_group_checks": "passed",
        "role_overlap_checks": "passed",
        "pre_score_review": "approved",
    }
    mismatches = {
        key: (receipt.get(key), expected)
        for key, expected in required.items()
        if receipt.get(key) != expected
    }
    if mismatches:
        raise ContractError(f"host receipt has unresolved finalization gates: {mismatches}")
    return receipt


def assert_score_blind_columns(columns: list[str] | tuple[str, ...]) -> None:
    forbidden = [
        name
        for name in columns
        if any(token in name.lower() for token in SCORE_TOKENS)
    ]
    if forbidden:
        raise ContractError(f"pre-score artifact contains outcome columns: {forbidden}")


def runtime_record() -> dict[str, Any]:
    import pandas
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "torch": torch.__version__,
        "torchvision": _optional_version("torchvision"),
        "device_names": (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        ),
    }


def _optional_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
