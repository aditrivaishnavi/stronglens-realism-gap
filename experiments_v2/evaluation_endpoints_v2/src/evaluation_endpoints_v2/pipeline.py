from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow
import scipy
import torch
from grec_pnoise.backend import backend_profile
from grec_pnoise.contracts import load_config as load_grec_config
from grec_pnoise.scoring import (
    _import_fixed_modules,
    fixed_scorer_closure,
    validate_scoring_authority,
)


class ContractError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(f"{prefix}:{value}".encode()).hexdigest()[:32]
    return f"{prefix}-{digest}"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"{path} must contain one JSON object")
    return payload


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def source_tree() -> dict[str, str]:
    root = package_root()
    files = sorted((root / "src/evaluation_endpoints_v2").glob("*.py"))
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in files}


def load_endpoint_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    config = read_json(config_path)
    root = Path(config["repository_root"]).resolve()
    if (
        config.get("schema_version") != 1
        or config.get("study_id") != "stronglens_replacement_evaluation_endpoints_v2"
        or set(config.get("roles", {})) != {"achieved_fpr", "terminal_confirmed"}
    ):
        raise ContractError("endpoint configuration contract differs")
    return config, root


def write_seal(config_path: str | Path) -> dict[str, Any]:
    config, _ = load_endpoint_config(config_path)
    lock = package_root() / "uv.lock"
    if not lock.is_file():
        raise ContractError("endpoint dependency lock is missing")
    seal = {
        "schema_version": 1,
        "status": "evaluation_endpoints_v2_code_and_config_sealed",
        "config_sha256": sha256_file(config_path),
        "dependency_lock_sha256": sha256_file(lock),
        "source_files": source_tree(),
        "scientific_outcomes_accessed": False,
    }
    atomic_json(package_root() / "config/CONFIG_SEAL.json", seal)
    return seal


def validate_seal(config_path: str | Path) -> dict[str, Any]:
    seal_path = package_root() / "config/CONFIG_SEAL.json"
    seal = read_json(seal_path)
    expected = {
        "status": "evaluation_endpoints_v2_code_and_config_sealed",
        "config_sha256": sha256_file(config_path),
        "dependency_lock_sha256": sha256_file(package_root() / "uv.lock"),
        "source_files": source_tree(),
        "scientific_outcomes_accessed": False,
    }
    if any(seal.get(field) != value for field, value in expected.items()):
        raise ContractError("endpoint code/config seal differs")
    return seal


def _bound(root: Path, binding: dict[str, Any]) -> Path:
    path = root / str(binding["path"])
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise ContractError(f"frozen input differs: {binding['path']}")
    return path


def _validate_sources(
    config: dict[str, Any], root: Path, role: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = config["roles"][role]
    source_path = _bound(root, spec["source_manifest"])
    receipt_path = _bound(root, spec["source_receipt"])
    source = pd.read_parquet(source_path)
    receipt = read_json(receipt_path)
    if len(source) != int(spec["rows"]):
        raise ContractError(f"{role} source row count differs")
    if role == "achieved_fpr":
        if (
            receipt.get("status") != "latest_test_negative_roles_frozen_pre_score"
            or receipt.get("roles", {}).get("achieved_fpr", {}).get("sha256")
            != spec["source_manifest"]["sha256"]
            or not source["global_role"].eq(role).all()
            or not source["label"].eq(0).all()
        ):
            raise ContractError("achieved-FPR source authority differs")
        payloads = source
    else:
        payload_path = _bound(root, spec["payload_manifest"])
        payloads = pd.read_parquet(payload_path)
        if (
            receipt.get("status")
            != "payload_validated_reference_cohort_pending_spatial_and_support_checks"
            or receipt.get("manifest_sha256") != spec["source_manifest"]["sha256"]
            or receipt.get("payload_validated_manifest_sha256")
            != spec["payload_manifest"]["sha256"]
            or len(payloads) != 43
            or not source["endpoint_role"].eq("terminal_confirmed").all()
            or not source["final_lensing_status"].eq("confirmed_lens").all()
            or set(source["galaxy_id"].astype(str))
            != set(payloads["galaxy_id"].astype(str))
        ):
            raise ContractError("terminal-confirmed source authority differs")
    return source, payloads


def _expected_manifest(config: dict[str, Any], root: Path, role: str) -> pd.DataFrame:
    spec = config["roles"][role]
    _, payloads = _validate_sources(config, root, role)
    payload_root = root / str(config["payload_root"])
    rows: list[dict[str, Any]] = []
    for row in payloads.sort_values("galaxy_id").itertuples(index=False):
        path = payload_root / str(row.local_relpath)
        file_hash = sha256_file(path)
        if file_hash != str(row.payload_sha256):
            raise ContractError(f"payload file hash differs for {row.galaxy_id}")
        with np.load(path, allow_pickle=False) as archive:
            if "cutout" not in archive.files:
                raise ContractError(f"payload lacks cutout for {row.galaxy_id}")
            image = np.asarray(archive["cutout"])
        if (
            image.shape != (101, 101, 3)
            or image.dtype != np.float32
            or not np.isfinite(image).all()
        ):
            raise ContractError(f"payload array differs for {row.galaxy_id}")
        rows.append(
            {
                "endpoint_id": stable_id(
                    str(spec["identity_prefix"]), str(row.galaxy_id)
                ),
                "galaxy_id": str(row.galaxy_id),
                "global_role": role,
                "payload_path": path.relative_to(root).as_posix(),
                "payload_file_sha256": file_hash,
                "pixel_array_sha256": array_sha256(image),
                "pixel_array_key": "cutout",
                "pixel_layout": "HWC",
                "healpix_128": int(row.healpix_128),
                "physical_group_id": str(row.physical_group_id),
                "spatial_group_id": str(
                    row.evaluation_spatial_group_id
                    if role == "achieved_fpr"
                    else row.spatial_component_id
                ),
            }
        )
    manifest = pd.DataFrame(rows).sort_values("endpoint_id", ignore_index=True)
    if (
        len(manifest) != int(spec["rows"])
        or manifest["endpoint_id"].duplicated().any()
        or manifest["galaxy_id"].duplicated().any()
    ):
        raise ContractError("prepared endpoint manifest support differs")
    return manifest


def prepare_manifest(
    config_path: str | Path, role: str, output_dir: str | Path
) -> dict[str, Any]:
    config, root = load_endpoint_config(config_path)
    validate_seal(config_path)
    if config.get("status") != "frozen_pre_score":
        raise ContractError("endpoint preparation requires frozen_pre_score config")
    spec = config["roles"][role]
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    manifest = _expected_manifest(config, root, role)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "endpoint_manifest.parquet"
    atomic_parquet(manifest_path, manifest)
    receipt = {
        "schema_version": 1,
        "status": spec["manifest_status"],
        "role": role,
        "rows": len(manifest),
        "manifest_sha256": sha256_file(manifest_path),
        "source_manifest_sha256": spec["source_manifest"]["sha256"],
        "source_receipt_sha256": spec["source_receipt"]["sha256"],
        "payload_manifest_sha256": (spec.get("payload_manifest", {}).get("sha256")),
        "payload_files_rehashed": len(manifest),
        "pixel_arrays_rehashed": len(manifest),
        "detector_scores_used": False,
        "package_seal_sha256": sha256_file(package_root() / "config/CONFIG_SEAL.json"),
    }
    atomic_json(output / "ENDPOINT_MANIFEST_RECEIPT.json", receipt)
    return receipt


def _validate_manifest(
    config: dict[str, Any],
    root: Path,
    role: str,
    manifest_path: str | Path,
    receipt_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = config["roles"][role]
    receipt = read_json(receipt_path)
    if (
        receipt.get("status") != spec["manifest_status"]
        or receipt.get("role") != role
        or receipt.get("manifest_sha256") != sha256_file(manifest_path)
        or receipt.get("package_seal_sha256")
        != sha256_file(package_root() / "config/CONFIG_SEAL.json")
        or receipt.get("source_manifest_sha256") != spec["source_manifest"]["sha256"]
        or receipt.get("source_receipt_sha256") != spec["source_receipt"]["sha256"]
        or receipt.get("payload_manifest_sha256")
        != spec.get("payload_manifest", {}).get("sha256")
    ):
        raise ContractError("endpoint manifest receipt differs")
    frame = pd.read_parquet(manifest_path)
    required = {
        "endpoint_id",
        "galaxy_id",
        "global_role",
        "payload_path",
        "payload_file_sha256",
        "pixel_array_sha256",
        "pixel_array_key",
        "pixel_layout",
        "healpix_128",
        "physical_group_id",
        "spatial_group_id",
    }
    if (
        required - set(frame)
        or len(frame) != int(spec["rows"])
        or not frame["global_role"].eq(role).all()
        or frame["endpoint_id"].duplicated().any()
    ):
        raise ContractError("endpoint manifest differs")
    expected = _expected_manifest(config, root, role)
    try:
        pd.testing.assert_frame_equal(frame, expected, check_exact=True)
    except AssertionError as error:
        raise ContractError(
            "endpoint manifest differs from frozen source authorities"
        ) from error
    return frame, receipt


def _images(root: Path, manifest: pd.DataFrame) -> Iterable[np.ndarray]:
    for row in manifest.itertuples(index=False):
        path = root / str(row.payload_path)
        if sha256_file(path) != str(row.payload_file_sha256):
            raise ContractError(f"endpoint payload drift: {row.galaxy_id}")
        with np.load(path, allow_pickle=False) as archive:
            image = np.asarray(archive[str(row.pixel_array_key)])
        if (
            image.shape != (101, 101, 3)
            or image.dtype != np.float32
            or not np.isfinite(image).all()
            or array_sha256(image) != str(row.pixel_array_sha256)
        ):
            raise ContractError(f"endpoint pixel drift: {row.galaxy_id}")
        yield image.transpose(2, 0, 1)


def _wilson(successes: int, n: int, confidence: float) -> list[float]:
    z = float(scipy.stats.norm.ppf(0.5 + confidence / 2.0))
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return [center - half, center + half]


def _spatial_interval(
    frame: pd.DataFrame, draws: int, seed: int, confidence: float
) -> list[float]:
    grouped = (
        frame.groupby("spatial_group_id", as_index=False)
        .agg(
            detections=("detected_at_frozen_threshold", "sum"),
            rows=("endpoint_id", "size"),
        )
        .sort_values("spatial_group_id", ignore_index=True)
    )
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    estimates = np.empty(draws, dtype=np.float64)
    detections = grouped["detections"].to_numpy(dtype=np.float64)
    rows = grouped["rows"].to_numpy(dtype=np.float64)
    for index in range(draws):
        multiplier = rng.exponential(size=len(grouped))
        estimates[index] = float(
            np.sum(multiplier * detections) / np.sum(multiplier * rows)
        )
    tail = (1.0 - confidence) / 2.0
    return [
        float(np.quantile(estimates, tail)),
        float(np.quantile(estimates, 1.0 - tail)),
    ]


def _result_summary(
    config: dict[str, Any], role: str, scores: pd.DataFrame
) -> dict[str, Any]:
    confidence = float(config["inference"]["confidence_level"])
    detections = int(scores["detected_at_frozen_threshold"].sum())
    return {
        "schema_version": 1,
        "status": config["roles"][role]["result_status"],
        "role": role,
        "rows": len(scores),
        "detections": detections,
        "fraction": detections / len(scores),
        "row_wilson_interval": (
            _wilson(detections, len(scores), confidence)
            if role == "achieved_fpr"
            else None
        ),
        "spatial_groups": int(scores["spatial_group_id"].nunique()),
        "physical_groups": int(scores["physical_group_id"].nunique()),
        "spatial_multiplier_interval": (
            _spatial_interval(
                scores,
                int(config["inference"]["spatial_multiplier_draws"]),
                int(config["inference"]["spatial_multiplier_seed"]),
                confidence,
            )
            if role == "achieved_fpr"
            else None
        ),
        "interpretation": (
            "Catalogue-negative exceedance at the once-frozen threshold; "
            "this result did not recalibrate the threshold."
            if role == "achieved_fpr"
            else "Exact selected finite-cohort response at the once-frozen threshold."
        ),
    }


def score_endpoint(
    config_path: str | Path,
    role: str,
    manifest_path: str | Path,
    manifest_receipt_path: str | Path,
    output_dir: str | Path,
    *,
    device: str,
) -> dict[str, Any]:
    config, root = load_endpoint_config(config_path)
    validate_seal(config_path)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest, _ = _validate_manifest(
        config, root, role, manifest_path, manifest_receipt_path
    )
    grec_config_path = _bound(root, config["grec_config"])
    threshold_path = _bound(root, config["threshold_receipt"])
    grec_config = load_grec_config(grec_config_path)
    threshold_receipt, threshold = validate_scoring_authority(
        grec_config, threshold_path, device=device
    )
    if int(np.asarray(threshold).view(np.uint32)) != int(
        config["threshold_receipt"]["float32_bits"]
    ):
        raise ContractError("frozen threshold bits differ")
    fixed, preprocess, utilities = _import_fixed_modules(grec_config)
    model, preprocess_kwargs = utilities.load_model_and_spec(
        grec_config["scoring"]["checkpoint"], device
    )
    model = model.to(device).eval()
    kernels = fixed.pin_deterministic_kernels()
    decoupling = fixed.assert_no_batch_coupling(model)

    def processed() -> Iterable[np.ndarray]:
        for image in _images(root, manifest):
            yield preprocess.preprocess_stack(image, **preprocess_kwargs)

    width = int(grec_config["scoring"]["batch_size"])
    logits64, batches = fixed.fixed_width_logits(
        model,
        fixed.array_batches(processed(), width),
        batch_size=width,
        device=device,
        filler=str(grec_config["scoring"]["filler_policy"]),
    )
    logits = logits64.astype(np.float32)
    if logits.shape != (len(manifest),):
        raise ContractError("endpoint scorer row coverage differs")
    scores = manifest[
        [
            "endpoint_id",
            "galaxy_id",
            "global_role",
            "spatial_group_id",
            "physical_group_id",
        ]
    ].copy()
    scores["raw_logit_float32"] = logits
    scores["raw_logit_float32_bits"] = logits.view(np.uint32)
    scores["detected_at_frozen_threshold"] = logits >= threshold
    scores_path = output / "endpoint_scores.parquet"
    atomic_parquet(scores_path, scores)
    result = _result_summary(config, role, scores)
    receipt = {
        "schema_version": 1,
        "status": config["roles"][role]["score_status"],
        "role": role,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_receipt_sha256": sha256_file(manifest_receipt_path),
        "scores_sha256": sha256_file(scores_path),
        "rows": len(scores),
        "threshold_receipt_sha256": sha256_file(threshold_path),
        "threshold_float32_bits": int(np.asarray(threshold).view(np.uint32)),
        "comparison": ">=",
        "checkpoint_sha256": grec_config["scoring"]["checkpoint_sha256"],
        "requested_device": device,
        "scorer_closure": fixed_scorer_closure(grec_config, device),
        "backend_profile": backend_profile(grec_config, device),
        "batches": batches,
        "batch_decoupling": decoupling,
        "kernels": kernels,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "torch": torch.__version__,
        },
        "threshold_was_recalibrated": False,
        "threshold_source_status": threshold_receipt["status"],
        "result": result,
        "package_seal_sha256": sha256_file(package_root() / "config/CONFIG_SEAL.json"),
    }
    atomic_json(output / "ENDPOINT_SCORE_RECEIPT.json", receipt)
    atomic_json(output / "ENDPOINT_RESULT.json", result)
    return receipt


def verify_endpoint(
    config_path: str | Path,
    role: str,
    manifest_path: str | Path,
    manifest_receipt_path: str | Path,
    scores_path: str | Path,
    score_receipt_path: str | Path,
    result_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    config, root = load_endpoint_config(config_path)
    validate_seal(config_path)
    manifest, _ = _validate_manifest(
        config, root, role, manifest_path, manifest_receipt_path
    )
    receipt = read_json(score_receipt_path)
    scores = pd.read_parquet(scores_path)
    result = read_json(result_path)
    threshold_path = _bound(root, config["threshold_receipt"])
    grec_config = load_grec_config(_bound(root, config["grec_config"]))
    device = str(receipt.get("requested_device", ""))
    _, threshold = validate_scoring_authority(
        grec_config, threshold_path, device=device
    )
    scorer_closure = fixed_scorer_closure(grec_config, device)
    profile = backend_profile(grec_config, device)
    if (
        receipt.get("status") != config["roles"][role]["score_status"]
        or receipt.get("role") != role
        or receipt.get("scores_sha256") != sha256_file(scores_path)
        or receipt.get("manifest_sha256") != sha256_file(manifest_path)
        or receipt.get("manifest_receipt_sha256") != sha256_file(manifest_receipt_path)
        or receipt.get("threshold_receipt_sha256") != sha256_file(threshold_path)
        or receipt.get("threshold_float32_bits")
        != int(np.asarray(threshold).view(np.uint32))
        or receipt.get("comparison") != ">="
        or receipt.get("checkpoint_sha256")
        != grec_config["scoring"]["checkpoint_sha256"]
        or receipt.get("scorer_closure") != scorer_closure
        or receipt.get("backend_profile") != profile
        or receipt.get("package_seal_sha256")
        != sha256_file(package_root() / "config/CONFIG_SEAL.json")
        or receipt.get("threshold_was_recalibrated") is not False
        or len(scores) != len(manifest)
        or set(scores["endpoint_id"]) != set(manifest["endpoint_id"])
    ):
        raise ContractError("endpoint score authority differs")
    identity_columns = [
        "endpoint_id",
        "galaxy_id",
        "global_role",
        "spatial_group_id",
        "physical_group_id",
    ]
    try:
        pd.testing.assert_frame_equal(
            scores[identity_columns].reset_index(drop=True),
            manifest[identity_columns].reset_index(drop=True),
            check_exact=True,
        )
    except AssertionError as error:
        raise ContractError(
            "endpoint score identities or groups differ from manifest"
        ) from error
    authoritative_scores = manifest[identity_columns].merge(
        scores[
            [
                "endpoint_id",
                "raw_logit_float32",
                "raw_logit_float32_bits",
                "detected_at_frozen_threshold",
            ]
        ],
        on="endpoint_id",
        validate="one_to_one",
    )
    for _ in _images(root, manifest):
        pass
    if (
        scores["raw_logit_float32"].dtype != np.dtype("float32")
        or scores["raw_logit_float32_bits"].dtype != np.dtype("uint32")
    ):
        raise ContractError("endpoint scores are not native float32/uint32")
    logits = authoritative_scores["raw_logit_float32"].to_numpy(copy=True)
    if not np.array_equal(
        logits.view(np.uint32),
        authoritative_scores["raw_logit_float32_bits"].to_numpy(dtype=np.uint32),
    ) or not np.array_equal(
        logits >= threshold,
        authoritative_scores["detected_at_frozen_threshold"].to_numpy(dtype=bool),
    ):
        raise ContractError("endpoint score bits or decisions differ")
    expected_result = _result_summary(config, role, authoritative_scores)
    if result != expected_result or receipt.get("result") != result:
        raise ContractError("endpoint result arithmetic differs")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    verification = {
        "schema_version": 1,
        "status": "PASS",
        "role": role,
        "rows": len(scores),
        "payload_files_and_arrays_rehashed": len(scores),
        "score_bits_recomputed": True,
        "threshold_decisions_recomputed": True,
        "result_recomputed": True,
        "manifest_sha256": sha256_file(manifest_path),
        "scores_sha256": sha256_file(scores_path),
        "score_receipt_sha256": sha256_file(score_receipt_path),
        "result_sha256": sha256_file(result_path),
    }
    atomic_json(output_path, verification)
    return verification
