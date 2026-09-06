from __future__ import annotations

import json
import os
import platform
import subprocess
from collections.abc import Mapping
from typing import Any

import torch

from .contracts import ContractError, runtime_record, sha256_json


def configure_determinism(config: Mapping[str, Any]) -> None:
    scoring = config["scoring"]
    if bool(scoring["deterministic_algorithms"]):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = bool(scoring["cudnn_deterministic"])
    torch.backends.cudnn.benchmark = bool(scoring["cudnn_benchmark"])
    torch.set_float32_matmul_precision(str(scoring["float32_matmul_precision"]))
    for name, value in scoring.get("mps_environment", {}).items():
        existing = os.environ.get(name)
        if existing is not None and existing != str(value):
            raise ContractError(
                f"MPS environment {name}={existing!r}, expected {value!r}"
            )
        os.environ[name] = str(value)


MPS_ENVIRONMENT_KEYS = (
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "PYTORCH_MPS_FAST_MATH",
    "PYTORCH_MPS_PREFER_METAL",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO",
    "PYTORCH_DEBUG_MPS_ALLOCATOR",
    "MTL_CAPTURE_ENABLED",
    "METAL_DEVICE_WRAPPER_TYPE",
)


def _apple_hardware_identity() -> dict[str, Any]:
    if platform.system() != "Darwin":
        raise ContractError("MPS requires Darwin hardware identity")
    try:
        completed = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        item = json.loads(completed.stdout)["SPHardwareDataType"][0]
        sysctl_model = subprocess.run(
            ["sysctl", "-n", "hw.model"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        sysctl_chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (KeyError, ValueError, OSError, subprocess.SubprocessError) as error:
        raise ContractError("could not resolve exact Apple hardware identity") from error
    identity = {
        "chip_type": str(item.get("chip_type", "")),
        "machine_model": str(item.get("machine_model", "")),
        "machine_name": str(item.get("machine_name", "")),
        "model_number": str(item.get("model_number", "")),
        "number_processors": str(item.get("number_processors", "")),
        "physical_memory": str(item.get("physical_memory", "")),
        "sysctl_hw_model": sysctl_model,
        "sysctl_cpu_brand_string": sysctl_chip,
    }
    if (
        not identity["chip_type"]
        or not identity["machine_model"]
        or identity["chip_type"] != sysctl_chip
        or identity["machine_model"] != sysctl_model
    ):
        raise ContractError(f"inconsistent Apple hardware identity: {identity}")
    return identity


def backend_profile(config: Mapping[str, Any], requested_device: str) -> dict[str, Any]:
    """Complete hardware/runtime/determinism identity used by threshold and scores."""
    configure_determinism(config)
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ContractError("requested CUDA backend is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ContractError("requested MPS backend is unavailable")
    device_index = device.index
    if device.type == "cuda":
        device_index = torch.cuda.current_device() if device_index is None else device_index
        properties = torch.cuda.get_device_properties(device_index)
        hardware = {
            "name": properties.name,
            "total_memory": int(properties.total_memory),
            "capability": list(torch.cuda.get_device_capability(device_index)),
            "multi_processor_count": int(properties.multi_processor_count),
        }
    elif device.type == "mps":
        hardware = {
            "name": "Apple Metal Performance Shaders",
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "apple": _apple_hardware_identity(),
        }
    else:
        hardware = {
            "name": platform.processor() or platform.machine(),
            "machine": platform.machine(),
        }
    profile: dict[str, Any] = {
        "schema_version": 1,
        "requested_device": requested_device,
        "device_type": device.type,
        "device_index": device_index,
        "hardware": hardware,
        "runtime": runtime_record(),
        "torch_build_config_sha256": sha256_json(torch.__config__.show()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "determinism": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_debug_mode": int(torch.get_deterministic_debug_mode()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "mps_environment": {
                name: os.environ.get(name) for name in MPS_ENVIRONMENT_KEYS
            },
        },
    }
    profile["profile_sha256"] = sha256_json(profile)
    return profile


def verify_backend_profile(
    expected: Mapping[str, Any],
    config: Mapping[str, Any],
    requested_device: str,
) -> dict[str, Any]:
    observed = backend_profile(config, requested_device)
    if expected.get("profile_sha256") != observed["profile_sha256"]:
        raise ContractError(
            "backend/hardware/determinism profile differs from threshold scoring: "
            f"{observed['profile_sha256']}"
        )
    if dict(expected) != observed:
        raise ContractError("backend profile body differs despite its claimed digest")
    return observed
