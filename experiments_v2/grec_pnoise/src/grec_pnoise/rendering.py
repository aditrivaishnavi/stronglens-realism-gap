from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_functional

from .contracts import (
    ContractError,
    array_bytes_sha256,
    assert_config_integrity,
    assert_score_blind_columns,
    atomic_write_json,
    config_provenance,
    load_json,
    package_source_sha256,
    runtime_record,
    scientific_section_sha256,
    sha256_file,
)
from .scene_table import verify_scene_identity


def _grid(
    height: int,
    width: int,
    pixel_scale_arcsec: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    yy = (
        torch.arange(height, device=device, dtype=dtype) - (height - 1) / 2.0
    ) * pixel_scale_arcsec
    xx = (
        torch.arange(width, device=device, dtype=dtype) - (width - 1) / 2.0
    ) * pixel_scale_arcsec
    y, x = torch.meshgrid(yy, xx, indexing="ij")
    return x, y


def _sie_deflection(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    theta_e: float,
    q: float,
    phi: float,
    x0: float,
    y0: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (0.0 < q <= 1.0):
        raise ContractError(f"SIE axis ratio must be in (0,1], got {q}")
    theta = x.new_tensor(theta_e)
    dx, dy = x - x0, y - y0
    cosine, sine = math.cos(phi), math.sin(phi)
    xp = cosine * dx + sine * dy
    yp = -sine * dx + cosine * dy
    epsilon = 1.0e-6
    if abs(q - 1.0) < 1.0e-6:
        radius = torch.sqrt(xp.square() + yp.square() + epsilon)
        axp, ayp = theta * xp / radius, theta * yp / radius
    else:
        q_tensor = x.new_tensor(q)
        f = torch.sqrt(1.0 - q_tensor.square())
        omega = torch.sqrt(q_tensor.square() * xp.square() + yp.square() + epsilon**2)
        scale = theta * torch.sqrt(q_tensor) / f
        axp = scale * torch.atan(f * xp / omega.clamp_min(epsilon))
        argument = (f * yp / omega.clamp_min(epsilon)).clamp(-0.9999, 0.9999)
        ayp = scale * torch.atanh(argument)
    return cosine * axp - sine * ayp, sine * axp + cosine * ayp


def _sersic_integral(re_arcsec: float, n_sersic: float, q: float) -> float:
    b_n = 2.0 * n_sersic - 1.0 / 3.0
    log_value = (
        math.log(2.0 * math.pi)
        + math.log(q)
        + math.log(n_sersic)
        + 2.0 * math.log(re_arcsec)
        + b_n
        + math.lgamma(2.0 * n_sersic)
        - 2.0 * n_sersic * math.log(b_n)
    )
    return math.exp(log_value)


def zero_padded_gaussian_convolution(
    image: torch.Tensor,
    sigma_pixels: float,
    *,
    truncate_sigma: float = 8.0,
) -> torch.Tensor:
    """Convolve a 2-D image with an explicitly zero-padded Gaussian kernel."""
    if image.ndim != 2:
        raise ValueError("image must be two-dimensional")
    if not math.isfinite(sigma_pixels) or sigma_pixels <= 0.0:
        raise ContractError("Gaussian sigma must be positive and finite")
    if truncate_sigma < 4.0:
        raise ContractError("Gaussian truncation below 4 sigma is not accepted")
    radius = max(1, int(math.ceil(truncate_sigma * sigma_pixels)))
    coordinate = torch.arange(
        -radius, radius + 1, device=image.device, dtype=image.dtype
    )
    kernel_1d = torch.exp(-0.5 * (coordinate / sigma_pixels).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    padded = torch_functional.pad(
        image[None, None], (radius, radius, radius, radius), mode="constant", value=0.0
    )
    return torch_functional.conv2d(padded, kernel_2d[None, None])[0, 0]


@dataclass(frozen=True)
class AdaptiveRenderResult:
    image_chw: np.ndarray
    selected_order: int
    unlensed_flux_relative_error: float
    previous_order_relative_l1: float
    previous_order_relative_max: float
    order_history: tuple[dict[str, float | int], ...]


def validate_renderer_validation_receipt(
    receipt_path: str | Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = load_json(receipt_path)
    required_true = (
        "full_source_prior_flux_gate_passed",
        "lensed_image_convergence_gate_passed",
        "logit_convergence_gate_passed",
        "sie_parity_gate_passed",
        "psf_gate_passed",
        "production_functions_used",
        "final_outcomes_used_false",
    )
    if receipt.get("status") != config["render"]["required_validation_receipt_status"]:
        raise ContractError("renderer validation receipt is not frozen")
    if receipt.get("config") != config_provenance(config):
        raise ContractError("renderer validation receipt is stale")
    expected_source = package_source_sha256(
        "cells.py", "contracts.py", "rendering.py", "scene_table.py", "validation.py"
    )
    if receipt.get("render_implementation_source_sha256") != expected_source:
        raise ContractError("renderer validation code closure is stale")
    failed = [field for field in required_true if receipt.get(field) is not True]
    if failed:
        raise ContractError(f"renderer validation gates failed: {failed}")
    return receipt


@lru_cache(maxsize=32)
def _legendre_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 2:
        raise ContractError("Gauss-Legendre order must be at least two")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return nodes.astype(np.float64), weights.astype(np.float64)


def _integrated_sersic_shape(
    row: Mapping[str, Any],
    shape_hwc: tuple[int, int, int],
    *,
    order: int,
    lensed: bool,
    device: str = "cpu",
    native_row_chunk: int = 8,
) -> torch.Tensor:
    """Pixel-integrate the dimensionless source with tensor Gauss-Legendre."""
    height, width, channels = shape_hwc
    if channels != 3:
        raise ContractError("host must have exactly three HWC bands")
    pixel_scale = float(row["pixel_scale_arcsec"])
    dtype = torch.float32
    torch_device = torch.device(device)
    nodes_np, weights_np = _legendre_rule(order)
    nodes = torch.tensor(nodes_np, device=torch_device, dtype=dtype)
    weights = torch.tensor(weights_np, device=torch_device, dtype=dtype)
    offsets = nodes * (pixel_scale / 2.0)
    x_centres = (
        torch.arange(width, device=torch_device, dtype=dtype) - (width - 1) / 2.0
    ) * pixel_scale
    y_centres = (
        torch.arange(height, device=torch_device, dtype=dtype) - (height - 1) / 2.0
    ) * pixel_scale
    phi = float(row["source_phi_rad"])
    cosine, sine = math.cos(phi), math.sin(phi)
    source_q = float(row["source_axis_ratio"])
    re_arcsec = float(row["source_re_arcsec"])
    n_sersic = float(row["source_n_sersic"])
    b_n = 2.0 * n_sersic - 1.0 / 3.0
    weighted_pixels: list[torch.Tensor] = []
    weight_2d = weights[:, None] * weights[None, :]
    with torch.no_grad():
        for start in range(0, height, native_row_chunk):
            stop = min(height, start + native_row_chunk)
            # Shape: (native rows, native columns, y nodes, x nodes).
            x = (
                x_centres[None, :, None, None]
                + offsets[None, None, None, :]
            ).expand(stop - start, width, order, order)
            y = (
                y_centres[start:stop, None, None, None]
                + offsets[None, None, :, None]
            ).expand(stop - start, width, order, order)
            if lensed:
                ax_lens, ay_lens = _sie_deflection(
                    x,
                    y,
                    theta_e=float(row["theta_e_arcsec"]),
                    q=float(row["lens_axis_ratio"]),
                    phi=float(row["lens_phi_rad"]),
                    x0=float(row["lens_x0_arcsec"]),
                    y0=float(row["lens_y0_arcsec"]),
                )
                shear_g1, shear_g2 = float(row["shear_g1"]), float(row["shear_g2"])
                beta_x = x - ax_lens - shear_g1 * x - shear_g2 * y
                beta_y = y - ay_lens - shear_g2 * x + shear_g1 * y
            else:
                beta_x, beta_y = x, y
            shifted_x = beta_x - float(row["source_beta_x_arcsec"])
            shifted_y = beta_y - float(row["source_beta_y_arcsec"])
            xr = cosine * shifted_x + sine * shifted_y
            yr = -sine * shifted_x + cosine * shifted_y
            elliptical_radius = torch.sqrt(
                xr.square() + yr.square() / (source_q**2 + 1.0e-8) + 1.0e-12
            )
            profile = torch.exp(
                -b_n
                * (
                    (elliptical_radius / (re_arcsec + 1.0e-8)).clamp_min(0.0)
                    ** (1.0 / n_sersic)
                    - 1.0
                )
            ).clamp_min(0.0)
            integrated = (
                (profile * weight_2d[None, None]).sum(dim=(-2, -1))
                * (pixel_scale**2 / 4.0)
            )
            weighted_pixels.append(integrated)
    return torch.cat(weighted_pixels, dim=0)


def render_source_expectation_at_order(
    row: Mapping[str, Any],
    shape_hwc: tuple[int, int, int],
    config: Mapping[str, Any],
    *,
    order: int,
    device: str = "cpu",
) -> tuple[np.ndarray, float]:
    """Render at a fixed quadrature order; used by adaptive and validation gates."""
    render = config["render"]
    integrated_lensed = _integrated_sersic_shape(
        row, shape_hwc, order=order, lensed=True, device=device
    )
    integrated_unlensed = _integrated_sersic_shape(
        row, shape_hwc, order=order, lensed=False, device=device
    )

    re_arcsec = float(row["source_re_arcsec"])
    n_sersic = float(row["source_n_sersic"])
    source_q = float(row["source_axis_ratio"])
    normalization = _sersic_integral(re_arcsec, n_sersic, source_q)
    captured_fraction = float(integrated_unlensed.sum().item() / normalization)
    flux_relative_error = abs(captured_fraction - 1.0)
    pixel_scale = float(row["pixel_scale_arcsec"])
    fwhm_r = float(row["psf_fwhm_r_arcsec"])
    band_fwhm = (
        fwhm_r * float(render["psf_fwhm_scale_g"]),
        fwhm_r,
        fwhm_r * float(render["psf_fwhm_scale_z"]),
    )
    band_flux = (
        float(row["source_flux_nmgy_g"]),
        float(row["source_flux_nmgy_r"]),
        float(row["source_flux_nmgy_z"]),
    )
    bands: list[torch.Tensor] = []
    for flux, fwhm in zip(band_flux, band_fwhm, strict=True):
        # The analytic source-plane integral sets surface brightness. There is no
        # image-plane renormalization: lensing magnification remains untouched.
        native_flux = integrated_lensed * (flux / normalization)
        bands.append(
            zero_padded_gaussian_convolution(
                native_flux,
                fwhm / (2.355 * pixel_scale),
                truncate_sigma=float(render["gaussian_truncate_sigma"]),
            )
        )
    output = torch.stack(bands).detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(output).all() or (output < -1.0e-7).any():
        raise ContractError("renderer produced invalid source expectation pixels")
    return np.maximum(output, np.float32(0.0)), flux_relative_error


def unlensed_flux_relative_error_at_order(
    row: Mapping[str, Any],
    shape_hwc: tuple[int, int, int],
    *,
    order: int,
    device: str = "cpu",
) -> float:
    integrated = _integrated_sersic_shape(
        row, shape_hwc, order=order, lensed=False, device=device
    )
    normalization = _sersic_integral(
        float(row["source_re_arcsec"]),
        float(row["source_n_sersic"]),
        float(row["source_axis_ratio"]),
    )
    return abs(float(integrated.sum().item() / normalization) - 1.0)


def adaptive_render_source_expectation(
    row: Mapping[str, Any],
    shape_hwc: tuple[int, int, int],
    config: Mapping[str, Any],
    *,
    device: str = "cpu",
    validate_scene: bool = True,
) -> AdaptiveRenderResult:
    """Select the first pixel quadrature order satisfying flux and image gates."""
    assert_config_integrity(config)
    if validate_scene:
        verify_scene_identity(row, config)
    render = config["render"]
    orders = tuple(int(value) for value in render["quadrature_orders"])
    if tuple(sorted(set(orders))) != orders or len(orders) < 2:
        raise ContractError("quadrature orders must be unique, increasing, and plural")
    flux_tolerance = float(render["unlensed_flux_relative_tolerance"])
    l1_tolerance = float(render["lensed_relative_l1_tolerance"])
    max_tolerance = float(render["lensed_relative_max_tolerance"])
    previous: np.ndarray | None = None
    history: list[dict[str, float | int]] = []
    for order in orders:
        image, flux_error = render_source_expectation_at_order(
            row, shape_hwc, config, order=order, device=device
        )
        if previous is None:
            relative_l1 = float("inf")
            relative_max = float("inf")
        else:
            difference = np.abs(image.astype(np.float64) - previous.astype(np.float64))
            relative_l1 = float(
                difference.sum() / max(np.abs(image.astype(np.float64)).sum(), 1.0e-30)
            )
            relative_max = float(
                difference.max() / max(np.abs(image.astype(np.float64)).max(), 1.0e-30)
            )
        history.append(
            {
                "order": order,
                "unlensed_flux_relative_error": flux_error,
                "previous_order_relative_l1": relative_l1,
                "previous_order_relative_max": relative_max,
            }
        )
        if (
            previous is not None
            and flux_error <= flux_tolerance
            and relative_l1 <= l1_tolerance
            and relative_max <= max_tolerance
        ):
            return AdaptiveRenderResult(
                image_chw=image,
                selected_order=order,
                unlensed_flux_relative_error=flux_error,
                previous_order_relative_l1=relative_l1,
                previous_order_relative_max=relative_max,
                order_history=tuple(history),
            )
        previous = image
    raise ContractError(
        "adaptive pixel integration failed preregistered flux/image convergence: "
        f"{history[-1]}"
    )


def render_source_expectation(
    row: Mapping[str, Any],
    shape_hwc: tuple[int, int, int],
    config: Mapping[str, Any],
    *,
    device: str = "cpu",
) -> np.ndarray:
    return adaptive_render_source_expectation(
        row, shape_hwc, config, device=device
    ).image_chw


def _load_host(row: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(row["host_pixel_path"]))
    if sha256_file(path) != str(row["host_payload_sha256"]):
        raise ContractError(f"host payload digest mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "cutout" not in archive.files:
            raise ContractError(f"host payload has no cutout array: {path}")
        host = np.array(archive["cutout"], copy=True)
    if host.dtype != np.float32 or host.ndim != 3 or host.shape[-1] != 3:
        raise ContractError(
            f"host must be finite float32 HWC with 3 bands, got {host.dtype} {host.shape}"
        )
    if not np.isfinite(host).all():
        raise ContractError(f"host has non-finite pixels: {path}")
    if array_bytes_sha256(host) != str(row["host_pixel_sha256"]):
        raise ContractError(f"host pixel bytes do not match manifest: {path}")
    return host


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def render_parent_bank(
    scenes: pd.DataFrame,
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    device: str = "cpu",
    limit: int | None = None,
    renderer_validation_receipt_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Render raw parent pixels. This module has no detector or scoring imports."""
    assert_config_integrity(config)
    if device != config["render"]["authoritative_device"]:
        raise ContractError("parent rendering must use the frozen render backend")
    if limit is None:
        if renderer_validation_receipt_path is None:
            raise ContractError("final parent rendering lacks renderer validation")
        validate_renderer_validation_receipt(
            renderer_validation_receipt_path, config
        )
    assert_score_blind_columns(list(scenes.columns))
    required = {
        "grec_row_id",
        "scene_id",
        "parameter_row_sha256",
        "host_galaxy_id",
        "host_pixel_sha256",
        "host_payload_sha256",
        "host_pixel_path",
    }
    missing = sorted(required - set(scenes.columns))
    if missing:
        raise ContractError(f"scene table lacks rendering fields: {missing}")
    table = scenes if limit is None else scenes.iloc[:limit]
    output = Path(output_dir)
    rows: list[dict[str, Any]] = []
    host_cache: dict[str, np.ndarray] = {}
    for row in table.to_dict(orient="records"):
        verify_scene_identity(row, config)
        host_hash = str(row["host_pixel_sha256"])
        host = host_cache.get(host_hash)
        if host is None:
            host = _load_host(row)
            host_cache[host_hash] = host
            stored_host = output / "raw_hosts" / host_hash[:2] / f"{host_hash}.npy"
            if not stored_host.exists():
                _atomic_save_npy(stored_host, host)
            restored = np.load(stored_host, allow_pickle=False)
            if (
                restored.dtype != host.dtype
                or restored.shape != host.shape
                or array_bytes_sha256(restored) != host_hash
            ):
                raise ContractError("raw-host store did not preserve exact element bytes")
        else:
            stored_host = output / "raw_hosts" / host_hash[:2] / f"{host_hash}.npy"

        adaptive = adaptive_render_source_expectation(
            row, host.shape, config, device=device
        )
        arc = adaptive.image_chw
        expectation = host.transpose(2, 0, 1) + arc
        expectation = np.asarray(expectation, dtype=np.float32)
        if not np.array_equal(
            expectation - arc, host.transpose(2, 0, 1)
        ):
            # Float addition/subtraction is not an exact ancestry proof; the raw host
            # store above is authoritative. Record this diagnostic, but do not alter.
            arithmetic_host_recovery_exact = False
        else:
            arithmetic_host_recovery_exact = True
        artifact = (
            output
            / "grec_parents"
            / str(row["scene_id"])[:2]
            / f"{row['scene_id']}.npz"
        )
        _atomic_save_npz(
            artifact,
            expectation_chw=expectation,
            injection_expectation_chw=arc,
        )
        rows.append(
            {
                **row,
                "grec_row_id": str(row["grec_row_id"]),
                "scene_id": str(row["scene_id"]),
                "parameter_row_sha256": str(row["parameter_row_sha256"]),
                "host_galaxy_id": str(row["host_galaxy_id"]),
                "host_parent_id": str(row["host_parent_id"]),
                "host_pixel_sha256": host_hash,
                "host_store_path": str(stored_host),
                "host_store_file_sha256": sha256_file(stored_host),
                "parent_pixel_path": str(artifact),
                "parent_artifact_sha256": sha256_file(artifact),
                "expectation_pixel_sha256": array_bytes_sha256(expectation),
                "injection_expectation_sha256": array_bytes_sha256(arc),
                "host_arithmetic_roundtrip_exact": arithmetic_host_recovery_exact,
                "added_source_photon_noise": False,
                "added_sky_noise": False,
                "quadrature_selected_order": adaptive.selected_order,
                "unlensed_flux_relative_error": (
                    adaptive.unlensed_flux_relative_error
                ),
                "lensed_previous_order_relative_l1": (
                    adaptive.previous_order_relative_l1
                ),
                "lensed_previous_order_relative_max": (
                    adaptive.previous_order_relative_max
                ),
            }
        )
    manifest = pd.DataFrame(rows)
    assert_score_blind_columns(list(manifest.columns))
    receipt = {
        "schema_version": 1,
        "status": (
            "design_only_grec_render_canary"
            if limit is not None
            else "grec_parent_bank_frozen_pre_score"
        ),
        "rows": len(manifest),
        "distinct_hosts": int(manifest["host_galaxy_id"].nunique()),
        "raw_host_element_bytes_preserved": True,
        "source_model": "single_sersic",
        "lens_model": "SIE_plus_external_shear",
        "added_source_photon_noise": False,
        "additional_sky_noise": False,
        "convolution_boundary": "explicit_zero_padding",
        "operation_order": config["render"]["operation_order"],
        "render_scientific_config_sha256": scientific_section_sha256(
            config, "source", "lens", "render", "host", "cells", "grec"
        ),
        "config": config_provenance(config),
        "quadrature_selected_order_counts": {
            str(key): int(value)
            for key, value in manifest["quadrature_selected_order"]
            .value_counts()
            .sort_index()
            .items()
        },
        "maximum_unlensed_flux_relative_error": float(
            manifest["unlensed_flux_relative_error"].max()
        ),
        "maximum_lensed_previous_order_relative_l1": float(
            manifest["lensed_previous_order_relative_l1"].max()
        ),
        "maximum_lensed_previous_order_relative_max": float(
            manifest["lensed_previous_order_relative_max"].max()
        ),
        "runtime": runtime_record(),
    }
    return manifest, receipt


def write_render_outputs(
    manifest: pd.DataFrame,
    receipt: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "grec_parent_manifest.parquet"
    temporary = output / "grec_parent_manifest.partial.parquet"
    manifest.to_parquet(temporary, index=False)
    temporary.replace(manifest_path)
    receipt["parent_manifest_sha256"] = sha256_file(manifest_path)
    receipt_path = output / "GREC_RENDER_RECEIPT.json"
    atomic_write_json(receipt_path, receipt)
    return manifest_path, receipt_path
