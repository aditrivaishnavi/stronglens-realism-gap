from __future__ import annotations

import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .backend import backend_profile
from .contracts import (
    ContractError,
    config_provenance,
    package_source_sha256,
    sha256_file,
    stable_id,
)
from .pnoise import verify_design_role_ledger
from .rendering import (
    _grid,
    _load_host,
    _sie_deflection,
    adaptive_render_source_expectation,
    render_source_expectation_at_order,
    unlensed_flux_relative_error_at_order,
    zero_padded_gaussian_convolution,
)
from .scene_table import bind_scene_identity
from .scoring import design_only_raw_logits


def _source_variant(
    original: Mapping[str, Any], re_arcsec: float, n_sersic: float, q: float
) -> dict[str, Any]:
    row = dict(original)
    row["source_re_arcsec"] = float(re_arcsec)
    row["source_n_sersic"] = float(n_sersic)
    row["source_axis_ratio"] = float(q)
    shared = {
        key: row[key]
        for key in row
        if key.startswith("source_")
        and key not in {"source_nuisance_id", "source_parameter_id"}
    }
    row["source_nuisance_id"] = stable_id("srcn", shared)
    row["source_parameter_id"] = stable_id(
        "src",
        {
            "nuisance": row["source_nuisance_id"],
            "theta_e_arcsec": row["theta_e_arcsec"],
            "beta_x_arcsec": row["source_beta_x_arcsec"],
            "beta_y_arcsec": row["source_beta_y_arcsec"],
        },
    )
    return bind_scene_identity(row)


def _sie_parity() -> float:
    x, y = _grid(
        31, 31, 0.08, device=torch.device("cpu"), dtype=torch.float32
    )
    theta_e, x0, y0, phi = 1.2, 0.03, -0.04, 0.71
    ax, ay = _sie_deflection(
        x, y, theta_e=theta_e, q=1.0, phi=phi, x0=x0, y0=y0
    )
    dx, dy = x - x0, y - y0
    radius = torch.sqrt(dx.square() + dy.square() + 1.0e-6)
    expected_x, expected_y = theta_e * dx / radius, theta_e * dy / radius
    return float(
        max(
            torch.max(torch.abs(ax - expected_x)).item(),
            torch.max(torch.abs(ay - expected_y)).item(),
        )
    )


def _psf_diagnostics(sigma: float, truncate: float) -> dict[str, float]:
    image = torch.zeros((101, 101), dtype=torch.float32)
    image[50, 50] = 1.0
    convolved = zero_padded_gaussian_convolution(
        image, sigma, truncate_sigma=truncate
    )
    yy = torch.arange(101, dtype=torch.float32) - 50
    xx = torch.arange(101, dtype=torch.float32) - 50
    y, x = torch.meshgrid(yy, xx, indexing="ij")
    total = convolved.sum()
    variance_x = (convolved * x.square()).sum() / total
    edge = torch.zeros((101, 101), dtype=torch.float32)
    edge[50, 0] = 1.0
    edge_blur = zero_padded_gaussian_convolution(
        edge, sigma, truncate_sigma=truncate
    )
    return {
        "central_kernel_sum_error": abs(float(total.item()) - 1.0),
        "relative_variance_error": abs(float(variance_x.item()) / sigma**2 - 1.0),
        "opposite_edge_wrap_value": float(edge_blur[50, -1].item()),
    }


def validate_renderer_design(
    scenes: pd.DataFrame,
    *,
    design_role_ledger_path: str | Path,
    design_role_receipt_path: str | Path,
    config: Mapping[str, Any],
    render_device: str,
    logit_device: str,
    authoritative_design: bool,
) -> dict[str, Any]:
    """Run pre-score physics and raw-logit gates using production functions."""
    if scenes.empty:
        raise ContractError("renderer validation requires design scenes")
    if authoritative_design and (
        render_device != config["render"]["authoritative_device"]
        or logit_device != config["scoring"]["authoritative_device"]
    ):
        raise ContractError("renderer validation used a non-authoritative backend")
    verify_design_role_ledger(
        scenes["scene_id"].astype(str),
        scenes["host_galaxy_id"].astype(str),
        design_role_ledger_path,
        design_role_receipt_path,
        config,
    )
    base = scenes.iloc[0].to_dict()
    render = config["render"]
    reference_order = int(render["logit_reference_order"])
    convergence_order = int(render["quadrature_orders"][-2])
    source = config["source"]
    axis_nodes = int(render["flux_validation_axis_nodes"])
    re_grid = np.linspace(
        source["re_arcsec_min"], source["re_arcsec_max"], axis_nodes
    )
    n_grid = np.linspace(source["n_min"], source["n_max"], axis_nodes)
    q_grid = np.linspace(
        source["axis_ratio_min"], source["axis_ratio_max"], axis_nodes
    )
    flux_errors = []
    flux_start = time.perf_counter()
    for re_arcsec in re_grid:
        for n_sersic in n_grid:
            for q in q_grid:
                variant = _source_variant(base, re_arcsec, n_sersic, q)
                flux_errors.append(
                    unlensed_flux_relative_error_at_order(
                        variant,
                        (101, 101, 3),
                        order=reference_order,
                        device=render_device,
                    )
                )
    rng = np.random.Generator(
        np.random.PCG64DXSM(int(render["flux_validation_seed"]))
    )
    for _ in range(int(render["flux_validation_random_samples"])):
        variant = _source_variant(
            base,
            rng.uniform(source["re_arcsec_min"], source["re_arcsec_max"]),
            rng.uniform(source["n_min"], source["n_max"]),
            rng.uniform(source["axis_ratio_min"], source["axis_ratio_max"]),
        )
        beta_fraction = math.sqrt(
            rng.uniform(
                source["beta_fraction_min"] ** 2,
                source["beta_fraction_max"] ** 2,
            )
        )
        theta_e = rng.uniform(
            min(config["grec"]["radii_arcsec"]),
            max(config["grec"]["radii_arcsec"]),
        )
        angle = rng.uniform(0.0, 2.0 * math.pi)
        variant["source_beta_x_arcsec"] = (
            beta_fraction * theta_e * math.cos(angle)
        )
        variant["source_beta_y_arcsec"] = (
            beta_fraction * theta_e * math.sin(angle)
        )
        variant["source_phi_rad"] = rng.uniform(0.0, 2.0 * math.pi)
        flux_errors.append(
            unlensed_flux_relative_error_at_order(
                variant,
                (101, 101, 3),
                order=reference_order,
                device=render_device,
            )
        )
    flux_seconds = time.perf_counter() - flux_start

    corner_values = [
        (re_arcsec, n_sersic, q)
        for re_arcsec in (source["re_arcsec_min"], source["re_arcsec_max"])
        for n_sersic in (source["n_min"], source["n_max"])
        for q in (source["axis_ratio_min"], source["axis_ratio_max"])
    ]
    templates = scenes.head(len(corner_values)).to_dict(orient="records")
    corner_variants = [
        _source_variant(
            templates[index % len(templates)], re_arcsec, n_sersic, q
        )
        for index, (re_arcsec, n_sersic, q) in enumerate(corner_values)
    ]
    adaptive_images = []
    reference_images = []
    image_l1 = []
    image_max = []
    selected_orders = []
    render_start = time.perf_counter()
    for variant in corner_variants:
        adaptive = adaptive_render_source_expectation(
            variant, (101, 101, 3), config, device=render_device
        )
        convergence_image, _ = render_source_expectation_at_order(
            variant,
            (101, 101, 3),
            config,
            order=convergence_order,
            device=render_device,
        )
        reference, _ = render_source_expectation_at_order(
            variant,
            (101, 101, 3),
            config,
            order=reference_order,
            device=render_device,
        )
        difference = np.abs(
            convergence_image.astype(np.float64) - reference.astype(np.float64)
        )
        image_l1.append(
            float(
                difference.sum()
                / max(np.abs(reference.astype(np.float64)).sum(), 1.0e-30)
            )
        )
        image_max.append(
            float(
                difference.max()
                / max(np.abs(reference.astype(np.float64)).max(), 1.0e-30)
            )
        )
        selected_orders.append(adaptive.selected_order)
        host = _load_host(variant).transpose(2, 0, 1)
        adaptive_images.append(host + convergence_image)
        reference_images.append(host + reference)
    render_seconds = time.perf_counter() - render_start
    logit_start = time.perf_counter()
    logits, logit_receipt = design_only_raw_logits(
        [*adaptive_images, *reference_images], config, device=logit_device
    )
    logit_seconds = time.perf_counter() - logit_start
    half = len(adaptive_images)
    logit_differences = np.abs(
        logits[:half].astype(np.float64) - logits[half:].astype(np.float64)
    )
    psf = _psf_diagnostics(
        1.2 / (2.355 * 0.262), float(render["gaussian_truncate_sigma"])
    )
    sie_error = _sie_parity()
    flux_pass = max(flux_errors) <= float(
        render["unlensed_flux_relative_tolerance"]
    )
    image_pass = max(image_l1) <= float(
        render["lensed_relative_l1_tolerance"]
    ) and max(image_max) <= float(render["lensed_relative_max_tolerance"])
    logit_pass = max(logit_differences) <= float(render["logit_absolute_tolerance"])
    psf_pass = (
        psf["central_kernel_sum_error"] <= 1.0e-6
        and psf["relative_variance_error"] <= 0.01
        and psf["opposite_edge_wrap_value"] == 0.0
    )
    receipt = {
        "schema_version": 1,
        "status": (
            render["required_validation_receipt_status"]
            if authoritative_design
            else "renderer_validation_canary_complete_not_authority"
        ),
        "design_only": True,
        "final_outcomes_used_false": True,
        "production_functions_used": True,
        "full_source_prior_flux_gate_passed": flux_pass,
        "lensed_image_convergence_gate_passed": image_pass,
        "logit_convergence_gate_passed": logit_pass,
        "sie_parity_gate_passed": sie_error <= 2.0e-6,
        "psf_gate_passed": psf_pass,
        "source_prior_grid_points": len(flux_errors),
        "maximum_unlensed_flux_relative_error": float(max(flux_errors)),
        "maximum_lensed_reference_relative_l1": float(max(image_l1)),
        "maximum_lensed_reference_relative_max": float(max(image_max)),
        "maximum_reference_logit_absolute_difference": float(
            max(logit_differences)
        ),
        "selected_quadrature_orders": selected_orders,
        "sie_sis_max_absolute_error": sie_error,
        "psf": psf,
        "benchmark": {
            "flux_grid_seconds": flux_seconds,
            "lensed_corner_seconds": render_seconds,
            "seconds_per_lensed_corner_with_reference": render_seconds
            / len(corner_variants),
            "logit_convergence_images": len(adaptive_images) + len(reference_images),
            "logit_convergence_seconds": logit_seconds,
        },
        "render_device": render_device,
        "backend_profile": backend_profile(config, logit_device),
        "scorer_closure_sha256": logit_receipt["scorer_closure_sha256"],
        "design_role_ledger_sha256": sha256_file(design_role_ledger_path),
        "render_implementation_source_sha256": package_source_sha256(
            "cells.py",
            "contracts.py",
            "rendering.py",
            "scene_table.py",
            "validation.py",
        ),
        "config": config_provenance(config),
    }
    failed = [
        field
        for field in (
            "full_source_prior_flux_gate_passed",
            "lensed_image_convergence_gate_passed",
            "logit_convergence_gate_passed",
            "sie_parity_gate_passed",
            "psf_gate_passed",
        )
        if not receipt[field]
    ]
    if failed:
        raise ContractError(f"renderer validation failed: {failed}; {receipt}")
    return receipt
