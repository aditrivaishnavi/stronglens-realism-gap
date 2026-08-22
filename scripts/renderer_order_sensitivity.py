#!/usr/bin/env python3
"""Measure sensitivity to PSF-convolution and pixel-integration order.

The retained renderer evaluates the lensed source on an oversampled grid,
block-integrates to native pixels, and applies a circular Gaussian blur on the
native grid. This script compares that executed order with a reference order
that applies the Gaussian on the oversampled grid before native-pixel
integration.

The diagnostic is synthetic and detector-free. It reads no observed cutouts,
model scores, validation rows, or test rows.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dhs.injection_engine import (
    _fft_gaussian_blur,
    _sersic_shape,
    _sersic_source_integral,
    _shear_deflection,
    _sie_deflection,
    _torch_meshgrid_arcsec,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "renderer_order_sensitivity_v1.json"
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "paired_sources"
    / "renderer_order_sensitivity.json"
)


@dataclass(frozen=True)
class RenderCase:
    """One deterministic lens/source/seeing configuration."""

    case_id: str
    theta_e_arcsec: float
    source_re_arcsec: float
    source_n_sersic: float
    source_q: float
    source_phi_rad: float
    beta_fraction: float
    beta_angle_rad: float
    lens_q: float
    lens_phi_rad: float
    lens_x0_arcsec: float
    lens_y0_arcsec: float
    shear_g1: float
    shear_g2: float
    psf_fwhm_arcsec: float


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the diagnostic configuration."""
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["oversample"] < 2:
        raise ValueError("oversample must be at least 2")
    if config["stamp_pixels"] % 2 != 1:
        raise ValueError("stamp_pixels must be odd")
    expected = config["n_corner_cases"] + config["n_random_cases"]
    if expected <= 0:
        raise ValueError("at least one case is required")
    return config


def _range(config: dict[str, Any], name: str) -> tuple[float, float]:
    values = config["ranges"][name]
    return float(values[0]), float(values[1])


def build_cases(config: dict[str, Any]) -> list[RenderCase]:
    """Construct declared corner cases followed by seeded random cases."""
    ranges = config["ranges"]
    seed = int(config["seed"])
    cases: list[RenderCase] = []

    corner_axes = [
        _range(config, "theta_e_arcsec"),
        _range(config, "source_re_arcsec"),
        _range(config, "source_n_sersic"),
        _range(config, "source_q"),
        _range(config, "lens_q"),
        _range(config, "psf_fwhm_arcsec"),
    ]
    corners = list(itertools.product(*corner_axes))
    if len(corners) != int(config["n_corner_cases"]):
        raise ValueError(
            f"declared {config['n_corner_cases']} corner cases, built {len(corners)}"
        )

    beta_lo, beta_hi = _range(config, "source_beta_fraction")
    for index, values in enumerate(corners):
        theta_e, source_re, source_n, source_q, lens_q, fwhm = values
        beta_fraction = beta_lo if index % 2 == 0 else beta_hi
        cases.append(
            RenderCase(
                case_id=f"corner_{index:03d}",
                theta_e_arcsec=theta_e,
                source_re_arcsec=source_re,
                source_n_sersic=source_n,
                source_q=source_q,
                source_phi_rad=(index % 7) * math.pi / 7.0,
                beta_fraction=beta_fraction,
                beta_angle_rad=(index % 11) * 2.0 * math.pi / 11.0,
                lens_q=lens_q,
                lens_phi_rad=(index % 13) * math.pi / 13.0,
                lens_x0_arcsec=0.0,
                lens_y0_arcsec=0.0,
                shear_g1=0.0,
                shear_g2=0.0,
                psf_fwhm_arcsec=fwhm,
            )
        )

    rng = np.random.default_rng(seed)
    theta_lo, theta_hi = _range(config, "theta_e_arcsec")
    re_lo, re_hi = _range(config, "source_re_arcsec")
    n_lo, n_hi = _range(config, "source_n_sersic")
    source_q_lo, source_q_hi = _range(config, "source_q")
    lens_q_lo, lens_q_hi = _range(config, "lens_q")
    fwhm_lo, fwhm_hi = _range(config, "psf_fwhm_arcsec")
    center_sigma = float(ranges["lens_center_sigma_arcsec"])
    shear_sigma = float(ranges["shear_component_sigma"])

    for index in range(int(config["n_random_cases"])):
        cases.append(
            RenderCase(
                case_id=f"random_{index:03d}",
                theta_e_arcsec=float(rng.uniform(theta_lo, theta_hi)),
                source_re_arcsec=float(rng.uniform(re_lo, re_hi)),
                source_n_sersic=float(rng.uniform(n_lo, n_hi)),
                source_q=float(rng.uniform(source_q_lo, source_q_hi)),
                source_phi_rad=float(rng.uniform(0.0, 2.0 * math.pi)),
                beta_fraction=float(rng.uniform(beta_lo, beta_hi)),
                beta_angle_rad=float(rng.uniform(0.0, 2.0 * math.pi)),
                lens_q=float(rng.uniform(lens_q_lo, lens_q_hi)),
                lens_phi_rad=float(rng.uniform(0.0, math.pi)),
                lens_x0_arcsec=float(rng.normal(0.0, center_sigma)),
                lens_y0_arcsec=float(rng.normal(0.0, center_sigma)),
                shear_g1=float(rng.normal(0.0, shear_sigma)),
                shear_g2=float(rng.normal(0.0, shear_sigma)),
                psf_fwhm_arcsec=float(rng.uniform(fwhm_lo, fwhm_hi)),
            )
        )
    return cases


def _block_average(image: torch.Tensor, native_pixels: int, sub: int) -> torch.Tensor:
    """Average a square oversampled image into native pixels."""
    return image.view(native_pixels, sub, native_pixels, sub).mean(dim=(1, 3))


def render_orders(
    case: RenderCase,
    *,
    native_pixels: int,
    pixel_scale_arcsec: float,
    oversample: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return executed-order and reference-order noiseless r-band maps."""
    dtype = torch.float64
    device = torch.device("cpu")
    sub_scale = pixel_scale_arcsec / oversample
    high_pixels = native_pixels * oversample
    x, y = _torch_meshgrid_arcsec(
        high_pixels,
        high_pixels,
        sub_scale,
        device,
        dtype,
    )

    theta_e = torch.tensor(case.theta_e_arcsec, dtype=dtype)
    lens_q = torch.tensor(case.lens_q, dtype=dtype)
    lens_phi = torch.tensor(case.lens_phi_rad, dtype=dtype)
    lens_x0 = torch.tensor(case.lens_x0_arcsec, dtype=dtype)
    lens_y0 = torch.tensor(case.lens_y0_arcsec, dtype=dtype)
    alpha_x, alpha_y = _sie_deflection(
        x,
        y,
        theta_e,
        lens_q,
        lens_phi,
        lens_x0,
        lens_y0,
    )
    shear_x, shear_y = _shear_deflection(
        x,
        y,
        torch.tensor(case.shear_g1, dtype=dtype),
        torch.tensor(case.shear_g2, dtype=dtype),
    )
    source_x = x - alpha_x - shear_x
    source_y = y - alpha_y - shear_y

    beta_radius = case.beta_fraction * case.theta_e_arcsec
    beta_x = beta_radius * math.cos(case.beta_angle_rad)
    beta_y = beta_radius * math.sin(case.beta_angle_rad)
    source_re = torch.tensor(case.source_re_arcsec, dtype=dtype)
    source_n = torch.tensor(case.source_n_sersic, dtype=dtype)
    source_q = torch.tensor(case.source_q, dtype=dtype)
    source_phi = torch.tensor(case.source_phi_rad, dtype=dtype)
    high_shape = _sersic_shape(
        source_x,
        source_y,
        torch.tensor(beta_x, dtype=dtype),
        torch.tensor(beta_y, dtype=dtype),
        source_re,
        source_n,
        source_q,
        source_phi,
    ).clamp(min=0.0)

    source_integral = _sersic_source_integral(source_re, source_n, source_q)
    high_sb = high_shape / source_integral
    native_pixel_area = pixel_scale_arcsec**2

    native_unblurred = _block_average(high_sb, native_pixels, oversample)
    native_flux = native_unblurred * native_pixel_area
    native_sigma = torch.tensor(
        [case.psf_fwhm_arcsec / (2.355 * pixel_scale_arcsec)],
        dtype=dtype,
    )
    executed = _fft_gaussian_blur(native_flux[None, :, :], native_sigma)[0]

    high_sigma = torch.tensor(
        [case.psf_fwhm_arcsec / (2.355 * sub_scale)],
        dtype=dtype,
    )
    high_blurred_sb = _fft_gaussian_blur(high_sb[None, :, :], high_sigma)[0]
    reference = (
        _block_average(high_blurred_sb, native_pixels, oversample) * native_pixel_area
    )
    return executed, reference


def comparison_metrics(
    executed: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float]:
    """Calculate declared relative map-difference metrics."""
    difference = executed - reference
    reference_l1 = torch.sum(torch.abs(reference))
    reference_l2 = torch.linalg.vector_norm(reference)
    reference_peak = torch.max(torch.abs(reference))
    reference_flux = torch.sum(reference)
    executed_flux = torch.sum(executed)
    correlation = torch.corrcoef(
        torch.stack((executed.flatten(), reference.flatten()))
    )[0, 1]
    return {
        "relative_l1": float(torch.sum(torch.abs(difference)) / reference_l1),
        "relative_l2": float(torch.linalg.vector_norm(difference) / reference_l2),
        "relative_peak": float(
            torch.abs(torch.max(executed) - torch.max(reference)) / reference_peak
        ),
        "pixel_correlation": float(correlation),
        "relative_total_flux": float(
            torch.abs(executed_flux - reference_flux) / torch.abs(reference_flux)
        ),
    }


def summarize(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    """Summarize one metric over all cases."""
    values = np.asarray([row["metrics"][metric] for row in rows], dtype=float)
    return {
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
        "minimum": float(np.min(values)),
    }


def run(config_path: Path, output_path: Path) -> dict[str, Any]:
    """Execute the detector-free sensitivity and write its result."""
    config = load_config(config_path)
    cases = build_cases(config)
    rows: list[dict[str, Any]] = []
    for case in cases:
        executed, reference = render_orders(
            case,
            native_pixels=int(config["stamp_pixels"]),
            pixel_scale_arcsec=float(config["pixel_scale_arcsec"]),
            oversample=int(config["oversample"]),
        )
        rows.append(
            {
                "case": asdict(case),
                "metrics": comparison_metrics(executed, reference),
            }
        )

    metric_summary = {
        metric: summarize(rows, metric)
        for metric in (
            "relative_l1",
            "relative_l2",
            "relative_peak",
            "relative_total_flux",
        )
    }
    correlation_values = np.asarray(
        [row["metrics"]["pixel_correlation"] for row in rows],
        dtype=float,
    )
    metric_summary["pixel_correlation"] = {
        "median": float(np.median(correlation_values)),
        "p05": float(np.quantile(correlation_values, 0.05)),
        "minimum": float(np.min(correlation_values)),
        "maximum": float(np.max(correlation_values)),
    }

    result = {
        "schema_version": 1,
        "status": "complete",
        "scope": config["scope"],
        "data_access": {
            "observed_cutouts_read": 0,
            "validation_rows_read": 0,
            "test_rows_read": 0,
            "detector_scores_computed": 0,
        },
        "orders": {
            "executed": (
                "oversampled ray tracing and source evaluation; block integration "
                "to native pixels; native-grid circular-Gaussian blur"
            ),
            "reference": (
                "oversampled ray tracing and source evaluation; oversampled-grid "
                "circular-Gaussian blur; block integration to native pixels"
            ),
        },
        "cases": {
            "total": len(cases),
            "corner": int(config["n_corner_cases"]),
            "random": int(config["n_random_cases"]),
        },
        "summary": metric_summary,
        "per_case": rows,
        "provenance": {
            "config_path": str(config_path.relative_to(ROOT)),
            "config_sha256": sha256(config_path),
            "script_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "script_sha256": sha256(Path(__file__).resolve()),
            "injection_engine_path": "dhs/injection_engine.py",
            "injection_engine_sha256": sha256(ROOT / "dhs" / "injection_engine.py"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "dtype": "float64",
            "device": "cpu",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.config.resolve(), args.output.resolve())
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
