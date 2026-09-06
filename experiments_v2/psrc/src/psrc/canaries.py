from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import galsim
import numpy as np
import pandas as pd

from .inference import global_spatial_source_bootstrap
from .rendering import (
    Lens,
    coupled_photon_draw,
    gaussian_kernel,
    gaussian_psf_kernels,
    render_expectation,
    zero_padded_convolution,
)
from .scoring import score_design_only_canary
from .source import _align_nonnegative
from .threshold import threshold_evidence
from .utils import atomic_json, sha256_file


def _unit_gaussian(size: int, sigma: float) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    center = (size - 1) / 2.0
    image = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma**2))
    return image / image.sum()


def synthetic_deconvolution_canary() -> dict[str, float]:
    pixel_scale = 0.03
    size = 161
    hst = galsim.Gaussian(fwhm=0.10)
    reconstruction = galsim.Gaussian(fwhm=0.15)
    truth = (
        galsim.Exponential(half_light_radius=0.28, flux=0.75)
        + galsim.Gaussian(sigma=0.035, flux=0.15).shift(0.13, -0.05)
        + galsim.Gaussian(sigma=0.045, flux=0.10).shift(-0.11, 0.09)
    )

    def draw(profile: galsim.GSObject) -> np.ndarray:
        return np.asarray(
            profile.drawImage(
                nx=size,
                ny=size,
                scale=pixel_scale,
                method="no_pixel",
            ).array,
            dtype=np.float64,
        )

    observed = draw(galsim.Convolve(truth, hst))
    noisy = observed + np.random.default_rng(20260916).normal(
        0.0, observed.max() / 150.0, observed.shape
    )
    empirical = galsim.InterpolatedImage(
        galsim.Image(noisy, scale=pixel_scale),
        normalization="flux",
        pad_factor=4,
    )
    reconstructed_raw = draw(
        galsim.Convolve(empirical, galsim.Deconvolve(hst), reconstruction)
    )
    target_raw = draw(galsim.Convolve(truth, reconstruction))
    y, x = np.mgrid[:size, :size]
    support = np.hypot(x - (size - 1) / 2, y - (size - 1) / 2) * pixel_scale <= 1.2
    reconstructed, reconstructed_diagnostic = _align_nonnegative(
        reconstructed_raw,
        support=support,
        pixel_scale=pixel_scale,
        tolerance_arcsec=3e-4,
    )
    target, _ = _align_nonnegative(
        target_raw,
        support=support,
        pixel_scale=pixel_scale,
        tolerance_arcsec=3e-4,
    )
    relative_l1 = float(
        np.sum(np.abs(reconstructed - target), dtype=np.float64)
        / np.sum(np.abs(target), dtype=np.float64)
    )
    if relative_l1 >= 0.08:
        raise AssertionError("synthetic regularized deconvolution recovery failed")
    return {
        "input_peak_to_pixel_noise": 150.0,
        "relative_l1_to_known_regularized_truth": relative_l1,
        "preprojection_negative_to_positive_flux": reconstructed_diagnostic[
            "preprojection_negative_to_positive_flux"
        ],
    }


def renderer_canary(config: dict[str, Any]) -> dict[str, float]:
    render = config["rendering"]
    pixel_scale = 0.262
    kernel = gaussian_kernel(
        1.2,
        pixel_scale,
        truncate_sigma=float(render["psf"]["truncate_sigma"]),
    )
    central_delta = np.zeros((101, 101), dtype=np.float64)
    central_delta[50, 50] = 1.0
    central = zero_padded_convolution(central_delta, kernel)
    if abs(central.sum(dtype=np.float64) - 1.0) > 1e-12:
        raise AssertionError("central delta PSF flux is not conserved")
    corner_delta = np.zeros((101, 101), dtype=np.float64)
    corner_delta[0, 0] = 1.0
    corner = zero_padded_convolution(corner_delta, kernel)
    if abs(corner[-1, -1]) > 1e-14:
        raise AssertionError("zero-padded PSF wrapped to opposite corner")

    source = _unit_gaussian(101, 7.0)
    lens = Lens(1.2, 0.72, 0.31, 0.02, -0.01, 0.01, -0.02)
    images: dict[int, np.ndarray] = {}
    for oversample in render["convergence_oversamples"]:
        kernels = gaussian_psf_kernels(
            fwhm_grz_arcsec=(1.05, 1.0, 0.94),
            target_pixel_scale_arcsec=pixel_scale,
            oversample=int(oversample),
            convention=render["psf"]["convention"],
            truncate_sigma=float(render["psf"]["truncate_sigma"]),
        )
        images[int(oversample)] = render_expectation(
            source,
            source_pixel_scale_arcsec=0.03,
            intrinsic_flux_nmgy_grz=(1.0, 2.0, 1.5),
            lens=lens,
            source_center_x_arcsec=0.22,
            source_center_y_arcsec=-0.10,
            target_shape=(101, 101),
            target_pixel_scale_arcsec=pixel_scale,
            psf_kernels_grz=kernels,
            psf_convention=render["psf"]["convention"],
            oversample=int(oversample),
        )
    relative_4_8 = float(
        np.sum(np.abs(images[4] - images[8])) / np.sum(np.abs(images[8]))
    )
    relative_8_16 = float(
        np.sum(np.abs(images[8] - images[16])) / np.sum(np.abs(images[16]))
    )
    if relative_8_16 >= float(render["convergence_relative_l1_max"]):
        raise AssertionError("renderer did not converge by 8x")
    return {
        "central_delta_psf_flux": float(central.sum(dtype=np.float64)),
        "opposite_corner_wrap_value": float(corner[-1, -1]),
        "relative_l1_4x_to_8x": relative_4_8,
        "relative_l1_8x_to_16x": relative_8_16,
    }


def photon_canary() -> dict[str, float | bool]:
    a = np.full((2, 2, 3), 0.2)
    b = np.full((2, 2, 3), 0.35)
    gain = 100.0
    draws_a = []
    draws_b = []
    forward = {}
    for draw in range(4000):
        sampled_a, sampled_b, _ = coupled_photon_draw(
            a,
            b,
            gain_e_per_nmgy=gain,
            noise_key_base="design-canary",
            draw_index=draw,
        )
        draws_a.append(sampled_a[0, 0, 0])
        draws_b.append(sampled_b[0, 0, 0])
        if draw < 16:
            forward[draw] = (sampled_a, sampled_b)
    reverse = {
        draw: coupled_photon_draw(
            a,
            b,
            gain_e_per_nmgy=gain,
            noise_key_base="design-canary",
            draw_index=draw,
        )[:2]
        for draw in reversed(range(16))
    }
    worker_invariant = all(
        np.array_equal(forward[index][0], reverse[index][0])
        and np.array_equal(forward[index][1], reverse[index][1])
        for index in range(16)
    )
    if not worker_invariant:
        raise AssertionError("photon keys changed under worker/order permutation")
    same_a, same_b, _ = coupled_photon_draw(
        a,
        a,
        gain_e_per_nmgy=gain,
        noise_key_base="equal-means",
        draw_index=0,
    )
    if not np.array_equal(same_a, same_b):
        raise AssertionError("equal means did not receive identical counts")
    mean_a = float(np.mean(draws_a))
    mean_b = float(np.mean(draws_b))
    variance_a = float(np.var(draws_a, ddof=1))
    variance_b = float(np.var(draws_b, ddof=1))
    expected_variance_a = 0.2 / gain
    expected_variance_b = 0.35 / gain
    variance_relative_error = max(
        abs(variance_a - expected_variance_a) / expected_variance_a,
        abs(variance_b - expected_variance_b) / expected_variance_b,
    )
    if abs(mean_a - 0.2) > 5 * np.sqrt(0.2 / gain / len(draws_a)) or abs(
        mean_b - 0.35
    ) > 5 * np.sqrt(0.35 / gain / len(draws_b)):
        raise AssertionError("photon marginal mean failed")
    if variance_relative_error > 0.08:
        raise AssertionError("photon marginal variance failed")
    return {
        "structured_sample_mean_nmgy": mean_a,
        "sersic_sample_mean_nmgy": mean_b,
        "paired_draw_correlation": float(np.corrcoef(draws_a, draws_b)[0, 1]),
        "structured_sample_variance_nmgy2": variance_a,
        "sersic_sample_variance_nmgy2": variance_b,
        "maximum_variance_relative_error": variance_relative_error,
        "worker_order_invariant": worker_invariant,
    }


def threshold_canary() -> dict[str, Any]:
    values = np.linspace(-10, 10, 20_000, dtype=np.float32)
    values[-3:] = np.float32(9.5)
    evidence, threshold = threshold_evidence(values, 0.001)
    return {
        **evidence,
        "recomputed_threshold_float32_bits": int(threshold.view(np.uint32)),
    }


def _coverage_frame(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    group_effect = rng.normal(0, 0.45, 88)
    source_effect = rng.normal(0, 0.35, 144)
    for cell in range(9):
        for radius_index in range(11):
            block = cell * 11 + radius_index
            for slot in range(16):
                stratum = (slot + block) % 16
                source = stratum * 9 + (block % 9)
                group = (radius_index * 16 + slot) % 88
                linear = (
                    -1.4
                    + group_effect[group]
                    + source_effect[source]
                    + 0.04 * (cell - 4)
                    + 0.03 * (radius_index - 5)
                )
                probability = 1.0 / (1.0 + np.exp(-linear))
                structured = rng.binomial(8, probability) / 8.0
                sersic = rng.binomial(8, probability) / 8.0
                rows.append(
                    {
                        "host_cell_id": f"c{cell}",
                        "host_spatial_group_id": f"g{group}",
                        "source_id": f"s{source}",
                        "source_stratum": f"q{stratum:02d}",
                        "source_analysis_weight": 1.0,
                        "paired_detection_risk_difference": (sersic - structured),
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_coverage_canary(config: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(20260916)
    replications = int(config["canaries"]["bootstrap_coverage_replications"])
    bootstrap_draws = int(
        config["canaries"]["bootstrap_draws_per_coverage_replication"]
    )
    covered = 0
    assignment = None
    for replication in range(replications):
        frame = _coverage_frame(rng)
        if assignment is None:
            frame = frame.copy()
            frame["block_id"] = np.repeat(np.arange(99), 16)
            block_strata = frame.groupby("block_id")["source_stratum"].nunique()
            source_reuse = frame.groupby("source_id").size()
            assignment = {
                "blocks": int(frame["block_id"].nunique()),
                "rows_per_block": int(frame.groupby("block_id").size().min()),
                "source_strata_per_block_minimum": int(block_strata.min()),
                "source_strata_per_block_maximum": int(block_strata.max()),
                "distinct_sources": int(frame["source_id"].nunique()),
                "source_reuse_minimum": int(source_reuse.min()),
                "source_reuse_maximum": int(source_reuse.max()),
                "parent_scene_weight": 1.0,
            }
            frame = frame.drop(columns="block_id")
        estimates, _ = global_spatial_source_bootstrap(
            frame,
            draws=bootstrap_draws,
            seed_namespace=f"coverage-{replication}",
        )
        interval = np.quantile(estimates, [0.025, 0.975])
        covered += int(interval[0] <= 0.0 <= interval[1])
    coverage = covered / replications
    return {
        "replications": float(replications),
        "bootstrap_draws_per_replication": float(bootstrap_draws),
        "coverage": coverage,
        "pairs_per_replication": 1584.0,
        "binary_photon_draws_per_pair": 8.0,
        "interval_inflation": 1.0,
        "role": "conservative_diagnostic_only_never_a_pass_gate",
        "method_limit": config["inference"]["method_limitation"],
        "exact_assignment": assignment,
    }


def actual_photon_scoring_canary(
    base_images_nchw: np.ndarray,
    config: dict[str, Any],
    *,
    scorer_receipt_path: str | Path,
    device: str,
) -> dict[str, Any]:
    """Use real scorer code on design-only photonized images; no threshold."""
    images = np.asarray(base_images_nchw)
    if images.dtype != np.float32 or images.ndim != 4:
        raise ValueError("actual scoring canary needs native float32 NCHW")
    logits, receipt = score_design_only_canary(
        images,
        config,
        scorer_receipt_path=scorer_receipt_path,
        device=device,
    )
    return {
        "status": "actual_design_only_photon_raw_logit_canary",
        "images": len(images),
        "input_array_sha256": hashlib.sha256(
            np.ascontiguousarray(images).view(np.uint8).tobytes()
        ).hexdigest(),
        "logit_bits_sha256": hashlib.sha256(
            logits.view(np.uint32).tobytes()
        ).hexdigest(),
        "threshold_applied": False,
        "final_outcomes_used": False,
        "scoring": receipt,
    }


def write_photonized_scoring_canary_images(
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Build synthetic, photonized NCHW images for the actual fixed scorer."""
    if output.exists():
        raise FileExistsError(output)
    render = config["rendering"]
    base_structured = _unit_gaussian(101, 7.0)
    y, x = np.mgrid[:101, :101]
    base_structured += 0.2 * np.exp(-((x - 58) ** 2 + (y - 43) ** 2) / (2 * 2.0**2))
    base_sersic = _unit_gaussian(101, 8.5)
    images = []
    labels = []
    for scene in range(8):
        lens = Lens(
            0.75 + 0.25 * (scene % 4),
            0.65 + 0.05 * (scene % 4),
            0.2 * scene,
            0.01,
            -0.02,
            0.0,
            0.0,
        )
        kernels = gaussian_psf_kernels(
            fwhm_grz_arcsec=(1.15, 1.1, 1.03),
            target_pixel_scale_arcsec=0.262,
            oversample=8,
            convention=render["psf"]["convention"],
            truncate_sigma=float(render["psf"]["truncate_sigma"]),
        )
        common = {
            "source_pixel_scale_arcsec": 0.03,
            "intrinsic_flux_nmgy_grz": (0.2, 0.4, 0.8),
            "lens": lens,
            "source_center_x_arcsec": 0.1,
            "source_center_y_arcsec": -0.05,
            "target_shape": (101, 101),
            "target_pixel_scale_arcsec": 0.262,
            "psf_kernels_grz": kernels,
            "psf_convention": render["psf"]["convention"],
            "oversample": 8,
        }
        structured = render_expectation(base_structured, **common)
        sersic = render_expectation(base_sersic, **common)
        host = np.random.default_rng(1000 + scene).normal(0.0, 0.01, structured.shape)
        for draw in range(4):
            structured_arc, sersic_arc, _ = coupled_photon_draw(
                structured,
                sersic,
                gain_e_per_nmgy=150.0,
                noise_key_base=f"scoring-canary-{scene}",
                draw_index=draw,
            )
            images.extend(
                [
                    np.transpose(host + structured_arc, (2, 0, 1)),
                    np.transpose(host + sersic_arc, (2, 0, 1)),
                ]
            )
            labels.extend(
                [
                    f"scene-{scene}-draw-{draw}-structured",
                    f"scene-{scene}-draw-{draw}-sersic",
                ]
            )
    array = np.asarray(images, dtype=np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        images_nchw=array,
        design_labels=np.asarray(labels),
    )
    receipt = {
        "schema_version": 1,
        "status": "synthetic_photonized_scoring_canary_images",
        "images": len(array),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "artifact_sha256": sha256_file(output),
        "final_outcomes_used": False,
    }
    atomic_json(output.with_suffix(".receipt.json"), receipt)
    return receipt


def run_canaries(
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    result = {
        "schema_version": 2,
        "status": "pass",
        "detector_outcomes_used": False,
        "synthetic_deconvolution": synthetic_deconvolution_canary(),
        "renderer": renderer_canary(config),
        "photon_noise": photon_canary(),
        "threshold": threshold_canary(),
        "inference_coverage": bootstrap_coverage_canary(config),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return result
