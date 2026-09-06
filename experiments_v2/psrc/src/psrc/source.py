from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import galsim
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import ndimage

from .contracts import (
    ContractError,
    assert_config_integrity,
    config_provenance,
    source_closure,
)
from .utils import atomic_json, md5_file, sha256_file, stable_hex, stable_seed


def _between(values: np.ndarray, bounds: list[float]) -> np.ndarray:
    return (values >= float(bounds[0])) & (values <= float(bounds[1]))


def _array_bytes_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def source_eligibility_ledger(config: dict[str, Any]) -> pd.DataFrame:
    """Build a complete score-blind COSMOS eligibility and support ledger."""
    assert_config_integrity(config)
    root = Path(config["inputs"]["cosmos_dir"])
    archive = Path(config["inputs"]["cosmos_archive"])
    expected_md5 = str(config["inputs"]["cosmos_archive_md5"])
    if md5_file(archive) != expected_md5:
        raise ValueError("COSMOS archive MD5 differs from the contract")

    fit = fits.getdata(root / "real_galaxy_catalog_25.2_fits.fits", 1)
    selection = fits.getdata(root / "real_galaxy_catalog_25.2_selection.fits", 1)
    catalogue = galsim.COSMOSCatalog(
        sample=str(config["selection"]["source"]["sample"]),
        dir=str(root),
    )
    original = np.asarray(catalogue.orig_index, dtype=np.int64)
    sersic = np.asarray(fit["SERSICFIT"], dtype=np.float64)[original]
    status = np.asarray(fit["FIT_STATUS"])[original, 4].astype(np.int64)
    viable = np.asarray(fit["VIABLE_SERSIC"])[original].astype(bool)
    magnitude = np.asarray(fit["MAG_AUTO"], dtype=np.float64)[original]
    fit_mad = np.asarray(fit["FIT_MAD_S"], dtype=np.float64)[original]
    sn = np.asarray(selection["SN_ELLIP_GAUSS"], dtype=np.float64)[original]
    mask_distance = np.asarray(selection["MIN_MASK_DIST_PIXELS"], dtype=np.float64)[
        original
    ]
    hlr = 0.03 * np.sqrt(np.maximum(sersic[:, 3], 0.0)) * sersic[:, 1]
    rules = config["selection"]["source"]

    frame = pd.DataFrame(
        {
            "cosmos_catalog_index": np.arange(len(original), dtype=np.int64),
            "cosmos_original_index": original,
            "cosmos_ident": np.asarray(fit["IDENT"])[original].astype(str),
            "f814w_mag_auto": magnitude,
            "sn_ellip_gauss": sn,
            "min_mask_distance_pixels": mask_distance,
            "single_sersic_fit_mad": fit_mad,
            "sersic_n": sersic[:, 2],
            "sersic_q": sersic[:, 3],
            "sersic_phi_catalog_rad": sersic[:, 7],
            "sersic_circularized_hlr_arcsec": hlr,
            "fit_status_sersic": status,
            "viable_sersic": viable,
            "galsim_use_real": np.asarray(catalogue.use_real, dtype=bool),
        }
    )
    finite = np.isfinite(
        frame[
            [
                "f814w_mag_auto",
                "sn_ellip_gauss",
                "min_mask_distance_pixels",
                "single_sersic_fit_mad",
                "sersic_n",
                "sersic_q",
                "sersic_phi_catalog_rad",
                "sersic_circularized_hlr_arcsec",
            ]
        ].to_numpy(dtype=np.float64)
    ).all(axis=1)
    frame["eligibility_reason"] = "eligible_pending_support"
    checks: list[tuple[np.ndarray, str]] = [
        (finite, "nonfinite_catalogue_value"),
        (frame["galsim_use_real"].to_numpy(), "not_in_galsim_real_sample"),
        (
            (~rules["viable_single_sersic_required"])
            | frame["viable_sersic"].to_numpy(),
            "no_viable_single_sersic_twin",
        ),
        (
            ~frame["fit_status_sersic"]
            .isin([int(value) for value in rules["forbidden_fit_status"]])
            .to_numpy(),
            "forbidden_single_sersic_fit_status",
        ),
        (
            frame["sn_ellip_gauss"].to_numpy() >= rules["sn_ellip_gauss_min"],
            "below_high_sn_threshold",
        ),
        (
            frame["min_mask_distance_pixels"].to_numpy()
            >= rules["min_mask_distance_pixels"],
            "mask_too_near_source",
        ),
        (
            _between(frame["f814w_mag_auto"].to_numpy(), rules["magnitude_auto"]),
            "outside_magnitude_support",
        ),
        (
            _between(
                frame["sersic_circularized_hlr_arcsec"].to_numpy(),
                rules["circularized_hlr_arcsec"],
            ),
            "outside_half_light_radius_support",
        ),
        (
            _between(frame["sersic_n"].to_numpy(), rules["sersic_n"]),
            "outside_sersic_index_support",
        ),
        (
            _between(frame["sersic_q"].to_numpy(), rules["axis_ratio"]),
            "outside_axis_ratio_support",
        ),
    ]
    still_eligible = np.ones(len(frame), dtype=bool)
    for passed, reason in checks:
        failed = still_eligible & ~np.asarray(passed, dtype=bool)
        frame.loc[failed, "eligibility_reason"] = reason
        still_eligible &= np.asarray(passed, dtype=bool)

    frame["native_height_pixels"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["native_width_pixels"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["native_pixel_scale_arcsec"] = np.nan
    frame["original_psf_fwhm_arcsec"] = np.nan
    frame["source_image_sha256"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame["source_psf_image_sha256"] = pd.Series(
        pd.NA, index=frame.index, dtype="string"
    )
    frame["common_support_radius_arcsec"] = np.nan
    frame["support_hlr"] = np.nan
    pixel_scale = float(config["source_construction"]["source_pixel_scale_arcsec"])
    support_fraction = float(
        config["source_construction"][
            "native_inscribed_radius_fraction_for_eligibility"
        ]
    )
    minimum_support = float(rules["minimum_native_inscribed_support_hlr"])
    reconstruction_fwhm = float(
        config["source_construction"]["reconstruction_psf_fwhm_arcsec"]
    )
    maximum_common_radius = float(
        config["source_construction"]["maximum_common_working_support_radius_arcsec"]
    )
    frame["common_support_radius_arcsec"] = np.nan
    frame["structured_positive_flux_retained_fraction"] = np.nan
    frame["sersic_positive_flux_retained_fraction"] = np.nan
    frame["structured_negative_to_positive_flux"] = np.nan
    frame["sersic_negative_to_positive_flux"] = np.nan
    frame["construction_preflight_passed"] = False
    for position, index in enumerate(np.flatnonzero(still_eligible), start=1):
        galaxy_image, psf_image, _, native_scale, _ = catalogue.getRealParams(
            int(index)
        )
        height, width = galaxy_image.array.shape
        psf = galsim.InterpolatedImage(psf_image)
        psf_fwhm = float(psf.calculateFWHM())
        support_radius = (
            support_fraction * 0.5 * (min(height, width) - 1) * float(native_scale)
        )
        support_hlr = support_radius / float(
            frame.at[index, "sersic_circularized_hlr_arcsec"]
        )
        common_radius = min(support_radius, maximum_common_radius)
        frame.at[index, "native_height_pixels"] = height
        frame.at[index, "native_width_pixels"] = width
        frame.at[index, "native_pixel_scale_arcsec"] = float(native_scale)
        frame.at[index, "original_psf_fwhm_arcsec"] = psf_fwhm
        frame.at[index, "source_image_sha256"] = _array_bytes_sha256(galaxy_image.array)
        frame.at[index, "source_psf_image_sha256"] = _array_bytes_sha256(
            psf_image.array
        )
        frame.at[index, "common_support_radius_arcsec"] = support_radius
        frame.at[index, "support_hlr"] = support_hlr
        frame.at[index, "common_support_radius_arcsec"] = common_radius
        if support_hlr < minimum_support:
            frame.at[index, "eligibility_reason"] = (
                "insufficient_native_inscribed_support"
            )
        elif (
            config["source_construction"][
                "reconstruction_psf_must_exceed_original_fwhm"
            ]
            and psf_fwhm > reconstruction_fwhm
        ):
            frame.at[index, "eligibility_reason"] = (
                "original_psf_broader_than_reconstruction_psf"
            )
        else:
            row = SimpleNamespace(
                **frame.loc[index].to_dict(),
                source_id=f"cosmos-{frame.at[index, 'cosmos_ident']}",
            )
            structured_raw, sersic_raw, _ = _raw_source_profiles(
                row, catalogue=catalogue, config=config
            )
            stamp_size = int(config["source_construction"]["stamp_size_pixels"])
            target = (stamp_size - 1) / 2.0
            grid_y, grid_x = np.mgrid[:stamp_size, :stamp_size]
            working_radius = np.hypot(grid_x - target, grid_y - target) * pixel_scale
            support = working_radius <= common_radius

            def projection_diagnostics(
                raw: np.ndarray, support_mask: np.ndarray
            ) -> tuple[float, float]:
                positive = np.maximum(raw, 0.0)
                positive_total = float(positive.sum(dtype=np.float64))
                if positive_total <= 0:
                    return float("nan"), float("nan")
                retained = float(
                    positive[support_mask].sum(dtype=np.float64) / positive_total
                )
                negative = float(-raw[raw < 0].sum(dtype=np.float64))
                return retained, negative / positive_total

            structured_retained, structured_negative = projection_diagnostics(
                structured_raw, support
            )
            sersic_retained, sersic_negative = projection_diagnostics(
                sersic_raw, support
            )
            frame.at[index, "structured_positive_flux_retained_fraction"] = (
                structured_retained
            )
            frame.at[index, "sersic_positive_flux_retained_fraction"] = sersic_retained
            frame.at[index, "structured_negative_to_positive_flux"] = (
                structured_negative
            )
            frame.at[index, "sersic_negative_to_positive_flux"] = sersic_negative
            construction = config["source_construction"]
            if (
                structured_retained
                < construction["minimum_positive_flux_retained_in_common_support"]
                or sersic_retained
                < construction["minimum_positive_flux_retained_in_common_support"]
            ):
                frame.at[index, "eligibility_reason"] = (
                    "common_support_retains_too_little_positive_flux"
                )
            elif (
                structured_negative
                > construction["maximum_structured_negative_to_positive_flux"]
            ):
                frame.at[index, "eligibility_reason"] = (
                    "structured_nonnegative_projection_exceeds_gate"
                )
            elif (
                sersic_negative
                > construction["maximum_sersic_negative_to_positive_flux"]
            ):
                frame.at[index, "eligibility_reason"] = (
                    "sersic_nonnegative_projection_exceeds_gate"
                )
            else:
                frame.at[index, "eligibility_reason"] = "eligible"
                frame.at[index, "construction_preflight_passed"] = True
        if position % 1000 == 0:
            print(f"Inspected source support {position}", flush=True)
    frame["eligible"] = frame["eligibility_reason"].eq("eligible")
    return frame


def select_sources(
    ledger: pd.DataFrame,
    config: dict[str, Any],
    *,
    source_role: str = "experiment",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a deterministic stratified source sample without outcomes."""
    rules = config["selection"]["source"]
    role_rules = config["selection"].get("source_roles", {}).get(source_role)
    if role_rules is None:
        raise ContractError(f"unknown source role: {source_role}")
    eligible = ledger[ledger["eligible"]].copy()
    if not len(eligible):
        raise ValueError("no eligible high-S/N COSMOS sources")
    seed_namespace = str(config["selection"]["seed_namespace"])
    eligible["selection_hash"] = eligible["cosmos_ident"].map(
        lambda value: stable_hex(seed_namespace, "source", value)
    )
    strata = rules["strata"]
    eligible["hlr_stratum"] = pd.qcut(
        eligible["sersic_circularized_hlr_arcsec"].rank(method="first", ascending=True),
        q=int(strata["half_light_radius_quantiles"]),
        labels=False,
    ).astype(int)
    eligible["fit_mad_stratum"] = pd.qcut(
        eligible["single_sersic_fit_mad"].rank(method="first", ascending=True),
        q=int(strata["single_sersic_fit_mad_quantiles"]),
        labels=False,
    ).astype(int)
    eligible["source_stratum"] = (
        "h"
        + eligible["hlr_stratum"].astype(str)
        + "_m"
        + eligible["fit_mad_stratum"].astype(str)
    )
    n_strata = int(strata["half_light_radius_quantiles"]) * int(
        strata["single_sersic_fit_mad_quantiles"]
    )
    requested = int(role_rules["per_stratum"]) * n_strata
    if requested != int(rules["selected_sources"]):
        raise ContractError("source-role count differs from selected_sources")
    if requested % n_strata:
        raise ValueError("selected source count must divide evenly across strata")
    per_stratum = requested // n_strata
    counts = eligible["source_stratum"].value_counts()
    if len(counts) != n_strata or (counts < per_stratum).any():
        raise ValueError("source strata do not support the requested balanced sample")
    offset = int(role_rules["rank_offset"])
    ordered = eligible.sort_values(
        ["source_stratum", "selection_hash"], kind="mergesort"
    ).copy()
    ordered["source_role_rank"] = ordered.groupby("source_stratum").cumcount()
    selected = ordered[
        ordered["source_role_rank"].between(
            offset, offset + per_stratum - 1, inclusive="both"
        )
    ].copy()
    if len(selected) != requested:
        raise ContractError("source role lacks the requested stratum support")
    selected["source_role"] = source_role
    selected["source_stratum_population"] = selected["source_stratum"].map(counts)
    selected["source_selection_probability"] = (
        per_stratum / selected["source_stratum_population"]
    )
    selected["source_population_sensitivity_weight"] = (
        selected["source_stratum_population"] / per_stratum
    )
    selected["source_population_sensitivity_weight"] /= selected[
        "source_population_sensitivity_weight"
    ].mean()
    selected["source_analysis_weight"] = 1.0
    selected["source_id"] = selected["cosmos_ident"].map(
        lambda value: f"cosmos-{value}"
    )
    selected = selected.sort_values("source_id", kind="mergesort").reset_index(
        drop=True
    )
    selected["source_index"] = np.arange(len(selected), dtype=np.int64)
    augmented = ledger.copy()
    augmented[f"selected_{source_role}"] = augmented["cosmos_catalog_index"].isin(
        selected["cosmos_catalog_index"]
    )
    if source_role == "experiment":
        augmented["selected"] = augmented["selected_experiment"]
    return augmented, selected


def _centroid(image: np.ndarray) -> tuple[float, float]:
    total = float(image.sum(dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("source map has no positive finite flux")
    y, x = np.mgrid[: image.shape[0], : image.shape[1]]
    return (
        float((x * image).sum(dtype=np.float64) / total),
        float((y * image).sum(dtype=np.float64) / total),
    )


def _align_nonnegative(
    raw: np.ndarray,
    *,
    support: np.ndarray,
    pixel_scale: float,
    tolerance_arcsec: float,
) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(raw, dtype=np.float64)
    negative_flux = float(-values[values < 0].sum(dtype=np.float64))
    positive_flux = float(values[values > 0].sum(dtype=np.float64))
    values = np.maximum(values, 0.0)
    values[~support] = 0.0
    target = (values.shape[0] - 1) / 2.0
    for _ in range(12):
        centroid_x, centroid_y = _centroid(values)
        separation = np.hypot(centroid_x - target, centroid_y - target) * pixel_scale
        if separation <= tolerance_arcsec / 4:
            break
        values = ndimage.shift(
            values,
            shift=(target - centroid_y, target - centroid_x),
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
        values = np.maximum(values, 0.0)
        values[~support] = 0.0
    integral = float(values.sum(dtype=np.float64) * pixel_scale**2)
    if integral <= 0:
        raise ValueError("source map vanished during centroid alignment")
    values /= integral
    centroid_x, centroid_y = _centroid(values)
    residual = float(np.hypot(centroid_x - target, centroid_y - target) * pixel_scale)
    if residual > tolerance_arcsec:
        raise ValueError(f"centroid alignment failed: {residual} arcsec")
    return values, {
        "preprojection_negative_to_positive_flux": (
            negative_flux / positive_flux if positive_flux > 0 else float("nan")
        ),
        "centroid_x_pixels": centroid_x,
        "centroid_y_pixels": centroid_y,
        "centroid_residual_arcsec": residual,
    }


def _sersic_profile(row: Any) -> galsim.GSObject:
    return galsim.Sersic(
        n=float(row.sersic_n),
        half_light_radius=float(row.sersic_circularized_hlr_arcsec),
    ).shear(
        q=float(row.sersic_q),
        beta=float(row.sersic_phi_catalog_rad) * galsim.radians,
    )


def _raw_source_profiles(
    row: Any,
    *,
    catalogue: galsim.COSMOSCatalog,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Draw the two common-resolution profiles before support/projection."""
    construction = config["source_construction"]
    pixel_scale = float(construction["source_pixel_scale_arcsec"])
    stamp_size = int(construction["stamp_size_pixels"])
    reconstruction_fwhm = float(construction["reconstruction_psf_fwhm_arcsec"])
    base = catalogue.makeGalaxy(
        int(row.cosmos_catalog_index),
        gal_type="real",
        noise_pad_size=0,
    )
    original_fwhm = float(base.original_psf.calculateFWHM())
    if (
        construction["reconstruction_psf_must_exceed_original_fwhm"]
        and original_fwhm > reconstruction_fwhm
    ):
        raise ContractError(
            f"reconstruction PSF is too narrow for source {row.source_id}"
        )
    angle = np.random.default_rng(
        stable_seed(
            str(config["selection"]["seed_namespace"]),
            "rotation",
            row.source_id,
        )
    ).uniform(0.0, 2.0 * np.pi)
    reconstruction_psf = galsim.Gaussian(fwhm=reconstruction_fwhm)
    structured_profile = galsim.Convolve(
        base.rotate(angle * galsim.radians),
        reconstruction_psf,
    )
    twin_profile = galsim.Convolve(
        _sersic_profile(row).rotate(angle * galsim.radians),
        reconstruction_psf,
    )

    def draw(profile: galsim.GSObject) -> np.ndarray:
        return np.asarray(
            profile.drawImage(
                nx=stamp_size,
                ny=stamp_size,
                scale=pixel_scale,
                method="no_pixel",
            ).array,
            dtype=np.float64,
        )

    return (
        draw(structured_profile),
        draw(twin_profile),
        {
            "rotation_rad": float(angle),
            "original_hst_psf_fwhm_arcsec": original_fwhm,
        },
    )


def construct_source_maps(
    selected: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    *,
    final_status: bool = False,
) -> pd.DataFrame:
    """Construct aligned, common-support structured/Sersic source pairs."""
    assert_config_integrity(config)
    if final_status and config["status"] != "frozen_pre_scoring":
        raise ContractError("final source construction requires frozen config")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    map_dir = output_dir / "maps"
    map_dir.mkdir()
    root = Path(config["inputs"]["cosmos_dir"])
    catalogue = galsim.COSMOSCatalog(
        sample=str(config["selection"]["source"]["sample"]),
        dir=str(root),
    )
    construction = config["source_construction"]
    pixel_scale = float(construction["source_pixel_scale_arcsec"])
    stamp_size = int(construction["stamp_size_pixels"])
    tolerance = float(construction["centroid_separation_tolerance_arcsec"])
    target = (stamp_size - 1) / 2.0
    y, x = np.mgrid[:stamp_size, :stamp_size]
    radius = np.hypot(x - target, y - target) * pixel_scale
    records: list[dict[str, Any]] = []
    for position, row in enumerate(selected.itertuples(index=False), start=1):
        structured_raw, sersic_raw, raw_diagnostics = _raw_source_profiles(
            row, catalogue=catalogue, config=config
        )
        support = radius <= float(row.common_support_radius_arcsec)
        structured, structured_diagnostics = _align_nonnegative(
            structured_raw,
            support=support,
            pixel_scale=pixel_scale,
            tolerance_arcsec=tolerance,
        )
        twin, twin_diagnostics = _align_nonnegative(
            sersic_raw,
            support=support,
            pixel_scale=pixel_scale,
            tolerance_arcsec=tolerance,
        )
        separation = float(
            np.hypot(
                structured_diagnostics["centroid_x_pixels"]
                - twin_diagnostics["centroid_x_pixels"],
                structured_diagnostics["centroid_y_pixels"]
                - twin_diagnostics["centroid_y_pixels"],
            )
            * pixel_scale
        )
        if separation > tolerance:
            raise ValueError(f"arm centroid separation failed: {separation}")
        expected_metrics = {
            "structured_preprojection_negative_to_positive_flux": float(
                row.structured_negative_to_positive_flux
            ),
            "sersic_preprojection_negative_to_positive_flux": float(
                row.sersic_negative_to_positive_flux
            ),
        }
        observed_metrics = {
            "structured_preprojection_negative_to_positive_flux": (
                structured_diagnostics["preprojection_negative_to_positive_flux"]
            ),
            "sersic_preprojection_negative_to_positive_flux": twin_diagnostics[
                "preprojection_negative_to_positive_flux"
            ],
        }
        if any(
            not np.isclose(observed_metrics[key], expected, rtol=0, atol=1e-12)
            for key, expected in expected_metrics.items()
        ):
            raise ContractError("source preflight projection diagnostics drift")
        if (
            float(row.structured_positive_flux_retained_fraction)
            < construction["minimum_positive_flux_retained_in_common_support"]
            or float(row.sersic_positive_flux_retained_fraction)
            < construction["minimum_positive_flux_retained_in_common_support"]
            or observed_metrics["structured_preprojection_negative_to_positive_flux"]
            > construction["maximum_structured_negative_to_positive_flux"]
            or observed_metrics["sersic_preprojection_negative_to_positive_flux"]
            > construction["maximum_sersic_negative_to_positive_flux"]
        ):
            raise ContractError("selected source violates support/projection gate")
        structured32 = structured.astype(np.float32)
        twin32 = twin.astype(np.float32)
        filename = f"{int(row.source_index):04d}_{row.source_id}.npz"
        path = map_dir / filename
        temporary_path = map_dir / f".{filename}.partial.npz"
        np.savez_compressed(
            temporary_path,
            structured=structured32,
            sersic=twin32,
            common_support=support,
        )
        os.replace(temporary_path, path)
        record = row._asdict()
        record.update(
            {
                "rotation_rad": raw_diagnostics["rotation_rad"],
                "source_map_relpath": f"maps/{filename}",
                "source_map_sha256": sha256_file(path),
                "source_pixel_scale_arcsec": pixel_scale,
                "source_stamp_size_pixels": stamp_size,
                "reconstruction_psf_fwhm_arcsec": float(
                    construction["reconstruction_psf_fwhm_arcsec"]
                ),
                "structured_original_hst_psf_fwhm_arcsec": raw_diagnostics[
                    "original_hst_psf_fwhm_arcsec"
                ],
                "common_support_pixels": int(support.sum()),
                "structured_centroid_residual_arcsec": structured_diagnostics[
                    "centroid_residual_arcsec"
                ],
                "sersic_centroid_residual_arcsec": twin_diagnostics[
                    "centroid_residual_arcsec"
                ],
                "arm_centroid_separation_arcsec": separation,
                **observed_metrics,
                "structured_positive_flux_retained_fraction": float(
                    row.structured_positive_flux_retained_fraction
                ),
                "sersic_positive_flux_retained_fraction": float(
                    row.sersic_positive_flux_retained_fraction
                ),
                "support_treatment": construction["support_treatment"],
                "nonnegative_projection_treatment": construction[
                    "nonnegative_projection"
                ],
                "structured_is_latent_truth": False,
                "structured_contains_residual_measurement_uncertainty": True,
            }
        )
        records.append(record)
        if position % 24 == 0 or position == len(selected):
            print(f"Constructed sources {position}/{len(selected)}", flush=True)
    manifest = pd.DataFrame(records)
    manifest_path = output_dir / "source_manifest.parquet"
    temporary_manifest = output_dir / ".source_manifest.parquet.partial"
    manifest.to_parquet(temporary_manifest, index=False)
    os.replace(temporary_manifest, manifest_path)
    inventory = hashlib.sha256()
    for row in manifest.sort_values("source_id").itertuples(index=False):
        inventory.update(
            f"{row.source_map_relpath}\t{row.source_map_sha256}\n".encode()
        )
    receipt = {
        "schema_version": 2,
        "status": (
            config["execution"]["required_source_receipt_status"]
            if final_status
            else "restricted_empirical_source_construction_design_only"
        ),
        "sources": len(manifest),
        "source_role": str(manifest["source_role"].iloc[0]),
        "construction_label": config["claim"]["construction_label"],
        "is_latent_source_reconstruction": False,
        "is_multiband_morphology": False,
        "selection_used_detector_outcomes": False,
        "regularization": {
            "operator": "GalSim RealGalaxy deconvolution then common Gaussian reconvolution",
            "reconstruction_psf_fwhm_arcsec": float(
                construction["reconstruction_psf_fwhm_arcsec"]
            ),
            "nonnegative_projection": construction["nonnegative_projection"],
            "support_treatment": construction["support_treatment"],
            "minimum_positive_flux_retained": construction[
                "minimum_positive_flux_retained_in_common_support"
            ],
            "maximum_structured_negative_to_positive_flux": construction[
                "maximum_structured_negative_to_positive_flux"
            ],
        },
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_map_inventory_sha256": inventory.hexdigest(),
        "galsim_version": galsim.__version__,
        "config": config_provenance(config),
        "source_closure": source_closure(config),
    }
    atomic_json(output_dir / "receipt.json", receipt)
    return manifest


def write_selection(
    ledger: pd.DataFrame,
    selected: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    assert_config_integrity(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "source_eligibility.parquet"
    selected_path = output_dir / "selected_sources.parquet"
    ledger.to_parquet(ledger_path, index=False)
    selected.to_parquet(selected_path, index=False)
    summary = {
        "schema_version": 2,
        "status": "restricted_empirical_source_selection_design_only",
        "catalogue_rows": len(ledger),
        "eligible_rows": int(ledger["eligible"].sum()),
        "selected_rows": len(selected),
        "source_role": str(selected["source_role"].iloc[0]),
        "selection_used_detector_outcomes": False,
        "reasons": ledger["eligibility_reason"].value_counts().to_dict(),
        "source_eligibility_sha256": sha256_file(ledger_path),
        "selected_sources_sha256": sha256_file(selected_path),
        "config": config_provenance(config),
    }
    atomic_json(output_dir / "selection_receipt.json", summary)
    return summary
