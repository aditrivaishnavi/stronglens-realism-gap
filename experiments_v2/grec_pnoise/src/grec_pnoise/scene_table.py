from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .cells import derive_condition_cells
from .contracts import (
    ContractError,
    assert_config_integrity,
    assert_score_blind_columns,
    config_provenance,
    package_source_sha256,
    seed_from_key,
    sha256_json,
    stable_id,
)

AB_ZEROPOINT_MAG = 22.5
REQUIRED_HOST_COLUMNS = {
    "host_galaxy_id",
    "host_parent_id",
    "host_pixel_path",
    "host_payload_sha256",
    "host_pixel_sha256",
    "host_spatial_group_id",
    "host_spatial_coordinate_id",
    "host_ra_deg",
    "host_dec_deg",
    "final_role",
    "psf_bin",
    "depth_bin",
    "depth_mag",
    "psfsize_r",
    "psfdepth_r",
    "psf_fwhm_r_arcsec",
    "pixel_scale_arcsec",
    "host_pixel_unit",
    "host_flux_calibration_system",
    "host_calibration_provenance_sha256",
}


def _flux_nmgy(magnitude: float) -> float:
    return float(10.0 ** ((AB_ZEROPOINT_MAG - magnitude) / 2.5))


def _host_draw(host: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    namespace = str(config["grec"]["seed_namespace"])
    key_parts = (
        str(host["host_galaxy_id"]),
        str(host["host_pixel_sha256"]),
        str(host["host_spatial_group_id"]),
    )
    seed = seed_from_key(namespace, *key_parts)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    source = config["source"]
    lens = config["lens"]

    beta_fraction = math.sqrt(
        rng.uniform(
            float(source["beta_fraction_min"]) ** 2,
            float(source["beta_fraction_max"]) ** 2,
        )
    )
    beta_angle = float(rng.uniform(0.0, 2.0 * math.pi))
    r_mag = float(rng.uniform(source["r_mag_min"], source["r_mag_max_exclusive"]))
    g_minus_r = float(rng.normal(source["g_minus_r_mean"], source["g_minus_r_sigma"]))
    r_minus_z = float(rng.normal(source["r_minus_z_mean"], source["r_minus_z_sigma"]))
    g_mag = r_mag + g_minus_r
    z_mag = r_mag - r_minus_z

    values = {
        "base_draw_key": stable_id(
            "draw", [namespace, *key_parts], length=40
        ),
        "source_r_mag": r_mag,
        "source_g_minus_r": g_minus_r,
        "source_r_minus_z": r_minus_z,
        "source_re_arcsec": float(
            rng.uniform(source["re_arcsec_min"], source["re_arcsec_max"])
        ),
        "source_n_sersic": float(rng.uniform(source["n_min"], source["n_max"])),
        "source_axis_ratio": float(
            rng.uniform(source["axis_ratio_min"], source["axis_ratio_max"])
        ),
        "source_phi_rad": float(rng.uniform(0.0, 2.0 * math.pi)),
        "source_beta_fraction": beta_fraction,
        "source_beta_angle_rad": beta_angle,
        "source_flux_nmgy_g": _flux_nmgy(g_mag),
        "source_flux_nmgy_r": _flux_nmgy(r_mag),
        "source_flux_nmgy_z": _flux_nmgy(z_mag),
        "lens_axis_ratio": float(
            rng.uniform(lens["axis_ratio_min"], lens["axis_ratio_max"])
        ),
        "lens_phi_rad": float(rng.uniform(0.0, math.pi)),
        "shear_g1": float(rng.normal(0.0, lens["shear_component_sigma"])),
        "shear_g2": float(rng.normal(0.0, lens["shear_component_sigma"])),
        "lens_x0_arcsec": float(
            rng.normal(0.0, lens["centre_offset_sigma_arcsec"])
        ),
        "lens_y0_arcsec": float(
            rng.normal(0.0, lens["centre_offset_sigma_arcsec"])
        ),
    }
    source_shared = {k: v for k, v in values.items() if k.startswith("source_")}
    lens_shared = {
        k: v
        for k, v in values.items()
        if k.startswith("lens_") or k.startswith("shear_")
    }
    values["source_nuisance_id"] = stable_id("srcn", source_shared)
    values["lens_nuisance_id"] = stable_id("lensn", lens_shared)
    return values


def render_contract_payload(
    config: Mapping[str, Any], radii: Sequence[float] | None = None
) -> dict[str, Any]:
    provenance = config_provenance(config)
    return {
        "config_file_sha256": provenance["config_file_sha256"],
        "config_contract_sha256": provenance["config_contract_sha256"],
        "governing_document_sha256": provenance["governing_document_sha256"],
        "source": config["source"],
        "lens": config["lens"],
        "render": config["render"],
        "host": config["host"],
        "cells": config["cells"],
        "radii_arcsec": list(radii or config["grec"]["radii_arcsec"]),
        "seed_namespace": config["grec"]["seed_namespace"],
        "implementation_source_sha256": package_source_sha256(
            "cells.py", "contracts.py", "rendering.py", "scene_table.py"
        ),
    }


def build_scene_table(
    hosts: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    expected_hosts: int | None = None,
    expected_cells: int | None = None,
    hosts_per_cell: int | None = None,
    radii_arcsec: Sequence[float] | None = None,
    required_role: str = "experiment_host",
) -> pd.DataFrame:
    """Expand finalized hosts over radii while sharing all nuisance draws per host."""
    assert_config_integrity(config)
    assert_score_blind_columns(list(hosts.columns))
    missing = sorted(REQUIRED_HOST_COLUMNS - set(hosts.columns))
    if missing:
        raise ContractError(f"host manifest lacks required columns: {missing}")
    if hosts["host_galaxy_id"].astype(str).duplicated().any():
        raise ContractError("host_galaxy_id must be unique before radius expansion")
    if hosts["host_pixel_sha256"].astype(str).duplicated().any():
        raise ContractError("duplicate host pixel hashes are forbidden")
    if hosts["host_spatial_coordinate_id"].astype(str).duplicated().any():
        raise ContractError("duplicate spatial objects are forbidden")
    if not hosts["final_role"].eq(required_role).all():
        raise ContractError(f"every scene host must have final_role={required_role}")
    if not hosts["host_pixel_unit"].eq(config["host"]["pixel_unit"]).all():
        raise ContractError("host pixel units differ from rendering contract")
    if not hosts["host_flux_calibration_system"].eq(
        config["host"]["flux_calibration_system"]
    ).all():
        raise ContractError("host flux calibration differs from rendering contract")

    grec = config["grec"]
    expected_hosts = int(expected_hosts or grec["expected_hosts"])
    expected_cells = int(expected_cells or grec["expected_cells"])
    hosts_per_cell = int(hosts_per_cell or grec["hosts_per_cell"])
    radii = tuple(float(x) for x in (radii_arcsec or grec["radii_arcsec"]))
    if len(set(radii)) != len(radii) or tuple(sorted(radii)) != radii:
        raise ContractError("Einstein radii must be unique and increasing")
    if len(hosts) != expected_hosts:
        raise ContractError(f"expected {expected_hosts} hosts, found {len(hosts)}")

    counts = hosts.groupby(["psf_bin", "depth_bin"], dropna=False).size()
    if len(counts) != expected_cells or not counts.eq(hosts_per_cell).all():
        raise ContractError(
            f"expected {expected_cells} cells x {hosts_per_cell} hosts; "
            f"observed {counts.to_dict()}"
        )
    finite_columns = ["psf_bin", "depth_bin", "psf_fwhm_r_arcsec", "pixel_scale_arcsec"]
    if not np.isfinite(hosts[finite_columns].to_numpy(dtype=float)).all():
        raise ContractError("host observing metadata contains non-finite values")
    if (hosts["psf_fwhm_r_arcsec"] <= 0).any() or (
        hosts["pixel_scale_arcsec"] <= 0
    ).any():
        raise ContractError("PSF FWHM and pixel scale must be positive")
    derived_psf, derived_depth, derived_depth_mag = derive_condition_cells(
        hosts["psfsize_r"].to_numpy(dtype=float),
        hosts["psfdepth_r"].to_numpy(dtype=float),
        config,
    )
    if (
        not np.array_equal(hosts["psf_bin"].to_numpy(dtype=float), derived_psf)
        or not np.array_equal(
            hosts["depth_bin"].to_numpy(dtype=float), derived_depth
        )
        or not np.allclose(
            hosts["depth_mag"].to_numpy(dtype=float),
            derived_depth_mag,
            rtol=0.0,
            atol=1.0e-12,
        )
        or not np.array_equal(
            hosts["psf_fwhm_r_arcsec"].to_numpy(dtype=float),
            hosts["psfsize_r"].to_numpy(dtype=float),
        )
    ):
        raise ContractError("host condition cells/PSF are stale relative to raw values")

    provenance = config_provenance(config)
    render_contract = render_contract_payload(config, radii)
    render_contract_sha256 = sha256_json(render_contract)
    render_id = f"render_{render_contract_sha256[:32]}"
    rows: list[dict[str, Any]] = []
    ordered_hosts = hosts.sort_values(
        ["psf_bin", "depth_bin", "host_galaxy_id"], kind="mergesort"
    )
    for raw_host in ordered_hosts.to_dict(orient="records"):
        host = {k: _python_scalar(v) for k, v in raw_host.items()}
        shared = _host_draw(host, config)
        for radius_index, radius in enumerate(radii):
            beta = shared["source_beta_fraction"] * radius
            beta_x = beta * math.cos(shared["source_beta_angle_rad"])
            beta_y = beta * math.sin(shared["source_beta_angle_rad"])
            source_parameter_id = stable_id(
                "src",
                {
                    "nuisance": shared["source_nuisance_id"],
                    "theta_e_arcsec": radius,
                    "beta_x_arcsec": beta_x,
                    "beta_y_arcsec": beta_y,
                },
            )
            lens_parameter_id = stable_id(
                "lens",
                {
                    "nuisance": shared["lens_nuisance_id"],
                    "theta_e_arcsec": radius,
                },
            )
            row = {
                **host,
                **shared,
                "radius_index": radius_index,
                "theta_e_arcsec": radius,
                "source_beta_x_arcsec": beta_x,
                "source_beta_y_arcsec": beta_y,
                "source_parameter_id": source_parameter_id,
                "lens_parameter_id": lens_parameter_id,
                "render_config_id": render_id,
                "render_contract_sha256": render_contract_sha256,
                "config_file_sha256": provenance["config_file_sha256"],
                "config_contract_sha256": provenance["config_contract_sha256"],
                "psf_fwhm_g_arcsec": float(host["psf_fwhm_r_arcsec"])
                * float(config["render"]["psf_fwhm_scale_g"]),
                "psf_fwhm_z_arcsec": float(host["psf_fwhm_r_arcsec"])
                * float(config["render"]["psf_fwhm_scale_z"]),
                "host_scientific_weight": 1.0 / expected_hosts,
                "scene_within_host_weight": 1.0 / len(radii),
                "equal_cell_scene_weight": 1.0
                / (expected_cells * hosts_per_cell * len(radii)),
                "added_source_photon_noise": False,
                "added_sky_noise": False,
            }
            rows.append(bind_scene_identity(row))

    result = pd.DataFrame(rows)
    expected_rows = expected_hosts * len(radii)
    if len(result) != expected_rows:
        raise AssertionError("scene expansion produced an unexpected row count")
    shared_columns = [
        "base_draw_key",
        "source_nuisance_id",
        "lens_nuisance_id",
        "source_r_mag",
        "source_re_arcsec",
        "source_n_sersic",
        "source_axis_ratio",
        "source_phi_rad",
        "source_beta_fraction",
        "source_beta_angle_rad",
        "source_flux_nmgy_g",
        "source_flux_nmgy_r",
        "source_flux_nmgy_z",
        "lens_axis_ratio",
        "lens_phi_rad",
        "shear_g1",
        "shear_g2",
        "lens_x0_arcsec",
        "lens_y0_arcsec",
    ]
    nuisance_counts = result.groupby("host_galaxy_id")[shared_columns].nunique()
    if not nuisance_counts.eq(1).all().all():
        raise AssertionError("a host's nuisance draw changed across radius rows")
    if not result.groupby("host_galaxy_id")["theta_e_arcsec"].nunique().eq(
        len(radii)
    ).all():
        raise AssertionError("a host does not have exactly one row at every radius")
    if result["scene_id"].duplicated().any() or result["grec_row_id"].duplicated().any():
        raise AssertionError("scene identifiers are not unique")
    assert_score_blind_columns(list(result.columns))
    return result


SCENE_RENDER_INPUT_COLUMNS = (
    "host_galaxy_id",
    "host_parent_id",
    "host_pixel_sha256",
    "host_payload_sha256",
    "host_spatial_group_id",
    "host_spatial_coordinate_id",
    "host_ra_deg",
    "host_dec_deg",
    "host_pixel_unit",
    "host_flux_calibration_system",
    "host_calibration_provenance_sha256",
    "final_role",
    "psfsize_r",
    "psfdepth_r",
    "psf_bin",
    "depth_bin",
    "depth_mag",
    "pixel_scale_arcsec",
    "psf_fwhm_r_arcsec",
    "psf_fwhm_g_arcsec",
    "psf_fwhm_z_arcsec",
    "radius_index",
    "theta_e_arcsec",
    "source_r_mag",
    "source_g_minus_r",
    "source_r_minus_z",
    "source_re_arcsec",
    "source_n_sersic",
    "source_axis_ratio",
    "source_phi_rad",
    "source_beta_fraction",
    "source_beta_angle_rad",
    "source_beta_x_arcsec",
    "source_beta_y_arcsec",
    "source_flux_nmgy_g",
    "source_flux_nmgy_r",
    "source_flux_nmgy_z",
    "source_nuisance_id",
    "source_parameter_id",
    "lens_axis_ratio",
    "lens_phi_rad",
    "shear_g1",
    "shear_g2",
    "lens_x0_arcsec",
    "lens_y0_arcsec",
    "lens_nuisance_id",
    "lens_parameter_id",
    "base_draw_key",
    "render_config_id",
    "render_contract_sha256",
    "config_file_sha256",
    "config_contract_sha256",
    "added_source_photon_noise",
    "added_sky_noise",
)


def scene_identity_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in SCENE_RENDER_INPUT_COLUMNS if name not in row]
    if missing:
        raise ContractError(f"scene identity lacks render inputs: {missing}")
    return {name: _python_scalar(row[name]) for name in SCENE_RENDER_INPUT_COLUMNS}


def bind_scene_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    bound = dict(row)
    payload = scene_identity_payload(bound)
    bound["parameter_row_sha256"] = sha256_json(payload)
    bound["scene_id"] = stable_id("scene", payload, length=40)
    bound["grec_row_id"] = stable_id(
        "grec", [bound["scene_id"], bound["parameter_row_sha256"]], length=40
    )
    return bound


def verify_scene_identity(row: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    assert_config_integrity(config)
    provenance = config_provenance(config)
    if row.get("config_file_sha256") != provenance["config_file_sha256"]:
        raise ContractError("scene was built from a different configuration file")
    if row.get("config_contract_sha256") != provenance["config_contract_sha256"]:
        raise ContractError("scene scientific configuration drift")
    expected_render_contract = sha256_json(render_contract_payload(config))
    if row.get("render_contract_sha256") != expected_render_contract:
        raise ContractError("scene renderer implementation/configuration drift")
    if row.get("render_config_id") != f"render_{expected_render_contract[:32]}":
        raise ContractError("scene render configuration ID drift")
    payload = scene_identity_payload(row)
    if sha256_json(payload) != row.get("parameter_row_sha256"):
        raise ContractError("scene parameter digest does not match render inputs")
    expected_scene = stable_id("scene", payload, length=40)
    if expected_scene != row.get("scene_id"):
        raise ContractError("scene ID does not bind its complete render inputs")


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
