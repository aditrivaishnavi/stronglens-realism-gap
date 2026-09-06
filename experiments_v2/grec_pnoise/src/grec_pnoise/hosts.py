from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cells import derive_condition_cells
from .contracts import (
    ContractError,
    array_bytes_sha256,
    assert_score_blind_columns,
    atomic_write_json,
    config_provenance,
    load_json,
    require_receipt,
    sha256_file,
    stable_id,
)

FINAL_SELECTION_STATUS = "final_experiment_host_selection_approved"
REQUIRED_INPUT_COLUMNS = {
    "galaxy_id",
    "local_relpath",
    "payload_sha256",
    "evaluation_spatial_group_id",
    "final_role",
    "psfsize_r",
    "psfdepth_r",
    "pixel_scale",
    "ra",
    "dec",
    "pixel_unit",
    "flux_calibration_system",
    "calibration_provenance_path",
    "calibration_provenance_sha256",
}


def prepare_final_hosts(
    selection: pd.DataFrame,
    payload_root: str | Path,
    config: Mapping[str, Any],
    *,
    expected_hosts: int | None = None,
    expected_cells: int | None = None,
    hosts_per_cell: int | None = None,
) -> pd.DataFrame:
    """Validate final selected payloads and record exact cutout element hashes."""
    assert_score_blind_columns(list(selection.columns))
    missing = sorted(REQUIRED_INPUT_COLUMNS - set(selection.columns))
    if missing:
        raise ContractError(f"final host selection lacks columns: {missing}")
    if "proposed_role" in selection.columns:
        raise ContractError("a proposal column cannot serve as final role authority")
    if not selection["final_role"].eq("experiment_host").all():
        raise ContractError("final host selection contains a non-host role")
    if selection["galaxy_id"].astype(str).duplicated().any():
        raise ContractError("final host galaxy IDs must be unique")
    if selection["payload_sha256"].astype(str).duplicated().any():
        raise ContractError("duplicate payload hashes cannot receive independent weight")
    psf_bins, depth_bins, depth_magnitudes = derive_condition_cells(
        selection["psfsize_r"].to_numpy(dtype=float),
        selection["psfdepth_r"].to_numpy(dtype=float),
        config,
    )
    selection = selection.copy()
    for supplied, derived in (
        ("psf_bin", psf_bins),
        ("depth_bin", depth_bins),
    ):
        if supplied in selection and not np.array_equal(
            selection[supplied].to_numpy(dtype=float), derived
        ):
            raise ContractError(f"supplied {supplied} differs from raw-value assignment")
    selection["psf_bin"] = psf_bins
    selection["depth_bin"] = depth_bins
    selection["depth_mag"] = depth_magnitudes
    if not np.isfinite(selection[["ra", "dec"]].to_numpy(dtype=float)).all():
        raise ContractError("host coordinates must be finite")
    if (
        (selection["ra"] < 0).any()
        or (selection["ra"] >= 360).any()
        or (selection["dec"] < -90).any()
        or (selection["dec"] > 90).any()
    ):
        raise ContractError("host coordinates are outside valid sky ranges")
    spatial_keys = (
        selection["ra"].map(lambda value: f"{float(value):.10f}")
        + "/"
        + selection["dec"].map(lambda value: f"{float(value):.10f}")
    )
    if spatial_keys.duplicated().any():
        raise ContractError("duplicate sky coordinates cannot receive independent weight")
    expected_unit = str(config["host"]["pixel_unit"])
    expected_calibration = str(config["host"]["flux_calibration_system"])
    if not selection["pixel_unit"].eq(expected_unit).all():
        raise ContractError("host pixel units differ from the nanomaggy contract")
    if not selection["flux_calibration_system"].eq(expected_calibration).all():
        raise ContractError("host flux calibration system differs from contract")
    expected_hosts = int(expected_hosts or config["grec"]["expected_hosts"])
    expected_cells = int(expected_cells or config["grec"]["expected_cells"])
    hosts_per_cell = int(hosts_per_cell or config["grec"]["hosts_per_cell"])
    if len(selection) != expected_hosts:
        raise ContractError(f"final selection has {len(selection)}, expected {expected_hosts}")
    counts = selection.groupby(["psf_bin", "depth_bin"]).size()
    if len(counts) != expected_cells or not counts.eq(hosts_per_cell).all():
        raise ContractError("final selected cells are not exactly 9 x 500 hosts")

    root = Path(payload_root)
    rows: list[dict[str, Any]] = []
    calibration_cache: dict[str, dict[str, Any]] = {}
    for source in selection.sort_values(
        ["psf_bin", "depth_bin", "galaxy_id"], kind="mergesort"
    ).to_dict(orient="records"):
        payload = (root / str(source["local_relpath"])).resolve()
        try:
            payload.relative_to(root.resolve())
        except ValueError as error:
            raise ContractError(f"payload escapes root: {payload}") from error
        if sha256_file(payload) != str(source["payload_sha256"]):
            raise ContractError(f"payload digest mismatch: {payload}")
        calibration_path = Path(str(source["calibration_provenance_path"])).resolve()
        calibration_sha256 = str(source["calibration_provenance_sha256"])
        if sha256_file(calibration_path) != calibration_sha256:
            raise ContractError("host calibration provenance digest mismatch")
        calibration_receipt = calibration_cache.get(calibration_sha256)
        if calibration_receipt is None:
            calibration_receipt = load_json(calibration_path)
            calibration_cache[calibration_sha256] = calibration_receipt
        calibration_checks = {
            "status": config["host"]["required_calibration_receipt_status"],
            "pixel_unit": expected_unit,
            "flux_calibration_system": expected_calibration,
        }
        if any(
            calibration_receipt.get(field) != expected
            for field, expected in calibration_checks.items()
        ):
            raise ContractError("host calibration provenance receipt is incompatible")
        with np.load(payload, allow_pickle=False) as archive:
            if "cutout" not in archive.files:
                raise ContractError(f"payload lacks cutout: {payload}")
            cutout = np.array(archive["cutout"], copy=True)
        if (
            cutout.dtype != np.float32
            or cutout.shape != (101, 101, 3)
            or not np.isfinite(cutout).all()
        ):
            raise ContractError(
                f"host cutout must be finite float32 (101,101,3): {payload}"
            )
        host_pixel_sha256 = array_bytes_sha256(cutout)
        host_parent_id = stable_id(
            "host",
            {
                "galaxy_id": str(source["galaxy_id"]),
                "payload_sha256": str(source["payload_sha256"]),
                "host_pixel_sha256": host_pixel_sha256,
                "spatial_group": str(source["evaluation_spatial_group_id"]),
            },
            length=40,
        )
        rows.append(
            {
                "host_galaxy_id": str(source["galaxy_id"]),
                "host_parent_id": host_parent_id,
                "host_pixel_path": str(payload),
                "host_payload_sha256": str(source["payload_sha256"]),
                "host_pixel_sha256": host_pixel_sha256,
                "host_spatial_group_id": str(
                    source["evaluation_spatial_group_id"]
                ),
                "host_ra_deg": float(source["ra"]),
                "host_dec_deg": float(source["dec"]),
                "host_spatial_coordinate_id": stable_id(
                    "coord",
                    [round(float(source["ra"]), 10), round(float(source["dec"]), 10)],
                    length=40,
                ),
                "final_role": "experiment_host",
                "psf_bin": float(source["psf_bin"]),
                "depth_bin": float(source["depth_bin"]),
                "depth_mag": float(source["depth_mag"]),
                "psfsize_r": float(source["psfsize_r"]),
                "psfdepth_r": float(source["psfdepth_r"]),
                "psf_fwhm_r_arcsec": float(source["psfsize_r"]),
                "pixel_scale_arcsec": float(source["pixel_scale"]),
                "host_pixel_unit": expected_unit,
                "host_flux_calibration_system": expected_calibration,
                "host_calibration_provenance_path": str(calibration_path),
                "host_calibration_provenance_sha256": calibration_sha256,
            }
        )
    result = pd.DataFrame(rows)
    if result["host_pixel_sha256"].duplicated().any():
        raise ContractError("duplicate pixel arrays cannot receive independent weight")
    if result["host_spatial_coordinate_id"].duplicated().any():
        raise ContractError("duplicate spatial objects cannot receive independent weight")
    maximum_group = int(config["grec"]["maximum_hosts_per_spatial_group_per_cell"])
    group_counts = result.groupby(
        ["psf_bin", "depth_bin", "host_spatial_group_id"]
    ).size()
    if (group_counts > maximum_group).any():
        raise ContractError("a spatial group exceeds its preregistered per-cell cap")
    assert_score_blind_columns(list(result.columns))
    return result


def prepare_and_write_final_hosts(
    selection_path: str | Path,
    selection_receipt_path: str | Path,
    payload_root: str | Path,
    config: Mapping[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    selection_receipt = require_receipt(
        selection_receipt_path,
        status=FINAL_SELECTION_STATUS,
        artifact_path=selection_path,
        artifact_digest_field="selection_manifest_sha256",
    )
    gates = {
        "detector_scores_used": False,
        "full_tractor_eligibility": "passed",
        "physical_group_checks": "passed",
        "role_overlap_checks": "passed",
        "pre_score_review": "approved",
    }
    failures = {
        field: (selection_receipt.get(field), expected)
        for field, expected in gates.items()
        if selection_receipt.get(field) != expected
    }
    if failures:
        raise ContractError(f"final selection receipt has unresolved gates: {failures}")
    selection = pd.read_parquet(selection_path)
    hosts = prepare_final_hosts(selection, payload_root, config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "final_grec_hosts.parquet"
    partial = output / "final_grec_hosts.partial.parquet"
    hosts.to_parquet(partial, index=False)
    partial.replace(manifest_path)
    receipt = {
        "schema_version": 1,
        "status": config["grec"]["required_host_receipt_status"],
        "host_manifest_sha256": sha256_file(manifest_path),
        "selection_manifest_sha256": sha256_file(selection_path),
        "selection_receipt_sha256": sha256_file(selection_receipt_path),
        "rows": len(hosts),
        "distinct_cells": int(
            hosts[["psf_bin", "depth_bin"]].drop_duplicates().shape[0]
        ),
        **gates,
        "host_payload_sha256_verified": True,
        "host_pixel_element_sha256_recorded": True,
        "duplicate_pixel_hashes": 0,
        "duplicate_spatial_coordinates": 0,
        "condition_cells_derived_from": ["psfsize_r", "psfdepth_r"],
        "condition_midpoint_tie_rule": "lower_node",
        "host_pixel_unit": config["host"]["pixel_unit"],
        "host_flux_calibration_system": config["host"]["flux_calibration_system"],
        "config": config_provenance(config),
    }
    receipt_path = output / "FINAL_GREC_HOSTS_RECEIPT.json"
    atomic_write_json(receipt_path, receipt)
    return manifest_path, receipt_path
