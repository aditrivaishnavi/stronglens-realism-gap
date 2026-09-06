from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


class ContractError(RuntimeError):
    """Fail-closed contract error."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def stable_id(prefix: str, value: Any, length: int = 40) -> str:
    return f"{prefix}_{sha256_json(value)[:length]}"


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


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 2:
        raise ContractError("expected parent-bound P_src schema_version 2")
    if config.get("status") not in {
        "implementation_candidate_not_execution_authority",
        "frozen_pre_scoring",
    }:
        raise ContractError("unknown P_src configuration status")
    for document, expected in config["governance"]["documents"].items():
        observed = sha256_file(document)
        if observed != expected:
            raise ContractError(
                f"governing document drift at {document}: {observed} != {expected}"
            )
    if (
        config["rendering"]["primary_flux_policy"] != "fixed_parent_intrinsic_per_band"
        or not config["rendering"]["prohibit_primary_post_lensing_rescale"]
        or not config["photon_noise"]["separated_from_renderer"]
        or config["threshold"]["comparator"] != ">="
        or config["scoring"]["comparator"] != ">="
    ):
        raise ContractError("primary flux/noise/threshold contract drift")
    project_root = Path(__file__).resolve().parents[2]
    config["__provenance__"] = {
        "config_relpath": os.path.relpath(config_path, project_root),
        "config_file_sha256": sha256_file(config_path),
        "config_contract_sha256": sha256_json(config),
        "governing_document_sha256": dict(config["governance"]["documents"]),
    }
    return config


def config_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("__provenance__")
    if not isinstance(value, dict):
        raise ContractError("configuration was not loaded through load_config")
    return dict(value)


def assert_config_integrity(config: Mapping[str, Any]) -> None:
    provenance = config_provenance(config)
    current = {key: value for key, value in config.items() if key != "__provenance__"}
    if sha256_json(current) != provenance["config_contract_sha256"]:
        raise ContractError("in-memory configuration drift")
    project_root = Path(__file__).resolve().parents[2]
    config_path = (project_root / provenance["config_relpath"]).resolve()
    if sha256_file(config_path) != provenance["config_file_sha256"]:
        raise ContractError("configuration file drift")


def source_closure(config: Mapping[str, Any]) -> dict[str, Any]:
    assert_config_integrity(config)
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parents[1]
    files = {
        str(path.relative_to(project_root)): sha256_file(path)
        for path in sorted(package_root.glob("*.py"))
    }
    files["pyproject.toml"] = sha256_file(project_root / "pyproject.toml")
    lock = project_root / "uv.lock"
    if not lock.is_file():
        raise ContractError("P_src dependency lock is missing")
    files["uv.lock"] = sha256_file(lock)
    closure = {
        "schema_version": 1,
        "config": config_provenance(config),
        "files": files,
    }
    closure["aggregate_sha256"] = sha256_json(closure)
    return closure


def verify_source_closure(
    expected: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    current = source_closure(config)
    if dict(expected) != current:
        raise ContractError("complete P_src source/config closure drift")


def fixed_scorer_closure(
    config: Mapping[str, Any], *, verify_runtime: bool
) -> dict[str, Any]:
    assert_config_integrity(config)
    scoring = config["scoring"]
    root = Path(scoring["repo_root"]).resolve()
    checkpoint = Path(scoring["checkpoint"]).resolve()
    if sha256_file(checkpoint) != scoring["checkpoint_sha256"]:
        raise ContractError("fixed checkpoint digest drift")
    for relative, expected in scoring["source_sha256"].items():
        observed = sha256_file(root / relative)
        if observed != expected:
            raise ContractError(f"fixed scorer source drift: {relative}")
    if verify_runtime:
        observed_versions = {
            "python": sys.version.split()[0],
            **{
                package: importlib.metadata.version(package)
                for package in scoring["runtime_versions"]
                if package != "python"
            },
        }
        mismatches = {
            package: (observed_versions.get(package), expected)
            for package, expected in scoring["runtime_versions"].items()
            if observed_versions.get(package) != str(expected)
        }
        if mismatches:
            raise ContractError(f"fixed scorer runtime drift: {mismatches}")
    closure = {
        "schema_version": 1,
        "checkpoint_locator": "configured_external_checkpoint",
        "checkpoint_sha256": scoring["checkpoint_sha256"],
        "checkpoint_epoch": int(scoring["checkpoint_epoch"]),
        "checkpoint_architecture": str(scoring["checkpoint_architecture"]),
        "scorer_source_sha256": dict(scoring["source_sha256"]),
        "runtime_versions": dict(scoring["runtime_versions"]),
        "batch_size": int(scoring["batch_size"]),
        "filler_policy": str(scoring["filler_policy"]),
        "native_logit_dtype": str(scoring["native_logit_dtype"]),
        "comparator": str(scoring["comparator"]),
        "authoritative_device": str(scoring["authoritative_device"]),
    }
    closure["aggregate_sha256"] = sha256_json(closure)
    return closure


def write_scorer_receipt(
    config: Mapping[str, Any],
    output: str | Path,
    *,
    final_status: bool = False,
) -> dict[str, Any]:
    closure = fixed_scorer_closure(config, verify_runtime=True)
    receipt = {
        "schema_version": 1,
        "status": (
            config["scoring"]["required_scorer_receipt_status"]
            if final_status
            else "fixed_scorer_verified_design_only"
        ),
        "final_outcomes_used": False,
        "config": config_provenance(config),
        "scorer_closure": closure,
        "scorer_closure_sha256": sha256_json(closure),
    }
    atomic_write_json(output, receipt)
    return receipt


def validate_scorer_receipt(
    path: str | Path,
    config: Mapping[str, Any],
    *,
    require_final: bool,
) -> dict[str, Any]:
    receipt = load_json(path)
    expected_status = (
        config["scoring"]["required_scorer_receipt_status"]
        if require_final
        else "fixed_scorer_verified_design_only"
    )
    if receipt.get("status") != expected_status:
        raise ContractError("fixed scorer receipt status is not authorized")
    closure = fixed_scorer_closure(config, verify_runtime=require_final)
    if (
        receipt.get("config") != config_provenance(config)
        or receipt.get("scorer_closure") != closure
        or receipt.get("scorer_closure_sha256") != sha256_json(closure)
    ):
        raise ContractError("fixed scorer receipt drift")
    return receipt


GREC_SCENE_RENDER_INPUT_COLUMNS = (
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


def verify_grec_scene_identity(row: Mapping[str, Any]) -> None:
    missing = [name for name in GREC_SCENE_RENDER_INPUT_COLUMNS if name not in row]
    if missing:
        raise ContractError(f"G_rec scene identity lacks: {missing}")
    payload = {
        name: _python_scalar(row[name]) for name in GREC_SCENE_RENDER_INPUT_COLUMNS
    }
    parameter_digest = sha256_json(payload)
    expected_scene_id = stable_id("scene", payload)
    expected_grec_id = stable_id("grec", [expected_scene_id, parameter_digest])
    if (
        row.get("parameter_row_sha256") != parameter_digest
        or row.get("scene_id") != expected_scene_id
        or row.get("grec_row_id") != expected_grec_id
    ):
        raise ContractError("G_rec scene parameter digest or identity drift")


def verify_parent_grec_authority(
    *,
    final_host_manifest_path: str | Path,
    final_host_receipt_path: str | Path,
    scene_table_path: str | Path,
    scene_receipt_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate the complete final-host -> scene -> rendered-parent authority."""
    assert_config_integrity(config)
    parent_contract = config["parent_grec"]
    host_receipt = load_json(final_host_receipt_path)
    host_sha = sha256_file(final_host_manifest_path)
    required_host = {
        "status": parent_contract["required_final_host_status"],
        "host_manifest_sha256": host_sha,
        "detector_scores_used": False,
        "full_tractor_eligibility": "passed",
        "physical_group_checks": "passed",
        "role_overlap_checks": "passed",
        "pre_score_review": "approved",
    }
    if any(host_receipt.get(key) != value for key, value in required_host.items()):
        raise ContractError("final G_rec host receipt is absent, stale, or unapproved")
    final_hosts = pd.read_parquet(final_host_manifest_path)
    if "host_physical_group_id" not in final_hosts:
        selection_binding = parent_contract["final_selection_manifest"]
        selection_path = (
            Path(config["scoring"]["repo_root"]).resolve()
            / str(selection_binding["path"])
        )
        if (
            not selection_path.is_file()
            or sha256_file(selection_path) != selection_binding["sha256"]
        ):
            raise ContractError("final host physical-group authority differs")
        selection = pd.read_parquet(
            selection_path,
            columns=["galaxy_id", "physical_group_id"],
        )
        if (
            len(selection) != len(final_hosts)
            or selection["galaxy_id"].astype(str).duplicated().any()
        ):
            raise ContractError("final host physical-group roster differs")
        final_hosts = final_hosts.merge(
            selection.rename(
                columns={
                    "galaxy_id": "host_galaxy_id",
                    "physical_group_id": "host_physical_group_id",
                }
            ),
            on="host_galaxy_id",
            how="left",
            validate="one_to_one",
        )
    host_ancestry_columns = [
        "host_galaxy_id",
        "host_physical_group_id",
        "host_spatial_group_id",
        "host_spatial_coordinate_id",
        "host_parent_id",
        "host_pixel_sha256",
    ]
    if (
        set(host_ancestry_columns) - set(final_hosts)
        or final_hosts["host_galaxy_id"].astype(str).duplicated().any()
        or final_hosts[host_ancestry_columns].isna().any().any()
    ):
        raise ContractError("final G_rec hosts lack unique complete ancestry fields")
    scene_receipt = load_json(scene_receipt_path)
    scene_sha = sha256_file(scene_table_path)
    required_scene = {
        "status": parent_contract["required_scene_status"],
        "scene_table_sha256": scene_sha,
        "host_manifest_sha256": host_sha,
        "host_receipt_sha256": sha256_file(final_host_receipt_path),
        "host_receipt_status": parent_contract["required_final_host_status"],
        "detector_scores_used": False,
    }
    if any(scene_receipt.get(key) != value for key, value in required_scene.items()):
        raise ContractError("registered G_rec scene receipt is invalid")
    grec_provenance = scene_receipt.get("config", {})
    if (
        grec_provenance.get("config_file_sha256")
        != parent_contract["required_grec_config_file_sha256"]
        or grec_provenance.get("config_contract_sha256")
        != parent_contract["required_grec_config_contract_sha256"]
    ):
        raise ContractError("G_rec configuration authority differs from P_src contract")
    parent_receipt = load_json(parent_receipt_path)
    parent_sha = sha256_file(parent_manifest_path)
    required_parent = {
        "status": parent_contract["required_parent_bank_status"],
        "parent_manifest_sha256": parent_sha,
        "scene_table_sha256": scene_sha,
        "scene_table_receipt_sha256": sha256_file(scene_receipt_path),
    }
    if any(parent_receipt.get(key) != value for key, value in required_parent.items()):
        raise ContractError("rendered G_rec parent-bank receipt is invalid")
    if parent_receipt.get("config") != grec_provenance:
        raise ContractError("G_rec parent bank and scene table use different config")
    scenes = pd.read_parquet(scene_table_path)
    parents = pd.read_parquet(parent_manifest_path)
    if (
        len(scenes) != scene_receipt.get("rows")
        or len(parents) != parent_receipt.get("rows")
        or scenes["scene_id"].astype(str).duplicated().any()
        or parents["scene_id"].astype(str).duplicated().any()
        or set(scenes["scene_id"].astype(str)) != set(parents["scene_id"].astype(str))
    ):
        raise ContractError("G_rec parent bank does not exactly cover scene table")
    required_radii = np.asarray(
        parent_contract["required_radii_arcsec"], dtype=np.float64
    )
    observed_radii = np.sort(scenes["theta_e_arcsec"].unique().astype(float))
    if not np.array_equal(observed_radii, required_radii):
        raise ContractError("G_rec radius population differs from registered parent")
    for row in scenes.to_dict(orient="records"):
        verify_grec_scene_identity(row)
    host_ancestry = final_hosts[host_ancestry_columns].copy()
    scene_hosts = scenes.merge(
        host_ancestry,
        on="host_galaxy_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_authority"),
    )
    for column in (
        "host_spatial_group_id",
        "host_spatial_coordinate_id",
        "host_parent_id",
        "host_pixel_sha256",
    ):
        authority_column = f"{column}_authority"
        if (
            not scene_hosts[column]
            .astype(str)
            .equals(scene_hosts[authority_column].astype(str))
        ):
            raise ContractError(f"G_rec scene/host ancestry drift: {column}")
        scene_hosts = scene_hosts.drop(columns=authority_column)
    if scene_hosts["host_physical_group_id"].isna().any():
        raise ContractError("G_rec scenes lack physical-group ancestry")
    authority = {
        "final_host_manifest_sha256": host_sha,
        "final_host_receipt_sha256": sha256_file(final_host_receipt_path),
        "final_selection_manifest_sha256": parent_contract[
            "final_selection_manifest"
        ]["sha256"],
        "scene_table_sha256": scene_sha,
        "scene_receipt_sha256": sha256_file(scene_receipt_path),
        "parent_manifest_sha256": parent_sha,
        "parent_receipt_sha256": sha256_file(parent_receipt_path),
        "grec_config": grec_provenance,
    }
    return scene_hosts, authority


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")
