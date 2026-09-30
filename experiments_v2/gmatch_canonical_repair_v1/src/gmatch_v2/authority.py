"""Direct, score-blind adapters for immutable G_match authorities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    ContractError,
    array_sha256,
    authority_paths,
    canonical_json_sha256,
    canonical_row_sha256,
    frame_content_sha256,
    geographic_spatial_key,
    relative_to_root,
    require_columns,
    require_nonempty,
    sha256_file,
    stable_digest,
)
from .schemas import assert_context_schema

COVARIATES = (
    "observed_zmag",
    "sersic_n",
    "axis_ratio_q",
    "psf_fwhm_r_arcsec",
    "depth5_r_mag",
)
RAW_TRACTOR = (
    "flux_z",
    "sersic",
    "shape_e1",
    "shape_e2",
    "psfsize_r",
    "psfdepth_r",
)


@dataclass(frozen=True)
class AuthorityBundle:
    root: Path
    paths: dict[str, Path]
    source_bindings: dict[str, str]
    reference: pd.DataFrame
    host: pd.DataFrame
    host_tractor: pd.DataFrame
    parent_theta15: pd.DataFrame
    authority_digest: str
    detector_exclusion: dict[str, Any]
    pixel_verification: dict[str, int]


def derive_covariates(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive only the five registered catalogue covariates, without clipping."""
    require_columns(frame, RAW_TRACTOR, context="raw Tractor table")
    result = frame.copy()
    flux_z = pd.to_numeric(result["flux_z"], errors="coerce").to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result["observed_zmag"] = np.where(
            flux_z > 0.0, 22.5 - 2.5 * np.log10(flux_z), np.nan
        )
    result["sersic_n"] = pd.to_numeric(result["sersic"], errors="coerce")
    e1 = pd.to_numeric(result["shape_e1"], errors="coerce").to_numpy(float)
    e2 = pd.to_numeric(result["shape_e2"], errors="coerce").to_numpy(float)
    ellipticity = np.hypot(e1, e2)
    axis_ratio = (1.0 - ellipticity) / (1.0 + ellipticity)
    invalid = ~np.isfinite(axis_ratio) | (axis_ratio < 0.0) | (axis_ratio > 1.0)
    axis_ratio[invalid] = np.nan
    result["axis_ratio_q"] = axis_ratio
    result["psf_fwhm_r_arcsec"] = pd.to_numeric(result["psfsize_r"], errors="coerce")
    psfdepth = pd.to_numeric(result["psfdepth_r"], errors="coerce").to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result["depth5_r_mag"] = np.where(
            psfdepth > 0.0,
            22.5 - 2.5 * np.log10(5.0 / np.sqrt(psfdepth)),
            np.nan,
        )
    return result


def _load_receipt(path: Path, status: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != status:
        raise ContractError(f"{path.name} has unexpected status")
    return payload


def _row_hash(frame: pd.DataFrame) -> list[str]:
    columns = list(frame.columns)
    return [
        canonical_row_sha256(row, columns) for row in frame.to_dict(orient="records")
    ]


def _float_identity(value: object) -> str:
    number = float(value)
    if not np.isfinite(number):
        raise ContractError("identity-defining float is not finite")
    return number.hex()


def _validate_reference(
    root: Path,
    paths: dict[str, Path],
    payload_root: Path,
) -> tuple[pd.DataFrame, int]:
    roster = pd.read_parquet(paths["reference_roster"])
    payloads = pd.read_parquet(paths["reference_payload_manifest"])
    tractor = pd.read_parquet(paths["reference_tractor_manifest"])
    for frame, context in (
        (roster, "input.reference_roster"),
        (payloads, "input.reference_payload_manifest"),
        (tractor, "input.reference_tractor_manifest"),
    ):
        assert_context_schema(frame, context=context)
        require_nonempty(frame, {"galaxy_id"}, context=context)
    required_roster = (
        "galaxy_id",
        "healpix_128",
        "physical_group_id",
        "spatial_component_id",
        "endpoint_role",
        "final_lensing_status",
        "split",
    )
    require_columns(roster, required_roster, context="reference roster")
    require_columns(
        payloads,
        {"galaxy_id", "payload_sha256", "local_relpath"},
        context="reference payload manifest",
    )
    require_columns(
        tractor,
        {"galaxy_id", "payload_sha256", "psfsize_g", "psfsize_z", *RAW_TRACTOR},
        context="reference Tractor manifest",
    )
    ids = set(roster["galaxy_id"].astype(str))
    if (
        len(roster) != 43
        or len(ids) != 43
        or set(payloads["galaxy_id"].astype(str)) != ids
        or set(tractor["galaxy_id"].astype(str)) != ids
    ):
        raise ContractError("reference authorities do not contain the exact same 43")
    if (
        not roster["endpoint_role"].eq("terminal_confirmed").all()
        or not roster["final_lensing_status"].eq("confirmed_lens").all()
        or not roster["split"].eq("test").all()
    ):
        raise ContractError("reference roster status has drifted")
    payload_columns = ["galaxy_id", "payload_sha256", "local_relpath"]
    merged = (
        roster[list(required_roster)]
        .merge(
            payloads[payload_columns],
            on="galaxy_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            tractor[
                ["galaxy_id", "payload_sha256", "psfsize_g", "psfsize_z", *RAW_TRACTOR]
            ],
            on="galaxy_id",
            how="inner",
            validate="one_to_one",
            suffixes=("", "_tractor"),
        )
    )
    if not merged["payload_sha256"].eq(merged["payload_sha256_tractor"]).all():
        raise ContractError("reference Tractor rows changed payload identity")
    pixel_hashes: list[str] = []
    pixel_relpaths: list[str] = []
    for row in merged.itertuples(index=False):
        pixel_path = payload_root / str(row.local_relpath)
        if sha256_file(pixel_path) != str(row.payload_sha256):
            raise ContractError(f"reference payload bytes changed: {row.galaxy_id}")
        with np.load(pixel_path, allow_pickle=False) as archive:
            if "cutout" not in archive.files:
                raise ContractError(f"reference payload lacks cutout: {row.galaxy_id}")
            pixels = np.asarray(archive["cutout"])
        if (
            pixels.shape != (101, 101, 3)
            or pixels.dtype != np.float32
            or not np.isfinite(pixels).all()
        ):
            raise ContractError(f"reference pixels are invalid: {row.galaxy_id}")
        pixel_hashes.append(array_sha256(pixels))
        pixel_relpaths.append(relative_to_root(pixel_path, root))
    adapter = derive_covariates(merged)
    adapter["geographic_spatial_key"] = adapter["healpix_128"].map(
        geographic_spatial_key
    )
    adapter["pixel_relpath"] = pixel_relpaths
    adapter["pixel_file_sha256"] = adapter["payload_sha256"].astype(str)
    adapter["pixel_sha256"] = pixel_hashes
    adapter["pixel_array_key"] = "cutout"
    adapter["pixel_layout"] = "HWC"
    adapter["pixel_dtype"] = "float32"
    adapter["pixel_shape"] = "101x101x3"
    adapter["psf_id"] = [
        "reference_psf_"
        + stable_digest(
            "reference_psf_v1",
            galaxy_id,
            _float_identity(g),
            _float_identity(r),
            _float_identity(z),
        )[:40]
        for galaxy_id, g, r, z in adapter[
            ["galaxy_id", "psfsize_g", "psfsize_r", "psfsize_z"]
        ].itertuples(index=False)
    ]
    adapter["render_parameter_id"] = [
        "observed_reference_render_"
        + stable_digest("observed_reference_render_v1", galaxy_id, payload_hash)[:40]
        for galaxy_id, payload_hash in adapter[
            ["galaxy_id", "payload_sha256"]
        ].itertuples(index=False)
    ]
    adapter["reference_roster_sha256"] = sha256_file(paths["reference_roster"])
    adapter["reference_payload_manifest_sha256"] = sha256_file(
        paths["reference_payload_manifest"]
    )
    adapter["reference_tractor_manifest_sha256"] = sha256_file(
        paths["reference_tractor_manifest"]
    )
    adapter = adapter.drop(columns="payload_sha256_tractor")
    adapter["adapter_row_sha256"] = _row_hash(adapter)
    adapter = adapter.sort_values("galaxy_id", ignore_index=True)
    assert_context_schema(adapter, context="intermediate.reference_adapter")
    return adapter, len(pixel_hashes)


def _validate_hosts(
    root: Path,
    paths: dict[str, Path],
    *,
    verify_pixels: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    roles = pd.read_parquet(paths["host_role_proposal"])
    selection = pd.read_parquet(paths["final_grec_selection"])
    hosts = pd.read_parquet(paths["grec_host_manifest"])
    tractor = pd.read_parquet(paths["host_tractor_manifest"])
    for frame, context in (
        (roles, "input.host_role_proposal"),
        (selection, "input.final_grec_selection"),
        (hosts, "input.grec_host_manifest"),
        (tractor, "input.host_tractor_manifest"),
    ):
        assert_context_schema(frame, context=context)
    require_columns(
        roles,
        {
            "galaxy_id",
            "payload_sha256",
            "evaluation_spatial_group_id",
            "proposed_role",
        },
        context="host role proposal",
    )
    require_columns(
        selection,
        {
            "galaxy_id",
            "payload_sha256",
            "evaluation_spatial_group_id",
            "healpix_128",
            "physical_group_id",
            "final_role",
            "directly_prohibited",
            "component_prohibited",
            "development_healpix_contact",
            "development_38arcsec_contact",
            "local_relpath",
        },
        context="final G_rec selection",
    )
    require_columns(
        hosts,
        {
            "host_galaxy_id",
            "host_parent_id",
            "host_pixel_path",
            "host_payload_sha256",
            "host_pixel_sha256",
            "host_spatial_group_id",
            "host_store_path",
        }
        - {"host_store_path"},
        context="G_rec hosts",
    )
    require_columns(
        tractor,
        {
            "galaxy_id",
            "payload_sha256",
            "flux_z",
            "sersic",
            "shape_e1",
            "shape_e2",
            "psfsize_r_tractor",
            "psfdepth_r_tractor",
        },
        context="host Tractor manifest",
    )
    if any(len(frame) != 4500 for frame in (selection, hosts, tractor)):
        raise ContractError("a final G_rec authority is not exactly 4,500 rows")
    for frame, column in (
        (selection, "galaxy_id"),
        (hosts, "host_galaxy_id"),
        (tractor, "galaxy_id"),
    ):
        if frame[column].astype(str).nunique() != 4500:
            raise ContractError("a final G_rec authority has duplicate hosts")
    role_hosts = roles.loc[
        roles["proposed_role"].eq("experiment_host"),
        [
            "galaxy_id",
            "payload_sha256",
            "evaluation_spatial_group_id",
            "proposed_role",
        ],
    ]
    joined = hosts.merge(
        selection[
            [
                "galaxy_id",
                "payload_sha256",
                "evaluation_spatial_group_id",
                "healpix_128",
                "physical_group_id",
                "final_role",
                "directly_prohibited",
                "component_prohibited",
                "development_healpix_contact",
                "development_38arcsec_contact",
                "local_relpath",
            ]
        ],
        left_on="host_galaxy_id",
        right_on="galaxy_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_selection"),
    ).merge(
        role_hosts,
        on="galaxy_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_role"),
    )
    if len(joined) != 4500:
        raise ContractError("G_rec hosts are not the exact experiment-host subset")
    checks = (
        ("host_payload_sha256", "payload_sha256"),
        ("payload_sha256", "payload_sha256_role"),
        ("host_spatial_group_id", "evaluation_spatial_group_id"),
        ("evaluation_spatial_group_id", "evaluation_spatial_group_id_role"),
        ("final_role", "proposed_role"),
    )
    for left, right in checks:
        if not joined[left].astype(str).eq(joined[right].astype(str)).all():
            raise ContractError(f"host authority mismatch: {left} != {right}")
    exclusion_flags = (
        "directly_prohibited",
        "component_prohibited",
        "development_healpix_contact",
        "development_38arcsec_contact",
    )
    if joined[list(exclusion_flags)].astype(bool).any().any():
        raise ContractError("a final G_rec host contacts detector development")
    verified = 0
    for row in joined.itertuples(index=False):
        pixel_path = Path(str(row.host_pixel_path))
        expected_path = (
            root / "evaluation_dataset_prep/candidate_payloads" / str(row.local_relpath)
        )
        if pixel_path.resolve() != expected_path.resolve():
            raise ContractError(f"host path changed: {row.host_galaxy_id}")
        if verify_pixels:
            if sha256_file(pixel_path) != str(row.host_payload_sha256):
                raise ContractError(f"host payload bytes changed: {row.host_galaxy_id}")
            with np.load(pixel_path, allow_pickle=False) as archive:
                pixels = np.asarray(archive["cutout"])
            if (
                pixels.shape != (101, 101, 3)
                or pixels.dtype != np.float32
                or not np.isfinite(pixels).all()
                or array_sha256(pixels) != str(row.host_pixel_sha256)
            ):
                raise ContractError(f"host array bytes changed: {row.host_galaxy_id}")
            verified += 1
    host_adapter = pd.DataFrame(
        {
            "galaxy_id": joined["host_galaxy_id"].astype(str),
            "proposed_role": joined["proposed_role"].astype(str),
            "final_role": joined["final_role"].astype(str),
            "host_parent_id": joined["host_parent_id"].astype(str),
            "host_physical_group_id": joined["physical_group_id"].astype(str),
            "host_spatial_group_id": joined["host_spatial_group_id"].astype(str),
            "host_healpix_128": joined["healpix_128"].astype("int64"),
            "geographic_spatial_key": joined["healpix_128"].map(
                geographic_spatial_key
            ),
            "host_pixel_relpath": [
                relative_to_root(Path(path), root) for path in joined["host_pixel_path"]
            ],
            "host_payload_sha256": joined["host_payload_sha256"].astype(str),
            "host_pixel_sha256": joined["host_pixel_sha256"].astype(str),
            "host_array_key": "cutout",
            "host_layout": "HWC",
            "host_dtype": "float32",
            "host_shape": "101x101x3",
            "source_host_manifest_sha256": sha256_file(paths["grec_host_manifest"]),
            "source_selection_manifest_sha256": sha256_file(
                paths["final_grec_selection"]
            ),
            "source_role_manifest_sha256": sha256_file(paths["host_role_proposal"]),
        }
    )
    host_adapter["adapter_row_sha256"] = _row_hash(host_adapter)
    host_authority_columns = [
        "galaxy_id",
        "proposed_role",
        "host_payload_sha256",
        "host_pixel_sha256",
        "host_physical_group_id",
        "host_spatial_group_id",
    ]
    # Some raw Tractor exports carry role-like convenience columns. Discard
    # those copies and attach the role only from the direct final authorities.
    tractor_for_join = tractor.drop(
        columns=[
            column
            for column in host_authority_columns
            if column != "galaxy_id" and column in tractor.columns
        ]
    )
    tractor_joined = tractor_for_join.merge(
        host_adapter[host_authority_columns],
        on="galaxy_id",
        how="inner",
        validate="one_to_one",
    )
    if (
        not tractor_joined["payload_sha256"]
        .astype(str)
        .eq(tractor_joined["host_payload_sha256"].astype(str))
        .all()
    ):
        raise ContractError("host Tractor payload hashes differ from final hosts")
    raw = pd.DataFrame(
        {
            "galaxy_id": tractor_joined["galaxy_id"].astype(str),
            "proposed_role": tractor_joined["proposed_role"].astype(str),
            "payload_sha256": tractor_joined["payload_sha256"].astype(str),
            "host_pixel_sha256": tractor_joined["host_pixel_sha256"].astype(str),
            "host_physical_group_id": tractor_joined["host_physical_group_id"].astype(
                str
            ),
            "host_spatial_group_id": tractor_joined["host_spatial_group_id"].astype(
                str
            ),
            "flux_z": tractor_joined["flux_z"],
            "sersic": tractor_joined["sersic"],
            "shape_e1": tractor_joined["shape_e1"],
            "shape_e2": tractor_joined["shape_e2"],
            "psfsize_r": tractor_joined["psfsize_r_tractor"],
            "psfdepth_r": tractor_joined["psfdepth_r_tractor"],
        }
    )
    raw = derive_covariates(raw)
    source_columns = list(tractor.columns)
    source_lookup = tractor.set_index(tractor["galaxy_id"].astype(str))
    raw["source_tractor_row_sha256"] = [
        canonical_row_sha256(source_lookup.loc[galaxy_id].to_dict(), source_columns)
        for galaxy_id in raw["galaxy_id"]
    ]
    raw["source_host_tractor_manifest_sha256"] = sha256_file(
        paths["host_tractor_manifest"]
    )
    raw["adapter_row_sha256"] = _row_hash(raw)
    host_adapter = host_adapter.sort_values("galaxy_id", ignore_index=True)
    raw = raw.sort_values("galaxy_id", ignore_index=True)
    assert_context_schema(host_adapter, context="intermediate.host_adapter")
    assert_context_schema(raw, context="intermediate.host_tractor_adapter")
    return host_adapter, raw, verified


def _validate_parents(
    paths: dict[str, Path],
    host: pd.DataFrame,
) -> pd.DataFrame:
    scenes = pd.read_parquet(paths["grec_scene_manifest"])
    parents = pd.read_parquet(paths["grec_parent_manifest"])
    assert_context_schema(scenes, context="input.grec_scene_manifest")
    assert_context_schema(parents, context="input.grec_parent_manifest")
    if (
        len(scenes) != 49500
        or len(parents) != 49500
        or scenes["grec_row_id"].astype(str).nunique() != 49500
        or parents["grec_row_id"].astype(str).nunique() != 49500
    ):
        raise ContractError("G_rec scene/parent authority is not exactly 49,500 rows")
    try:
        pd.testing.assert_frame_equal(
            scenes.sort_values("grec_row_id", ignore_index=True),
            parents[list(scenes.columns)].sort_values("grec_row_id", ignore_index=True),
            check_exact=True,
        )
    except AssertionError as error:
        raise ContractError("G_rec parent rows differ from frozen scenes") from error
    required = {
        "grec_row_id",
        "scene_id",
        "host_galaxy_id",
        "host_parent_id",
        "host_pixel_path",
        "host_payload_sha256",
        "host_pixel_sha256",
        "host_store_path",
        "host_store_file_sha256",
        "host_spatial_group_id",
        "theta_e_arcsec",
        "source_parameter_id",
        "source_nuisance_id",
        "lens_parameter_id",
        "lens_nuisance_id",
        "base_draw_key",
        "render_config_id",
        "render_contract_sha256",
        "parameter_row_sha256",
        "psf_fwhm_g_arcsec",
        "psf_fwhm_r_arcsec",
        "psf_fwhm_z_arcsec",
        "source_flux_nmgy_g",
        "source_flux_nmgy_r",
        "source_flux_nmgy_z",
        "parent_pixel_path",
        "parent_artifact_sha256",
        "expectation_pixel_sha256",
        "injection_expectation_sha256",
    }
    require_nonempty(parents, required, context="G_rec parent manifest")
    selected = parents.loc[
        pd.to_numeric(parents["theta_e_arcsec"], errors="coerce").eq(1.5)
    ].copy()
    if (
        len(selected) != 4500
        or selected["host_galaxy_id"].astype(str).nunique() != 4500
        or selected["grec_row_id"].astype(str).nunique() != 4500
    ):
        raise ContractError("theta_E=1.5 does not have exactly one scene per host")
    host_lookup = host.set_index(host["galaxy_id"].astype(str))
    if set(selected["host_galaxy_id"].astype(str)) != set(host_lookup.index):
        raise ContractError("theta_E=1.5 does not cover the exact final hosts")
    selected["host_physical_group_id"] = (
        selected["host_galaxy_id"]
        .astype(str)
        .map(host_lookup["host_physical_group_id"])
    )
    selected["proposed_role"] = (
        selected["host_galaxy_id"].astype(str).map(host_lookup["proposed_role"])
    )
    for parent_column, host_column in (
        ("host_parent_id", "host_parent_id"),
        ("host_payload_sha256", "host_payload_sha256"),
        ("host_pixel_sha256", "host_pixel_sha256"),
        ("host_spatial_group_id", "host_spatial_group_id"),
    ):
        expected = selected["host_galaxy_id"].astype(str).map(host_lookup[host_column])
        if not selected[parent_column].astype(str).eq(expected.astype(str)).all():
            raise ContractError(f"theta_E=1.5 changed {parent_column}")
    selected["native_source_parameter_id"] = selected["source_parameter_id"].astype(str)
    selected["native_lens_parameter_id"] = selected["lens_parameter_id"].astype(str)
    selected["source_parameter_id"] = selected["source_nuisance_id"].astype(str)
    selected["lens_parameter_id"] = selected["lens_nuisance_id"].astype(str)
    selected["psf_id"] = [
        "grec_psf_"
        + stable_digest(
            "grec_psf_v1",
            galaxy_id,
            _float_identity(g),
            _float_identity(r),
            _float_identity(z),
        )[:40]
        for galaxy_id, g, r, z in selected[
            [
                "host_galaxy_id",
                "psf_fwhm_g_arcsec",
                "psf_fwhm_r_arcsec",
                "psf_fwhm_z_arcsec",
            ]
        ].itertuples(index=False)
    ]
    selected["rendering_id"] = selected["base_draw_key"].astype(str)
    selected["render_parameter_id"] = selected["base_draw_key"].astype(str)
    selected["random_draw_id"] = selected["base_draw_key"].astype(str)
    selected["intrinsic_flux_g_nmgy"] = selected["source_flux_nmgy_g"]
    selected["intrinsic_flux_r_nmgy"] = selected["source_flux_nmgy_r"]
    selected["intrinsic_flux_z_nmgy"] = selected["source_flux_nmgy_z"]
    selected["source_flux_parameter_sha256"] = [
        canonical_json_sha256(
            {
                "g": _float_identity(g),
                "r": _float_identity(r),
                "z": _float_identity(z),
            }
        )
        for g, r, z in selected[
            ["source_flux_nmgy_g", "source_flux_nmgy_r", "source_flux_nmgy_z"]
        ].itertuples(index=False)
    ]
    selected["parent_artifact_file_sha256"] = selected["parent_artifact_sha256"].astype(
        str
    )
    selected["source_grec_scene_manifest_sha256"] = sha256_file(
        paths["grec_scene_manifest"]
    )
    selected["source_grec_parent_manifest_sha256"] = sha256_file(
        paths["grec_parent_manifest"]
    )
    selected["adapter_row_sha256"] = _row_hash(selected)
    selected = selected.sort_values("grec_row_id", ignore_index=True)
    assert_context_schema(selected, context="intermediate.parent_theta15_adapter")
    return selected


def _validate_receipts(
    paths: dict[str, Path],
) -> None:
    reference_summary = _load_receipt(
        paths["reference_summary"],
        "payload_validated_reference_cohort_pending_spatial_and_support_checks",
    )
    tractor_summary = _load_receipt(
        paths["reference_tractor_summary"],
        "payload_and_tractor_validated_reference_cohort",
    )
    selection = _load_receipt(
        paths["final_grec_selection_receipt"],
        "final_experiment_host_selection_approved",
    )
    hosts = _load_receipt(
        paths["grec_host_receipt"], "final_pre_score_grec_hosts_frozen"
    )
    scenes = _load_receipt(
        paths["grec_scene_receipt"], "grec_scene_table_frozen_pre_score"
    )
    render = _load_receipt(
        paths["grec_render_receipt"], "grec_parent_bank_frozen_pre_score"
    )
    resume = _load_receipt(
        paths["grec_render_resume_audit"], "grec_render_resume_complete"
    )
    _load_receipt(
        paths["grec_quadrature_amendment"],
        "score_blind_quadrature_extension_authorized",
    )
    _load_receipt(paths["common_threshold_receipt"], "common_threshold_frozen")
    _load_receipt(
        paths["governance_amendment"],
        "governance_provenance_amended_no_scientific_change",
    )
    _load_receipt(
        paths["negative_role_receipt"],
        "latest_test_negative_roles_frozen_pre_score",
    )
    links = (
        (reference_summary, "manifest_sha256", "reference_roster"),
        (
            reference_summary,
            "payload_validated_manifest_sha256",
            "reference_payload_manifest",
        ),
        (tractor_summary, "manifest_sha256", "reference_tractor_manifest"),
        (selection, "selection_manifest_sha256", "final_grec_selection"),
        (selection, "role_proposal_sha256", "host_role_proposal"),
        (selection, "tractor_manifest_sha256", "host_tractor_manifest"),
        (hosts, "host_manifest_sha256", "grec_host_manifest"),
        (scenes, "scene_table_sha256", "grec_scene_manifest"),
        (render, "parent_manifest_sha256", "grec_parent_manifest"),
        (render, "scene_table_sha256", "grec_scene_manifest"),
        (resume, "parent_manifest_sha256", "grec_parent_manifest"),
    )
    for receipt, field, authority in links:
        if receipt.get(field) != sha256_file(paths[authority]):
            raise ContractError(f"receipt has wrong {field}: {authority}")
    if any(
        receipt.get("detector_scores_used") is not False
        for receipt in (selection, hosts, scenes)
    ):
        raise ContractError("source preparation does not prove score blindness")
    if reference_summary.get("selection_used_detector_scores") is not False:
        raise ContractError("reference selection does not prove score blindness")
    if tractor_summary.get("score_columns_used") is not False:
        raise ContractError("reference matching table was not score-blind")


def _detector_exclusion(
    paths: dict[str, Path],
    reference: pd.DataFrame,
    hosts: pd.DataFrame,
) -> dict[str, Any]:
    train = pd.read_parquet(paths["train_manifest"])
    validation = pd.read_parquet(paths["validation_manifest"])
    test = pd.read_parquet(paths["test_manifest"])
    for frame, context in (
        (train, "input.train_manifest"),
        (validation, "input.validation_manifest"),
        (test, "input.test_manifest"),
    ):
        assert_context_schema(frame, context=context)
        require_columns(
            frame,
            {"galaxy_id", "physical_group_id", "split"},
            context=context,
        )
    development = pd.concat([train, validation], ignore_index=True)
    development_ids = set(development["galaxy_id"].astype(str))
    development_groups = set(development["physical_group_id"].astype(str))
    reference_ids = set(reference["galaxy_id"].astype(str))
    host_ids = set(hosts["galaxy_id"].astype(str))
    reference_groups = set(reference["physical_group_id"].astype(str))
    host_groups = set(hosts["host_physical_group_id"].astype(str))
    counts = {
        "reference_id_overlap_train_validation": len(reference_ids & development_ids),
        "host_id_overlap_train_validation": len(host_ids & development_ids),
        "reference_physical_group_overlap_train_validation": len(
            reference_groups & development_groups
        ),
        "host_physical_group_overlap_train_validation": len(
            host_groups & development_groups
        ),
        "references_present_in_test": len(
            reference_ids & set(test["galaxy_id"].astype(str))
        ),
    }
    if any(counts[key] for key in counts if "overlap" in key):
        raise ContractError(f"detector-development exclusion failed: {counts}")
    if counts["references_present_in_test"] != 43:
        raise ContractError("the 43 references are not all in the fixed test manifest")
    return {
        **counts,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "passed": True,
    }


def load_authorities(
    config: dict[str, Any],
    *,
    root: Path,
    verify_all_host_pixels: bool = True,
) -> AuthorityBundle:
    """Authenticate direct sources and construct G_match-specific adapters."""
    paths = authority_paths(config, root)
    _validate_receipts(paths)
    payload_root = root / str(config["paths"]["reference_payload_root"])
    reference, reference_pixels = _validate_reference(root, paths, payload_root)
    host, host_tractor, host_pixels = _validate_hosts(
        root, paths, verify_pixels=verify_all_host_pixels
    )
    parent = _validate_parents(paths, host)
    exclusion = _detector_exclusion(paths, reference, host)
    source_bindings = {name: sha256_file(path) for name, path in paths.items()}
    adapter_digests = {
        "reference": frame_content_sha256(
            reference,
            columns=list(reference.columns),
            sort_by=["galaxy_id"],
        ),
        "host": frame_content_sha256(
            host,
            columns=list(host.columns),
            sort_by=["galaxy_id"],
        ),
        "host_tractor": frame_content_sha256(
            host_tractor,
            columns=list(host_tractor.columns),
            sort_by=["galaxy_id"],
        ),
        "parent_theta15": frame_content_sha256(
            parent,
            columns=list(parent.columns),
            sort_by=["grec_row_id"],
        ),
    }
    authority_digest = canonical_json_sha256(
        {
            "source_bindings": source_bindings,
            "adapter_content": adapter_digests,
            "detector_exclusion": exclusion,
        }
    )
    return AuthorityBundle(
        root=root.resolve(),
        paths=paths,
        source_bindings=source_bindings,
        reference=reference,
        host=host,
        host_tractor=host_tractor,
        parent_theta15=parent,
        authority_digest=authority_digest,
        detector_exclusion=exclusion,
        pixel_verification={
            "reference_payloads_reopened": reference_pixels,
            "host_payloads_reopened": host_pixels,
        },
    )
