"""Immutable end-to-end G_match v2 preparation, stopping before features."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .authority import AuthorityBundle, load_authorities
from .contracts import (
    ContractError,
    array_sha256,
    atomic_write_json,
    atomic_write_parquet,
    canonical_row_sha256,
    load_config,
    require_nonempty,
    sha256_file,
    source_tree_sha256,
)
from .matching import MatchResult, match
from .schemas import ALLOWLIST_CONTRACT_SHA256, assert_context_schema

OUTPUT_FILES = {
    "reference_adapter": "reference_adapter.parquet",
    "host_adapter": "host_adapter.parquet",
    "host_tractor_adapter": "host_tractor_adapter.parquet",
    "parent_theta15_adapter": "parent_theta15_adapter.parquet",
    "matches": "matches.parquet",
    "reference_disposition": "reference_disposition.parquet",
    "host_disposition": "host_disposition.parquet",
    "balance": "balance_diagnostics.parquet",
    "scaler_comparison": "scaler_policy_comparison.parquet",
    "pair_scene_links": "pair_scene_links.parquet",
    "pair_plan": "pair_plan.parquet",
    "rendered_pairs": "rendered_pairs_zero_copy.parquet",
    "analysis_manifest": "gmatch_analysis_manifest.parquet",
}


def write_config_seal(
    *,
    package_root: Path,
    config_path: Path,
    documentation_path: Path,
) -> dict[str, Any]:
    """Seal adopted config, implementation, lock, and scientific authority text."""
    seal_path = package_root / "config/CONFIG_SEAL.json"
    if seal_path.exists():
        raise ContractError("configuration seal already exists")
    load_config(config_path)
    tree_hash, source_hashes = source_tree_sha256(package_root)
    lock_path = package_root / "uv.lock"
    if not lock_path.is_file():
        raise ContractError("dependency lock is missing")
    payload = {
        "schema_version": 1,
        "status": "gmatch_v2_config_adopted_and_sealed_pre_feature_revision_2",
        "study_id": "stronglens_gmatch_v2_final_preparation",
        "config_path": config_path.relative_to(package_root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "source_tree_sha256": tree_hash,
        "source_file_sha256": source_hashes,
        "dependency_lock_path": lock_path.relative_to(package_root).as_posix(),
        "dependency_lock_sha256": sha256_file(lock_path),
        "authority_document_path": documentation_path.as_posix(),
        "authority_document_sha256": sha256_file(documentation_path),
        "strict_schema_allowlist_contract_sha256": ALLOWLIST_CONTRACT_SHA256,
        "scaler_policy": ("component_wise_all_finite_values_exact_authorized_roster"),
        "matching_candidate_policy": "all_five_covariates_finite",
        "nonfinite_host_policy": (
            "retain_in_scaler_authority_accounting_mark_ineligible_no_imputation"
        ),
        "detector_scores_authorized": False,
        "feature_extraction_authorized": False,
        "probe_fitting_authorized": False,
        "bootstrap_authorized": False,
    }
    atomic_write_json(payload, seal_path)
    return payload


def validate_config_seal(
    *,
    package_root: Path,
    config_path: Path,
    documentation_path: Path,
) -> dict[str, Any]:
    seal_path = package_root / "config/CONFIG_SEAL.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    tree_hash, source_hashes = source_tree_sha256(package_root)
    expected = {
        "status": "gmatch_v2_config_adopted_and_sealed_pre_feature_revision_2",
        "config_sha256": sha256_file(config_path),
        "source_tree_sha256": tree_hash,
        "source_file_sha256": source_hashes,
        "dependency_lock_sha256": sha256_file(package_root / "uv.lock"),
        "authority_document_sha256": sha256_file(documentation_path),
        "strict_schema_allowlist_contract_sha256": ALLOWLIST_CONTRACT_SHA256,
        "scaler_policy": ("component_wise_all_finite_values_exact_authorized_roster"),
        "matching_candidate_policy": "all_five_covariates_finite",
        "nonfinite_host_policy": (
            "retain_in_scaler_authority_accounting_mark_ineligible_no_imputation"
        ),
        "detector_scores_authorized": False,
        "feature_extraction_authorized": False,
        "probe_fitting_authorized": False,
        "bootstrap_authorized": False,
    }
    for field, value in expected.items():
        if seal.get(field) != value:
            raise ContractError(f"configuration seal has wrong {field}")
    return seal


def _row_hashes(frame: pd.DataFrame) -> list[str]:
    columns = list(frame.columns)
    return [
        canonical_row_sha256(row, columns) for row in frame.to_dict(orient="records")
    ]


def _write_frame(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    atomic_write_parquet(frame, path)
    return {"path": path.name, "sha256": sha256_file(path), "rows": len(frame)}


def _build_pair_scene_links(
    bundle: AuthorityBundle,
    matched: MatchResult,
    *,
    matches_sha256: str,
) -> pd.DataFrame:
    parent = bundle.parent_theta15.set_index(
        bundle.parent_theta15["host_galaxy_id"].astype(str)
    )
    rows: list[dict[str, Any]] = []
    for row in matched.matches.itertuples(index=False):
        target = str(row.target_galaxy_id)
        if target not in parent.index:
            raise ContractError(f"matched target lacks theta_E=1.5 parent: {target}")
        scene = parent.loc[target]
        rows.append(
            {
                "pair_id": str(row.pair_id),
                "reference_id": str(row.reference_id),
                "target_galaxy_id": target,
                "grec_row_id": str(scene["grec_row_id"]),
                "scene_id": str(scene["scene_id"]),
                "theta_e_arcsec": float(scene["theta_e_arcsec"]),
                "selection_rule": "exact_radius_then_lexicographic_grec_row_id",
                "matches_manifest_sha256": matches_sha256,
                "grec_scene_manifest_sha256": bundle.source_bindings[
                    "grec_scene_manifest"
                ],
                "grec_parent_manifest_sha256": bundle.source_bindings[
                    "grec_parent_manifest"
                ],
                "parent_adapter_row_sha256": str(scene["adapter_row_sha256"]),
                "authority_digest": bundle.authority_digest,
            }
        )
    links = pd.DataFrame(rows).sort_values("pair_id", ignore_index=True)
    links["link_row_sha256"] = _row_hashes(links)
    assert_context_schema(links, context="output.pair_scene_links")
    return links


def _build_pair_plan(
    bundle: AuthorityBundle,
    matched: MatchResult,
    links: pd.DataFrame,
    *,
    matches_sha256: str,
    links_sha256: str,
) -> pd.DataFrame:
    parent = bundle.parent_theta15.set_index(
        bundle.parent_theta15["grec_row_id"].astype(str)
    )
    host = bundle.host.set_index(bundle.host["galaxy_id"].astype(str))
    match_lookup = matched.matches.set_index("pair_id")
    rows: list[dict[str, Any]] = []
    for link in links.itertuples(index=False):
        scene = parent.loc[str(link.grec_row_id)]
        selected_host = host.loc[str(link.target_galaxy_id)]
        matched_row = match_lookup.loc[str(link.pair_id)]
        shared = {
            "pair_id": str(link.pair_id),
            "reference_id": str(link.reference_id),
            "target_galaxy_id": str(link.target_galaxy_id),
            "grec_row_id": str(scene["grec_row_id"]),
            "scene_id": str(scene["scene_id"]),
            "theta_e_arcsec": float(scene["theta_e_arcsec"]),
            "proposed_role": str(scene["proposed_role"]),
            "host_parent_id": str(scene["host_parent_id"]),
            "host_physical_group_id": str(scene["host_physical_group_id"]),
            "host_spatial_group_id": str(scene["host_spatial_group_id"]),
            "host_payload_sha256": str(scene["host_payload_sha256"]),
            "host_pixel_sha256": str(scene["host_pixel_sha256"]),
            "host_store_relpath": str(scene["host_store_path"]),
            "host_store_file_sha256": str(scene["host_store_file_sha256"]),
            "source_parameter_id": str(scene["source_parameter_id"]),
            "native_source_parameter_id": str(scene["native_source_parameter_id"]),
            "source_nuisance_id": str(scene["source_nuisance_id"]),
            "lens_parameter_id": str(scene["lens_parameter_id"]),
            "native_lens_parameter_id": str(scene["native_lens_parameter_id"]),
            "lens_nuisance_id": str(scene["lens_nuisance_id"]),
            "psf_id": str(scene["psf_id"]),
            "psf_fwhm_g_arcsec": float(scene["psf_fwhm_g_arcsec"]),
            "psf_fwhm_r_arcsec": float(scene["psf_fwhm_r_arcsec"]),
            "psf_fwhm_z_arcsec": float(scene["psf_fwhm_z_arcsec"]),
            "rendering_id": str(scene["rendering_id"]),
            "render_parameter_id": str(scene["render_parameter_id"]),
            "random_draw_id": str(scene["random_draw_id"]),
            "base_draw_key": str(scene["base_draw_key"]),
            "render_config_id": str(scene["render_config_id"]),
            "render_contract_sha256": str(scene["render_contract_sha256"]),
            "parameter_row_sha256": str(scene["parameter_row_sha256"]),
            "source_flux_policy": "fixed_intrinsic",
            "intrinsic_flux_g_nmgy": float(scene["intrinsic_flux_g_nmgy"]),
            "intrinsic_flux_r_nmgy": float(scene["intrinsic_flux_r_nmgy"]),
            "intrinsic_flux_z_nmgy": float(scene["intrinsic_flux_z_nmgy"]),
            "source_flux_parameter_sha256": str(scene["source_flux_parameter_sha256"]),
            "parent_pixel_relpath": str(scene["parent_pixel_path"]),
            "parent_artifact_file_sha256": str(scene["parent_artifact_file_sha256"]),
            "parent_expectation_pixel_sha256": str(scene["expectation_pixel_sha256"]),
            "injection_expectation_sha256": str(scene["injection_expectation_sha256"]),
            "match_row_sha256": str(matched_row["match_row_sha256"]),
            "parent_adapter_row_sha256": str(scene["adapter_row_sha256"]),
            "pair_scene_links_manifest_sha256": links_sha256,
            "matches_manifest_sha256": matches_sha256,
            "reference_manifest_sha256": bundle.source_bindings["reference_roster"],
            "reference_payload_manifest_sha256": bundle.source_bindings[
                "reference_payload_manifest"
            ],
            "reference_tractor_manifest_sha256": bundle.source_bindings[
                "reference_tractor_manifest"
            ],
            "host_role_manifest_sha256": bundle.source_bindings["host_role_proposal"],
            "final_selection_manifest_sha256": bundle.source_bindings[
                "final_grec_selection"
            ],
            "host_tractor_manifest_sha256": bundle.source_bindings[
                "host_tractor_manifest"
            ],
            "grec_host_manifest_sha256": bundle.source_bindings["grec_host_manifest"],
            "grec_scene_manifest_sha256": bundle.source_bindings["grec_scene_manifest"],
            "grec_parent_manifest_sha256": bundle.source_bindings[
                "grec_parent_manifest"
            ],
            "grec_render_receipt_sha256": bundle.source_bindings["grec_render_receipt"],
            "authority_digest": bundle.authority_digest,
        }
        for arm in ("bare", "injected"):
            if arm == "bare":
                pixel_relpath = str(selected_host["host_pixel_relpath"])
                pixel_file_sha256 = str(selected_host["host_payload_sha256"])
                pixel_sha256 = str(selected_host["host_pixel_sha256"])
                pixel_key, pixel_layout, pixel_shape = (
                    "cutout",
                    "HWC",
                    "101x101x3",
                )
            else:
                pixel_relpath = str(scene["parent_pixel_path"])
                pixel_file_sha256 = str(scene["parent_artifact_file_sha256"])
                pixel_sha256 = str(scene["expectation_pixel_sha256"])
                pixel_key, pixel_layout, pixel_shape = (
                    "expectation_chw",
                    "CHW",
                    "3x101x101",
                )
            rows.append(
                {
                    **shared,
                    "arm": arm,
                    "pixel_relpath": pixel_relpath,
                    "pixel_file_sha256": pixel_file_sha256,
                    "pixel_sha256": pixel_sha256,
                    "pixel_array_key": pixel_key,
                    "pixel_layout": pixel_layout,
                    "pixel_dtype": "float32",
                    "pixel_shape": pixel_shape,
                    "zero_copy": True,
                }
            )
    plan = pd.DataFrame(rows).sort_values(["pair_id", "arm"], ignore_index=True)
    plan["pair_plan_row_sha256"] = _row_hashes(plan)
    assert_context_schema(plan, context="output.pair_plan")
    return plan


def validate_zero_copy_pairs(
    pairs: pd.DataFrame,
    *,
    bundle: AuthorityBundle,
    matches: pd.DataFrame,
    pair_plan_sha256: str,
) -> dict[str, Any]:
    """Reopen every selected bare and injected parent and verify exact bytes."""
    required_shared = {
        "reference_id",
        "target_galaxy_id",
        "grec_row_id",
        "scene_id",
        "theta_e_arcsec",
        "proposed_role",
        "host_parent_id",
        "host_physical_group_id",
        "host_spatial_group_id",
        "host_payload_sha256",
        "host_pixel_sha256",
        "source_parameter_id",
        "native_source_parameter_id",
        "source_nuisance_id",
        "lens_parameter_id",
        "native_lens_parameter_id",
        "lens_nuisance_id",
        "psf_id",
        "rendering_id",
        "render_parameter_id",
        "random_draw_id",
        "source_flux_policy",
        "intrinsic_flux_g_nmgy",
        "intrinsic_flux_r_nmgy",
        "intrinsic_flux_z_nmgy",
        "source_flux_parameter_sha256",
        "parent_artifact_file_sha256",
        "parent_expectation_pixel_sha256",
        "injection_expectation_sha256",
        "match_row_sha256",
        "parent_adapter_row_sha256",
        "authority_digest",
    }
    require_nonempty(
        pairs,
        {
            "pair_id",
            "arm",
            "pixel_relpath",
            "pixel_file_sha256",
            "pixel_sha256",
            "pixel_array_key",
            "pixel_layout",
            *required_shared,
        },
        context="zero-copy pair manifest",
    )
    if any(
        "floor" in str(column).lower()
        and any(
            token in str(column).lower() for token in ("host", "flux", "brightness")
        )
        for column in pairs.columns
    ):
        raise ContractError("zero-copy pairs contain a host flux-floor field")
    if (
        len(pairs) != 2 * len(matches)
        or set(pairs["pair_id"].astype(str)) != set(matches["pair_id"].astype(str))
        or pairs["target_galaxy_id"].astype(str).nunique() != len(matches)
        or not pairs["pair_plan_manifest_sha256"].eq(pair_plan_sha256).all()
    ):
        raise ContractError("zero-copy pairs do not exactly cover frozen matches")
    match_lookup = matches.set_index("pair_id")
    parent_lookup = bundle.parent_theta15.set_index(
        bundle.parent_theta15["grec_row_id"].astype(str)
    )
    bare_files = 0
    injected_files = 0
    exact_host_store_links = 0
    exact_host_plus_injection = 0
    for pair_id, frame in pairs.groupby("pair_id", sort=False):
        if len(frame) != 2 or set(frame["arm"]) != {"bare", "injected"}:
            raise ContractError(f"{pair_id} is not exactly one bare/injected pair")
        for column in required_shared:
            if frame[column].nunique(dropna=False) != 1:
                raise ContractError(f"{pair_id} changed shared identity {column}")
        matched = match_lookup.loc[str(pair_id)]
        if (
            str(frame["reference_id"].iloc[0]) != str(matched["reference_id"])
            or str(frame["target_galaxy_id"].iloc[0])
            != str(matched["target_galaxy_id"])
            or str(frame["match_row_sha256"].iloc[0])
            != str(matched["match_row_sha256"])
        ):
            raise ContractError(f"{pair_id} differs from its frozen match")
        scene = parent_lookup.loc[str(frame["grec_row_id"].iloc[0])]
        parent_identity = {
            "target_galaxy_id": "host_galaxy_id",
            "scene_id": "scene_id",
            "host_parent_id": "host_parent_id",
            "host_physical_group_id": "host_physical_group_id",
            "host_spatial_group_id": "host_spatial_group_id",
            "host_payload_sha256": "host_payload_sha256",
            "host_pixel_sha256": "host_pixel_sha256",
            "source_parameter_id": "source_parameter_id",
            "native_source_parameter_id": "native_source_parameter_id",
            "source_nuisance_id": "source_nuisance_id",
            "lens_parameter_id": "lens_parameter_id",
            "native_lens_parameter_id": "native_lens_parameter_id",
            "lens_nuisance_id": "lens_nuisance_id",
            "psf_id": "psf_id",
            "rendering_id": "rendering_id",
            "render_parameter_id": "render_parameter_id",
            "random_draw_id": "random_draw_id",
            "base_draw_key": "base_draw_key",
            "render_config_id": "render_config_id",
            "render_contract_sha256": "render_contract_sha256",
            "parameter_row_sha256": "parameter_row_sha256",
            "source_flux_parameter_sha256": "source_flux_parameter_sha256",
            "parent_artifact_file_sha256": "parent_artifact_file_sha256",
            "parent_expectation_pixel_sha256": "expectation_pixel_sha256",
            "injection_expectation_sha256": "injection_expectation_sha256",
            "parent_adapter_row_sha256": "adapter_row_sha256",
        }
        for output_column, parent_column in parent_identity.items():
            if str(frame[output_column].iloc[0]) != str(scene[parent_column]):
                raise ContractError(f"{pair_id} changed parent {output_column}")
        if (
            float(frame["theta_e_arcsec"].iloc[0]) != 1.5
            or not frame["proposed_role"].eq("experiment_host").all()
            or not frame["authority_digest"].eq(bundle.authority_digest).all()
        ):
            raise ContractError(f"{pair_id} changed radius, role, or authority")
        bare = frame.loc[frame["arm"].eq("bare")].iloc[0]
        injected = frame.loc[frame["arm"].eq("injected")].iloc[0]
        bare_path = bundle.root / str(bare["pixel_relpath"])
        injected_path = bundle.root / str(injected["pixel_relpath"])
        if (
            sha256_file(bare_path) != str(bare["pixel_file_sha256"])
            or str(bare["pixel_file_sha256"]) != str(bare["host_payload_sha256"])
            or str(bare["pixel_array_key"]) != "cutout"
            or str(bare["pixel_layout"]) != "HWC"
        ):
            raise ContractError(f"{pair_id} bare file binding is invalid")
        with np.load(bare_path, allow_pickle=False) as archive:
            bare_pixels = np.asarray(archive["cutout"])
        if (
            bare_pixels.shape != (101, 101, 3)
            or bare_pixels.dtype != np.float32
            or array_sha256(bare_pixels) != str(bare["host_pixel_sha256"])
            or str(bare["pixel_sha256"]) != str(bare["host_pixel_sha256"])
        ):
            raise ContractError(f"{pair_id} bare HWC array bytes changed")
        bare_files += 1
        if (
            sha256_file(injected_path) != str(injected["pixel_file_sha256"])
            or str(injected["pixel_file_sha256"])
            != str(injected["parent_artifact_file_sha256"])
            or str(injected["pixel_array_key"]) != "expectation_chw"
            or str(injected["pixel_layout"]) != "CHW"
        ):
            raise ContractError(f"{pair_id} injected file binding is invalid")
        with np.load(injected_path, allow_pickle=False) as archive:
            expectation = np.asarray(archive["expectation_chw"])
            injection = np.asarray(archive["injection_expectation_chw"])
        if (
            expectation.shape != (3, 101, 101)
            or expectation.dtype != np.float32
            or injection.shape != (3, 101, 101)
            or injection.dtype != np.float32
            or array_sha256(expectation)
            != str(injected["parent_expectation_pixel_sha256"])
            or str(injected["pixel_sha256"])
            != str(injected["parent_expectation_pixel_sha256"])
            or array_sha256(injection) != str(injected["injection_expectation_sha256"])
        ):
            raise ContractError(f"{pair_id} injected expectation bytes changed")
        injected_files += 1
        host_store_path = bundle.root / str(scene["host_store_path"])
        if sha256_file(host_store_path) != str(scene["host_store_file_sha256"]):
            raise ContractError(f"{pair_id} G_rec raw-host store file changed")
        host_hwc = np.load(host_store_path, allow_pickle=False)
        if not np.array_equal(bare_pixels, host_hwc) or array_sha256(host_hwc) != str(
            scene["host_pixel_sha256"]
        ):
            raise ContractError(f"{pair_id} bare host differs from G_rec raw host")
        exact_host_store_links += 1
        host_chw = np.transpose(host_hwc, (2, 0, 1))
        if not np.array_equal(expectation, host_chw + injection):
            raise ContractError(
                f"{pair_id} injected parent is not exact host plus source"
            )
        exact_host_plus_injection += 1
        if str(injected["parent_artifact_file_sha256"]) != str(
            scene["parent_artifact_sha256"]
        ):
            raise ContractError(f"{pair_id} changed parent artifact identity")
        for column in (
            "intrinsic_flux_g_nmgy",
            "intrinsic_flux_r_nmgy",
            "intrinsic_flux_z_nmgy",
        ):
            if float(frame[column].iloc[0]) != float(scene[column]):
                raise ContractError(f"{pair_id} rescaled {column}")
        if not frame["source_flux_policy"].eq("fixed_intrinsic").all():
            raise ContractError(f"{pair_id} changed intrinsic flux policy")
    return {
        "pair_plan_manifest_sha256": pair_plan_sha256,
        "bare_payload_files_reopened": bare_files,
        "injected_parent_files_reopened": injected_files,
        "bare_to_grec_host_store_exact": exact_host_store_links,
        "parent_host_plus_injection_exact": exact_host_plus_injection,
        "flux_rescaling_applied": False,
        "host_total_flux_floor_applied": False,
    }


def _analysis_manifest(
    bundle: AuthorityBundle,
    matches: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    matches_sha256: str,
    pair_plan_sha256: str,
) -> pd.DataFrame:
    match_by_reference = matches.set_index("reference_id")
    host_by_id = bundle.host.set_index(bundle.host["galaxy_id"].astype(str))
    rows: list[dict[str, Any]] = []
    for reference in bundle.reference.sort_values("galaxy_id").itertuples(index=False):
        reference_id = str(reference.galaxy_id)
        matched = (
            match_by_reference.loc[reference_id]
            if reference_id in match_by_reference.index
            else None
        )
        rows.append(
            {
                "row_id": f"gmatch2:observed:{reference_id}",
                "population": "observed_reference",
                "data_role": "final",
                "pair_id": str(matched["pair_id"]) if matched is not None else "",
                "reference_id": reference_id,
                "target_galaxy_id": (
                    str(matched["target_galaxy_id"])
                    if matched is not None
                    else "not_applicable:unmatched"
                ),
                "host_galaxy_id": "not_applicable:observed",
                "host_parent_id": "not_applicable:observed",
                "proposed_role": "not_applicable:observed",
                "physical_group_id": str(reference.physical_group_id),
                "geographic_spatial_key": str(
                    reference.geographic_spatial_key
                ),
                "spatial_group_id": str(reference.spatial_component_id),
                "source_parameter_id": "not_applicable:observed",
                "native_source_parameter_id": "not_applicable:observed",
                "source_nuisance_id": "not_applicable:observed",
                "lens_parameter_id": "not_applicable:observed",
                "native_lens_parameter_id": "not_applicable:observed",
                "lens_nuisance_id": "not_applicable:observed",
                "psf_id": str(reference.psf_id),
                "psf_fwhm_g_arcsec": float(reference.psfsize_g),
                "psf_fwhm_r_arcsec": float(reference.psfsize_r),
                "psf_fwhm_z_arcsec": float(reference.psfsize_z),
                "rendering_id": (f"observed_survey_payload:{reference.payload_sha256}"),
                "render_parameter_id": str(reference.render_parameter_id),
                "random_draw_id": "not_applicable:observed",
                "base_draw_key": "not_applicable:observed",
                "render_config_id": "not_applicable:observed",
                "render_contract_sha256": "not_applicable:observed",
                "parameter_row_sha256": "not_applicable:observed",
                "base_pixel_sha256": str(reference.pixel_sha256),
                "host_payload_sha256": "not_applicable:observed",
                "host_pixel_sha256": "not_applicable:observed",
                "host_store_relpath": "not_applicable:observed",
                "host_store_file_sha256": "not_applicable:observed",
                "pixel_relpath": str(reference.pixel_relpath),
                "pixel_file_sha256": str(reference.pixel_file_sha256),
                "pixel_sha256": str(reference.pixel_sha256),
                "pixel_array_key": "cutout",
                "pixel_layout": "HWC",
                "pixel_dtype": "float32",
                "pixel_shape": "101x101x3",
                "source_flux_policy": "not_applicable:observed",
                "intrinsic_flux_g_nmgy": np.nan,
                "intrinsic_flux_r_nmgy": np.nan,
                "intrinsic_flux_z_nmgy": np.nan,
                "source_flux_parameter_sha256": "not_applicable:observed",
                "grec_row_id": "not_applicable:observed",
                "scene_id": "not_applicable:observed",
                "parent_artifact_file_sha256": "not_applicable:observed",
                "parent_expectation_pixel_sha256": "not_applicable:observed",
                "injection_expectation_sha256": "not_applicable:observed",
                "match_row_sha256": (
                    str(matched["match_row_sha256"])
                    if matched is not None
                    else "not_applicable:unmatched"
                ),
                "parent_adapter_row_sha256": "not_applicable:observed",
                "pair_scene_links_manifest_sha256": "not_applicable:observed",
                "matches_manifest_sha256": matches_sha256,
                "pair_plan_manifest_sha256": pair_plan_sha256,
                "reference_manifest_sha256": bundle.source_bindings["reference_roster"],
                "reference_payload_manifest_sha256": bundle.source_bindings[
                    "reference_payload_manifest"
                ],
                "host_manifest_sha256": bundle.source_bindings["grec_host_manifest"],
                "host_tractor_manifest_sha256": bundle.source_bindings[
                    "host_tractor_manifest"
                ],
                "final_selection_manifest_sha256": bundle.source_bindings[
                    "final_grec_selection"
                ],
                "parent_manifest_sha256": bundle.source_bindings[
                    "grec_parent_manifest"
                ],
                "authority_digest": bundle.authority_digest,
            }
        )
    for rendered in pairs.sort_values(["pair_id", "arm"]).itertuples(index=False):
        host = host_by_id.loc[str(rendered.target_galaxy_id)]
        rows.append(
            {
                "row_id": f"gmatch2:{rendered.pair_id}:{rendered.arm}",
                "population": str(rendered.arm),
                "data_role": "final",
                "pair_id": str(rendered.pair_id),
                "reference_id": str(rendered.reference_id),
                "target_galaxy_id": str(rendered.target_galaxy_id),
                "host_galaxy_id": str(rendered.target_galaxy_id),
                "host_parent_id": str(rendered.host_parent_id),
                "proposed_role": str(rendered.proposed_role),
                "physical_group_id": str(rendered.host_physical_group_id),
                "geographic_spatial_key": str(
                    host["geographic_spatial_key"]
                ),
                "spatial_group_id": str(rendered.host_spatial_group_id),
                "source_parameter_id": str(rendered.source_parameter_id),
                "native_source_parameter_id": str(rendered.native_source_parameter_id),
                "source_nuisance_id": str(rendered.source_nuisance_id),
                "lens_parameter_id": str(rendered.lens_parameter_id),
                "native_lens_parameter_id": str(rendered.native_lens_parameter_id),
                "lens_nuisance_id": str(rendered.lens_nuisance_id),
                "psf_id": str(rendered.psf_id),
                "psf_fwhm_g_arcsec": float(rendered.psf_fwhm_g_arcsec),
                "psf_fwhm_r_arcsec": float(rendered.psf_fwhm_r_arcsec),
                "psf_fwhm_z_arcsec": float(rendered.psf_fwhm_z_arcsec),
                "rendering_id": str(rendered.rendering_id),
                "render_parameter_id": str(rendered.render_parameter_id),
                "random_draw_id": str(rendered.random_draw_id),
                "base_draw_key": str(rendered.base_draw_key),
                "render_config_id": str(rendered.render_config_id),
                "render_contract_sha256": str(rendered.render_contract_sha256),
                "parameter_row_sha256": str(rendered.parameter_row_sha256),
                "base_pixel_sha256": str(rendered.host_pixel_sha256),
                "host_payload_sha256": str(rendered.host_payload_sha256),
                "host_pixel_sha256": str(rendered.host_pixel_sha256),
                "host_store_relpath": str(rendered.host_store_relpath),
                "host_store_file_sha256": str(rendered.host_store_file_sha256),
                "pixel_relpath": str(rendered.pixel_relpath),
                "pixel_file_sha256": str(rendered.pixel_file_sha256),
                "pixel_sha256": str(rendered.pixel_sha256),
                "pixel_array_key": str(rendered.pixel_array_key),
                "pixel_layout": str(rendered.pixel_layout),
                "pixel_dtype": str(rendered.pixel_dtype),
                "pixel_shape": str(rendered.pixel_shape),
                "source_flux_policy": str(rendered.source_flux_policy),
                "intrinsic_flux_g_nmgy": float(rendered.intrinsic_flux_g_nmgy),
                "intrinsic_flux_r_nmgy": float(rendered.intrinsic_flux_r_nmgy),
                "intrinsic_flux_z_nmgy": float(rendered.intrinsic_flux_z_nmgy),
                "source_flux_parameter_sha256": str(
                    rendered.source_flux_parameter_sha256
                ),
                "grec_row_id": str(rendered.grec_row_id),
                "scene_id": str(rendered.scene_id),
                "parent_artifact_file_sha256": str(
                    rendered.parent_artifact_file_sha256
                ),
                "parent_expectation_pixel_sha256": str(
                    rendered.parent_expectation_pixel_sha256
                ),
                "injection_expectation_sha256": str(
                    rendered.injection_expectation_sha256
                ),
                "match_row_sha256": str(rendered.match_row_sha256),
                "parent_adapter_row_sha256": str(rendered.parent_adapter_row_sha256),
                "pair_scene_links_manifest_sha256": str(
                    rendered.pair_scene_links_manifest_sha256
                ),
                "matches_manifest_sha256": matches_sha256,
                "pair_plan_manifest_sha256": pair_plan_sha256,
                "reference_manifest_sha256": bundle.source_bindings["reference_roster"],
                "reference_payload_manifest_sha256": bundle.source_bindings[
                    "reference_payload_manifest"
                ],
                "host_manifest_sha256": bundle.source_bindings["grec_host_manifest"],
                "host_tractor_manifest_sha256": bundle.source_bindings[
                    "host_tractor_manifest"
                ],
                "final_selection_manifest_sha256": bundle.source_bindings[
                    "final_grec_selection"
                ],
                "parent_manifest_sha256": bundle.source_bindings[
                    "grec_parent_manifest"
                ],
                "authority_digest": bundle.authority_digest,
            }
        )
    result = pd.DataFrame(rows)
    if (
        len(result) != 43 + 2 * len(matches)
        or len(result.loc[result["population"].eq("observed_reference")]) != 43
        or set(result["data_role"]) != {"final"}
    ):
        raise ContractError("final analysis manifest has wrong population rows")
    dependency_columns = {
        "physical_group_id",
        "geographic_spatial_key",
        "spatial_group_id",
        "reference_id",
        "host_galaxy_id",
        "host_parent_id",
        "source_parameter_id",
        "lens_parameter_id",
        "psf_id",
        "rendering_id",
        "render_parameter_id",
        "random_draw_id",
        "base_pixel_sha256",
        "pixel_sha256",
    }
    require_nonempty(result, dependency_columns, context="analysis dependencies")
    result["analysis_row_sha256"] = _row_hashes(result)
    result = result.sort_values("row_id", ignore_index=True)
    assert_context_schema(result, context="output.analysis_manifest")
    return result


def _strict_schema_check(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    contexts = {
        "reference_adapter": "intermediate.reference_adapter",
        "host_adapter": "intermediate.host_adapter",
        "host_tractor_adapter": "intermediate.host_tractor_adapter",
        "parent_theta15_adapter": "intermediate.parent_theta15_adapter",
        "matches": "output.matches",
        "reference_disposition": "output.reference_disposition",
        "host_disposition": "output.host_disposition",
        "balance": "output.balance",
        "scaler_comparison": "output.scaler_comparison",
        "pair_scene_links": "output.pair_scene_links",
        "pair_plan": "output.pair_plan",
        "rendered_pairs": "output.rendered_pairs",
        "analysis_manifest": "output.analysis_manifest",
    }
    if set(frames) != set(contexts):
        raise ContractError("strict schema validation did not receive every frame")
    for name, frame in frames.items():
        assert_context_schema(frame, context=contexts[name])
    return {
        "status": "strict_context_schema_allowlists_passed",
        "allowlist_contract_sha256": ALLOWLIST_CONTRACT_SHA256,
        "unknown_column_policy": "reject",
        "missing_column_policy": "reject",
        "duplicate_column_policy": "reject",
        "input_contexts_validated": [
            "input.reference_roster",
            "input.reference_payload_manifest",
            "input.reference_tractor_manifest",
            "input.host_role_proposal",
            "input.final_grec_selection",
            "input.host_tractor_manifest",
            "input.grec_host_manifest",
            "input.grec_scene_manifest",
            "input.grec_parent_manifest",
            "input.train_manifest",
            "input.validation_manifest",
            "input.test_manifest",
        ],
        "intermediate_and_output_contexts_validated": sorted(contexts.values()),
        "columns_validated": sum(len(frame.columns) for frame in frames.values()),
        "detector_scores_read": False,
        "detector_outcomes_read": False,
        "feature_arrays_created": False,
    }


def prepare(
    *,
    package_root: Path,
    config_path: Path,
    documentation_path: Path,
) -> dict[str, Any]:
    """Write all final preparation artifacts once, atomically."""
    config = load_config(config_path)
    root = Path(config["paths"]["repository_root"]).resolve()
    seal = validate_config_seal(
        package_root=package_root,
        config_path=config_path,
        documentation_path=documentation_path,
    )
    output = root / str(config["paths"]["output_root"])
    if output.exists():
        if any(output.iterdir()):
            raise ContractError("frozen G_match preparation output already exists")
        output.rmdir()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".gmatch-preparation-", dir=output.parent))
    try:
        bundle = load_authorities(config, root=root, verify_all_host_pixels=True)
        frames: dict[str, pd.DataFrame] = {
            "reference_adapter": bundle.reference,
            "host_adapter": bundle.host,
            "host_tractor_adapter": bundle.host_tractor,
            "parent_theta15_adapter": bundle.parent_theta15,
        }
        output_receipts = {
            name: _write_frame(frame, temporary / OUTPUT_FILES[name])
            for name, frame in frames.items()
        }
        authority_receipt = {
            "schema_version": 1,
            "status": "gmatch_v2_direct_authority_adapters_complete",
            "authority_digest": bundle.authority_digest,
            "source_bindings": bundle.source_bindings,
            "outputs": output_receipts,
            "detector_development_exclusion": bundle.detector_exclusion,
            "pixel_verification": bundle.pixel_verification,
            "payload_file_hash_field": "host_payload_sha256",
            "host_array_hash_field": "host_pixel_sha256",
            "hash_domains_distinct": bool(
                bundle.host["host_payload_sha256"]
                .ne(bundle.host["host_pixel_sha256"])
                .all()
            ),
            "strict_schema_allowlist_contract_sha256": (ALLOWLIST_CONTRACT_SHA256),
            "detector_scores_used": False,
            "detector_outcomes_used": False,
            "feature_extraction_performed": False,
        }
        atomic_write_json(
            authority_receipt, temporary / "GMATCH_AUTHORITY_RECEIPT.json"
        )

        matched = match(bundle, config)
        frames.update(
            {
                "matches": matched.matches,
                "reference_disposition": matched.reference_disposition,
                "host_disposition": matched.host_disposition,
                "balance": matched.balance,
                "scaler_comparison": matched.scaler_comparison,
            }
        )
        for name in (
            "matches",
            "reference_disposition",
            "host_disposition",
            "balance",
            "scaler_comparison",
        ):
            output_receipts[name] = _write_frame(
                frames[name], temporary / OUTPUT_FILES[name]
            )
        matches_sha256 = output_receipts["matches"]["sha256"]
        matching_receipt = {
            **matched.receipt,
            "matches_manifest_sha256": matches_sha256,
            "reference_disposition_sha256": output_receipts["reference_disposition"][
                "sha256"
            ],
            "host_disposition_sha256": output_receipts["host_disposition"]["sha256"],
            "balance_diagnostics_sha256": output_receipts["balance"]["sha256"],
            "scaler_policy_comparison_sha256": output_receipts["scaler_comparison"][
                "sha256"
            ],
        }
        atomic_write_json(matching_receipt, temporary / "GMATCH_MATCH_RECEIPT.json")

        links = _build_pair_scene_links(bundle, matched, matches_sha256=matches_sha256)
        frames["pair_scene_links"] = links
        output_receipts["pair_scene_links"] = _write_frame(
            links, temporary / OUTPUT_FILES["pair_scene_links"]
        )
        plan = _build_pair_plan(
            bundle,
            matched,
            links,
            matches_sha256=matches_sha256,
            links_sha256=output_receipts["pair_scene_links"]["sha256"],
        )
        frames["pair_plan"] = plan
        output_receipts["pair_plan"] = _write_frame(
            plan, temporary / OUTPUT_FILES["pair_plan"]
        )
        pairs = plan.copy()
        pairs["pair_plan_manifest_sha256"] = output_receipts["pair_plan"]["sha256"]
        pairs["rendered_pair_row_sha256"] = _row_hashes(pairs)
        assert_context_schema(pairs, context="output.rendered_pairs")
        frames["rendered_pairs"] = pairs
        pair_validation = validate_zero_copy_pairs(
            pairs,
            bundle=bundle,
            matches=matched.matches,
            pair_plan_sha256=output_receipts["pair_plan"]["sha256"],
        )
        output_receipts["rendered_pairs"] = _write_frame(
            pairs, temporary / OUTPUT_FILES["rendered_pairs"]
        )
        pair_receipt = {
            "schema_version": 1,
            "status": "gmatch_v2_zero_copy_pairs_validated",
            "authority_digest": bundle.authority_digest,
            "matched_targets": len(matched.matches),
            "pair_rows": len(pairs),
            "pair_scene_links_manifest_sha256": output_receipts["pair_scene_links"][
                "sha256"
            ],
            "pair_plan_manifest_sha256": output_receipts["pair_plan"]["sha256"],
            "rendered_pairs_manifest_sha256": output_receipts["rendered_pairs"][
                "sha256"
            ],
            **pair_validation,
        }
        atomic_write_json(pair_receipt, temporary / "GMATCH_PAIR_RECEIPT.json")

        analysis = _analysis_manifest(
            bundle,
            matched.matches,
            pairs,
            matches_sha256=matches_sha256,
            pair_plan_sha256=output_receipts["pair_plan"]["sha256"],
        )
        frames["analysis_manifest"] = analysis
        output_receipts["analysis_manifest"] = _write_frame(
            analysis, temporary / OUTPUT_FILES["analysis_manifest"]
        )
        schema_validation = _strict_schema_check(frames)
        schema_validation["source_access_ledger"] = sorted(bundle.source_bindings)
        atomic_write_json(
            schema_validation,
            temporary / "STRICT_SCHEMA_ALLOWLIST_RECEIPT.json",
        )

        registration = {
            "schema_version": 1,
            "status": "gmatch_v2_score_blind_preparation_registered",
            "study_id": config["study"]["study_id"],
            "data_role": "final",
            "authority_digest": bundle.authority_digest,
            "config_seal_sha256": sha256_file(package_root / "config/CONFIG_SEAL.json"),
            "config_sha256": seal["config_sha256"],
            "source_tree_sha256": seal["source_tree_sha256"],
            "dependency_lock_sha256": seal["dependency_lock_sha256"],
            "strict_schema_allowlist_contract_sha256": (ALLOWLIST_CONTRACT_SHA256),
            "scaler_diagnostics": {
                "policy": matched.receipt["scaler_value_policy"],
                "authority_rows": matched.receipt["scaler_authority_rows"],
                "finite_count": matched.receipt["scaler_finite_count"],
                "missing_count": matched.receipt["scaler_missing_count"],
                "median": matched.receipt["host_universe_median"],
                "iqr": matched.receipt["host_universe_iqr"],
                "matching_candidates": matched.receipt["hosts_complete_covariates"],
                "ineligible_hosts": matched.receipt[
                    "hosts_ineligible_nonfinite_covariate"
                ],
                "prior_complete_case_comparison": matched.receipt[
                    "prior_complete_case_scaler"
                ],
            },
            "scientific_contract": {
                "reference_rows": 43,
                "host_rows": 4500,
                "theta_e_arcsec": 1.5,
                "covariates": list(config["matching"]["covariates"]),
                "matching": config["matching"],
                "pairing": config["pairing"],
            },
            "output_bindings": output_receipts,
            "feature_extraction_authorized": False,
            "probe_fitting_authorized": False,
            "bootstrap_authorized": False,
            "final_scoring_authorized": False,
        }
        atomic_write_json(registration, temporary / "GMATCH_REGISTRATION.json")
        preparation_receipt = {
            "schema_version": 1,
            "status": "gmatch_v2_preparation_complete_pre_feature_review",
            "verdict": "PASS",
            "data_role": "final",
            "authority_digest": bundle.authority_digest,
            "scaler_value_policy": matched.receipt["scaler_value_policy"],
            "scaler_authority_rows": matched.receipt["scaler_authority_rows"],
            "scaler_finite_count": matched.receipt["scaler_finite_count"],
            "scaler_missing_count": matched.receipt["scaler_missing_count"],
            "scaler_median": matched.receipt["host_universe_median"],
            "scaler_iqr": matched.receipt["host_universe_iqr"],
            "matching_candidate_rows": matched.receipt["hosts_complete_covariates"],
            "ineligible_nonfinite_host_rows": matched.receipt[
                "hosts_ineligible_nonfinite_covariate"
            ],
            "prior_complete_case_comparison": matched.receipt[
                "prior_complete_case_scaler"
            ],
            "matched_references": len(matched.matches),
            "unmatched_references": 43 - len(matched.matches),
            "analysis_rows": len(analysis),
            "analysis_reference_rows": int(
                analysis["population"].eq("observed_reference").sum()
            ),
            "analysis_bare_rows": int(analysis["population"].eq("bare").sum()),
            "analysis_injected_rows": int(analysis["population"].eq("injected").sum()),
            "output_bindings": output_receipts,
            "authority_receipt_sha256": sha256_file(
                temporary / "GMATCH_AUTHORITY_RECEIPT.json"
            ),
            "matching_receipt_sha256": sha256_file(
                temporary / "GMATCH_MATCH_RECEIPT.json"
            ),
            "pair_receipt_sha256": sha256_file(temporary / "GMATCH_PAIR_RECEIPT.json"),
            "registration_sha256": sha256_file(temporary / "GMATCH_REGISTRATION.json"),
            "strict_schema_allowlist_receipt_sha256": sha256_file(
                temporary / "STRICT_SCHEMA_ALLOWLIST_RECEIPT.json"
            ),
            "strict_schema_allowlist_contract_sha256": (ALLOWLIST_CONTRACT_SHA256),
            "config_seal_sha256": sha256_file(package_root / "config/CONFIG_SEAL.json"),
            "source_tree_sha256": seal["source_tree_sha256"],
            "dependency_lock_sha256": seal["dependency_lock_sha256"],
            "detector_scores_used": False,
            "detector_outcomes_used": False,
            "feature_extraction_performed": False,
            "probe_fitting_performed": False,
            "bootstrap_performed": False,
            "feature_authorization_issued": False,
            "host_total_flux_floor_applied": False,
            "flux_rescaling_applied": False,
        }
        atomic_write_json(
            preparation_receipt, temporary / "GMATCH_PREPARATION_RECEIPT.json"
        )
        os.replace(temporary, output)
        return preparation_receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
