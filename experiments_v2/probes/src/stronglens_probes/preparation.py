"""Score-blind adapters for the finalized G_rec and reference schemas."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    ContractError,
    assert_score_blind,
    atomic_write_json,
    atomic_write_parquet,
    canonical_json_sha256,
    canonical_row_sha256,
    require_columns,
    require_nonempty_columns,
    sha256_file,
    stable_digest,
)

ADAPTER_CONTRACT = {
    "schema_version": 2,
    "host": {
        "galaxy_id": "final_grec_hosts.host_galaxy_id",
        "payload_sha256": "final_grec_hosts.host_payload_sha256",
        "host_parent_id": "final_grec_hosts.host_parent_id",
        "host_physical_group_id": (
            "approved_final_selection.physical_group_id"
        ),
        "host_spatial_group_id": "final_grec_hosts.host_spatial_group_id",
        "host_healpix_128": "approved_final_selection.healpix_128",
        "authority_membership": (
            "proposed_roles[proposed_role == 'experiment_host']"
        ),
    },
    "host_tractor": {
        "psfsize_r": "tractor_hosts.psfsize_r_tractor",
        "psfdepth_r": "tractor_hosts.psfdepth_r_tractor",
        "other_predicate_fields": "exact-name projection from tractor_hosts",
    },
    "reference": {
        "psf_id": (
            "sha256(reference_psf_v1, galaxy_id, exact float hex "
            "psfsize_g/psfsize_r/psfsize_z)"
        ),
        "render_parameter_id": (
            "sha256(observed_reference_render_v1, galaxy_id, payload_sha256)"
        ),
        "pixel_sha256": "sha256 exact C-order cutout element bytes",
        "pixel_file_sha256": "validated payload file SHA-256",
    },
    "parent": {
        "host_physical_group_id": "host adapter host_physical_group_id",
        "source_parameter_id": "grec_parent.source_nuisance_id",
        "lens_parameter_id": "grec_parent.lens_nuisance_id",
        "psf_id": (
            "sha256(grec_psf_v1, host_galaxy_id, exact float hex "
            "psf_fwhm_g_arcsec/psf_fwhm_r_arcsec/psf_fwhm_z_arcsec)"
        ),
        "rendering_id": "grec_parent.base_draw_key",
        "render_parameter_id": "grec_parent.base_draw_key",
        "random_draw_id": "grec_parent.base_draw_key",
        "intrinsic_flux_*_nmgy": "grec_parent.source_flux_nmgy_*",
        "source_flux_parameter_sha256": (
            "canonical SHA-256 of exact float-hex source flux triplet"
        ),
        "final_image_sha256": "grec_parent.parent_artifact_sha256",
        "scientific_pixel_sha256": "grec_parent.expectation_pixel_sha256",
        "pixel_path": "grec_parent.parent_pixel_path",
    },
}
ADAPTER_CONTRACT_SHA256 = canonical_json_sha256(ADAPTER_CONTRACT)


@dataclass(frozen=True)
class FinalSchemaSourcePaths:
    repository_root: Path
    reference_manifest: Path
    reference_payload_manifest: Path
    reference_tractor_manifest: Path
    reference_payload_root: Path
    host_universe_manifest: Path
    final_selection_manifest: Path
    final_selection_receipt: Path
    host_tractor_manifest: Path
    grec_host_manifest: Path
    grec_host_receipt: Path
    grec_scene_manifest: Path
    grec_scene_receipt: Path
    grec_parent_manifest: Path
    grec_render_receipt: Path


@dataclass(frozen=True)
class FinalSchemaAdapterPaths:
    reference_payload_adapter: Path
    host_adapter: Path
    host_tractor_adapter: Path
    grec_parent_adapter: Path
    receipt: Path


def adapter_paths(output_dir: Path) -> FinalSchemaAdapterPaths:
    return FinalSchemaAdapterPaths(
        reference_payload_adapter=output_dir / "reference_payload_adapter.parquet",
        host_adapter=output_dir / "host_authority_adapter.parquet",
        host_tractor_adapter=output_dir / "host_tractor_adapter.parquet",
        grec_parent_adapter=output_dir / "grec_parent_adapter.parquet",
        receipt=output_dir / "FINAL_SCHEMA_ADAPTER_RECEIPT.json",
    )


def _load_receipt(path: Path, *, status: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != status:
        raise ContractError(f"{path.name} has unexpected status")
    return payload


def _array_bytes_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ContractError(f"pixel path escapes repository root: {path}") from error


def _float_identity(value: object) -> str:
    converted = float(value)
    if not np.isfinite(converted):
        raise ContractError("identity-defining float is not finite")
    return converted.hex()


def _row_hashes(frame: pd.DataFrame) -> list[str]:
    columns = list(frame.columns)
    return [
        canonical_row_sha256(row, columns)
        for row in frame.to_dict(orient="records")
    ]


def _derive_reference_adapter(
    sources: FinalSchemaSourcePaths,
) -> pd.DataFrame:
    roster = pd.read_parquet(sources.reference_manifest)
    payloads = pd.read_parquet(sources.reference_payload_manifest)
    tractor = pd.read_parquet(sources.reference_tractor_manifest)
    for frame, context in (
        (roster, "reference roster source"),
        (payloads, "reference payload source"),
        (tractor, "reference Tractor source"),
    ):
        assert_score_blind(frame, context=context)
        require_nonempty_columns(frame, {"galaxy_id"}, context=context)
    require_columns(
        payloads,
        {"galaxy_id", "payload_sha256", "local_relpath"},
        context="reference payload source",
    )
    require_columns(
        tractor,
        {
            "galaxy_id",
            "payload_sha256",
            "psfsize_g",
            "psfsize_r",
            "psfsize_z",
        },
        context="reference Tractor source",
    )
    roster_ids = set(roster["galaxy_id"].astype(str))
    if (
        len(roster) != 43
        or len(roster_ids) != 43
        or set(payloads["galaxy_id"].astype(str)) != roster_ids
        or set(tractor["galaxy_id"].astype(str)) != roster_ids
    ):
        raise ContractError("reference sources do not contain the same exact 43 IDs")
    merged = payloads.merge(
        tractor[
            [
                "galaxy_id",
                "payload_sha256",
                "psfsize_g",
                "psfsize_r",
                "psfsize_z",
            ]
        ],
        on="galaxy_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_tractor"),
    )
    if not merged["payload_sha256"].eq(merged["payload_sha256_tractor"]).all():
        raise ContractError("reference Tractor rows changed payload identities")
    pixel_hashes: list[str] = []
    pixel_relpaths: list[str] = []
    for row in merged.itertuples(index=False):
        payload_path = sources.reference_payload_root / str(row.local_relpath)
        if sha256_file(payload_path) != str(row.payload_sha256):
            raise ContractError(f"reference payload bytes changed: {row.galaxy_id}")
        with np.load(payload_path, allow_pickle=False) as archive:
            if "cutout" not in archive.files:
                raise ContractError(f"reference payload lacks cutout: {row.galaxy_id}")
            cutout = np.asarray(archive["cutout"])
        if (
            cutout.dtype != np.float32
            or cutout.shape != (101, 101, 3)
            or not np.isfinite(cutout).all()
        ):
            raise ContractError(f"reference cutout is invalid: {row.galaxy_id}")
        pixel_hashes.append(_array_bytes_sha256(cutout))
        pixel_relpaths.append(_relative_path(payload_path, sources.repository_root))
    merged["psf_id"] = [
        "reference_psf_"
        + stable_digest(
            "reference_psf_v1",
            galaxy_id,
            _float_identity(g),
            _float_identity(r),
            _float_identity(z),
        )[:40]
        for galaxy_id, g, r, z in merged[
            ["galaxy_id", "psfsize_g", "psfsize_r", "psfsize_z"]
        ].itertuples(index=False)
    ]
    merged["render_parameter_id"] = [
        "observed_reference_render_"
        + stable_digest(
            "observed_reference_render_v1",
            galaxy_id,
            payload_sha256,
        )[:40]
        for galaxy_id, payload_sha256 in merged[
            ["galaxy_id", "payload_sha256"]
        ].itertuples(index=False)
    ]
    merged["pixel_sha256"] = pixel_hashes
    merged["pixel_file_sha256"] = merged["payload_sha256"].astype(str)
    merged["pixel_relpath"] = pixel_relpaths
    merged["source_reference_manifest_sha256"] = sha256_file(
        sources.reference_manifest
    )
    merged["source_reference_payload_manifest_sha256"] = sha256_file(
        sources.reference_payload_manifest
    )
    merged["source_reference_tractor_manifest_sha256"] = sha256_file(
        sources.reference_tractor_manifest
    )
    merged["adapter_contract_sha256"] = ADAPTER_CONTRACT_SHA256
    merged = merged.drop(columns="payload_sha256_tractor")
    merged["adapter_row_sha256"] = _row_hashes(merged)
    return merged.sort_values("galaxy_id", ignore_index=True)


def _derive_host_adapter(
    sources: FinalSchemaSourcePaths,
) -> pd.DataFrame:
    universe = pd.read_parquet(sources.host_universe_manifest)
    selection = pd.read_parquet(sources.final_selection_manifest)
    hosts = pd.read_parquet(sources.grec_host_manifest)
    for frame, context in (
        (universe, "host universe source"),
        (selection, "final selection source"),
        (hosts, "G_rec host source"),
    ):
        assert_score_blind(frame, context=context)
    require_columns(
        universe,
        {
            "galaxy_id",
            "payload_sha256",
            "evaluation_spatial_group_id",
            "proposed_role",
        },
        context="host universe source",
    )
    require_columns(
        selection,
        {
            "galaxy_id",
            "payload_sha256",
            "evaluation_spatial_group_id",
            "physical_group_id",
            "healpix_128",
            "final_role",
        },
        context="final selection source",
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
            "final_role",
            "psf_bin",
            "depth_bin",
        },
        context="G_rec host source",
    )
    if (
        len(selection) != 4_500
        or len(hosts) != 4_500
        or selection["galaxy_id"].astype(str).nunique() != 4_500
        or hosts["host_galaxy_id"].astype(str).nunique() != 4_500
    ):
        raise ContractError("final host sources are not exact 4,500-row manifests")
    universe_hosts = universe[
        universe["proposed_role"].eq("experiment_host")
    ][
        [
            "galaxy_id",
            "payload_sha256",
            "evaluation_spatial_group_id",
            "proposed_role",
        ]
    ]
    joined = hosts.merge(
        selection[
            [
                "galaxy_id",
                "payload_sha256",
                "evaluation_spatial_group_id",
                "physical_group_id",
                "healpix_128",
                "final_role",
            ]
        ],
        left_on="host_galaxy_id",
        right_on="galaxy_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_selection"),
    ).merge(
        universe_hosts,
        on="galaxy_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_universe"),
    )
    if len(joined) != 4_500:
        raise ContractError("final hosts are not an exact authority-universe subset")
    checks = {
        "host_payload_sha256": "payload_sha256",
        "host_spatial_group_id": "evaluation_spatial_group_id",
        "final_role": "final_role_selection",
        "payload_sha256": "payload_sha256_universe",
        "evaluation_spatial_group_id": "evaluation_spatial_group_id_universe",
    }
    for left, right in checks.items():
        if not joined[left].astype(str).eq(joined[right].astype(str)).all():
            raise ContractError(f"host adapter source mismatch: {left} != {right}")
    adapter = pd.DataFrame(
        {
            "galaxy_id": joined["host_galaxy_id"].astype(str),
            "payload_sha256": joined["host_payload_sha256"].astype(str),
            "host_parent_id": joined["host_parent_id"].astype(str),
            "host_physical_group_id": joined["physical_group_id"].astype(str),
            "host_spatial_group_id": joined["host_spatial_group_id"].astype(str),
            "host_healpix_128": joined["healpix_128"].astype("int64"),
            "evaluation_spatial_group_id": joined[
                "evaluation_spatial_group_id"
            ].astype(str),
            "proposed_role": joined["proposed_role"].astype(str),
            "final_role": joined["final_role"].astype(str),
            "psf_bin": joined["psf_bin"].astype(float),
            "depth_bin": joined["depth_bin"].astype(float),
            "host_pixel_path": joined["host_pixel_path"].astype(str),
            "source_host_pixel_sha256": joined["host_pixel_sha256"].astype(str),
            "source_host_manifest_sha256": sha256_file(sources.grec_host_manifest),
            "source_final_selection_manifest_sha256": sha256_file(
                sources.final_selection_manifest
            ),
            "source_host_universe_manifest_sha256": sha256_file(
                sources.host_universe_manifest
            ),
            "adapter_contract_sha256": ADAPTER_CONTRACT_SHA256,
        }
    )
    adapter["adapter_row_sha256"] = _row_hashes(adapter)
    return adapter.sort_values("galaxy_id", ignore_index=True)


def _derive_host_tractor_adapter(
    sources: FinalSchemaSourcePaths,
    hosts: pd.DataFrame,
) -> pd.DataFrame:
    tractor = pd.read_parquet(sources.host_tractor_manifest)
    assert_score_blind(tractor, context="host Tractor source")
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
        context="host Tractor source",
    )
    if (
        len(tractor) != 4_500
        or tractor["galaxy_id"].astype(str).nunique() != 4_500
        or set(tractor["galaxy_id"].astype(str))
        != set(hosts["galaxy_id"].astype(str))
    ):
        raise ContractError("host Tractor source is not the exact final host roster")
    joined = tractor.merge(
        hosts[["galaxy_id", "payload_sha256"]],
        on="galaxy_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_host"),
    )
    if not joined["payload_sha256"].eq(joined["payload_sha256_host"]).all():
        raise ContractError("host Tractor payloads differ from final hosts")
    adapter = pd.DataFrame(
        {
            "galaxy_id": joined["galaxy_id"].astype(str),
            "payload_sha256": joined["payload_sha256"].astype(str),
            "flux_z": joined["flux_z"],
            "sersic": joined["sersic"],
            "shape_e1": joined["shape_e1"],
            "shape_e2": joined["shape_e2"],
            "psfsize_r": joined["psfsize_r_tractor"],
            "psfdepth_r": joined["psfdepth_r_tractor"],
            "source_host_tractor_manifest_sha256": sha256_file(
                sources.host_tractor_manifest
            ),
            "source_psfsize_r_column": "psfsize_r_tractor",
            "source_psfdepth_r_column": "psfdepth_r_tractor",
            "adapter_contract_sha256": ADAPTER_CONTRACT_SHA256,
        }
    )
    adapter["source_tractor_row_sha256"] = [
        canonical_row_sha256(row, tractor.columns)
        for row in tractor.to_dict(orient="records")
    ]
    adapter["adapter_row_sha256"] = _row_hashes(adapter)
    return adapter.sort_values("galaxy_id", ignore_index=True)


def _derive_parent_adapter(
    sources: FinalSchemaSourcePaths,
    hosts: pd.DataFrame,
) -> pd.DataFrame:
    scene = pd.read_parquet(sources.grec_scene_manifest)
    parent = pd.read_parquet(sources.grec_parent_manifest)
    assert_score_blind(scene, context="G_rec scene source")
    assert_score_blind(parent, context="G_rec physical parent source")
    require_columns(
        parent,
        {
            *scene.columns,
            "source_nuisance_id",
            "lens_nuisance_id",
            "base_draw_key",
            "source_flux_nmgy_g",
            "source_flux_nmgy_r",
            "source_flux_nmgy_z",
            "psf_fwhm_g_arcsec",
            "psf_fwhm_r_arcsec",
            "psf_fwhm_z_arcsec",
            "parent_pixel_path",
            "parent_artifact_sha256",
            "expectation_pixel_sha256",
            "injection_expectation_sha256",
        },
        context="G_rec physical parent source",
    )
    if (
        len(scene) != 49_500
        or len(parent) != 49_500
        or scene["grec_row_id"].astype(str).nunique() != 49_500
        or parent["grec_row_id"].astype(str).nunique() != 49_500
    ):
        raise ContractError("G_rec source is not the exact 49,500-row parent bank")
    scene_sorted = scene.sort_values("grec_row_id", ignore_index=True)
    parent_sorted = parent.sort_values("grec_row_id", ignore_index=True)
    try:
        pd.testing.assert_frame_equal(
            scene_sorted,
            parent_sorted[list(scene.columns)],
            check_exact=True,
        )
    except AssertionError as error:
        raise ContractError(
            "physical parent scientific rows differ from frozen scene table"
        ) from error
    host_lookup = hosts.set_index(hosts["galaxy_id"].astype(str))
    if set(parent["host_galaxy_id"].astype(str)) != set(host_lookup.index):
        raise ContractError("physical parent does not use the exact final hosts")
    adapter = parent.copy()
    adapter["native_source_parameter_id"] = adapter["source_parameter_id"].astype(str)
    adapter["native_lens_parameter_id"] = adapter["lens_parameter_id"].astype(str)
    adapter["source_parameter_id"] = adapter["source_nuisance_id"].astype(str)
    adapter["lens_parameter_id"] = adapter["lens_nuisance_id"].astype(str)
    adapter["host_physical_group_id"] = adapter["host_galaxy_id"].astype(str).map(
        host_lookup["host_physical_group_id"]
    )
    adapter["psf_id"] = [
        "grec_psf_"
        + stable_digest(
            "grec_psf_v1",
            galaxy_id,
            _float_identity(g),
            _float_identity(r),
            _float_identity(z),
        )[:40]
        for galaxy_id, g, r, z in adapter[
            [
                "host_galaxy_id",
                "psf_fwhm_g_arcsec",
                "psf_fwhm_r_arcsec",
                "psf_fwhm_z_arcsec",
            ]
        ].itertuples(index=False)
    ]
    adapter["rendering_id"] = adapter["base_draw_key"].astype(str)
    adapter["render_parameter_id"] = adapter["base_draw_key"].astype(str)
    adapter["random_draw_id"] = adapter["base_draw_key"].astype(str)
    adapter["intrinsic_flux_g_nmgy"] = adapter["source_flux_nmgy_g"]
    adapter["intrinsic_flux_r_nmgy"] = adapter["source_flux_nmgy_r"]
    adapter["intrinsic_flux_z_nmgy"] = adapter["source_flux_nmgy_z"]
    adapter["source_flux_parameter_sha256"] = [
        canonical_json_sha256(
            {
                "g": _float_identity(g),
                "r": _float_identity(r),
                "z": _float_identity(z),
            }
        )
        for g, r, z in adapter[
            ["source_flux_nmgy_g", "source_flux_nmgy_r", "source_flux_nmgy_z"]
        ].itertuples(index=False)
    ]
    adapter["final_image_sha256"] = adapter["parent_artifact_sha256"].astype(str)
    adapter["final_image_path"] = adapter["parent_pixel_path"].astype(str)
    adapter["source_grec_scene_manifest_sha256"] = sha256_file(
        sources.grec_scene_manifest
    )
    adapter["source_grec_parent_manifest_sha256"] = sha256_file(
        sources.grec_parent_manifest
    )
    adapter["source_host_adapter_content_sha256"] = canonical_json_sha256(
        {
            str(row.galaxy_id): str(row.adapter_row_sha256)
            for row in hosts[["galaxy_id", "adapter_row_sha256"]].itertuples(
                index=False
            )
        }
    )
    adapter["adapter_contract_sha256"] = ADAPTER_CONTRACT_SHA256
    adapter["adapter_row_sha256"] = _row_hashes(adapter)
    return adapter.sort_values("grec_row_id", ignore_index=True)


def _validate_source_receipts(sources: FinalSchemaSourcePaths) -> None:
    selection = _load_receipt(
        sources.final_selection_receipt,
        status="final_experiment_host_selection_approved",
    )
    host = _load_receipt(
        sources.grec_host_receipt,
        status="final_pre_score_grec_hosts_frozen",
    )
    scene = _load_receipt(
        sources.grec_scene_receipt,
        status="grec_scene_table_frozen_pre_score",
    )
    render = _load_receipt(
        sources.grec_render_receipt,
        status="grec_parent_bank_frozen_pre_score",
    )
    checks = (
        (selection, "selection_manifest_sha256", sources.final_selection_manifest),
        (selection, "role_proposal_sha256", sources.host_universe_manifest),
        (selection, "tractor_manifest_sha256", sources.host_tractor_manifest),
        (host, "host_manifest_sha256", sources.grec_host_manifest),
        (scene, "scene_table_sha256", sources.grec_scene_manifest),
        (render, "scene_table_sha256", sources.grec_scene_manifest),
        (render, "parent_manifest_sha256", sources.grec_parent_manifest),
    )
    for receipt, field, path in checks:
        if receipt.get(field) != sha256_file(path):
            raise ContractError(f"source receipt has wrong {field}")
    if (
        selection.get("detector_scores_used") is not False
        or host.get("detector_scores_used") is not False
        or scene.get("detector_scores_used") is not False
    ):
        raise ContractError("a source receipt does not prove score-blind preparation")


def _derive_all(
    sources: FinalSchemaSourcePaths,
) -> dict[str, pd.DataFrame]:
    _validate_source_receipts(sources)
    reference = _derive_reference_adapter(sources)
    host = _derive_host_adapter(sources)
    tractor = _derive_host_tractor_adapter(sources, host)
    parent = _derive_parent_adapter(sources, host)
    return {
        "reference_payload_adapter": reference,
        "host_adapter": host,
        "host_tractor_adapter": tractor,
        "grec_parent_adapter": parent,
    }


def write_final_schema_adapters(
    sources: FinalSchemaSourcePaths,
    output_dir: Path,
) -> dict[str, Any]:
    """Write new adapters once; source scientific artifacts remain untouched."""
    paths = adapter_paths(output_dir)
    if any(path.exists() for path in vars(paths).values()):
        raise ContractError("final-schema adapter outputs already exist")
    frames = _derive_all(sources)
    for name, frame in frames.items():
        atomic_write_parquet(frame, getattr(paths, name))
    receipt = {
        "schema_version": 1,
        "status": "score_blind_final_schema_adapters_complete",
        "adapter_contract": ADAPTER_CONTRACT,
        "adapter_contract_sha256": ADAPTER_CONTRACT_SHA256,
        "source_bindings": {
            field: sha256_file(getattr(sources, field))
            for field in (
                "reference_manifest",
                "reference_payload_manifest",
                "reference_tractor_manifest",
                "host_universe_manifest",
                "final_selection_manifest",
                "final_selection_receipt",
                "host_tractor_manifest",
                "grec_host_manifest",
                "grec_host_receipt",
                "grec_scene_manifest",
                "grec_scene_receipt",
                "grec_parent_manifest",
                "grec_render_receipt",
            )
        },
        "outputs": {
            name: {
                "sha256": sha256_file(getattr(paths, name)),
                "rows": len(frame),
            }
            for name, frame in frames.items()
        },
        "reference_pixel_arrays_hashed": 43,
        "grec_parent_pixel_arrays_opened": 0,
        "detector_scores_used": False,
        "detector_outcomes_used": False,
        "feature_extraction_performed": False,
        "stein_probe_run": False,
    }
    atomic_write_json(receipt, paths.receipt)
    return receipt


def validate_final_schema_adapters(
    sources: FinalSchemaSourcePaths,
    output_dir: Path,
) -> dict[str, Any]:
    """Independently reconstruct and compare every adapter row and binding."""
    paths = adapter_paths(output_dir)
    receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
    expected_frames = _derive_all(sources)
    if (
        receipt.get("status") != "score_blind_final_schema_adapters_complete"
        or receipt.get("adapter_contract_sha256") != ADAPTER_CONTRACT_SHA256
        or receipt.get("detector_scores_used") is not False
        or receipt.get("feature_extraction_performed") is not False
    ):
        raise ContractError("final-schema adapter receipt is invalid")
    for name, expected in expected_frames.items():
        path = getattr(paths, name)
        bound = receipt.get("outputs", {}).get(name, {})
        if (
            bound.get("sha256") != sha256_file(path)
            or bound.get("rows") != len(expected)
        ):
            raise ContractError(f"{name} has an invalid receipt binding")
        observed = pd.read_parquet(path)
        try:
            pd.testing.assert_frame_equal(observed, expected, check_exact=True)
        except AssertionError as error:
            raise ContractError(f"{name} differs from exact reconstruction") from error
    return {
        "status": "score_blind_final_schema_adapters_validated",
        "adapter_receipt_sha256": sha256_file(paths.receipt),
        "reference_rows": len(expected_frames["reference_payload_adapter"]),
        "host_rows": len(expected_frames["host_adapter"]),
        "tractor_rows": len(expected_frames["host_tractor_adapter"]),
        "parent_rows": len(expected_frames["grec_parent_adapter"]),
        "detector_scores_used": False,
        "feature_extraction_performed": False,
    }
