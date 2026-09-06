from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .authority import resolve_binding, validate_execution_lock
from .contracts import (
    ContractError,
    assert_config_integrity,
    config_provenance,
    load_json,
    source_closure,
)
from .draw_count import validate_draw_count_receipt
from .rendering import (
    Lens,
    coupled_photon_draw,
    gaussian_psf_kernels,
    render_expectation,
)
from .utils import atomic_json, safe_relative, sha256_file, stable_hex


def _array_sha256(array: np.ndarray) -> str:
    value = np.asarray(array)
    if not value.flags.c_contiguous:
        value = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def _load_host(row: Any) -> np.ndarray:
    path = Path(str(row.host_pixel_path))
    if sha256_file(path) != str(row.host_payload_sha256):
        raise ContractError(f"host payload drift: {row.host_galaxy_id}")
    with np.load(path, allow_pickle=False) as payload:
        if "cutout" not in payload.files:
            raise ContractError("host payload lacks cutout")
        host = np.array(payload["cutout"], copy=True)
    if (
        host.dtype != np.float32
        or host.shape != (101, 101, 3)
        or not np.isfinite(host).all()
        or _array_sha256(host) != str(row.host_pixel_sha256)
    ):
        raise ContractError(f"host pixel identity drift: {row.host_galaxy_id}")
    return host


def _load_source(
    source_root: Path,
    row: Any,
    cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    source_id = str(row.source_id)
    if source_id not in cache:
        path = safe_relative(source_root, str(row.source_map_relpath))
        if sha256_file(path) != str(row.source_map_sha256):
            raise ContractError(f"source map drift: {source_id}")
        with np.load(path, allow_pickle=False) as payload:
            structured = np.asarray(payload["structured"], dtype=np.float64)
            sersic = np.asarray(payload["sersic"], dtype=np.float64)
            support = np.asarray(payload["common_support"], dtype=bool)
        if (
            structured.shape != sersic.shape
            or structured.shape != support.shape
            or not np.isfinite(structured).all()
            or not np.isfinite(sersic).all()
            or np.any(structured < 0)
            or np.any(sersic < 0)
            or np.any(structured[~support] != 0)
            or np.any(sersic[~support] != 0)
        ):
            raise ContractError(f"invalid source map: {source_id}")
        cache[source_id] = structured, sersic
    return cache[source_id]


def _validate_design(
    design_path: str | Path,
    design_receipt_path: str | Path,
    config: dict[str, Any],
    *,
    require_final: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    receipt = load_json(design_receipt_path)
    expected_status = (
        config["execution"]["required_design_receipt_status"]
        if require_final
        else "psrc_parent_child_design_complete_design_only"
    )
    if (
        receipt.get("status") != expected_status
        or receipt.get("pair_design_sha256") != sha256_file(design_path)
        or receipt.get("config") != config_provenance(config)
        or receipt.get("source_closure") != source_closure(config)
    ):
        raise ContractError("pair design receipt/config/code closure drift")
    design = pd.read_parquet(design_path)
    required = {
        "pair_id",
        "parent_grec_scene_id",
        "parent_grec_row_id",
        "parent_parameter_row_sha256",
        "host_galaxy_id",
        "host_spatial_group_id",
        "host_cell_id",
        "host_pixel_path",
        "host_payload_sha256",
        "host_pixel_sha256",
        "source_id",
        "source_stratum",
        "source_analysis_weight",
        "parent_scene_analysis_weight",
        "source_map_relpath",
        "source_map_sha256",
        "source_flux_nmgy_g",
        "source_flux_nmgy_r",
        "source_flux_nmgy_z",
        "source_beta_x_arcsec",
        "source_beta_y_arcsec",
        "lens_axis_ratio",
        "lens_phi_rad",
        "shear_g1",
        "shear_g2",
        "lens_x0_arcsec",
        "lens_y0_arcsec",
        "theta_e_arcsec",
        "pixel_scale_arcsec",
        "psf_fwhm_g_arcsec",
        "psf_fwhm_r_arcsec",
        "psf_fwhm_z_arcsec",
        "host_key",
        "lens_key",
        "position_key",
        "psf_key",
        "intrinsic_flux_key",
        "renderer_key",
        "photon_uniform_key_base",
    }
    if required - set(design):
        raise ContractError("pair design lacks parent/common identity fields")
    if (
        design["pair_id"].duplicated().any()
        or design["parent_grec_scene_id"].duplicated().any()
        or design["host_galaxy_id"].duplicated().any()
        or not design["source_analysis_weight"].eq(1.0).all()
        or not design["parent_scene_analysis_weight"].eq(1.0).all()
    ):
        raise ContractError("pair/parent/host identities are not one-to-one")
    return design, receipt


def stratified_canary_subset(design: pd.DataFrame, pairs: int) -> pd.DataFrame:
    """Choose score-blind canaries spanning cells, radii and source strata."""
    if not (1 <= pairs <= 64):
        raise ContractError("design-only canary size must be in 1..64")
    ordered = design.copy()
    ordered["canary_order"] = ordered["pair_id"].map(
        lambda value: stable_hex("P_src_v2_stratified_canary", value)
    )
    ordered = ordered.sort_values("canary_order", kind="mergesort")
    dimensions = ["host_cell_id", "theta_e_arcsec", "source_stratum"]
    selected: list[int] = []
    uncovered = {name: set(ordered[name].astype(str).unique()) for name in dimensions}
    while len(selected) < min(pairs, len(ordered)):
        remaining = ordered.drop(index=selected)
        if remaining.empty:
            break
        scores = []
        for index, row in remaining.iterrows():
            gain = sum(str(row[name]) in uncovered[name] for name in dimensions)
            scores.append((gain, str(row["canary_order"]), index))
        _, _, best = max(scores, key=lambda value: (value[0], value[1]))
        selected.append(best)
        for name in dimensions:
            uncovered[name].discard(str(ordered.at[best, name]))
    return ordered.loc[selected].drop(columns="canary_order").reset_index(drop=True)


def render_expectation_bank(
    *,
    design_path: str | Path,
    design_receipt_path: str | Path,
    source_root: str | Path,
    config: dict[str, Any],
    output_dir: Path,
    execution_lock_path: str | Path | None,
    design_only_pairs: int | None = None,
) -> dict[str, Any]:
    """Render deterministic paired expectations; photon noise is a later stage."""
    assert_config_integrity(config)
    final_run = design_only_pairs is None
    lock = None
    if final_run:
        if execution_lock_path is None:
            raise ContractError("final rendering requires an execution lock")
        lock = validate_execution_lock(execution_lock_path, config)
        bindings = lock["bindings"]
        if (
            sha256_file(design_path) != bindings["pair_design"]["sha256"]
            or sha256_file(design_receipt_path)
            != bindings["pair_design_receipt"]["sha256"]
        ):
            raise ContractError("execution lock binds another pair design")
    design, design_receipt = _validate_design(
        design_path, design_receipt_path, config, require_final=final_run
    )
    selected = (
        design
        if final_run
        else stratified_canary_subset(design, int(design_only_pairs))
    )
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    map_dir = output_dir / "expectations"
    map_dir.mkdir()
    source_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    records = []
    inventory = hashlib.sha256()
    render = config["rendering"]
    convergence_maximum = 0.0
    for position, row in enumerate(selected.itertuples(index=False), start=1):
        host = _load_host(row)
        structured, sersic = _load_source(Path(source_root), row, source_cache)
        lens = Lens(
            theta_e_arcsec=float(row.theta_e_arcsec),
            q=float(row.lens_axis_ratio),
            phi_rad=float(row.lens_phi_rad),
            shear_g1=float(row.shear_g1),
            shear_g2=float(row.shear_g2),
            x0_arcsec=float(row.lens_x0_arcsec),
            y0_arcsec=float(row.lens_y0_arcsec),
        )
        oversample = int(render["integration_oversample"])
        kernels = gaussian_psf_kernels(
            fwhm_grz_arcsec=(
                float(row.psf_fwhm_g_arcsec),
                float(row.psf_fwhm_r_arcsec),
                float(row.psf_fwhm_z_arcsec),
            ),
            target_pixel_scale_arcsec=float(row.pixel_scale_arcsec),
            oversample=oversample,
            convention=str(render["psf"]["convention"]),
            truncate_sigma=float(render["psf"]["truncate_sigma"]),
        )
        common = {
            "source_pixel_scale_arcsec": float(
                config["source_construction"]["source_pixel_scale_arcsec"]
            ),
            "intrinsic_flux_nmgy_grz": (
                float(row.source_flux_nmgy_g),
                float(row.source_flux_nmgy_r),
                float(row.source_flux_nmgy_z),
            ),
            "lens": lens,
            "source_center_x_arcsec": float(row.source_beta_x_arcsec),
            "source_center_y_arcsec": float(row.source_beta_y_arcsec),
            "target_shape": tuple(render["target_shape_pixels"]),
            "target_pixel_scale_arcsec": float(row.pixel_scale_arcsec),
            "psf_kernels_grz": kernels,
            "psf_convention": str(render["psf"]["convention"]),
            "oversample": oversample,
        }
        structured_expectation = render_expectation(structured, **common)
        sersic_expectation = render_expectation(sersic, **common)
        convergence = {}
        if final_run:
            rendered_by_oversample = {
                oversample: (structured_expectation, sersic_expectation)
            }
            for check_oversample in render["convergence_oversamples"]:
                check_oversample = int(check_oversample)
                if check_oversample in rendered_by_oversample:
                    continue
                check_kernels = gaussian_psf_kernels(
                    fwhm_grz_arcsec=(
                        float(row.psf_fwhm_g_arcsec),
                        float(row.psf_fwhm_r_arcsec),
                        float(row.psf_fwhm_z_arcsec),
                    ),
                    target_pixel_scale_arcsec=float(row.pixel_scale_arcsec),
                    oversample=check_oversample,
                    convention=str(render["psf"]["convention"]),
                    truncate_sigma=float(render["psf"]["truncate_sigma"]),
                )
                check_common = {
                    **common,
                    "psf_kernels_grz": check_kernels,
                    "oversample": check_oversample,
                }
                rendered_by_oversample[check_oversample] = (
                    render_expectation(structured, **check_common),
                    render_expectation(sersic, **check_common),
                )
            ordered = sorted(rendered_by_oversample)
            for lower, upper in zip(ordered[:-1], ordered[1:], strict=True):
                for arm_index, arm in enumerate(("structured", "sersic")):
                    low = rendered_by_oversample[lower][arm_index].astype(np.float64)
                    high = rendered_by_oversample[upper][arm_index].astype(np.float64)
                    relative_l1 = float(
                        np.abs(low - high).sum() / max(np.abs(high).sum(), 1e-30)
                    )
                    convergence[f"{arm}_{lower}_to_{upper}_relative_l1"] = relative_l1
                    convergence_maximum = max(convergence_maximum, relative_l1)
                    if relative_l1 > float(render["convergence_relative_l1_max"]):
                        raise ContractError(
                            f"P_src renderer convergence failed for "
                            f"{row.pair_id} {arm} {lower}->{upper}: "
                            f"{relative_l1}"
                        )
        filename = f"{row.pair_id}.npz"
        path = map_dir / filename
        temporary_path = map_dir / f".{filename}.partial.npz"
        np.savez_compressed(
            temporary_path,
            structured_expectation=structured_expectation.astype(np.float32),
            sersic_expectation=sersic_expectation.astype(np.float32),
            structured_host_plus_expectation=(host + structured_expectation).astype(
                np.float32
            ),
            sersic_host_plus_expectation=(host + sersic_expectation).astype(np.float32),
        )
        os.replace(temporary_path, path)
        digest = sha256_file(path)
        inventory.update(f"{filename}\t{digest}\n".encode())
        records.append(
            {
                "rendered_pair_row_id": stable_hex(
                    "P_src_v2_expectation", row.pair_id, digest
                ),
                "pair_id": row.pair_id,
                "parent_grec_scene_id": row.parent_grec_scene_id,
                "parent_grec_row_id": row.parent_grec_row_id,
                "host_galaxy_id": row.host_galaxy_id,
                "host_spatial_group_id": row.host_spatial_group_id,
                "host_cell_id": row.host_cell_id,
                "source_id": row.source_id,
                "source_stratum": row.source_stratum,
                "expectation_relpath": f"expectations/{filename}",
                "expectation_sha256": digest,
                "structured_expectation_pixel_sha256": _array_sha256(
                    structured_expectation.astype(np.float32)
                ),
                "sersic_expectation_pixel_sha256": _array_sha256(
                    sersic_expectation.astype(np.float32)
                ),
                "structured_lensed_flux_r": float(
                    structured_expectation[..., 1].sum(dtype=np.float64)
                ),
                "sersic_lensed_flux_r": float(
                    sersic_expectation[..., 1].sum(dtype=np.float64)
                ),
                "post_lensing_rescale_applied": False,
                "photon_noise_applied": False,
                "host_key": row.host_key,
                "lens_key": row.lens_key,
                "position_key": row.position_key,
                "psf_key": row.psf_key,
                "intrinsic_flux_key": row.intrinsic_flux_key,
                "renderer_key": row.renderer_key,
                "photon_uniform_key_base": row.photon_uniform_key_base,
                **convergence,
            }
        )
        if position % 64 == 0:
            print(f"Rendered expectations {position}/{len(selected)}", flush=True)
    manifest = pd.DataFrame(records)
    manifest_path = output_dir / "manifest.parquet"
    temporary_manifest = output_dir / ".manifest.parquet.partial"
    manifest.to_parquet(temporary_manifest, index=False)
    os.replace(temporary_manifest, manifest_path)
    receipt = {
        "schema_version": 2,
        "status": (
            config["execution"]["required_expectation_receipt_status"]
            if final_run
            else "psrc_expectation_bank_design_only_canary"
        ),
        "pairs": len(manifest),
        "detector_scored": False,
        "photon_noise_applied": False,
        "post_lensing_rescale_applied": False,
        "convergence_oversamples": list(map(int, render["convergence_oversamples"])),
        "convergence_relative_l1_maximum": convergence_maximum,
        "convergence_gate": float(render["convergence_relative_l1_max"]),
        "convergence_passed": True,
        "pair_design_sha256": sha256_file(design_path),
        "pair_design_receipt_sha256": sha256_file(design_receipt_path),
        "pair_design_receipt_status": design_receipt["status"],
        "manifest_sha256": sha256_file(manifest_path),
        "expectation_inventory_sha256": inventory.hexdigest(),
        "execution_lock_sha256": (
            sha256_file(execution_lock_path) if final_run else None
        ),
        "config": config_provenance(config),
        "source_closure": source_closure(config),
    }
    atomic_json(output_dir / "EXPECTATION_RECEIPT.json", receipt)
    return receipt


def materialize_photon_bank(
    *,
    design_path: str | Path,
    expectation_root: str | Path,
    expectation_receipt_path: str | Path,
    draw_count_receipt_path: str | Path,
    execution_lock_path: str | Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Materialize complete parent-bound photon draws without scoring."""
    lock = validate_execution_lock(execution_lock_path, config)
    if (
        sha256_file(draw_count_receipt_path)
        != lock["bindings"]["draw_count_receipt"]["sha256"]
    ):
        raise ContractError("execution lock binds another draw-count receipt")
    threshold_path = resolve_binding(
        execution_lock_path, lock["bindings"]["threshold_receipt"]
    )
    draw_receipt, draws = validate_draw_count_receipt(
        draw_count_receipt_path,
        config,
        expected_final_design_path=design_path,
        expected_final_design_receipt_path=resolve_binding(
            execution_lock_path, lock["bindings"]["pair_design_receipt"]
        ),
        expected_scorer_receipt_path=resolve_binding(
            execution_lock_path, lock["bindings"]["scorer_receipt"]
        ),
        expected_threshold_receipt_path=threshold_path,
        production=True,
    )
    expectation_receipt = load_json(expectation_receipt_path)
    expectation_manifest_path = Path(expectation_root) / "manifest.parquet"
    if (
        expectation_receipt.get("status")
        != config["execution"]["required_expectation_receipt_status"]
        or expectation_receipt.get("manifest_sha256")
        != sha256_file(expectation_manifest_path)
        or expectation_receipt.get("pair_design_sha256") != sha256_file(design_path)
        or expectation_receipt.get("execution_lock_sha256")
        != sha256_file(execution_lock_path)
        or expectation_receipt.get("config") != config_provenance(config)
    ):
        raise ContractError("expectation bank receipt is not frozen/bound")
    design = pd.read_parquet(design_path)
    expectations = pd.read_parquet(expectation_manifest_path)
    joined = design.merge(
        expectations[
            [
                "pair_id",
                "rendered_pair_row_id",
                "expectation_relpath",
                "expectation_sha256",
            ]
        ],
        on="pair_id",
        validate="one_to_one",
    )
    if len(joined) != len(design):
        raise ContractError("expectation bank does not exactly cover design")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    pair_dir = output_dir / "pairs"
    pair_dir.mkdir()
    gain = float(config["photon_noise"]["gain_e_per_nmgy"])
    records = []
    inventory = hashlib.sha256()
    for position, row in enumerate(
        joined.sort_values("pair_id").itertuples(index=False), start=1
    ):
        path = safe_relative(Path(expectation_root), row.expectation_relpath)
        if sha256_file(path) != row.expectation_sha256:
            raise ContractError("expectation artifact drift")
        with np.load(path, allow_pickle=False) as payload:
            structured_expectation = np.asarray(
                payload["structured_expectation"], dtype=np.float64
            )
            sersic_expectation = np.asarray(
                payload["sersic_expectation"], dtype=np.float64
            )
        host = _load_host(row)
        structured_draws = np.empty((draws, *host.shape), dtype=np.float32)
        sersic_draws = np.empty((draws, *host.shape), dtype=np.float32)
        for draw_index in range(draws):
            structured_arc, sersic_arc, keys = coupled_photon_draw(
                structured_expectation,
                sersic_expectation,
                gain_e_per_nmgy=gain,
                noise_key_base=row.photon_uniform_key_base,
                draw_index=draw_index,
            )
            structured_draws[draw_index] = host + structured_arc
            sersic_draws[draw_index] = host + sersic_arc
            records.append(
                {
                    "pair_id": row.pair_id,
                    "rendered_pair_row_id": row.rendered_pair_row_id,
                    "draw_index": draw_index,
                    "host_galaxy_id": row.host_galaxy_id,
                    "host_spatial_group_id": row.host_spatial_group_id,
                    "host_cell_id": row.host_cell_id,
                    "source_id": row.source_id,
                    "source_stratum": row.source_stratum,
                    "photon_pair_relpath": f"pairs/{row.pair_id}.npz",
                    "shared_noise_key": keys["shared_noise_key"],
                    "structured_extra_noise_key": keys["structured_extra_noise_key"],
                    "sersic_extra_noise_key": keys["sersic_extra_noise_key"],
                }
            )
        output_path = pair_dir / f"{row.pair_id}.npz"
        temporary_output = pair_dir / f".{row.pair_id}.partial.npz"
        np.savez_compressed(
            temporary_output,
            structured_cutouts=structured_draws,
            sersic_cutouts=sersic_draws,
        )
        os.replace(temporary_output, output_path)
        digest = sha256_file(output_path)
        inventory.update(f"{row.pair_id}.npz\t{digest}\n".encode())
        for record in records[-draws:]:
            record["photon_pair_sha256"] = digest
        if position % 64 == 0:
            print(f"Materialized photons {position}/{len(joined)}", flush=True)
    manifest = pd.DataFrame(records)
    expected_draws = set(range(draws))
    observed_sets = manifest.groupby("pair_id")["draw_index"].apply(
        lambda values: set(map(int, values))
    )
    if (
        len(manifest) != len(design) * draws
        or manifest.duplicated(["pair_id", "draw_index"]).any()
        or not observed_sets.map(lambda value: value == expected_draws).all()
    ):
        raise ContractError("photon bank draw-index coverage is incomplete")
    manifest_path = output_dir / "manifest.parquet"
    temporary_manifest = output_dir / ".manifest.parquet.partial"
    manifest.to_parquet(temporary_manifest, index=False)
    os.replace(temporary_manifest, manifest_path)
    receipt = {
        "schema_version": 2,
        "status": config["execution"]["required_photon_receipt_status"],
        "pairs": len(design),
        "draw_count": draws,
        "rows": len(manifest),
        "draw_indices": list(range(draws)),
        "detector_scored": False,
        "expectation_manifest_sha256": sha256_file(expectation_manifest_path),
        "expectation_receipt_sha256": sha256_file(expectation_receipt_path),
        "draw_count_receipt_sha256": sha256_file(draw_count_receipt_path),
        "draw_count_receipt_status": draw_receipt["status"],
        "execution_lock_sha256": sha256_file(execution_lock_path),
        "manifest_sha256": sha256_file(manifest_path),
        "pair_inventory_sha256": inventory.hexdigest(),
        "config": config_provenance(config),
        "source_closure": source_closure(config),
    }
    atomic_json(output_dir / "PHOTON_RECEIPT.json", receipt)
    return receipt
