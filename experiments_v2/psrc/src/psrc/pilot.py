from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    ContractError,
    assert_config_integrity,
    config_provenance,
    load_json,
    source_closure,
    validate_scorer_receipt,
)
from .design import build_parent_bound_design, validate_source_receipt
from .draw_count import _validate_pilot_and_final_designs
from .execution import _load_host, _load_source
from .rendering import (
    Lens,
    coupled_photon_draw,
    gaussian_psf_kernels,
    render_expectation,
)
from .scoring import _native_float32_logits
from .threshold import validate_threshold_receipt
from .utils import atomic_json, sha256_file, stable_hex


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _bound(root: Path, binding: dict[str, Any]) -> Path:
    path = root / str(binding["path"])
    if not path.is_file() or sha256_file(path) != str(binding["sha256"]):
        raise ContractError(f"pilot parent authority differs: {binding['path']}")
    return path


def _select_pilot_parents(scenes: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rules = config["pilot_parent"]
    required = {
        "scene_id",
        "grec_row_id",
        "host_galaxy_id",
        "pilot_cell",
        "theta_e_arcsec",
        "physical_group_id",
        "historical_healpix_128",
        "host_spatial_coordinate_id",
    }
    if required - set(scenes):
        raise ContractError("pilot parent scenes lack required identities")
    hosts = (
        scenes.sort_values("host_galaxy_id")
        .groupby("host_galaxy_id", as_index=False)
        .first()
    )
    hosts["pilot_order"] = hosts["host_galaxy_id"].map(
        lambda value: stable_hex(str(rules["selection_namespace"]), "host", value)
    )
    selected_hosts = (
        hosts.sort_values(["pilot_cell", "pilot_order"], kind="mergesort")
        .groupby("pilot_cell", group_keys=False)
        .head(int(rules["hosts_per_cell"]))
        .copy()
    )
    selected_hosts["parent_magnitude_quantile"] = (
        selected_hosts.groupby("pilot_cell", group_keys=False)["source_r_mag"]
        .transform(
            lambda values: pd.qcut(
                values.rank(method="first"),
                q=int(
                    config["parent_grec"]["subset"][
                        "source_magnitude_quantiles_within_cell"
                    ]
                ),
                labels=False,
            )
        )
        .astype(int)
    )
    selected = scenes.merge(
        selected_hosts[["host_galaxy_id", "parent_magnitude_quantile"]],
        on="host_galaxy_id",
        how="inner",
        validate="many_to_one",
    )
    selected["host_cell_id"] = (
        selected["psf_bin"].map(lambda value: f"{float(value):.2f}")
        + "/"
        + selected["depth_bin"].map(lambda value: f"{float(value):.1f}")
    )
    selected["host_physical_group_id"] = selected["physical_group_id"].astype(str)
    selected["host_healpix_128"] = selected["historical_healpix_128"].astype("int64")
    if (
        len(selected) != int(rules["expected_pairs"])
        or selected["host_galaxy_id"].nunique() != int(rules["expected_distinct_hosts"])
        or selected["scene_id"].duplicated().any()
        or not selected.groupby(["host_cell_id", "theta_e_arcsec"])
        .size()
        .eq(int(config["crossing"]["rows_per_cell_radius"]))
        .all()
    ):
        raise ContractError("pilot parent subset balance differs")
    return selected


def prepare_pilot_design(
    *,
    source_manifest_path: str | Path,
    source_receipt_path: str | Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    assert_config_integrity(config)
    if config["status"] != "frozen_pre_scoring":
        raise ContractError("pilot design requires frozen P_src configuration")
    root = Path(config["scoring"]["repo_root"]).resolve()
    scene_path = _bound(root, config["pilot_parent"]["scene_manifest"])
    scene_receipt_path = _bound(root, config["pilot_parent"]["scene_receipt"])
    freeze_path = _bound(root, config["pilot_parent"]["design_freeze_receipt"])
    scene_receipt = load_json(scene_receipt_path)
    freeze = load_json(freeze_path)
    if (
        scene_receipt.get("status") != config["pilot_parent"]["scene_receipt"]["status"]
        or scene_receipt.get("design_scene_manifest_sha256") != sha256_file(scene_path)
        or freeze.get("status")
        != config["pilot_parent"]["design_freeze_receipt"]["status"]
        or freeze.get("design_scene_manifest_sha256") != sha256_file(scene_path)
        or freeze.get("final_outcomes_used") is not False
    ):
        raise ContractError("P_noise design-parent authority is not reusable")
    sources, source_receipt = validate_source_receipt(
        source_manifest_path,
        source_receipt_path,
        config,
        require_final=False,
        expected_role="pilot",
    )
    parents = _select_pilot_parents(pd.read_parquet(scene_path), config)
    design = build_parent_bound_design(
        parents,
        sources,
        config,
        require_unique_hosts=False,
    )
    design["global_role"] = "design"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    path = output_dir / "pair_design.parquet"
    _atomic_parquet(design, path)
    receipt = {
        "schema_version": 1,
        "status": config["photon_noise"]["required_pilot_design_receipt_status"],
        "global_role": "design",
        "pairs": len(design),
        "distinct_hosts": int(design["host_galaxy_id"].nunique()),
        "distinct_sources": int(design["source_id"].nunique()),
        "pair_design_sha256": sha256_file(path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_receipt_sha256": sha256_file(source_receipt_path),
        "source_receipt_status": source_receipt["status"],
        "parent_scene_manifest_sha256": sha256_file(scene_path),
        "parent_scene_receipt_sha256": sha256_file(scene_receipt_path),
        "parent_design_freeze_receipt_sha256": sha256_file(freeze_path),
        "detector_scores_used": False,
        "final_outcomes_used": False,
        "config": config_provenance(config),
        "source_closure": source_closure(config),
    }
    atomic_json(output_dir / "PAIR_DESIGN_RECEIPT.json", receipt)
    return receipt


def build_role_ledger(
    *,
    pilot_design_path: str | Path,
    pilot_design_receipt_path: str | Path,
    final_design_path: str | Path,
    final_design_receipt_path: str | Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    pilot = pd.read_parquet(pilot_design_path)
    final = pd.read_parquet(final_design_path)
    pilot_receipt = load_json(pilot_design_receipt_path)
    final_receipt = load_json(final_design_receipt_path)
    if (
        pilot_receipt.get("pair_design_sha256") != sha256_file(pilot_design_path)
        or final_receipt.get("pair_design_sha256") != sha256_file(final_design_path)
        or pilot_receipt.get("global_role") != "design"
        or final_receipt.get("status")
        != config["execution"]["required_design_receipt_status"]
    ):
        raise ContractError("role-ledger design authority differs")
    columns = [
        str(value)
        for value in config["photon_noise"]["pilot_anti_join_identity_columns"]
    ]
    if set(columns) - set(pilot) or set(columns) - set(final):
        raise ContractError("design lacks role-ledger identity columns")
    ledger = pd.concat(
        [
            pilot[columns].assign(global_role="design"),
            final[columns].assign(global_role="experiment"),
        ],
        ignore_index=True,
    )
    if ledger["pair_id"].duplicated().any() or set(
        pilot["source_id"].astype(str)
    ) & set(final["source_id"].astype(str)):
        raise ContractError("pilot and experiment roles are not disjoint")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    path = output_dir / "ROLE_LEDGER.parquet"
    _atomic_parquet(ledger, path)
    receipt = {
        "schema_version": 1,
        "status": config["photon_noise"]["required_role_ledger_receipt_status"],
        "role_ledger_sha256": sha256_file(path),
        "pilot_design_sha256": sha256_file(pilot_design_path),
        "pilot_design_receipt_sha256": sha256_file(pilot_design_receipt_path),
        "final_design_sha256": sha256_file(final_design_path),
        "final_design_receipt_sha256": sha256_file(final_design_receipt_path),
        "pilot_rows": len(pilot),
        "final_rows": len(final),
        "detector_scores_used": False,
        "final_outcomes_used": False,
        "config": config_provenance(config),
        "source_closure": source_closure(config),
    }
    atomic_json(output_dir / "ROLE_LEDGER_RECEIPT.json", receipt)
    return receipt


def _pair_expectations(
    row: Any,
    *,
    source_root: Path,
    source_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    host = _load_host(row)
    structured, sersic = _load_source(source_root, row, source_cache)
    lens = Lens(
        theta_e_arcsec=float(row.theta_e_arcsec),
        q=float(row.lens_axis_ratio),
        phi_rad=float(row.lens_phi_rad),
        shear_g1=float(row.shear_g1),
        shear_g2=float(row.shear_g2),
        x0_arcsec=float(row.lens_x0_arcsec),
        y0_arcsec=float(row.lens_y0_arcsec),
    )
    rendering = config["rendering"]
    oversample = int(rendering["integration_oversample"])
    kernels = gaussian_psf_kernels(
        fwhm_grz_arcsec=(
            float(row.psf_fwhm_g_arcsec),
            float(row.psf_fwhm_r_arcsec),
            float(row.psf_fwhm_z_arcsec),
        ),
        target_pixel_scale_arcsec=float(row.pixel_scale_arcsec),
        oversample=oversample,
        convention=str(rendering["psf"]["convention"]),
        truncate_sigma=float(rendering["psf"]["truncate_sigma"]),
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
        "target_shape": tuple(rendering["target_shape_pixels"]),
        "target_pixel_scale_arcsec": float(row.pixel_scale_arcsec),
        "psf_kernels_grz": kernels,
        "psf_convention": str(rendering["psf"]["convention"]),
        "oversample": oversample,
    }
    return (
        host,
        render_expectation(structured, **common),
        render_expectation(sersic, **common),
    )


def run_draw_pilot(
    *,
    pilot_design_path: str | Path,
    pilot_design_receipt_path: str | Path,
    final_design_path: str | Path,
    final_design_receipt_path: str | Path,
    role_ledger_path: str | Path,
    role_ledger_receipt_path: str | Path,
    source_root: str | Path,
    scorer_receipt_path: str | Path,
    threshold_receipt_path: str | Path,
    config: dict[str, Any],
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    pilot, _, _ = _validate_pilot_and_final_designs(
        pilot_design_path=pilot_design_path,
        pilot_design_receipt_path=pilot_design_receipt_path,
        final_design_path=final_design_path,
        final_design_receipt_path=final_design_receipt_path,
        role_ledger_path=role_ledger_path,
        role_ledger_receipt_path=role_ledger_receipt_path,
        config=config,
        canary_mode=False,
    )
    validate_scorer_receipt(scorer_receipt_path, config, require_final=True)
    _, threshold = validate_threshold_receipt(
        threshold_receipt_path, config, require_final=True
    )
    pilot = pilot.sort_values("pair_id", ignore_index=True)
    candidates = [
        int(value) for value in config["photon_noise"]["candidate_draw_counts"]
    ]
    draws = max(candidates)
    metadata = [
        (str(row.pair_id), draw)
        for row in pilot.itertuples(index=False)
        for draw in range(draws)
    ]

    def images() -> Iterable[np.ndarray]:
        source_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for position, row in enumerate(pilot.itertuples(index=False), start=1):
            host, structured, sersic = _pair_expectations(
                row,
                source_root=Path(source_root),
                source_cache=source_cache,
                config=config,
            )
            for draw in range(draws):
                structured_draw, sersic_draw, _ = coupled_photon_draw(
                    structured,
                    sersic,
                    gain_e_per_nmgy=float(config["photon_noise"]["gain_e_per_nmgy"]),
                    noise_key_base=str(row.photon_uniform_key_base),
                    draw_index=draw,
                )
                yield np.transpose(
                    (host + structured_draw).astype(np.float32), (2, 0, 1)
                )
                yield np.transpose((host + sersic_draw).astype(np.float32), (2, 0, 1))
            if position % 64 == 0:
                print(
                    f"Streamed P_src pilot pairs {position}/{len(pilot)}",
                    flush=True,
                )

    logits, scoring = _native_float32_logits(
        images(),
        config,
        scorer_receipt_path=scorer_receipt_path,
        require_final_scorer=True,
        device=device,
    )
    if len(logits) != 2 * len(metadata):
        raise ContractError("pilot scorer row coverage differs")
    scores = pd.DataFrame(metadata, columns=["pair_id", "draw_index"])
    structured = logits[0::2]
    sersic = logits[1::2]
    scores["structured_raw_logit_float32"] = structured
    scores["structured_raw_logit_float32_bits"] = structured.view(np.uint32)
    scores["sersic_raw_logit_float32"] = sersic
    scores["sersic_raw_logit_float32_bits"] = sersic.view(np.uint32)
    scores["structured_detected"] = structured >= threshold
    scores["sersic_detected"] = sersic >= threshold
    scores_path = output_dir / "pilot_scores.parquet"
    _atomic_parquet(scores, scores_path)
    receipt = {
        "schema_version": 1,
        "status": config["photon_noise"]["required_pilot_score_receipt_status"],
        "global_role": "design",
        "rows": len(scores),
        "pairs": int(scores["pair_id"].nunique()),
        "draw_count": draws,
        "candidate_draw_counts": candidates,
        "precision_target": config["photon_noise"]["paired_mcse_target_absolute_risk"],
        "pilot_scores_sha256": sha256_file(scores_path),
        "pilot_design_sha256": sha256_file(pilot_design_path),
        "pilot_design_receipt_sha256": sha256_file(pilot_design_receipt_path),
        "final_design_sha256": sha256_file(final_design_path),
        "final_design_receipt_sha256": sha256_file(final_design_receipt_path),
        "role_ledger_sha256": sha256_file(role_ledger_path),
        "role_ledger_receipt_sha256": sha256_file(role_ledger_receipt_path),
        "scorer_receipt_sha256": sha256_file(scorer_receipt_path),
        "threshold_receipt_sha256": sha256_file(threshold_receipt_path),
        "threshold_float32_bits": int(threshold.view(np.uint32)),
        "comparison": ">=",
        "scorer_closure_sha256": validate_scorer_receipt(
            scorer_receipt_path, config, require_final=True
        )["scorer_closure_sha256"],
        "scoring": scoring,
        "final_outcomes_used": False,
        "config": config_provenance(config),
        "source_closure": source_closure(config),
    }
    atomic_json(output_dir / "PILOT_SCORE_RECEIPT.json", receipt)
    return receipt
