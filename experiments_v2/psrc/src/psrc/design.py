from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    ContractError,
    assert_config_integrity,
    config_provenance,
    load_json,
    sha256_json,
    source_closure,
    stable_id,
    verify_parent_grec_authority,
    verify_source_closure,
)
from .utils import (
    atomic_json,
    reject_outcome_columns,
    safe_relative,
    sha256_file,
    stable_hex,
)


def _host_cell(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["psf_bin"].map(lambda value: f"{float(value):.2f}")
        + "/"
        + frame["depth_bin"].map(lambda value: f"{float(value):.1f}")
    )


def select_parent_scene_subset(
    scenes: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Select exact parent rows; never generate or modify scene parameters."""
    assert_config_integrity(config)
    reject_outcome_columns(
        scenes,
        [str(value) for value in config["selection"]["forbidden_column_tokens"]],
    )
    required = {
        "scene_id",
        "grec_row_id",
        "parameter_row_sha256",
        "host_galaxy_id",
        "host_spatial_group_id",
        "host_pixel_sha256",
        "host_pixel_path",
        "source_parameter_id",
        "source_nuisance_id",
        "lens_parameter_id",
        "lens_nuisance_id",
        "render_config_id",
        "render_contract_sha256",
        "base_draw_key",
        "psf_bin",
        "depth_bin",
        "source_r_mag",
        "theta_e_arcsec",
    }
    missing = sorted(required - set(scenes))
    if missing:
        raise ContractError(f"G_rec parent scene table lacks: {missing}")
    if scenes["scene_id"].astype(str).duplicated().any():
        raise ContractError("G_rec parent scene IDs are not unique")
    radii = np.asarray(config["parent_grec"]["required_radii_arcsec"], dtype=np.float64)
    per_host = scenes.groupby("host_galaxy_id")["theta_e_arcsec"].apply(
        lambda values: np.array_equal(np.sort(values.to_numpy(dtype=np.float64)), radii)
    )
    if not per_host.all():
        raise ContractError("a parent host lacks the complete registered radius set")
    shared = [
        "source_r_mag",
        "source_nuisance_id",
        "lens_nuisance_id",
        "base_draw_key",
        "psf_bin",
        "depth_bin",
        "host_spatial_group_id",
    ]
    if not scenes.groupby("host_galaxy_id")[shared].nunique().eq(1).all().all():
        raise ContractError("parent shared nuisance changed across radius rows")

    rules = config["parent_grec"]["subset"]
    magnitude_quantiles = int(rules["source_magnitude_quantiles_within_cell"])
    per_stratum = int(rules["hosts_per_cell_radius_magnitude_quantile"])
    namespace = str(rules["selection_namespace"])
    hosts = (
        scenes.sort_values(["host_galaxy_id", "theta_e_arcsec"])
        .groupby("host_galaxy_id", as_index=False)
        .first()
    )
    hosts["host_cell_id"] = _host_cell(hosts)
    hosts["parent_subset_order"] = hosts["host_galaxy_id"].map(
        lambda value: stable_hex(namespace, "host", value)
    )
    hosts["parent_magnitude_quantile"] = (
        hosts.groupby("host_cell_id", group_keys=False)["source_r_mag"]
        .transform(
            lambda values: pd.qcut(
                values.rank(method="first"),
                q=magnitude_quantiles,
                labels=False,
            )
        )
        .astype(int)
    )
    cells = sorted(hosts["host_cell_id"].unique())
    if len(cells) != int(rules["host_cells"]):
        raise ContractError("G_rec host-cell count differs from subset contract")
    selected_rows: list[pd.Series] = []
    scene_index = scenes.set_index(["host_galaxy_id", "theta_e_arcsec"])
    for cell in cells:
        for magnitude_quantile in range(magnitude_quantiles):
            candidates = hosts[
                hosts["host_cell_id"].eq(cell)
                & hosts["parent_magnitude_quantile"].eq(magnitude_quantile)
            ].sort_values("parent_subset_order", kind="mergesort")
            needed = len(radii) * per_stratum
            if len(candidates) < needed:
                raise ContractError(
                    f"parent cell/magnitude stratum {cell}/{magnitude_quantile} "
                    f"has {len(candidates)} hosts, needs {needed}"
                )
            chosen = candidates.head(needed).reset_index(drop=True)
            for radius_index, radius in enumerate(radii):
                start = radius_index * per_stratum
                for host in chosen.iloc[start : start + per_stratum].itertuples(
                    index=False
                ):
                    parent = scene_index.loc[
                        (str(host.host_galaxy_id), float(radius))
                    ].copy()
                    parent["host_galaxy_id"] = str(host.host_galaxy_id)
                    parent["theta_e_arcsec"] = float(radius)
                    parent["host_cell_id"] = cell
                    parent["parent_magnitude_quantile"] = magnitude_quantile
                    selected_rows.append(parent)
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    expected_rows = int(rules["expected_parent_scenes"])
    expected_hosts = int(rules["expected_distinct_hosts"])
    if (
        len(selected) != expected_rows
        or selected["host_galaxy_id"].nunique() != expected_hosts
        or selected["host_galaxy_id"].duplicated().any()
        or not selected.groupby(
            ["host_cell_id", "theta_e_arcsec", "parent_magnitude_quantile"]
        )
        .size()
        .eq(per_stratum)
        .all()
    ):
        raise ContractError("parent scene subset balance/host uniqueness failed")
    return selected


def build_parent_bound_design(
    parent_subset: pd.DataFrame,
    sources: pd.DataFrame,
    config: dict[str, Any],
    *,
    require_unique_hosts: bool = True,
) -> pd.DataFrame:
    """Pair source constructions with exact immutable G_rec parent rows."""
    assert_config_integrity(config)
    crossing = config["crossing"]
    expected_pairs = int(crossing["expected_pairs"])
    expected_source_uses = int(crossing["expected_source_uses"])
    if len(parent_subset) != expected_pairs:
        raise ContractError("parent subset count differs from crossing contract")
    if len(sources) * expected_source_uses != expected_pairs:
        raise ContractError("source count/use degree does not cover parent subset")
    expected_strata = int(crossing["expected_source_strata"])
    source_groups = {
        str(stratum): frame.sort_values("selection_hash", kind="mergesort").reset_index(
            drop=True
        )
        for stratum, frame in sources.groupby("source_stratum", sort=True)
    }
    strata = sorted(source_groups)
    if (
        len(strata) != expected_strata
        or len({len(frame) for frame in source_groups.values()}) != 1
    ):
        raise ContractError("selected source strata are absent or unbalanced")
    sources_per_stratum = len(next(iter(source_groups.values())))
    if sources_per_stratum * expected_source_uses != (
        len(parent_subset) // expected_strata
    ):
        raise ContractError("source cycles cannot evenly cover parent blocks")
    parent_order = parent_subset.copy()
    namespace = str(crossing["assignment_namespace"])
    parent_order["assignment_order"] = parent_order["scene_id"].map(
        lambda value: stable_hex(namespace, "parent", value)
    )
    records: list[dict[str, Any]] = []
    parent_columns = list(parent_subset.columns)
    block_columns = ["host_cell_id", "theta_e_arcsec"]
    blocks = list(parent_order.groupby(block_columns, sort=True, observed=True))
    rows_per_block = int(crossing["rows_per_cell_radius"])
    if len(blocks) * rows_per_block != expected_pairs:
        raise ContractError("parent cell/radius block count differs from contract")
    for block_index, (_, block) in enumerate(blocks):
        block = block.sort_values(
            ["parent_magnitude_quantile", "assignment_order"],
            kind="mergesort",
        ).reset_index(drop=True)
        if len(block) != rows_per_block:
            raise ContractError("parent cell/radius block is not exactly 16 rows")
        for slot, parent_record in enumerate(block.to_dict(orient="records")):
            stratum = strata[(slot + block_index) % expected_strata]
            source = source_groups[stratum].iloc[block_index % sources_per_stratum]
            source_record = source.to_dict()
            pair_id = stable_id(
                "psrcpair",
                [
                    parent_record["scene_id"],
                    parent_record["parameter_row_sha256"],
                    source_record["source_id"],
                    source_record["source_map_sha256"],
                ],
            )
            flux_id = stable_id(
                "flux",
                {
                    "source_nuisance_id": parent_record["source_nuisance_id"],
                    "g": parent_record["source_flux_nmgy_g"],
                    "r": parent_record["source_flux_nmgy_r"],
                    "z": parent_record["source_flux_nmgy_z"],
                },
            )
            position_id = stable_id(
                "position",
                {
                    "source_parameter_id": parent_record["source_parameter_id"],
                    "x": parent_record["source_beta_x_arcsec"],
                    "y": parent_record["source_beta_y_arcsec"],
                },
            )
            psf_id = stable_id(
                "psf",
                {
                    "render_config_id": parent_record["render_config_id"],
                    "g": parent_record["psf_fwhm_g_arcsec"],
                    "r": parent_record["psf_fwhm_r_arcsec"],
                    "z": parent_record["psf_fwhm_z_arcsec"],
                    "pixel_scale": parent_record["pixel_scale_arcsec"],
                },
            )
            derivative_ancestry_sha256 = sha256_json(
                {
                    "parent_grec_scene_id": parent_record["scene_id"],
                    "parent_grec_row_id": parent_record["grec_row_id"],
                    "parent_parameter_row_sha256": parent_record[
                        "parameter_row_sha256"
                    ],
                    "host_parent_id": parent_record["host_parent_id"],
                    "host_physical_group_id": parent_record["host_physical_group_id"],
                    "host_spatial_group_id": parent_record["host_spatial_group_id"],
                    "host_spatial_coordinate_id": parent_record[
                        "host_spatial_coordinate_id"
                    ],
                    "host_pixel_sha256": parent_record["host_pixel_sha256"],
                    "source_id": source_record["source_id"],
                    "source_image_sha256": source_record["source_image_sha256"],
                    "source_map_sha256": source_record["source_map_sha256"],
                }
            )
            pair_id = stable_id("psrcpair", [pair_id, derivative_ancestry_sha256])
            record = {
                **parent_record,
                "pair_id": pair_id,
                "parent_grec_scene_id": str(parent_record["scene_id"]),
                "parent_grec_row_id": str(parent_record["grec_row_id"]),
                "parent_parameter_row_sha256": str(
                    parent_record["parameter_row_sha256"]
                ),
                "parent_source_id": str(parent_record["source_parameter_id"]),
                "parent_source_nuisance_id": str(parent_record["source_nuisance_id"]),
                "parent_lens_id": str(parent_record["lens_parameter_id"]),
                "parent_lens_nuisance_id": str(parent_record["lens_nuisance_id"]),
                "parent_source_position_id": position_id,
                "parent_psf_id": psf_id,
                "parent_intrinsic_flux_id": flux_id,
                "empirical_source_id": str(source_record["source_id"]),
                "source_id": str(source_record["source_id"]),
                "source_stratum": str(source_record["source_stratum"]),
                "source_stratum_slot": slot,
                "parent_cell_radius_block_index": block_index,
                "source_analysis_weight": 1.0,
                "source_population_sensitivity_weight": float(
                    source_record["source_population_sensitivity_weight"]
                ),
                "parent_scene_analysis_weight": 1.0,
                "source_map_relpath": str(source_record["source_map_relpath"]),
                "source_map_sha256": str(source_record["source_map_sha256"]),
                "source_image_sha256": str(source_record["source_image_sha256"]),
                "source_psf_image_sha256": str(
                    source_record["source_psf_image_sha256"]
                ),
                "derivative_ancestry_sha256": derivative_ancestry_sha256,
                "structured_arm_id": f"{pair_id}_structured",
                "sersic_arm_id": f"{pair_id}_sersic",
                "intrinsic_flux_policy": "fixed_parent_intrinsic_per_band",
                "host_key": str(parent_record["host_pixel_sha256"]),
                "lens_key": str(parent_record["lens_parameter_id"]),
                "position_key": position_id,
                "psf_key": psf_id,
                "intrinsic_flux_key": flux_id,
                "renderer_key": str(config["rendering"]["primary_renderer_id"]),
                "photon_uniform_key_base": stable_id(
                    "photon", [pair_id, parent_record["base_draw_key"]]
                ),
            }
            records.append(record)
    design = pd.DataFrame(records)
    if (
        len(design) != expected_pairs
        or design["pair_id"].duplicated().any()
        or not design.groupby("source_id").size().eq(expected_source_uses).all()
        or design["parent_grec_scene_id"].duplicated().any()
        or (require_unique_hosts and design["host_galaxy_id"].duplicated().any())
        or not design.groupby(["host_cell_id", "theta_e_arcsec"])["source_stratum"]
        .nunique()
        .eq(expected_strata)
        .all()
        or not design["parent_scene_analysis_weight"].eq(1.0).all()
    ):
        raise ContractError("parent/source crossing identities or degrees failed")
    parent_by_scene = parent_subset.set_index("scene_id")
    for row in design.to_dict(orient="records"):
        original = parent_by_scene.loc[row["parent_grec_scene_id"]]
        for column in parent_columns:
            observed = row[column]
            expected = (
                row["parent_grec_scene_id"]
                if column == "scene_id"
                else original[column]
            )
            if pd.isna(observed) and pd.isna(expected):
                continue
            if observed != expected:
                raise ContractError(f"parent parameter changed in child: {column}")
    return design.sort_values("pair_id", kind="mergesort").reset_index(drop=True)


def validate_source_receipt(
    source_manifest_path: str | Path,
    source_receipt_path: str | Path,
    config: dict[str, Any],
    *,
    require_final: bool,
    expected_role: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    receipt = load_json(source_receipt_path)
    expected_status = (
        config["execution"]["required_source_receipt_status"]
        if require_final
        else "restricted_empirical_source_construction_design_only"
    )
    if (
        receipt.get("status") != expected_status
        or receipt.get("source_manifest_sha256") != sha256_file(source_manifest_path)
        or receipt.get("config") != config_provenance(config)
        or (expected_role is not None and receipt.get("source_role") != expected_role)
    ):
        raise ContractError("source construction receipt is stale")
    verify_source_closure(receipt.get("source_closure", {}), config)
    manifest = pd.read_parquet(source_manifest_path)
    root = Path(source_manifest_path).parent
    inventory = hashlib.sha256()
    for row in manifest.sort_values("source_id").itertuples(index=False):
        path = safe_relative(root, str(row.source_map_relpath))
        observed = sha256_file(path)
        if observed != str(row.source_map_sha256):
            raise ContractError(f"source map drift: {row.source_id}")
        inventory.update(f"{row.source_map_relpath}\t{observed}\n".encode())
    if inventory.hexdigest() != receipt.get("source_map_inventory_sha256"):
        raise ContractError("complete source-map inventory drift")
    return manifest, receipt


def prepare_parent_child_design(
    *,
    final_host_manifest_path: str | Path,
    final_host_receipt_path: str | Path,
    scene_table_path: str | Path,
    scene_receipt_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    source_manifest_path: str | Path,
    source_receipt_path: str | Path,
    config: dict[str, Any],
    output_dir: Path,
    final_status: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    assert_config_integrity(config)
    if final_status and config["status"] != "frozen_pre_scoring":
        raise ContractError("final child design requires frozen P_src config")
    scenes, authority = verify_parent_grec_authority(
        final_host_manifest_path=final_host_manifest_path,
        final_host_receipt_path=final_host_receipt_path,
        scene_table_path=scene_table_path,
        scene_receipt_path=scene_receipt_path,
        parent_manifest_path=parent_manifest_path,
        parent_receipt_path=parent_receipt_path,
        config=config,
    )
    sources, source_receipt = validate_source_receipt(
        source_manifest_path,
        source_receipt_path,
        config,
        require_final=final_status,
        expected_role="experiment" if final_status else None,
    )
    subset = select_parent_scene_subset(scenes, config)
    design = build_parent_bound_design(subset, sources, config)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    design_path = output_dir / "pair_design.parquet"
    design.to_parquet(design_path, index=False)
    block_balance = design.groupby(
        ["host_cell_id", "theta_e_arcsec", "source_stratum"]
    ).size()
    source_reuse = design.groupby("source_id").size()
    receipt = {
        "schema_version": 2,
        "status": (
            config["execution"]["required_design_receipt_status"]
            if final_status
            else "psrc_parent_child_design_complete_design_only"
        ),
        "pairs": len(design),
        "distinct_parent_scenes": int(design["parent_grec_scene_id"].nunique()),
        "distinct_hosts": int(design["host_galaxy_id"].nunique()),
        "distinct_host_spatial_groups": int(design["host_spatial_group_id"].nunique()),
        "distinct_sources": int(design["source_id"].nunique()),
        "radii_arcsec": sorted(design["theta_e_arcsec"].unique().tolist()),
        "host_cell_radius_blocks": int(
            design[["host_cell_id", "theta_e_arcsec"]].drop_duplicates().shape[0]
        ),
        "source_strata_per_host_cell_radius": int(
            design.groupby(["host_cell_id", "theta_e_arcsec"])["source_stratum"]
            .nunique()
            .min()
        ),
        "rows_per_host_cell_radius_source_stratum": {
            "minimum": int(block_balance.min()),
            "maximum": int(block_balance.max()),
        },
        "source_reuse": {
            "minimum": int(source_reuse.min()),
            "maximum": int(source_reuse.max()),
        },
        "parent_scene_analysis_weight": 1.0,
        "parent_parameters_preserved_exactly": True,
        "selection_used_detector_outcomes": False,
        "pair_design_sha256": sha256_file(design_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_receipt_sha256": sha256_file(source_receipt_path),
        "source_receipt_status": source_receipt["status"],
        "parent_authority": authority,
        "config": config_provenance(config),
        "source_closure": source_closure(config),
    }
    atomic_json(output_dir / "PAIR_DESIGN_RECEIPT.json", receipt)
    return design, receipt
