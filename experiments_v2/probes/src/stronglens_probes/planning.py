"""Pre-feature row planning for both probes and their shared dependency graph."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .authorities import AuthorityContext
from .contracts import (
    ContractError,
    canonical_row_sha256,
    require_columns,
    require_nonempty_columns,
    sha256_file,
)
from .dependencies import (
    assign_dependency_folds,
    build_dependency_components,
    geographic_spatial_key,
)
from .gdomain import GDomainSubsetArtifact, load_gdomain_subset
from .matching import (
    FrozenMatchArtifact,
    FrozenPairPlanArtifact,
    RenderedPairArtifact,
    assert_frozen_match_artifact,
    load_rendered_pairs,
    validate_rendered_pairs,
)
from .parents import RAW_TRACTOR_COLUMNS, GRecArtifacts


@dataclass(frozen=True)
class FrozenPairSceneLinks:
    frame: pd.DataFrame
    path: Path
    receipt_path: Path
    manifest_sha256: str
    receipt: dict[str, Any]


def build_gdomain_analysis_rows(
    grec: GRecArtifacts,
    subset_artifact: GDomainSubsetArtifact,
    *,
    config: Mapping[str, Any],
    authority: AuthorityContext,
) -> pd.DataFrame:
    """Build rows only; no image is opened and no representation is extracted."""
    authority.assert_config(config)
    grec.revalidate(config=dict(config), authority=authority)
    subset = load_gdomain_subset(
        path=subset_artifact.path,
        receipt_path=subset_artifact.receipt_path,
        grec=grec,
        config=config,
        authority=authority,
    ).frame
    references = pd.read_parquet(authority.reference_manifest_path)
    reference_payloads = pd.read_parquet(authority.reference_payload_manifest_path)
    require_columns(
        references,
        {
            "galaxy_id",
            "healpix_128",
            "physical_group_id",
            "spatial_component_id",
        },
        context="G_domain references",
    )
    require_columns(
        reference_payloads,
        {"galaxy_id", "payload_sha256", "psf_id", "render_parameter_id"},
        context="G_domain reference payloads",
    )
    authority.assert_exact_references(
        references, id_column="galaxy_id", context="G_domain references"
    )
    authority.assert_reference_metadata(
        references,
        id_column="galaxy_id",
        physical_column="physical_group_id",
        spatial_column="spatial_component_id",
        exact=True,
        context="G_domain references",
    )
    authority.assert_reference_payloads(
        reference_payloads,
        id_column="galaxy_id",
        payload_column="payload_sha256",
        exact=True,
        context="G_domain reference payloads",
    )
    authority.assert_reference_render_metadata(
        reference_payloads,
        id_column="galaxy_id",
        psf_column="psf_id",
        render_column="render_parameter_id",
        exact=True,
        context="G_domain reference payloads",
    )
    payload_columns = [
        "galaxy_id",
        "payload_sha256",
        "psf_id",
        "render_parameter_id",
        *[
            column
            for column in ("pixel_sha256", "pixel_file_sha256", "pixel_relpath")
            if column in reference_payloads
        ],
    ]
    observed = references[
        ["galaxy_id", "healpix_128", "physical_group_id", "spatial_component_id"]
    ].merge(
        reference_payloads[payload_columns],
        on="galaxy_id",
        how="left",
        validate="one_to_one",
    )
    if (
        len(observed) != 43
        or observed["galaxy_id"].astype(str).nunique() != 43
        or observed["payload_sha256"].isna().any()
    ):
        raise ContractError("G_domain rows do not contain the exact 43 payloads")
    reference_pixel_sha256 = (
        observed["pixel_sha256"].astype(str)
        if "pixel_sha256" in observed
        else observed["payload_sha256"].astype(str)
    )
    reference_pixel_file_sha256 = (
        observed["pixel_file_sha256"].astype(str)
        if "pixel_file_sha256" in observed
        else observed["payload_sha256"].astype(str)
    )
    reference_pixel_relpath = (
        observed["pixel_relpath"].astype(str)
        if "pixel_relpath" in observed
        else pd.Series("unavailable:design_fixture", index=observed.index)
    )
    observed_rows = pd.DataFrame(
        {
            "row_id": "gdomain:observed:" + observed["galaxy_id"].astype(str),
            "population": "observed_reference",
            "reference_id": observed["galaxy_id"].astype(str),
            "host_galaxy_id": "not_applicable:observed",
            "host_parent_id": "not_applicable:observed",
            "physical_group_id": observed["physical_group_id"].astype(str),
            "geographic_spatial_key": observed["healpix_128"].map(
                geographic_spatial_key
            ),
            "spatial_group_id": observed["spatial_component_id"].astype(str),
            "source_parameter_id": "not_applicable:observed",
            "lens_parameter_id": "not_applicable:observed",
            "psf_id": observed["psf_id"].astype(str),
            "rendering_id": (
                "observed_survey_payload:" + observed["payload_sha256"].astype(str)
            ),
            "render_parameter_id": observed["render_parameter_id"].astype(str),
            "random_draw_id": "not_applicable:observed",
            "base_pixel_sha256": observed["payload_sha256"].astype(str),
            "pixel_sha256": reference_pixel_sha256,
            "pixel_file_sha256": reference_pixel_file_sha256,
            "pixel_relpath": reference_pixel_relpath,
            "pixel_array_key": "cutout",
            "pixel_layout": "HWC",
            "pair_id": "",
            "target_galaxy_id": "",
        }
    )
    simulated_pixel_sha256 = (
        subset["expectation_pixel_sha256"].astype(str)
        if "expectation_pixel_sha256" in subset
        else subset["final_image_sha256"].astype(str)
    )
    simulated_pixel_file_sha256 = (
        subset["parent_artifact_sha256"].astype(str)
        if "parent_artifact_sha256" in subset
        else subset["final_image_sha256"].astype(str)
    )
    simulated_pixel_relpath = (
        subset["parent_pixel_path"].astype(str)
        if "parent_pixel_path" in subset
        else pd.Series("unavailable:design_fixture", index=subset.index)
    )
    simulated_rows = pd.DataFrame(
        {
            "row_id": "gdomain:simulated:" + subset["grec_row_id"].astype(str),
            "population": "simulated",
            "reference_id": "not_applicable:simulated",
            "host_galaxy_id": subset["host_galaxy_id"].astype(str),
            "host_parent_id": subset["host_parent_id"].astype(str),
            "physical_group_id": subset["host_physical_group_id"].astype(str),
            "geographic_spatial_key": subset["host_galaxy_id"]
            .astype(str)
            .map(
                grec.hosts.set_index(
                    grec.hosts["galaxy_id"].astype(str)
                )["host_healpix_128"]
            )
            .map(geographic_spatial_key),
            "spatial_group_id": subset["host_spatial_group_id"].astype(str),
            "source_parameter_id": subset["source_parameter_id"].astype(str),
            "lens_parameter_id": subset["lens_parameter_id"].astype(str),
            "psf_id": subset["psf_id"].astype(str),
            "rendering_id": subset["rendering_id"].astype(str),
            "render_parameter_id": subset["render_parameter_id"].astype(str),
            "random_draw_id": subset["random_draw_id"].astype(str),
            "base_pixel_sha256": subset["host_pixel_sha256"].astype(str),
            "pixel_sha256": simulated_pixel_sha256,
            "pixel_file_sha256": simulated_pixel_file_sha256,
            "pixel_relpath": simulated_pixel_relpath,
            "pixel_array_key": "expectation_chw",
            "pixel_layout": "CHW",
            "pair_id": "",
            "target_galaxy_id": subset["host_galaxy_id"].astype(str),
        }
    )
    result = pd.concat([observed_rows, simulated_rows], ignore_index=True)
    result["authority_digest"] = authority.authority_digest
    result["data_role"] = (
        "design"
        if config["execution"].get("allow_reduced_grec_design_fixture")
        else "final"
    )
    require_nonempty_columns(
        result,
        config["probe"]["dependency_columns"],
        context="G_domain analysis rows",
    )
    return result


def build_pair_scene_links(
    frozen_matches: FrozenMatchArtifact,
    grec: GRecArtifacts,
    *,
    config: Mapping[str, Any],
    authority: AuthorityContext,
) -> pd.DataFrame:
    """Select one score-blind registered-radius G_rec scene per matched target."""
    authority.assert_config(config)
    assert_frozen_match_artifact(
        frozen_matches,
        config=config,
        authority=authority,
    )
    grec.revalidate(config=dict(config), authority=authority)
    selection = config["G_match"]["scene_selection"]
    radius = float(selection["theta_e_arcsec"])
    candidates = grec.scenes[
        pd.to_numeric(grec.scenes["theta_e_arcsec"]).eq(radius)
    ].sort_values(["host_galaxy_id", "grec_row_id"], kind="mergesort")
    if candidates["host_galaxy_id"].astype(str).duplicated().any():
        raise ContractError(
            "registered G_match radius has multiple G_rec scenes per host"
        )
    lookup = candidates.set_index(candidates["host_galaxy_id"].astype(str))
    rows = []
    for match in frozen_matches.frame.itertuples(index=False):
        target = str(match.target_galaxy_id)
        if target not in lookup.index:
            raise ContractError(f"G_match target {target} lacks registered G_rec scene")
        scene = lookup.loc[target]
        rows.append(
            {
                "pair_id": str(match.pair_id),
                "reference_id": str(match.reference_id),
                "target_galaxy_id": target,
                "grec_row_id": str(scene["grec_row_id"]),
                "theta_e_arcsec": radius,
                "selection_rule": str(selection["rule"]),
                "matches_manifest_sha256": frozen_matches.manifest_sha256,
                "G_rec_scene_manifest_sha256": grec.scene_sha256,
            }
        )
    return pd.DataFrame(rows).sort_values("pair_id", ignore_index=True)


def load_pair_scene_links(
    *,
    path: Path,
    receipt_path: Path,
    frozen_matches: FrozenMatchArtifact,
    grec: GRecArtifacts,
    config: Mapping[str, Any],
    authority: AuthorityContext,
) -> FrozenPairSceneLinks:
    """Reload and exactly recompute the deterministic pair-scene link rule."""
    manifest_sha256 = sha256_file(path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "complete_frozen_pair_scene_links"
        or receipt.get("manifest_sha256") != manifest_sha256
        or receipt.get("matches_manifest_sha256") != frozen_matches.manifest_sha256
        or receipt.get("G_rec_scene_manifest_sha256") != grec.scene_sha256
        or receipt.get("authority_digest") != authority.authority_digest
    ):
        raise ContractError("pair-scene link receipt has invalid parent authorities")
    observed = pd.read_parquet(path)
    expected = build_pair_scene_links(
        frozen_matches,
        grec,
        config=config,
        authority=authority,
    )
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_exact=True,
        )
    except AssertionError as error:
        raise ContractError(
            "pair-scene links differ from deterministic rule"
        ) from error
    return FrozenPairSceneLinks(
        frame=observed,
        path=path.resolve(),
        receipt_path=receipt_path.resolve(),
        manifest_sha256=manifest_sha256,
        receipt=receipt,
    )


def build_gmatch_pair_plan(
    frozen_matches: FrozenMatchArtifact,
    grec: GRecArtifacts,
    pair_scene_links: FrozenPairSceneLinks,
    *,
    config: Mapping[str, Any],
    authority: AuthorityContext,
) -> pd.DataFrame:
    """Create exactly two planned arms per match without rendering either arm."""
    authority.assert_config(config)
    assert_frozen_match_artifact(
        frozen_matches,
        config=config,
        authority=authority,
    )
    grec.revalidate(config=dict(config), authority=authority)
    verified_links = load_pair_scene_links(
        path=pair_scene_links.path,
        receipt_path=pair_scene_links.receipt_path,
        frozen_matches=frozen_matches,
        grec=grec,
        config=config,
        authority=authority,
    )
    pair_scene_link_manifest_sha256 = verified_links.manifest_sha256
    matches = frozen_matches.frame
    require_columns(
        matches,
        {"pair_id", "reference_id", "target_galaxy_id", "match_row_sha256"},
        context="frozen G_match matches",
    )
    required_scene = {"pair_id", "grec_row_id"}
    require_columns(
        verified_links.frame, required_scene, context="G_match pair-scene links"
    )
    if (
        verified_links.frame["pair_id"].astype(str).duplicated().any()
        or verified_links.frame["grec_row_id"].astype(str).duplicated().any()
    ):
        raise ContractError("G_match pair-scene links must be one-to-one")
    joined = matches.merge(
        verified_links.frame[list(required_scene)],
        on="pair_id",
        how="left",
        validate="one_to_one",
    )
    joined = joined.merge(
        grec.scenes,
        on="grec_row_id",
        how="left",
        validate="one_to_one",
    )
    if (
        not joined["target_galaxy_id"]
        .astype(str)
        .eq(joined["host_galaxy_id"].astype(str))
        .all()
    ):
        raise ContractError("G_match linked G_rec scene has the wrong target host")
    tractor_lookup = grec.tractor.frame.set_index(
        grec.tractor.frame["galaxy_id"].astype(str)
    )
    tractor_columns = [*RAW_TRACTOR_COLUMNS, "payload_sha256"]
    joined["tractor_row_sha256"] = [
        canonical_row_sha256(
            tractor_lookup.loc[str(host)].to_dict(),
            tractor_columns,
        )
        for host in joined["host_galaxy_id"]
    ]
    if (
        joined[
            [
                "host_pixel_sha256",
                "source_parameter_id",
                "lens_parameter_id",
                "psf_id",
                "rendering_id",
                "render_parameter_id",
                "random_draw_id",
                "intrinsic_flux_g_nmgy",
                "intrinsic_flux_r_nmgy",
                "intrinsic_flux_z_nmgy",
                "source_flux_parameter_sha256",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ContractError("G_match pair plan lacks a host or scene parameter")
    rows = []
    for row in joined.itertuples(index=False):
        for arm in ("bare", "injected"):
            rows.append(
                {
                    "pair_id": str(row.pair_id),
                    "reference_id": str(row.reference_id),
                    "target_galaxy_id": str(row.target_galaxy_id),
                    "arm": arm,
                    "grec_row_id": str(row.grec_row_id),
                    "scene_id": str(row.scene_id),
                    "base_pixel_sha256": str(row.host_pixel_sha256),
                    "host_parent_id": str(row.host_parent_id),
                    "host_physical_group_id": str(row.host_physical_group_id),
                    "host_spatial_group_id": str(row.host_spatial_group_id),
                    "source_parameter_id": str(row.source_parameter_id),
                    "lens_parameter_id": str(row.lens_parameter_id),
                    "psf_id": str(row.psf_id),
                    "rendering_id": str(row.rendering_id),
                    "render_parameter_id": str(row.render_parameter_id),
                    "random_draw_id": str(row.random_draw_id),
                    "source_flux_policy": "fixed_intrinsic",
                    "intrinsic_flux_g_nmgy": float(row.intrinsic_flux_g_nmgy),
                    "intrinsic_flux_r_nmgy": float(row.intrinsic_flux_r_nmgy),
                    "intrinsic_flux_z_nmgy": float(row.intrinsic_flux_z_nmgy),
                    "source_flux_parameter_sha256": str(
                        row.source_flux_parameter_sha256
                    ),
                    "pair_scene_link_manifest_sha256": (
                        pair_scene_link_manifest_sha256
                    ),
                    "grec_scene_manifest_sha256": grec.scene_sha256,
                    "grec_host_manifest_sha256": grec.host_sha256,
                    "tractor_manifest_sha256": grec.tractor.sha256,
                    "tractor_row_sha256": str(row.tractor_row_sha256),
                    "match_row_sha256": str(row.match_row_sha256),
                    "reference_manifest_sha256": (authority.reference_manifest_sha256),
                    "reference_payload_manifest_sha256": (
                        authority.reference_payload_manifest_sha256
                    ),
                    "host_universe_manifest_sha256": (
                        authority.experiment_host_universe_sha256
                    ),
                    "matches_manifest_sha256": frozen_matches.manifest_sha256,
                }
            )
    plan = pd.DataFrame(rows)
    hash_columns = list(plan.columns)
    plan["pair_plan_row_sha256"] = [
        canonical_row_sha256(row, hash_columns)
        for row in plan.to_dict(orient="records")
    ]
    if plan.groupby("pair_id")["arm"].nunique().ne(2).any() or plan[
        "target_galaxy_id"
    ].nunique() != len(matches):
        raise ContractError("G_match plan is not one pair per unique target")
    return plan.sort_values(["pair_id", "arm"], ignore_index=True)


def build_gmatch_analysis_rows(
    rendered_pair_artifact: RenderedPairArtifact,
    frozen_matches: FrozenMatchArtifact,
    frozen_pair_plan: FrozenPairPlanArtifact,
    grec: GRecArtifacts,
    *,
    config: Mapping[str, Any],
    authority: AuthorityContext,
) -> pd.DataFrame:
    """Build observed/bare/injected trios after rendering, before feature access."""
    authority.assert_config(config)
    grec.revalidate(config=dict(config), authority=authority)
    rendered_pairs = load_rendered_pairs(
        path=rendered_pair_artifact.path,
        receipt_path=rendered_pair_artifact.receipt_path,
        frozen_matches=frozen_matches,
        frozen_pair_plan=frozen_pair_plan,
        config=config,
        authority=authority,
    ).frame
    references = pd.read_parquet(authority.reference_manifest_path)
    reference_payloads = pd.read_parquet(authority.reference_payload_manifest_path)
    matches = frozen_matches.frame
    pair_manifest_sha256 = frozen_pair_plan.manifest_sha256
    validate_rendered_pairs(
        rendered_pairs,
        frozen_matches=frozen_matches,
        frozen_pair_plan=frozen_pair_plan,
        config=config,
        authority=authority,
    )
    reference = references[
        ["galaxy_id", "healpix_128", "physical_group_id", "spatial_component_id"]
    ].merge(
        reference_payloads[
            ["galaxy_id", "payload_sha256", "psf_id", "render_parameter_id"]
        ],
        on="galaxy_id",
        how="left",
        validate="one_to_one",
    )
    authority.assert_reference_payloads(
        reference_payloads,
        id_column="galaxy_id",
        payload_column="payload_sha256",
        exact=True,
        context="G_match reference payloads",
    )
    authority.assert_reference_render_metadata(
        reference_payloads,
        id_column="galaxy_id",
        psf_column="psf_id",
        render_column="render_parameter_id",
        exact=True,
        context="G_match reference payloads",
    )
    authority.assert_reference_metadata(
        references,
        id_column="galaxy_id",
        physical_column="physical_group_id",
        spatial_column="spatial_component_id",
        exact=True,
        context="G_match references",
    )
    reference = reference.set_index(reference["galaxy_id"].astype(str))
    hosts = grec.hosts.set_index(grec.hosts["galaxy_id"].astype(str))
    rows: list[dict[str, Any]] = []
    for match in matches.itertuples(index=False):
        reference_id = str(match.reference_id)
        target_id = str(match.target_galaxy_id)
        if reference_id not in reference.index or target_id not in hosts.index:
            raise ContractError("match lacks reference or target dependency metadata")
        ref = reference.loc[reference_id]
        pair = rendered_pairs[
            rendered_pairs["pair_id"].astype(str).eq(str(match.pair_id))
        ].set_index("arm")
        common = {
            "pair_id": str(match.pair_id),
            "reference_id": reference_id,
            "target_galaxy_id": target_id,
        }
        rows.append(
            {
                **common,
                "row_id": f"gmatch:{match.pair_id}:observed",
                "population": "observed_reference",
                "host_galaxy_id": "not_applicable:observed",
                "host_parent_id": "not_applicable:observed",
                "physical_group_id": str(ref["physical_group_id"]),
                "geographic_spatial_key": geographic_spatial_key(
                    ref["healpix_128"]
                ),
                "spatial_group_id": str(ref["spatial_component_id"]),
                "source_parameter_id": "not_applicable:observed",
                "lens_parameter_id": "not_applicable:observed",
                "psf_id": str(ref["psf_id"]),
                "rendering_id": f"observed_survey_payload:{ref['payload_sha256']}",
                "render_parameter_id": str(ref["render_parameter_id"]),
                "random_draw_id": "not_applicable:observed",
                "base_pixel_sha256": str(ref["payload_sha256"]),
                "pixel_sha256": str(ref["payload_sha256"]),
            }
        )
        for arm in ("bare", "injected"):
            rendered = pair.loc[arm]
            rows.append(
                {
                    **common,
                    "row_id": f"gmatch:{match.pair_id}:{arm}",
                    "population": arm,
                    "host_galaxy_id": target_id,
                    "host_parent_id": str(rendered["host_parent_id"]),
                    "physical_group_id": str(rendered["host_physical_group_id"]),
                    "geographic_spatial_key": geographic_spatial_key(
                        hosts.loc[target_id, "host_healpix_128"]
                    ),
                    "spatial_group_id": str(rendered["host_spatial_group_id"]),
                    "source_parameter_id": str(rendered["source_parameter_id"]),
                    "lens_parameter_id": str(rendered["lens_parameter_id"]),
                    "psf_id": str(rendered["psf_id"]),
                    "rendering_id": str(rendered["rendering_id"]),
                    "render_parameter_id": str(rendered["render_parameter_id"]),
                    "random_draw_id": str(rendered["random_draw_id"]),
                    "base_pixel_sha256": str(rendered["base_pixel_sha256"]),
                    "pixel_sha256": str(rendered["final_image_sha256"]),
                }
            )
    result = pd.DataFrame(rows)
    result["authority_digest"] = authority.authority_digest
    result["matches_manifest_sha256"] = frozen_matches.manifest_sha256
    result["pair_manifest_sha256"] = pair_manifest_sha256
    require_nonempty_columns(
        result,
        config["probe"]["dependency_columns"],
        context="G_match analysis rows",
    )
    return result


def finalize_joint_dependencies_and_folds(
    domain_rows: pd.DataFrame,
    gmatch_rows: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    authority: AuthorityContext,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Union shared ancestry once, then make experiment-specific fold maps."""
    authority.assert_config(config)
    dependency_columns = list(config["probe"]["dependency_columns"])
    require_nonempty_columns(
        domain_rows, dependency_columns, context="G_domain dependencies"
    )
    require_nonempty_columns(
        gmatch_rows, dependency_columns, context="G_match dependencies"
    )
    if not domain_rows["authority_digest"].eq(authority.authority_digest).all():
        raise ContractError("G_domain rows have the wrong authority")
    if not gmatch_rows["authority_digest"].eq(authority.authority_digest).all():
        raise ContractError("G_match rows have the wrong authority")
    domain = domain_rows.copy()
    match = gmatch_rows.copy()
    domain["_experiment"] = "G_domain"
    match["_experiment"] = "G_match"
    domain["_global_row_id"] = domain["_experiment"] + ":" + domain["row_id"]
    match["_global_row_id"] = match["_experiment"] + ":" + match["row_id"]
    combined = pd.concat([domain, match], ignore_index=True)
    combined = build_dependency_components(
        combined,
        row_id_column="_global_row_id",
        dependency_columns=dependency_columns,
        output_column="joint_dependency_id",
    )
    domain = combined[combined["_experiment"].eq("G_domain")].copy()
    match = combined[combined["_experiment"].eq("G_match")].copy()
    for frame in (domain, match):
        frame["dependency_component_id"] = frame["joint_dependency_id"]
        frame.drop(columns=["_experiment", "_global_row_id"], inplace=True)
    domain["fold_label"] = domain["population"].eq("observed_reference").astype(int)
    domain_folds = assign_dependency_folds(
        domain,
        component_column="dependency_component_id",
        label_column="fold_label",
        n_splits=int(config["probe"]["folds"]),
        seed=int(config["probe"]["fold_seed"]),
    )
    match_fit = match[match["population"].isin(["observed_reference", "bare"])].copy()
    match_fit["fold_label"] = (
        match_fit["population"].eq("observed_reference").astype(int)
    )
    match_folds = assign_dependency_folds(
        match_fit,
        component_column="dependency_component_id",
        label_column="fold_label",
        n_splits=int(config["probe"]["folds"]),
        seed=int(config["probe"]["fold_seed"]),
    )
    return (
        domain.drop(columns="fold_label").reset_index(drop=True),
        domain_folds,
        match.reset_index(drop=True),
        match_folds,
    )
