from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from conftest import AuthorityBundle, digest

from stronglens_probes.contracts import ContractError, sha256_file
from stronglens_probes.dependencies import (
    build_dependency_components,
    geographic_spatial_key,
)
from stronglens_probes.gdomain import build_gdomain_subset, load_gdomain_subset
from stronglens_probes.matching import (
    load_frozen_matches,
    load_frozen_pair_plan,
    load_rendered_pairs,
    match_targets,
)
from stronglens_probes.planning import (
    build_gdomain_analysis_rows,
    build_gmatch_analysis_rows,
    build_gmatch_pair_plan,
    build_pair_scene_links,
    finalize_joint_dependencies_and_folds,
    load_pair_scene_links,
)


def test_common_geography_connects_distinct_complete_triplets() -> None:
    rows = []
    for pair, reference, geography in (
        ("pair-a", "reference-a", "healpix128:10"),
        ("pair-b", "reference-b", "healpix128:20"),
    ):
        for population in ("observed_reference", "bare", "injected"):
            rows.append(
                {
                    "row_id": f"{pair}:{population}",
                    "pair_id": pair,
                    "reference_id": reference,
                    "population": population,
                    "geographic_spatial_key": (
                        "healpix128:20"
                        if pair == "pair-a" and population == "observed_reference"
                        else geography
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    result = build_dependency_components(
        frame,
        row_id_column="row_id",
        dependency_columns=["reference_id", "geographic_spatial_key"],
    )
    assert result["dependency_component_id"].nunique() == 1
    assert result.groupby("pair_id")["dependency_component_id"].nunique().eq(1).all()
    assert geographic_spatial_key(20) == "healpix128:20"


def _frozen(bundle: AuthorityBundle, tmp_path: Path):
    result = match_targets(
        bundle.reference_tractor,
        bundle.host_tractor,
        bundle.grec,
        config=bundle.config,
        authority=bundle.context,
    )
    match_path = tmp_path / "matches.parquet"
    match_receipt_path = tmp_path / "matches.json"
    result.matches.to_parquet(match_path, index=False)
    match_receipt = dict(result.receipt)
    match_receipt["matches_manifest_sha256"] = sha256_file(match_path)
    match_receipt_path.write_text(json.dumps(match_receipt))
    frozen_match = load_frozen_matches(
        matches_path=match_path,
        receipt_path=match_receipt_path,
        reference_tractor=bundle.reference_tractor,
        host_tractor=bundle.host_tractor,
        grec=bundle.grec,
        config=bundle.config,
        authority=bundle.context,
    )
    links = build_pair_scene_links(
        frozen_match,
        bundle.grec,
        config=bundle.config,
        authority=bundle.context,
    )
    links_path = tmp_path / "links.parquet"
    links.to_parquet(links_path, index=False)
    links_receipt_path = tmp_path / "links.json"
    links_receipt_path.write_text(
        json.dumps(
            {
                "status": "complete_frozen_pair_scene_links",
                "manifest_sha256": sha256_file(links_path),
                "matches_manifest_sha256": frozen_match.manifest_sha256,
                "G_rec_scene_manifest_sha256": bundle.grec.scene_sha256,
                "authority_digest": bundle.context.authority_digest,
            }
        )
    )
    frozen_links = load_pair_scene_links(
        path=links_path,
        receipt_path=links_receipt_path,
        frozen_matches=frozen_match,
        grec=bundle.grec,
        config=bundle.config,
        authority=bundle.context,
    )
    plan = build_gmatch_pair_plan(
        frozen_match,
        bundle.grec,
        frozen_links,
        config=bundle.config,
        authority=bundle.context,
    )
    plan_path = tmp_path / "pair_plan.parquet"
    plan_receipt_path = tmp_path / "pair_plan.json"
    plan.to_parquet(plan_path, index=False)
    plan_receipt_path.write_text(
        json.dumps(
            {
                "status": "complete_frozen_G_match_pair_plan",
                "pair_manifest_sha256": sha256_file(plan_path),
                "matches_manifest_sha256": frozen_match.manifest_sha256,
                "authority_digest": bundle.context.authority_digest,
                "G_rec_scene_manifest_sha256": bundle.grec.scene_sha256,
                "G_rec_host_manifest_sha256": bundle.grec.host_sha256,
                "G_rec_tractor_manifest_sha256": bundle.grec.tractor.sha256,
            }
        )
    )
    frozen_plan = load_frozen_pair_plan(
        pair_plan_path=plan_path,
        receipt_path=plan_receipt_path,
        frozen_matches=frozen_match,
        grec=bundle.grec,
        config=bundle.config,
        authority=bundle.context,
    )
    return frozen_match, frozen_plan


def test_planning_uses_bound_parents_and_connected_groups(
    authority_bundle: AuthorityBundle,
    tmp_path: Path,
) -> None:
    assert len(authority_bundle.grec.hosts) == 80
    assert len(authority_bundle.grec.scenes) == 80 * 11
    assert (
        authority_bundle.grec.scenes.groupby("host_galaxy_id")["theta_e_arcsec"]
        .nunique()
        .eq(11)
        .all()
    )
    subset_frame, _ = build_gdomain_subset(
        authority_bundle.grec,
        config=authority_bundle.config,
        authority=authority_bundle.context,
    )
    subset_path = tmp_path / "subset.parquet"
    subset_receipt_path = tmp_path / "subset.json"
    subset_frame.to_parquet(subset_path, index=False)
    subset_receipt_path.write_text(
        json.dumps(
            {
                "status": "pre_score_exact_G_domain_subset_complete",
                "subset_sha256": sha256_file(subset_path),
                "G_rec_parent_sha256": authority_bundle.grec.scene_sha256,
                "G_rec_host_sha256": authority_bundle.grec.host_sha256,
                "G_rec_tractor_sha256": authority_bundle.grec.tractor.sha256,
                "authority_digest": authority_bundle.context.authority_digest,
            }
        )
    )
    subset = load_gdomain_subset(
        path=subset_path,
        receipt_path=subset_receipt_path,
        grec=authority_bundle.grec,
        config=authority_bundle.config,
        authority=authority_bundle.context,
    )
    domain_rows = build_gdomain_analysis_rows(
        authority_bundle.grec,
        subset,
        config=authority_bundle.config,
        authority=authority_bundle.context,
    )
    frozen_match, frozen_plan = _frozen(authority_bundle, tmp_path)
    rendered = frozen_plan.frame.copy()
    rendered["pair_manifest_sha256"] = frozen_plan.manifest_sha256
    rendered["final_image_sha256"] = [
        base if arm == "bare" else digest(f"injected:{pair}")
        for base, arm, pair in rendered[
            ["base_pixel_sha256", "arm", "pair_id"]
        ].itertuples(index=False)
    ]
    rendered_path = tmp_path / "rendered.parquet"
    rendered_receipt_path = tmp_path / "rendered.json"
    rendered.to_parquet(rendered_path, index=False)
    rendered_receipt_path.write_text(
        json.dumps(
            {
                "status": "complete_G_match_rendered_pair_validation",
                "rendered_pairs_sha256": sha256_file(rendered_path),
                "pair_manifest_sha256": frozen_plan.manifest_sha256,
                "matches_manifest_sha256": frozen_match.manifest_sha256,
                "authority_digest": authority_bundle.context.authority_digest,
            }
        )
    )
    rendered_artifact = load_rendered_pairs(
        path=rendered_path,
        receipt_path=rendered_receipt_path,
        frozen_matches=frozen_match,
        frozen_pair_plan=frozen_plan,
        config=authority_bundle.config,
        authority=authority_bundle.context,
    )
    match_rows = build_gmatch_analysis_rows(
        rendered_artifact,
        frozen_match,
        frozen_plan,
        authority_bundle.grec,
        config=authority_bundle.config,
        authority=authority_bundle.context,
    )
    domain, domain_folds, match, match_folds = finalize_joint_dependencies_and_folds(
        domain_rows,
        match_rows,
        config=authority_bundle.config,
        authority=authority_bundle.context,
    )
    domain_reference = domain[domain["population"].eq("observed_reference")].set_index(
        "reference_id"
    )["joint_dependency_id"]
    match_reference = match[match["population"].eq("observed_reference")].set_index(
        "reference_id"
    )["joint_dependency_id"]
    assert domain_reference.to_dict() == match_reference.to_dict()
    assert match.groupby("pair_id")["dependency_component_id"].nunique().max() == 1
    assert set(domain_folds["fold"]) == set(range(5))
    assert set(match_folds["fold"]) == set(range(5))


def test_missing_dependency_and_reference_render_authority_fail(
    authority_bundle: AuthorityBundle,
) -> None:
    subset_frame, _ = build_gdomain_subset(
        authority_bundle.grec,
        config=authority_bundle.config,
        authority=authority_bundle.context,
    )
    assert not subset_frame.empty
    original = authority_bundle.context.reference_payload_manifest_path.read_bytes()
    wrong_payloads = pd.read_parquet(
        authority_bundle.context.reference_payload_manifest_path
    )
    wrong_payloads.loc[0, "psf_id"] = "wrong"
    wrong_payloads.to_parquet(
        authority_bundle.context.reference_payload_manifest_path,
        index=False,
    )
    try:
        with pytest.raises(ContractError, match="authority file changed"):
            build_gdomain_analysis_rows(
                authority_bundle.grec,
                object(),  # authority rehash fails before artifact use
                config=authority_bundle.config,
                authority=authority_bundle.context,
            )
    finally:
        authority_bundle.context.reference_payload_manifest_path.write_bytes(original)
