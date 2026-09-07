from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
from conftest import AuthorityBundle, digest

from stronglens_probes.contracts import ContractError, sha256_file
from stronglens_probes.matching import (
    load_frozen_matches,
    load_frozen_pair_plan,
    match_targets,
    validate_rendered_pairs,
)
from stronglens_probes.planning import (
    build_gmatch_pair_plan,
    build_pair_scene_links,
    load_pair_scene_links,
)


def _frozen_matches(bundle: AuthorityBundle, tmp_path: Path):
    result = match_targets(
        bundle.reference_tractor,
        bundle.host_tractor,
        bundle.grec,
        config=bundle.config,
        authority=bundle.context,
    )
    path = tmp_path / "matches.parquet"
    receipt_path = tmp_path / "matches.json"
    result.matches.to_parquet(path, index=False)
    receipt = dict(result.receipt)
    receipt["matches_manifest_sha256"] = sha256_file(path)
    receipt_path.write_text(json.dumps(receipt))
    return load_frozen_matches(
        matches_path=path,
        receipt_path=receipt_path,
        reference_tractor=bundle.reference_tractor,
        host_tractor=bundle.host_tractor,
        grec=bundle.grec,
        config=bundle.config,
        authority=bundle.context,
    )


def _frozen_pair_plan(
    bundle: AuthorityBundle,
    frozen_matches,
    tmp_path: Path,
):
    links = build_pair_scene_links(
        frozen_matches,
        bundle.grec,
        config=bundle.config,
        authority=bundle.context,
    )
    link_path = tmp_path / "pair_links.parquet"
    links.to_parquet(link_path, index=False)
    link_receipt_path = tmp_path / "pair_links.json"
    link_receipt_path.write_text(
        json.dumps(
            {
                "status": "complete_frozen_pair_scene_links",
                "manifest_sha256": sha256_file(link_path),
                "matches_manifest_sha256": frozen_matches.manifest_sha256,
                "G_rec_scene_manifest_sha256": bundle.grec.scene_sha256,
                "authority_digest": bundle.context.authority_digest,
            }
        )
    )
    frozen_links = load_pair_scene_links(
        path=link_path,
        receipt_path=link_receipt_path,
        frozen_matches=frozen_matches,
        grec=bundle.grec,
        config=bundle.config,
        authority=bundle.context,
    )
    plan = build_gmatch_pair_plan(
        frozen_matches,
        bundle.grec,
        frozen_links,
        config=bundle.config,
        authority=bundle.context,
    )
    path = tmp_path / "pair_plan.parquet"
    receipt_path = tmp_path / "pair_plan.json"
    plan.to_parquet(path, index=False)
    receipt = {
        "status": "complete_frozen_G_match_pair_plan",
        "pair_manifest_sha256": sha256_file(path),
        "matches_manifest_sha256": frozen_matches.manifest_sha256,
        "authority_digest": bundle.context.authority_digest,
        "G_rec_scene_manifest_sha256": bundle.grec.scene_sha256,
        "G_rec_host_manifest_sha256": bundle.grec.host_sha256,
        "G_rec_tractor_manifest_sha256": bundle.grec.tractor.sha256,
    }
    receipt_path.write_text(json.dumps(receipt))
    return load_frozen_pair_plan(
        pair_plan_path=path,
        receipt_path=receipt_path,
        frozen_matches=frozen_matches,
        grec=bundle.grec,
        config=bundle.config,
        authority=bundle.context,
    )


def _rendered(frozen_plan) -> pd.DataFrame:
    frame = frozen_plan.frame.copy()
    frame["pair_manifest_sha256"] = frozen_plan.manifest_sha256
    frame["final_image_sha256"] = [
        base if arm == "bare" else digest(f"injected:{pair}")
        for base, arm, pair in frame[
            ["base_pixel_sha256", "arm", "pair_id"]
        ].itertuples(index=False)
    ]
    return frame


def test_raw_tractor_matching_is_deterministic_and_receipted(
    authority_bundle: AuthorityBundle,
) -> None:
    first = match_targets(
        authority_bundle.reference_tractor,
        authority_bundle.host_tractor,
        authority_bundle.grec,
        config=authority_bundle.config,
        authority=authority_bundle.context,
    )
    second = match_targets(
        authority_bundle.reference_tractor,
        authority_bundle.host_tractor,
        authority_bundle.grec,
        config=authority_bundle.config,
        authority=authority_bundle.context,
    )
    pd.testing.assert_frame_equal(first.matches, second.matches)
    assert len(first.matches) == 43
    assert first.receipt["reference_tractor_manifest_sha256"] == (
        authority_bundle.reference_tractor.sha256
    )
    assert first.receipt["host_tractor_manifest_sha256"] == (
        authority_bundle.host_tractor.sha256
    )
    assert first.receipt["tie_rule"] == "sha256_sorted_identifiers"
    assert first.receipt["candidate_population"] == "exact_G_rec_hosts"
    assert first.receipt["scaling_population"] == "exact_G_rec_hosts_only"
    assert first.receipt["G_rec_host_manifest_sha256"] == (
        authority_bundle.grec.host_sha256
    )


def test_matching_rejects_config_and_raw_tractor_mutation(
    authority_bundle: AuthorityBundle,
) -> None:
    changed = copy.deepcopy(authority_bundle.config)
    changed["G_match"]["matching"]["tie_rule"] = "input_order"
    with pytest.raises(ContractError, match="config differs|contract"):
        match_targets(
            authority_bundle.reference_tractor,
            authority_bundle.host_tractor,
            authority_bundle.grec,
            config=changed,
            authority=authority_bundle.context,
        )

    original = authority_bundle.host_tractor.path.read_bytes()
    tampered = pd.read_parquet(authority_bundle.host_tractor.path)
    tampered.loc[0, "flux_z"] *= 2
    tampered.to_parquet(authority_bundle.host_tractor.path, index=False)
    try:
        with pytest.raises(ContractError, match="changed after loading"):
            match_targets(
                authority_bundle.reference_tractor,
                authority_bundle.host_tractor,
                authority_bundle.grec,
                config=authority_bundle.config,
                authority=authority_bundle.context,
            )
    finally:
        authority_bundle.host_tractor.path.write_bytes(original)


def test_persisted_match_rows_are_exactly_recomputed(
    authority_bundle: AuthorityBundle,
    tmp_path: Path,
) -> None:
    result = match_targets(
        authority_bundle.reference_tractor,
        authority_bundle.host_tractor,
        authority_bundle.grec,
        config=authority_bundle.config,
        authority=authority_bundle.context,
    )
    path = tmp_path / "matches.parquet"
    receipt_path = tmp_path / "matches.json"
    changed = result.matches.copy()
    changed.loc[0, "distance"] += 0.01
    changed.to_parquet(path, index=False)
    receipt = dict(result.receipt)
    receipt["matches_manifest_sha256"] = sha256_file(path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ContractError, match="exact recomputation"):
        load_frozen_matches(
            matches_path=path,
            receipt_path=receipt_path,
            reference_tractor=authority_bundle.reference_tractor,
            host_tractor=authority_bundle.host_tractor,
            grec=authority_bundle.grec,
            config=authority_bundle.config,
            authority=authority_bundle.context,
        )


def test_pair_plan_binds_all_grec_and_tractor_parents(
    authority_bundle: AuthorityBundle,
    tmp_path: Path,
) -> None:
    frozen = _frozen_matches(authority_bundle, tmp_path)
    pair_plan = _frozen_pair_plan(authority_bundle, frozen, tmp_path)
    rendered = _rendered(pair_plan)
    validate_rendered_pairs(
        rendered,
        frozen_matches=frozen,
        frozen_pair_plan=pair_plan,
        config=authority_bundle.config,
        authority=authority_bundle.context,
    )
    for field in (
        "grec_row_id",
        "scene_id",
        "host_parent_id",
        "host_physical_group_id",
        "host_spatial_group_id",
        "source_parameter_id",
        "lens_parameter_id",
        "psf_id",
        "rendering_id",
        "render_parameter_id",
        "random_draw_id",
        "intrinsic_flux_g_nmgy",
        "intrinsic_flux_r_nmgy",
        "intrinsic_flux_z_nmgy",
        "tractor_row_sha256",
    ):
        assert rendered.groupby("pair_id")[field].nunique().eq(1).all()

    tampered = rendered.copy()
    pair = tampered.iloc[0]["pair_id"]
    tampered.loc[tampered["pair_id"].eq(pair), "lens_parameter_id"] = "wrong"
    with pytest.raises(ContractError, match="pair plan"):
        validate_rendered_pairs(
            tampered,
            frozen_matches=frozen,
            frozen_pair_plan=pair_plan,
            config=authority_bundle.config,
            authority=authority_bundle.context,
        )

    floor = rendered.copy()
    floor["host_total_flux_floor"] = 1.05
    with pytest.raises(ContractError, match="floor"):
        validate_rendered_pairs(
            floor,
            frozen_matches=frozen,
            frozen_pair_plan=pair_plan,
            config=authority_bundle.config,
            authority=authority_bundle.context,
        )


def test_pair_plan_postload_parent_mutation_is_rejected(
    authority_bundle: AuthorityBundle,
    tmp_path: Path,
) -> None:
    frozen = _frozen_matches(authority_bundle, tmp_path)
    pair_plan = _frozen_pair_plan(authority_bundle, frozen, tmp_path)
    pair_plan.frame.loc[0, "psf_id"] = "mutated"
    with pytest.raises(ContractError, match="row hashes|scene field|in-memory"):
        validate_rendered_pairs(
            _rendered(pair_plan),
            frozen_matches=frozen,
            frozen_pair_plan=pair_plan,
            config=authority_bundle.config,
            authority=authority_bundle.context,
        )
