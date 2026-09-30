from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gmatch_v2.authority import derive_covariates, load_authorities
from gmatch_v2.contracts import (
    ContractError,
    array_sha256,
    load_config,
    sha256_file,
)
from gmatch_v2.matching import match, solve_assignment
from gmatch_v2.preparation import (
    _analysis_manifest,
    _build_pair_plan,
    _build_pair_scene_links,
    validate_zero_copy_pairs,
)
from gmatch_v2.schemas import FORBIDDEN_ALIASES, assert_context_schema

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
CONFIG_PATH = PACKAGE / "config/production.toml"


@pytest.fixture(scope="session")
def config() -> dict:
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="session")
def bundle(config: dict):
    try:
        return load_authorities(config, root=ROOT, verify_all_host_pixels=False)
    except (ContractError, FileNotFoundError):
        pytest.skip("external G_match pixel and execution authorities are unavailable")


@pytest.fixture(scope="session")
def matched(bundle, config: dict):
    return match(bundle, config)


@pytest.fixture(scope="session")
def planned(bundle, matched):
    links = _build_pair_scene_links(bundle, matched, matches_sha256="a" * 64)
    plan = _build_pair_plan(
        bundle,
        matched,
        links,
        matches_sha256="a" * 64,
        links_sha256="b" * 64,
    )
    return links, plan


def test_config_is_adopted_and_stops_before_features(config: dict) -> None:
    assert config["study"]["status"] == "adopted_execution_authority_pre_feature"
    assert config["study"]["data_role"] == "final"
    assert config["matching"]["host_total_flux_floor"] == "forbidden"
    assert (
        config["matching"]["scaler_value_policy"]
        == "component_wise_all_finite_values_exact_authorized_roster"
    )
    assert config["matching"]["candidate_eligibility"] == "all_five_covariates_finite"
    assert config["matching"]["imputation"] is False
    assert config["schema_policy"]["unknown_columns"] == "reject"
    assert config["execution"]["feature_extraction_forbidden"] is True
    assert config["execution"]["probe_fitting_forbidden"] is True
    assert config["execution"]["bootstrap_forbidden"] is True


def test_maximum_cardinality_precedes_distance() -> None:
    distances = np.asarray([[0.0, 1.0], [0.1, 0.0]])
    admissible = np.asarray([[True, True], [True, False]])
    result = solve_assignment(
        distances,
        admissible,
        pd.Series(["r0", "r1"]),
        pd.Series(["h0", "h1"]),
        total_distance_caliper=3.0,
    )
    assert set(result) == {(0, 1), (1, 0)}


def test_tie_rule_is_invariant_to_input_permutation() -> None:
    distances = np.ones((3, 3))
    admissible = np.ones((3, 3), dtype=bool)
    references = pd.Series(["r-a", "r-b", "r-c"])
    hosts = pd.Series(["h-a", "h-b", "h-c"])
    first = solve_assignment(
        distances,
        admissible,
        references,
        hosts,
        total_distance_caliper=3.0,
    )
    first_pairs = {
        (references.iloc[r], hosts.iloc[h]) for r, h in first if h is not None
    }
    reference_order = [2, 0, 1]
    host_order = [1, 2, 0]
    second = solve_assignment(
        distances[np.ix_(reference_order, host_order)],
        admissible[np.ix_(reference_order, host_order)],
        references.iloc[reference_order].reset_index(drop=True),
        hosts.iloc[host_order].reset_index(drop=True),
        total_distance_caliper=3.0,
    )
    second_pairs = {
        (
            references.iloc[reference_order].reset_index(drop=True).iloc[r],
            hosts.iloc[host_order].reset_index(drop=True).iloc[h],
        )
        for r, h in second
        if h is not None
    }
    assert first_pairs == second_pairs


def test_covariates_are_raw_registered_transformations() -> None:
    source = pd.DataFrame(
        {
            "flux_z": [10.0, -1.0],
            "sersic": [4.0, 1.0],
            "shape_e1": [0.0, 2.0],
            "shape_e2": [0.0, 0.0],
            "psfsize_r": [1.2, 1.3],
            "psfdepth_r": [100.0, -1.0],
        }
    )
    result = derive_covariates(source)
    assert result.loc[0, "observed_zmag"] == pytest.approx(20.0)
    assert result.loc[0, "axis_ratio_q"] == 1.0
    assert np.isnan(result.loc[1, "observed_zmag"])
    assert np.isnan(result.loc[1, "axis_ratio_q"])
    assert np.isnan(result.loc[1, "depth5_r_mag"])


def test_benign_exact_schema_is_accepted(bundle) -> None:
    assert_context_schema(bundle.host.copy(), context="intermediate.host_adapter")


@pytest.mark.parametrize("alias", sorted(FORBIDDEN_ALIASES))
def test_every_forbidden_alias_is_rejected_by_default(bundle, alias: str) -> None:
    tampered = bundle.host.copy()
    tampered[alias] = 0.0
    with pytest.raises(ContractError, match="strict schema allowlist"):
        assert_context_schema(tampered, context="intermediate.host_adapter")


def test_file_and_array_hash_domains_are_distinct(tmp_path: Path) -> None:
    pixels = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    path = tmp_path / "pixels.npz"
    np.savez(path, cutout=pixels)
    assert sha256_file(path) != array_sha256(pixels)
    assert (
        array_sha256(pixels)
        == hashlib.sha256(
            memoryview(np.ascontiguousarray(pixels)).cast("B")
        ).hexdigest()
    )


def test_direct_authority_cardinality_and_roles(bundle) -> None:
    assert len(bundle.reference) == 43
    assert len(bundle.host) == 4500
    assert len(bundle.host_tractor) == 4500
    assert len(bundle.parent_theta15) == 4500
    assert bundle.host["proposed_role"].eq("experiment_host").all()
    assert bundle.parent_theta15["theta_e_arcsec"].eq(1.5).all()
    assert bundle.host["host_payload_sha256"].ne(bundle.host["host_pixel_sha256"]).all()
    assert bundle.detector_exclusion["passed"] is True
    assert bundle.detector_exclusion["references_present_in_test"] == 43


def test_real_matching_is_unique_calipered_and_deterministic(
    bundle, config: dict, matched
) -> None:
    replay = match(bundle, config)
    pd.testing.assert_frame_equal(matched.matches, replay.matches, check_exact=True)
    assert matched.matches["reference_id"].is_unique
    assert matched.matches["target_galaxy_id"].is_unique
    assert matched.matches["distance"].le(3.0).all()
    for covariate in config["matching"]["covariates"]:
        assert matched.matches[f"delta_standardised__{covariate}"].abs().le(2.0).all()
    assert len(matched.reference_disposition) == 43
    assert len(matched.host_disposition) == 4500
    assert matched.host_disposition["scaler_authority_member"].all()
    assert matched.host_disposition["matching_candidate_eligible"].sum() == 4265
    ineligible = matched.host_disposition.loc[
        ~matched.host_disposition["matching_candidate_eligible"]
    ]
    assert len(ineligible) == 235
    assert ineligible["missing_covariates"].eq("observed_zmag").all()
    assert matched.receipt["scaler_finite_count"] == {
        "observed_zmag": 4265,
        "sersic_n": 4500,
        "axis_ratio_q": 4500,
        "psf_fwhm_r_arcsec": 4500,
        "depth5_r_mag": 4500,
    }
    changed = matched.scaler_comparison.loc[
        matched.scaler_comparison["assignment_changed"]
    ]
    assert changed["reference_id"].tolist() == ["1601p187_3957"]
    assert changed["prior_target_galaxy_id"].tolist() == ["2013p022_3066"]
    assert changed["governing_target_galaxy_id"].tolist() == ["0659m037_4323"]
    assert matched.receipt["host_total_flux_floor_applied"] is False


def test_pair_plan_preserves_all_parent_identities(bundle, matched, planned) -> None:
    links, plan = planned
    assert len(links) == len(matched.matches)
    assert len(plan) == 2 * len(matched.matches)
    assert set(plan["arm"]) == {"bare", "injected"}
    shared = [
        "host_parent_id",
        "host_physical_group_id",
        "host_spatial_group_id",
        "source_parameter_id",
        "native_source_parameter_id",
        "lens_parameter_id",
        "native_lens_parameter_id",
        "psf_id",
        "rendering_id",
        "render_parameter_id",
        "random_draw_id",
        "source_flux_parameter_sha256",
    ]
    for column in shared:
        assert plan.groupby("pair_id")[column].nunique(dropna=False).eq(1).all()
    assert plan.loc[plan["arm"].eq("bare"), "pixel_layout"].eq("HWC").all()
    assert plan.loc[plan["arm"].eq("injected"), "pixel_layout"].eq("CHW").all()


def test_zero_copy_validation_and_adversarial_identity_checks(
    bundle, matched, planned
) -> None:
    _, plan = planned
    pairs = plan.copy()
    pairs["pair_plan_manifest_sha256"] = "c" * 64
    pairs["rendered_pair_row_sha256"] = "d" * 64
    result = validate_zero_copy_pairs(
        pairs,
        bundle=bundle,
        matches=matched.matches,
        pair_plan_sha256="c" * 64,
    )
    assert result["bare_payload_files_reopened"] == len(matched.matches)
    assert result["injected_parent_files_reopened"] == len(matched.matches)
    assert result["flux_rescaling_applied"] is False
    changed_identity = pairs.copy()
    first_pair = changed_identity["pair_id"].iloc[0]
    row = changed_identity.index[
        changed_identity["pair_id"].eq(first_pair)
        & changed_identity["arm"].eq("injected")
    ][0]
    changed_identity.loc[row, "source_parameter_id"] = "tampered"
    with pytest.raises(ContractError, match="shared identity"):
        validate_zero_copy_pairs(
            changed_identity,
            bundle=bundle,
            matches=matched.matches,
            pair_plan_sha256="c" * 64,
        )
    changed_both_arms = pairs.copy()
    changed_both_arms.loc[
        changed_both_arms["pair_id"].eq(first_pair), "source_parameter_id"
    ] = "tampered"
    with pytest.raises(ContractError, match="parent source_parameter_id"):
        validate_zero_copy_pairs(
            changed_both_arms,
            bundle=bundle,
            matches=matched.matches,
            pair_plan_sha256="c" * 64,
        )
    changed_flux = pairs.copy()
    changed_flux.loc[row, "intrinsic_flux_r_nmgy"] *= 2.0
    with pytest.raises(ContractError):
        validate_zero_copy_pairs(
            changed_flux,
            bundle=bundle,
            matches=matched.matches,
            pair_plan_sha256="c" * 64,
        )
    changed_file = pairs.copy()
    bare_row = changed_file.index[
        changed_file["pair_id"].eq(first_pair) & changed_file["arm"].eq("bare")
    ][0]
    changed_file.loc[bare_row, "pixel_file_sha256"] = "0" * 64
    with pytest.raises(ContractError):
        validate_zero_copy_pairs(
            changed_file,
            bundle=bundle,
            matches=matched.matches,
            pair_plan_sha256="c" * 64,
        )


def test_analysis_manifest_has_exact_final_populations(
    bundle, matched, planned
) -> None:
    _, plan = planned
    analysis = _analysis_manifest(
        bundle,
        matched.matches,
        plan,
        matches_sha256="a" * 64,
        pair_plan_sha256="c" * 64,
    )
    assert len(analysis) == 43 + 2 * len(matched.matches)
    assert analysis["population"].eq("observed_reference").sum() == 43
    assert analysis["population"].eq("bare").sum() == len(matched.matches)
    assert analysis["population"].eq("injected").sum() == len(matched.matches)
    assert set(analysis["data_role"]) == {"final"}
    assert (
        not analysis[
            [
                "pixel_relpath",
                "pixel_file_sha256",
                "pixel_sha256",
                "pixel_array_key",
                "pixel_layout",
            ]
        ]
        .isna()
        .any()
        .any()
    )


def test_package_never_names_grec_score_artifacts() -> None:
    for path in [
        *sorted((PACKAGE / "src").rglob("*.py")),
        *sorted((PACKAGE / "scripts").glob("*.py")),
        CONFIG_PATH,
    ]:
        text = path.read_text(encoding="utf-8").lower()
        assert "grec_scores.parquet" not in text
        assert "results_v2/grec/scores" not in text
