from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results_v2/canonical_test_host_repair_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_public_authority_binds_every_included_artifact() -> None:
    authority = load(
        "results_v2/canonical_test_host_repair_v1/PUBLIC_STUDY_AUTHORITY.json"
    )
    assert authority["status"] == "canonical_public_study_authority"
    for relative, expected in authority["artifacts"].items():
        path = ROOT / relative
        assert path.is_file(), relative
        assert sha256(path) == expected


def test_primary_endpoint_values_are_canonical() -> None:
    confirmed = load(
        "results_v2/evaluation_v2/terminal_confirmed_scores/ENDPOINT_RESULT.json"
    )
    achieved = load(
        "results_v2/evaluation_v2/achieved_fpr_scores/ENDPOINT_RESULT.json"
    )
    assert confirmed["detections"] == 39
    assert confirmed["rows"] == 43
    assert achieved["rows"] == 12_512
    assert achieved["detections"] == 13


def test_recovery_and_intervention_values_are_canonical() -> None:
    grec = load(
        "results_v2/canonical_test_host_repair_v1/grec/GREC_ANALYSIS.json"
    )["grec"]
    pnoise = load(
        "results_v2/canonical_test_host_repair_v1/pnoise/PNOISE_ANALYSIS.json"
    )["pnoise"]
    psrc = load(
        "results_v2/canonical_test_host_repair_v1/psrc/analysis/RESULT.json"
    )
    assert grec["distinct_hosts"] == 4_500
    assert grec["estimate"] == pytest.approx(0.034545454545454546)
    assert pnoise["distinct_scenes"] == 49_500
    assert pnoise["estimate"] == pytest.approx(-0.01203030303030303)
    assert psrc["risk_difference_sersic_minus_structured"] == pytest.approx(
        0.00560290404040404
    )


def test_probe_values_are_canonical() -> None:
    domain = load(
        "results_v2/canonical_test_host_repair_v1/gdomain/"
        "GDOMAIN_REPAIR_RESULT.json"
    )["results"]
    movement = load(
        "results_v2/canonical_test_host_repair_v1/gmatch/analysis/"
        "GMATCH_REPAIR_RESULT.json"
    )["results"]
    assert domain["operational_penultimate"][
        "component_equal_oof_auc"
    ] == pytest.approx(0.9913400552935434)
    assert (
        domain["stein_ssl_resnet50"]["component_equal_oof_auc"]
        == 0.9581869680706872
    )
    assert movement["operational_penultimate"]["mean_movement"] == pytest.approx(
        0.023183100799986617
    )
    assert movement["operational_penultimate"]["positive_movements"] == 27
    assert movement["operational_penultimate"]["negative_movements"] == 16


def test_public_reporting_summary_matches_canonical_outputs() -> None:
    summary = load(
        "results_v2/canonical_test_host_repair_v1/PUBLIC_REPORTING_SUMMARY.json"
    )
    domain = load(
        "results_v2/canonical_test_host_repair_v1/gdomain/"
        "GDOMAIN_REPAIR_RESULT.json"
    )
    movement = load(
        "results_v2/canonical_test_host_repair_v1/gmatch/analysis/"
        "GMATCH_REPAIR_RESULT.json"
    )
    assert summary["status"] == "canonical_public_reporting_summary"
    assert summary["gdomain_verification"]["rows"] == domain["rows"]
    assert summary["gdomain_verification"]["dependency_components"] == domain[
        "dependency_components"
    ]
    assert summary["matched_stein_movement"]["mean"] == movement["results"][
        "stein_ssl_resnet50"
    ]["mean_movement"]
    assert summary["psrc"]["pairs"] == 1_584
    for relative, specification in summary["derived_artifacts"].items():
        path = ROOT / relative
        assert path.is_file(), relative
        assert sha256(path) == specification["sha256"]
