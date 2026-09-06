from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from psrc.contracts import (
    ContractError,
    fixed_scorer_closure,
    load_config,
    sha256_file,
)
from psrc.report import validate_status_report
from psrc.validation import validate_source_package

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/psrc_v2_parent_bound.yaml"


def test_retained_cosmos_checkpoint_and_scorer_sources_match() -> None:
    config = load_config(CONFIG_PATH)
    cosmos_dir = Path(config["inputs"]["cosmos_dir"])
    checkpoint = Path(config["scoring"]["checkpoint"])
    if not cosmos_dir.is_dir() or not checkpoint.is_file():
        pytest.skip("external COSMOS sample or detector checkpoint is unavailable")
    assert (
        sha256_file(checkpoint)
        == config["scoring"]["checkpoint_sha256"]
    )
    closure = fixed_scorer_closure(config, verify_runtime=False)
    assert closure["native_logit_dtype"] == "float32"
    assert closure["comparator"] == ">="


def test_prior_actual_photon_canary_is_non_outcome_and_report_is_stale(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG_PATH)
    canary_path = ROOT / "artifacts_v7/actual_photon_scoring_canary.json"
    if not canary_path.is_file():
        pytest.skip("external photon-scoring canary is unavailable")
    canary = json.loads(canary_path.read_text())
    assert canary["status"] == "actual_design_only_photon_raw_logit_canary"
    assert canary["threshold_applied"] is False
    assert canary["final_outcomes_used"] is False
    assert canary["scoring"]["exact_float32_roundtrip_verified"] is True
    assert canary["scoring"]["worker_order_invariant"] is True
    with pytest.raises(ContractError, match="status report is stale"):
        validate_status_report(ROOT / "artifacts_v7/report", config)
    copied = tmp_path / "report"
    shutil.copytree(ROOT / "artifacts_v7/report", copied)
    with (copied / "STATUS_REPORT.md").open("a") as handle:
        handle.write("stale\n")
    with pytest.raises(ContractError, match="stale"):
        validate_status_report(copied, config)


@pytest.mark.skipif(
    "PSRC_V2_SOURCE_ROOT" not in os.environ,
    reason="set PSRC_V2_SOURCE_ROOT after parent-bound source preparation",
)
def test_real_source_package_passes_every_projection_support_gate(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG_PATH)
    root = Path(os.environ["PSRC_V2_SOURCE_ROOT"])
    result = validate_source_package(
        selection_root=root / "selection",
        source_root=root / "sources",
        config=config,
        output=tmp_path / "validation.json",
    )
    assert result["status"] == "pass"
    assert result["selected_sources"] == 144
    assert result["minimum_structured_positive_flux_retained"] >= 0.90
    assert result["minimum_sersic_positive_flux_retained"] >= 0.90
    assert result["maximum_structured_negative_to_positive_flux"] <= 0.20
    assert result["maximum_sersic_negative_to_positive_flux"] <= 0.001
    manifest = pd.read_parquet(root / "sources/source_manifest.parquet")
    assert manifest["structured_is_latent_truth"].eq(False).all()
    assert manifest["structured_contains_residual_measurement_uncertainty"].all()
    measured_native = (
        0.95
        * 0.5
        * (manifest[["native_height_pixels", "native_width_pixels"]].min(axis=1) - 1)
        * manifest["native_pixel_scale_arcsec"]
    )
    expected_support = measured_native.clip(upper=2.35)
    assert np.allclose(
        manifest["common_support_radius_arcsec"],
        expected_support,
        rtol=0,
        atol=1e-12,
    )
