from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_methods_document_matches_canonical_populations() -> None:
    text = (ROOT / "docs/METHODS_AND_LIMITATIONS.md").read_text(encoding="utf-8")
    for required in (
        "49,500 injections on 4,500 distinct DR10 targets",
        "43 selected observed systems and 6,336 injections",
        "AUC 0.958187",
        "The 1,584 paired comparisons",
        "-1.203030 percentage points",
        "+0.560290 percentage points",
    ):
        assert required in text
    for stale in ("115,500", "28,625", "0.9637", "Sixty-four catalogue-positive"):
        assert stale not in text


def test_public_docs_contain_no_internal_workflow_language() -> None:
    files = [
        ROOT / "README.md",
        ROOT / "docs/DATA.md",
        ROOT / "docs/METHODS_AND_LIMITATIONS.md",
        ROOT / "docs/REPRODUCIBILITY.md",
        ROOT / "evaluation_dataset_prep/README.md",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in files)
    for phrase in (
        "Parent conversation",
        "continue overnight",
        "five-experiment rerun",
        "user instruction",
        "no final evaluation dataset",
    ):
        assert phrase not in text


def test_external_artifact_inventory_names_required_inputs() -> None:
    inventory = yaml.safe_load(
        (ROOT / "data/downloads/artifacts.yaml").read_text(encoding="utf-8")
    )["artifacts"]
    for required in (
        "stein_ssl_resnet50",
        "primary_checkpoint",
        "checkpoint_panel_weights",
        "recovery_grid",
        "recovery_grid_metadata",
        "source_photon_banks",
        "historical_sampled_negatives",
        "gmatch_pixel_payloads",
        "psrc_photon_canary",
        "cosmos_25_2",
        "legacy_survey_dr10",
    ):
        assert required in inventory
        assert inventory[required]["availability"]
