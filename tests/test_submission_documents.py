from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_bound_input(relative: str, expected: str) -> None:
    path = ROOT / relative
    if path.is_file():
        assert sha256(path) == expected
        return
    assert len(expected) == 64
    assert relative.startswith(
        (
            "evaluation_dataset_prep/candidate_payloads/",
            "results_v2/canonical_test_host_repair_v1/psrc/expectations/expectations/",
            "results_v2/canonical_test_host_repair_v1/psrc/photons/pairs/",
        )
    )


def test_final_psrc_figure_uses_current_bound_artifacts() -> None:
    figures = json.loads((MANUSCRIPT / "figures/sources.json").read_text())["figures"]
    spec = figures["section3_final_psrc_construction.pdf"]
    assert spec["generator"] == "scripts/generate_final_psrc_illustration.py"
    for relative, expected in spec["input_sha256"].items():
        assert_bound_input(relative, expected)
    assert (
        sha256(MANUSCRIPT / "figures/section3_final_psrc_construction.pdf")
        == spec["output_sha256"]
    )
    assert (
        sha256(MANUSCRIPT / "figures/section3_final_psrc_construction.png")
        == spec["png_sha256"]
    )
    methods = (MANUSCRIPT / "04_injection_models_experiments.tex").read_text()
    assert "figures/section3_final_psrc_construction.pdf" in methods
    assert "selected without detector scores" in normalize_whitespace(methods)
    assert "pair remains in the final paired analysis" in normalize_whitespace(methods)
    expectation_path = next(iter(spec["input_sha256"]))
    illustrated_pair_id = Path(expectation_path).stem
    design = pd.read_parquet(
        ROOT
        / "results_v2/canonical_test_host_repair_v1/psrc/final_design/"
        "pair_design.parquet",
        columns=["pair_id"],
    )
    assert (design["pair_id"] == illustrated_pair_id).sum() == 1
    assert "section3_paired_source_construction.pdf" not in methods


def test_clean_manuscript_describes_current_science_without_revision_history() -> None:
    clean_files = [
        MANUSCRIPT / f"{number:02d}_{name}.tex"
        for number, name in (
            (1, "abstract"),
            (2, "introduction"),
            (3, "data_samples_detector"),
            (4, "injection_models_experiments"),
            (5, "analysis_design_statistical_methods"),
            (6, "results"),
            (7, "discussion"),
            (8, "conclusions"),
            (9, "declarations"),
        )
    ]
    clean_files.extend(
        [
            MANUSCRIPT / "A2_supplementary_analyses.tex",
            MANUSCRIPT / "tables/confirmed_43_provenance.tex",
        ]
    )
    clean_text = normalize_whitespace(
        "\n".join(path.read_text() for path in clean_files)
    ).lower()
    for forbidden in (
        "superseded detector",
        "earlier exploratory analysis",
        "grouping defect",
        "development-contacted",
        "terminal stratum",
        "test-facing",
        "outcomes were opened",
        "machinery variability",
        "the apparatus",
        "factor-four replay",
        "fixed-width contract",
        "audit trail",
        "visibility-enriched",
        "historical run",
        "historical audit",
        "submitted analysis",
        "superseded experiment",
        "previous run",
        "32.5--45",
        "clip 20",
        "30/43",
    ):
        assert forbidden not in clean_text

    declarations = normalize_whitespace(
        (MANUSCRIPT / "09_declarations.tex").read_text()
    )
    assert "Aditrivaishnavi Balaji conceived and led the project" in declarations
    assert "designed and executed the experiments and statistical analyses" in declarations
    assert "Parker T. Johnson supervised and mentored the project" in declarations
    assert "critically reviewed the manuscript" in declarations


def test_terminal_endpoint_and_psrc_scope_are_precise() -> None:
    abstract = normalize_whitespace((MANUSCRIPT / "01_abstract.tex").read_text())
    assert "Dependency-aware linear probes compare" in abstract
    assert "43 selected observed systems" in abstract
    assert "single-S\\'{e}rsic construction is recovered more often" in abstract
    assert "recovers 3.45 per cent" in abstract
    assert "49,500-scene grid with no visibility cut" in abstract
    assert "operational detector's pooled penultimate features" in abstract
    assert "On 43 fixed targets, expectation-image recovery" in abstract
    assert "peaks at 18.6 per cent" in abstract
    assert "varies non-monotonically with source brightness" in abstract
    assert "reveal differences between the selected populations" in abstract
    assert "provide a practical basis" in abstract
    assert "0.991" in abstract
    assert "0.978" not in abstract
    assert "unique morphology cause" not in abstract
    assert "certify physical realism" not in abstract
    assert "43 held-out confirmed systems" not in abstract

    conclusions = normalize_whitespace((MANUSCRIPT / "08_conclusions.tex").read_text())
    assert "no tested construction" not in conclusions
    assert "single-S\\'{e}rsic construction has higher recovery" in conclusions
    assert "A leading pixel-statistic offset largely predates insertion" in conclusions
    assert "near-zero mean signed movement with heterogeneous individual shifts" in conclusions
    assert "future injection pipelines should be compared" in conclusions
    assert "factor of several" not in conclusions
    assert conclusions.index("representation-level diagnostic") < conclusions.index(
        "Separately, the finder recovers"
    )

    discussion = normalize_whitespace((MANUSCRIPT / "07_discussion.tex").read_text())
    assert "The near-zero cohort mean therefore coexists with substantial individual movements" in discussion
    assert "remain outside that matching set" in discussion
    assert "do not estimate a causal foreground contribution" not in discussion
    assert "heterogeneous movements do not establish equivalence" not in discussion
    assert "that effect is not estimated here" not in discussion

    data_section = normalize_whitespace(
        (MANUSCRIPT / "03_data_samples_detector.tex").read_text()
    )
    assert (
        "63 confirmed systems were excluded from detector fitting, validation, "
        "checkpoint selection, and threshold calibration"
    ) in data_section
    assert "superseded submitted checkpoint" not in data_section
    assert "prior project development" not in data_section
    assert (
        "documented general scientific eligibility rule that independently "
        "selects those 43 from all 63 confirmed systems"
    ) in data_section
    assert "Exact footprint polygons are unavailable" in data_section
    assert "Twenty had an exploratory feature-analysis-development role" in data_section
    assert "not document a scientific reason for this exemption" in data_section
    assert "needed to recheck those two eligibility criteria row by row" in data_section
    assert "The repeated seeds probe sensitivity to training stochasticity" in data_section
    assert (
        "classified as candidates---rather than confirmed lenses, non-lenses, "
        "or uncertain systems"
    ) in data_section

    methods = normalize_whitespace(
        (MANUSCRIPT / "05_analysis_design_statistical_methods.tex").read_text()
    )
    assert r"z=\Phi^{-1}(0.975)\simeq1.96" in methods
    assert "so that a draw does not change stratum mass" not in methods
    assert r"(a_g,a_r,a_z)=(6.0,3.4,2.2)" in methods
    assert r"(R,G,B)&=\operatorname{clip}_{[0,1]}" in methods
    assert "comparisons between these checkpoint policies are exploratory" in methods

    results = normalize_whitespace((MANUSCRIPT / "06_results.tex").read_text())
    assert "Across all eight magnitudes, expectation-image recovery is non-monotonic" in results
    assert "non-monotonic across the three brightest nodes" not in results

    appendix = normalize_whitespace(
        (MANUSCRIPT / "A2_supplementary_analyses.tex").read_text()
    )
    assert "local-pixel criterion is met in 1/16 cases" in appendix
    assert "shape criterion in 0/16" in appendix

    experiments = normalize_whitespace(
        (MANUSCRIPT / "04_injection_models_experiments.tex").read_text()
    )
    assert "continuous cross-fitted readouts in the four feature protocols" in experiments
    assert "primary detector's global-average-pooling layer" in methods
    assert r"\(N1\) contains the remaining candidates" in data_section
    assert r"\(N2\) uses proxies for lens-like negative morphology" in data_section


def test_manuscript_training_and_experiment_narrative_is_scoped() -> None:
    source = normalize_whitespace(
        "\n".join(path.read_text() for path in sorted(MANUSCRIPT.glob("*.tex")))
    )
    source_lower = source.lower()
    for forbidden in (
        "each fit runs",
        "clean-universe",
        "factor-of-several",
        "physical-minus-expectation",
        "physical draws",
        "physical-source decision",
    ):
        assert forbidden not in source_lower

    data = normalize_whitespace(
        (MANUSCRIPT / "03_data_samples_detector.tex").read_text()
    )
    assert r"\subsection{Primary detector protocol}" in data
    assert r"\subsection{Detector-sensitivity training panel}" in data
    assert "seeded 50:50 class-balanced weighted sampling with replacement" in data
    assert "halves the learning rate after epoch 130" in data
    assert "select the checkpoint with the highest validation average precision" in data

    experiments = normalize_whitespace(
        (MANUSCRIPT / "04_injection_models_experiments.tex").read_text()
    )
    assert experiments.index(r"$P_{\rm noise}$") < experiments.index(r"$P_{\rm src}$")
    assert "each arm reuses its own spatial map across the three bands" in experiments

    appendix = normalize_whitespace(
        (MANUSCRIPT / "A2_supplementary_analyses.tex").read_text()
    )
    assert "Val.\\ AP" in appendix
    assert "A separate author-run implementation checks the" in appendix


def test_reader_facing_training_terms_use_plain_set_composition() -> None:
    clean_text = "\n".join(
        path.read_text()
        for path in sorted(MANUSCRIPT.glob("*.tex"))
    )
    for internal_term in ("Tier-A", "Tier-B", "C+TB", "candidate-plus-Tier-B"):
        assert internal_term not in clean_text

    normalized = normalize_whitespace(clean_text)
    for invented_term in (
        "evidence-review cohort",
        "supplementary-candidate cohort",
        "reviewed-candidate cohort",
        "expanded mixture",
        "reviewed-only mixture",
    ):
        assert invented_term not in normalized

    for required in (
        "389 highest-confidence records underwent object-level literature review",
        "4,399 lower-confidence catalogue candidates provide an additional source",
        "The primary training set combines reviewed and additional catalogue candidates",
        "the comparison set uses reviewed candidates only",
        r"\caption{Detector-development and evaluation roles.",
        "Variant & Positive examples",
    ):
        assert required in normalized

def test_submission_uses_canonical_psrc_result() -> None:
    result = json.loads(
        (
            ROOT
            / "results_v2/canonical_test_host_repair_v1/psrc/analysis/RESULT.json"
        ).read_text()
    )
    assert result["status"] == "complete"
    assert result["risk_difference_sersic_minus_structured"] == pytest.approx(
        0.00560290404040404
    )


def test_pdf_link_configuration_uses_one_hyperref_load() -> None:
    main = (MANUSCRIPT / "main.tex").read_text()
    assert r"\usepackage{hyperref}" not in main
    assert r"\hypersetup{hypertexnames=false}" in main


def test_abstract_stays_within_rasti_word_limit() -> None:
    text = (MANUSCRIPT / "01_abstract.tex").read_text()
    text = re.sub(r"%.*", "", text)
    text = re.sub(r"\\begin\{[^}]+\}|\\end\{[^}]+\}", " ", text)
    text = re.sub(r"\\[A-Za-z]+\*?(?:\[[^]]*\])?", " ", text)
    text = re.sub(r"[{}$\\_^~]", " ", text)
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text)
    assert len(words) <= 250


def test_principal_contribution_and_pnoise_pilot_are_synchronized() -> None:
    files = {
        name: normalize_whitespace((MANUSCRIPT / name).read_text())
        for name in (
            "01_abstract.tex",
            "02_introduction.tex",
            "06_results.tex",
            "07_discussion.tex",
            "08_conclusions.tex",
        )
    }
    assert "0.991, 0.958, 0.979, and 0.981, respectively" in files[
        "01_abstract.tex"
    ]
    assert "Comparing the four protocols shows" in files[
        "02_introduction.tex"
    ]
    assert "distinguishable in all four feature protocols" in files["06_results.tex"]
    assert "representation can serve as a detector-specific diagnostic" in files[
        "07_discussion.tex"
    ]
    assert "representation-level diagnostic" in files["08_conclusions.tex"]
    assert "complementary central measurements" not in " ".join(files.values())

    appendix = normalize_whitespace(
        (MANUSCRIPT / "A2_supplementary_analyses.tex").read_text()
    )
    for required in (
        r"\label{tab:pnoise_draw_pilot}",
        "0.003553",
        "0.001119",
        "0.000791",
        r"\textbf{0.000559}",
        "unavailable",
    ):
        assert required in appendix
def test_public_checkpoint_sensitivity_uses_best_validation_panel() -> None:
    result = json.loads(
        (
            ROOT / "results_v2/checkpoint_sensitivity_v2/"
            "FINAL_CHECKPOINT_SENSITIVITY_RESULTS.json"
        ).read_text()
    )
    models = result["models"]
    assert len(models) == 12
    assert [
        min(row["model"]["checkpoint_epoch"] for row in models),
        max(row["model"]["checkpoint_epoch"] for row in models),
    ] == [17, 103]
    selected_counts = sorted(row["terminal_confirmed"]["detections"] for row in models)
    assert [selected_counts[0], selected_counts[-1]] == [35, 39]
    repaired = list(
        json.loads(
            (
                ROOT
                / "results_v2/canonical_test_host_repair_v1/checkpoint_sensitivity/"
                "interventions/INTERVENTION_PANEL_REPAIR_INFERENCE.json"
            ).read_text()
        )["models"].values()
    )
    assert (
        sum(
            row["psrc"]["threshold_conditional_95_interval"][0] > 0
            for row in repaired
        )
        == 7
    )
    assert all(
        row["pnoise"]["confidence_interval_sampling"][1] < 0
        for row in repaired
    )

    source = normalize_whitespace(
        "\n".join(path.read_text() for path in sorted(MANUSCRIPT.glob("*.tex")))
    )
    source_lower = source.lower()
    assert "within-run selection metric is average precision from float32 sigmoid outputs" in source
    assert "selected-cohort response ranges from 35/43 to 39/43" in source_lower
    assert "selected-cohort response ranges from 34/43 to 39/43" not in source_lower


def test_manuscript_uses_corrected_probe_results() -> None:
    results = normalize_whitespace((MANUSCRIPT / "06_results.tex").read_text())
    for expected in (
        "0.991",
        "0.958",
        "0.979",
        "0.981",
        "0.980",
        r"\(+0.023\)",
        r"\(+0.235\)",
        r"\(+0.025\)",
        r"\(-0.029\)",
        "0.998 & 0.998 & 0.509",
        "0.979 & 0.982 & 0.488",
        "0.973 & 0.971 & 0.500",
        "0.928 & 0.933 & 0.478",
    ):
        assert expected in results
    appendix = normalize_whitespace(
        (MANUSCRIPT / "A2_supplementary_analyses.tex").read_text()
    )
    for expected in (
        "0.991340 / 0.958187 / 0.978528 / 0.980745",
        "+0.023183 / +0.235027 / +0.024792 / -0.029207",
        "13.953 & 12.500",
    ):
        assert expected in appendix
    for superseded in (
        "0.935714",
        "0.965580",
        r"\(-0.031476\)",
        r"\(+0.037830\)",
        "0.99243",
        "0.99405",
        "0.49432",
    ):
        assert superseded not in results


def test_final_table_and_recovery_figure_provenance() -> None:
    table_spec = json.loads((MANUSCRIPT / "tables/sources.json").read_text())["tables"][
        "confirmed_43_provenance"
    ]
    assert sha256(ROOT / table_spec["generator"]) == table_spec["generator_sha256"]
    assert sha256(ROOT / table_spec["input"]) == table_spec["input_sha256"]
    for relative, expected in table_spec["outputs"].items():
        assert sha256(ROOT / relative) == expected

    figure_spec = json.loads((MANUSCRIPT / "figures/sources.json").read_text())[
        "figures"
    ]["section5_grouped_recovery.pdf"]
    assert sha256(ROOT / figure_spec["generator"]) == figure_spec["generator_sha256"]
    assert (
        sha256(MANUSCRIPT / "figures/section5_grouped_recovery.pdf")
        == (figure_spec["output_sha256"])
    )
    figures = json.loads((MANUSCRIPT / "figures/sources.json").read_text())[
        "figures"
    ]
    for name in (
        "appendix_grec_cell_radius_response.pdf",
        "appendix_brightness_response.pdf",
    ):
        spec = figures[name]
        assert sha256(ROOT / spec["generator"]) == spec["generator_sha256"]
        for relative, expected in spec["input_sha256"].items():
            assert_bound_input(relative, expected)
        assert sha256(MANUSCRIPT / "figures" / name) == spec["output_sha256"]


def test_retained_process_and_missed_system_figures_are_bound() -> None:
    figures = json.loads((MANUSCRIPT / "figures/sources.json").read_text())["figures"]
    for filename in (
        "section2_data_training_flow.pdf",
        "appendix_missed_selected_systems.pdf",
    ):
        spec = figures[filename]
        assert sha256(ROOT / spec["generator"]) == spec["generator_sha256"]
        for relative, expected in spec["input_sha256"].items():
            assert_bound_input(relative, expected)
        assert sha256(MANUSCRIPT / "figures" / filename) == spec["output_sha256"]
        assert (
            sha256(MANUSCRIPT / "figures" / Path(filename).with_suffix(".png"))
            == spec["png_sha256"]
        )

    data_text = (MANUSCRIPT / "03_data_samples_detector.tex").read_text()
    experiment_text = (MANUSCRIPT / "04_injection_models_experiments.tex").read_text()
    appendix_text = (MANUSCRIPT / "A2_supplementary_analyses.tex").read_text()
    assert "figures/section2_data_training_flow.pdf" in data_text
    assert "figures/section3_experiment_dependency_flow.pdf" not in experiment_text
    assert "figures/appendix_missed_selected_systems.pdf" in appendix_text

    scores = pd.read_parquet(
        ROOT
        / "results_v2/evaluation_v2/terminal_confirmed_scores/endpoint_scores.parquet",
        columns=["galaxy_id", "detected_at_frozen_threshold"],
    )
    misses = set(
        scores.loc[~scores["detected_at_frozen_threshold"], "galaxy_id"].astype(str)
    )
    assert misses == {
        "3450p222_3227",
        "3277p137_6699",
        "0753m245_3999",
        "0728p000_6851",
    }
