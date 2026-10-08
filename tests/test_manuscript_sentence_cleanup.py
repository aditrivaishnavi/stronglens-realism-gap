from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"


def text(name: str) -> str:
    return (MANUSCRIPT / name).read_text(encoding="utf-8")


def normalized(name: str) -> str:
    return " ".join(text(name).split())


def test_required_sentence_corrections_are_present() -> None:
    data = normalized("03_data_samples_detector.tex")
    methods = normalized("05_analysis_design_statistical_methods.tex")
    construction = normalized("04_injection_models_experiments.tex")
    results = normalized("06_results.tex")
    discussion = normalized("07_discussion.tex")
    appendix = normalized("A2_supplementary_analyses.tex")

    assert "each training row is used once per epoch" not in data
    assert "sampled without replacement each epoch" in data
    assert "with the incomplete final mini-batch omitted" in data

    assert "No member entered fitting" not in methods
    assert "None was used for detector fitting or validation" in methods
    assert "detector-model selection, threshold calibration" in methods

    assert "deconvolved, regularized, truncated to common support" not in discussion
    assert "deconvolution and reconvolution to a common resolution" in discussion
    assert "followed by non-negative projection, centring" in discussion

    assert "design crosses the nine target cells" not in construction
    assert "design has 99 target-cell--radius blocks" in construction
    assert "Each block contains 16 distinct targets" in construction

    assert "dependency weighting preventing" not in discussion
    assert "dependency-closed folds keep linked images together" in discussion
    assert "component-equal weights prevent repeated injections" in discussion

    assert "4,568-row replay below" not in appendix
    assert "4,568-row replay in" in appendix
    assert r"Appendix~\ref{app:photon_followup}" in appendix
    assert "for the estimator reported in" not in appendix
    assert "for the estimate reported in" in appendix

    assert "Monte Carlo standard errors (MCSEs)" in construction
    assert construction.index("Monte Carlo standard errors (MCSEs)") == (
        construction.index("MCSE") - len("Monte Carlo standard errors (")
    )

    assert "values retain each protocol's training history" not in results


def test_selected_polish_replaces_imprecise_or_mechanical_prose() -> None:
    data = normalized("03_data_samples_detector.tex")
    construction = normalized("04_injection_models_experiments.tex")
    methods = normalized("05_analysis_design_statistical_methods.tex")
    results = normalized("06_results.tex")
    discussion = normalized("07_discussion.tex")
    conclusions = normalized("08_conclusions.tex")
    appendix = normalized("A2_supplementary_analyses.tex")

    for stale in (
        "only as noisy training labels",
        "Every reported quantity retains",
        '``None\'\' in',
        "three thresholded response experiments",
        "operationally defines the stratum",
        "Gaussian source clumps are disabled",
        "before the complete pair allocation is defined below",
    ):
        assert stale not in data + construction

    assert "These point summaries condition on the frozen score-blind assignment" in methods
    assert "Each finite-cohort AUC describes its feature protocol" in results
    assert "The Stein movement distribution is asymmetric" in results
    assert "The elementary-image result provides a concrete constraint" not in results
    assert "do not separate the possible contributions listed above" in discussion
    assert "expectation-image recovery peaks at magnitude 20.5" in discussion
    assert "different realism channels" not in discussion
    assert "the effective PSF smooths the rendered lensed images" in discussion
    assert "14,850,866 pair-pixel bytes" not in appendix
    assert "Direct checks of the stored bare and injected arrays" in appendix
    assert "future injection pipelines should be compared" in conclusions


def test_appendix_tables_render_in_numerical_order() -> None:
    appendix = text("A2_supplementary_analyses.tex")
    provenance_input = r"\input{tables/confirmed_43_provenance}"
    assert provenance_input + "\n\\FloatBarrier" in appendix
    assert r"\begin{minipage}{\columnwidth}" in appendix

    rendered = subprocess.check_output(
        ["pdftotext", "-layout", str(MANUSCRIPT / "manuscript.pdf"), "-"],
        text=True,
    )
    table_a11 = rendered.index(
        "Table A11. Object-level provenance of the selected 43-system endpoint."
    )
    table_a12 = rendered.index(
        "Table A12. Measured-covariate support for the 43 references"
    )
    assert table_a11 < table_a12


def test_followup_terminology_and_accessibility_corrections() -> None:
    data = normalized("03_data_samples_detector.tex")
    construction = normalized("04_injection_models_experiments.tex")
    methods = normalized("05_analysis_design_statistical_methods.tex")
    results = normalized("06_results.tex")
    conclusions = normalized("08_conclusions.tex")
    appendix = normalized("A2_supplementary_analyses.tex")
    alt_text = normalized("FIGURE_ALT_TEXT.md")

    assert "uncertain catalogue candidates" not in data
    assert "literature-reviewed candidate set" in data
    assert "lower-confidence catalogue-candidate set" in data
    assert "both candidate sources" not in methods
    assert "reviewed and additional catalogue candidates" in methods
    assert "nominal source-plane centre" in construction
    assert "equal weighting of observing cells, injection targets, and Einstein radii" in results
    assert "authority MPS backend" not in appendix
    assert "authority backend" not in appendix
    assert appendix.count("recorded MPS backend") >= 2
    assert "no-visibility-cut grid" not in conclusions
    assert "full grid without a visibility cut" in conclusions
    assert "about 6.2 per cent" in alt_text
    assert "about 5.8 per cent" not in alt_text
