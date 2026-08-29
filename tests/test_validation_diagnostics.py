"""Tests for the Amendment 5 validation diagnostics.

The diagnostics exist to adjudicate between two readings of a declining ROC AUC,
so their own arithmetic has to be trustworthy before their output means anything.
These tests check the properties the adjudication actually leans on: that the
false-positive budget is never exceeded, that the stratified comparison holds the
negatives fixed, that the paired tests are genuinely paired, and that each
function refuses rather than guesses when the input cannot support it.
"""

from __future__ import annotations

import numpy as np
import pytest

from dhs.train import negative_only_threshold
from dhs.validation_diagnostics import (
    DiagnosticError,
    mcnemar_exact,
    paired_detection_comparison,
    paired_score_comparison,
    partial_roc_auc,
    score_concentration,
    stratified_ranking_report,
    threshold_at_false_positive_count,
    tpr_at_false_positive_counts,
    wilson_interval,
)


def _cohort_scores(seed: int = 0):
    """Negatives, clean positives, and noisy positives that rank like negatives.

    The noisy cohort is drawn from the negative distribution on purpose: it is a
    stand-in for mislabeled Tier-B rows, so a stratified report should score it
    near chance while the clean cohort scores well.
    """
    rng = np.random.default_rng(seed)
    neg = rng.uniform(0.0, 0.6, size=5000)
    clean = rng.uniform(0.7, 1.0, size=64)
    noisy = rng.uniform(0.0, 0.6, size=200)
    scores = np.concatenate([neg, clean, noisy])
    labels = np.concatenate([np.zeros(5000), np.ones(64), np.ones(200)]).astype(int)
    cohorts = np.array(["neg"] * 5000 + ["candidate"] * 64 + ["tier_b"] * 200, dtype=object)
    return labels, scores, cohorts


def test_the_false_positive_budget_is_never_exceeded():
    """The realised count may fall below the budget but must never exceed it."""
    rng = np.random.default_rng(1)
    # Heavy ties force the tie policy to bind rather than sit idle.
    negatives = np.round(rng.uniform(0, 1, size=4000), 2)
    for count in (0, 1, 6, 30, 61, 123, 400):
        threshold, allowance = threshold_at_false_positive_count(negatives, count)
        assert allowance == count
        assert int(np.sum(negatives >= threshold)) <= count


def test_the_budget_delegates_to_the_frozen_trainer_threshold():
    """A diagnostic must not restate the operating point it is auditing."""
    rng = np.random.default_rng(2)
    negatives = rng.uniform(0, 1, size=61437)
    for count in (6, 30, 61, 123):
        mine = threshold_at_false_positive_count(negatives, count)
        theirs = negative_only_threshold(negatives, count / negatives.size)
        assert mine == theirs
    # And at the budget the fits actually run at, the allowance is 61 objects.
    assert threshold_at_false_positive_count(negatives, 61)[1] == 61
    assert negative_only_threshold(negatives, 1e-3)[1] == 61


def test_tpr_rises_with_the_false_positive_budget():
    labels, scores, _ = _cohort_scores()
    out = tpr_at_false_positive_counts(labels, scores, (6, 30, 61, 123))
    tprs = [out[f"fp_{k}"]["tpr"] for k in (6, 30, 61, 123)]
    assert tprs == sorted(tprs), "a looser budget cannot detect fewer positives"
    for k in (6, 30, 61, 123):
        row = out[f"fp_{k}"]
        assert row["realised_false_positives"] <= k
        assert row["tpr_wilson_low"] <= row["tpr"] <= row["tpr_wilson_high"]


def test_stratified_report_scores_every_cohort_against_the_same_negatives():
    """The strata must differ only in positives, or they are not comparable."""
    labels, scores, cohorts = _cohort_scores()
    rep = stratified_ranking_report(labels, scores, cohorts)
    by = rep["by_positive_cohort"]
    assert set(by) == {"candidate", "tier_b"}
    assert by["candidate"]["negatives"] == by["tier_b"]["negatives"] == 5000
    assert by["candidate"]["positives"] == 64
    assert by["tier_b"]["positives"] == 200


def test_stratification_separates_label_noise_from_lost_ranking():
    """The distinguishing power the diagnostic is pre-registered to provide.

    Positives drawn from the negative distribution must score near chance while
    genuinely separable positives score near one. If a stratified report cannot
    tell those apart it cannot adjudicate the ROC decline either.
    """
    labels, scores, cohorts = _cohort_scores()
    rep = stratified_ranking_report(labels, scores, cohorts)
    clean = rep["by_positive_cohort"]["candidate"]["roc_auc"]
    noisy = rep["by_positive_cohort"]["tier_b"]["roc_auc"]
    assert clean > 0.99
    assert 0.4 < noisy < 0.6
    # And the pooled figure sits between them, which is precisely why pooling
    # cannot answer the question.
    assert noisy < rep["pooled"]["roc_auc"] < clean


def test_partial_auc_is_normalised_to_mean_tpr_below_the_boundary():
    labels, scores, _ = _cohort_scores()
    out = partial_roc_auc(labels, scores, 1e-3)
    assert out["partial_auc_normalised"] == pytest.approx(
        out["partial_auc"] / 1e-3, rel=1e-9
    )
    assert 0.0 <= out["partial_auc_normalised"] <= 1.0
    # A perfect ranker attains the whole strip.
    perfect = np.concatenate([np.zeros(5000), np.ones(64)])
    sc = np.concatenate([np.zeros(5000), np.ones(64)])
    assert partial_roc_auc(perfect.astype(int), sc, 1e-3)[
        "partial_auc_normalised"
    ] == pytest.approx(1.0, abs=1e-9)


def test_partial_auc_widens_monotonically():
    labels, scores, _ = _cohort_scores()
    areas = [partial_roc_auc(labels, scores, f)["partial_auc"] for f in (1e-3, 1e-2, 1e-1)]
    assert areas == sorted(areas)


def test_saturation_is_detected_where_ranking_cannot_express_a_change():
    labels, scores, _ = _cohort_scores()
    saturated = scores.copy()
    saturated[:100] = 1.0  # a hundred negatives pinned at the representable top
    out = score_concentration(labels, saturated, top=200)
    assert out["negatives_at_or_above_one_minus_1e-6"] == 100
    assert out["tied_fraction_in_top_n"] > 0.4
    clean = score_concentration(labels, scores, top=200)
    assert clean["negatives_at_or_above_one_minus_1e-6"] == 0


def test_wilson_interval_covers_the_point_and_stays_in_range():
    for successes, trials in ((0, 64), (1, 64), (48, 64), (64, 64)):
        low, high = wilson_interval(successes, trials)
        assert 0.0 <= low <= successes / trials <= high <= 1.0
    # The interval must not collapse at the boundaries, which is the reason for
    # preferring Wilson over the normal approximation on 64 objects.
    assert wilson_interval(64, 64)[0] < 1.0
    assert wilson_interval(0, 64)[1] > 0.0


def test_wilson_refuses_impossible_input():
    with pytest.raises(DiagnosticError):
        wilson_interval(65, 64)
    with pytest.raises(DiagnosticError):
        wilson_interval(0, 0)


def test_mcnemar_is_symmetric_and_ignores_concordant_pairs():
    assert mcnemar_exact(5, 1)["p_value"] == mcnemar_exact(1, 5)["p_value"]
    assert mcnemar_exact(0, 0)["p_value"] == 1.0
    # A lopsided split is significant; an even one is not.
    assert mcnemar_exact(12, 0)["p_value"] < 0.01
    assert mcnemar_exact(6, 6)["p_value"] > 0.9
    # Against the textbook value: 2 * P(X <= 1), X ~ Binom(10, 1/2) = 2*11/1024.
    assert mcnemar_exact(9, 1)["p_value"] == pytest.approx(2 * 11 / 1024, rel=1e-12)


def test_paired_detection_thresholds_each_arm_on_its_own_negatives():
    """Equal budgets, not equal scores: the arms are matched on false positives."""
    rng = np.random.default_rng(3)
    labels = np.concatenate([np.zeros(20000), np.ones(64)]).astype(int)
    a = np.concatenate([rng.uniform(0, 0.5, 20000), rng.uniform(0.6, 1.0, 64)])
    # Arm b is arm a squashed by a monotone map, so its scores are on a
    # different scale but its ranking -- and so its detections -- are identical.
    b = a**3
    out = paired_detection_comparison(labels, a, b, 1e-3)
    assert out["threshold_a"] != out["threshold_b"]
    assert out["only_a"] == out["only_b"] == 0
    assert out["recall_a"] == out["recall_b"]
    assert out["mcnemar"]["p_value"] == 1.0


def test_paired_detection_counts_partition_the_positives():
    labels, scores, _ = _cohort_scores(seed=4)
    rng = np.random.default_rng(5)
    other = scores + rng.normal(0, 0.05, size=scores.size)
    out = paired_detection_comparison(labels, scores, other, 1e-3)
    assert out["both"] + out["only_a"] + out["only_b"] + out["neither"] == out["positives"]
    assert out["recall_difference"] == pytest.approx(
        out["recall_a"] - out["recall_b"], abs=1e-12
    )


def test_paired_score_comparison_is_paired_not_marginal():
    """A shared shift must be detected exactly, with no interval ambiguity."""
    labels, scores, _ = _cohort_scores(seed=6)
    shifted = np.clip(scores + 0.01, 0, 1)
    out = paired_score_comparison(labels, shifted, scores, resamples=2000)
    assert out["mean_difference"] > 0
    assert out["b_higher_on"] == 0
    lo, hi = out["mean_difference_ci95"]
    assert lo <= out["mean_difference"] <= hi
    # Identical arms must give an exactly zero difference, not a small one.
    same = paired_score_comparison(labels, scores, scores, resamples=200)
    assert same["mean_difference"] == 0.0
    assert same["tied_on"] == same["positives"]


def test_diagnostics_refuse_a_single_class_or_a_ragged_input():
    ones = np.ones(10, dtype=int)
    with pytest.raises(DiagnosticError):
        tpr_at_false_positive_counts(ones, np.linspace(0, 1, 10))
    with pytest.raises(DiagnosticError):
        partial_roc_auc(ones, np.linspace(0, 1, 10))
    with pytest.raises(DiagnosticError):
        stratified_ranking_report(ones, np.linspace(0, 1, 10), np.array(["a"] * 10))
    with pytest.raises(DiagnosticError):
        tpr_at_false_positive_counts(np.array([0, 1]), np.array([0.1]))
    with pytest.raises(DiagnosticError):
        tpr_at_false_positive_counts(np.array([0, 1]), np.array([0.1, np.nan]))
    with pytest.raises(DiagnosticError):
        stratified_ranking_report(
            np.array([0, 1]), np.array([0.1, 0.9]), np.array(["a", "b", "c"])
        )


def test_a_budget_smaller_than_the_saturated_tie_reports_undefined_not_zero():
    """Sigmoid saturation can make a strict operating point unreachable.

    Seven validation negatives of the tier_ab arm reach 1.0 in float32 at epoch
    160, so a six-false-positive budget cannot be spent: admitting the tie whole
    would overspend it, the cut lands above every score, and nothing is detected.
    Reported as a rate that would read 0 per cent recall, which is a statement
    about the float range rather than the model.
    """
    import numpy as np

    from dhs.validation_diagnostics import tpr_at_false_positive_counts

    # Seven negatives pinned at the ceiling, the rest well below it.
    negatives = np.concatenate([np.ones(7), np.full(993, 0.10)])
    positives = np.full(50, 0.95)
    scores = np.concatenate([positives, negatives])
    labels = np.concatenate([np.ones(50), np.zeros(1000)])

    out = tpr_at_false_positive_counts(labels, scores, counts=(6, 100))

    strict = out["fp_6"]
    assert strict["operating_point_defined"] is False
    assert strict["negatives_tied_at_maximum"] == 7
    assert strict["tpr"] is None, "an unreachable budget must not report a rate"
    assert strict["detected"] is None
    assert strict["tpr_wilson_low"] is None and strict["tpr_wilson_high"] is None

    # A budget larger than the tie is spendable, so the operating point exists and
    # the rate is reported. It is 0 here, and that zero is a real statement: seven
    # negatives sit at the ceiling above every positive. Distinguishing this from
    # the case above is the entire point of the flag, since the two are identical
    # in every other field.
    loose = out["fp_100"]
    assert loose["operating_point_defined"] is True
    assert loose["realised_false_positives"] == 7
    assert loose["tpr"] == 0.0
    assert loose["detected"] == 0


def test_an_unsaturated_arm_keeps_every_budget_defined():
    """The guard must not fire on the ordinary case."""
    import numpy as np

    from dhs.validation_diagnostics import tpr_at_false_positive_counts

    rng = np.random.default_rng(0)
    negatives = rng.uniform(0.0, 0.5, size=1000)
    positives = rng.uniform(0.6, 0.99, size=50)
    scores = np.concatenate([positives, negatives])
    labels = np.concatenate([np.ones(50), np.zeros(1000)])

    for entry in tpr_at_false_positive_counts(labels, scores, counts=(6, 61)).values():
        assert entry["operating_point_defined"] is True
        assert entry["negatives_tied_at_maximum"] == 1
        assert entry["tpr"] == 1.0
