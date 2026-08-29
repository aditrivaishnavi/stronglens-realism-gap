"""Validation-only diagnostics pre-registered by Amendment 5.

Two questions are answered here, both on validation data and neither ever on
``test_heldout.parquet``.

Diagnostic 5 asks why validation ROC AUC declines while recall at FPR 10**-3
rises. The two statistics are both rank statistics, so a drifting score
distribution cannot be the cause; only rank changes, ties, saturation, or noise
in the order statistic that fixes the threshold can move them. The decline also
admits two opposite readings: if late training learns to down-rank mislabeled
Tier-B positives, pooled ROC AUC falls while genuine discrimination improves,
whereas a genuine loss of ranking quality lowers it for every positive cohort.
Stratifying by positive provenance separates those, which pooled metrics cannot.

Diagnostic 4 compares the two mixtures on the population they share. Their
validation negatives are set-equal and the 64 candidate positives of the thin
mixture are the complete intersection of the two positive cohorts, so a paired
per-object comparison is available with no construction choices left to make.

Everything in this module is a pure function of scores and labels so that it can
be tested without a GPU, a checkpoint, or the corpus.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .train import negative_only_threshold

__all__ = [
    "DiagnosticError",
    "mcnemar_exact",
    "paired_detection_comparison",
    "paired_score_comparison",
    "partial_roc_auc",
    "score_concentration",
    "stratified_ranking_report",
    "threshold_at_false_positive_count",
    "tpr_at_false_positive_counts",
    "wilson_interval",
]

# The false-positive counts reported alongside the stratified ROC. 61 is the
# allowance the frozen operating point actually uses on 61,437 negatives; the
# others bracket it by factors of about ten and two so that a change confined to
# one budget is distinguishable from a change across the whole tail.
FALSE_POSITIVE_COUNTS = (6, 30, 61, 123)


class DiagnosticError(RuntimeError):
    """Raised when a diagnostic is asked for something it cannot honestly give."""


def _validate(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape:
        raise DiagnosticError(
            f"labels and scores must align, got {labels.shape} and {scores.shape}"
        )
    if labels.size == 0:
        raise DiagnosticError("no rows to score")
    if not np.isfinite(scores).all():
        raise DiagnosticError("scores contain non-finite values")
    uniq = set(np.unique(labels).astype(int).tolist())
    if not uniq <= {0, 1}:
        raise DiagnosticError(f"labels must be binary, saw {sorted(uniq)}")
    return labels.astype(int), scores


def threshold_at_false_positive_count(
    negatives: np.ndarray, count: int
) -> tuple[float, int]:
    """Return the cut admitting at most ``count`` negatives.

    Delegates to the frozen trainer implementation rather than restating its tie
    policy, so a diagnostic can never disagree with the operating point the fits
    actually used. Ties are admitted whole or not at all, hence the realised
    count may fall below ``count`` but can never exceed it.
    """
    negatives = np.asarray(negatives, dtype=float)
    if negatives.size == 0:
        raise DiagnosticError("a false-positive budget needs negatives")
    if count < 0:
        raise DiagnosticError(f"false-positive count must be non-negative, got {count}")
    # floor((count / n) * n) == count recovers the allowance exactly.
    return negative_only_threshold(negatives, count / negatives.size)


def tpr_at_false_positive_counts(
    labels: np.ndarray,
    scores: np.ndarray,
    counts: tuple[int, ...] = FALSE_POSITIVE_COUNTS,
) -> dict[str, dict[str, float | int]]:
    """True-positive rate at each fixed false-positive budget.

    Reported in counts rather than rates because the budget a survey lives under
    is a number of follow-ups, and because at 61,437 negatives a rate of 10**-3
    is 61 objects -- a quantity whose sampling noise is visible and worth seeing.
    """
    labels, scores = _validate(labels, scores)
    positive = labels == 1
    negatives = scores[~positive]
    if negatives.size == 0:
        raise DiagnosticError("a false-positive budget needs negatives")
    n_pos = int(positive.sum())
    if n_pos == 0:
        raise DiagnosticError("a true-positive rate needs positives")

    # Ties at the ceiling of the score range can exceed a small budget. Admitting
    # them whole would overspend it, so none are admitted, and the cut lands above
    # every score -- including every positive, since nothing can outrank a
    # saturated negative. The measured rate is then 0 for a reason that says
    # nothing about the model: at this budget the operating point does not exist.
    # It is reported as undefined rather than as zero, because a zero here would be
    # quoted as a recall.
    tied_at_max = int(np.sum(negatives >= negatives.max()))

    out: dict[str, dict[str, float | int | bool | None]] = {}
    for count in counts:
        threshold, allowance = threshold_at_false_positive_count(negatives, count)
        detected = int(np.sum(scores[positive] >= threshold))
        realised_fp = int(np.sum(negatives >= threshold))
        defined = not (allowance > 0 and realised_fp == 0 and tied_at_max > allowance)
        low, high = wilson_interval(detected, n_pos) if defined else (None, None)
        out[f"fp_{count}"] = {
            "false_positive_allowance": allowance,
            "realised_false_positives": realised_fp,
            "threshold": float(threshold),
            "detected": detected if defined else None,
            "positives": n_pos,
            "tpr": detected / n_pos if defined else None,
            "tpr_wilson_low": low,
            "tpr_wilson_high": high,
            "operating_point_defined": defined,
            "negatives_tied_at_maximum": tied_at_max,
        }
    return out


def partial_roc_auc(
    labels: np.ndarray, scores: np.ndarray, max_fpr: float = 1e-3
) -> dict[str, float]:
    """Area under the ROC curve restricted to ``FPR <= max_fpr``.

    Returned both raw and normalised by ``max_fpr``. The normalised form is the
    mean true-positive rate over the admitted false-positive range, which is
    comparable across choices of ``max_fpr``; the raw form is not, and is
    included only because it is what the phrase "partial AUC" usually denotes.
    """
    labels, scores = _validate(labels, scores)
    if not 0.0 < max_fpr <= 1.0:
        raise DiagnosticError(f"max_fpr must lie in (0, 1], got {max_fpr}")
    if labels.sum() == 0 or (labels == 0).sum() == 0:
        raise DiagnosticError("partial ROC AUC needs both classes")

    fpr, tpr, _ = roc_curve(labels, scores)
    # Interpolate the curve onto the boundary so the area is not truncated at
    # whichever grid point happens to fall below it.
    if fpr[-1] < max_fpr:
        raise DiagnosticError("ROC curve does not reach the requested max_fpr")
    keep = fpr <= max_fpr
    x = np.append(fpr[keep], max_fpr)
    y = np.append(tpr[keep], np.interp(max_fpr, fpr, tpr))
    raw = float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))
    return {
        "max_fpr": float(max_fpr),
        "partial_auc": raw,
        "partial_auc_normalised": raw / max_fpr,
    }


def score_concentration(
    labels: np.ndarray, scores: np.ndarray, top: int = 200
) -> dict[str, float | int]:
    """Ties and saturation among the highest-scoring negatives.

    A rank statistic can only be disturbed by ties or by saturation against the
    representable maximum. This counts both in the region that sets the
    threshold, so "the ranking changed" can be told apart from "the ranking
    became unable to express a change".
    """
    labels, scores = _validate(labels, scores)
    negatives = np.sort(scores[labels == 0])[::-1]
    if negatives.size == 0:
        raise DiagnosticError("score concentration needs negatives")
    head = negatives[: min(top, negatives.size)]
    return {
        "top_n": int(head.size),
        "distinct_values_in_top_n": int(np.unique(head).size),
        "tied_fraction_in_top_n": float(1.0 - np.unique(head).size / head.size),
        "at_or_above_one_minus_1e-6": int(np.sum(scores >= 1.0 - 1e-6)),
        "negatives_at_or_above_one_minus_1e-6": int(np.sum(negatives >= 1.0 - 1e-6)),
        "max_negative": float(negatives[0]),
        "negative_quantile_0.999": float(np.quantile(negatives, 0.999)),
        "negative_quantile_0.9999": float(np.quantile(negatives, 0.9999)),
    }


def stratified_ranking_report(
    labels: np.ndarray,
    scores: np.ndarray,
    cohorts: np.ndarray,
    max_fpr: float = 1e-3,
    counts: tuple[int, ...] = FALSE_POSITIVE_COUNTS,
) -> dict:
    """Ranking quality overall and within each positive cohort.

    Every stratum is scored against the *same* negatives -- the full validation
    negative set -- so the strata differ only in which positives they contain.
    Comparing them is therefore a comparison of how well the model ranks
    different kinds of positive, not of two different problems.
    """
    labels, scores = _validate(labels, scores)
    cohorts = np.asarray(cohorts, dtype=object)
    if cohorts.shape != labels.shape:
        raise DiagnosticError(
            f"cohorts must align with labels, got {cohorts.shape} and {labels.shape}"
        )
    negative = labels == 0
    if not negative.any():
        raise DiagnosticError("stratified ranking needs negatives")

    def stratum(mask: np.ndarray) -> dict:
        sel = mask | negative
        y, p = labels[sel], scores[sel]
        report: dict = {
            "positives": int(mask.sum()),
            "negatives": int(negative.sum()),
            "roc_auc": float(roc_auc_score(y, p)),
            "tpr_at_fixed_false_positives": tpr_at_false_positive_counts(y, p, counts),
        }
        report.update({"partial_roc": partial_roc_auc(y, p, max_fpr)})
        return report

    out: dict = {
        "pooled": stratum(labels == 1),
        "by_positive_cohort": {},
        "score_concentration": score_concentration(labels, scores),
    }
    for name in sorted({str(c) for c in cohorts[labels == 1]}):
        mask = (labels == 1) & (cohorts == name)
        if mask.sum() == 0:
            continue
        out["by_positive_cohort"][name] = stratum(mask)
    return out


def wilson_interval(
    successes: int, trials: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used in preference to the normal approximation because the shared positive
    cohort has 64 members, where recall moves in steps of 1/64 and the normal
    interval can leave [0, 1] outright.
    """
    if trials <= 0:
        raise DiagnosticError("a proportion needs trials")
    if not 0 <= successes <= trials:
        raise DiagnosticError(f"{successes} successes in {trials} trials is impossible")
    phat = successes / trials
    denom = 1.0 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denom
    half = (z / denom) * np.sqrt(
        phat * (1 - phat) / trials + z * z / (4 * trials * trials)
    )
    low, high = centre - half, centre + half
    # At the boundaries the interval is closed exactly, so pin it there rather
    # than reporting the residue of the square root. Recall on 64 positives sits
    # at a boundary whenever an arm detects all or none of them, and a bound of
    # 3e-18 instead of 0 would be reported as though it were a measurement.
    if successes == 0:
        low = 0.0
    if successes == trials:
        high = 1.0
    return float(max(0.0, low)), float(min(1.0, high))


def mcnemar_exact(only_a: int, only_b: int) -> dict[str, float | int]:
    """Exact McNemar test on discordant pairs.

    The exact binomial form is used rather than the chi-squared approximation
    because with 64 paired positives the discordant count is small enough that
    the approximation is not trustworthy.
    """
    if only_a < 0 or only_b < 0:
        raise DiagnosticError("discordant counts cannot be negative")
    n = only_a + only_b
    if n == 0:
        return {
            "only_a": 0,
            "only_b": 0,
            "discordant": 0,
            "p_value": 1.0,
            "note": "no discordant pairs; the two arms detected the same objects",
        }
    # Two-sided exact binomial p-value against p = 1/2, by summing the tail at
    # or below the observed minimum on both sides.
    k = min(only_a, only_b)
    coef = np.array([_log_binom(n, i) for i in range(0, k + 1)])
    tail = float(np.exp(coef - n * np.log(2.0)).sum())
    return {
        "only_a": int(only_a),
        "only_b": int(only_b),
        "discordant": int(n),
        "p_value": float(min(1.0, 2.0 * tail)),
    }


def _log_binom(n: int, k: int) -> float:
    from math import lgamma

    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def paired_detection_comparison(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    target_fpr: float = 1e-3,
) -> dict:
    """Compare two arms object by object at their own matched operating points.

    Each arm is thresholded on its own negative distribution, so both sit at the
    same false-positive budget rather than the same score. Because the negatives
    are identical across arms, that budget is the same set of follow-ups, which
    is what makes the comparison paired rather than merely simultaneous.
    """
    labels, scores_a = _validate(labels, scores_a)
    _, scores_b = _validate(labels, scores_b)
    positive = labels == 1
    if not positive.any():
        raise DiagnosticError("a paired detection comparison needs positives")

    ta, allowance = negative_only_threshold(scores_a[~positive], target_fpr)
    tb, _ = negative_only_threshold(scores_b[~positive], target_fpr)
    det_a = scores_a[positive] >= ta
    det_b = scores_b[positive] >= tb

    both = int(np.sum(det_a & det_b))
    only_a = int(np.sum(det_a & ~det_b))
    only_b = int(np.sum(det_b & ~det_a))
    neither = int(np.sum(~det_a & ~det_b))
    n = int(positive.sum())
    la, ha = wilson_interval(both + only_a, n)
    lb, hb = wilson_interval(both + only_b, n)
    return {
        "positives": n,
        "target_fpr": float(target_fpr),
        "false_positive_allowance": allowance,
        "threshold_a": float(ta),
        "threshold_b": float(tb),
        "both": both,
        "only_a": only_a,
        "only_b": only_b,
        "neither": neither,
        "recall_a": (both + only_a) / n,
        "recall_b": (both + only_b) / n,
        "recall_a_wilson": [la, ha],
        "recall_b_wilson": [lb, hb],
        "recall_difference": (only_a - only_b) / n,
        "mcnemar": mcnemar_exact(only_a, only_b),
    }


def paired_score_comparison(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    resamples: int = 10000,
    seed: int = 42,
) -> dict:
    """Paired score differences on the positives, with a bootstrap interval.

    Reported alongside the detection comparison because detection at a threshold
    discards the margin: two arms can agree on every detection while differing
    substantially in how confidently they rank the same objects, and a paired
    difference sees that where two marginal recalls cannot.
    """
    labels, scores_a = _validate(labels, scores_a)
    _, scores_b = _validate(labels, scores_b)
    positive = labels == 1
    delta = scores_a[positive] - scores_b[positive]
    if delta.size == 0:
        raise DiagnosticError("a paired score comparison needs positives")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, delta.size, size=(resamples, delta.size))
    means = delta[idx].mean(axis=1)
    return {
        "positives": int(delta.size),
        "mean_difference": float(delta.mean()),
        "median_difference": float(np.median(delta)),
        "a_higher_on": int(np.sum(delta > 0)),
        "b_higher_on": int(np.sum(delta < 0)),
        "tied_on": int(np.sum(delta == 0)),
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
        "mean_difference_ci95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
    }
