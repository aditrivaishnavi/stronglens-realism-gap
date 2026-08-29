#!/usr/bin/env python
"""The section 5 secondary classifier endpoints, from re-scored validation scores.

Revision requirement R11. Consumes the per-object score files written by
``scripts/score_validation_logits.py``, so it needs no GPU, no corpus and no
checkpoint, and it reads the validation split only.

Every statistic here is computed by code that already existed. This module
selects the populations, applies the thresholds and arranges the output; it
implements no estimator of its own:

* reliability curves -- ``dhs.calibration.reliability_curve``
* recall and precision at a threshold -- the local helpers below
* operating points at a false-positive budget -- ``dhs.validation_diagnostics``
  (which delegates the tie policy to ``dhs.train.negative_only_threshold``, so a
  threshold reported here cannot disagree with the one the fits used)
* ties and saturation at the top of the negative ranking --
  ``dhs.validation_diagnostics.score_concentration``
* ROC AUC, PR AUC and Brier score -- scikit-learn

Two constraints from the analysis plan are enforced rather than remembered.
Section 5 does not list ECE or MCE, and section 5's closing sentence forbids
reporting them as normal calibration measures for a positive-only group; no ECE
or MCE is computed anywhere in this file. The selected identifiers are checked
against the separate test manifest, and the measured overlap count is written
to the output.

Usage:

    python scripts/run_validation_calibration.py \\
        --arm tier_ab_historical/seed_42=<scores dir> \\
        --confirmatory tier_ab_historical/seed_42 \\
        --epoch 160 \\
        --out results/<run>/validation_calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dhs.calibration import reliability_curve  # noqa: E402
from dhs.data_splits import count_test_rows  # noqa: E402
from dhs.validation_diagnostics import (  # noqa: E402
    DiagnosticError,
    score_concentration,
    threshold_at_false_positive_count,
    wilson_interval,
)

# The false-positive rates the plan declares as operating points (section 4 for
# 10**-3, section 5 for 10**-4). Expressed as rates and converted to a count
# against the arm's own negatives, because the threshold is an order statistic
# of the negatives and the count is what a follow-up programme actually spends.
TARGET_FPRS = (1e-3, 1e-4)


def recall_at_threshold(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> float:
    positive = np.asarray(labels).astype(int) == 1
    if not positive.any():
        return float("nan")
    return float((np.asarray(scores)[positive] >= threshold).mean())


def precision_at_threshold(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> float:
    selected = np.asarray(scores) >= threshold
    if not selected.any():
        return float("nan")
    return float((np.asarray(labels).astype(int)[selected] == 1).mean())

# Section 5: "recall at fixed score thresholds 0.3 and 0.5 as legacy descriptive
# results". Descriptive because a newly trained model may have different
# calibration, which is precisely what this file is measuring.
LEGACY_SCORE_THRESHOLDS = (0.3, 0.5)

# The plan names a "high-score-region reliability curve" without pinning the
# region, so the boundary is declared here rather than chosen per run. 0.5 is
# where the model asserts a positive; 0.9 is the region the operating points
# actually live in at 1.25 per cent prevalence.
HIGH_SCORE_FLOORS = (0.5, 0.9)

RELIABILITY_BINS = 15
HIGH_SCORE_BINS = 10

# Quantiles reported for every score distribution. The upper tail is resolved
# finely because the operating points are order statistics of it and because
# float32 saturation is only visible above 0.999.
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 0.9999, 1.0)


def _curve_to_dict(curve) -> dict:
    """Serialise a ReliabilityCurve, keeping empty bins as null rather than 0.

    ``run_evaluation`` writes ``nan_to_num(..., nan=0.0)``, which turns an empty
    bin into a bin with zero observed frequency. At 1.25 per cent prevalence
    most equal-width bins are empty, so that substitution would manufacture a
    calibration curve that sits on the floor.
    """
    return {
        "bin_edges": [float(x) for x in curve.bin_edges],
        "bin_observed_frequency": [
            None if not np.isfinite(x) else float(x) for x in curve.bin_acc
        ],
        "bin_mean_predicted": [
            None if not np.isfinite(x) else float(x) for x in curve.bin_conf
        ],
        "bin_counts": [int(x) for x in curve.bin_counts],
    }


def _distribution(scores: np.ndarray) -> dict:
    if scores.size == 0:
        return {"n": 0}
    return {
        "n": int(scores.size),
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)) if scores.size > 1 else None,
        "quantiles": {str(q): float(np.quantile(scores, q)) for q in QUANTILES},
        "at_float32_one": int(np.sum(scores >= np.float32(1.0))),
        "at_or_above_one_minus_1e-6": int(np.sum(scores >= 1.0 - 1e-6)),
    }


def _operating_points(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Every declared operating point, with recall and precision at each.

    The budgeted points take their threshold from the negatives alone, which is
    what the plan requires: the threshold is chosen on validation negatives and
    then applied, so it does not see the positives it is about to be scored on.
    """
    positive = labels == 1
    negatives = scores[~positive]
    n_pos = int(positive.sum())
    if negatives.size == 0 or n_pos == 0:
        raise DiagnosticError("an operating point needs both classes")
    tied_at_max = int(np.sum(negatives >= negatives.max()))

    points: dict[str, dict] = {}
    for target in TARGET_FPRS:
        count = int(np.floor(target * negatives.size))
        threshold, allowance = threshold_at_false_positive_count(negatives, count)
        realised_fp = int(np.sum(negatives >= threshold))
        # Ties at the representable ceiling can exceed a small budget. The
        # trainer's tie policy then admits none of them and the cut lands above
        # every score, including every positive. Recall is 0 for a reason that
        # says nothing about the model, so it is reported as undefined rather
        # than as a recall of zero.
        defined = not (allowance > 0 and realised_fp == 0 and tied_at_max > allowance)
        detected = int(np.sum(scores[positive] >= threshold))
        low, high = wilson_interval(detected, n_pos) if defined else (None, None)
        points[f"fpr_{target:.0e}"] = {
            "kind": "validation-derived threshold at a false-positive budget",
            "target_fpr": float(target),
            "false_positive_allowance": allowance,
            "realised_false_positives": realised_fp,
            "negatives": int(negatives.size),
            "threshold": float(threshold),
            "operating_point_defined": defined,
            "negatives_tied_at_maximum": tied_at_max,
            "recall": detected / n_pos if defined else None,
            "recall_numerator": detected if defined else None,
            "recall_denominator": n_pos,
            "recall_wilson_95": [low, high],
            "precision": float(precision_at_threshold(labels, scores, threshold))
            if defined
            else None,
            "predicted_positive": int(np.sum(scores >= threshold)),
        }

    for threshold in LEGACY_SCORE_THRESHOLDS:
        predicted_positive = int(np.sum(scores >= threshold))
        detected = int(np.sum(scores[positive] >= threshold))
        low, high = wilson_interval(detected, n_pos)
        points[f"score_{threshold}"] = {
            "kind": "fixed score threshold, legacy descriptive",
            "threshold": float(threshold),
            "recall": float(recall_at_threshold(labels, scores, threshold)),
            "recall_numerator": detected,
            "recall_denominator": n_pos,
            "recall_wilson_95": [low, high],
            "precision": float(precision_at_threshold(labels, scores, threshold)),
            "predicted_positive": predicted_positive,
            "realised_false_positives": int(np.sum(scores[~positive] >= threshold)),
        }
    return points


def calibration_report(
    labels: np.ndarray,
    scores: np.ndarray,
    cohorts: np.ndarray,
) -> dict:
    """The section 5 endpoint set for one arm at one epoch."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape:
        raise DiagnosticError("labels and scores must align")
    if not np.isfinite(scores).all():
        raise DiagnosticError("scores contain non-finite values")

    positive = labels == 1
    report: dict = {
        "n": int(labels.size),
        "n_positive": int(positive.sum()),
        "n_negative": int((~positive).sum()),
        "prevalence": float(positive.mean()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc_average_precision": float(average_precision_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
        # At 1.25 per cent prevalence the pooled Brier score is dominated by
        # negatives, so class-conditional values are reported beside it.
        "brier_positives_only": float(np.mean((scores[positive] - 1.0) ** 2))
        if positive.any()
        else None,
        "brier_negatives_only": float(np.mean(scores[~positive] ** 2))
        if (~positive).any()
        else None,
        "operating_points": _operating_points(labels, scores),
        "reliability_full_sample": {
            "equal_width": _curve_to_dict(
                reliability_curve(labels, scores, RELIABILITY_BINS, equal_mass=False)
            ),
            "equal_mass": _curve_to_dict(
                reliability_curve(labels, scores, RELIABILITY_BINS, equal_mass=True)
            ),
        },
        "reliability_high_score_region": {},
        "class_conditional_scores": {
            "positive": _distribution(scores[positive]),
            "negative": _distribution(scores[~positive]),
        },
        "negative_tail_concentration": score_concentration(labels, scores),
        "score_distribution_by_provenance_group": {},
    }

    for floor in HIGH_SCORE_FLOORS:
        keep = scores >= floor
        block: dict = {
            "floor": float(floor),
            "n": int(keep.sum()),
            "n_positive": int((labels[keep] == 1).sum()),
            "n_negative": int((labels[keep] == 0).sum()),
        }
        if keep.sum() > 0:
            block["curve_equal_width"] = _curve_to_dict(
                reliability_curve(
                    labels[keep], scores[keep], HIGH_SCORE_BINS, equal_mass=False
                )
            )
            block["observed_frequency"] = float((labels[keep] == 1).mean())
            block["mean_predicted"] = float(scores[keep].mean())
            # Both classes are present in the high-score region, so this is a
            # calibration statement rather than a positive-only one.
            block["single_class"] = bool(
                block["n_positive"] == 0 or block["n_negative"] == 0
            )
        report["reliability_high_score_region"][f"score_ge_{floor}"] = block

    # Provenance groups. The negatives carry a cohort label too, so both classes
    # appear here; a positive-only group is marked as such and no calibration
    # statement is attached to it, per section 5.
    cohorts = np.asarray(cohorts, dtype=object)
    for name in sorted({str(c) for c in cohorts}):
        mask = cohorts == name
        group_labels = labels[mask]
        block = {
            "n": int(mask.sum()),
            "n_positive": int((group_labels == 1).sum()),
            "n_negative": int((group_labels == 0).sum()),
            "positive_only": bool((group_labels == 0).sum() == 0),
            "negative_only": bool((group_labels == 1).sum() == 0),
            "scores": _distribution(scores[mask]),
        }
        if not block["positive_only"] and not block["negative_only"]:
            block["roc_auc_within_group"] = float(
                roc_auc_score(group_labels, scores[mask])
            )
        else:
            block["calibration_note"] = (
                "Single-class provenance group. Section 5 of the analysis plan "
                "forbids reporting ECE or MCE for a positive-only group, and no "
                "calibration measure is attached here: the group contains no "
                "outcome of the other class against which a predicted "
                "probability could be checked."
            )
        report["score_distribution_by_provenance_group"][name] = block

    return report


def _load_epoch(directory: Path, epoch: int) -> pd.DataFrame:
    path = directory / f"epoch_{epoch:03d}.parquet"
    if not path.exists():
        raise DiagnosticError(f"{path} does not exist")
    return pd.read_parquet(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="arm label and the directory of its re-scored validation scores",
    )
    ap.add_argument(
        "--confirmatory",
        required=True,
        help="the arm name that is the pre-designated confirmatory model",
    )
    ap.add_argument("--epoch", type=int, default=160)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report: dict = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": (
            "revision requirement R11: the section 5 secondary classifier "
            "endpoints, computed on validation only"
        ),
        "analysis_plan": "docs/RASTI_REVISION_STATISTICAL_ANALYSIS_PLAN.md section 5",
        "test_facing": False,
        "split": "val",
        "epoch": int(args.epoch),
        "confirmatory_arm": args.confirmatory,
        "ece_mce_reported": False,
        "ece_mce_note": (
            "Section 5 does not list ECE or MCE among the secondary endpoints, "
            "and forbids reporting them as normal calibration measures for a "
            "positive-only group. Neither is computed in this run."
        ),
        "high_score_region_definition": (
            "score >= 0.5 and score >= 0.9; the plan names the curve but does "
            "not pin the region, so the boundaries are declared here"
        ),
        "arms": {},
    }

    for spec in args.arm:
        if "=" not in spec:
            raise DiagnosticError(f"--arm expects NAME=DIR, got {spec!r}")
        name, directory = spec.split("=", 1)
        frame = _load_epoch(Path(directory), args.epoch)
        test = count_test_rows(frame)
        if test:
            raise DiagnosticError(
                f"{name}: {test} scored rows are in the test evaluation "
                "test manifest. This analysis is restricted to validation rows."
            )
        block = calibration_report(
            frame["label"].to_numpy().astype(int),
            frame["score"].to_numpy(),
            frame["cohort"].to_numpy(dtype=object),
        )
        block["source_directory"] = str(Path(directory).resolve())
        block["test_rows_selected"] = int(test)
        block["is_confirmatory"] = name == args.confirmatory
        report["arms"][name] = block

    if args.confirmatory not in report["arms"]:
        raise DiagnosticError(
            f"confirmatory arm {args.confirmatory!r} was not among the arms given"
        )

    print(
        f"{'arm':<46} {'ROC AUC':>8} {'PR AUC':>8} {'Brier':>9} "
        f"{'R@1e-3':>8} {'P@1e-3':>8} {'R@1e-4':>8}"
    )
    for name, block in report["arms"].items():
        p3 = block["operating_points"]["fpr_1e-03"]
        p4 = block["operating_points"]["fpr_1e-04"]
        mark = "*" if block["is_confirmatory"] else " "
        r4 = "undefined" if p4["recall"] is None else f"{p4['recall']:8.4f}"
        print(
            f"{mark}{name:<45} {block['roc_auc']:8.4f} "
            f"{block['pr_auc_average_precision']:8.4f} {block['brier_score']:9.6f} "
            f"{p3['recall']:8.4f} {p3['precision']:8.4f} {r4:>8}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
