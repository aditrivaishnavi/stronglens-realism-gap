#!/usr/bin/env python
"""Stratified and paired diagnostics from re-scored validation scores.

Consumes the per-object score files written by ``scripts/score_validation_logits.py``
and needs no GPU, no corpus and no checkpoint, so the statistics can be reviewed
and re-run independently of the machine that produced the scores.

Diagnostic 5, per arm and per grid epoch: validation ROC AUC pooled and stratified
by positive provenance, against the same negatives throughout; partial ROC AUC
below FPR 10**-3; TPR at false-positive counts 6, 30, 61 and 123; and tie and
saturation counts at the top of the negative ranking.

Diagnostic 4, per epoch across the two mixtures: a paired per-object comparison
restricted to the positives and negatives the two arms share, by exact McNemar on
detections at each arm's own matched operating point, and by paired score
differences with a bootstrap interval.

Usage:

    python scripts/run_validation_diagnostics.py \\
        --arm tier_a_candidates/seed_42=<scores dir> \\
        --arm tier_ab_historical/seed_42=<scores dir> \\
        --out runs/<run>/analysis/validation_diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dhs.validation_diagnostics import (  # noqa: E402
    DiagnosticError,
    paired_detection_comparison,
    paired_score_comparison,
    stratified_ranking_report,
)

# The cohort whose membership is identical across mixtures, and therefore the one
# the paired comparison is restricted to.
SHARED_POSITIVE_COHORT = "candidate_holdout"


def _load_arm(directory: Path) -> dict[int, pd.DataFrame]:
    frames = {}
    for path in sorted(directory.glob("epoch_*.parquet")):
        df = pd.read_parquet(path)
        frames[int(df["epoch"].iloc[0])] = df
    if not frames:
        raise DiagnosticError(f"no epoch_*.parquet under {directory}")
    return frames


def _paired(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    restrict_positives_to_shared_cohort: bool = False,
) -> dict:
    """Restrict two arms to their shared population and compare per object.

    The restriction is checked rather than assumed: the shared positives must be
    exactly the intersection of the two positive cohorts and the negatives must be
    set-equal, which is what makes the comparison free of construction choices. If
    either fails, the arms do not define a paired comparison and the diagnostic
    refuses.

    ``restrict_positives_to_shared_cohort`` narrows both arms to
    ``SHARED_POSITIVE_COHORT`` before those checks. It exists for same-mixture
    pairs of the wide mixture, where both arms carry the 714 unreviewed positives
    as well as the shared 64: the intersection is then 778, the declared cohort is
    64, and the check above refuses. That refusal is correct — it is what stops an
    unexamined population entering a paired test — so the flag is opt-in.
    Restricting makes both mixtures use the same 64 shared objects.
    """
    if restrict_positives_to_shared_cohort:
        a = a[(a.label == 0) | (a.cohort == SHARED_POSITIVE_COHORT)]
        b = b[(b.label == 0) | (b.cohort == SHARED_POSITIVE_COHORT)]

    pos_a = set(a.loc[a.label == 1, "galaxy_id"])
    pos_b = set(b.loc[b.label == 1, "galaxy_id"])
    neg_a = set(a.loc[a.label == 0, "galaxy_id"])
    neg_b = set(b.loc[b.label == 0, "galaxy_id"])

    if neg_a != neg_b:
        raise DiagnosticError(
            f"validation negatives are not set-equal ({len(neg_a)} vs {len(neg_b)}, "
            f"{len(neg_a ^ neg_b)} differing). The paired comparison depends on "
            f"their identity; without it the two arms face different false-positive budgets "
            f"and the pairing is not defined."
        )
    shared_pos = pos_a & pos_b
    cohort_pos = set(a.loc[(a.label == 1) & (a.cohort == SHARED_POSITIVE_COHORT), "galaxy_id"])
    if shared_pos != cohort_pos:
        raise DiagnosticError(
            f"the shared positives ({len(shared_pos)}) are not exactly the "
            f"{SHARED_POSITIVE_COHORT} cohort ({len(cohort_pos)}). The declared "
            f"population is the complete intersection of the two cohorts."
        )

    keep = shared_pos | neg_a
    aa = a[a.galaxy_id.isin(keep)].set_index("galaxy_id").sort_index()
    bb = b[b.galaxy_id.isin(keep)].set_index("galaxy_id").sort_index()
    if not aa.index.equals(bb.index):
        raise DiagnosticError("shared rows failed to align after restriction")
    if not (aa.label.to_numpy() == bb.label.to_numpy()).all():
        raise DiagnosticError("the same object carries different labels in the two arms")

    labels = aa.label.to_numpy().astype(int)
    out = {
        "shared_positives": int(len(shared_pos)),
        "shared_negatives": int(len(neg_a)),
        "negatives_set_equal": True,
        "positives_restricted_to_shared_cohort": bool(
            restrict_positives_to_shared_cohort
        ),
        "detection": {},
        "score": paired_score_comparison(labels, aa.score.to_numpy(), bb.score.to_numpy()),
    }
    for fpr in (1e-3, 1e-4):
        out["detection"][f"fpr_{fpr:.0e}"] = paired_detection_comparison(
            labels, aa.score.to_numpy(), bb.score.to_numpy(), fpr
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="arm label and the directory of its re-scored validation scores",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--restrict-positives-to-shared-cohort",
        action="store_true",
        help=f"narrow every arm's positives to the {SHARED_POSITIVE_COHORT} cohort "
        "before pairing for same-mixture comparisons",
    )
    args = ap.parse_args()

    arms: dict[str, dict[int, pd.DataFrame]] = {}
    for spec in args.arm:
        if "=" not in spec:
            raise DiagnosticError(f"--arm expects NAME=DIR, got {spec!r}")
        name, directory = spec.split("=", 1)
        arms[name] = _load_arm(Path(directory))

    report: dict = {
        "scope": "validation-only stratified and paired diagnostics",
        "test_facing": False,
        "split": "val",
        "diagnostic_5_stratified_ranking": {},
        "diagnostic_4_paired_common_population": {},
        "paired_positives_restricted_to_shared_cohort": bool(
            args.restrict_positives_to_shared_cohort
        ),
    }

    for name, frames in arms.items():
        per_epoch = {}
        for epoch in sorted(frames):
            df = frames[epoch]
            per_epoch[str(epoch)] = stratified_ranking_report(
                df.label.to_numpy().astype(int),
                df.score.to_numpy(),
                df.cohort.to_numpy(dtype=object),
            )
        report["diagnostic_5_stratified_ranking"][name] = per_epoch
        pooled = {e: r["pooled"]["roc_auc"] for e, r in per_epoch.items()}
        cohorts = sorted(
            {c for r in per_epoch.values() for c in r["by_positive_cohort"]}
        )
        print(f"\n=== {name} ===")
        header = "  epoch   pooled" + "".join(f"  {c[:18]:>18}" for c in cohorts)
        print(header)
        for e in sorted(pooled, key=int):
            row = f"  {int(e):5d}  {pooled[e]:.4f}"
            for c in cohorts:
                by = per_epoch[e]["by_positive_cohort"].get(c)
                row += f"  {by['roc_auc']:18.4f}" if by else f"  {'-':>18}"
            print(row)

    names = list(arms)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            shared_epochs = sorted(set(arms[a]) & set(arms[b]))
            if not shared_epochs:
                continue
            pair = {}
            for epoch in shared_epochs:
                pair[str(epoch)] = _paired(
                    arms[a][epoch],
                    arms[b][epoch],
                    restrict_positives_to_shared_cohort=(
                        args.restrict_positives_to_shared_cohort
                    ),
                )
            report["diagnostic_4_paired_common_population"][f"{a}__vs__{b}"] = pair
            print(f"\n=== paired: {a} vs {b} ===")
            print("  epoch  recall_a  recall_b  onlyA  onlyB  mcnemar_p  mean_dscore")
            for e in sorted(pair, key=int):
                d = pair[e]["detection"]["fpr_1e-03"]
                s = pair[e]["score"]
                print(
                    f"  {int(e):5d}  {d['recall_a']:8.4f}  {d['recall_b']:8.4f}  "
                    f"{d['only_a']:5d}  {d['only_b']:5d}  {d['mcnemar']['p_value']:9.4f}  "
                    f"{s['mean_difference']:+.4f}"
                )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
