"""Contract tests for the Amendment 5 diagnostic drivers.

These run the drivers end to end on synthetic score files, so that the plumbing
-- argument parsing, restriction to the shared population, report shape -- is
known good before either driver is pointed at a real fit. The scoring script's
refusal to read a test partition is checked here too, because that guard is the
one whose failure would matter most and it needs no GPU to exercise.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
ANALYSE = REPO / "scripts" / "run_validation_diagnostics.py"

N_NEG = 2000
N_SHARED_POS = 64
N_EXTRA_POS = 200


def _write_arm(directory: Path, epochs, with_extra_positives, seed=0, shift=0.0):
    """Write score files shaped like the real ones.

    ``with_extra_positives`` distinguishes the wide mixture, which carries an
    unreviewed cohort on top of the shared candidates, from the thin one, which
    carries only the candidates. Negative identifiers are shared by construction,
    mirroring the set-equality the real manifests have.
    """
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    neg_ids = [f"neg_{i:05d}" for i in range(N_NEG)]
    pos_ids = [f"cand_{i:03d}" for i in range(N_SHARED_POS)]
    extra_ids = [f"tierb_{i:03d}" for i in range(N_EXTRA_POS)]

    for epoch in epochs:
        ids = list(neg_ids) + list(pos_ids)
        labels = [0] * N_NEG + [1] * N_SHARED_POS
        cohorts = ["not_applicable"] * N_NEG + ["candidate_holdout"] * N_SHARED_POS
        tiers = ["N"] * N_NEG + ["A"] * N_SHARED_POS
        scores = list(rng.uniform(0, 0.6, N_NEG)) + list(
            np.clip(rng.uniform(0.7, 1.0, N_SHARED_POS) + shift, 0, 1)
        )
        if with_extra_positives:
            ids += extra_ids
            labels += [1] * N_EXTRA_POS
            cohorts += ["tier_b_unreviewed_holdout"] * N_EXTRA_POS
            tiers += ["B"] * N_EXTRA_POS
            # Drawn from the negative range, standing in for label noise.
            scores += list(rng.uniform(0, 0.6, N_EXTRA_POS))
        pd.DataFrame(
            {
                "galaxy_id": ids,
                "label": labels,
                "cohort": cohorts,
                "tier": tiers,
                "score": scores,
                "epoch": epoch,
            }
        ).to_parquet(directory / f"epoch_{epoch:03d}.parquet", index=False)


def _run(args):
    return subprocess.run(
        [sys.executable, *args], capture_output=True, text=True, cwd=REPO
    )


def test_the_driver_reports_both_diagnostics_over_the_grid(tmp_path):
    thin, wide = tmp_path / "thin", tmp_path / "wide"
    _write_arm(thin, (19, 110), with_extra_positives=False, seed=1)
    _write_arm(wide, (19, 110), with_extra_positives=True, seed=2)
    out = tmp_path / "report.json"

    proc = _run(
        [
            str(ANALYSE),
            "--arm", f"tier_a_candidates/seed_42={thin}",
            "--arm", f"tier_ab_historical/seed_42={wide}",
            "--out", str(out),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(out.read_text())
    assert report["test_facing"] is False
    assert report["split"] == "val"

    strat = report["diagnostic_5_stratified_ranking"]
    assert set(strat) == {"tier_a_candidates/seed_42", "tier_ab_historical/seed_42"}
    wide_19 = strat["tier_ab_historical/seed_42"]["19"]
    assert set(wide_19["by_positive_cohort"]) == {
        "candidate_holdout",
        "tier_b_unreviewed_holdout",
    }
    # The synthetic noisy cohort must read as near-chance and the clean one as
    # near-perfect: the separation the diagnostic exists to detect.
    assert wide_19["by_positive_cohort"]["candidate_holdout"]["roc_auc"] > 0.99
    assert wide_19["by_positive_cohort"]["tier_b_unreviewed_holdout"]["roc_auc"] < 0.6

    paired = report["diagnostic_4_paired_common_population"]
    key = "tier_a_candidates/seed_42__vs__tier_ab_historical/seed_42"
    assert set(paired[key]) == {"19", "110"}
    entry = paired[key]["19"]
    assert entry["shared_positives"] == N_SHARED_POS
    assert entry["shared_negatives"] == N_NEG
    assert entry["negatives_set_equal"] is True
    assert "mcnemar" in entry["detection"]["fpr_1e-03"]


def test_the_paired_comparison_uses_only_the_shared_population(tmp_path):
    """The wide arm's extra positives must not enter the paired comparison."""
    thin, wide = tmp_path / "thin", tmp_path / "wide"
    _write_arm(thin, (19,), with_extra_positives=False, seed=3)
    _write_arm(wide, (19,), with_extra_positives=True, seed=4)
    out = tmp_path / "report.json"
    assert _run(
        [str(ANALYSE), "--arm", f"a={thin}", "--arm", f"b={wide}", "--out", str(out)]
    ).returncode == 0
    entry = json.loads(out.read_text())["diagnostic_4_paired_common_population"]["a__vs__b"]["19"]
    # 64, not 264: the pairing is on the intersection, not the union.
    assert entry["shared_positives"] == N_SHARED_POS
    d = entry["detection"]["fpr_1e-03"]
    assert d["both"] + d["only_a"] + d["only_b"] + d["neither"] == N_SHARED_POS


def test_a_shared_shift_is_visible_as_a_paired_difference(tmp_path):
    """Two arms differing by a known positive-side shift must be ordered correctly."""
    lo, hi = tmp_path / "lo", tmp_path / "hi"
    _write_arm(lo, (19,), with_extra_positives=False, seed=5, shift=0.0)
    _write_arm(hi, (19,), with_extra_positives=False, seed=5, shift=0.05)
    out = tmp_path / "report.json"
    assert _run(
        [str(ANALYSE), "--arm", f"hi={hi}", "--arm", f"lo={lo}", "--out", str(out)]
    ).returncode == 0
    entry = json.loads(out.read_text())["diagnostic_4_paired_common_population"]["hi__vs__lo"]["19"]
    assert entry["score"]["mean_difference"] > 0
    assert entry["score"]["b_higher_on"] == 0


def test_mismatched_negatives_are_refused(tmp_path):
    """Set-equality of the negatives is the premise; it must be checked, not assumed."""
    a, b = tmp_path / "a", tmp_path / "b"
    _write_arm(a, (19,), with_extra_positives=False, seed=6)
    _write_arm(b, (19,), with_extra_positives=False, seed=7)
    frame = pd.read_parquet(b / "epoch_019.parquet")
    frame.loc[frame.label == 0, "galaxy_id"] = [
        f"other_{i:05d}" for i in range(int((frame.label == 0).sum()))
    ]
    frame.to_parquet(b / "epoch_019.parquet", index=False)

    proc = _run(
        [str(ANALYSE), "--arm", f"a={a}", "--arm", f"b={b}", "--out", str(tmp_path / "r.json")]
    )
    assert proc.returncode != 0
    assert "not set-equal" in (proc.stderr + proc.stdout)


def test_a_positive_cohort_that_is_not_the_intersection_is_refused(tmp_path):
    """Guards against a manifest that is not the one Amendment 5 describes."""
    a, b = tmp_path / "a", tmp_path / "b"
    _write_arm(a, (19,), with_extra_positives=False, seed=8)
    _write_arm(b, (19,), with_extra_positives=False, seed=9)
    frame = pd.read_parquet(a / "epoch_019.parquet")
    # Relabel one shared candidate as an unreviewed row, so the intersection and
    # the declared cohort no longer coincide.
    idx = frame.index[frame.cohort == "candidate_holdout"][0]
    frame.loc[idx, "cohort"] = "tier_b_unreviewed_holdout"
    frame.to_parquet(a / "epoch_019.parquet", index=False)

    proc = _run(
        [str(ANALYSE), "--arm", f"a={a}", "--arm", f"b={b}", "--out", str(tmp_path / "r.json")]
    )
    assert proc.returncode != 0
    assert "complete intersection" in (proc.stderr + proc.stdout)


def test_a_same_mixture_wide_pair_is_refused_by_default(tmp_path):
    """Two wide arms share all 264 positives, so the declared cohort check fires.

    This is the situation the gen7 recipe-parity and imbalance arms are in on the
    wide mixture. The refusal is the behaviour worth pinning: without it a paired
    test would silently run over the unreviewed cohort.
    """
    a, b = tmp_path / "a", tmp_path / "b"
    _write_arm(a, (19,), with_extra_positives=True, seed=10)
    _write_arm(b, (19,), with_extra_positives=True, seed=11)

    proc = _run(
        [str(ANALYSE), "--arm", f"a={a}", "--arm", f"b={b}", "--out", str(tmp_path / "r.json")]
    )
    assert proc.returncode != 0
    assert "complete intersection" in (proc.stderr + proc.stdout)


def test_restricting_to_the_shared_cohort_makes_a_wide_pair_comparable(tmp_path):
    """The pre-registered restriction, and the audit flag that records it."""
    a, b = tmp_path / "a", tmp_path / "b"
    _write_arm(a, (19,), with_extra_positives=True, seed=12)
    _write_arm(b, (19,), with_extra_positives=True, seed=13)
    out = tmp_path / "r.json"

    proc = _run(
        [
            str(ANALYSE),
            "--arm", f"a={a}",
            "--arm", f"b={b}",
            "--out", str(out),
            "--restrict-positives-to-shared-cohort",
        ]
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(out.read_text())
    assert report["paired_positives_restricted_to_shared_cohort"] is True

    entry = report["diagnostic_4_paired_common_population"]["a__vs__b"]["19"]
    # The 200 unreviewed positives are excluded; the shared 64 remain.
    assert entry["shared_positives"] == N_SHARED_POS
    assert entry["shared_negatives"] == N_NEG
    assert entry["positives_restricted_to_shared_cohort"] is True
    d = entry["detection"]["fpr_1e-03"]
    assert d["both"] + d["only_a"] + d["only_b"] + d["neither"] == N_SHARED_POS

    # Diagnostic 5 is per arm and must be unaffected: it still sees both cohorts,
    # because the restriction is declared only for the paired comparison.
    strat = report["diagnostic_5_stratified_ranking"]["a"]["19"]
    assert set(strat["by_positive_cohort"]) == {
        "candidate_holdout",
        "tier_b_unreviewed_holdout",
    }


def test_the_restriction_flag_is_recorded_as_false_when_unused(tmp_path):
    thin, wide = tmp_path / "thin", tmp_path / "wide"
    _write_arm(thin, (19,), with_extra_positives=False, seed=14)
    _write_arm(wide, (19,), with_extra_positives=True, seed=15)
    out = tmp_path / "r.json"
    assert _run(
        [str(ANALYSE), "--arm", f"a={thin}", "--arm", f"b={wide}", "--out", str(out)]
    ).returncode == 0
    report = json.loads(out.read_text())
    assert report["paired_positives_restricted_to_shared_cohort"] is False
    entry = report["diagnostic_4_paired_common_population"]["a__vs__b"]["19"]
    assert entry["positives_restricted_to_shared_cohort"] is False


def test_a_malformed_arm_specification_is_refused(tmp_path):
    proc = _run([str(ANALYSE), "--arm", "no-equals-sign", "--out", str(tmp_path / "r.json")])
    assert proc.returncode != 0
    assert "NAME=DIR" in (proc.stderr + proc.stdout)


def test_the_scoring_script_refuses_a_manifest_carrying_test_rows(tmp_path):
    """The validation-only tool rejects rows from the separate test manifest."""
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame(
        {
            "split": ["train", "val", "test"],
            "galaxy_id": ["a", "b", "c"],
            "label": [0, 1, 1],
        }
    ).to_parquet(manifest, index=False)

    sys.path.insert(0, str(REPO))
    from scripts.score_validation_logits import ScoringError, refuse_any_test_partition

    with pytest.raises(ScoringError) as err:
        refuse_any_test_partition(str(manifest))
    assert "model-development rows" in str(err.value)

    # A development manifest with no test partition must pass unobstructed.
    ok = tmp_path / "ok.parquet"
    pd.DataFrame({"split": ["train", "val"], "galaxy_id": ["a", "b"], "label": [0, 1]}).to_parquet(
        ok, index=False
    )
    refuse_any_test_partition(str(ok))
