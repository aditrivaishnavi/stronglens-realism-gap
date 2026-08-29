"""The operating point is s_(k+1), not s_(k), and the difference is not cosmetic.

§14 of the plan fixes the threshold prospectively: for allowance k, sorting the
validation negatives descending, the threshold is the smallest representable
value strictly greater than s_(k+1), classifying by ``score >= t``.

The first implementation used s_(k) and still produced the correct realised
false-positive count, which is why it survived review. Both rules admit exactly
k negatives, so nothing about the validation set distinguishes them. What differs
is where the boundary sits in the open gap between s_(k+1) and s_(k), and every
object that has not been seen yet is classified by that boundary. At the 1e-3
rung the gap is 0.019 in logit units, which is wide compared with the spacing of
injection scores nearby, so the choice moves real counts.

The rule was frozen before any of these outcomes existed. That is the whole
reason a prospective rule is worth having, so these tests pin it to the plan's
text rather than to whichever value happens to look defensible now.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dhs.fixed_width_scoring import operating_points

# Every rung the plan reads, plus neighbours that reveal ties.
ALLOWANCES = (1, 2, 3, 4, 5, 6, 7, 8, 12, 30, 60, 61, 62, 123)

LOGITS = Path(
    "results/unified_scoring_contract_20260828/artifacts/validation_logits.parquet"
)


def rungs_for(scores: np.ndarray, allowances=ALLOWANCES) -> dict[int, dict]:
    """Index the rungs by allowance, which is how every test below reads them."""
    return {r["allowance"]: r for r in operating_points(scores, allowances)}


@pytest.fixture(scope="module")
def negatives() -> np.ndarray:
    if not LOGITS.exists():  # pragma: no cover
        pytest.skip("the retained validation logits are absent")
    frame = pd.read_parquet(LOGITS, columns=["label", "logit"])
    return frame.loc[frame["label"] == 0, "logit"].to_numpy()


def test_the_threshold_sits_strictly_above_the_k_plus_first_not_at_the_kth():
    """The rule, on scores contrived so every gap is unambiguous."""
    # Descending: 10, 9, 8, ... so s_(1)=10, s_(2)=9, s_(3)=8.
    scores = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0], dtype=np.float64)
    rungs = rungs_for(scores)

    rung = rungs[2]
    assert rung["s_k"] == 9.0
    assert rung["s_k_plus_1"] == 8.0

    # Strictly above s_(3)=8, and nowhere near s_(2)=9.
    assert rung["logit_threshold"] > 8.0
    assert rung["logit_threshold"] < 9.0
    assert rung["negatives_at_or_above"] == 2
    assert rung["identified"] is True


def test_the_two_rules_disagree_on_an_object_between_the_order_statistics():
    """Why this is blocking rather than presentational.

    An object landing in the gap is accepted by the plan's rule and rejected by
    the superseded one. On the validation negatives the two are indistinguishable,
    so only an unseen object exposes the difference, which is precisely the
    population the threshold exists to classify.
    """
    scores = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0], dtype=np.float64)
    rungs = rungs_for(scores)
    correct = rungs[2]["logit_threshold"]
    superseded = rungs[2]["s_k"]

    candidate = 8.5
    assert candidate >= correct, "the plan's rule accepts it"
    assert not candidate >= superseded, "the superseded rule rejects it"

    # And both rules still admit the same number of validation negatives, which
    # is the reason the defect was invisible.
    assert int((scores >= correct).sum()) == int((scores >= superseded).sum()) == 2


def test_nextafter_is_evaluated_in_float32_so_the_threshold_is_representable():
    """A float64 step would land in a gap no float32 score can occupy.

    The model emits float32. Stepping in float64 produces a threshold that no
    achievable score can equal, which converts the plan's ``>=`` into a strict
    ``>`` for one specific value. Same count here, wrong rule, and it would
    diverge the moment a score landed exactly on the boundary.
    """
    scores = np.array([3.0, 2.0, 1.5, 1.0], dtype=np.float64)
    rung = rungs_for(scores)[2]

    threshold = np.float32(rung["logit_threshold"])
    s_next = np.float32(rung["s_k_plus_1"])

    # Exactly one float32 step above s_(k+1): nothing representable in between.
    assert threshold == np.nextafter(s_next, np.float32(np.inf), dtype=np.float32)
    assert float(threshold) == rung["logit_threshold"]
    assert rung["logit_threshold_float32_bits"] == int(
        np.float32(threshold).view(np.uint32)
    )

    # A float64 step would be smaller than a float32 step, hence not representable.
    float64_step = np.nextafter(np.float64(s_next), np.inf)
    assert float64_step < float(threshold)
    assert np.float32(float64_step) == s_next


def test_ties_resolve_above_the_whole_group_and_the_count_falls_below_k():
    """The plan's stated tie behaviour, which is conservative by construction.

    With s_(k) = s_(k+1) the threshold clears both, so the realised count comes
    in under the allowance rather than over it. Reporting the looser adjacent
    threshold alongside is what the plan requires in that situation, so it must
    be present on every rung and not only when a tie happens to occur.
    """
    scores = np.array([10.0, 7.0, 7.0, 7.0, 5.0], dtype=np.float64)
    rung = rungs_for(scores)[2]

    assert rung["s_k"] == rung["s_k_plus_1"] == 7.0
    assert rung["tied_at_s_k_plus_1"] == 3
    assert rung["logit_threshold"] > 7.0

    # Under the allowance, never over it.
    assert rung["negatives_at_or_above"] == 1
    assert rung["negatives_at_or_above"] < rung["allowance"]
    assert rung["identified"] is False

    # The looser threshold admits the whole tied group, and is reported.
    assert rung["loose_negatives_at_or_above"] == 4
    assert rung["loose_realised_fpr"] > rung["realised_fpr"]


def test_every_rung_carries_the_adjacent_order_statistic_sensitivity(negatives):
    rungs = operating_points(negatives, ALLOWANCES)
    assert rungs
    for rung in rungs:
        assert rung["loose_negatives_at_or_above"] == rung["allowance"] + 1
        assert rung["loose_realised_fpr"] > rung["realised_fpr"]
        assert rung["s_k"] > rung["s_k_plus_1"]


def test_the_governing_thresholds_on_the_retained_logits(negatives):
    """Regression on the two rungs the plan reads, computed from the artifact.

    These are the values Amendment 11 freezes. Pinning them here means a future
    change to the ranking code cannot move the operating point silently.
    """
    assert negatives.size == 61437
    rungs = rungs_for(negatives)

    primary = rungs[61]
    assert primary["s_k"] == pytest.approx(1.352764368057251, abs=1e-12)
    assert primary["s_k_plus_1"] == pytest.approx(1.3337491750717163, abs=1e-12)
    assert primary["logit_threshold"] == pytest.approx(1.3337492942810059, abs=1e-12)
    assert primary["probability_threshold_float64"] == pytest.approx(
        0.7914601357633129, abs=1e-15
    )
    assert primary["negatives_at_or_above"] == 61
    assert primary["identified"] is True

    secondary = rungs[6]
    assert secondary["s_k"] == pytest.approx(17.69274139404297, abs=1e-12)
    assert secondary["s_k_plus_1"] == pytest.approx(17.135696411132812, abs=1e-12)
    assert secondary["logit_threshold"] == pytest.approx(17.135698318481445, abs=1e-12)
    assert secondary["probability_threshold_float64"] == pytest.approx(
        0.999999963853957, abs=1e-15
    )
    assert secondary["negatives_at_or_above"] == 6
    assert secondary["identified"] is True


def test_threshold_is_placed_above_s_k_plus_1(negatives):
    """The boundary must not be placed directly on s_(k)."""
    rungs = rungs_for(negatives)

    assert rungs[61]["logit_threshold"] != rungs[61]["s_k"]
    assert rungs[6]["logit_threshold"] != rungs[6]["s_k"]


def test_a_rung_is_skipped_when_there_is_no_k_plus_first_negative():
    """The rule needs one more negative than the allowance, so the edge is real."""
    scores = np.array([3.0, 2.0], dtype=np.float64)
    allowances = set(rungs_for(scores, (1, 2, 3)))

    assert 1 in allowances
    assert 2 not in allowances, "s_(3) does not exist, so the rung cannot be set"


def test_non_finite_negative_logits_are_rejected_not_sorted():
    """A NaN sorts arbitrarily and compares false, so it must not enter the rule."""
    from dhs.fixed_width_scoring import ScoringError

    with pytest.raises(ScoringError, match="non-finite"):
        operating_points(np.array([np.nan, 2.0, 1.0, -1.0]), [1])
    with pytest.raises(ScoringError, match="non-finite"):
        operating_points(np.array([3.0, np.inf, 1.0]), [1])
    with pytest.raises(ScoringError, match="one-dimensional"):
        operating_points(np.array([[3.0, 2.0], [1.0, 0.0]]), [1])
