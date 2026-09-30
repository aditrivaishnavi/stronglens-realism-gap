from __future__ import annotations

import numpy as np
import pytest

from dhs.matching import (
    MatchingError,
    match_hosts,
    rematching_draws,
    standardised_mean_difference,
)


def test_optimal_matching_assigns_unique_nearby_hosts() -> None:
    observed = np.array([[0.0], [10.0]])
    candidates = np.array([[0.1], [9.9], [20.0], [-20.0]])
    result = match_hosts(
        observed,
        candidates,
        columns=("brightness",),
        hosts_per_lens=1,
        seed=42,
    )
    selected = [matched.host_indices[0] for matched in result.matched_sets]
    assert selected == [0, 1]
    assert len(set(selected)) == 2


def test_standardised_mean_difference_is_zero_for_equal_samples() -> None:
    values = np.array([1.0, 2.0, 3.0])
    assert standardised_mean_difference(values, values) == 0.0


def test_zero_covariate_matching_respects_forbidden_pairs() -> None:
    observed = np.empty((2, 0))
    candidates = np.empty((6, 0))
    forbidden = np.zeros((2, 6), dtype=bool)
    forbidden[0, :3] = True
    forbidden[1, 3:] = True

    result = match_hosts(
        observed,
        candidates,
        columns=(),
        hosts_per_lens=2,
        seed=42,
        forbidden=forbidden,
    )

    for matched in result.matched_sets:
        assert all(not forbidden[matched.real_index, host] for host in matched.host_indices)


def test_zero_covariate_matching_rejects_no_admissible_assignment() -> None:
    with pytest.raises(MatchingError, match="no admissible candidate"):
        match_hosts(
            np.empty((1, 0)),
            np.empty((3, 0)),
            columns=(),
            hosts_per_lens=1,
            seed=42,
            forbidden=np.ones((1, 3), dtype=bool),
        )


def test_with_replacement_matching_never_selects_infinite_cost() -> None:
    observed = np.array([[0.0]])
    candidates = np.array([[0.0], [1.0], [2.0]])
    forbidden = np.array([[True, False, False]])

    result = match_hosts(
        observed,
        candidates,
        columns=("x",),
        hosts_per_lens=2,
        seed=42,
        with_replacement=True,
        forbidden=forbidden,
    )

    assert result.matched_sets[0].host_indices == (1, 2)
    assert np.isfinite(result.matched_sets[0].distances).all()


def test_rematching_subsets_forbidden_mask_with_candidate_pool() -> None:
    observed = np.array([[0.0], [5.0]])
    candidates = np.arange(20, dtype=float)[:, None]
    forbidden = np.zeros((2, 20), dtype=bool)
    forbidden[0, ::2] = True
    forbidden[1, 1::2] = True

    results = rematching_draws(
        observed,
        candidates,
        columns=("x",),
        hosts_per_lens=1,
        base_seed=42,
        draws=3,
        forbidden=forbidden,
    )

    for result in results:
        for matched in result.matched_sets:
            assert all(
                not forbidden[matched.real_index, host]
                for host in matched.host_indices
            )
