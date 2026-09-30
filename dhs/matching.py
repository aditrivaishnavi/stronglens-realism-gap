"""Host matching: Mahalanobis distance and optimal assignment.

Design 8.3 replaces two v1 choices:

* **Optimal assignment** (``scipy.optimize.linear_sum_assignment``) rather than
  a greedy sweep with a refinement pass. Greedy matching is order-dependent, so
  the matched set depends on the order rows happened to arrive in, and it is
  dominated by the optimal solution on the same cost matrix. There is no
  tuning-parameter argument for greedy; it is simply worse.
* **Mahalanobis distance** rather than Euclidean distance on independently
  standardised columns. The ``g``, ``r``, ``z`` fluxes are strongly correlated,
  so Euclidean-on-standardised counts the same photometric information three
  times and under-weights everything else.

The covariance used by the Mahalanobis metric is estimated on the **pooled**
real-plus-candidate sample. Estimating it on the reals alone would let a
111-row covariance drive a 40,000-row assignment; estimating it separately per
arm would make the distance asymmetric, so d(a, b) would depend on which arm you
started from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


class MatchingError(RuntimeError):
    """Raised when a matched set cannot be constructed as specified."""


def standardised_mean_difference(
    observed: np.ndarray,
    matched: np.ndarray,
) -> float:
    """Difference in means divided by the pooled sample standard deviation."""
    observed = np.asarray(observed, dtype=float)
    matched = np.asarray(matched, dtype=float)
    pooled = np.sqrt(
        (np.var(observed, ddof=1) + np.var(matched, ddof=1)) / 2.0
    )
    if pooled == 0:
        return 0.0 if np.mean(observed) == np.mean(matched) else float("inf")
    return float((np.mean(matched) - np.mean(observed)) / pooled)


@dataclass(frozen=True)
class MatchedSet:
    """One real lens and the pool rows assigned to it.

    ``distances`` are in the same metric as the assignment cost, so they can be
    compared against the pairwise caliper without recomputation.
    """

    real_index: int
    host_indices: tuple[int, ...]
    distances: tuple[float, ...]

    @property
    def worst_distance(self) -> float:
        return max(self.distances) if self.distances else 0.0


@dataclass(frozen=True)
class MatchingResult:
    matched_sets: tuple[MatchedSet, ...]
    columns: tuple[str, ...]
    hosts_per_lens: int
    with_replacement: bool
    seed: int
    covariance_condition_number: float
    metric: str
    assignment: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def real_indices(self) -> np.ndarray:
        return np.array([s.real_index for s in self.matched_sets], dtype=int)

    @property
    def host_indices(self) -> np.ndarray:
        """Flat array of pool indices, grouped by real lens in order."""
        return np.array(
            [h for s in self.matched_sets for h in s.host_indices], dtype=int
        )

    @property
    def group_labels(self) -> np.ndarray:
        """Cluster id per host row, for grouped CV and the cluster bootstrap.

        Every host carries the id of the real lens it was matched to, so that a
        matched set is never split across cross-validation folds and is
        resampled as a unit by the bootstrap.
        """
        return np.array(
            [s.real_index for s in self.matched_sets for _ in s.host_indices],
            dtype=int,
        )

    @property
    def pairwise_distances(self) -> np.ndarray:
        return np.array(
            [d for s in self.matched_sets for d in s.distances], dtype=float
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "hosts_per_lens": self.hosts_per_lens,
            "with_replacement": self.with_replacement,
            "seed": self.seed,
            "metric": self.metric,
            "assignment": self.assignment,
            "covariance_condition_number": self.covariance_condition_number,
            "n_matched_sets": len(self.matched_sets),
            "worst_pair_distance": float(self.pairwise_distances.max())
            if self.matched_sets
            else 0.0,
            "diagnostics": self.diagnostics,
        }


def pooled_covariance(
    real: np.ndarray, candidates: np.ndarray, *, ridge: float = 1e-8
) -> tuple[np.ndarray, float]:
    """Covariance of the pooled sample, and its condition number.

    A small ridge is added to the diagonal so that a covariate which is constant
    within the pooled sample (a degenerate but legal situation, e.g. a cell
    matching a variable that happens not to vary) does not make the matrix
    singular. The ridge is reported through the condition number rather than
    being tuned to make a particular run succeed.
    """
    pooled = np.vstack([real, candidates])
    if pooled.shape[1] == 0:
        return np.zeros((0, 0)), 1.0
    covariance = np.cov(pooled, rowvar=False)
    covariance = np.atleast_2d(covariance)
    scale = float(np.trace(covariance)) / max(covariance.shape[0], 1)
    covariance = covariance + np.eye(covariance.shape[0]) * ridge * max(scale, 1.0)
    condition = float(np.linalg.cond(covariance))
    return covariance, condition


def mahalanobis_cost(
    real: np.ndarray, candidates: np.ndarray, covariance: np.ndarray
) -> np.ndarray:
    """Mahalanobis distance from every real row to every candidate row.

    Implemented by whitening both arms with the inverse Cholesky factor and then
    taking Euclidean distances, which is numerically better behaved than forming
    the quadratic form directly and is what makes the 444 x 40,000 cost matrix
    tractable.
    """
    if real.shape[1] == 0:
        return np.zeros((real.shape[0], candidates.shape[0]))
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:  # pragma: no cover - guarded by ridge
        raise MatchingError(f"covariance is not positive definite: {error}") from error
    whitened_real = np.linalg.solve(cholesky, real.T).T
    whitened_candidates = np.linalg.solve(cholesky, candidates.T).T
    squared = (
        (whitened_real**2).sum(axis=1)[:, None]
        + (whitened_candidates**2).sum(axis=1)[None, :]
        - 2.0 * whitened_real @ whitened_candidates.T
    )
    return np.sqrt(np.maximum(squared, 0.0))


def euclidean_cost(real: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Euclidean distance on independently standardised columns."""
    if real.shape[1] == 0:
        return np.zeros((real.shape[0], candidates.shape[0]))
    pooled = np.vstack([real, candidates])
    centre = pooled.mean(axis=0)
    spread = pooled.std(axis=0)
    spread = np.where(spread > 0, spread, 1.0)
    a = (real - centre) / spread
    b = (candidates - centre) / spread
    squared = (a**2).sum(axis=1)[:, None] + (b**2).sum(axis=1)[None, :] - 2.0 * a @ b.T
    return np.sqrt(np.maximum(squared, 0.0))


def match_hosts(
    real: np.ndarray,
    candidates: np.ndarray,
    *,
    columns: tuple[str, ...],
    hosts_per_lens: int,
    seed: int,
    metric: str = "mahalanobis",
    assignment: str = "optimal",
    with_replacement: bool = False,
    forbidden: np.ndarray | None = None,
) -> MatchingResult:
    """Assign ``hosts_per_lens`` pool rows to each real lens.

    ``forbidden`` is an optional boolean mask over real x candidate marking pairs
    that may never be assigned -- used for the 63-cohort's requirement that hosts
    within 5 arcsec of any real lens or its physical group are excluded, and for
    within-split matching. Forbidden pairs are given an infinite cost rather than
    being removed from the pool, so that the pool size stays constant across
    cells and the assignment problem stays comparable.

    With zero matching covariates (the M0 cells) there is no meaningful distance,
    so hosts are drawn at random under the supplied seed. That is the honest
    representation of "no matching" and it keeps M0 inside the same machinery as
    every other cell rather than making it a special case elsewhere.
    """
    if hosts_per_lens < 1:
        raise MatchingError("hosts_per_lens must be at least 1")
    n_real = real.shape[0]
    n_candidates = candidates.shape[0]
    required = n_real * hosts_per_lens
    if not with_replacement and n_candidates < required:
        raise MatchingError(
            f"pool of {n_candidates} cannot supply {required} hosts without replacement"
        )

    rng = np.random.default_rng(seed)

    if forbidden is not None and forbidden.shape != (n_real, n_candidates):
        raise MatchingError("forbidden mask shape does not match real x candidate rows")

    if real.shape[1] == 0:
        return _random_assignment(
            n_real,
            n_candidates,
            hosts_per_lens=hosts_per_lens,
            with_replacement=with_replacement,
            rng=rng,
            seed=seed,
            columns=columns,
            forbidden=forbidden,
        )

    if metric == "mahalanobis":
        covariance, condition = pooled_covariance(real, candidates)
        cost = mahalanobis_cost(real, candidates, covariance)
    elif metric == "euclidean":
        condition = 1.0
        cost = euclidean_cost(real, candidates)
    else:
        raise MatchingError(f"unknown metric {metric!r}")

    if forbidden is not None:
        cost = np.where(forbidden, np.inf, cost)

    if with_replacement:
        matched_sets = _assign_with_replacement(cost, hosts_per_lens)
    elif assignment == "optimal":
        matched_sets = _assign_optimal(cost, hosts_per_lens)
    elif assignment == "greedy":
        matched_sets = _assign_greedy(cost, hosts_per_lens)
    else:
        raise MatchingError(f"unknown assignment {assignment!r}")

    return MatchingResult(
        matched_sets=matched_sets,
        columns=columns,
        hosts_per_lens=hosts_per_lens,
        with_replacement=with_replacement,
        seed=seed,
        covariance_condition_number=condition,
        metric=metric,
        assignment=assignment,
    )


def _assign_optimal(cost: np.ndarray, hosts_per_lens: int) -> tuple[MatchedSet, ...]:
    """Solve the rectangular assignment problem on the replicated cost matrix.

    Each real lens is replicated ``hosts_per_lens`` times, giving a
    (n_real * hosts_per_lens) x n_candidates problem whose optimal solution is
    the optimal without-replacement matching. This is the construction design 8.3
    specifies as "the 444 x pool cost matrix".
    """
    n_real = cost.shape[0]
    replicated = np.repeat(cost, hosts_per_lens, axis=0)
    if not np.isfinite(replicated).any(axis=1).all():
        offenders = np.where(~np.isfinite(replicated).any(axis=1))[0]
        raise MatchingError(
            f"{offenders.size} real slots have no admissible candidate; the "
            "forbidden mask or the pool is too restrictive"
        )
    rows, cols = linear_sum_assignment(replicated)
    if rows.size < replicated.shape[0]:
        raise MatchingError("assignment did not saturate every real slot")

    by_real: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n_real)}
    for row, col in zip(rows, cols, strict=True):
        real_index = int(row // hosts_per_lens)
        by_real[real_index].append((int(col), float(cost[real_index, col])))

    return tuple(
        MatchedSet(
            real_index=i,
            host_indices=tuple(h for h, _ in sorted(by_real[i])),
            distances=tuple(d for _, d in sorted(by_real[i])),
        )
        for i in range(n_real)
    )


def _assign_greedy(cost: np.ndarray, hosts_per_lens: int) -> tuple[MatchedSet, ...]:
    """Order-dependent nearest-available sweep. Retained for comparison only."""
    n_real = cost.shape[0]
    taken: set[int] = set()
    sets = []
    for i in range(n_real):
        order = np.argsort(cost[i])
        chosen: list[int] = []
        for candidate in order:
            if len(chosen) == hosts_per_lens:
                break
            if int(candidate) in taken or not np.isfinite(cost[i, candidate]):
                continue
            chosen.append(int(candidate))
            taken.add(int(candidate))
        if len(chosen) < hosts_per_lens:
            raise MatchingError(f"greedy exhausted the pool at real index {i}")
        sets.append(
            MatchedSet(
                real_index=i,
                host_indices=tuple(chosen),
                distances=tuple(float(cost[i, c]) for c in chosen),
            )
        )
    return tuple(sets)


def _assign_with_replacement(
    cost: np.ndarray, hosts_per_lens: int
) -> tuple[MatchedSet, ...]:
    """Nearest ``hosts_per_lens`` candidates per real lens, reuse permitted.

    Only the historical M0a cell uses this, because the published artifact was
    built with replacement. Reproducing it requires reproducing that.
    """
    sets = []
    for i in range(cost.shape[0]):
        finite = np.flatnonzero(np.isfinite(cost[i]))
        if finite.size < hosts_per_lens:
            raise MatchingError(
                f"real index {i} has only {finite.size} admissible candidates"
            )
        order = finite[np.argsort(cost[i, finite])[:hosts_per_lens]]
        sets.append(
            MatchedSet(
                real_index=i,
                host_indices=tuple(int(c) for c in order),
                distances=tuple(float(cost[i, c]) for c in order),
            )
        )
    return tuple(sets)


def _random_assignment(
    n_real: int,
    n_candidates: int,
    *,
    hosts_per_lens: int,
    with_replacement: bool,
    rng: np.random.Generator,
    seed: int,
    columns: tuple[str, ...],
    forbidden: np.ndarray | None,
) -> MatchingResult:
    if with_replacement:
        selected = []
        for real_index in range(n_real):
            allowed = np.arange(n_candidates)
            if forbidden is not None:
                allowed = allowed[~forbidden[real_index]]
            if allowed.size < hosts_per_lens:
                raise MatchingError(
                    f"real index {real_index} has only {allowed.size} admissible "
                    "candidates"
                )
            selected.extend(
                rng.choice(allowed, size=hosts_per_lens, replace=False).tolist()
            )
        picks = np.asarray(selected, dtype=np.int64)
    else:
        random_cost = rng.random((n_real, n_candidates))
        if forbidden is not None:
            random_cost = np.where(forbidden, np.inf, random_cost)
        assigned = _assign_optimal(random_cost, hosts_per_lens)
        picks = np.asarray(
            [host for matched in assigned for host in matched.host_indices],
            dtype=np.int64,
        )
    sets = tuple(
        MatchedSet(
            real_index=i,
            host_indices=tuple(
                int(p) for p in picks[i * hosts_per_lens : (i + 1) * hosts_per_lens]
            ),
            distances=tuple(0.0 for _ in range(hosts_per_lens)),
        )
        for i in range(n_real)
    )
    return MatchingResult(
        matched_sets=sets,
        columns=columns,
        hosts_per_lens=hosts_per_lens,
        with_replacement=with_replacement,
        seed=seed,
        covariance_condition_number=1.0,
        metric="none",
        assignment="random",
        diagnostics={"note": "no matching covariates; hosts drawn at random"},
    )


def rematching_draws(
    real: np.ndarray,
    candidates: np.ndarray,
    *,
    columns: tuple[str, ...],
    hosts_per_lens: int,
    base_seed: int,
    draws: int,
    **kwargs: Any,
) -> tuple[MatchingResult, ...]:
    """K independent host draws over the same pool and covariates.

    Design 8.3's rematching sensitivity: the same matching problem is re-solved
    under different seeds, and the post-injection balance is recomputed in each.
    Whether an imbalance persists across all draws, appears in some, or moves
    between covariates is what distinguishes an engine-level defect from a
    host-selection artefact from a joint-support tail.

    Optimal assignment is deterministic given a cost matrix, so the draws are
    made independent by resampling the candidate pool: each draw solves the
    assignment against a distinct random subsample. That varies which hosts are
    available without varying the covariates, the calipers, or the metric, which
    is what the design asks for.

    Selecting among these draws on any basis other than the pre-registered seed
    is forbidden; this function returns all of them so the report can show all of
    them.
    """
    if draws < 1:
        raise MatchingError("draws must be at least 1")
    n_candidates = candidates.shape[0]
    required = real.shape[0] * hosts_per_lens
    # Leave enough headroom that the assignment has real freedom in each draw,
    # while keeping every draw's pool the same size so the draws are comparable.
    subsample_size = max(required * 2, int(n_candidates * 0.8))
    subsample_size = min(subsample_size, n_candidates)

    results = []
    for k in range(draws):
        seed = base_seed + 1000 * (k + 1)
        rng = np.random.default_rng(seed)
        if subsample_size < n_candidates:
            subset = np.sort(
                rng.choice(n_candidates, size=subsample_size, replace=False)
            )
        else:
            subset = np.arange(n_candidates)
        draw_kwargs = dict(kwargs)
        forbidden = draw_kwargs.pop("forbidden", None)
        subset_forbidden = None
        if forbidden is not None:
            forbidden = np.asarray(forbidden, dtype=bool)
            if forbidden.shape != (real.shape[0], n_candidates):
                raise MatchingError(
                    "forbidden mask shape does not match real x candidate rows"
                )
            subset_forbidden = forbidden[:, subset]
        result = match_hosts(
            real,
            candidates[subset],
            columns=columns,
            hosts_per_lens=hosts_per_lens,
            seed=seed,
            forbidden=subset_forbidden,
            **draw_kwargs,
        )
        # Map pool indices back onto the full pool so callers can index the
        # original frame.
        remapped = tuple(
            MatchedSet(
                real_index=s.real_index,
                host_indices=tuple(int(subset[h]) for h in s.host_indices),
                distances=s.distances,
            )
            for s in result.matched_sets
        )
        results.append(
            MatchingResult(
                matched_sets=remapped,
                columns=result.columns,
                hosts_per_lens=result.hosts_per_lens,
                with_replacement=result.with_replacement,
                seed=seed,
                covariance_condition_number=result.covariance_condition_number,
                metric=result.metric,
                assignment=result.assignment,
                diagnostics={"draw_index": k, "pool_subsample": int(subsample_size)},
            )
        )
    return tuple(results)
