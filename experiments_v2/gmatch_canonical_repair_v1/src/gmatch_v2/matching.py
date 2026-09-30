"""Deterministic maximum-cardinality robust-IQR matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .authority import COVARIATES, AuthorityBundle
from .contracts import (
    ContractError,
    canonical_json_sha256,
    canonical_row_sha256,
    frame_content_sha256,
    stable_digest,
)
from .schemas import assert_context_schema


@dataclass(frozen=True)
class MatchResult:
    matches: pd.DataFrame
    reference_disposition: pd.DataFrame
    host_disposition: pd.DataFrame
    balance: pd.DataFrame
    scaler_comparison: pd.DataFrame
    receipt: dict[str, Any]


def matching_contract_sha256(config: dict[str, Any]) -> str:
    return canonical_json_sha256(config["matching"])


def _stable_order(values: pd.Series, *, namespace: str) -> np.ndarray:
    ordered = sorted(
        (
            stable_digest(namespace, value),
            str(value),
            index,
        )
        for index, value in enumerate(values.astype(str))
    )
    return np.asarray([entry[2] for entry in ordered], dtype=np.int64)


def solve_assignment(
    distances: np.ndarray,
    admissible: np.ndarray,
    reference_ids: pd.Series,
    host_ids: pd.Series,
    *,
    total_distance_caliper: float,
) -> list[tuple[int, int | None]]:
    """Solve maximum cardinality, then minimum distance, under a frozen tie rule."""
    if distances.shape != admissible.shape:
        raise ContractError("distance and admissibility matrices differ")
    n_reference, n_host = distances.shape
    if len(reference_ids) != n_reference or len(host_ids) != n_host:
        raise ContractError("assignment identifiers differ from matrix dimensions")
    reference_order = _stable_order(reference_ids, namespace="G_match_v2:reference")
    host_order = _stable_order(host_ids, namespace="G_match_v2:host")
    ordered_distance = distances[np.ix_(reference_order, host_order)]
    ordered_admissible = admissible[np.ix_(reference_order, host_order)]
    # One lost match is always more expensive than every admissible distance
    # combined, making cardinality the exact primary objective.
    unmatched_penalty = (n_reference + 1) * (total_distance_caliper + 1.0)
    cost = np.full((n_reference, n_host + n_reference), np.inf, dtype=float)
    cost[:, :n_host] = np.where(ordered_admissible, ordered_distance, np.inf)
    for index in range(n_reference):
        cost[index, n_host + index] = unmatched_penalty
    rows, columns = linear_sum_assignment(cost)
    if len(rows) != n_reference:
        raise ContractError("assignment omitted a reference disposition")
    result: list[tuple[int, int | None]] = []
    for ordered_reference, selected in zip(rows, columns, strict=True):
        original_reference = int(reference_order[ordered_reference])
        original_host = int(host_order[selected]) if selected < n_host else None
        result.append((original_reference, original_host))
    return sorted(result)


def _row_hashes(frame: pd.DataFrame) -> list[str]:
    columns = list(frame.columns)
    return [
        canonical_row_sha256(row, columns) for row in frame.to_dict(orient="records")
    ]


def _componentwise_scaler(
    hosts: pd.DataFrame,
    covariates: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = hosts[covariates].to_numpy(float)
    centres = []
    scales = []
    finite_counts = []
    missing_counts = []
    for index, _covariate in enumerate(covariates):
        values = matrix[:, index]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ContractError("a registered covariate has no finite host values")
        q25, q75 = np.quantile(finite, [0.25, 0.75], method="linear")
        centres.append(float(np.median(finite)))
        scales.append(float(q75 - q25))
        finite_counts.append(int(finite.size))
        missing_counts.append(int(len(values) - finite.size))
    centre = np.asarray(centres)
    scale = np.asarray(scales)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        bad = [covariates[index] for index in np.flatnonzero(scale <= 0)]
        raise ContractError(f"component-wise host IQR is degenerate: {bad}")
    return (
        centre,
        scale,
        np.asarray(finite_counts, dtype=int),
        np.asarray(missing_counts, dtype=int),
    )


def _distance_and_assignment(
    eligible_reference: pd.DataFrame,
    eligible_host: pd.DataFrame,
    covariates: list[str],
    scale: np.ndarray,
    *,
    component_caliper: float,
    distance_caliper: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int | None]]]:
    reference_matrix = eligible_reference[covariates].to_numpy(float)
    host_matrix = eligible_host[covariates].to_numpy(float)
    differences = (reference_matrix[:, None, :] - host_matrix[None, :, :]) / scale
    distances = np.sqrt(np.square(differences).sum(axis=2))
    admissible = (np.abs(differences) <= component_caliper).all(axis=2) & (
        distances <= distance_caliper
    )
    assignments = solve_assignment(
        distances,
        admissible,
        eligible_reference["galaxy_id"],
        eligible_host["galaxy_id"],
        total_distance_caliper=distance_caliper,
    )
    return differences, distances, admissible, assignments


def match(bundle: AuthorityBundle, config: dict[str, Any]) -> MatchResult:
    specification = config["matching"]
    covariates = list(specification["covariates"])
    if tuple(covariates) != COVARIATES:
        raise ContractError("registered five-covariate order has drifted")
    references = bundle.reference.copy()
    hosts = bundle.host_tractor.copy()
    if (
        len(references) != 43
        or len(hosts) != 4500
        or set(hosts["galaxy_id"].astype(str))
        != set(bundle.host["galaxy_id"].astype(str))
    ):
        raise ContractError("matching populations differ from direct authorities")
    if not hosts["proposed_role"].eq("experiment_host").all():
        raise ContractError("matching candidates are not all experiment hosts")
    if set(references["galaxy_id"].astype(str)) & set(hosts["galaxy_id"].astype(str)):
        raise ContractError("a reference appears in the candidate host population")
    reference_complete = np.isfinite(references[covariates].to_numpy(float)).all(axis=1)
    host_complete = np.isfinite(hosts[covariates].to_numpy(float)).all(axis=1)
    eligible_reference = references.loc[reference_complete].reset_index(drop=True)
    eligible_host = hosts.loc[host_complete].reset_index(drop=True)
    if eligible_reference.empty or eligible_host.empty:
        raise ContractError("matching has no complete reference or host rows")
    component_caliper = float(specification["component_caliper_standardised"])
    distance_caliper = float(specification["total_distance_caliper"])
    centre, scale, finite_counts, missing_counts = _componentwise_scaler(
        hosts, covariates
    )
    expected_scaler = specification["expected_scaler"]
    for index, covariate in enumerate(covariates):
        expected = expected_scaler[covariate]
        observed = {
            "finite": int(finite_counts[index]),
            "missing": int(missing_counts[index]),
            "median": float(centre[index]),
            "iqr": float(scale[index]),
        }
        if observed != expected:
            raise ContractError(f"governing scaler differs for {covariate}")
    differences, distances, admissible, assignments = _distance_and_assignment(
        eligible_reference,
        eligible_host,
        covariates,
        scale,
        component_caliper=component_caliper,
        distance_caliper=distance_caliper,
    )
    complete_matrix = eligible_host[covariates].to_numpy(float)
    prior_centre = np.median(complete_matrix, axis=0)
    prior_q25, prior_q75 = np.quantile(
        complete_matrix, [0.25, 0.75], axis=0, method="linear"
    )
    prior_scale = prior_q75 - prior_q25
    (
        _prior_differences,
        prior_distances,
        prior_admissible,
        prior_assignments,
    ) = _distance_and_assignment(
        eligible_reference,
        eligible_host,
        covariates,
        prior_scale,
        component_caliper=component_caliper,
        distance_caliper=distance_caliper,
    )
    reference_matrix = eligible_reference[covariates].to_numpy(float)
    host_matrix = eligible_host[covariates].to_numpy(float)
    match_rows: list[dict[str, Any]] = []
    selected_hosts: set[str] = set()
    selected_by_reference: dict[str, str] = {}
    support_by_reference: dict[str, tuple[int, float]] = {}
    for reference_index in range(len(eligible_reference)):
        support = distances[reference_index, admissible[reference_index]]
        reference_id = str(eligible_reference.iloc[reference_index]["galaxy_id"])
        support_by_reference[reference_id] = (
            int(admissible[reference_index].sum()),
            float(support.min()) if support.size else float("nan"),
        )
    for reference_index, host_index in assignments:
        reference_id = str(eligible_reference.iloc[reference_index]["galaxy_id"])
        if host_index is None:
            continue
        target_id = str(eligible_host.iloc[host_index]["galaxy_id"])
        pair_id = f"gmatch2-{stable_digest(reference_id, target_id)[:24]}"
        row: dict[str, Any] = {
            "pair_id": pair_id,
            "reference_id": reference_id,
            "target_galaxy_id": target_id,
            "distance": float(distances[reference_index, host_index]),
            "admissible_host_count": support_by_reference[reference_id][0],
        }
        for index, covariate in enumerate(covariates):
            row[f"reference__{covariate}"] = float(
                reference_matrix[reference_index, index]
            )
            row[f"target__{covariate}"] = float(host_matrix[host_index, index])
            row[f"delta_standardised__{covariate}"] = float(
                differences[reference_index, host_index, index]
            )
        match_rows.append(row)
        selected_hosts.add(target_id)
        selected_by_reference[reference_id] = target_id
    matches = pd.DataFrame(match_rows).sort_values(
        "reference_id", kind="mergesort", ignore_index=True
    )
    if (
        matches["reference_id"].duplicated().any()
        or matches["target_galaxy_id"].duplicated().any()
        or (matches["distance"] > distance_caliper).any()
    ):
        raise ContractError("matching violates uniqueness or total caliper")
    for covariate in covariates:
        if (
            matches[f"delta_standardised__{covariate}"].abs() > component_caliper
        ).any():
            raise ContractError(f"matching violates {covariate} component caliper")
    minimum = int(
        specification["minimum_matched_references_for_future_five_fold_analysis"]
    )
    if len(matches) < minimum:
        raise ContractError(f"only {len(matches)} references matched; need {minimum}")
    matches["match_row_sha256"] = _row_hashes(matches)
    governing_by_reference = {
        str(eligible_reference.iloc[reference_index]["galaxy_id"]): (
            (
                str(eligible_host.iloc[host_index]["galaxy_id"])
                if host_index is not None
                else "not_matched"
            ),
            (
                float(distances[reference_index, host_index])
                if host_index is not None
                else float("nan")
            ),
            int(admissible[reference_index].sum()),
        )
        for reference_index, host_index in assignments
    }
    prior_by_reference = {
        str(eligible_reference.iloc[reference_index]["galaxy_id"]): (
            (
                str(eligible_host.iloc[host_index]["galaxy_id"])
                if host_index is not None
                else "not_matched"
            ),
            (
                float(prior_distances[reference_index, host_index])
                if host_index is not None
                else float("nan")
            ),
            int(prior_admissible[reference_index].sum()),
        )
        for reference_index, host_index in prior_assignments
    }
    comparison_rows = []
    for reference_id in sorted(governing_by_reference):
        target, distance, support = governing_by_reference[reference_id]
        prior_target, prior_distance, prior_support = prior_by_reference[reference_id]
        comparison_rows.append(
            {
                "reference_id": reference_id,
                "prior_target_galaxy_id": prior_target,
                "governing_target_galaxy_id": target,
                "assignment_changed": prior_target != target,
                "prior_distance": prior_distance,
                "governing_distance": distance,
                "prior_admissible_host_count": prior_support,
                "governing_admissible_host_count": support,
            }
        )
    scaler_comparison = pd.DataFrame(comparison_rows)
    scaler_comparison["comparison_row_sha256"] = _row_hashes(scaler_comparison)
    assert_context_schema(scaler_comparison, context="output.scaler_comparison")

    reference_disposition = references[["galaxy_id"]].copy()
    reference_disposition["covariates_complete"] = reference_complete
    reference_disposition["admissible_host_count"] = [
        support_by_reference.get(str(identifier), (0, float("nan")))[0]
        for identifier in reference_disposition["galaxy_id"]
    ]
    reference_disposition["nearest_admissible_distance"] = [
        support_by_reference.get(str(identifier), (0, float("nan")))[1]
        for identifier in reference_disposition["galaxy_id"]
    ]
    reference_disposition["matched"] = (
        reference_disposition["galaxy_id"].astype(str).isin(selected_by_reference)
    )
    reference_disposition["target_galaxy_id"] = (
        reference_disposition["galaxy_id"].astype(str).map(selected_by_reference)
    )
    reference_disposition["disposition"] = np.where(
        ~reference_disposition["covariates_complete"],
        "unmatched_missing_covariate",
        np.where(
            reference_disposition["matched"],
            "matched",
            "unmatched_no_common_support",
        ),
    )
    reference_disposition["disposition_row_sha256"] = _row_hashes(reference_disposition)
    reference_disposition = reference_disposition.sort_values(
        "galaxy_id", ignore_index=True
    )
    host_disposition = hosts[["galaxy_id"]].copy()
    host_disposition["scaler_authority_member"] = True
    for covariate in covariates:
        host_disposition[f"finite__{covariate}"] = np.isfinite(
            hosts[covariate].to_numpy(float)
        )
    host_disposition["covariates_complete"] = host_complete
    host_disposition["missing_covariates"] = [
        "|".join(
            covariate
            for covariate in covariates
            if not bool(host_disposition.loc[index, f"finite__{covariate}"])
        )
        or "none"
        for index in host_disposition.index
    ]
    host_disposition["matching_candidate_eligible"] = host_complete
    host_disposition["selected"] = (
        host_disposition["galaxy_id"].astype(str).isin(selected_hosts)
    )
    host_disposition["disposition"] = np.where(
        ~host_disposition["covariates_complete"],
        "excluded_missing_covariate",
        np.where(host_disposition["selected"], "selected", "eligible_not_selected"),
    )
    host_disposition["disposition_row_sha256"] = _row_hashes(host_disposition)
    host_disposition = host_disposition.sort_values("galaxy_id", ignore_index=True)
    if (
        int((~host_disposition["matching_candidate_eligible"]).sum())
        != int(config["matching"]["expected_nonfinite_observed_zmag_hosts"])
        or not host_disposition["scaler_authority_member"].all()
        or not host_disposition.loc[
            ~host_disposition["matching_candidate_eligible"], "missing_covariates"
        ]
        .eq("observed_zmag")
        .all()
    ):
        raise ContractError("nonfinite-host disposition policy was not preserved")

    balance_rows: list[dict[str, Any]] = []
    matched_references = matches[
        [f"reference__{covariate}" for covariate in covariates]
    ].to_numpy(float)
    matched_hosts = matches[
        [f"target__{covariate}" for covariate in covariates]
    ].to_numpy(float)
    for index, covariate in enumerate(covariates):
        deltas = (matched_references[:, index] - matched_hosts[:, index]) / scale[index]
        reference_variance = float(np.var(matched_references[:, index], ddof=1))
        target_variance = float(np.var(matched_hosts[:, index], ddof=1))
        balance_rows.append(
            {
                "covariate": covariate,
                "scaler_finite_count": int(finite_counts[index]),
                "scaler_missing_count": int(missing_counts[index]),
                "scaler_policy": (
                    "component_wise_all_finite_values_exact_authorized_roster"
                ),
                "host_universe_median": float(centre[index]),
                "host_universe_iqr": float(scale[index]),
                "pre_match_robust_standardised_mean_difference": float(
                    (
                        eligible_reference[covariate].mean()
                        - eligible_host[covariate].mean()
                    )
                    / scale[index]
                ),
                "post_match_robust_standardised_mean_difference": float(deltas.mean()),
                "post_match_mean_absolute_pair_difference": float(
                    np.abs(deltas).mean()
                ),
                "post_match_max_absolute_pair_difference": float(np.abs(deltas).max()),
                "post_match_reference_mean": float(matched_references[:, index].mean()),
                "post_match_target_mean": float(matched_hosts[:, index].mean()),
                "post_match_target_to_reference_variance_ratio": (
                    target_variance / reference_variance
                    if reference_variance > 0
                    else float("nan")
                ),
            }
        )
    balance = pd.DataFrame(balance_rows)
    balance["diagnostic_row_sha256"] = _row_hashes(balance)
    assert_context_schema(balance, context="output.balance")
    support_counts = reference_disposition.loc[
        reference_disposition["covariates_complete"], "admissible_host_count"
    ].to_numpy(int)
    assert_context_schema(matches, context="output.matches")
    assert_context_schema(reference_disposition, context="output.reference_disposition")
    assert_context_schema(host_disposition, context="output.host_disposition")
    changed_comparison = scaler_comparison.loc[scaler_comparison["assignment_changed"]]
    receipt = {
        "schema_version": 1,
        "status": "score_blind_G_match_v2_matching_complete",
        "method": "maximum_cardinality_then_minimum_robust_iqr_distance",
        "score_blind": True,
        "candidate_population": "exact_final_grec_hosts",
        "references": 43,
        "host_universe_rows": 4500,
        "scaler_authority_rows": 4500,
        "scaler_value_policy": (
            "component_wise_all_finite_values_exact_authorized_roster"
        ),
        "matching_candidate_policy": "all_five_covariates_finite",
        "missing_covariate_policy": (
            "retain_in_scaler_authority_accounting_mark_ineligible_no_imputation"
        ),
        "imputation_performed": False,
        "references_complete_covariates": int(reference_complete.sum()),
        "hosts_complete_covariates": int(host_complete.sum()),
        "hosts_ineligible_nonfinite_covariate": int((~host_complete).sum()),
        "scaler_finite_count": dict(
            zip(covariates, finite_counts.tolist(), strict=True)
        ),
        "scaler_missing_count": dict(
            zip(covariates, missing_counts.tolist(), strict=True)
        ),
        "matched_references": len(matches),
        "unmatched_references": 43 - len(matches),
        "covariates": covariates,
        "host_universe_median": dict(zip(covariates, centre.tolist(), strict=True)),
        "host_universe_iqr": dict(zip(covariates, scale.tolist(), strict=True)),
        "prior_complete_case_scaler": {
            "status": "non_governing_comparator",
            "rows": len(eligible_host),
            "median": dict(zip(covariates, prior_centre.tolist(), strict=True)),
            "iqr": dict(zip(covariates, prior_scale.tolist(), strict=True)),
            "changed_assignment_count": len(changed_comparison),
            "changed_references": changed_comparison["reference_id"].tolist(),
        },
        "component_caliper_standardised": component_caliper,
        "total_distance_caliper": distance_caliper,
        "distance_minimum": float(matches["distance"].min()),
        "distance_median": float(matches["distance"].median()),
        "distance_maximum": float(matches["distance"].max()),
        "admissible_host_count_minimum": int(support_counts.min()),
        "admissible_host_count_median": float(np.median(support_counts)),
        "admissible_host_count_maximum": int(support_counts.max()),
        "replacement": False,
        "targets_per_reference": 1,
        "tie_rule": "sha256_sorted_identifiers",
        "host_total_flux_floor_applied": False,
        "matching_contract_sha256": matching_contract_sha256(config),
        "authority_digest": bundle.authority_digest,
        "reference_catalogue_content_sha256": frame_content_sha256(
            references,
            columns=["galaxy_id", "payload_sha256", *covariates],
            sort_by=["galaxy_id"],
        ),
        "host_catalogue_content_sha256": frame_content_sha256(
            hosts,
            columns=["galaxy_id", "payload_sha256", "proposed_role", *covariates],
            sort_by=["galaxy_id"],
        ),
        "matches_content_sha256": frame_content_sha256(
            matches,
            columns=list(matches.columns),
            sort_by=["reference_id"],
        ),
    }
    return MatchResult(
        matches=matches,
        reference_disposition=reference_disposition,
        host_disposition=host_disposition,
        balance=balance,
        scaler_comparison=scaler_comparison,
        receipt=receipt,
    )
