"""Independently recompute every final G_domain AUC from OOF predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stronglens_probes.contracts import (
    ContractError,
    atomic_write_json,
    canonical_frame_sha256,
    sha256_file,
)

REPRESENTATIONS = (
    "stein_ssl_resnet50",
    "imagenet_effnet_v2_s",
    "pixel_statistics",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return payload


def _weighted_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute weighted AUC from weighted positive-negative pair orderings."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        labels.ndim != 1
        or scores.shape != labels.shape
        or weights.shape != labels.shape
        or set(np.unique(labels)) != {0, 1}
        or not np.isfinite(scores).all()
        or not np.isfinite(weights).all()
        or (weights < 0).any()
    ):
        raise ContractError("invalid arrays for independent weighted AUC")
    positive_total = float(weights[labels == 1].sum())
    negative_total = float(weights[labels == 0].sum())
    if positive_total <= 0 or negative_total <= 0:
        raise ContractError("independent AUC lacks class support")

    order = np.argsort(scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    ordered_weights = weights[order]
    numerator = 0.0
    negative_below = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        while (
            stop < len(order)
            and ordered_scores[stop] == ordered_scores[start]
        ):
            stop += 1
        group_labels = ordered_labels[start:stop]
        group_weights = ordered_weights[start:stop]
        positive = float(group_weights[group_labels == 1].sum())
        negative = float(group_weights[group_labels == 0].sum())
        numerator += positive * (negative_below + 0.5 * negative)
        negative_below += negative
        start = stop
    return numerator / (positive_total * negative_total)


def _assert_close(observed: float, expected: float, context: str) -> None:
    if not np.isclose(observed, expected, rtol=0.0, atol=2e-12):
        raise ContractError(
            f"{context} differs: independently {observed}, receipt {expected}"
        )


def verify(root: Path, output: Path) -> dict[str, Any]:
    result_dir = root / "results_v2/probes/gdomain_v2"
    receipt_path = result_dir / "GDOMAIN_FINAL_RESULT_RECEIPT.json"
    contract_path = result_dir / "GDOMAIN_FINAL_RESULT_CONTRACT.json"
    authorization_path = (
        result_dir / "GDOMAIN_FINAL_ANALYSIS_AUTHORIZATION.json"
    )
    receipt = _load_json(receipt_path)
    contract = _load_json(contract_path)
    authorization = _load_json(authorization_path)
    if output.exists():
        raise FileExistsError("independent verification refuses to overwrite output")
    if receipt.get("status") != "complete_final_G_domain_linear_probe":
        raise ContractError("final G_domain result receipt is incomplete")
    contract_sha256 = sha256_file(contract_path)
    if (
        receipt.get("result_contract_sha256") != contract_sha256
        or contract.get("status") != "sealed_final_G_domain_result_contract"
        or contract.get("analysis") != "G_domain"
        or authorization.get("result_contract_sha256s") != [contract_sha256]
        or authorization.get("allowed_stages") != ["probe_analysis"]
        or receipt.get("significance_test") is not None
        or receipt.get("population_confidence_interval") is not None
        or receipt.get("bootstrap_run") is not False
        or receipt.get("G_match_run") is not False
    ):
        raise ContractError("final result authority or inference scope is invalid")

    recomputed: dict[str, Any] = {}
    reference_rows: pd.DataFrame | None = None
    for representation in REPRESENTATIONS:
        binding = receipt["outputs"][representation]
        path = result_dir / binding["path"]
        if sha256_file(path) != binding["sha256"]:
            raise ContractError(f"{representation} prediction file hash differs")
        frame = pd.read_parquet(path)
        if (
            canonical_frame_sha256(
                frame,
                columns=frame.columns,
                sort_by=["row_id"],
            )
            != binding["content_sha256"]
        ):
            raise ContractError(f"{representation} prediction content differs")
        if (
            len(frame) != 7_149
            or frame["row_id"].astype(str).nunique() != 7_149
            or set(frame["fold"].astype(int)) != set(range(5))
            or set(frame["label"].astype(int)) != {0, 1}
            or frame["representation"].nunique() != 1
            or frame["representation"].iloc[0] != representation
        ):
            raise ContractError(f"{representation} prediction support is invalid")
        identity = frame[
            [
                "row_id",
                "population",
                "reference_id",
                "host_galaxy_id",
                "dependency_component_id",
                "fold",
                "label",
                "analysis_weight",
            ]
        ]
        if reference_rows is None:
            reference_rows = identity
        else:
            try:
                pd.testing.assert_frame_equal(
                    reference_rows,
                    identity,
                    check_exact=True,
                )
            except AssertionError as error:
                raise ContractError(
                    "representations do not share exact rows, folds, and weights"
                ) from error

        labels = frame["label"].to_numpy(dtype=np.int8)
        scores = frame["oof_score"].to_numpy(dtype=np.float64)
        weights = frame["analysis_weight"].to_numpy(dtype=np.float64)
        primary = _weighted_auc(labels, scores, weights)
        row_diagnostic = _weighted_auc(
            labels,
            scores,
            np.ones(len(frame), dtype=np.float64),
        )
        registered = receipt["results"][representation]
        _assert_close(
            primary,
            float(registered["primary_oof_auc"]),
            f"{representation} primary AUC",
        )
        _assert_close(
            row_diagnostic,
            float(registered["row_weighted_oof_auc_diagnostic"]),
            f"{representation} row-weighted AUC",
        )
        fold_aucs = []
        by_fold = {
            int(entry["fold"]): entry for entry in registered["folds"]
        }
        for fold in range(5):
            selected = frame["fold"].to_numpy(dtype=int) == fold
            fold_auc = _weighted_auc(
                labels[selected],
                scores[selected],
                weights[selected],
            )
            _assert_close(
                fold_auc,
                float(by_fold[fold]["weighted_auc"]),
                f"{representation} fold {fold} AUC",
            )
            fold_aucs.append({"fold": fold, "weighted_auc": fold_auc})
        recomputed[representation] = {
            "dependency_component_equal_oof_roc_auc": primary,
            "row_weighted_oof_roc_auc_diagnostic": row_diagnostic,
            "fold_dependency_component_equal_oof_roc_auc": fold_aucs,
        }

    verification = {
        "schema_version": 1,
        "status": "PASS",
        "analysis": "G_domain",
        "verification": "independent_AUC_recomputation_from_OOF_predictions",
        "verifier_sha256": sha256_file(Path(__file__)),
        "result_receipt_sha256": sha256_file(receipt_path),
        "result_contract_sha256": contract_sha256,
        "analysis_authorization_receipt_sha256": sha256_file(
            authorization_path
        ),
        "AUCs": recomputed,
        "all_global_and_fold_AUCs_match": True,
        "prediction_files_authenticated": True,
        "shared_rows_folds_and_weights_exact": True,
        "significance_test": None,
        "population_confidence_interval": None,
        "bootstrap_run": False,
        "G_match_run": False,
    }
    atomic_write_json(verification, output)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = verify(
        arguments.repository_root.resolve(),
        arguments.output.resolve(),
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
