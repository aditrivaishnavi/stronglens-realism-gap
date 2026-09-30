"""Independent-formula recomputation of every metric from committed OOF rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    ContractError,
    atomic_write_json,
    canonical_frame_sha256,
    load_config,
    read_json,
    sha256_file,
    validate_config_seal,
    validate_runtime,
)

REPRESENTATIONS = (
    "stein_ssl_resnet50",
    "imagenet_effnet_v2_s",
    "pixel_statistics",
)


def _pairwise_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Compute AUC directly from all positive-negative score orderings."""
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if (
        positive.shape != (43,)
        or negative.shape != (43,)
        or not np.isfinite(positive).all()
        or not np.isfinite(negative).all()
    ):
        raise ContractError("independent pooled AUC lacks exact finite support")
    comparisons = positive[:, None] - negative[None, :]
    numerator = np.sum(comparisons > 0.0) + 0.5 * np.sum(comparisons == 0.0)
    return float(numerator / comparisons.size)


def _recompute(frame: pd.DataFrame) -> dict[str, Any]:
    if (
        len(frame) != 129
        or frame["row_id"].nunique() != 129
        or frame["pair_id"].nunique() != 43
        or frame["population"].value_counts().sort_index().to_dict()
        != {"bare": 43, "injected": 43, "observed_reference": 43}
        or set(frame["fold"].astype(int)) != set(range(5))
        or not np.isfinite(frame["axis_score"].to_numpy(dtype=float)).all()
    ):
        raise ContractError("OOF prediction support is invalid")
    if frame.groupby("pair_id")["fold"].nunique().max() != 1:
        raise ContractError("an OOF triplet crosses folds")
    wide = frame.pivot(
        index=["pair_id", "reference_id", "target_galaxy_id"],
        columns="population",
        values="axis_score",
    ).reset_index()
    wide.columns.name = None
    if len(wide) != 43 or wide.isna().any().any():
        raise ContractError("OOF predictions do not form exact triplets")
    movement = wide["injected"].to_numpy(dtype=np.float64) - wide["bare"].to_numpy(
        dtype=np.float64
    )
    above = int(np.sum(movement > 0.0))
    ties = int(np.sum(movement == 0.0))
    below = int(np.sum(movement < 0.0))
    return {
        "primary_within_target_movement": float(np.sum(movement) / 43.0),
        "diagnostics": {
            "pooled_observed_vs_bare_auc": _pairwise_auc(
                wide["observed_reference"].to_numpy(),
                wide["bare"].to_numpy(),
            ),
            "pooled_observed_vs_injected_auc": _pairwise_auc(
                wide["observed_reference"].to_numpy(),
                wide["injected"].to_numpy(),
            ),
            "pooled_injected_vs_bare_auc": _pairwise_auc(
                wide["injected"].to_numpy(),
                wide["bare"].to_numpy(),
            ),
            "paired_injected_above_bare_concordance": (above + 0.5 * ties) / 43.0,
            "paired_counts": {
                "injected_above_bare": above,
                "ties": ties,
                "injected_below_bare": below,
                "total": 43,
            },
        },
        "movement_distribution": {
            "minimum": float(np.min(movement)),
            "median": float(np.median(movement)),
            "maximum": float(np.max(movement)),
        },
    }


def _assert_close(observed: float, expected: float, context: str) -> None:
    if not np.isclose(observed, expected, rtol=0.0, atol=2e-12):
        raise ContractError(f"{context} differs: {observed} != {expected}")


def _compare_metrics(
    recomputed: dict[str, Any],
    registered: dict[str, Any],
    representation: str,
) -> None:
    _assert_close(
        recomputed["primary_within_target_movement"],
        float(registered["primary_within_target_movement"]),
        f"{representation} primary movement",
    )
    for name in (
        "pooled_observed_vs_bare_auc",
        "pooled_observed_vs_injected_auc",
        "pooled_injected_vs_bare_auc",
        "paired_injected_above_bare_concordance",
    ):
        _assert_close(
            recomputed["diagnostics"][name],
            float(registered["diagnostics"][name]),
            f"{representation} {name}",
        )
    if (
        recomputed["diagnostics"]["paired_counts"]
        != registered["diagnostics"]["paired_counts"]
    ):
        raise ContractError(f"{representation} paired counts differ")
    for name in ("minimum", "median", "maximum"):
        _assert_close(
            recomputed["movement_distribution"][name],
            float(registered["movement_distribution"][name]),
            f"{representation} movement {name}",
        )


def verify(root: Path) -> dict[str, Any]:
    config, _ = load_config(root)
    validate_config_seal(root, config)
    validate_runtime(root, config)
    result_dir = root / "results_v2/gmatch/analysis"
    receipt_path = result_dir / config["outputs"]["result_receipt"]
    contract_path = result_dir / config["outputs"]["result_contract"]
    authorization_path = result_dir / config["outputs"]["analysis_authorization"]
    output = result_dir / config["outputs"]["metric_recomputation"]
    if output.exists():
        raise FileExistsError(f"verification refuses to overwrite {output}")
    receipt = read_json(receipt_path)
    contract = read_json(contract_path)
    authorization = read_json(authorization_path)
    if (
        receipt.get("status") != "complete_final_G_match_linear_axis_analysis"
        or receipt.get("result_contract_sha256") != sha256_file(contract_path)
        or receipt.get("analysis_authorization_sha256")
        != sha256_file(authorization_path)
        or contract.get("status") != "sealed_final_G_match_result_contract"
        or authorization.get("status") != "authorized_final_G_match_analysis"
        or authorization.get("result_contract_sha256") != sha256_file(contract_path)
        or receipt.get("feature_suite_authority_sha256")
        != config["inputs"]["feature_suite_authority"]["sha256"]
        or receipt.get("feature_independent_validation_sha256")
        != config["inputs"]["feature_independent_validation"]["sha256"]
        or receipt.get("formal_confidence_interval") is not None
        or receipt.get("significance_test") is not None
        or receipt.get("bootstrap_run") is not False
        or receipt.get("causal_attribution") is not False
        or receipt.get("realism_percentage_reported") is not False
        or receipt.get("independent_replication_claimed") is not False
    ):
        raise ContractError("final G_match receipt or authority is invalid")

    recomputed: dict[str, Any] = {}
    identity: pd.DataFrame | None = None
    for representation in REPRESENTATIONS:
        binding = receipt["outputs"][representation]
        path = result_dir / binding["path"]
        if sha256_file(path) != binding["sha256"]:
            raise ContractError(f"{representation} OOF file hash differs")
        frame = pd.read_parquet(path)
        if (
            canonical_frame_sha256(
                frame,
                columns=frame.columns.tolist(),
                sort_by=["row_id"],
            )
            != binding["content_sha256"]
            or frame["representation"].nunique() != 1
            or frame["representation"].iloc[0] != representation
        ):
            raise ContractError(f"{representation} OOF content differs")
        current_identity = frame[
            [
                "row_id",
                "pair_id",
                "population",
                "reference_id",
                "target_galaxy_id",
                "dependency_component_id",
                "fold",
            ]
        ]
        if identity is None:
            identity = current_identity
        else:
            try:
                pd.testing.assert_frame_equal(
                    identity,
                    current_identity,
                    check_exact=True,
                )
            except AssertionError as error:
                raise ContractError(
                    "representations do not share exact OOF identities and folds"
                ) from error
        metrics = _recompute(frame)
        _compare_metrics(metrics, receipt["results"][representation], representation)
        recomputed[representation] = metrics

    primary = recomputed["stein_ssl_resnet50"]["primary_within_target_movement"]
    _assert_close(
        primary,
        float(receipt["primary"]["point_estimate"]),
        "top-level primary movement",
    )
    verification = {
        "schema_version": 1,
        "status": "PASS",
        "analysis": "G_match",
        "verification": (
            "independent_formula_recomputation_of_all_metrics_from_committed_OOF_rows"
        ),
        "metric_verifier_uuid": config["metric_verifier_uuid"],
        "verifier_sha256": sha256_file(Path(__file__)),
        "result_receipt_sha256": sha256_file(receipt_path),
        "result_contract_sha256": sha256_file(contract_path),
        "analysis_authorization_sha256": sha256_file(authorization_path),
        "feature_suite_authority_sha256": config["inputs"]["feature_suite_authority"][
            "sha256"
        ],
        "feature_independent_validation_sha256": config["inputs"][
            "feature_independent_validation"
        ]["sha256"],
        "metrics": recomputed,
        "all_metrics_match": True,
        "all_prediction_files_authenticated": True,
        "all_representations_share_exact_rows_pairs_and_folds": True,
        "all_triplets_fold_closed": True,
        "pooled_AUCs_labelled_nonpaired_diagnostics": True,
        "formal_confidence_interval": None,
        "significance_test": None,
        "bootstrap_run": False,
        "causal_attribution": False,
        "realism_percentage_reported": False,
        "independent_replication_claimed": False,
        "this_is_not_separate_final_independent_verification": True,
        "ready_for_separate_final_verification": True,
    }
    atomic_write_json(verification, output)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    arguments = parser.parse_args()
    result = verify(arguments.repository_root.resolve())
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
