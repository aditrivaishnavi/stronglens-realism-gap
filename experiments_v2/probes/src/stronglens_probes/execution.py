"""Receipt-bound, resumable, bounded execution for joint bootstrap shards."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from itertools import islice
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn
from joblib import Parallel, delayed, parallel_config

from .authorities import AuthorityContext
from .contracts import (
    ContractError,
    atomic_write_json,
    atomic_write_parquet,
    canonical_frame_sha256,
    canonical_json_sha256,
    require_sha256,
    sha256_file,
)
from .estimator_canary import run_estimator_coverage_canary
from .features import FeatureArtifact, FeatureSuiteAuthority
from .inference import (
    evaluate_joint_resample,
    iter_joint_resamples,
    stratified_dependency_support_canary,
)

SHARD_COLUMNS = [
    "draw",
    "gdomain_oof_auc",
    "gmatch_within_target_movement",
    "minimum_shared_reference_multiplier",
]


def _source_tree_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _runtime_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _validate_shard_frame(
    frame: pd.DataFrame,
    *,
    start: int,
    stop: int,
) -> None:
    if list(frame.columns) != SHARD_COLUMNS:
        raise ContractError("bootstrap shard schema is not exact")
    if frame["draw"].tolist() != list(range(start, stop)):
        raise ContractError("bootstrap shard has wrong draw IDs")
    metrics = frame[SHARD_COLUMNS[1:]].to_numpy(float)
    if not np.isfinite(metrics).all():
        raise ContractError("bootstrap shard contains non-finite metrics")
    if not frame["gdomain_oof_auc"].between(0.0, 1.0, inclusive="both").all():
        raise ContractError("bootstrap shard AUC is outside [0,1]")
    if frame["minimum_shared_reference_multiplier"].le(0).any():
        raise ContractError("bootstrap shard lost shared-reference support")


def _authorization(
    *,
    config: dict[str, Any],
    domain_artifact: FeatureArtifact,
    gmatch_artifact: FeatureArtifact,
    design_only: bool,
) -> None:
    if design_only:
        if {domain_artifact.data_role, gmatch_artifact.data_role} != {"design"}:
            raise ContractError("design-only execution received non-design artifacts")
        return
    execution = config["execution"]
    if execution["final_analysis_authorized"] is not True:
        raise ContractError("final analysis is not authorized by the frozen config")
    require_sha256(
        execution.get("analysis_authorization_receipt_sha256"),
        context="analysis authorization receipt",
    )


def _run_contract(
    *,
    config: dict[str, Any],
    authority: AuthorityContext,
    domain_artifacts: dict[str, FeatureArtifact],
    gmatch_artifacts: dict[str, FeatureArtifact],
    domain_folds: pd.DataFrame,
    gmatch_folds: pd.DataFrame,
    feature_suite_receipt_sha256: str,
    design_only: bool,
) -> dict[str, Any]:
    source_tree_sha256 = _source_tree_sha256()
    expected_source = config["execution"].get("source_tree_sha256")
    require_sha256(expected_source, context="registered source tree")
    if source_tree_sha256 != expected_source:
        raise ContractError("runtime source tree differs from registered digest")
    lock_path = Path(__file__).resolve().parents[2] / "uv.lock"
    lock_sha256 = sha256_file(lock_path)
    if lock_sha256 != config["execution"].get("dependency_lock_sha256"):
        raise ContractError("runtime dependency lock differs from registered digest")
    return {
        "schema_version": 1,
        "config_sha256": authority.config_sha256,
        "authority_digest": authority.authority_digest,
        "source_tree_sha256": source_tree_sha256,
        "dependency_lock_sha256": lock_sha256,
        "runtime_environment": _runtime_environment(),
        "domain_feature_artifacts": {
            name: {
                "features": artifact.features_sha256,
                "receipt": artifact.receipt_sha256,
            }
            for name, artifact in sorted(domain_artifacts.items())
        },
        "gmatch_feature_artifacts": {
            name: {
                "features": artifact.features_sha256,
                "receipt": artifact.receipt_sha256,
            }
            for name, artifact in sorted(gmatch_artifacts.items())
        },
        "authorization_receipt_sha256": next(
            iter(domain_artifacts.values())
        ).authorization.receipt_sha256,
        "domain_folds_sha256": canonical_frame_sha256(
            domain_folds,
            columns=domain_folds.columns,
            sort_by=["dependency_component_id"],
        ),
        "gmatch_folds_sha256": canonical_frame_sha256(
            gmatch_folds,
            columns=gmatch_folds.columns,
            sort_by=["dependency_component_id"],
        ),
        "feature_suite_receipt_sha256": feature_suite_receipt_sha256,
        "bootstrap_draws": int(config["inference"]["bootstrap_draws"]),
        "bootstrap_seed": int(config["inference"]["seed"]),
        "multiplier_scheme": config["inference"]["multiplier_scheme"],
        "design_only": design_only,
    }


def run_joint_bootstrap_shard(
    *,
    domain_suite: FeatureSuiteAuthority,
    domain_folds: pd.DataFrame,
    gmatch_suite: FeatureSuiteAuthority,
    gmatch_folds: pd.DataFrame,
    config: dict[str, Any],
    authority: AuthorityContext,
    start: int,
    stop: int,
    jobs: int,
    output_path: Path,
    receipt_path: Path,
    design_only: bool,
) -> dict[str, Any]:
    """Run or authenticate one immutable, resumable draw shard."""
    authority.assert_config(config)
    domain_suite.revalidate(config=config, authority=authority, stage="bootstrap")
    gmatch_suite.revalidate(config=config, authority=authority, stage="bootstrap")
    domain_artifacts = domain_suite.artifacts
    gmatch_artifacts = gmatch_suite.artifacts
    primary_id = config["representations"]["primary"]["id"]
    domain_artifact = domain_artifacts[primary_id]
    gmatch_artifact = gmatch_artifacts[primary_id]
    feature_suite_receipt_sha256 = canonical_json_sha256(
        {"domain": domain_suite.sha256, "gmatch": gmatch_suite.sha256}
    )
    _authorization(
        config=config,
        domain_artifact=domain_artifact,
        gmatch_artifact=gmatch_artifact,
        design_only=design_only,
    )
    total = int(config["inference"]["bootstrap_draws"])
    maximum_jobs = int(config["execution"]["maximum_parallel_jobs"])
    shard_draws = int(config["execution"]["shard_draws"])
    if not (0 <= start < stop <= total):
        raise ContractError("bootstrap shard range is outside registered draws")
    if stop - start > shard_draws:
        raise ContractError(f"bootstrap shard exceeds registered maximum {shard_draws}")
    if not design_only and stop < total and stop - start != shard_draws:
        raise ContractError("production nonterminal shards must use default size")
    if not 1 <= jobs <= maximum_jobs:
        raise ContractError(f"jobs must be between 1 and {maximum_jobs}")
    run_contract = _run_contract(
        config=config,
        authority=authority,
        domain_artifacts=domain_artifacts,
        gmatch_artifacts=gmatch_artifacts,
        domain_folds=domain_folds,
        gmatch_folds=gmatch_folds,
        feature_suite_receipt_sha256=feature_suite_receipt_sha256,
        design_only=design_only,
    )
    run_fingerprint = canonical_json_sha256(run_contract)
    shard_contract = {
        "run_fingerprint": run_fingerprint,
        "start": start,
        "stop": stop,
    }
    shard_fingerprint = canonical_json_sha256(shard_contract)
    all_resamples = iter_joint_resamples(
        pd.read_parquet(domain_artifact.manifest_path),
        pd.read_parquet(gmatch_artifact.manifest_path),
        domain_folds,
        gmatch_folds,
        draws=stop,
        seed=int(config["inference"]["seed"]),
        config=config,
        authority=authority,
    )
    resamples = list(islice(all_resamples, start, stop))

    def compute() -> pd.DataFrame:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            records = Parallel(
                n_jobs=jobs,
                max_nbytes="1M",
                mmap_mode="r",
                batch_size=1,
            )(
                delayed(evaluate_joint_resample)(
                    resample,
                    domain_artifact=domain_artifact,
                    domain_folds=domain_folds,
                    gmatch_artifact=gmatch_artifact,
                    gmatch_folds=gmatch_folds,
                    config=config,
                    authority=authority,
                )
                for resample in resamples
            )
        frame = pd.DataFrame(records).sort_values("draw", ignore_index=True)
        _validate_shard_frame(frame, start=start, stop=stop)
        return frame

    if output_path.exists() or receipt_path.exists():
        if not output_path.exists() or not receipt_path.exists():
            raise ContractError("bootstrap resume has a partial output/receipt pair")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        required_receipt = {
            "schema_version",
            "status",
            "run_contract",
            "run_fingerprint",
            "shard_fingerprint",
            "content_sha256",
            "start",
            "stop",
            "draws",
            "jobs",
            "bounded_parallelism",
            "memory_mapped_feature_bases",
            "elapsed_seconds",
            "seconds_per_draw",
            "output_sha256",
        }
        if set(receipt) != required_receipt:
            raise ContractError("bootstrap resume receipt schema is not exact")
        if (
            receipt.get("run_contract") != run_contract
            or receipt.get("run_fingerprint") != run_fingerprint
            or receipt.get("shard_fingerprint") != shard_fingerprint
            or receipt.get("output_sha256") != sha256_file(output_path)
        ):
            raise ContractError("bootstrap resume receipt differs from invocation")
        frame = pd.read_parquet(output_path)
        _validate_shard_frame(frame, start=start, stop=stop)
        content_sha256 = canonical_frame_sha256(
            frame,
            columns=frame.columns,
            sort_by=["draw"],
        )
        if receipt.get("content_sha256") != content_sha256:
            raise ContractError("bootstrap resume content digest is invalid")
        recomputed = compute()
        try:
            pd.testing.assert_frame_equal(frame, recomputed, check_exact=True)
        except AssertionError as error:
            raise ContractError(
                "bootstrap resume shard differs from deterministic recomputation"
            ) from error
        return {**receipt, "resumed": True}

    started = time.perf_counter()
    frame = compute()
    elapsed = time.perf_counter() - started
    atomic_write_parquet(frame, output_path)
    receipt = {
        "schema_version": 1,
        "status": "complete_joint_bootstrap_shard",
        "run_contract": run_contract,
        "run_fingerprint": run_fingerprint,
        "shard_fingerprint": shard_fingerprint,
        "content_sha256": canonical_frame_sha256(
            frame,
            columns=frame.columns,
            sort_by=["draw"],
        ),
        "start": start,
        "stop": stop,
        "draws": stop - start,
        "jobs": jobs,
        "bounded_parallelism": True,
        "memory_mapped_feature_bases": True,
        "elapsed_seconds": elapsed,
        "seconds_per_draw": elapsed / (stop - start),
        "output_sha256": sha256_file(output_path),
    }
    atomic_write_json(receipt, receipt_path)
    return {**receipt, "resumed": False}


def aggregate_joint_bootstrap_shards(
    *,
    shard_files: list[tuple[Path, Path]],
    domain_suite: FeatureSuiteAuthority,
    domain_folds: pd.DataFrame,
    gmatch_suite: FeatureSuiteAuthority,
    gmatch_folds: pd.DataFrame,
    config: dict[str, Any],
    authority: AuthorityContext,
    jobs: int,
    output_path: Path,
    receipt_path: Path,
    design_only: bool,
) -> dict[str, Any]:
    """Recompute-validate and aggregate every registered draw exactly once."""
    total = int(config["inference"]["bootstrap_draws"])
    expected_start = 0
    frames = []
    run_fingerprint: str | None = None
    shard_receipts = []
    for output, receipt in shard_files:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        start = int(payload.get("start", -1))
        stop = int(payload.get("stop", -1))
        if start != expected_start:
            raise ContractError(
                "bootstrap shards are missing, overlapping, or unordered"
            )
        validated = run_joint_bootstrap_shard(
            domain_suite=domain_suite,
            domain_folds=domain_folds,
            gmatch_suite=gmatch_suite,
            gmatch_folds=gmatch_folds,
            config=config,
            authority=authority,
            start=start,
            stop=stop,
            jobs=jobs,
            output_path=output,
            receipt_path=receipt,
            design_only=design_only,
        )
        if run_fingerprint is None:
            run_fingerprint = str(validated["run_fingerprint"])
        elif validated["run_fingerprint"] != run_fingerprint:
            raise ContractError("bootstrap shards have different run fingerprints")
        frames.append(pd.read_parquet(output))
        shard_receipts.append(
            {
                "start": start,
                "stop": stop,
                "receipt_sha256": sha256_file(receipt),
                "output_sha256": sha256_file(output),
            }
        )
        expected_start = stop
    if expected_start != total:
        raise ContractError(
            f"bootstrap shards stop at {expected_start}, expected {total}"
        )
    combined = pd.concat(frames, ignore_index=True)
    _validate_shard_frame(combined, start=0, stop=total)
    atomic_write_parquet(combined, output_path)
    summary = {
        "schema_version": 1,
        "status": "complete_joint_bootstrap_aggregate",
        "draws": total,
        "run_fingerprint": run_fingerprint,
        "shards": shard_receipts,
        "output_sha256": sha256_file(output_path),
        "content_sha256": canonical_frame_sha256(
            combined,
            columns=combined.columns,
            sort_by=["draw"],
        ),
        "gdomain_auc_descriptive_quantiles_2.5_97.5": np.quantile(
            combined["gdomain_oof_auc"], [0.025, 0.975]
        ).tolist(),
        "gmatch_movement_descriptive_quantiles_2.5_97.5": np.quantile(
            combined["gmatch_within_target_movement"], [0.025, 0.975]
        ).tolist(),
        "joint_covariance": float(
            np.cov(
                combined["gdomain_oof_auc"],
                combined["gmatch_within_target_movement"],
            )[0, 1]
        ),
        "independent_replication_claim": False,
        "formal_confidence_intervals": None,
        "coverage_validated_for_final_estimators": False,
        "uncertainty_label": (
            "finite_cohort_model_conditional_resampling_stability_quantiles"
        ),
        "population_inference": False,
        "significance_test": None,
        "conservative_parameter_bounds": {
            "gdomain_auc": [0.0, 1.0],
            "gmatch_movement": None,
        },
    }
    atomic_write_json(summary, receipt_path)
    return summary


def run_design_resampling_benchmark(
    *,
    config: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Run the practical 10k support canary and a design-only coverage check."""
    draws = int(config["execution"]["support_canary_draws"])
    component_folds = np.tile(np.arange(int(config["probe"]["folds"])), 20)
    started = time.perf_counter()
    component_folds = np.tile(np.arange(int(config["probe"]["folds"])), 40)
    component_classes = np.repeat([0, 1], len(component_folds) // 2)
    support = stratified_dependency_support_canary(
        component_folds,
        component_classes,
        draws=draws,
        seed=int(config["inference"]["seed"]),
    )
    estimator = run_estimator_coverage_canary(
        benchmark_draws=int(config["execution"]["design_benchmark_draws"]),
        oracle_studies=100,
        coverage_studies=40,
        posterior_draws=100,
        seed=int(config["inference"]["seed"]) + 2,
    )
    result = {
        "schema_version": 1,
        "status": "complete_design_only_resampling_benchmark",
        "config_sha256": canonical_json_sha256(config),
        "support": support,
        "actual_estimator": estimator,
        "elapsed_seconds": time.perf_counter() - started,
        "contains_final_features_or_scores": False,
    }
    atomic_write_json(result, output_path)
    return result
