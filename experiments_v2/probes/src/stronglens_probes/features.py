"""Authenticated, memory-mapped feature artifacts and mandatory probe suites."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .authorities import AuthorityContext
from .authorization import ExecutionAuthorization, protocol_config_sha256
from .contracts import (
    ContractError,
    atomic_write_json,
    canonical_frame_sha256,
    require_columns,
    require_nonempty_columns,
    require_sha256,
    require_sha256_column,
    sha256_file,
)


@dataclass(frozen=True)
class FeatureArtifact:
    representation_id: str
    values: np.ndarray
    manifest: pd.DataFrame
    receipt: dict[str, Any]
    features_path: Path
    receipt_path: Path
    manifest_path: Path
    features_sha256: str
    receipt_sha256: str
    manifest_sha256: str
    manifest_content_sha256: str
    row_ids_sha256: str
    pixel_hashes_sha256: str
    authority_digest: str
    data_role: str
    checkpoint_path: Path | None
    authorization: ExecutionAuthorization


@dataclass(frozen=True)
class FeatureSuiteAuthority:
    analysis: str
    data_role: str
    artifacts: dict[str, FeatureArtifact]
    analysis_manifest_path: Path
    recomputed_manifest: pd.DataFrame
    parent_bindings: dict[str, str]
    payload: dict[str, Any]
    path: Path
    sha256: str
    analysis_authorization: ExecutionAuthorization
    parent_artifacts: dict[str, Any]

    def revalidate(
        self,
        *,
        config: dict[str, Any],
        authority: AuthorityContext,
        stage: str,
    ) -> None:
        authority.assert_config(config)
        if (
            sha256_file(self.path) != self.sha256
            or json.loads(self.path.read_text(encoding="utf-8")) != self.payload
        ):
            raise ContractError("external feature-suite authority changed")
        self.analysis_authorization.revalidate(
            config=config,
            authority=authority,
            stage=stage,
            manifest_sha256=sha256_file(self.analysis_manifest_path),
            feature_suite_authority_sha256=self.sha256,
        )
        validate_feature_suite(
            self.artifacts,
            config=config,
            authority=authority,
            stage=stage,
        )
        if (
            config["execution"].get("allow_reduced_grec_design_fixture")
            and "design_recomputed_manifest" in self.parent_artifacts
        ):
            current_recomputed = self.parent_artifacts[
                "design_recomputed_manifest"
            ].copy()
        elif self.analysis == "G_domain":
            current_recomputed = recompute_domain_manifest(
                grec=self.parent_artifacts["grec"],
                subset=self.parent_artifacts["subset"],
                config=config,
                authority=authority,
            )
        else:
            current_recomputed = recompute_gmatch_manifest(
                grec=self.parent_artifacts["grec"],
                frozen_matches=self.parent_artifacts["frozen_matches"],
                frozen_pair_plan=self.parent_artifacts["frozen_pair_plan"],
                rendered=self.parent_artifacts["rendered"],
                config=config,
                authority=authority,
            )
        physical = pd.read_parquet(self.analysis_manifest_path)
        try:
            pd.testing.assert_frame_equal(
                physical.reset_index(drop=True),
                current_recomputed.reset_index(drop=True),
                check_exact=True,
            )
        except AssertionError as error:
            raise ContractError(
                "analysis manifest differs from authenticated parent recomputation"
            ) from error
        expected = _feature_suite_authority_payload(
            analysis=self.analysis,
            artifacts=self.artifacts,
            analysis_manifest_path=self.analysis_manifest_path,
            recomputed_manifest=current_recomputed,
            parent_bindings=self.parent_bindings,
            config=config,
            authority=authority,
        )
        if expected != self.payload:
            raise ContractError("feature-suite authority content is inconsistent")


def representation_spec(
    config: dict[str, Any], representation_id: str
) -> dict[str, Any]:
    entries = [
        config["representations"]["primary"],
        *config["representations"]["required_diagnostics"],
    ]
    by_id = {entry["id"]: entry for entry in entries}
    if representation_id not in by_id:
        raise ContractError(f"unregistered representation {representation_id}")
    return by_id[representation_id]


def _sequence_sha256(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_finite_mmap(values: np.ndarray, *, chunk_rows: int = 4096) -> None:
    for start in range(0, len(values), chunk_rows):
        if not np.isfinite(values[start : start + chunk_rows]).all():
            raise ContractError("feature matrix contains non-finite values")


def load_feature_artifact(
    *,
    features_path: Path,
    receipt_path: Path,
    manifest_path: Path,
    config: dict[str, Any],
    authority: AuthorityContext,
    authorization: ExecutionAuthorization,
    checkpoint_path: Path | None,
    expected_data_role: str,
) -> FeatureArtifact:
    """Authenticate one feature matrix without copying the full matrix."""
    authority.assert_config(config)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    representation_id = str(receipt.get("representation_id"))
    specification = representation_spec(config, representation_id)
    if expected_data_role != authorization.data_role:
        raise ContractError("feature role differs from execution authorization")
    features_sha256 = sha256_file(features_path)
    receipt_sha256 = sha256_file(receipt_path)
    manifest_sha256 = sha256_file(manifest_path)
    authorization.revalidate(
        config=config,
        authority=authority,
        stage="feature_load",
        manifest_sha256=manifest_sha256,
    )
    for field, observed in (
        ("features_file_sha256", features_sha256),
        ("analysis_manifest_sha256", manifest_sha256),
    ):
        require_sha256(receipt.get(field), context=f"feature receipt {field}")
        if receipt[field] != observed:
            raise ContractError(f"feature receipt has wrong {field}")
    if receipt.get("schema_version") != 2 or receipt.get("status") != "complete":
        raise ContractError("feature extraction receipt is incomplete")
    if receipt.get("data_role") != expected_data_role:
        raise ContractError("feature artifact has the wrong data role")
    if receipt.get("authority_digest") != authority.authority_digest:
        raise ContractError("feature artifact has the wrong authority context")
    if receipt.get("authorization_receipt_sha256") != authorization.receipt_sha256:
        raise ContractError("feature artifact has the wrong authorization receipt")
    if receipt.get("checkpoint_sha256") != specification["checkpoint_sha256"]:
        raise ContractError("feature artifact has the wrong encoder checkpoint")
    if specification["checkpoint_sha256"] is None:
        if checkpoint_path is not None:
            raise ContractError("pixel statistics must not have a checkpoint")
    else:
        if checkpoint_path is None:
            raise ContractError("encoder checkpoint path is required")
        if sha256_file(checkpoint_path) != specification["checkpoint_sha256"]:
            raise ContractError("encoder checkpoint bytes have the wrong SHA-256")
    if (
        receipt.get("preprocessing_contract_sha256")
        != specification["preprocessing_contract_sha256"]
    ):
        raise ContractError("feature artifact has the wrong preprocessing contract")
    if (
        receipt.get("extraction_contract_sha256")
        != specification["extraction_contract_sha256"]
    ):
        raise ContractError("feature artifact has the wrong extraction contract")
    package_root = Path(__file__).resolve().parents[1]
    extractor_path = package_root / specification["extractor_code_path"]
    preprocessing_path = package_root / specification["preprocessing_code_path"]
    code_expectations = {
        "extractor_code_sha256": (
            extractor_path,
            specification["extractor_code_sha256"],
        ),
        "preprocessing_code_sha256": (
            preprocessing_path,
            specification["preprocessing_code_sha256"],
        ),
    }
    for field, (path, expected) in code_expectations.items():
        require_sha256(expected, context=f"config {field}")
        if sha256_file(path) != expected or receipt.get(field) != expected:
            raise ContractError(f"feature artifact has the wrong {field}")

    manifest = pd.read_parquet(manifest_path)
    require_columns(
        manifest,
        {"row_id", "pixel_sha256", "authority_digest", "data_role"},
        context="feature analysis manifest",
    )
    require_nonempty_columns(
        manifest, {"row_id", "pixel_sha256"}, context="feature analysis manifest"
    )
    require_sha256_column(manifest, "pixel_sha256", context="feature analysis manifest")
    if manifest["row_id"].astype(str).duplicated().any():
        raise ContractError("feature analysis manifest has duplicate row IDs")
    if not manifest["authority_digest"].eq(authority.authority_digest).all():
        raise ContractError("feature manifest has the wrong authority context")
    if not manifest["data_role"].eq(expected_data_role).all():
        raise ContractError("feature manifest has the wrong data role")
    manifest_content_sha256 = canonical_frame_sha256(
        manifest,
        columns=manifest.columns,
        sort_by=["row_id"],
    )
    row_ids_sha256 = _sequence_sha256(manifest["row_id"])
    pixel_hashes_sha256 = _sequence_sha256(manifest["pixel_sha256"])
    expected_receipt_values = {
        "analysis_manifest_content_sha256": manifest_content_sha256,
        "row_ids_sha256": row_ids_sha256,
        "pixel_hashes_sha256": pixel_hashes_sha256,
        "rows": len(manifest),
        "feature_dimension": int(specification["feature_dimension"]),
        "dtype": "float32",
    }
    for field, expected in expected_receipt_values.items():
        if receipt.get(field) != expected:
            raise ContractError(f"feature receipt has wrong {field}")

    values = np.load(features_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(manifest), int(specification["feature_dimension"]))
    if values.shape != expected_shape:
        raise ContractError(
            f"{representation_id} feature shape {values.shape} != {expected_shape}"
        )
    if values.dtype != np.float32:
        raise ContractError("feature matrix must be native float32")
    _validate_finite_mmap(values)
    return FeatureArtifact(
        representation_id=representation_id,
        values=values,
        manifest=manifest,
        receipt=receipt,
        features_path=features_path,
        receipt_path=receipt_path,
        manifest_path=manifest_path.resolve(),
        features_sha256=features_sha256,
        receipt_sha256=receipt_sha256,
        manifest_sha256=manifest_sha256,
        manifest_content_sha256=manifest_content_sha256,
        row_ids_sha256=row_ids_sha256,
        pixel_hashes_sha256=pixel_hashes_sha256,
        authority_digest=authority.authority_digest,
        data_role=expected_data_role,
        checkpoint_path=checkpoint_path.resolve() if checkpoint_path else None,
        authorization=authorization,
    )


def revalidate_feature_artifact(
    artifact: FeatureArtifact,
    *,
    config: dict[str, Any],
    authority: AuthorityContext,
    stage: str,
) -> None:
    """Rehash all mutable bytes and reopen the matrix before every use."""
    authority.assert_config(config)
    artifact.authorization.revalidate(
        config=config,
        authority=authority,
        stage="feature_load",
        manifest_sha256=artifact.manifest_sha256,
    )
    if (
        sha256_file(artifact.features_path) != artifact.features_sha256
        or sha256_file(artifact.receipt_path) != artifact.receipt_sha256
        or json.loads(artifact.receipt_path.read_text(encoding="utf-8"))
        != artifact.receipt
        or sha256_file(
            Path(__file__).resolve().parents[1]
            / representation_spec(config, artifact.representation_id)[
                "extractor_code_path"
            ]
        )
        != representation_spec(config, artifact.representation_id)[
            "extractor_code_sha256"
        ]
        or sha256_file(
            Path(__file__).resolve().parents[1]
            / representation_spec(config, artifact.representation_id)[
                "preprocessing_code_path"
            ]
        )
        != representation_spec(config, artifact.representation_id)[
            "preprocessing_code_sha256"
        ]
    ):
        raise ContractError("feature artifact or implementation changed after loading")
    if (
        artifact.checkpoint_path is not None
        and sha256_file(artifact.checkpoint_path)
        != representation_spec(config, artifact.representation_id)["checkpoint_sha256"]
    ):
        raise ContractError("feature checkpoint changed after loading")
    if sha256_file(artifact.manifest_path) != artifact.manifest_sha256:
        raise ContractError("feature manifest changed after loading")
    fresh_manifest = pd.read_parquet(artifact.manifest_path)
    try:
        pd.testing.assert_frame_equal(
            artifact.manifest.reset_index(drop=True),
            fresh_manifest.reset_index(drop=True),
            check_exact=True,
        )
    except AssertionError as error:
        raise ContractError(
            "feature manifest mutated in memory after loading"
        ) from error
    reopened = np.load(artifact.features_path, mmap_mode="r", allow_pickle=False)
    if reopened.shape != artifact.values.shape or reopened.dtype != np.float32:
        raise ContractError("feature matrix changed shape or dtype after loading")
    _validate_finite_mmap(reopened)


def validate_feature_suite(
    artifacts: dict[str, FeatureArtifact],
    *,
    config: dict[str, Any],
    authority: AuthorityContext,
    stage: str = "probe_analysis",
) -> None:
    """Require Stein primary plus both ImageNet and pixel diagnostics."""
    authority.assert_config(config)
    expected = {
        config["representations"]["primary"]["id"],
        *(entry["id"] for entry in config["representations"]["required_diagnostics"]),
    }
    if set(artifacts) != expected:
        raise ContractError(
            "feature suite must contain Stein primary, ImageNet, and pixel diagnostics"
        )
    baseline = next(iter(artifacts.values()))
    shared_fields = (
        "manifest_sha256",
        "manifest_content_sha256",
        "row_ids_sha256",
        "pixel_hashes_sha256",
        "authority_digest",
        "data_role",
    )
    for name, artifact in artifacts.items():
        revalidate_feature_artifact(
            artifact,
            config=config,
            authority=authority,
            stage=stage,
        )
        if artifact.representation_id != name:
            raise ContractError("feature-suite key differs from receipt representation")
        for field in shared_fields:
            if getattr(artifact, field) != getattr(baseline, field):
                raise ContractError(f"feature suite differs on {field}")


def feature_suite_receipt(
    artifacts: dict[str, FeatureArtifact],
    *,
    config: dict[str, Any],
    authority: AuthorityContext,
) -> dict[str, Any]:
    """Return a persistable receipt proving both diagnostics were enforced."""
    validate_feature_suite(artifacts, config=config, authority=authority)
    return {
        "schema_version": 1,
        "status": "complete_authenticated_feature_suite",
        "primary_representation": config["representations"]["primary"]["id"],
        "primary_pretraining_image_overlap_status": "unproven",
        "required_diagnostics": [
            entry["id"] for entry in config["representations"]["required_diagnostics"]
        ],
        "pixel_diagnostic_operationally_required": True,
        "authority_digest": authority.authority_digest,
        "manifest_sha256": next(iter(artifacts.values())).manifest_sha256,
        "artifacts": {
            name: {
                "features_sha256": artifact.features_sha256,
                "extraction_receipt_sha256": artifact.receipt_sha256,
                "dimension": int(artifact.values.shape[1]),
            }
            for name, artifact in sorted(artifacts.items())
        },
    }


def _feature_suite_authority_payload(
    *,
    analysis: str,
    artifacts: dict[str, FeatureArtifact],
    analysis_manifest_path: Path,
    recomputed_manifest: pd.DataFrame,
    parent_bindings: dict[str, str],
    config: dict[str, Any],
    authority: AuthorityContext,
) -> dict[str, Any]:
    validate_feature_suite(
        artifacts,
        config=config,
        authority=authority,
        stage="feature_load",
    )
    physical = pd.read_parquet(analysis_manifest_path)
    try:
        pd.testing.assert_frame_equal(
            physical.reset_index(drop=True),
            recomputed_manifest.reset_index(drop=True),
            check_exact=True,
        )
    except AssertionError as error:
        raise ContractError(
            "cannot seal a feature suite against a non-recomputed manifest"
        ) from error
    return {
        "schema_version": 1,
        "status": "authenticated_external_feature_suite_authority",
        "analysis": analysis,
        "data_role": next(iter(artifacts.values())).data_role,
        "protocol_config_sha256": protocol_config_sha256(config),
        "authority_digest": authority.authority_digest,
        "analysis_manifest_sha256": sha256_file(analysis_manifest_path),
        "analysis_manifest_content_sha256": canonical_frame_sha256(
            recomputed_manifest,
            columns=recomputed_manifest.columns,
            sort_by=["row_id"],
        ),
        "parent_bindings": dict(sorted(parent_bindings.items())),
        "representations": {
            name: {
                "features_sha256": artifact.features_sha256,
                "extraction_receipt_sha256": artifact.receipt_sha256,
                "checkpoint_sha256": artifact.receipt["checkpoint_sha256"],
                "preprocessing_contract_sha256": artifact.receipt[
                    "preprocessing_contract_sha256"
                ],
                "extraction_contract_sha256": artifact.receipt[
                    "extraction_contract_sha256"
                ],
                "extractor_code_sha256": artifact.receipt["extractor_code_sha256"],
                "preprocessing_code_sha256": artifact.receipt[
                    "preprocessing_code_sha256"
                ],
                "row_ids_sha256": artifact.row_ids_sha256,
                "pixel_hashes_sha256": artifact.pixel_hashes_sha256,
                "dimension": int(artifact.values.shape[1]),
            }
            for name, artifact in sorted(artifacts.items())
        },
    }


def write_feature_suite_authority(
    *,
    path: Path,
    analysis: str,
    artifacts: dict[str, FeatureArtifact],
    analysis_manifest_path: Path,
    recomputed_manifest: pd.DataFrame,
    parent_bindings: dict[str, str],
    config: dict[str, Any],
    authority: AuthorityContext,
) -> dict[str, Any]:
    """Write the external authority after authenticated extraction completes."""
    if path.exists():
        raise ContractError("feature-suite authority refuses to overwrite")
    payload = _feature_suite_authority_payload(
        analysis=analysis,
        artifacts=artifacts,
        analysis_manifest_path=analysis_manifest_path,
        recomputed_manifest=recomputed_manifest,
        parent_bindings=parent_bindings,
        config=config,
        authority=authority,
    )
    atomic_write_json(payload, path)
    return payload


def load_feature_suite_authority(
    *,
    path: Path,
    artifacts: dict[str, FeatureArtifact],
    analysis_manifest_path: Path,
    recomputed_manifest: pd.DataFrame,
    parent_bindings: dict[str, str],
    analysis_authorization: ExecutionAuthorization,
    parent_artifacts: dict[str, Any],
    config: dict[str, Any],
    authority: AuthorityContext,
) -> FeatureSuiteAuthority:
    """Load an externally pinned suite and recompute every parent binding."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    observed_sha256 = sha256_file(path)
    analysis = str(payload.get("analysis"))
    if analysis not in {"G_domain", "G_match"}:
        raise ContractError("feature-suite authority has an invalid analysis")
    if analysis_authorization.data_role == "final":
        expected = config["execution"]["final_feature_suite_authority_sha256"].get(
            analysis
        )
        require_sha256(expected, context=f"final {analysis} feature-suite authority")
        if expected != observed_sha256:
            raise ContractError("final feature-suite authority is not config-pinned")
    analysis_authorization.revalidate(
        config=config,
        authority=authority,
        stage="probe_analysis",
        manifest_sha256=sha256_file(analysis_manifest_path),
        feature_suite_authority_sha256=observed_sha256,
    )
    suite = FeatureSuiteAuthority(
        analysis=analysis,
        data_role=str(payload.get("data_role")),
        artifacts=artifacts,
        analysis_manifest_path=analysis_manifest_path.resolve(),
        recomputed_manifest=recomputed_manifest.copy(),
        parent_bindings=dict(parent_bindings),
        payload=payload,
        path=path.resolve(),
        sha256=observed_sha256,
        analysis_authorization=analysis_authorization,
        parent_artifacts=dict(parent_artifacts),
    )
    suite.revalidate(config=config, authority=authority, stage="probe_analysis")
    return suite


def domain_parent_bindings(grec, subset) -> dict[str, str]:
    return {
        "G_rec_scene_manifest_sha256": grec.scene_sha256,
        "G_rec_host_manifest_sha256": grec.host_sha256,
        "G_rec_tractor_manifest_sha256": grec.tractor.sha256,
        "G_domain_subset_manifest_sha256": subset.manifest_sha256,
        "G_domain_subset_receipt_sha256": sha256_file(subset.receipt_path),
    }


def gmatch_parent_bindings(
    grec, frozen_matches, frozen_pair_plan, rendered
) -> dict[str, str]:
    return {
        "G_rec_scene_manifest_sha256": grec.scene_sha256,
        "G_rec_host_manifest_sha256": grec.host_sha256,
        "G_rec_tractor_manifest_sha256": grec.tractor.sha256,
        "G_match_manifest_sha256": frozen_matches.manifest_sha256,
        "G_match_pair_plan_sha256": frozen_pair_plan.manifest_sha256,
        "G_match_rendered_pairs_sha256": rendered.manifest_sha256,
    }


def recompute_domain_manifest(*, grec, subset, config, authority) -> pd.DataFrame:
    from .planning import build_gdomain_analysis_rows

    return build_gdomain_analysis_rows(
        grec,
        subset,
        config=config,
        authority=authority,
    )


def recompute_gmatch_manifest(
    *,
    grec,
    frozen_matches,
    frozen_pair_plan,
    rendered,
    config,
    authority,
) -> pd.DataFrame:
    from .planning import build_gmatch_analysis_rows

    return build_gmatch_analysis_rows(
        rendered,
        frozen_matches,
        frozen_pair_plan,
        grec,
        config=config,
        authority=authority,
    )
