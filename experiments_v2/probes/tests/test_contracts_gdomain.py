from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import AuthorityBundle, digest

from stronglens_probes.authorization import (
    load_execution_authorization,
    protocol_config_sha256,
)
from stronglens_probes.contracts import (
    ContractError,
    assert_score_blind,
    canonical_row_sha256,
    representation_label,
    sha256_file,
    validate_config,
    validate_reference_payloads,
    validate_reference_roster,
)
from stronglens_probes.dependencies import (
    assert_dependency_closed_folds,
    assign_dependency_folds,
    build_dependency_components,
)
from stronglens_probes.gdomain import (
    build_gdomain_subset,
    cnn_recovery_cross_reference,
    load_gdomain_subset,
)


def _subset_artifact(bundle: AuthorityBundle, tmp_path: Path):
    subset, disposition = build_gdomain_subset(
        bundle.grec,
        config=bundle.config,
        authority=bundle.context,
    )
    path = tmp_path / "gdomain.parquet"
    receipt_path = tmp_path / "gdomain.json"
    subset.to_parquet(path, index=False)
    receipt = {
        "status": "pre_score_exact_G_domain_subset_complete",
        "subset_sha256": sha256_file(path),
        "G_rec_parent_sha256": bundle.grec.scene_sha256,
        "G_rec_host_sha256": bundle.grec.host_sha256,
        "G_rec_tractor_sha256": bundle.grec.tractor.sha256,
        "authority_digest": bundle.context.authority_digest,
    }
    receipt_path.write_text(json.dumps(receipt))
    return (
        load_gdomain_subset(
            path=path,
            receipt_path=receipt_path,
            grec=bundle.grec,
            config=bundle.config,
            authority=bundle.context,
        ),
        disposition,
    )


def _authorization(
    bundle: AuthorityBundle,
    tmp_path: Path,
    manifest_sha256: str,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "authorization.json"
    path.write_text(
        json.dumps(
            {
                "status": "authorized_design_analysis",
                "authorization_kind": "analysis",
                "data_role": "design",
                "allowed_stages": ["cnn_recovery", "recovery_inference"],
                "manifest_sha256s": [manifest_sha256],
                "protocol_config_sha256": protocol_config_sha256(bundle.config),
                "reference_manifest_sha256": (bundle.context.reference_manifest_sha256),
                "reference_payload_manifest_sha256": (
                    bundle.context.reference_payload_manifest_sha256
                ),
                "host_universe_manifest_sha256": (
                    bundle.context.experiment_host_universe_sha256
                ),
                "feature_suite_authority_sha256s": [],
                "result_contract_sha256s": ["f" * 64],
            }
        )
    )
    return load_execution_authorization(
        path=path,
        config=bundle.config,
        authority=bundle.context,
    )


def test_reference_and_hash_authorities_fail_closed(
    references: pd.DataFrame,
    config: dict,
    authority_bundle: AuthorityBundle,
) -> None:
    observed = validate_reference_roster(references, config=config)
    payloads = pd.DataFrame(
        {
            "galaxy_id": references["galaxy_id"],
            "payload_sha256": [f"{index:064x}" for index in range(43)],
            "valid": 1,
            "bucket": "positive",
        }
    )
    assert len(validate_reference_payloads(observed, payloads)) == 43
    contaminated = references.copy()
    contaminated["cnn_score"] = 0.0
    with pytest.raises(ContractError, match="outcome-bearing"):
        validate_reference_roster(contaminated, config=config)
    null = references.copy()
    null.loc[0, "physical_group_id"] = "nan"
    with pytest.raises(ContractError, match="null or empty|reserved missing"):
        validate_reference_roster(null, config=config)
    bad_payload = payloads.copy()
    bad_payload.loc[0, "payload_sha256"] = "invalid"
    with pytest.raises(ContractError, match="SHA-256"):
        validate_reference_payloads(observed, bad_payload)
    arbitrary = references.copy()
    arbitrary["galaxy_id"] = [f"arbitrary-{index}" for index in range(43)]
    with pytest.raises(ContractError, match="exact authority"):
        authority_bundle.context.assert_exact_references(
            arbitrary,
            id_column="galaxy_id",
            context="adversarial references",
        )


def test_config_locks_stein_primary_and_diagnostics(config: dict) -> None:
    assert (
        representation_label(config, "stein_ssl_resnet50", requested_role="primary")
        == "primary_stein_pretraining_image_overlap_unproven"
    )
    assert "diagnostic" in representation_label(
        config, "imagenet_effnet_v2_s", requested_role="diagnostic"
    )
    changed = copy.deepcopy(config)
    changed["shared_references"]["independent_replications"] = True
    with pytest.raises(ContractError, match="shared 43"):
        validate_config(changed)
    assert_score_blind(
        pd.DataFrame({"galaxy_id": ["x"], "sample_weight": [1.0]}),
        context="clean",
    )
    future = copy.deepcopy(config)
    future["execution"]["extraction_authorization_receipt_sha256"] = "a" * 64
    future["execution"]["analysis_authorization_receipt_sha256"] = "b" * 64
    future["execution"]["final_feature_suite_authority_sha256"] = {
        "G_domain": "c" * 64,
        "G_match": "d" * 64,
    }
    assert protocol_config_sha256(future) == protocol_config_sha256(config)


def test_gdomain_recomputes_raw_predicate_and_receipt(
    authority_bundle: AuthorityBundle,
    tmp_path: Path,
) -> None:
    artifact, disposition = _subset_artifact(authority_bundle, tmp_path)
    selected = set(artifact.frame["host_galaxy_id"])
    assert "host-000" in selected
    assert "host-001" not in selected
    assert "host-002" not in selected
    assert "host-003" not in selected
    indexed = disposition.set_index("galaxy_id")
    assert indexed.loc["host-000", "selected"]
    assert not indexed.loc["host-001", "selected"]

    changed = artifact.frame.copy()
    changed.loc[0, "final_image_sha256"] = digest("tampered")
    changed.to_parquet(artifact.path, index=False)
    receipt = dict(artifact.receipt)
    receipt["subset_sha256"] = sha256_file(artifact.path)
    artifact.receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ContractError, match="raw Tractor predicate|wrong subset"):
        load_gdomain_subset(
            path=artifact.path,
            receipt_path=artifact.receipt_path,
            grec=authority_bundle.grec,
            config=authority_bundle.config,
            authority=authority_bundle.context,
        )


def test_cnn_cross_reference_checks_all_parent_and_subset_fields(
    authority_bundle: AuthorityBundle,
    tmp_path: Path,
) -> None:
    subset, _ = _subset_artifact(authority_bundle, tmp_path)
    parent = authority_bundle.grec.scenes
    scores = parent.copy()
    scores["parent_manifest_sha256"] = authority_bundle.grec.scene_sha256
    scores["parent_row_sha256"] = [
        canonical_row_sha256(row, parent.columns)
        for row in parent.to_dict(orient="records")
    ]
    scores["cnn_logit"] = np.linspace(-1, 1, len(scores), dtype=np.float32)
    scores["scoring_receipt_sha256"] = "b" * 64
    scores["data_role"] = "design"
    scores_path = tmp_path / "scores.parquet"
    scores.to_parquet(scores_path, index=False)
    scores_sha256 = sha256_file(scores_path)
    authorization = _authorization(
        authority_bundle,
        tmp_path,
        scores_sha256,
    )
    summary, joined = cnn_recovery_cross_reference(
        authority_bundle.grec,
        subset,
        scores_path,
        config=authority_bundle.config,
        authority=authority_bundle.context,
        threshold=np.float32(0),
        scoring_receipt_sha256="b" * 64,
        authorization=authorization,
        design_only=True,
    )
    assert summary["all_parent_fields_compared"] is True
    assert joined["in_gdomain_subset"].any()
    tampered = scores.copy()
    tampered.loc[0, "scene_id"] = "wrong"
    tampered_path = tmp_path / "tampered-scores.parquet"
    tampered.to_parquet(tampered_path, index=False)
    tampered_authorization = _authorization(
        authority_bundle,
        tmp_path / "tampered",
        sha256_file(tampered_path),
    )
    with pytest.raises(ContractError, match="scene_id"):
        cnn_recovery_cross_reference(
            authority_bundle.grec,
            subset,
            tampered_path,
            config=authority_bundle.config,
            authority=authority_bundle.context,
            threshold=np.float32(0),
            scoring_receipt_sha256="b" * 64,
            authorization=tampered_authorization,
            design_only=True,
        )


def test_dependency_components_and_fold_canary() -> None:
    frame = pd.DataFrame(
        {
            "row_id": [f"row-{index}" for index in range(20)],
            "host_galaxy_id": [f"host-{index}" for index in range(20)],
            "source_parameter_id": [f"source-{index}" for index in range(20)],
            "label": [1] * 10 + [0] * 10,
        }
    )
    built = build_dependency_components(
        frame,
        row_id_column="row_id",
        dependency_columns=["host_galaxy_id", "source_parameter_id"],
    )
    folds = assign_dependency_folds(
        built,
        component_column="dependency_component_id",
        label_column="label",
        n_splits=5,
        seed=42,
    )
    assert_dependency_closed_folds(
        built,
        folds,
        component_column="dependency_component_id",
        dependency_columns=["host_galaxy_id", "source_parameter_id"],
    )
