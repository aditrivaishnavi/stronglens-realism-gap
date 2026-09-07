from __future__ import annotations

import hashlib
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
    canonical_frame_sha256,
    sha256_file,
)
from stronglens_probes.dependencies import assign_dependency_folds
from stronglens_probes.estimator_canary import run_estimator_coverage_canary
from stronglens_probes.execution import run_joint_bootstrap_shard
from stronglens_probes.feature_extraction import extract_feature_artifact
from stronglens_probes.features import (
    FeatureArtifact,
    feature_suite_receipt,
    load_feature_artifact,
    load_feature_suite_authority,
    representation_spec,
    validate_feature_suite,
    write_feature_suite_authority,
)
from stronglens_probes.inference import (
    bootstrap_recovery_cross_reference,
    iter_joint_resamples,
    joint_refit_bootstrap,
    stratified_dependency_support_canary,
)
from stronglens_probes.probes import (
    crossfit_domain_suite,
    crossfit_gmatch_axis,
    crossfit_gmatch_suite,
)


def _sequence_digest(values: pd.Series) -> str:
    hasher = hashlib.sha256()
    for value in values.astype(str):
        hasher.update(value.encode())
        hasher.update(b"\n")
    return hasher.hexdigest()


def _domain_data(
    bundle: AuthorityBundle,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(11)
    rows = []
    signal = []
    payloads = bundle.context.reference_payloads
    for index, reference in enumerate(bundle.context.reference_ids):
        rows.append(
            {
                "row_id": f"domain-reference-{index}",
                "population": "observed_reference",
                "reference_id": reference,
                "host_galaxy_id": "not_applicable:observed",
                "dependency_component_id": f"joint-ref-{index:02d}",
                "joint_dependency_id": f"joint-ref-{index:02d}",
                "pixel_sha256": payloads[reference],
                "physical_group_id": f"ref-physical-{index:02d}",
                "spatial_group_id": f"ref-spatial-{index:02d}",
                "psf_id": f"reference-psf-{index}",
                "render_parameter_id": f"reference-render-{index}",
                "authority_digest": bundle.context.authority_digest,
            }
        )
        signal.append(rng.normal(1.0, 0.4, size=6))
    for host in range(50):
        for scene in range(2):
            rows.append(
                {
                    "row_id": f"domain-sim-{host}-{scene}",
                    "population": "simulated",
                    "reference_id": "not_applicable:simulated",
                    "host_galaxy_id": f"host-{host:03d}",
                    "dependency_component_id": f"domain-host-{host:02d}",
                    "joint_dependency_id": f"domain-host-{host:02d}",
                    "pixel_sha256": digest(f"domain-sim-{host}-{scene}"),
                    "physical_group_id": f"host-physical-{host}",
                    "spatial_group_id": f"host-spatial-{host}",
                    "psf_id": f"host-psf-{host}",
                    "render_parameter_id": f"host-render-{host}",
                    "authority_digest": bundle.context.authority_digest,
                }
            )
            signal.append(rng.normal(-1.0, 0.4, size=6))
    manifest = pd.DataFrame(rows)
    manifest["data_role"] = "design"
    allocation = manifest[["dependency_component_id", "population"]].copy()
    allocation["label"] = allocation["population"].eq("observed_reference").astype(int)
    folds = assign_dependency_folds(
        allocation,
        component_column="dependency_component_id",
        label_column="label",
        n_splits=5,
        seed=42,
    )
    return manifest, np.asarray(signal, dtype=np.float32), folds


def _gmatch_data(
    bundle: AuthorityBundle,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(19)
    rows = []
    signal = []
    payloads = bundle.context.reference_payloads
    for index, reference in enumerate(bundle.context.reference_ids):
        pair = f"pair-{index:02d}"
        target = f"host-{index:03d}"
        component = f"joint-ref-{index:02d}"
        bare = rng.normal(-1.0, 0.25, size=6)
        for population, values, pixel in (
            (
                "observed_reference",
                rng.normal(1.0, 0.25, size=6),
                payloads[reference],
            ),
            ("bare", bare, bundle.context.host_payloads[target]),
            ("injected", bare + 0.7, digest(f"injected-{target}")),
        ):
            if population == "observed_reference":
                physical = f"ref-physical-{index:02d}"
                spatial = f"ref-spatial-{index:02d}"
            else:
                physical = f"host-physical-{index}"
                spatial = f"host-spatial-{index}"
            rows.append(
                {
                    "row_id": f"{pair}-{population}",
                    "pair_id": pair,
                    "population": population,
                    "reference_id": reference,
                    "target_galaxy_id": target,
                    "dependency_component_id": component,
                    "joint_dependency_id": component,
                    "pixel_sha256": pixel,
                    "physical_group_id": physical,
                    "spatial_group_id": spatial,
                    "psf_id": (
                        f"reference-psf-{index}"
                        if population == "observed_reference"
                        else f"host-psf-{index}"
                    ),
                    "render_parameter_id": (
                        f"reference-render-{index}"
                        if population == "observed_reference"
                        else f"host-render-{index}"
                    ),
                    "authority_digest": bundle.context.authority_digest,
                }
            )
            signal.append(values)
    manifest = pd.DataFrame(rows)
    manifest["data_role"] = "design"
    fitting = manifest[manifest["population"].isin(["observed_reference", "bare"])][
        ["dependency_component_id", "population"]
    ].copy()
    fitting["label"] = fitting["population"].eq("observed_reference").astype(int)
    folds = assign_dependency_folds(
        fitting,
        component_column="dependency_component_id",
        label_column="label",
        n_splits=5,
        seed=42,
    )
    return manifest, np.asarray(signal, dtype=np.float32), folds


def _feature_values(signal: np.ndarray, dimension: int) -> np.ndarray:
    values = np.zeros((len(signal), dimension), dtype=np.float32)
    values[:, : signal.shape[1]] += signal
    return values


def _design_authorization(
    tmp_path: Path,
    *,
    name: str,
    manifest_path: Path,
    bundle: AuthorityBundle,
    stages: list[str],
    feature_suite_authority_sha256s: list[str] | None = None,
):
    authorization_kind = (
        "extraction"
        if set(stages) <= {"feature_extract", "feature_load"}
        else "analysis"
    )
    representation_entries = [
        bundle.config["representations"]["primary"],
        *bundle.config["representations"]["required_diagnostics"],
    ]
    authorization_path = tmp_path / f"{name}-authorization.json"
    authorization_path.write_text(
        json.dumps(
            {
                "status": (
                    "authorized_design_extraction"
                    if authorization_kind == "extraction"
                    else "authorized_design_analysis"
                ),
                "authorization_kind": authorization_kind,
                "data_role": "design",
                "allowed_stages": stages,
                "manifest_sha256s": [sha256_file(manifest_path)],
                "protocol_config_sha256": protocol_config_sha256(bundle.config),
                "reference_manifest_sha256": (bundle.context.reference_manifest_sha256),
                "reference_payload_manifest_sha256": (
                    bundle.context.reference_payload_manifest_sha256
                ),
                "host_universe_manifest_sha256": (
                    bundle.context.experiment_host_universe_sha256
                ),
                "feature_suite_authority_sha256s": (
                    feature_suite_authority_sha256s or []
                ),
                "result_contract_sha256s": (
                    ["f" * 64] if authorization_kind == "analysis" else []
                ),
                "representation_bindings": (
                    {
                        entry["id"]: {
                            "checkpoint_sha256": entry["checkpoint_sha256"],
                            "extractor_code_sha256": entry["extractor_code_sha256"],
                            "preprocessing_code_sha256": entry[
                                "preprocessing_code_sha256"
                            ],
                            "preprocessing_contract_sha256": entry[
                                "preprocessing_contract_sha256"
                            ],
                            "extraction_contract_sha256": entry[
                                "extraction_contract_sha256"
                            ],
                        }
                        for entry in representation_entries
                    }
                    if authorization_kind == "extraction"
                    else None
                ),
            }
        )
    )
    return load_execution_authorization(
        path=authorization_path,
        config=bundle.config,
        authority=bundle.context,
    )


def _artifact(
    tmp_path: Path,
    *,
    name: str,
    manifest: pd.DataFrame,
    signal: np.ndarray,
    representation: str,
    bundle: AuthorityBundle,
) -> FeatureArtifact:
    specification = representation_spec(bundle.config, representation)
    manifest_path = tmp_path / f"{name}-manifest.parquet"
    features_path = tmp_path / f"{name}-{representation}.npy"
    receipt_path = tmp_path / f"{name}-{representation}.json"
    manifest.to_parquet(manifest_path, index=False)
    authorization = _design_authorization(
        tmp_path,
        name=name,
        manifest_path=manifest_path,
        bundle=bundle,
        stages=[
            "feature_load",
        ],
    )
    np.save(
        features_path,
        _feature_values(signal, int(specification["feature_dimension"])),
        allow_pickle=False,
    )
    receipt = {
        "schema_version": 2,
        "status": "complete",
        "data_role": "design",
        "representation_id": representation,
        "rows": len(manifest),
        "feature_dimension": int(specification["feature_dimension"]),
        "dtype": "float32",
        "features_file_sha256": sha256_file(features_path),
        "analysis_manifest_sha256": sha256_file(manifest_path),
        "analysis_manifest_content_sha256": canonical_frame_sha256(
            manifest,
            columns=manifest.columns,
            sort_by=["row_id"],
        ),
        "row_ids_sha256": _sequence_digest(manifest["row_id"]),
        "pixel_hashes_sha256": _sequence_digest(manifest["pixel_sha256"]),
        "authority_digest": bundle.context.authority_digest,
        "checkpoint_sha256": specification["checkpoint_sha256"],
        "preprocessing_contract_sha256": specification["preprocessing_contract_sha256"],
        "extraction_contract_sha256": specification["extraction_contract_sha256"],
        "extractor_code_sha256": "e" * 64,
        "preprocessing_code_sha256": "e" * 64,
        "authorization_receipt_sha256": authorization.receipt_sha256,
    }
    receipt["extractor_code_sha256"] = specification["extractor_code_sha256"]
    receipt["preprocessing_code_sha256"] = specification["preprocessing_code_sha256"]
    receipt_path.write_text(json.dumps(receipt))
    return load_feature_artifact(
        features_path=features_path,
        receipt_path=receipt_path,
        manifest_path=manifest_path,
        config=bundle.config,
        authority=bundle.context,
        authorization=authorization,
        checkpoint_path=bundle.checkpoint_paths.get(representation),
        expected_data_role="design",
    )


def _suite(
    tmp_path: Path,
    *,
    name: str,
    manifest: pd.DataFrame,
    signal: np.ndarray,
    bundle: AuthorityBundle,
) -> dict[str, FeatureArtifact]:
    return {
        representation: _artifact(
            tmp_path,
            name=name,
            manifest=manifest,
            signal=signal,
            representation=representation,
            bundle=bundle,
        )
        for representation in (
            "stein_ssl_resnet50",
            "imagenet_effnet_v2_s",
            "pixel_statistics",
        )
    }


def _suite_authority(
    tmp_path: Path,
    *,
    name: str,
    artifacts: dict[str, FeatureArtifact],
    bundle: AuthorityBundle,
):
    manifest_path = next(iter(artifacts.values())).manifest_path
    recomputed = pd.read_parquet(manifest_path)
    parent_bindings = {"design_parent_manifest_sha256": sha256_file(manifest_path)}
    path = tmp_path / f"{name}-suite-authority.json"
    write_feature_suite_authority(
        path=path,
        analysis="G_domain" if "domain" in name else "G_match",
        artifacts=artifacts,
        analysis_manifest_path=manifest_path,
        recomputed_manifest=recomputed,
        parent_bindings=parent_bindings,
        config=bundle.config,
        authority=bundle.context,
    )
    authorization = _design_authorization(
        tmp_path,
        name=f"{name}-analysis",
        manifest_path=manifest_path,
        bundle=bundle,
        stages=["probe_analysis", "joint_inference", "bootstrap"],
        feature_suite_authority_sha256s=[sha256_file(path)],
    )
    return load_feature_suite_authority(
        path=path,
        artifacts=artifacts,
        analysis_manifest_path=manifest_path,
        recomputed_manifest=recomputed,
        parent_bindings=parent_bindings,
        analysis_authorization=authorization,
        parent_artifacts={"design_recomputed_manifest": recomputed},
        config=bundle.config,
        authority=bundle.context,
    )


def test_feature_receipts_authenticate_dimensions_rows_pixels_and_suite(
    authority_bundle: AuthorityBundle,
    tmp_path: Path,
) -> None:
    manifest, signal, _ = _domain_data(authority_bundle)
    artifacts = _suite(
        tmp_path,
        name="domain",
        manifest=manifest,
        signal=signal,
        bundle=authority_bundle,
    )
    validate_feature_suite(
        artifacts,
        config=authority_bundle.config,
        authority=authority_bundle.context,
    )
    receipt = feature_suite_receipt(
        artifacts,
        config=authority_bundle.config,
        authority=authority_bundle.context,
    )
    assert receipt["primary_representation"] == "stein_ssl_resnet50"
    assert receipt["pixel_diagnostic_operationally_required"] is True
    assert receipt["artifacts"]["stein_ssl_resnet50"]["dimension"] == 2048
    assert receipt["artifacts"]["imagenet_effnet_v2_s"]["dimension"] == 1280
    assert receipt["artifacts"]["pixel_statistics"]["dimension"] == 18
    with pytest.raises(ContractError, match="pixel diagnostics"):
        validate_feature_suite(
            {
                key: value
                for key, value in artifacts.items()
                if key != "pixel_statistics"
            },
            config=authority_bundle.config,
            authority=authority_bundle.context,
        )

    bad_receipt = json.loads((tmp_path / "domain-stein_ssl_resnet50.json").read_text())
    bad_receipt["preprocessing_contract_sha256"] = "f" * 64
    (tmp_path / "bad-receipt.json").write_text(json.dumps(bad_receipt))
    with pytest.raises(ContractError, match="preprocessing"):
        load_feature_artifact(
            features_path=artifacts["stein_ssl_resnet50"].features_path,
            receipt_path=tmp_path / "bad-receipt.json",
            manifest_path=tmp_path / "domain-manifest.parquet",
            config=authority_bundle.config,
            authority=authority_bundle.context,
            authorization=artifacts["stein_ssl_resnet50"].authorization,
            checkpoint_path=authority_bundle.checkpoint_paths["stein_ssl_resnet50"],
            expected_data_role="design",
        )

    baseline_receipt = json.loads(
        (tmp_path / "domain-stein_ssl_resnet50.json").read_text()
    )
    mutations = {
        "feature_dimension": 2047,
        "checkpoint_sha256": "a" * 64,
        "extraction_contract_sha256": "b" * 64,
        "row_ids_sha256": "c" * 64,
        "pixel_hashes_sha256": "d" * 64,
        "extractor_code_sha256": "invalid",
    }
    for field, value in mutations.items():
        changed = dict(baseline_receipt)
        changed[field] = value
        path = tmp_path / f"bad-{field}.json"
        path.write_text(json.dumps(changed))
        with pytest.raises(ContractError):
            load_feature_artifact(
                features_path=artifacts["stein_ssl_resnet50"].features_path,
                receipt_path=path,
                manifest_path=tmp_path / "domain-manifest.parquet",
                config=authority_bundle.config,
                authority=authority_bundle.context,
                authorization=artifacts["stein_ssl_resnet50"].authorization,
                checkpoint_path=authority_bundle.checkpoint_paths["stein_ssl_resnet50"],
                expected_data_role="design",
            )

    stein = artifacts["stein_ssl_resnet50"]
    original_features = stein.features_path.read_bytes()
    original_receipt = stein.receipt_path.read_bytes()
    forged = np.load(stein.features_path)
    forged[0, 0] += 1.0
    np.save(stein.features_path, forged, allow_pickle=False)
    forged_receipt = json.loads(original_receipt)
    forged_receipt["features_file_sha256"] = sha256_file(stein.features_path)
    stein.receipt_path.write_text(json.dumps(forged_receipt))
    try:
        with pytest.raises(ContractError, match="changed after loading"):
            validate_feature_suite(
                artifacts,
                config=authority_bundle.config,
                authority=authority_bundle.context,
            )
    finally:
        stein.features_path.write_bytes(original_features)
        stein.receipt_path.write_bytes(original_receipt)

    checkpoint = authority_bundle.checkpoint_paths["stein_ssl_resnet50"]
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint.write_bytes(b"coordinated-checkpoint-forgery")
    try:
        with pytest.raises(ContractError, match="checkpoint"):
            validate_feature_suite(
                artifacts,
                config=authority_bundle.config,
                authority=authority_bundle.context,
            )
    finally:
        checkpoint.write_bytes(checkpoint_bytes)


def test_actual_pixel_extraction_binds_payloads_code_and_authorization(
    authority_bundle: AuthorityBundle,
    tmp_path: Path,
) -> None:
    pixel_root = tmp_path / "pixels"
    pixel_root.mkdir()
    rows = []
    for index in range(3):
        path = pixel_root / f"image-{index}.npz"
        image = np.full((101, 101, 3), index + 1, dtype=np.float32)
        if index == 2:
            native = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
            np.savez_compressed(path, expectation_chw=native)
            array_key = "expectation_chw"
            layout = "CHW"
        else:
            native = image
            np.savez_compressed(path, cutout=native)
            array_key = "cutout"
            layout = "HWC"
        rows.append(
            {
                "row_id": f"design-image-{index}",
                "pixel_sha256": hashlib.sha256(
                    memoryview(native).cast("B")
                ).hexdigest(),
                "pixel_file_sha256": sha256_file(path),
                "pixel_relpath": path.name,
                "pixel_array_key": array_key,
                "pixel_layout": layout,
                "authority_digest": authority_bundle.context.authority_digest,
                "data_role": "design",
            }
        )
    manifest = pd.DataFrame(rows)
    manifest_path = tmp_path / "extraction-manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    authorization = _design_authorization(
        tmp_path,
        name="extraction",
        manifest_path=manifest_path,
        bundle=authority_bundle,
        stages=["feature_extract", "feature_load"],
    )
    output = tmp_path / "pixel-features.npy"
    receipt = tmp_path / "pixel-features.json"
    extract_feature_artifact(
        representation_id="pixel_statistics",
        manifest_path=manifest_path,
        pixel_root=pixel_root,
        checkpoint_path=None,
        output_path=output,
        receipt_path=receipt,
        config=authority_bundle.config,
        authority=authority_bundle.context,
        authorization=authorization,
        batch_size=2,
        device="cpu",
    )
    artifact = load_feature_artifact(
        features_path=output,
        receipt_path=receipt,
        manifest_path=manifest_path,
        config=authority_bundle.config,
        authority=authority_bundle.context,
        authorization=authorization,
        checkpoint_path=None,
        expected_data_role="design",
    )
    assert artifact.values.shape == (3, 18)
    assert (
        artifact.receipt["extractor_code_sha256"]
        == (
            authority_bundle.config["representations"]["required_diagnostics"][1][
                "extractor_code_sha256"
            ]
        )
    )

    final_manifest = manifest.copy()
    final_manifest["data_role"] = "final"
    final_manifest_path = tmp_path / "unauthorized-final.parquet"
    final_manifest.to_parquet(final_manifest_path, index=False)
    design_authorization = _design_authorization(
        tmp_path,
        name="wrong-role",
        manifest_path=final_manifest_path,
        bundle=authority_bundle,
        stages=["feature_extract"],
    )
    with pytest.raises(ContractError, match="wrong data role"):
        extract_feature_artifact(
            representation_id="pixel_statistics",
            manifest_path=final_manifest_path,
            pixel_root=pixel_root,
            checkpoint_path=None,
            output_path=tmp_path / "forbidden.npy",
            receipt_path=tmp_path / "forbidden.json",
            config=authority_bundle.config,
            authority=authority_bundle.context,
            authorization=design_authorization,
            batch_size=2,
            device="cpu",
        )


def test_probe_suites_keep_stein_primary_and_injected_out_of_axis(
    authority_bundle: AuthorityBundle,
    tmp_path: Path,
) -> None:
    domain_manifest, domain_signal, domain_folds = _domain_data(authority_bundle)
    match_manifest, match_signal, match_folds = _gmatch_data(authority_bundle)
    domain_suite = _suite(
        tmp_path,
        name="domain",
        manifest=domain_manifest,
        signal=domain_signal,
        bundle=authority_bundle,
    )
    match_suite = _suite(
        tmp_path,
        name="match",
        manifest=match_manifest,
        signal=match_signal,
        bundle=authority_bundle,
    )
    domain_authority = _suite_authority(
        tmp_path,
        name="domain",
        artifacts=domain_suite,
        bundle=authority_bundle,
    )
    match_authority = _suite_authority(
        tmp_path,
        name="match",
        artifacts=match_suite,
        bundle=authority_bundle,
    )
    future = json.loads(json.dumps(authority_bundle.config))
    future["execution"]["extraction_authorization_receipt_sha256"] = domain_suite[
        "stein_ssl_resnet50"
    ].authorization.receipt_sha256
    future["execution"]["analysis_authorization_receipt_sha256"] = (
        domain_authority.analysis_authorization.receipt_sha256
    )
    future["execution"]["final_feature_suite_authority_sha256"] = {
        "G_domain": domain_authority.sha256,
        "G_match": match_authority.sha256,
    }
    future["execution"]["feature_extraction_authorized"] = True
    future["execution"]["final_analysis_authorized"] = True
    authority_bundle.context.assert_config(future)
    validate_feature_suite(
        domain_suite,
        config=future,
        authority=authority_bundle.context,
    )
    domain_authority.revalidate(
        config=future,
        authority=authority_bundle.context,
        stage="probe_analysis",
    )
    forged_authority_path = tmp_path / "forged-suite-authority.json"
    forged_payload = dict(domain_authority.payload)
    forged_payload["authority_digest"] = "f" * 64
    forged_authority_path.write_text(json.dumps(forged_payload))
    with pytest.raises(ContractError, match="does not permit"):
        load_feature_suite_authority(
            path=forged_authority_path,
            artifacts=domain_suite,
            analysis_manifest_path=domain_authority.analysis_manifest_path,
            recomputed_manifest=domain_authority.recomputed_manifest,
            parent_bindings=domain_authority.parent_bindings,
            analysis_authorization=domain_authority.analysis_authorization,
            parent_artifacts=domain_authority.parent_artifacts,
            config=authority_bundle.config,
            authority=authority_bundle.context,
        )
    domain = crossfit_domain_suite(
        domain_authority,
        domain_folds,
        config=authority_bundle.config,
        authority=authority_bundle.context,
    )
    match = crossfit_gmatch_suite(
        match_authority,
        match_folds,
        config=authority_bundle.config,
        authority=authority_bundle.context,
    )
    assert domain["stein_ssl_resnet50"][0]["primary_oof_auc"] > 0.98
    assert (
        domain["stein_ssl_resnet50"][0]["representation_result_label"]
        == "primary_stein_pretraining_image_overlap_unproven"
    )
    primary, predictions, pairs = match["stein_ssl_resnet50"]
    assert all(item["training_injected_rows"] == 0 for item in primary["fold_scales"])
    assert primary["pooled_auc_is_primary"] is False
    assert pairs["within_target_movement"].gt(0).mean() > 0.9

    altered_signal = match_signal.copy()
    altered_signal[match_manifest["population"].eq("injected").to_numpy()] += 100
    altered = _artifact(
        tmp_path,
        name="altered",
        manifest=match_manifest,
        signal=altered_signal,
        representation="stein_ssl_resnet50",
        bundle=authority_bundle,
    )
    altered_suite = dict(match_suite)
    altered_suite["stein_ssl_resnet50"] = altered
    altered_authority = _suite_authority(
        tmp_path,
        name="altered-match",
        artifacts=altered_suite,
        bundle=authority_bundle,
    )
    _, altered_predictions, _ = crossfit_gmatch_axis(
        altered,
        match_folds,
        config=authority_bundle.config,
        authority=authority_bundle.context,
        suite_authority=altered_authority,
    )
    not_injected = predictions["population"].ne("injected")
    assert predictions.loc[not_injected, "axis_score"].to_numpy() == pytest.approx(
        altered_predictions.loc[not_injected, "axis_score"].to_numpy(),
        abs=0.0,
    )


def test_positive_bayesian_resamples_preserve_support_and_shared_covariance(
    authority_bundle: AuthorityBundle,
    tmp_path: Path,
) -> None:
    domain_manifest, domain_signal, domain_folds = _domain_data(authority_bundle)
    match_manifest, match_signal, match_folds = _gmatch_data(authority_bundle)
    draw = next(
        iter_joint_resamples(
            domain_manifest,
            match_manifest,
            domain_folds,
            match_folds,
            draws=1,
            seed=7,
            config=authority_bundle.config,
            authority=authority_bundle.context,
        )
    )
    fold_map = domain_folds.set_index("dependency_component_id")["fold"].to_dict()
    for fold in range(5):
        assert (
            sum(
                weight
                for component, weight in (
                    draw.shared_reference_component_multipliers.items()
                )
                if fold_map[component] == fold
            )
            > 0
        )
    for index in range(43):
        component = f"joint-ref-{index:02d}"
        expected = draw.shared_reference_component_multipliers[component]
        assert draw.domain_component_multipliers[component] == expected
        assert draw.gmatch_component_multipliers[component] == expected

    matched_ids = {f"ref-{index:02d}" for index in range(33)}
    partial_match = match_manifest[
        match_manifest["reference_id"].isin(matched_ids)
    ].copy()
    partial_draw = next(
        iter_joint_resamples(
            domain_manifest,
            partial_match,
            domain_folds,
            match_folds,
            draws=1,
            seed=8,
            config=authority_bundle.config,
            authority=authority_bundle.context,
        )
    )
    assert len(partial_draw.shared_reference_component_multipliers) == 33
    for index in range(33, 43):
        component = f"joint-ref-{index:02d}"
        assert component in partial_draw.domain_component_multipliers
        assert component not in partial_draw.gmatch_component_multipliers

    domain_artifacts = _suite(
        tmp_path,
        name="domain-bootstrap",
        manifest=domain_manifest,
        signal=domain_signal,
        bundle=authority_bundle,
    )
    match_artifacts = _suite(
        tmp_path,
        name="match-bootstrap",
        manifest=match_manifest,
        signal=match_signal,
        bundle=authority_bundle,
    )
    domain_suite = _suite_authority(
        tmp_path,
        name="domain-bootstrap",
        artifacts=domain_artifacts,
        bundle=authority_bundle,
    )
    match_suite = _suite_authority(
        tmp_path,
        name="match-bootstrap",
        artifacts=match_artifacts,
        bundle=authority_bundle,
    )
    with pytest.raises(ContractError, match="pixel diagnostics"):
        write_feature_suite_authority(
            path=tmp_path / "invalid-suite.json",
            analysis="G_match",
            artifacts={
                key: value
                for key, value in match_artifacts.items()
                if key != "pixel_statistics"
            },
            analysis_manifest_path=match_artifacts["stein_ssl_resnet50"].manifest_path,
            recomputed_manifest=match_manifest,
            parent_bindings={"test": "f" * 64},
            config=authority_bundle.config,
            authority=authority_bundle.context,
        )
    summary, values = joint_refit_bootstrap(
        domain_suite,
        domain_folds,
        match_suite,
        match_folds,
        config=authority_bundle.config,
        authority=authority_bundle.context,
        draws=3,
        seed=7,
        design_only=True,
    )
    assert len(values) == 3
    assert summary["shared_reference_component_resamples"] is True
    assert summary["independent_replication_claim"] is False
    assert summary["formal_confidence_intervals"] is None
    assert summary["population_inference"] is False
    assert summary["significance_test"] is None
    assert values["minimum_shared_reference_multiplier"].gt(0).all()


def test_10k_support_and_coverage_canaries() -> None:
    support = stratified_dependency_support_canary(
        np.tile(np.arange(5), 40),
        np.repeat([0, 1], 100),
        draws=10_000,
        seed=20260915,
    )
    assert support["all_fold_class_strata_supported"] is True
    estimator = run_estimator_coverage_canary(
        benchmark_draws=50,
        oracle_studies=20,
        coverage_studies=10,
        posterior_draws=30,
        seed=20260917,
    )
    assert estimator["interval_decision"] in {
        "eligible_for_further_scientific_review",
        "decline_95pct_confidence_intervals_use_conservative_bounds",
    }
    assert estimator["full_refit_benchmark"]["draws"] == 50
    assert estimator["matching_no_common_support_rejected"] is True
    assert np.isfinite(estimator["full_refit_benchmark"]["joint_covariance"])


def test_recovery_bootstrap_validates_booleans_draws_and_support(
    authority_bundle: AuthorityBundle,
    tmp_path: Path,
) -> None:
    rows = []
    for group in range(10):
        for row in range(4):
            rows.append(
                {
                    "host_spatial_group_id": f"group-{group}",
                    "detected": (group + row) % 3 == 0,
                    "in_gdomain_subset": group < 8,
                    "authority_digest": authority_bundle.context.authority_digest,
                    "data_role": "design",
                }
            )
    frame = pd.DataFrame(rows)
    frame_path = tmp_path / "recovery.parquet"
    frame.to_parquet(frame_path, index=False)
    authorization = _design_authorization(
        tmp_path,
        name="recovery",
        manifest_path=frame_path,
        bundle=authority_bundle,
        stages=["recovery_inference"],
    )
    summary, values = bootstrap_recovery_cross_reference(
        frame_path,
        config=authority_bundle.config,
        authority=authority_bundle.context,
        draws=100,
        seed=5,
        design_only=True,
        authorization=authorization,
    )
    assert len(values) == 100
    assert summary["multiplier_scheme"] == "stratified_dependency_bootstrap"
    assert summary["formal_confidence_intervals"] is None
    assert summary["common_draws_for_full_and_subset"] is True
    invalid = frame.copy()
    invalid["detected"] = invalid["detected"].astype(int)
    invalid_path = tmp_path / "invalid-recovery.parquet"
    invalid.to_parquet(invalid_path, index=False)
    invalid_authorization = _design_authorization(
        tmp_path,
        name="invalid-recovery",
        manifest_path=invalid_path,
        bundle=authority_bundle,
        stages=["recovery_inference"],
    )
    with pytest.raises(ContractError, match="strict boolean"):
        bootstrap_recovery_cross_reference(
            invalid_path,
            config=authority_bundle.config,
            authority=authority_bundle.context,
            draws=100,
            seed=5,
            design_only=True,
            authorization=invalid_authorization,
        )
    with pytest.raises(ContractError, match="at least two"):
        bootstrap_recovery_cross_reference(
            frame_path,
            config=authority_bundle.config,
            authority=authority_bundle.context,
            draws=1,
            seed=5,
            design_only=True,
            authorization=authorization,
        )


def test_shard_execution_is_receipted_bounded_and_resumable(
    authority_bundle: AuthorityBundle,
    tmp_path: Path,
) -> None:
    domain_manifest, domain_signal, domain_folds = _domain_data(authority_bundle)
    match_manifest, match_signal, match_folds = _gmatch_data(authority_bundle)
    domain = _suite(
        tmp_path,
        name="domain-shard",
        manifest=domain_manifest,
        signal=domain_signal,
        bundle=authority_bundle,
    )
    match = _suite(
        tmp_path,
        name="match-shard",
        manifest=match_manifest,
        signal=match_signal,
        bundle=authority_bundle,
    )
    domain_suite = _suite_authority(
        tmp_path,
        name="domain-shard",
        artifacts=domain,
        bundle=authority_bundle,
    )
    match_suite = _suite_authority(
        tmp_path,
        name="match-shard",
        artifacts=match,
        bundle=authority_bundle,
    )
    output = tmp_path / "draws.parquet"
    receipt = tmp_path / "draws.json"
    first = run_joint_bootstrap_shard(
        domain_suite=domain_suite,
        domain_folds=domain_folds,
        gmatch_suite=match_suite,
        gmatch_folds=match_folds,
        config=authority_bundle.config,
        authority=authority_bundle.context,
        start=0,
        stop=1,
        jobs=1,
        output_path=output,
        receipt_path=receipt,
        design_only=True,
    )
    assert first["resumed"] is False
    assert first["memory_mapped_feature_bases"] is True
    second = run_joint_bootstrap_shard(
        domain_suite=domain_suite,
        domain_folds=domain_folds,
        gmatch_suite=match_suite,
        gmatch_folds=match_folds,
        config=authority_bundle.config,
        authority=authority_bundle.context,
        start=0,
        stop=1,
        jobs=1,
        output_path=output,
        receipt_path=receipt,
        design_only=True,
    )
    assert second["resumed"] is True
    forged_frame = pd.read_parquet(output)
    forged_frame.loc[0, "gdomain_oof_auc"] = max(
        0.0, forged_frame.loc[0, "gdomain_oof_auc"] - 0.01
    )
    forged_frame.to_parquet(output, index=False)
    forged_receipt = json.loads(receipt.read_text())
    forged_receipt["output_sha256"] = sha256_file(output)
    forged_receipt["content_sha256"] = canonical_frame_sha256(
        forged_frame,
        columns=forged_frame.columns,
        sort_by=["draw"],
    )
    receipt.write_text(json.dumps(forged_receipt))
    with pytest.raises(ContractError, match="deterministic recomputation"):
        run_joint_bootstrap_shard(
            domain_suite=domain_suite,
            domain_folds=domain_folds,
            gmatch_suite=match_suite,
            gmatch_folds=match_folds,
            config=authority_bundle.config,
            authority=authority_bundle.context,
            start=0,
            stop=1,
            jobs=1,
            output_path=output,
            receipt_path=receipt,
            design_only=True,
        )
    with pytest.raises(ContractError, match="jobs"):
        run_joint_bootstrap_shard(
            domain_suite=domain_suite,
            domain_folds=domain_folds,
            gmatch_suite=match_suite,
            gmatch_folds=match_folds,
            config=authority_bundle.config,
            authority=authority_bundle.context,
            start=1,
            stop=2,
            jobs=99,
            output_path=tmp_path / "bad.parquet",
            receipt_path=tmp_path / "bad.json",
            design_only=True,
        )
