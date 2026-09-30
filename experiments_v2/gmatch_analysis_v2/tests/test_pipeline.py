from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from stronglens_probes.contracts import ContractError as ProbeContractError
from stronglens_probes.dependencies import assert_dependency_closed_folds
from stronglens_probes.probes import crossfit_gmatch_axis

from gmatch_analysis_v2.contracts import (
    ContractError,
    canonical_frame_sha256,
    read_json,
    sequence_sha256,
    sha256_file,
)
from gmatch_analysis_v2.pipeline import (
    _AnalysisAuthority,
    _componentize_and_allocate,
    _validate_authorization_payload,
    _validate_manifest,
)
from gmatch_analysis_v2.verify import _pairwise_auc, _recompute

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]


def _config() -> dict:
    return read_json(PACKAGE_ROOT / "config/production.json")


def _hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _manifest() -> pd.DataFrame:
    rows = []
    for index in range(43):
        pair = f"pair-{index:02d}"
        reference = f"reference-{index:02d}"
        target = f"target-{index:02d}"
        for population in ("observed_reference", "bare", "injected"):
            rows.append(
                {
                    "row_id": f"{pair}:{population}",
                    "pair_id": pair,
                    "population": population,
                    "data_role": "final",
                    "reference_id": reference,
                    "target_galaxy_id": target,
                    "host_galaxy_id": f"host-{index:02d}",
                    "host_parent_id": f"parent-{index:02d}",
                    "physical_group_id": f"physical-{index:02d}",
                    "geographic_spatial_key": f"healpix128:{index}",
                    "spatial_group_id": f"spatial-{index:02d}",
                    "source_parameter_id": f"source-{index:02d}",
                    "lens_parameter_id": f"lens-{index:02d}",
                    "psf_id": f"psf-{index:02d}",
                    "rendering_id": f"rendering-{index:02d}",
                    "render_parameter_id": f"render-{index:02d}",
                    "random_draw_id": f"draw-{index:02d}",
                    "base_pixel_sha256": _hex(f"base-{index}"),
                    "pixel_sha256": _hex(f"pixel-{index}-{population}"),
                    "authority_digest": _hex("authority"),
                }
            )
    return pd.DataFrame(rows)


def _config_for_manifest(root: Path, frame: pd.DataFrame) -> dict:
    config = _config()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.parquet"
    frame.to_parquet(path, index=False)
    config["inputs"]["analysis_manifest"] = {
        "path": "manifest.parquet",
        "sha256": sha256_file(path),
    }
    config["manifest_contract"].update(
        {
            "authority_digest": _hex("authority"),
            "content_sha256": canonical_frame_sha256(
                frame,
                columns=frame.columns.tolist(),
                sort_by=["row_id"],
            ),
            "row_ids_sha256": sequence_sha256(frame["row_id"]),
            "pixel_hashes_sha256": sequence_sha256(frame["pixel_sha256"]),
            "reference_ids_sha256": sequence_sha256(sorted(set(frame["reference_id"]))),
            "pair_ids_sha256": sequence_sha256(sorted(set(frame["pair_id"]))),
        }
    )
    return config


def test_manifest_requires_exact_complete_triplets(tmp_path: Path) -> None:
    frame = _manifest()
    config = _config_for_manifest(tmp_path, frame)
    validated = _validate_manifest(tmp_path, config)
    assert len(validated) == 129

    broken = frame.copy()
    broken.loc[broken.index[-1], "target_galaxy_id"] = "other-target"
    broken_config = _config_for_manifest(tmp_path / "broken", broken)
    with pytest.raises(ContractError, match="exact bound triplet"):
        _validate_manifest(tmp_path / "broken", broken_config)


def test_fold_allocation_is_deterministic_and_triplet_closed() -> None:
    frame = _manifest()
    config = _config()
    first, first_folds = _componentize_and_allocate(frame, config)
    second, second_folds = _componentize_and_allocate(frame, config)
    pd.testing.assert_frame_equal(first_folds, second_folds, check_exact=True)
    assert first["dependency_component_id"].nunique() == 43
    attached = first.merge(first_folds, on="dependency_component_id")
    assert attached.groupby("pair_id")["fold"].nunique().max() == 1
    assert sorted(attached.groupby("fold")["pair_id"].nunique()) == [8, 8, 9, 9, 9]


def test_two_geographically_linked_triplets_form_one_valid_component() -> None:
    frame = _manifest()
    frame.loc[
        frame["pair_id"].eq("pair-00") & frame["population"].eq("observed_reference"),
        "geographic_spatial_key",
    ] = "healpix128:1"
    componentized, folds = _componentize_and_allocate(frame, _config())
    assert componentized["dependency_component_id"].nunique() == 42
    assert (
        componentized.groupby("pair_id")["dependency_component_id"]
        .nunique()
        .eq(1)
        .all()
    )
    attached = componentized.merge(folds, on="dependency_component_id")
    assert attached.groupby("dependency_component_id").size().max() == 6
    assert attached.groupby("dependency_component_id")["fold"].nunique().max() == 1


def test_adversarial_fold_crossing_is_rejected() -> None:
    frame = _manifest()
    config = _config()
    componentized, folds = _componentize_and_allocate(frame, config)
    forged = folds.copy()
    split = componentized.iloc[[0]].copy()
    original_component = str(split.iloc[0]["dependency_component_id"])
    split["dependency_component_id"] = "forged-component"
    original_fold = int(
        folds.set_index("dependency_component_id").loc[original_component, "fold"]
    )
    split["fold"] = (original_fold + 1) % 5
    forged = pd.concat(
        [forged, split[["dependency_component_id", "fold"]]],
        ignore_index=True,
    )
    altered = componentized.copy()
    altered.loc[altered.index[0], "dependency_component_id"] = "forged-component"
    with pytest.raises(ProbeContractError, match="dependency .* crosses folds"):
        assert_dependency_closed_folds(
            altered,
            forged,
            component_column="dependency_component_id",
            dependency_columns=config["probe"]["dependency_columns"],
        )


def _valid_authorization(config: dict, digest: str) -> dict:
    return {
        "status": "authorized_final_G_match_analysis",
        "authorization_kind": "analysis_only",
        "allowed_operations": [
            "read_authenticated_feature_matrices",
            "fit_registered_observed_vs_bare_axes",
            "score_held_out_triplets",
            "write_OOF_predictions",
            "write_final_result_receipt",
            "recompute_metrics_from_predictions",
        ],
        "result_contract_sha256": digest,
        "feature_suite_authority_sha256": config["inputs"]["feature_suite_authority"][
            "sha256"
        ],
        "feature_independent_validation_sha256": config["inputs"][
            "feature_independent_validation"
        ]["sha256"],
        "analysis_manifest_sha256": config["inputs"]["analysis_manifest"]["sha256"],
        "preparation_authority_sha256": config["inputs"]["preparation_authority"][
            "sha256"
        ],
        "feature_access_authorized": True,
        "probe_fitting_authorized": True,
        "fold_creation_authorized": True,
        "bootstrap_authorized": False,
        "detector_score_access_authorized": False,
        "detector_outcome_access_authorized": False,
        "G_rec_score_access_authorized": False,
        "manuscript_edit_authorized": False,
        "feature_bytes_opened_before_authorization": False,
        "analysis_producer_uuid": config["analysis_producer_uuid"],
        "metric_verifier_uuid": config["metric_verifier_uuid"],
        "feature_producer_uuid": config["feature_producer_uuid"],
        "feature_verifier_uuid": config["feature_verifier_uuid"],
        "independent_feature_pass_uuid": config["independent_feature_pass_uuid"],
        "upstream_source_closure": config["upstream_source_closure"],
        "feature_dependency_lock_sha256": config["inputs"]["feature_lock"]["sha256"],
        "config_sha256": digest,
        "config_seal_sha256": digest,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("feature_access_authorized", False),
        ("G_rec_score_access_authorized", True),
        ("bootstrap_authorized", True),
        ("result_contract_sha256", "b" * 64),
        ("allowed_operations", []),
        ("independent_feature_pass_uuid", "wrong-pass"),
    ],
)
def test_adversarial_authorization_is_rejected(field: str, value: object) -> None:
    config = _config()
    digest = "a" * 64
    authorization = _valid_authorization(config, digest)
    authorization[field] = value
    with pytest.raises(ContractError, match="authorization"):
        _validate_authorization_payload(
            authorization,
            config=config,
            expected_contract_sha256=digest,
            expected_config_sha256=digest,
            expected_config_seal_sha256=digest,
        )


def _synthetic_axis_inputs(
    tmp_path: Path,
) -> tuple[SimpleNamespace, pd.DataFrame, dict, _AnalysisAuthority, np.ndarray]:
    frame = _manifest()
    config = _config()
    componentized, folds = _componentize_and_allocate(frame, config)
    values = np.zeros((129, 2), dtype=np.float32)
    for row_index, row in componentized.iterrows():
        pair_index = int(str(row["pair_id"]).split("-")[1])
        base = pair_index / 20.0
        shift = {
            "bare": 0.0,
            "injected": 0.7,
            "observed_reference": 2.0,
        }[str(row["population"])]
        values[row_index] = [base + shift, np.sin(pair_index)]
    manifest_path = tmp_path / "manifest.parquet"
    features_path = tmp_path / "features.npy"
    componentized.to_parquet(manifest_path, index=False)
    np.save(features_path, values, allow_pickle=False)
    artifact = SimpleNamespace(
        representation_id="stein_ssl_resnet50",
        manifest_path=manifest_path,
        features_path=features_path,
        receipt_sha256="a" * 64,
        authority_digest=_hex("authority"),
    )
    authority = _AnalysisAuthority(componentized, _hex("authority"))
    return artifact, folds, config, authority, values


def test_axis_orientation_and_injected_exclusion(tmp_path: Path) -> None:
    artifact, folds, config, authority, values = _synthetic_axis_inputs(tmp_path)
    result, predictions, _ = crossfit_gmatch_axis(
        artifact,
        folds,
        config=config,
        authority=authority,
        _prevalidated=True,
    )
    assert result["primary_within_target_movement"] > 0.0
    assert result["diagnostics"]["pooled_observed_vs_bare_auc"] > 0.99
    assert all(scale["training_injected_rows"] == 0 for scale in result["fold_scales"])

    altered = values.copy()
    injected = pd.read_parquet(artifact.manifest_path)["population"].eq("injected")
    altered[injected.to_numpy()] += 100.0
    altered_path = tmp_path / "altered.npy"
    np.save(altered_path, altered, allow_pickle=False)
    altered_artifact = copy.copy(artifact)
    altered_artifact.features_path = altered_path
    _, altered_predictions, _ = crossfit_gmatch_axis(
        altered_artifact,
        folds,
        config=config,
        authority=authority,
        _prevalidated=True,
    )
    not_injected = predictions["population"].ne("injected")
    np.testing.assert_array_equal(
        predictions.loc[not_injected, "axis_score"].to_numpy(),
        altered_predictions.loc[not_injected, "axis_score"].to_numpy(),
    )


def test_independent_auc_formula_and_exact_concordance() -> None:
    assert _pairwise_auc(np.arange(43) + 100.0, np.arange(43)) == 1.0
    frame = _manifest()[
        [
            "row_id",
            "pair_id",
            "population",
            "reference_id",
            "target_galaxy_id",
        ]
    ]
    frame["dependency_component_id"] = frame["pair_id"]
    frame["fold"] = frame["pair_id"].str[-2:].astype(int) % 5
    frame["axis_score"] = frame["population"].map(
        {"bare": 0.0, "injected": 1.0, "observed_reference": 2.0}
    )
    metrics = _recompute(frame)
    assert metrics["primary_within_target_movement"] == 1.0
    assert metrics["diagnostics"]["paired_injected_above_bare_concordance"] == 1.0
    assert metrics["diagnostics"]["paired_counts"] == {
        "injected_above_bare": 43,
        "ties": 0,
        "injected_below_bare": 0,
        "total": 43,
    }


def test_registered_axis_source_is_byte_exact() -> None:
    config = _config()
    path = REPOSITORY_ROOT / "experiments_v2/probes/src/stronglens_probes/probes.py"
    assert sha256_file(path) == config["registered_implementation"]["sha256"]
