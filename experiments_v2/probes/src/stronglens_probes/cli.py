"""Pre-feature, pre-score command line interface for probe preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .authorities import load_authority_context
from .authorization import load_execution_authorization
from .contracts import (
    ContractError,
    atomic_write_json,
    atomic_write_parquet,
    canonical_json_sha256,
    load_config,
    sha256_file,
)
from .design_scale_benchmark import run_design_scale_refit_benchmark
from .execution import (
    aggregate_joint_bootstrap_shards,
    run_design_resampling_benchmark,
    run_joint_bootstrap_shard,
)
from .feature_extraction import extract_feature_artifact
from .features import (
    domain_parent_bindings,
    gmatch_parent_bindings,
    load_feature_artifact,
    load_feature_suite_authority,
    recompute_domain_manifest,
    recompute_gmatch_manifest,
    write_feature_suite_authority,
)
from .gdomain import build_gdomain_subset, load_gdomain_subset
from .matching import (
    load_frozen_matches,
    load_frozen_pair_plan,
    load_rendered_pairs,
    match_targets,
    validate_rendered_pairs,
)
from .parents import load_bound_tractor_rows, load_grec_artifacts
from .planning import (
    build_gdomain_analysis_rows,
    build_gmatch_pair_plan,
    build_pair_scene_links,
    load_pair_scene_links,
)
from .preparation import (
    FinalSchemaSourcePaths,
    validate_final_schema_adapters,
    write_final_schema_adapters,
)


def _development_ids(paths: list[Path]) -> set[str]:
    identifiers: set[str] = set()
    for path in paths:
        frame = pd.read_parquet(path, columns=["galaxy_id"])
        identifiers.update(frame["galaxy_id"].astype(str))
    return identifiers


def _authority(arguments: argparse.Namespace, config: dict[str, Any]):
    return load_authority_context(
        config=config,
        references_path=arguments.authority_references,
        reference_payloads_path=arguments.authority_reference_payloads,
        host_universe_path=arguments.authority_host_universe,
        reference_payload_adapter_path=getattr(
            arguments, "authority_reference_payload_adapter", None
        ),
        host_adapter_path=getattr(arguments, "authority_host_adapter", None),
        development_ids=_development_ids(arguments.development_manifest),
    )


def _grec(arguments: argparse.Namespace, config: dict[str, Any], authority):
    return load_grec_artifacts(
        scene_path=arguments.grec_scene_manifest,
        host_path=arguments.grec_host_manifest,
        host_tractor_path=arguments.host_tractor,
        config=config,
        authority=authority,
    )


def _preflight(arguments: argparse.Namespace) -> dict[str, Any]:
    config = load_config(arguments.config)
    authority = _authority(arguments, config)
    result = {
        "schema_version": 2,
        "status": "pre_score_authority_preflight_passed",
        "references": len(authority.reference_ids),
        "experiment_hosts": len(authority.host_payload_items),
        "reference_manifest_sha256": authority.reference_manifest_sha256,
        "reference_payload_manifest_sha256": (
            authority.reference_payload_manifest_sha256
        ),
        "host_universe_manifest_sha256": (authority.experiment_host_universe_sha256),
        "host_physical_authority_available": bool(authority.host_physical_items),
        "reference_psf_render_authority_available": bool(
            authority.reference_psf_items and authority.reference_render_items
        ),
        "bound_tractor_and_grec_hashes_available": all(
            config["authorities"].get(field)
            for field in (
                "reference_tractor_manifest_sha256",
                "experiment_host_tractor_manifest_sha256",
                "grec_host_manifest_sha256",
                "grec_scene_manifest_sha256",
            )
        ),
        "downstream_parent_ready": bool(
            authority.host_physical_items
            and authority.reference_psf_items
            and authority.reference_render_items
            and all(
                config["authorities"].get(field)
                for field in (
                    "reference_tractor_manifest_sha256",
                    "experiment_host_tractor_manifest_sha256",
                    "grec_host_manifest_sha256",
                    "grec_scene_manifest_sha256",
                )
            )
        ),
        "authority_digest": authority.authority_digest,
        "config_sha256": canonical_json_sha256(config),
        "development_overlap": 0,
        "score_columns_used": False,
        "feature_extraction_performed": False,
        "final_scoring_performed": False,
    }
    if arguments.output is not None:
        atomic_write_json(result, arguments.output)
    return result


def _build_domain(arguments: argparse.Namespace) -> dict[str, Any]:
    config = load_config(arguments.config)
    authority = _authority(arguments, config)
    grec = _grec(arguments, config, authority)
    subset, disposition = build_gdomain_subset(
        grec,
        config=config,
        authority=authority,
    )
    atomic_write_parquet(subset, arguments.output)
    atomic_write_parquet(disposition, arguments.disposition)
    result = {
        "schema_version": 2,
        "status": "pre_score_exact_G_domain_subset_complete",
        "parent_rows": len(grec.scenes),
        "subset_rows": len(subset),
        "subset_hosts": int(subset["host_galaxy_id"].nunique()),
        "predicate": config["G_domain"]["predicate"],
        "independently_rendered": False,
        "pixel_mismatches": 0,
        "G_rec_parent_sha256": grec.scene_sha256,
        "G_rec_host_sha256": grec.host_sha256,
        "G_rec_tractor_sha256": grec.tractor.sha256,
        "subset_sha256": sha256_file(arguments.output),
        "disposition_sha256": sha256_file(arguments.disposition),
        "feature_extraction_performed": False,
        "final_scoring_performed": False,
        "authority_digest": authority.authority_digest,
    }
    if arguments.receipt is not None:
        atomic_write_json(result, arguments.receipt)
    return result


def _schema_sources(arguments: argparse.Namespace) -> FinalSchemaSourcePaths:
    return FinalSchemaSourcePaths(
        repository_root=arguments.repository_root,
        reference_manifest=arguments.reference_manifest,
        reference_payload_manifest=arguments.reference_payload_manifest,
        reference_tractor_manifest=arguments.reference_tractor_manifest,
        reference_payload_root=arguments.reference_payload_root,
        host_universe_manifest=arguments.host_universe_manifest,
        final_selection_manifest=arguments.final_selection_manifest,
        final_selection_receipt=arguments.final_selection_receipt,
        host_tractor_manifest=arguments.host_tractor_manifest,
        grec_host_manifest=arguments.source_grec_host_manifest,
        grec_host_receipt=arguments.grec_host_receipt,
        grec_scene_manifest=arguments.source_grec_scene_manifest,
        grec_scene_receipt=arguments.grec_scene_receipt,
        grec_parent_manifest=arguments.source_grec_parent_manifest,
        grec_render_receipt=arguments.grec_render_receipt,
    )


def _write_schema_adapters(arguments: argparse.Namespace) -> dict[str, Any]:
    return write_final_schema_adapters(
        _schema_sources(arguments),
        arguments.output_dir,
    )


def _validate_schema_adapters(arguments: argparse.Namespace) -> dict[str, Any]:
    return validate_final_schema_adapters(
        _schema_sources(arguments),
        arguments.output_dir,
    )


def _prepare_domain(arguments: argparse.Namespace) -> dict[str, Any]:
    outputs = (
        arguments.subset,
        arguments.disposition,
        arguments.subset_receipt,
        arguments.analysis_manifest,
        arguments.receipt,
    )
    if any(path.exists() for path in outputs):
        raise ValueError("G_domain preparation refuses to overwrite outputs")
    config = load_config(arguments.config)
    authority = _authority(arguments, config)
    grec = _grec(arguments, config, authority)
    subset, disposition = build_gdomain_subset(
        grec,
        config=config,
        authority=authority,
    )
    atomic_write_parquet(subset, arguments.subset)
    atomic_write_parquet(disposition, arguments.disposition)
    subset_receipt = {
        "schema_version": 2,
        "status": "pre_score_exact_G_domain_subset_complete",
        "parent_rows": len(grec.scenes),
        "parent_hosts": int(grec.scenes["host_galaxy_id"].nunique()),
        "subset_rows": len(subset),
        "subset_hosts": int(subset["host_galaxy_id"].nunique()),
        "predicate": config["G_domain"]["predicate"],
        "independently_rendered": False,
        "pixel_mismatches": 0,
        "G_rec_parent_sha256": grec.scene_sha256,
        "G_rec_host_sha256": grec.host_sha256,
        "G_rec_tractor_sha256": grec.tractor.sha256,
        "subset_sha256": sha256_file(arguments.subset),
        "disposition_sha256": sha256_file(arguments.disposition),
        "feature_extraction_performed": False,
        "final_scoring_performed": False,
        "authority_digest": authority.authority_digest,
    }
    atomic_write_json(subset_receipt, arguments.subset_receipt)
    subset_artifact = load_gdomain_subset(
        path=arguments.subset,
        receipt_path=arguments.subset_receipt,
        grec=grec,
        config=config,
        authority=authority,
    )
    analysis = build_gdomain_analysis_rows(
        grec,
        subset_artifact,
        config=config,
        authority=authority,
    )
    data_roles = analysis["data_role"].astype(str).unique().tolist()
    if len(data_roles) != 1:
        raise ContractError("G_domain analysis manifest has mixed data roles")
    atomic_write_parquet(analysis, arguments.analysis_manifest)
    result = {
        "schema_version": 1,
        "status": "score_blind_G_domain_preparation_complete",
        "config_sha256": canonical_json_sha256(config),
        "authority_digest": authority.authority_digest,
        "reference_manifest_sha256": authority.reference_manifest_sha256,
        "reference_payload_source_manifest_sha256": (
            authority.reference_payload_manifest_sha256
        ),
        "reference_payload_adapter_manifest_sha256": (
            authority.reference_payload_adapter_manifest_sha256
        ),
        "experiment_host_universe_source_sha256": (
            authority.experiment_host_universe_sha256
        ),
        "experiment_host_adapter_manifest_sha256": (
            authority.experiment_host_adapter_manifest_sha256
        ),
        "grec_parent_adapter_manifest_sha256": grec.scene_sha256,
        "grec_host_adapter_manifest_sha256": grec.host_sha256,
        "host_tractor_adapter_manifest_sha256": grec.tractor.sha256,
        "subset_manifest_sha256": sha256_file(arguments.subset),
        "subset_receipt_sha256": sha256_file(arguments.subset_receipt),
        "disposition_manifest_sha256": sha256_file(arguments.disposition),
        "analysis_manifest_sha256": sha256_file(arguments.analysis_manifest),
        "parent_rows": len(grec.scenes),
        "parent_hosts": int(grec.scenes["host_galaxy_id"].nunique()),
        "subset_rows": len(subset),
        "subset_hosts": int(subset["host_galaxy_id"].nunique()),
        "analysis_rows": len(analysis),
        "analysis_reference_rows": int(
            analysis["population"].eq("observed_reference").sum()
        ),
        "analysis_reference_ids": int(
            analysis.loc[
                analysis["population"].eq("observed_reference"), "reference_id"
            ].nunique()
        ),
        "analysis_simulated_rows": int(analysis["population"].eq("simulated").sum()),
        "data_role": data_roles[0],
        "predicate": config["G_domain"]["predicate"],
        "parent_pixels_unchanged": True,
        "detector_scores_used": False,
        "detector_outcomes_used": False,
        "feature_extraction_performed": False,
        "stein_probe_run": False,
    }
    atomic_write_json(result, arguments.receipt)
    return result


def _validate_domain_preparation(arguments: argparse.Namespace) -> dict[str, Any]:
    config = load_config(arguments.config)
    authority = _authority(arguments, config)
    grec = _grec(arguments, config, authority)
    subset = load_gdomain_subset(
        path=arguments.subset,
        receipt_path=arguments.subset_receipt,
        grec=grec,
        config=config,
        authority=authority,
    )
    _, expected_disposition = build_gdomain_subset(
        grec,
        config=config,
        authority=authority,
    )
    observed_disposition = pd.read_parquet(arguments.disposition)
    try:
        pd.testing.assert_frame_equal(
            observed_disposition,
            expected_disposition,
            check_exact=True,
        )
    except AssertionError as error:
        raise ValueError(
            "G_domain disposition differs from exact reconstruction"
        ) from error
    expected_analysis = build_gdomain_analysis_rows(
        grec,
        subset,
        config=config,
        authority=authority,
    )
    observed_analysis = pd.read_parquet(arguments.analysis_manifest)
    try:
        pd.testing.assert_frame_equal(
            observed_analysis,
            expected_analysis,
            check_exact=True,
        )
    except AssertionError as error:
        raise ValueError(
            "G_domain analysis manifest differs from exact reconstruction"
        ) from error
    receipt = json.loads(arguments.receipt.read_text(encoding="utf-8"))
    expected_bindings = {
        "subset_manifest_sha256": sha256_file(arguments.subset),
        "subset_receipt_sha256": sha256_file(arguments.subset_receipt),
        "disposition_manifest_sha256": sha256_file(arguments.disposition),
        "analysis_manifest_sha256": sha256_file(arguments.analysis_manifest),
    }
    if receipt.get("status") != "score_blind_G_domain_preparation_complete":
        raise ValueError("G_domain preparation receipt is incomplete")
    for field, expected in expected_bindings.items():
        if receipt.get(field) != expected:
            raise ValueError(f"G_domain preparation receipt has wrong {field}")
    return {
        "status": "score_blind_G_domain_preparation_validated",
        "preparation_receipt_sha256": sha256_file(arguments.receipt),
        "parent_rows": len(grec.scenes),
        "parent_hosts": int(grec.scenes["host_galaxy_id"].nunique()),
        "subset_rows": len(subset.frame),
        "subset_hosts": int(subset.frame["host_galaxy_id"].nunique()),
        "analysis_rows": len(observed_analysis),
        "analysis_reference_rows": int(
            observed_analysis["population"].eq("observed_reference").sum()
        ),
        "detector_scores_used": False,
        "feature_extraction_performed": False,
    }


def _match(arguments: argparse.Namespace) -> dict[str, Any]:
    config = load_config(arguments.config)
    authority = _authority(arguments, config)
    reference_catalogue = load_bound_tractor_rows(
        path=arguments.reference_catalogue,
        authority_field="reference_tractor_manifest_sha256",
        config=config,
        authority=authority,
        population="reference",
    )
    host_universe = load_bound_tractor_rows(
        path=arguments.host_universe,
        authority_field="experiment_host_tractor_manifest_sha256",
        config=config,
        authority=authority,
        population="host",
    )
    result = match_targets(
        reference_catalogue,
        host_universe,
        _grec(arguments, config, authority),
        config=config,
        authority=authority,
    )
    atomic_write_parquet(result.matches, arguments.output)
    atomic_write_parquet(result.reference_disposition, arguments.reference_disposition)
    atomic_write_parquet(result.host_disposition, arguments.host_disposition)
    receipt = dict(result.receipt)
    matches_manifest_sha256 = sha256_file(arguments.output)
    receipt.update(
        {
            "status": "score_blind_match_complete_pre_render",
            "matches_manifest_sha256": matches_manifest_sha256,
            "inputs": {
                "reference_catalogue_sha256": sha256_file(
                    arguments.reference_catalogue
                ),
                "host_universe_sha256": sha256_file(arguments.host_universe),
            },
            "outputs": {
                "matches_sha256": sha256_file(arguments.output),
                "reference_disposition_sha256": sha256_file(
                    arguments.reference_disposition
                ),
                "host_disposition_sha256": sha256_file(arguments.host_disposition),
            },
            "feature_extraction_performed": False,
            "final_scoring_performed": False,
            "authority_digest": authority.authority_digest,
        }
    )
    atomic_write_json(receipt, arguments.receipt)
    return receipt


def _load_frozen_for_arguments(
    arguments: argparse.Namespace,
    *,
    config: dict[str, Any],
    authority,
):
    return load_frozen_matches(
        matches_path=arguments.matches,
        receipt_path=arguments.match_receipt,
        reference_tractor=load_bound_tractor_rows(
            path=arguments.reference_catalogue,
            authority_field="reference_tractor_manifest_sha256",
            config=config,
            authority=authority,
            population="reference",
        ),
        host_tractor=load_bound_tractor_rows(
            path=arguments.host_universe,
            authority_field="experiment_host_tractor_manifest_sha256",
            config=config,
            authority=authority,
            population="host",
        ),
        grec=_grec(arguments, config, authority),
        config=config,
        authority=authority,
    )


def _plan_pairs(arguments: argparse.Namespace) -> dict[str, Any]:
    config = load_config(arguments.config)
    authority = _authority(arguments, config)
    frozen = _load_frozen_for_arguments(arguments, config=config, authority=authority)
    grec = _grec(arguments, config, authority)
    if not arguments.pair_scene_links.exists():
        links = build_pair_scene_links(
            frozen,
            grec,
            config=config,
            authority=authority,
        )
        atomic_write_parquet(links, arguments.pair_scene_links)
        atomic_write_json(
            {
                "status": "complete_frozen_pair_scene_links",
                "manifest_sha256": sha256_file(arguments.pair_scene_links),
                "matches_manifest_sha256": frozen.manifest_sha256,
                "G_rec_scene_manifest_sha256": grec.scene_sha256,
                "authority_digest": authority.authority_digest,
            },
            arguments.pair_scene_links_receipt,
        )
    links = load_pair_scene_links(
        path=arguments.pair_scene_links,
        receipt_path=arguments.pair_scene_links_receipt,
        frozen_matches=frozen,
        grec=grec,
        config=config,
        authority=authority,
    )
    plan = build_gmatch_pair_plan(
        frozen,
        grec,
        links,
        config=config,
        authority=authority,
    )
    atomic_write_parquet(plan, arguments.output)
    receipt = {
        "schema_version": 1,
        "status": "complete_frozen_G_match_pair_plan",
        "pair_manifest_sha256": sha256_file(arguments.output),
        "matches_manifest_sha256": frozen.manifest_sha256,
        "pair_scene_links_sha256": sha256_file(arguments.pair_scene_links),
        "G_rec_scene_manifest_sha256": grec.scene_sha256,
        "G_rec_host_manifest_sha256": grec.host_sha256,
        "G_rec_tractor_manifest_sha256": grec.tractor.sha256,
        "authority_digest": authority.authority_digest,
        "rows": len(plan),
        "pairs": int(plan["pair_id"].nunique()),
        "fixed_intrinsic_per_band_flux": True,
        "host_total_flux_floor": None,
        "feature_extraction_performed": False,
        "final_scoring_performed": False,
    }
    atomic_write_json(receipt, arguments.receipt)
    return receipt


def _validate_rendered(arguments: argparse.Namespace) -> dict[str, Any]:
    config = load_config(arguments.config)
    authority = _authority(arguments, config)
    frozen = _load_frozen_for_arguments(arguments, config=config, authority=authority)
    grec = _grec(arguments, config, authority)
    pair_plan = load_frozen_pair_plan(
        pair_plan_path=arguments.pair_plan,
        receipt_path=arguments.pair_plan_receipt,
        frozen_matches=frozen,
        grec=grec,
        config=config,
        authority=authority,
    )
    rendered = pd.read_parquet(arguments.rendered_pairs)
    validate_rendered_pairs(
        rendered,
        frozen_matches=frozen,
        frozen_pair_plan=pair_plan,
        config=config,
        authority=authority,
    )
    receipt = {
        "schema_version": 1,
        "status": "complete_G_match_rendered_pair_validation",
        "rendered_pairs_sha256": sha256_file(arguments.rendered_pairs),
        "pair_manifest_sha256": pair_plan.manifest_sha256,
        "matches_manifest_sha256": frozen.manifest_sha256,
        "authority_digest": authority.authority_digest,
        "rows": len(rendered),
        "pairs": int(rendered["pair_id"].nunique()),
        "bare_host_payloads_exact": True,
        "fixed_intrinsic_flux_exact": True,
        "host_total_flux_floor": None,
    }
    atomic_write_json(receipt, arguments.receipt)
    return receipt


def _feature_suite(arguments: argparse.Namespace) -> dict[str, Any]:
    config = load_config(arguments.config)
    authority = _authority(arguments, config)
    authorization = load_execution_authorization(
        path=arguments.authorization_receipt,
        config=config,
        authority=authority,
    )
    artifacts = {}
    for representation in (
        "stein_ssl_resnet50",
        "imagenet_effnet_v2_s",
        "pixel_statistics",
    ):
        artifacts[representation] = load_feature_artifact(
            features_path=getattr(arguments, f"{representation}_features"),
            receipt_path=getattr(arguments, f"{representation}_receipt"),
            manifest_path=arguments.analysis_manifest,
            config=config,
            authority=authority,
            authorization=authorization,
            checkpoint_path=(
                None
                if representation == "pixel_statistics"
                else getattr(arguments, f"{representation}_checkpoint")
            ),
            expected_data_role=arguments.data_role,
        )
    grec = _grec(arguments, config, authority)
    if arguments.analysis == "G_domain":
        subset = load_gdomain_subset(
            path=arguments.gdomain_subset,
            receipt_path=arguments.gdomain_subset_receipt,
            grec=grec,
            config=config,
            authority=authority,
        )
        recomputed = recompute_domain_manifest(
            grec=grec,
            subset=subset,
            config=config,
            authority=authority,
        )
        bindings = domain_parent_bindings(grec, subset)
    else:
        frozen_matches = _load_frozen_for_arguments(
            arguments, config=config, authority=authority
        )
        pair_plan = load_frozen_pair_plan(
            pair_plan_path=arguments.pair_plan,
            receipt_path=arguments.pair_plan_receipt,
            frozen_matches=frozen_matches,
            grec=grec,
            config=config,
            authority=authority,
        )
        rendered = load_rendered_pairs(
            path=arguments.rendered_pairs,
            receipt_path=arguments.rendered_pairs_receipt,
            frozen_matches=frozen_matches,
            frozen_pair_plan=pair_plan,
            config=config,
            authority=authority,
        )
        recomputed = recompute_gmatch_manifest(
            grec=grec,
            frozen_matches=frozen_matches,
            frozen_pair_plan=pair_plan,
            rendered=rendered,
            config=config,
            authority=authority,
        )
        bindings = gmatch_parent_bindings(grec, frozen_matches, pair_plan, rendered)
    receipt = write_feature_suite_authority(
        path=arguments.output,
        analysis=arguments.analysis,
        artifacts=artifacts,
        analysis_manifest_path=arguments.analysis_manifest,
        recomputed_manifest=recomputed,
        parent_bindings=bindings,
        config=config,
        authority=authority,
    )
    return {
        **receipt,
        "feature_suite_authority_sha256": sha256_file(arguments.output),
    }


def _benchmark(arguments: argparse.Namespace) -> dict[str, Any]:
    config = load_config(arguments.config)
    return run_design_resampling_benchmark(
        config=config,
        output_path=arguments.output,
    )


def _benchmark_design_scale(arguments: argparse.Namespace) -> dict[str, Any]:
    config = load_config(arguments.config)
    return run_design_scale_refit_benchmark(
        work_path=arguments.work,
        output_path=arguments.output,
        draws=arguments.draws,
        seed=int(config["inference"]["seed"]),
    )


def _extract_features(arguments: argparse.Namespace) -> dict[str, Any]:
    config = load_config(arguments.config)
    authority = _authority(arguments, config)
    extraction_authorization = load_execution_authorization(
        path=arguments.extraction_authorization_receipt,
        config=config,
        authority=authority,
    )
    return extract_feature_artifact(
        representation_id=arguments.representation,
        manifest_path=arguments.analysis_manifest,
        pixel_root=arguments.pixel_root,
        checkpoint_path=arguments.checkpoint,
        output_path=arguments.output,
        receipt_path=arguments.receipt,
        config=config,
        authority=authority,
        authorization=extraction_authorization,
        batch_size=arguments.batch_size,
        device=arguments.device,
    )


def _bootstrap_shard(arguments: argparse.Namespace) -> dict[str, Any]:
    config = load_config(arguments.config)
    authority = _authority(arguments, config)
    extraction_authorization = load_execution_authorization(
        path=arguments.extraction_authorization_receipt,
        config=config,
        authority=authority,
    )
    suites = _load_analysis_suites(
        arguments,
        config=config,
        authority=authority,
        extraction_authorization=extraction_authorization,
    )
    return run_joint_bootstrap_shard(
        domain_suite=suites["domain"],
        domain_folds=pd.read_parquet(arguments.domain_folds),
        gmatch_suite=suites["gmatch"],
        gmatch_folds=pd.read_parquet(arguments.gmatch_folds),
        config=config,
        authority=authority,
        start=arguments.start,
        stop=arguments.stop,
        jobs=arguments.jobs,
        output_path=arguments.output,
        receipt_path=arguments.receipt,
        design_only=arguments.design_only,
    )


def _load_analysis_suites(
    arguments: argparse.Namespace,
    *,
    config: dict[str, Any],
    authority,
    extraction_authorization,
) -> dict[str, Any]:
    artifact_sets = {}
    for experiment in ("domain", "gmatch"):
        artifacts = {}
        for representation in (
            "stein_ssl_resnet50",
            "imagenet_effnet_v2_s",
            "pixel_statistics",
        ):
            artifacts[representation] = load_feature_artifact(
                features_path=getattr(
                    arguments, f"{experiment}_{representation}_features"
                ),
                receipt_path=getattr(
                    arguments, f"{experiment}_{representation}_receipt"
                ),
                manifest_path=getattr(arguments, f"{experiment}_manifest"),
                config=config,
                authority=authority,
                authorization=extraction_authorization,
                checkpoint_path=(
                    None
                    if representation == "pixel_statistics"
                    else getattr(
                        arguments,
                        f"{experiment}_{representation}_checkpoint",
                    )
                ),
                expected_data_role=arguments.data_role,
            )
        artifact_sets[experiment] = artifacts
    grec = _grec(arguments, config, authority)
    subset = load_gdomain_subset(
        path=arguments.gdomain_subset,
        receipt_path=arguments.gdomain_subset_receipt,
        grec=grec,
        config=config,
        authority=authority,
    )
    domain_manifest = recompute_domain_manifest(
        grec=grec,
        subset=subset,
        config=config,
        authority=authority,
    )
    frozen_matches = _load_frozen_for_arguments(
        arguments, config=config, authority=authority
    )
    frozen_pair_plan = load_frozen_pair_plan(
        pair_plan_path=arguments.pair_plan,
        receipt_path=arguments.pair_plan_receipt,
        frozen_matches=frozen_matches,
        grec=grec,
        config=config,
        authority=authority,
    )
    rendered = load_rendered_pairs(
        path=arguments.rendered_pairs,
        receipt_path=arguments.rendered_pairs_receipt,
        frozen_matches=frozen_matches,
        frozen_pair_plan=frozen_pair_plan,
        config=config,
        authority=authority,
    )
    gmatch_manifest = recompute_gmatch_manifest(
        grec=grec,
        frozen_matches=frozen_matches,
        frozen_pair_plan=frozen_pair_plan,
        rendered=rendered,
        config=config,
        authority=authority,
    )
    analysis_authorization = load_execution_authorization(
        path=arguments.analysis_authorization_receipt,
        config=config,
        authority=authority,
    )
    return {
        "domain": load_feature_suite_authority(
            path=arguments.domain_feature_suite_authority,
            artifacts=artifact_sets["domain"],
            analysis_manifest_path=arguments.domain_manifest,
            recomputed_manifest=domain_manifest,
            parent_bindings=domain_parent_bindings(grec, subset),
            analysis_authorization=analysis_authorization,
            parent_artifacts={"grec": grec, "subset": subset},
            config=config,
            authority=authority,
        ),
        "gmatch": load_feature_suite_authority(
            path=arguments.gmatch_feature_suite_authority,
            artifacts=artifact_sets["gmatch"],
            analysis_manifest_path=arguments.gmatch_manifest,
            recomputed_manifest=gmatch_manifest,
            parent_bindings=gmatch_parent_bindings(
                grec, frozen_matches, frozen_pair_plan, rendered
            ),
            analysis_authorization=analysis_authorization,
            parent_artifacts={
                "grec": grec,
                "frozen_matches": frozen_matches,
                "frozen_pair_plan": frozen_pair_plan,
                "rendered": rendered,
            },
            config=config,
            authority=authority,
        ),
    }


def _aggregate_shards(arguments: argparse.Namespace) -> dict[str, Any]:
    config = load_config(arguments.config)
    authority = _authority(arguments, config)
    extraction_authorization = load_execution_authorization(
        path=arguments.extraction_authorization_receipt,
        config=config,
        authority=authority,
    )
    suites = _load_analysis_suites(
        arguments,
        config=config,
        authority=authority,
        extraction_authorization=extraction_authorization,
    )
    if len(arguments.shard_output) != len(arguments.shard_receipt):
        raise ValueError("shard output and receipt counts differ")
    return aggregate_joint_bootstrap_shards(
        shard_files=list(
            zip(arguments.shard_output, arguments.shard_receipt, strict=True)
        ),
        domain_suite=suites["domain"],
        domain_folds=pd.read_parquet(arguments.domain_folds),
        gmatch_suite=suites["gmatch"],
        gmatch_folds=pd.read_parquet(arguments.gmatch_folds),
        config=config,
        authority=authority,
        jobs=arguments.jobs,
        output_path=arguments.output,
        receipt_path=arguments.receipt,
        design_only=arguments.design_only,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/probes.yaml"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_authorities(command: argparse.ArgumentParser) -> None:
        command.add_argument("--authority-references", type=Path, required=True)
        command.add_argument("--authority-reference-payloads", type=Path, required=True)
        command.add_argument("--authority-host-universe", type=Path, required=True)
        command.add_argument("--authority-reference-payload-adapter", type=Path)
        command.add_argument("--authority-host-adapter", type=Path)
        command.add_argument(
            "--development-manifest", type=Path, action="append", default=[]
        )

    def add_grec(command: argparse.ArgumentParser) -> None:
        command.add_argument("--grec-scene-manifest", type=Path, required=True)
        command.add_argument("--grec-host-manifest", type=Path, required=True)
        command.add_argument("--host-tractor", type=Path, required=True)

    def add_analysis_parents(command: argparse.ArgumentParser) -> None:
        add_grec(command)
        command.add_argument("--gdomain-subset", type=Path, required=True)
        command.add_argument("--gdomain-subset-receipt", type=Path, required=True)
        command.add_argument("--matches", type=Path, required=True)
        command.add_argument("--match-receipt", type=Path, required=True)
        command.add_argument("--reference-catalogue", type=Path, required=True)
        command.add_argument("--host-universe", type=Path, required=True)
        command.add_argument("--pair-plan", type=Path, required=True)
        command.add_argument("--pair-plan-receipt", type=Path, required=True)
        command.add_argument("--rendered-pairs", type=Path, required=True)
        command.add_argument("--rendered-pairs-receipt", type=Path, required=True)
        command.add_argument(
            "--domain-feature-suite-authority", type=Path, required=True
        )
        command.add_argument(
            "--gmatch-feature-suite-authority", type=Path, required=True
        )
        command.add_argument(
            "--extraction-authorization-receipt", type=Path, required=True
        )
        command.add_argument(
            "--analysis-authorization-receipt", type=Path, required=True
        )

    def add_schema_sources(command: argparse.ArgumentParser) -> None:
        command.add_argument("--repository-root", type=Path, required=True)
        command.add_argument("--reference-manifest", type=Path, required=True)
        command.add_argument(
            "--reference-payload-manifest", type=Path, required=True
        )
        command.add_argument(
            "--reference-tractor-manifest", type=Path, required=True
        )
        command.add_argument("--reference-payload-root", type=Path, required=True)
        command.add_argument("--host-universe-manifest", type=Path, required=True)
        command.add_argument("--final-selection-manifest", type=Path, required=True)
        command.add_argument("--final-selection-receipt", type=Path, required=True)
        command.add_argument("--host-tractor-manifest", type=Path, required=True)
        command.add_argument("--source-grec-host-manifest", type=Path, required=True)
        command.add_argument("--grec-host-receipt", type=Path, required=True)
        command.add_argument("--source-grec-scene-manifest", type=Path, required=True)
        command.add_argument("--grec-scene-receipt", type=Path, required=True)
        command.add_argument("--source-grec-parent-manifest", type=Path, required=True)
        command.add_argument("--grec-render-receipt", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)

    schema_adapters = subparsers.add_parser("write-final-schema-adapters")
    add_schema_sources(schema_adapters)
    schema_adapters.set_defaults(handler=_write_schema_adapters)

    validate_adapters = subparsers.add_parser("validate-final-schema-adapters")
    add_schema_sources(validate_adapters)
    validate_adapters.set_defaults(handler=_validate_schema_adapters)

    preflight = subparsers.add_parser("preflight")
    add_authorities(preflight)
    preflight.add_argument("--output", type=Path)
    preflight.set_defaults(handler=_preflight)

    domain = subparsers.add_parser("build-gdomain")
    add_authorities(domain)
    add_grec(domain)
    domain.add_argument("--output", type=Path, required=True)
    domain.add_argument("--disposition", type=Path, required=True)
    domain.add_argument("--receipt", type=Path)
    domain.set_defaults(handler=_build_domain)

    prepare_domain = subparsers.add_parser("prepare-gdomain")
    add_authorities(prepare_domain)
    add_grec(prepare_domain)
    prepare_domain.add_argument("--subset", type=Path, required=True)
    prepare_domain.add_argument("--disposition", type=Path, required=True)
    prepare_domain.add_argument("--subset-receipt", type=Path, required=True)
    prepare_domain.add_argument("--analysis-manifest", type=Path, required=True)
    prepare_domain.add_argument("--receipt", type=Path, required=True)
    prepare_domain.set_defaults(handler=_prepare_domain)

    validate_domain = subparsers.add_parser("validate-gdomain-preparation")
    add_authorities(validate_domain)
    add_grec(validate_domain)
    validate_domain.add_argument("--subset", type=Path, required=True)
    validate_domain.add_argument("--disposition", type=Path, required=True)
    validate_domain.add_argument("--subset-receipt", type=Path, required=True)
    validate_domain.add_argument("--analysis-manifest", type=Path, required=True)
    validate_domain.add_argument("--receipt", type=Path, required=True)
    validate_domain.set_defaults(handler=_validate_domain_preparation)

    matching = subparsers.add_parser("match-gmatch")
    add_authorities(matching)
    add_grec(matching)
    matching.add_argument("--reference-catalogue", type=Path, required=True)
    matching.add_argument("--host-universe", type=Path, required=True)
    matching.add_argument("--output", type=Path, required=True)
    matching.add_argument("--reference-disposition", type=Path, required=True)
    matching.add_argument("--host-disposition", type=Path, required=True)
    matching.add_argument("--receipt", type=Path, required=True)
    matching.set_defaults(handler=_match)

    pairs = subparsers.add_parser("plan-gmatch-pairs")
    add_authorities(pairs)
    add_grec(pairs)
    pairs.add_argument("--matches", type=Path, required=True)
    pairs.add_argument("--match-receipt", type=Path, required=True)
    pairs.add_argument("--reference-catalogue", type=Path, required=True)
    pairs.add_argument("--host-universe", type=Path, required=True)
    pairs.add_argument("--pair-scene-links", type=Path, required=True)
    pairs.add_argument("--pair-scene-links-receipt", type=Path, required=True)
    pairs.add_argument("--output", type=Path, required=True)
    pairs.add_argument("--receipt", type=Path, required=True)
    pairs.set_defaults(handler=_plan_pairs)

    rendered = subparsers.add_parser("validate-gmatch-rendered")
    add_authorities(rendered)
    add_grec(rendered)
    rendered.add_argument("--matches", type=Path, required=True)
    rendered.add_argument("--match-receipt", type=Path, required=True)
    rendered.add_argument("--reference-catalogue", type=Path, required=True)
    rendered.add_argument("--host-universe", type=Path, required=True)
    rendered.add_argument("--pair-plan", type=Path, required=True)
    rendered.add_argument("--pair-plan-receipt", type=Path, required=True)
    rendered.add_argument("--rendered-pairs", type=Path, required=True)
    rendered.add_argument("--receipt", type=Path, required=True)
    rendered.set_defaults(handler=_validate_rendered)

    features = subparsers.add_parser("validate-feature-suite")
    add_authorities(features)
    add_grec(features)
    features.add_argument("--analysis", choices=("G_domain", "G_match"), required=True)
    features.add_argument("--analysis-manifest", type=Path, required=True)
    features.add_argument("--authorization-receipt", type=Path, required=True)
    features.add_argument("--data-role", choices=("design", "final"), required=True)
    features.add_argument("--gdomain-subset", type=Path, required=True)
    features.add_argument("--gdomain-subset-receipt", type=Path, required=True)
    features.add_argument("--matches", type=Path, required=True)
    features.add_argument("--match-receipt", type=Path, required=True)
    features.add_argument("--reference-catalogue", type=Path, required=True)
    features.add_argument("--host-universe", type=Path, required=True)
    features.add_argument("--pair-plan", type=Path, required=True)
    features.add_argument("--pair-plan-receipt", type=Path, required=True)
    features.add_argument("--rendered-pairs", type=Path, required=True)
    features.add_argument("--rendered-pairs-receipt", type=Path, required=True)
    for representation in (
        "stein_ssl_resnet50",
        "imagenet_effnet_v2_s",
        "pixel_statistics",
    ):
        option = representation.replace("_", "-")
        features.add_argument(
            f"--{option}-features",
            dest=f"{representation}_features",
            type=Path,
            required=True,
        )
        if representation != "pixel_statistics":
            features.add_argument(
                f"--{option}-checkpoint",
                dest=f"{representation}_checkpoint",
                type=Path,
                required=True,
            )
        features.add_argument(
            f"--{option}-receipt",
            dest=f"{representation}_receipt",
            type=Path,
            required=True,
        )
    features.add_argument("--output", type=Path, required=True)
    features.set_defaults(handler=_feature_suite)

    benchmark = subparsers.add_parser("benchmark-resampling")
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.set_defaults(handler=_benchmark)

    scale_benchmark = subparsers.add_parser("benchmark-design-scale")
    scale_benchmark.add_argument("--work", type=Path, required=True)
    scale_benchmark.add_argument("--output", type=Path, required=True)
    scale_benchmark.add_argument("--draws", type=int, default=3)
    scale_benchmark.set_defaults(handler=_benchmark_design_scale)

    extraction = subparsers.add_parser("extract-features")
    add_authorities(extraction)
    extraction.add_argument(
        "--representation",
        choices=(
            "stein_ssl_resnet50",
            "imagenet_effnet_v2_s",
            "pixel_statistics",
        ),
        required=True,
    )
    extraction.add_argument("--analysis-manifest", type=Path, required=True)
    extraction.add_argument(
        "--authorization-receipt",
        dest="extraction_authorization_receipt",
        type=Path,
        required=True,
    )
    extraction.add_argument("--pixel-root", type=Path, required=True)
    extraction.add_argument("--checkpoint", type=Path)
    extraction.add_argument("--batch-size", type=int, default=32)
    extraction.add_argument("--device", default="cpu")
    extraction.add_argument("--output", type=Path, required=True)
    extraction.add_argument("--receipt", type=Path, required=True)
    extraction.set_defaults(handler=_extract_features)

    shard = subparsers.add_parser("run-bootstrap-shard")
    add_authorities(shard)
    add_analysis_parents(shard)
    shard.add_argument("--domain-manifest", type=Path, required=True)
    shard.add_argument("--domain-folds", type=Path, required=True)
    shard.add_argument("--gmatch-manifest", type=Path, required=True)
    shard.add_argument("--gmatch-folds", type=Path, required=True)
    shard.add_argument("--data-role", choices=("design", "final"), required=True)
    for experiment in ("domain", "gmatch"):
        for representation in (
            "stein_ssl_resnet50",
            "imagenet_effnet_v2_s",
            "pixel_statistics",
        ):
            option = f"{experiment}-{representation}".replace("_", "-")
            shard.add_argument(
                f"--{option}-features",
                dest=f"{experiment}_{representation}_features",
                type=Path,
                required=True,
            )
            if representation != "pixel_statistics":
                shard.add_argument(
                    f"--{option}-checkpoint",
                    dest=f"{experiment}_{representation}_checkpoint",
                    type=Path,
                    required=True,
                )
            shard.add_argument(
                f"--{option}-receipt",
                dest=f"{experiment}_{representation}_receipt",
                type=Path,
                required=True,
            )
    shard.add_argument("--start", type=int, required=True)
    shard.add_argument("--stop", type=int, required=True)
    shard.add_argument("--jobs", type=int, default=1)
    shard.add_argument("--design-only", action="store_true")
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument("--receipt", type=Path, required=True)
    shard.set_defaults(handler=_bootstrap_shard)

    aggregate = subparsers.add_parser("aggregate-bootstrap-shards")
    add_authorities(aggregate)
    add_analysis_parents(aggregate)
    aggregate.add_argument("--domain-manifest", type=Path, required=True)
    aggregate.add_argument("--domain-folds", type=Path, required=True)
    aggregate.add_argument("--gmatch-manifest", type=Path, required=True)
    aggregate.add_argument("--gmatch-folds", type=Path, required=True)
    aggregate.add_argument("--data-role", choices=("design", "final"), required=True)
    for experiment in ("domain", "gmatch"):
        for representation in (
            "stein_ssl_resnet50",
            "imagenet_effnet_v2_s",
            "pixel_statistics",
        ):
            option = f"{experiment}-{representation}".replace("_", "-")
            aggregate.add_argument(
                f"--{option}-features",
                dest=f"{experiment}_{representation}_features",
                type=Path,
                required=True,
            )
            aggregate.add_argument(
                f"--{option}-receipt",
                dest=f"{experiment}_{representation}_receipt",
                type=Path,
                required=True,
            )
            if representation != "pixel_statistics":
                aggregate.add_argument(
                    f"--{option}-checkpoint",
                    dest=f"{experiment}_{representation}_checkpoint",
                    type=Path,
                    required=True,
                )
    aggregate.add_argument("--shard-output", type=Path, action="append", required=True)
    aggregate.add_argument("--shard-receipt", type=Path, action="append", required=True)
    aggregate.add_argument("--jobs", type=int, default=1)
    aggregate.add_argument("--design-only", action="store_true")
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--receipt", type=Path, required=True)
    aggregate.set_defaults(handler=_aggregate_shards)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    result = arguments.handler(arguments)
    print(result)


if __name__ == "__main__":
    main()
