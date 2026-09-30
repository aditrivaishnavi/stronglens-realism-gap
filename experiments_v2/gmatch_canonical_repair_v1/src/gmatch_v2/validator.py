"""Independent reconstruction of frozen score-blind G_match preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .contracts import (
    ContractError,
    array_sha256,
    atomic_write_json,
    load_config,
    require_columns,
    sha256_file,
)
from .preparation import OUTPUT_FILES, validate_config_seal
from .schemas import ALLOWLIST_CONTRACT_SHA256, assert_context_schema


def _digest(*parts: object) -> str:
    encoded = json.dumps(
        [str(part) for part in parts],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _flux_digest(g: object, r: object, z: object) -> str:
    encoded = json.dumps(
        {"g": float(g).hex(), "r": float(r).hex(), "z": float(z).hex()},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _derive(frame: pd.DataFrame, *, host: bool) -> pd.DataFrame:
    result = frame.copy()
    psfsize = "psfsize_r_tractor" if host else "psfsize_r"
    psfdepth = "psfdepth_r_tractor" if host else "psfdepth_r"
    flux = pd.to_numeric(result["flux_z"], errors="coerce").to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result["observed_zmag"] = np.where(
            flux > 0, 22.5 - 2.5 * np.log10(flux), np.nan
        )
    result["sersic_n"] = pd.to_numeric(result["sersic"], errors="coerce")
    ellipticity = np.hypot(
        pd.to_numeric(result["shape_e1"], errors="coerce").to_numpy(float),
        pd.to_numeric(result["shape_e2"], errors="coerce").to_numpy(float),
    )
    axis_ratio = (1.0 - ellipticity) / (1.0 + ellipticity)
    axis_ratio[~np.isfinite(axis_ratio) | (axis_ratio < 0.0) | (axis_ratio > 1.0)] = (
        np.nan
    )
    result["axis_ratio_q"] = axis_ratio
    result["psf_fwhm_r_arcsec"] = pd.to_numeric(result[psfsize], errors="coerce")
    depth = pd.to_numeric(result[psfdepth], errors="coerce").to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result["depth5_r_mag"] = np.where(
            depth > 0, 22.5 - 2.5 * np.log10(5.0 / np.sqrt(depth)), np.nan
        )
    return result


def _independent_matches(
    references: pd.DataFrame,
    hosts: pd.DataFrame,
    *,
    covariates: list[str],
    component_caliper: float,
    total_caliper: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reference = _derive(references, host=False)
    candidate_authority = _derive(hosts, host=True)
    reference = reference.loc[
        np.isfinite(reference[covariates].to_numpy(float)).all(axis=1)
    ].reset_index(drop=True)
    candidate = candidate_authority.loc[
        np.isfinite(candidate_authority[covariates].to_numpy(float)).all(axis=1)
    ].reset_index(drop=True)
    authority_matrix = candidate_authority[covariates].to_numpy(float)
    centre = []
    scale = []
    finite_counts = []
    missing_counts = []
    for index in range(len(covariates)):
        values = authority_matrix[:, index]
        finite = values[np.isfinite(values)]
        q25, q75 = np.quantile(finite, [0.25, 0.75], method="linear")
        centre.append(float(np.median(finite)))
        scale.append(float(q75 - q25))
        finite_counts.append(int(finite.size))
        missing_counts.append(int(len(values) - finite.size))
    centre = np.asarray(centre)
    scale = np.asarray(scale)
    host_matrix = candidate[covariates].to_numpy(float)
    differences = (
        reference[covariates].to_numpy(float)[:, None, :] - host_matrix[None, :, :]
    ) / scale
    distances = np.sqrt(np.square(differences).sum(axis=2))
    admissible = (np.abs(differences) <= component_caliper).all(axis=2) & (
        distances <= total_caliper
    )
    reference_order = np.asarray(
        [
            item[2]
            for item in sorted(
                (
                    _digest("G_match_v2:reference", value),
                    str(value),
                    index,
                )
                for index, value in enumerate(reference["galaxy_id"].astype(str))
            )
        ]
    )
    host_order = np.asarray(
        [
            item[2]
            for item in sorted(
                (
                    _digest("G_match_v2:host", value),
                    str(value),
                    index,
                )
                for index, value in enumerate(candidate["galaxy_id"].astype(str))
            )
        ]
    )
    ordered_distances = distances[np.ix_(reference_order, host_order)]
    ordered_admissible = admissible[np.ix_(reference_order, host_order)]
    n_reference, n_host = ordered_distances.shape
    penalty = (n_reference + 1) * (total_caliper + 1.0)
    cost = np.full((n_reference, n_host + n_reference), np.inf)
    cost[:, :n_host] = np.where(ordered_admissible, ordered_distances, np.inf)
    for index in range(n_reference):
        cost[index, n_host + index] = penalty
    rows, columns = linear_sum_assignment(cost)
    reconstructed = []
    for ordered_reference, ordered_host in zip(rows, columns, strict=True):
        if ordered_host >= n_host:
            continue
        reference_index = int(reference_order[ordered_reference])
        host_index = int(host_order[ordered_host])
        item: dict[str, Any] = {
            "reference_id": str(reference.iloc[reference_index]["galaxy_id"]),
            "target_galaxy_id": str(candidate.iloc[host_index]["galaxy_id"]),
            "distance": float(distances[reference_index, host_index]),
            "admissible_host_count": int(admissible[reference_index].sum()),
        }
        for covariate_index, covariate in enumerate(covariates):
            item[f"reference__{covariate}"] = float(
                reference.iloc[reference_index][covariate]
            )
            item[f"target__{covariate}"] = float(candidate.iloc[host_index][covariate])
            item[f"delta_standardised__{covariate}"] = float(
                differences[reference_index, host_index, covariate_index]
            )
        reconstructed.append(item)
    prior_centre = np.median(host_matrix, axis=0)
    prior_q25, prior_q75 = np.quantile(
        host_matrix, [0.25, 0.75], axis=0, method="linear"
    )
    prior_scale = prior_q75 - prior_q25
    prior_differences = (
        reference[covariates].to_numpy(float)[:, None, :] - host_matrix[None, :, :]
    ) / prior_scale
    prior_distances = np.sqrt(np.square(prior_differences).sum(axis=2))
    prior_admissible = (np.abs(prior_differences) <= component_caliper).all(axis=2) & (
        prior_distances <= total_caliper
    )
    prior_ordered_distances = prior_distances[np.ix_(reference_order, host_order)]
    prior_ordered_admissible = prior_admissible[np.ix_(reference_order, host_order)]
    prior_cost = np.full((n_reference, n_host + n_reference), np.inf)
    prior_cost[:, :n_host] = np.where(
        prior_ordered_admissible, prior_ordered_distances, np.inf
    )
    for index in range(n_reference):
        prior_cost[index, n_host + index] = penalty
    prior_rows, prior_columns = linear_sum_assignment(prior_cost)
    prior_reconstructed = []
    for ordered_reference, ordered_host in zip(prior_rows, prior_columns, strict=True):
        if ordered_host >= n_host:
            continue
        reference_index = int(reference_order[ordered_reference])
        host_index = int(host_order[ordered_host])
        prior_reconstructed.append(
            {
                "reference_id": str(reference.iloc[reference_index]["galaxy_id"]),
                "target_galaxy_id": str(candidate.iloc[host_index]["galaxy_id"]),
                "distance": float(prior_distances[reference_index, host_index]),
                "admissible_host_count": int(prior_admissible[reference_index].sum()),
            }
        )
    return (
        pd.DataFrame(reconstructed).sort_values(
            "reference_id", kind="mergesort", ignore_index=True
        ),
        {
            "centre": centre,
            "scale": scale,
            "admissible": admissible,
            "reference": reference,
            "candidate": candidate,
            "finite_counts": np.asarray(finite_counts),
            "missing_counts": np.asarray(missing_counts),
            "prior_centre": prior_centre,
            "prior_scale": prior_scale,
            "prior": pd.DataFrame(prior_reconstructed).sort_values(
                "reference_id", kind="mergesort", ignore_index=True
            ),
        },
    )


def independent_validate(
    *,
    package_root: Path,
    config_path: Path,
    documentation_path: Path,
) -> dict[str, Any]:
    """Recompute assignment and reopen every matched pair from direct sources."""
    config = load_config(config_path)
    root = Path(config["paths"]["repository_root"]).resolve()
    seal = validate_config_seal(
        package_root=package_root,
        config_path=config_path,
        documentation_path=documentation_path,
    )
    output = root / str(config["paths"]["output_root"])
    destination = output / "GMATCH_INDEPENDENT_VALIDATION.json"
    if destination.exists():
        raise ContractError("independent validation receipt already exists")
    preparation = json.loads(
        (output / "GMATCH_PREPARATION_RECEIPT.json").read_text(encoding="utf-8")
    )
    if (
        preparation.get("status") != "gmatch_v2_preparation_complete_pre_feature_review"
        or preparation.get("verdict") != "PASS"
        or preparation.get("feature_authorization_issued") is not False
    ):
        raise ContractError("preparation receipt is not valid for review")
    for name, binding in preparation["output_bindings"].items():
        path = output / str(binding["path"])
        if sha256_file(path) != binding["sha256"]:
            raise ContractError(f"output changed after preparation: {name}")
    expected_receipts = {
        "authority_receipt_sha256": "GMATCH_AUTHORITY_RECEIPT.json",
        "matching_receipt_sha256": "GMATCH_MATCH_RECEIPT.json",
        "pair_receipt_sha256": "GMATCH_PAIR_RECEIPT.json",
        "registration_sha256": "GMATCH_REGISTRATION.json",
        "strict_schema_allowlist_receipt_sha256": (
            "STRICT_SCHEMA_ALLOWLIST_RECEIPT.json"
        ),
    }
    for field, filename in expected_receipts.items():
        if preparation.get(field) != sha256_file(output / filename):
            raise ContractError(f"preparation has wrong {field}")

    source_paths = {
        name: root / str(entry["path"]) for name, entry in config["authorities"].items()
    }
    for name, path in source_paths.items():
        if sha256_file(path) != config["authorities"][name]["sha256"]:
            raise ContractError(f"source authority changed: {name}")
    dataframe_authorities = (
        "reference_roster",
        "reference_payload_manifest",
        "reference_tractor_manifest",
        "host_role_proposal",
        "final_grec_selection",
        "host_tractor_manifest",
        "grec_host_manifest",
        "grec_scene_manifest",
        "grec_parent_manifest",
        "train_manifest",
        "validation_manifest",
        "test_manifest",
    )
    source_frames = {
        name: pd.read_parquet(source_paths[name]) for name in dataframe_authorities
    }
    for name, frame in source_frames.items():
        assert_context_schema(frame, context=f"input.{name}")
    for name in (
        "reference_adapter",
        "host_adapter",
        "host_tractor_adapter",
        "parent_theta15_adapter",
    ):
        adapter = pd.read_parquet(output / OUTPUT_FILES[name])
        assert_context_schema(adapter, context=f"intermediate.{name}")
    references = source_frames["reference_tractor_manifest"]
    host_tractor = source_frames["host_tractor_manifest"]
    selection = source_frames["final_grec_selection"]
    roles = source_frames["host_role_proposal"]
    parents = source_frames["grec_parent_manifest"]
    role_hosts = roles.loc[roles["proposed_role"].eq("experiment_host")]
    host_ids = set(selection["galaxy_id"].astype(str))
    if (
        len(references) != 43
        or len(selection) != 4500
        or len(host_ids) != 4500
        or set(host_tractor["galaxy_id"].astype(str)) != host_ids
        or not host_ids.issubset(set(role_hosts["galaxy_id"].astype(str)))
    ):
        raise ContractError("independent source population reconstruction failed")
    reconstructed, diagnostics = _independent_matches(
        references,
        host_tractor,
        covariates=list(config["matching"]["covariates"]),
        component_caliper=float(config["matching"]["component_caliper_standardised"]),
        total_caliper=float(config["matching"]["total_distance_caliper"]),
    )
    observed_matches = pd.read_parquet(output / OUTPUT_FILES["matches"])
    assert_context_schema(observed_matches, context="output.matches")
    core_columns = list(reconstructed.columns)
    try:
        pd.testing.assert_frame_equal(
            observed_matches[core_columns].reset_index(drop=True),
            reconstructed.reset_index(drop=True),
            check_exact=True,
        )
    except AssertionError as error:
        raise ContractError(
            "independent maximum-cardinality matching differs"
        ) from error
    if observed_matches["target_galaxy_id"].duplicated().any():
        raise ContractError("independent validation found host replacement")
    expected_scaler = config["matching"]["expected_scaler"]
    covariates = list(config["matching"]["covariates"])
    for index, covariate in enumerate(covariates):
        observed_scaler = {
            "finite": int(diagnostics["finite_counts"][index]),
            "missing": int(diagnostics["missing_counts"][index]),
            "median": float(diagnostics["centre"][index]),
            "iqr": float(diagnostics["scale"][index]),
        }
        if observed_scaler != expected_scaler[covariate]:
            raise ContractError(f"independent scaler differs for {covariate}")
    prior = diagnostics["prior"].set_index("reference_id")
    governing = reconstructed.set_index("reference_id")
    expected_comparison = []
    for reference_id in sorted(governing.index):
        old = prior.loc[reference_id]
        new = governing.loc[reference_id]
        expected_comparison.append(
            {
                "reference_id": reference_id,
                "prior_target_galaxy_id": str(old["target_galaxy_id"]),
                "governing_target_galaxy_id": str(new["target_galaxy_id"]),
                "assignment_changed": (
                    str(old["target_galaxy_id"]) != str(new["target_galaxy_id"])
                ),
                "prior_distance": float(old["distance"]),
                "governing_distance": float(new["distance"]),
                "prior_admissible_host_count": int(old["admissible_host_count"]),
                "governing_admissible_host_count": int(new["admissible_host_count"]),
            }
        )
    comparison = pd.read_parquet(output / OUTPUT_FILES["scaler_comparison"])
    assert_context_schema(comparison, context="output.scaler_comparison")
    try:
        pd.testing.assert_frame_equal(
            comparison[list(expected_comparison[0])].reset_index(drop=True),
            pd.DataFrame(expected_comparison).reset_index(drop=True),
            check_exact=True,
        )
    except AssertionError as error:
        raise ContractError("prior scaler comparison differs") from error

    reference_disposition = pd.read_parquet(
        output / OUTPUT_FILES["reference_disposition"]
    )
    host_disposition = pd.read_parquet(output / OUTPUT_FILES["host_disposition"])
    assert_context_schema(reference_disposition, context="output.reference_disposition")
    assert_context_schema(host_disposition, context="output.host_disposition")
    if (
        len(reference_disposition) != 43
        or len(host_disposition) != 4500
        or int(reference_disposition["matched"].sum()) != len(reconstructed)
        or int(host_disposition["selected"].sum()) != len(reconstructed)
    ):
        raise ContractError("frozen dispositions have wrong cardinality")
    if (
        int((~host_disposition["matching_candidate_eligible"]).sum())
        != int(config["matching"]["expected_nonfinite_observed_zmag_hosts"])
        or not host_disposition["scaler_authority_member"].all()
        or not host_disposition.loc[
            ~host_disposition["matching_candidate_eligible"],
            "missing_covariates",
        ]
        .eq("observed_zmag")
        .all()
    ):
        raise ContractError("nonfinite host dispositions differ from policy")
    reference_support = {
        str(row.galaxy_id): int(diagnostics["admissible"][index].sum())
        for index, row in diagnostics["reference"].iterrows()
    }
    for row in reference_disposition.itertuples(index=False):
        if (
            bool(row.covariates_complete)
            and int(row.admissible_host_count)
            != (reference_support[str(row.galaxy_id)])
        ):
            raise ContractError("reference support count differs from reconstruction")

    parent_theta = parents.loc[
        pd.to_numeric(parents["theta_e_arcsec"], errors="coerce").eq(1.5)
    ].copy()
    if (
        len(parent_theta) != 4500
        or parent_theta["host_galaxy_id"].astype(str).nunique() != 4500
    ):
        raise ContractError("independent theta_E=1.5 reconstruction failed")
    parent_by_host = parent_theta.set_index(parent_theta["host_galaxy_id"].astype(str))
    links = pd.read_parquet(output / OUTPUT_FILES["pair_scene_links"])
    assert_context_schema(links, context="output.pair_scene_links")
    if len(links) != len(reconstructed):
        raise ContractError("pair-scene link cardinality is wrong")
    for row in links.itertuples(index=False):
        parent = parent_by_host.loc[str(row.target_galaxy_id)]
        if (
            float(row.theta_e_arcsec) != 1.5
            or str(row.grec_row_id) != str(parent["grec_row_id"])
            or str(row.scene_id) != str(parent["scene_id"])
        ):
            raise ContractError("pair-scene link differs from direct parent")

    plan = pd.read_parquet(output / OUTPUT_FILES["pair_plan"])
    pairs = pd.read_parquet(output / OUTPUT_FILES["rendered_pairs"])
    assert_context_schema(plan, context="output.pair_plan")
    assert_context_schema(pairs, context="output.rendered_pairs")
    if (
        len(plan) != 2 * len(reconstructed)
        or len(pairs) != len(plan)
        or set(plan["arm"]) != {"bare", "injected"}
    ):
        raise ContractError("pair plan cardinality is wrong")
    comparison_columns = [
        column for column in plan.columns if column != "pair_plan_row_sha256"
    ]
    try:
        pd.testing.assert_frame_equal(
            plan[comparison_columns].reset_index(drop=True),
            pairs[comparison_columns].reset_index(drop=True),
            check_exact=True,
        )
    except AssertionError as error:
        raise ContractError("zero-copy rows differ from pair plan") from error
    selected_parent = parent_theta.set_index(parent_theta["grec_row_id"].astype(str))
    selected_host = selection.set_index(selection["galaxy_id"].astype(str))
    bytes_reopened = 0
    for pair_id, frame in pairs.groupby("pair_id", sort=False):
        if len(frame) != 2 or set(frame["arm"]) != {"bare", "injected"}:
            raise ContractError(f"{pair_id} is not an exact pair")
        shared = [
            "reference_id",
            "target_galaxy_id",
            "grec_row_id",
            "scene_id",
            "host_parent_id",
            "host_pixel_sha256",
            "source_parameter_id",
            "native_source_parameter_id",
            "lens_parameter_id",
            "native_lens_parameter_id",
            "psf_id",
            "rendering_id",
            "render_parameter_id",
            "random_draw_id",
            "intrinsic_flux_g_nmgy",
            "intrinsic_flux_r_nmgy",
            "intrinsic_flux_z_nmgy",
            "source_flux_parameter_sha256",
        ]
        if any(frame[column].nunique(dropna=False) != 1 for column in shared):
            raise ContractError(f"{pair_id} changes a paired identity")
        parent = selected_parent.loc[str(frame["grec_row_id"].iloc[0])]
        host = selected_host.loc[str(frame["target_galaxy_id"].iloc[0])]
        bare = frame.loc[frame["arm"].eq("bare")].iloc[0]
        injected = frame.loc[frame["arm"].eq("injected")].iloc[0]
        bare_path = root / str(bare["pixel_relpath"])
        parent_path = root / str(injected["pixel_relpath"])
        if (
            sha256_file(bare_path) != str(bare["host_payload_sha256"])
            or str(bare["pixel_file_sha256"]) != str(bare["host_payload_sha256"])
            or sha256_file(parent_path) != str(parent["parent_artifact_sha256"])
            or str(injected["pixel_file_sha256"])
            != str(parent["parent_artifact_sha256"])
        ):
            raise ContractError(f"{pair_id} file bytes differ")
        with np.load(bare_path, allow_pickle=False) as archive:
            bare_hwc = np.asarray(archive["cutout"])
        with np.load(parent_path, allow_pickle=False) as archive:
            expectation = np.asarray(archive["expectation_chw"])
            injection = np.asarray(archive["injection_expectation_chw"])
        host_store_path = root / str(parent["host_store_path"])
        host_hwc = np.load(host_store_path, allow_pickle=False)
        host_chw = np.transpose(host_hwc, (2, 0, 1))
        if (
            array_sha256(bare_hwc) != str(parent["host_pixel_sha256"])
            or not np.array_equal(bare_hwc, host_hwc)
            or array_sha256(expectation) != str(parent["expectation_pixel_sha256"])
            or array_sha256(injection) != str(parent["injection_expectation_sha256"])
            or not np.array_equal(expectation, host_chw + injection)
        ):
            raise ContractError(f"{pair_id} array bytes or ancestry differ")
        direct_parent_identities = (
            ("target_galaxy_id", "host_galaxy_id"),
            ("scene_id", "scene_id"),
            ("host_parent_id", "host_parent_id"),
            ("host_spatial_group_id", "host_spatial_group_id"),
            ("host_payload_sha256", "host_payload_sha256"),
            ("host_pixel_sha256", "host_pixel_sha256"),
            ("native_source_parameter_id", "source_parameter_id"),
            ("source_parameter_id", "source_nuisance_id"),
            ("source_nuisance_id", "source_nuisance_id"),
            ("native_lens_parameter_id", "lens_parameter_id"),
            ("lens_parameter_id", "lens_nuisance_id"),
            ("lens_nuisance_id", "lens_nuisance_id"),
            ("base_draw_key", "base_draw_key"),
            ("rendering_id", "base_draw_key"),
            ("render_parameter_id", "base_draw_key"),
            ("random_draw_id", "base_draw_key"),
            ("render_config_id", "render_config_id"),
            ("render_contract_sha256", "render_contract_sha256"),
            ("parameter_row_sha256", "parameter_row_sha256"),
            ("parent_artifact_file_sha256", "parent_artifact_sha256"),
            ("parent_expectation_pixel_sha256", "expectation_pixel_sha256"),
            ("injection_expectation_sha256", "injection_expectation_sha256"),
        )
        for output_column, parent_column in direct_parent_identities:
            if str(frame[output_column].iloc[0]) != str(parent[parent_column]):
                raise ContractError(f"{pair_id} changed {output_column}")
        expected_psf_id = (
            "grec_psf_"
            + _digest(
                "grec_psf_v1",
                parent["host_galaxy_id"],
                float(parent["psf_fwhm_g_arcsec"]).hex(),
                float(parent["psf_fwhm_r_arcsec"]).hex(),
                float(parent["psf_fwhm_z_arcsec"]).hex(),
            )[:40]
        )
        if (
            str(frame["host_physical_group_id"].iloc[0])
            != str(host["physical_group_id"])
            or not frame["proposed_role"].eq("experiment_host").all()
            or str(frame["psf_id"].iloc[0]) != expected_psf_id
            or str(frame["source_flux_parameter_sha256"].iloc[0])
            != _flux_digest(
                parent["source_flux_nmgy_g"],
                parent["source_flux_nmgy_r"],
                parent["source_flux_nmgy_z"],
            )
        ):
            raise ContractError(f"{pair_id} changed group, role, PSF, or flux identity")
        for output_column, parent_column in (
            ("intrinsic_flux_g_nmgy", "source_flux_nmgy_g"),
            ("intrinsic_flux_r_nmgy", "source_flux_nmgy_r"),
            ("intrinsic_flux_z_nmgy", "source_flux_nmgy_z"),
        ):
            if float(frame[output_column].iloc[0]) != float(parent[parent_column]):
                raise ContractError(f"{pair_id} changed {output_column}")
        bytes_reopened += bare_path.stat().st_size + parent_path.stat().st_size

    analysis = pd.read_parquet(output / OUTPUT_FILES["analysis_manifest"])
    assert_context_schema(analysis, context="output.analysis_manifest")
    observed = analysis.loc[analysis["population"].eq("observed_reference")]
    bare_rows = analysis.loc[analysis["population"].eq("bare")]
    injected_rows = analysis.loc[analysis["population"].eq("injected")]
    if (
        len(observed) != 43
        or set(observed["reference_id"].astype(str))
        != set(references["galaxy_id"].astype(str))
        or len(bare_rows) != len(reconstructed)
        or len(injected_rows) != len(reconstructed)
        or set(analysis["data_role"]) != {"final"}
    ):
        raise ContractError("analysis manifest population reconstruction failed")
    reference_lookup = references.set_index(references["galaxy_id"].astype(str))
    for row in observed.itertuples(index=False):
        source = reference_lookup.loc[str(row.reference_id)]
        expected_path = (
            root
            / "evaluation_dataset_prep/candidate_payloads"
            / str(source["local_relpath"])
        )
        if (
            (root / str(row.pixel_relpath)).resolve() != expected_path.resolve()
            or str(row.pixel_file_sha256) != str(source["payload_sha256"])
            or str(row.pixel_array_key) != "cutout"
            or str(row.pixel_layout) != "HWC"
        ):
            raise ContractError(
                f"observed reference binding changed: {row.reference_id}"
            )
    pair_lookup = pairs.set_index(["pair_id", "arm"])
    analysis_to_pair = {
        "reference_id": "reference_id",
        "target_galaxy_id": "target_galaxy_id",
        "host_parent_id": "host_parent_id",
        "physical_group_id": "host_physical_group_id",
        "spatial_group_id": "host_spatial_group_id",
        "source_parameter_id": "source_parameter_id",
        "native_source_parameter_id": "native_source_parameter_id",
        "source_nuisance_id": "source_nuisance_id",
        "lens_parameter_id": "lens_parameter_id",
        "native_lens_parameter_id": "native_lens_parameter_id",
        "lens_nuisance_id": "lens_nuisance_id",
        "psf_id": "psf_id",
        "rendering_id": "rendering_id",
        "render_parameter_id": "render_parameter_id",
        "random_draw_id": "random_draw_id",
        "base_draw_key": "base_draw_key",
        "pixel_relpath": "pixel_relpath",
        "pixel_file_sha256": "pixel_file_sha256",
        "pixel_sha256": "pixel_sha256",
        "pixel_array_key": "pixel_array_key",
        "pixel_layout": "pixel_layout",
        "source_flux_parameter_sha256": "source_flux_parameter_sha256",
        "parent_artifact_file_sha256": "parent_artifact_file_sha256",
        "parent_expectation_pixel_sha256": "parent_expectation_pixel_sha256",
    }
    for row in analysis.loc[
        analysis["population"].isin(["bare", "injected"])
    ].itertuples(index=False):
        parent_row = pair_lookup.loc[(str(row.pair_id), str(row.population))]
        for analysis_column, pair_column in analysis_to_pair.items():
            if str(getattr(row, analysis_column)) != str(parent_row[pair_column]):
                raise ContractError(
                    f"analysis row {row.row_id} changed {analysis_column}"
                )
    for row in analysis.itertuples(index=False):
        pixel_path = root / str(row.pixel_relpath)
        if sha256_file(pixel_path) != str(row.pixel_file_sha256):
            raise ContractError(f"analysis pixel file changed: {row.row_id}")
        with np.load(pixel_path, allow_pickle=False) as archive:
            pixels = np.asarray(archive[str(row.pixel_array_key)])
        if array_sha256(pixels) != str(row.pixel_sha256):
            raise ContractError(f"analysis pixel array changed: {row.row_id}")
    dependency_columns = {
        "physical_group_id",
        "spatial_group_id",
        "reference_id",
        "host_galaxy_id",
        "host_parent_id",
        "source_parameter_id",
        "lens_parameter_id",
        "psf_id",
        "rendering_id",
        "render_parameter_id",
        "random_draw_id",
        "base_pixel_sha256",
        "pixel_sha256",
    }
    require_columns(analysis, dependency_columns, context="analysis manifest")
    if any(
        analysis[column].isna().any()
        or analysis[column].astype(str).str.strip().eq("").any()
        for column in dependency_columns
    ):
        raise ContractError("analysis manifest has incomplete dependency identities")
    balance = pd.read_parquet(output / OUTPUT_FILES["balance"])
    assert_context_schema(balance, context="output.balance")
    balance_lookup = balance.set_index("covariate")
    for index, covariate in enumerate(covariates):
        row = balance_lookup.loc[covariate]
        if (
            int(row["scaler_finite_count"]) != int(diagnostics["finite_counts"][index])
            or int(row["scaler_missing_count"])
            != int(diagnostics["missing_counts"][index])
            or float(row["host_universe_median"]) != float(diagnostics["centre"][index])
            or float(row["host_universe_iqr"]) != float(diagnostics["scale"][index])
            or str(row["scaler_policy"])
            != "component_wise_all_finite_values_exact_authorized_roster"
        ):
            raise ContractError(f"balance scaler fields differ for {covariate}")
    schema_receipt = json.loads(
        (output / "STRICT_SCHEMA_ALLOWLIST_RECEIPT.json").read_text(encoding="utf-8")
    )
    if (
        schema_receipt.get("status") != "strict_context_schema_allowlists_passed"
        or schema_receipt.get("allowlist_contract_sha256") != ALLOWLIST_CONTRACT_SHA256
    ):
        raise ContractError("strict schema allowlist receipt is invalid")

    receipt = {
        "schema_version": 1,
        "status": "gmatch_v2_independent_pre_feature_validation_complete",
        "verdict": "PASS",
        "preparation_receipt_sha256": sha256_file(
            output / "GMATCH_PREPARATION_RECEIPT.json"
        ),
        "config_seal_sha256": sha256_file(package_root / "config/CONFIG_SEAL.json"),
        "source_tree_sha256": seal["source_tree_sha256"],
        "dependency_lock_sha256": seal["dependency_lock_sha256"],
        "strict_schema_allowlist_contract_sha256": (ALLOWLIST_CONTRACT_SHA256),
        "scaler_value_policy": (
            "component_wise_all_finite_values_exact_authorized_roster"
        ),
        "scaler_authority_rows": 4500,
        "scaler_finite_count": dict(
            zip(covariates, diagnostics["finite_counts"].tolist(), strict=True)
        ),
        "scaler_missing_count": dict(
            zip(covariates, diagnostics["missing_counts"].tolist(), strict=True)
        ),
        "scaler_median": dict(
            zip(covariates, diagnostics["centre"].tolist(), strict=True)
        ),
        "scaler_iqr": dict(zip(covariates, diagnostics["scale"].tolist(), strict=True)),
        "matching_candidate_rows": len(diagnostics["candidate"]),
        "ineligible_nonfinite_host_rows": 4500 - len(diagnostics["candidate"]),
        "prior_complete_case_changed_assignment_count": int(
            comparison["assignment_changed"].sum()
        ),
        "prior_complete_case_changed_references": comparison.loc[
            comparison["assignment_changed"], "reference_id"
        ].tolist(),
        "matched_references": len(reconstructed),
        "unmatched_references": 43 - len(reconstructed),
        "analysis_rows": len(analysis),
        "analysis_reference_rows": len(observed),
        "analysis_bare_rows": len(bare_rows),
        "analysis_injected_rows": len(injected_rows),
        "pair_pixel_bytes_reopened": bytes_reopened,
        "matching_recomputed_from_direct_raw_tractor": True,
        "maximum_cardinality_then_minimum_distance_reproduced": True,
        "calipers_rechecked": True,
        "exact_pairing_rechecked": True,
        "bare_hwc_bytes_rechecked": True,
        "injected_parent_chw_bytes_rechecked": True,
        "host_source_lens_psf_render_random_flux_identities_rechecked": True,
        "detector_scores_used": False,
        "detector_outcomes_used": False,
        "feature_extraction_performed": False,
        "probe_fitting_performed": False,
        "bootstrap_performed": False,
        "host_total_flux_floor_applied": False,
        "flux_rescaling_applied": False,
        "ready_for_independent_pre_feature_review": True,
    }
    atomic_write_json(receipt, destination)
    return receipt
