"""Exact G_rec-child construction and CNN recovery cross-reference for G_domain."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .authorities import AuthorityContext
from .authorization import ExecutionAuthorization
from .contracts import (
    ContractError,
    assert_score_blind,
    canonical_row_sha256,
    derive_catalogue_covariates,
    require_columns,
    require_nonempty_columns,
    require_sha256,
    require_sha256_column,
    sha256_file,
)
from .parents import PARENT_IDENTITY_COLUMNS, GRecArtifacts


@dataclass(frozen=True)
class GDomainSubsetArtifact:
    frame: pd.DataFrame
    path: Path
    receipt_path: Path
    manifest_sha256: str
    receipt: dict[str, Any]


def _host_disposition(catalogue: pd.DataFrame) -> pd.DataFrame:
    result = catalogue[
        [
            "galaxy_id",
            "observed_zmag",
            "sersic_n",
            "axis_ratio_q",
        ]
    ].copy()
    result["finite_zmag"] = np.isfinite(result["observed_zmag"].to_numpy(float))
    result["finite_sersic"] = np.isfinite(result["sersic_n"].to_numpy(float))
    result["finite_axis_ratio"] = np.isfinite(
        result["axis_ratio_q"].to_numpy(float)
    )
    result["passes_zmag"] = result["finite_zmag"] & result["observed_zmag"].le(19.5)
    result["passes_sersic"] = result["finite_sersic"] & result["sersic_n"].ge(4.0)
    result["passes_axis_ratio"] = (
        result["finite_axis_ratio"] & result["axis_ratio_q"].ge(0.6)
    )
    result["selected"] = result[
        ["passes_zmag", "passes_sersic", "passes_axis_ratio"]
    ].all(axis=1)

    def reason(row: pd.Series) -> str:
        failed = []
        if not row["finite_zmag"]:
            failed.append("observed_zmag_nonfinite")
        elif not row["passes_zmag"]:
            failed.append("observed_zmag_gt_19.5")
        if not row["finite_sersic"]:
            failed.append("sersic_nonfinite")
        elif not row["passes_sersic"]:
            failed.append("sersic_lt_4")
        if not row["finite_axis_ratio"]:
            failed.append("axis_ratio_q_nonfinite")
        elif not row["passes_axis_ratio"]:
            failed.append("axis_ratio_q_lt_0.6")
        return "selected" if not failed else ";".join(failed)

    result["disposition"] = result.apply(reason, axis=1)
    return result


def build_gdomain_subset(
    grec: GRecArtifacts,
    *,
    config: Mapping[str, Any],
    authority: AuthorityContext,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the inclusive catalogue predicate without creating any new pixels."""
    authority.assert_config(config)
    grec.revalidate(config=dict(config), authority=authority)
    grec_parent = grec.scenes
    parent_manifest_sha256 = grec.scene_sha256
    require_columns(
        grec_parent, PARENT_IDENTITY_COLUMNS, context="G_rec parent manifest"
    )
    assert_score_blind(grec_parent, context="pre-score G_rec parent manifest")
    require_nonempty_columns(
        grec_parent, PARENT_IDENTITY_COLUMNS, context="G_rec parent manifest"
    )
    require_sha256_column(
        grec_parent, "host_pixel_sha256", context="G_rec parent manifest"
    )
    require_sha256_column(
        grec_parent, "final_image_sha256", context="G_rec parent manifest"
    )
    if grec_parent["grec_row_id"].astype(str).duplicated().any():
        raise ContractError("G_rec row IDs must be non-null and unique")
    authority.assert_host_subset(
        grec_parent,
        id_column="host_galaxy_id",
        payload_column=(
            "host_payload_sha256"
            if "host_payload_sha256" in grec_parent
            else "host_pixel_sha256"
        ),
        parent_column="host_parent_id",
        physical_column="host_physical_group_id",
        spatial_column="host_spatial_group_id",
        context="G_rec parent manifest",
    )
    host_catalogue = grec.tractor.frame
    require_columns(host_catalogue, {"galaxy_id"}, context="host catalogue")
    assert_score_blind(host_catalogue, context="G_domain host catalogue")
    if host_catalogue["galaxy_id"].astype(str).duplicated().any():
        raise ContractError("host catalogue contains duplicate galaxy IDs")
    catalogue = derive_catalogue_covariates(host_catalogue)
    catalogue["galaxy_id"] = catalogue["galaxy_id"].astype(str)
    host_ids = set(grec_parent["host_galaxy_id"].astype(str))
    selected_catalogue = catalogue[catalogue["galaxy_id"].isin(host_ids)].copy()
    if set(selected_catalogue["galaxy_id"]) != host_ids:
        missing = sorted(host_ids.difference(selected_catalogue["galaxy_id"]))
        raise ContractError(f"G_rec hosts lack Tractor rows: {missing[:5]}")
    covariates = ["observed_zmag", "sersic_n", "axis_ratio_q"]
    disposition = _host_disposition(selected_catalogue).sort_values(
        "galaxy_id", ignore_index=True
    )
    selected_hosts = set(disposition.loc[disposition["selected"], "galaxy_id"])
    subset = grec_parent[
        grec_parent["host_galaxy_id"].astype(str).isin(selected_hosts)
    ].copy()
    subset["gdomain_row_id"] = subset["grec_row_id"].astype(str)
    subset["grec_parent_manifest_sha256"] = parent_manifest_sha256
    subset["authority_digest"] = authority.authority_digest
    subset = subset.merge(
        selected_catalogue[["galaxy_id", *covariates]],
        left_on="host_galaxy_id",
        right_on="galaxy_id",
        how="left",
        validate="many_to_one",
    ).drop(columns="galaxy_id")
    subset = subset.sort_values("grec_row_id", ignore_index=True)
    validate_gdomain_subset(
        grec,
        subset,
        config=config,
        authority=authority,
    )
    return subset, disposition


def validate_gdomain_subset(
    grec: GRecArtifacts,
    subset: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    authority: AuthorityContext,
) -> None:
    """Prove that the subset is exactly the predicate-selected parent rows."""
    authority.assert_config(config)
    grec.revalidate(config=dict(config), authority=authority)
    grec_parent = grec.scenes
    parent_manifest_sha256 = grec.scene_sha256
    require_columns(
        subset,
        {
            *PARENT_IDENTITY_COLUMNS,
            *grec_parent.columns,
            "gdomain_row_id",
            "grec_parent_manifest_sha256",
            "authority_digest",
        },
        context="G_domain subset",
    )
    if not subset["grec_parent_manifest_sha256"].eq(parent_manifest_sha256).all():
        raise ContractError("G_domain subset has the wrong G_rec parent authority")
    if not subset["authority_digest"].eq(authority.authority_digest).all():
        raise ContractError("G_domain subset has the wrong authority context")
    catalogue = derive_catalogue_covariates(grec.tractor.frame)
    parent_hosts = set(grec_parent["host_galaxy_id"].astype(str))
    selected_hosts = set(
        catalogue.loc[
            catalogue["observed_zmag"].le(19.5)
            & catalogue["sersic_n"].ge(4.0)
            & catalogue["axis_ratio_q"].ge(0.6),
            "galaxy_id",
        ].astype(str)
    ).intersection(parent_hosts)
    expected = grec_parent[
        grec_parent["host_galaxy_id"].astype(str).isin(selected_hosts)
    ].sort_values("grec_row_id", ignore_index=True)
    observed = subset.sort_values("grec_row_id", ignore_index=True)
    if (
        not observed["gdomain_row_id"]
        .astype(str)
        .equals(observed["grec_row_id"].astype(str))
    ):
        raise ContractError("G_domain row IDs do not preserve G_rec parent IDs")
    if len(expected) != len(observed):
        raise ContractError("G_domain row count is not the exact predicate subset")
    for column in grec_parent.columns:
        if (
            not expected[column]
            .reset_index(drop=True)
            .equals(observed[column].reset_index(drop=True))
        ):
            raise ContractError(f"G_domain changed parent field {column}")
    if set(observed["host_galaxy_id"].astype(str)) != selected_hosts:
        raise ContractError("G_domain host membership differs from exact predicate")


def load_gdomain_subset(
    *,
    path: Path,
    receipt_path: Path,
    grec: GRecArtifacts,
    config: Mapping[str, Any],
    authority: AuthorityContext,
) -> GDomainSubsetArtifact:
    """Reload a subset, verify receipt parents, and recompute the raw predicate."""
    authority.assert_config(config)
    grec.revalidate(config=dict(config), authority=authority)
    manifest_sha256 = sha256_file(path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "status": "pre_score_exact_G_domain_subset_complete",
        "subset_sha256": manifest_sha256,
        "G_rec_parent_sha256": grec.scene_sha256,
        "G_rec_host_sha256": grec.host_sha256,
        "G_rec_tractor_sha256": grec.tractor.sha256,
        "authority_digest": authority.authority_digest,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ContractError(f"G_domain subset receipt has wrong {field}")
    frame = pd.read_parquet(path)
    recomputed, _ = build_gdomain_subset(
        grec,
        config=config,
        authority=authority,
    )
    try:
        pd.testing.assert_frame_equal(
            frame.reset_index(drop=True),
            recomputed.reset_index(drop=True),
            check_exact=True,
        )
    except AssertionError as error:
        raise ContractError(
            "G_domain subset differs from recomputed raw Tractor predicate"
        ) from error
    return GDomainSubsetArtifact(
        frame=frame,
        path=path.resolve(),
        receipt_path=receipt_path.resolve(),
        manifest_sha256=manifest_sha256,
        receipt=receipt,
    )


def cnn_recovery_cross_reference(
    grec: GRecArtifacts,
    gdomain_subset: GDomainSubsetArtifact,
    scores_path: Path,
    *,
    config: Mapping[str, Any],
    authority: AuthorityContext,
    threshold: np.float32,
    scoring_receipt_sha256: str,
    authorization: ExecutionAuthorization,
    design_only: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute full/subset recovery from one exact score table and threshold."""
    authority.assert_config(config)
    grec.revalidate(config=dict(config), authority=authority)
    verified_subset = load_gdomain_subset(
        path=gdomain_subset.path,
        receipt_path=gdomain_subset.receipt_path,
        grec=grec,
        config=config,
        authority=authority,
    )
    grec_parent = grec.scenes
    subset_frame = verified_subset.frame
    parent_manifest_sha256 = grec.scene_sha256
    subset_manifest_sha256 = verified_subset.manifest_sha256
    require_sha256(scoring_receipt_sha256, context="CNN scoring receipt")
    scores_manifest_sha256 = sha256_file(scores_path)
    authorization.revalidate(
        config=dict(config),
        authority=authority,
        stage="cnn_recovery",
        manifest_sha256=scores_manifest_sha256,
    )
    scores = pd.read_parquet(scores_path)
    require_columns(scores, {"data_role"}, context="CNN score table")
    if not scores["data_role"].eq(authorization.data_role).all():
        raise ContractError("CNN scores have the wrong authorized data role")
    if design_only and authorization.data_role != "design":
        raise ContractError("design-only CNN recovery received final-role scores")
    if not design_only and authorization.data_role != "final":
        raise ContractError("final CNN recovery requires final authorization")
    payload_column = (
        "host_payload_sha256"
        if "host_payload_sha256" in grec_parent.columns
        else "host_pixel_sha256"
    )
    authority.assert_host_subset(
        grec_parent,
        id_column="host_galaxy_id",
        payload_column=payload_column,
        parent_column="host_parent_id",
        physical_column="host_physical_group_id",
        spatial_column="host_spatial_group_id",
        context="CNN recovery G_rec parent",
    )
    require_columns(
        subset_frame,
        {*grec_parent.columns, "grec_parent_manifest_sha256"},
        context="CNN recovery G_domain subset",
    )
    if subset_frame["grec_row_id"].astype(str).duplicated().any():
        raise ContractError("CNN recovery G_domain subset duplicates parent rows")
    if not subset_frame["grec_parent_manifest_sha256"].eq(parent_manifest_sha256).all():
        raise ContractError("CNN recovery subset has the wrong parent manifest")
    subset_ids_ordered = subset_frame["grec_row_id"].astype(str)
    expected_subset = grec_parent[
        grec_parent["grec_row_id"].astype(str).isin(subset_ids_ordered)
    ].sort_values("grec_row_id", ignore_index=True)
    observed_subset = subset_frame.sort_values("grec_row_id", ignore_index=True)
    if len(expected_subset) != len(observed_subset):
        raise ContractError("CNN recovery subset is not one-to-one with G_rec")
    for column in grec_parent.columns:
        if not expected_subset[column].equals(observed_subset[column]):
            raise ContractError(f"CNN recovery subset changed parent field {column}")
    require_columns(
        scores,
        {
            *grec_parent.columns,
            "parent_manifest_sha256",
            "parent_row_sha256",
            "cnn_logit",
            "scoring_receipt_sha256",
        },
        context="CNN score table",
    )
    require_sha256_column(scores, "final_image_sha256", context="CNN score table")
    require_sha256_column(scores, "parent_row_sha256", context="CNN score table")
    if not scores["parent_manifest_sha256"].eq(parent_manifest_sha256).all():
        raise ContractError("CNN score table has the wrong parent manifest")
    if scores["grec_row_id"].astype(str).duplicated().any():
        raise ContractError("CNN score table contains duplicate G_rec rows")
    if set(scores["grec_row_id"].astype(str)) != set(
        grec_parent["grec_row_id"].astype(str)
    ):
        raise ContractError("CNN scores do not cover exactly the full G_rec parent")
    if not scores["scoring_receipt_sha256"].eq(scoring_receipt_sha256).all():
        raise ContractError("CNN score rows do not share the declared receipt")
    logits = scores["cnn_logit"].to_numpy()
    if logits.dtype != np.float32 or not np.isfinite(logits).all():
        raise ContractError("CNN logits must be finite native float32 values")
    if not isinstance(threshold, np.float32) or not np.isfinite(threshold):
        raise ContractError("threshold must be one finite native float32 scalar")
    parent_sorted = grec_parent.sort_values("grec_row_id", ignore_index=True)
    score_sorted = scores.sort_values("grec_row_id", ignore_index=True)
    for column in grec_parent.columns:
        if not parent_sorted[column].equals(score_sorted[column]):
            raise ContractError(f"CNN score table changed parent field {column}")
    expected_row_hashes = [
        canonical_row_sha256(row, grec_parent.columns)
        for row in parent_sorted.to_dict(orient="records")
    ]
    if score_sorted["parent_row_sha256"].astype(str).tolist() != expected_row_hashes:
        raise ContractError("CNN score table has invalid parent row hashes")
    parent = parent_sorted.copy()
    parent["cnn_logit"] = score_sorted["cnn_logit"].to_numpy()
    parent["scoring_receipt_sha256"] = scoring_receipt_sha256
    parent["detected"] = parent["cnn_logit"].ge(threshold)
    subset_ids = set(subset_frame["grec_row_id"].astype(str))
    parent["in_gdomain_subset"] = parent["grec_row_id"].astype(str).isin(subset_ids)
    if int(parent["in_gdomain_subset"].sum()) != len(subset_frame):
        raise ContractError("G_domain subset cross-reference is not one-to-one")

    def summarize(frame: pd.DataFrame) -> dict[str, Any]:
        detections = int(frame["detected"].sum())
        return {
            "rows": len(frame),
            "hosts": int(frame["host_galaxy_id"].nunique()),
            "detections": detections,
            "recovery": float(detections / len(frame)) if len(frame) else None,
        }

    summary = {
        "schema_version": 1,
        "threshold_float32": float(threshold),
        "comparison": ">=",
        "scoring_receipt_sha256": scoring_receipt_sha256,
        "G_rec_parent_manifest_sha256": parent_manifest_sha256,
        "G_domain_subset_manifest_sha256": subset_manifest_sha256,
        "authority_digest": authority.authority_digest,
        "full_G_rec": summarize(parent),
        "exact_G_domain_subset": summarize(parent[parent["in_gdomain_subset"]]),
        "subset_is_exact_parent_rows": True,
        "subset_pixel_mismatch_count": 0,
        "all_parent_fields_compared": True,
        "interpretation": (
            "Two correlated recovery summaries from one G_rec score table; "
            "the subset is not an independent replication."
        ),
    }
    return summary, parent


def config_predicate(config: Mapping[str, Any]) -> dict[str, float]:
    predicate = config["G_domain"]["predicate"]
    return {key: float(value) for key, value in predicate.items()}
