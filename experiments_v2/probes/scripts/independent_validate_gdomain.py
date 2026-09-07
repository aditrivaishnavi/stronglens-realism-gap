"""Independent, score-blind validation of final G_domain preparation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

OUTCOME_TOKENS = {
    "auc",
    "decision",
    "detection",
    "detectoroutput",
    "logit",
    "logits",
    "oof",
    "prediction",
    "predictions",
    "probability",
    "probabilities",
    "score",
    "scores",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_score_blind(frame: pd.DataFrame, name: str) -> None:
    prohibited = []
    for column in frame.columns:
        lowered = str(column).lower()
        tokens = set(re.split(r"[^a-z0-9]+", lowered))
        tokens.add(re.sub(r"[^a-z0-9]+", "", lowered))
        if tokens & OUTCOME_TOKENS:
            prohibited.append(str(column))
    if prohibited:
        raise AssertionError(f"{name} contains outcome columns: {prohibited}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.repository_root.resolve()
    paths = {
        "config": root / "experiments_v2/probes/configs/probes.yaml",
        "raw_tractor": (
            root / "evaluation_dataset_prep/tractor_fallback_full/tractor_hosts.parquet"
        ),
        "reference_roster": (
            root / "evaluation_dataset_prep/reference_cohort/confirmed_43.parquet"
        ),
        "source_parent": (
            root / "results_v2/grec/parents/grec_parent_manifest.parquet"
        ),
        "reference_adapter": (
            root / "results_v2/probes/authorities/reference_payload_adapter.parquet"
        ),
        "host_adapter": (
            root / "results_v2/probes/authorities/host_authority_adapter.parquet"
        ),
        "tractor_adapter": (
            root / "results_v2/probes/authorities/host_tractor_adapter.parquet"
        ),
        "parent_adapter": (
            root / "results_v2/probes/authorities/grec_parent_adapter.parquet"
        ),
        "adapter_receipt": (
            root / "results_v2/probes/authorities/FINAL_SCHEMA_ADAPTER_RECEIPT.json"
        ),
        "subset": root / "results_v2/probes/gdomain/gdomain_subset.parquet",
        "disposition": (
            root / "results_v2/probes/gdomain/gdomain_host_disposition.parquet"
        ),
        "subset_receipt": (
            root / "results_v2/probes/gdomain/GDOMAIN_SUBSET_RECEIPT.json"
        ),
        "analysis": (
            root / "results_v2/probes/gdomain/gdomain_analysis_manifest.parquet"
        ),
        "preparation_receipt": (
            root / "results_v2/probes/gdomain/GDOMAIN_PREPARATION_RECEIPT.json"
        ),
        "authority_preflight": (
            root / "results_v2/probes/gdomain/GDOMAIN_AUTHORITY_PREFLIGHT.json"
        ),
    }
    if arguments.output.exists():
        raise FileExistsError("independent validation refuses to overwrite output")
    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    raw = pd.read_parquet(paths["raw_tractor"])
    references = pd.read_parquet(paths["reference_roster"])
    source_parent = pd.read_parquet(paths["source_parent"])
    reference_adapter = pd.read_parquet(paths["reference_adapter"])
    hosts = pd.read_parquet(paths["host_adapter"])
    tractor = pd.read_parquet(paths["tractor_adapter"])
    parent = pd.read_parquet(paths["parent_adapter"])
    subset = pd.read_parquet(paths["subset"])
    disposition = pd.read_parquet(paths["disposition"])
    analysis = pd.read_parquet(paths["analysis"])
    for name, frame in (
        ("raw Tractor", raw),
        ("reference adapter", reference_adapter),
        ("host adapter", hosts),
        ("Tractor adapter", tractor),
        ("parent adapter", parent),
        ("G_domain subset", subset),
        ("G_domain disposition", disposition),
        ("G_domain analysis", analysis),
    ):
        assert_score_blind(frame, name)

    assert len(hosts) == hosts["galaxy_id"].nunique() == 4_500
    assert len(tractor) == tractor["galaxy_id"].nunique() == 4_500
    assert set(hosts["galaxy_id"].astype(str)) == set(tractor["galaxy_id"].astype(str))
    assert len(parent) == parent["grec_row_id"].nunique() == 49_500
    assert parent["host_galaxy_id"].nunique() == 4_500
    assert parent.groupby("host_galaxy_id").size().eq(11).all()
    assert set(parent["theta_e_arcsec"].unique()) == {
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
    }
    source_sorted = source_parent.sort_values("grec_row_id", ignore_index=True)
    parent_sorted = parent.sort_values("grec_row_id", ignore_index=True)
    unchanged_source_columns = [
        column
        for column in source_parent.columns
        if column not in {"source_parameter_id", "lens_parameter_id"}
    ]
    pd.testing.assert_frame_equal(
        source_sorted[unchanged_source_columns],
        parent_sorted[unchanged_source_columns],
        check_exact=True,
    )
    assert source_sorted["source_parameter_id"].eq(
        parent_sorted["native_source_parameter_id"]
    ).all()
    assert source_sorted["lens_parameter_id"].eq(
        parent_sorted["native_lens_parameter_id"]
    ).all()
    assert parent["final_image_sha256"].eq(parent["parent_artifact_sha256"]).all()
    assert parent["final_image_path"].eq(parent["parent_pixel_path"]).all()
    assert parent["source_parameter_id"].eq(parent["source_nuisance_id"]).all()
    assert parent["lens_parameter_id"].eq(parent["lens_nuisance_id"]).all()

    flux_z = pd.to_numeric(raw["flux_z"], errors="coerce").to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        zmag = np.where(flux_z > 0, 22.5 - 2.5 * np.log10(flux_z), np.nan)
    sersic = pd.to_numeric(raw["sersic"], errors="coerce").to_numpy(float)
    ellipticity = np.hypot(
        pd.to_numeric(raw["shape_e1"], errors="coerce").to_numpy(float),
        pd.to_numeric(raw["shape_e2"], errors="coerce").to_numpy(float),
    )
    q = (1.0 - ellipticity) / (1.0 + ellipticity)
    q[(q < 0) | (q > 1) | ~np.isfinite(q)] = np.nan
    selected_mask = (zmag <= 19.5) & (sersic >= 4.0) & (q >= 0.6)
    selected_ids = set(raw.loc[selected_mask, "galaxy_id"].astype(str))
    assert len(selected_ids) == 646
    assert set(subset["host_galaxy_id"].astype(str)) == selected_ids
    assert len(subset) == 7_106
    assert subset.groupby("host_galaxy_id").size().eq(11).all()
    expected_subset = parent[
        parent["host_galaxy_id"].astype(str).isin(selected_ids)
    ].sort_values("grec_row_id", ignore_index=True)
    observed_subset = subset.sort_values("grec_row_id", ignore_index=True)
    pd.testing.assert_frame_equal(
        expected_subset,
        observed_subset[list(parent.columns)],
        check_exact=True,
    )
    disposition_index = disposition.set_index(disposition["galaxy_id"].astype(str))
    assert set(disposition_index.index) == set(raw["galaxy_id"].astype(str))
    assert set(
        disposition.loc[disposition["selected"], "galaxy_id"].astype(str)
    ) == selected_ids

    reference_ids = set(references["galaxy_id"].astype(str))
    observed = analysis[analysis["population"].eq("observed_reference")]
    simulated = analysis[analysis["population"].eq("simulated")]
    assert len(reference_ids) == 43
    assert len(observed) == observed["reference_id"].nunique() == 43
    assert set(observed["reference_id"].astype(str)) == reference_ids
    assert len(simulated) == 7_106
    assert set(simulated["row_id"].astype(str)) == {
        f"gdomain:simulated:{value}"
        for value in subset["grec_row_id"].astype(str)
    }
    reference_lookup = reference_adapter.set_index(
        reference_adapter["galaxy_id"].astype(str)
    )
    for row in observed.itertuples(index=False):
        source = reference_lookup.loc[str(row.reference_id)]
        assert str(row.pixel_sha256) == str(source["pixel_sha256"])
        assert str(row.pixel_file_sha256) == str(source["pixel_file_sha256"])
        assert str(row.pixel_relpath) == str(source["pixel_relpath"])
    subset_lookup = subset.set_index(subset["grec_row_id"].astype(str))
    for row in simulated.itertuples(index=False):
        source = subset_lookup.loc[str(row.row_id).removeprefix("gdomain:simulated:")]
        assert str(row.pixel_sha256) == str(source["expectation_pixel_sha256"])
        assert str(row.pixel_file_sha256) == str(source["parent_artifact_sha256"])
        assert str(row.pixel_relpath) == str(source["parent_pixel_path"])
    assert analysis["data_role"].eq("final").all()

    execution = config["execution"]
    assert execution["feature_extraction_authorized"] is False
    assert execution["extraction_authorization_receipt_sha256"] is None
    assert execution["final_scoring_authorized"] is False
    assert execution["final_analysis_authorized"] is False
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    output = {
        "schema_version": 1,
        "status": "independent_score_blind_G_domain_validation_passed",
        "validator_sha256": sha256_file(Path(__file__)),
        "hashes": hashes,
        "parent_hosts": 4_500,
        "parent_rows": 49_500,
        "selected_hosts": 646,
        "selected_rows": 7_106,
        "analysis_reference_rows": 43,
        "analysis_simulated_rows": 7_106,
        "analysis_rows": 7_149,
        "nonfinite_zmag_hosts": int((~np.isfinite(zmag)).sum()),
        "predicate": "zmag <= 19.5; sersic >= 4; q >= 0.6",
        "source_parent_columns_exact": True,
        "parent_pixel_paths_and_hashes_preserved": True,
        "detector_scores_used": False,
        "detector_outcomes_used": False,
        "feature_extraction_authorized": False,
        "feature_extraction_performed": False,
        "stein_probe_run": False,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
