#!/usr/bin/env python3
"""Prepare the score-blind 43-system observed reference cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path(
            "evaluation_dataset_prep/stage1_antijoin/"
            "candidate_positive_keys.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_dataset_prep/reference_cohort"),
    )
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(
        arguments.root / "data/manifests/train.parquet",
        columns=["galaxy_id"],
    )
    validation = pd.read_parquet(
        arguments.root / "data/manifests/validation.parquet",
        columns=["galaxy_id"],
    )
    test = pd.read_parquet(arguments.root / "data/manifests/test.parquet")
    candidates = pd.read_parquet(arguments.candidates)
    provenance = pd.read_csv(
        arguments.root / "data/catalogues/lens_provenance.csv"
    )

    references = test[
        test["endpoint_role"].eq("terminal_confirmed")
        & test["label"].eq(1)
        & test["lensing_status"].eq("confirmed_lens")
    ].copy()
    references["galaxy_id"] = references["galaxy_id"].astype(str)
    if len(references) != 43 or references["galaxy_id"].nunique() != 43:
        raise ValueError("expected exactly 43 unique terminal-confirmed systems")

    development_ids = set(train["galaxy_id"].astype(str)) | set(
        validation["galaxy_id"].astype(str)
    )
    if set(references["galaxy_id"]) & development_ids:
        raise ValueError("reference cohort overlaps latest detector development")

    candidate_columns = [
        "galaxy_id",
        "s3_key",
        "bytes",
        "etag",
        "canonical_key",
        "stage1_reason",
    ]
    references = references.merge(
        candidates[candidate_columns],
        on="galaxy_id",
        how="left",
        validate="one_to_one",
    )
    if references["s3_key"].isna().any():
        missing = references.loc[references["s3_key"].isna(), "galaxy_id"].tolist()
        raise ValueError(f"confirmed references missing from S3 candidates: {missing}")
    if not references["stage1_reason"].eq("candidate").all():
        raise ValueError("reference cohort contains a rejected S3 candidate")

    references = references.merge(
        provenance,
        on=["galaxy_id", "physical_group_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_provenance"),
    )
    if not references["final_lensing_status"].eq("confirmed_lens").all():
        raise ValueError("reference cohort lacks confirmed provenance")

    forbidden_score_columns = [
        column
        for column in references.columns
        if any(token in column.lower() for token in ("score", "logit", "probability"))
    ]
    if forbidden_score_columns:
        raise ValueError(
            f"score columns entered reference selection: {forbidden_score_columns}"
        )

    references = references.sort_values("galaxy_id").reset_index(drop=True)
    output = arguments.output_dir / "confirmed_43.parquet"
    references.to_parquet(output, index=False)
    summary = {
        "schema_version": 1,
        "status": "candidate_reference_cohort_pending_full_payload_checks",
        "rows": len(references),
        "unique_galaxy_ids": references["galaxy_id"].nunique(),
        "latest_detector_train_overlap": 0,
        "latest_detector_validation_overlap": 0,
        "selection_used_detector_scores": False,
        "used_by": ["G_domain", "G_match"],
        "shared_cohort_is_independent_replication": False,
        "manifest_sha256": sha256_file(output),
    }
    (arguments.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
