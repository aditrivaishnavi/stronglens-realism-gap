#!/usr/bin/env python3
"""Build a score-blind first-pass S3 cutout anti-join for the latest CNN.

This stage excludes only the training and validation members of the fixed
seed-42 epoch-160 detector. It does not treat contact with superseded models or
unpublished historical experiments as contamination of that detector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

EXPECTED_DEVELOPMENT = {
    "train.parquet": {
        "sha256": "a0caa3824568f96010275e59d89d013a23ac7e6e6ae280de2c636a401fa38e5b",
        "rows": 290_312,
    },
    "validation.parquet": {
        "sha256": "809729e63c9bd6fca96f51b4f8d244aeda604beca4135edcc428f431a9941ee9",
        "rows": 62_215,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_ids(
    destination: set[str],
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> int:
    before = len(destination)
    for column in columns:
        if column not in frame:
            continue
        values = frame[column].dropna().astype(str).str.strip()
        destination.update(value for value in values if value)
    return len(destination) - before


def load_detector_development_ids(
    current_root: Path,
) -> tuple[set[str], list[dict[str, object]]]:
    excluded: set[str] = set()
    evidence: list[dict[str, object]] = []

    sources: list[tuple[Path, tuple[str, ...]]] = [
        (current_root / "data/manifests/train.parquet", ("galaxy_id",)),
        (current_root / "data/manifests/validation.parquet", ("galaxy_id",)),
    ]

    for path, columns in sources:
        if not path.is_file():
            raise FileNotFoundError(f"required detector manifest is absent: {path}")
        expected = EXPECTED_DEVELOPMENT[path.name]
        digest = sha256_file(path)
        if digest != expected["sha256"]:
            raise ValueError(f"{path.name} SHA-256 mismatch: {digest}")
        available = pd.read_parquet(path).columns
        missing = sorted(set(columns) - set(available))
        if missing:
            raise ValueError(f"{path.name} lacks required columns: {missing}")
        frame = pd.read_parquet(path, columns=list(columns))
        if len(frame) != expected["rows"]:
            raise ValueError(f"{path.name} row-count mismatch: {len(frame)}")
        if frame["galaxy_id"].astype(str).duplicated().any():
            raise ValueError(f"{path.name} contains duplicate galaxy IDs")
        before = len(excluded)
        add_ids(excluded, frame, columns)
        evidence.append(
            {
                "logical_path": path.name,
                "status": "used",
                "sha256": digest,
                "rows": len(frame),
                "columns": list(columns),
                "new_unique_ids": len(excluded) - before,
            }
        )

    expected_unique = sum(
        int(record["rows"]) for record in EXPECTED_DEVELOPMENT.values()
    )
    if len(excluded) != expected_unique:
        raise ValueError(
            "latest detector train/validation identifiers are not disjoint and unique"
        )
    return excluded, evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("evaluation_dataset_prep/s3_cutout_inventory.tsv"),
    )
    parser.add_argument("--current-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_dataset_prep/stage1_antijoin"),
    )
    arguments = parser.parse_args()

    current_root = arguments.current_root.resolve()
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)

    excluded, evidence = load_detector_development_ids(current_root)
    inventory = pd.read_csv(
        arguments.inventory,
        sep="\t",
        names=["s3_key", "bytes", "etag", "last_modified"],
        dtype={"s3_key": "string", "bytes": "int64", "etag": "string"},
    )
    inventory["etag"] = inventory["etag"].str.strip('"')
    inventory["galaxy_id"] = inventory["s3_key"].str.extract(
        r"/([^/]+)\.npz$", expand=False
    )
    inventory["catalogue_bucket"] = inventory["s3_key"].str.extract(
        r"/cutouts/(negatives|positives)/", expand=False
    )
    inventory["is_npz"] = inventory["galaxy_id"].notna()
    inventory["latest_detector_development_contact"] = inventory[
        "galaxy_id"
    ].isin(excluded)
    inventory["valid_single_part_etag"] = inventory["etag"].str.fullmatch(
        r"[0-9a-fA-F]{32}", na=False
    )

    inventory["_key_depth"] = inventory["s3_key"].str.count("/")
    inventory["_direct_key"] = inventory["s3_key"].str.fullmatch(
        r"stronglens_calibration/cutouts/(negatives|positives)/[^/]+\.npz",
        na=False,
    )
    inventory["_direct_rank"] = (~inventory["_direct_key"]).astype(int)
    inventory = inventory.sort_values(
        ["galaxy_id", "_direct_rank", "_key_depth", "s3_key"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    by_id = inventory.groupby("galaxy_id", dropna=False).agg(
        key_count=("s3_key", "size"),
        distinct_etags=("etag", "nunique"),
        distinct_label_buckets=("catalogue_bucket", "nunique"),
        has_direct_key=("_direct_key", "any"),
        canonical_key=("s3_key", "first"),
        canonical_etag=("etag", "first"),
    )
    inventory = inventory.join(by_id, on="galaxy_id")
    etag_ids = inventory.groupby("etag", dropna=False)["galaxy_id"].nunique()
    inventory["etag_distinct_ids"] = inventory["etag"].map(etag_ids)
    inventory["canonical_version"] = (
        inventory["s3_key"] == inventory["canonical_key"]
    )
    inventory["has_alternate_pixel_versions"] = inventory["distinct_etags"].gt(1)
    inventory["same_as_canonical_etag"] = (
        inventory["etag"] == inventory["canonical_etag"]
    )
    inventory["ambiguous_timestamp_only_versions"] = (
        ~inventory["has_direct_key"] & inventory["distinct_etags"].gt(1)
    )

    inventory["stage1_reason"] = "candidate"
    reason_rules = [
        (~inventory["is_npz"], "not_npz_cutout"),
        (inventory["catalogue_bucket"].isna(), "unknown_catalogue_bucket"),
        (
            inventory["latest_detector_development_contact"],
            "latest_detector_train_or_validation",
        ),
        (inventory["distinct_label_buckets"].gt(1), "conflicting_label_buckets"),
        (inventory["etag_distinct_ids"].gt(1), "same_pixels_used_by_multiple_ids"),
        (~inventory["valid_single_part_etag"], "etag_not_single_part_md5"),
        (
            inventory["ambiguous_timestamp_only_versions"],
            "ambiguous_timestamp_only_versions",
        ),
        (
            ~inventory["canonical_version"]
            & inventory["same_as_canonical_etag"],
            "duplicate_identical_key_version",
        ),
        (
            ~inventory["canonical_version"]
            & ~inventory["same_as_canonical_etag"],
            "alternate_noncanonical_pixel_version",
        ),
    ]
    for mask, reason in reason_rules:
        inventory.loc[
            mask & inventory["stage1_reason"].eq("candidate"), "stage1_reason"
        ] = reason

    negatives = inventory[
        inventory["catalogue_bucket"].eq("negatives")
        & inventory["stage1_reason"].eq("candidate")
    ].copy()
    positives = inventory[
        inventory["catalogue_bucket"].eq("positives")
        & inventory["stage1_reason"].eq("candidate")
    ].copy()

    inventory.to_parquet(output / "annotated_inventory.parquet", index=False)
    negatives.to_parquet(output / "candidate_negative_keys.parquet", index=False)
    positives.to_parquet(output / "candidate_positive_keys.parquet", index=False)

    summary = {
        "schema_version": 1,
        "status": "stage1_metadata_only_not_evaluation_ready",
        "inventory": {
            "objects": len(inventory),
            "bytes": int(inventory["bytes"].sum()),
            "sha256": sha256_file(arguments.inventory),
        },
        "latest_detector_development_ids": len(excluded),
        "latest_detector_development_sources": evidence,
        "catalogue_bucket_counts": inventory["catalogue_bucket"]
        .fillna("unknown")
        .value_counts()
        .sort_index()
        .to_dict(),
        "reason_counts": inventory["stage1_reason"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "candidate_negative_keys": len(negatives),
        "candidate_positive_keys": len(positives),
        "remaining_required_steps": [
            "retrieve_candidate_metadata_and_coordinates",
            "remove_coordinate_and_overlapping_footprint_contacts",
            "verify_catalogue_label_and_selection_eligibility",
            "download_only_the_planned_candidate_pool",
            "compute_payload_sha256_and_validate_npz_schema",
            "freeze_population_definition_before_role_assignment",
            "do_not_use_detector_scores_for_selection",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
