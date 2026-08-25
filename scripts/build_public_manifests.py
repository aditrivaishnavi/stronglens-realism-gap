#!/usr/bin/env python3
"""Build portable public manifests from the final grouped data split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from validate_manifests import validate

COMMON_COLUMNS = (
    "galaxy_id",
    "ra",
    "dec",
    "healpix_128",
    "physical_group_id",
    "spatial_component_id",
    "label",
    "sample_weight",
    "pool",
    "tier",
    "cohort",
    "lensing_status",
    "confirmation_category",
)
OPTIONAL_DEVELOPMENT_COLUMNS = (
    "is_noisy_positive",
    "training_label_origin",
)


def portable_path(value: object) -> str:
    text = str(value)
    marker = "stronglens_calibration/"
    if marker not in text:
        raise ValueError(f"cutout path has no stable dataset root: {text}")
    return text[text.index(marker) :]


def public_frame(
    frame: pd.DataFrame,
    *,
    split: str,
    terminal_ids: set[str],
) -> pd.DataFrame:
    columns = [*COMMON_COLUMNS]
    columns.extend(
        column for column in OPTIONAL_DEVELOPMENT_COLUMNS if column in frame.columns
    )
    result = frame.loc[:, columns].copy()
    source_paths = (
        frame["cutout_relpath"]
        if "cutout_relpath" in frame.columns
        else frame["cutout_path"]
    )
    result.insert(1, "cutout_relpath", source_paths.map(portable_path))
    result["split"] = split

    if split != "test":
        result["endpoint_role"] = "model_development"
    else:
        positive = result["label"].astype(int).eq(1)
        terminal = result["galaxy_id"].astype(str).isin(terminal_ids)
        result["endpoint_role"] = "test_negative"
        result.loc[positive & terminal, "endpoint_role"] = "terminal_confirmed"
        result.loc[positive & ~terminal, "endpoint_role"] = (
            "development_contacted_confirmed"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--terminal-cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    development = pd.read_parquet(arguments.development_manifest)
    test = pd.read_parquet(arguments.test_manifest)
    terminal = json.loads(arguments.terminal_cohort.read_text(encoding="utf-8"))
    terminal_ids = set(terminal["galaxy_ids"])

    manifests = {
        "train": public_frame(
            development.loc[development["split"].eq("train")],
            split="train",
            terminal_ids=terminal_ids,
        ),
        "validation": public_frame(
            development.loc[development["split"].eq("val")],
            split="validation",
            terminal_ids=terminal_ids,
        ),
        "test": public_frame(test, split="test", terminal_ids=terminal_ids),
    }
    validate(manifests)

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in manifests.items():
        frame.to_parquet(
            arguments.output_dir / f"{name}.parquet",
            index=False,
            compression="zstd",
        )
    development_validation = manifests["validation"].copy()
    development_validation["split"] = "val"
    pd.concat(
        [manifests["train"], development_validation],
        ignore_index=True,
    ).to_parquet(
        arguments.output_dir / "development.parquet",
        index=False,
        compression="zstd",
    )

    summary = {
        "schema_version": 1,
        "minimum_cross_split_separation_arcsec": 38.0,
        "splits": {
            name: {
                "rows": len(frame),
                "positive_labels": int(frame["label"].sum()),
                "negative_labels": int(frame["label"].eq(0).sum()),
                "endpoint_roles": {
                    str(role): int(count)
                    for role, count in frame["endpoint_role"].value_counts().items()
                },
            }
            for name, frame in manifests.items()
        },
        "validation": {
            "pairwise_disjoint": [
                "galaxy_id",
                "cutout_relpath",
                "healpix_128",
                "physical_group_id",
                "spatial_component_id",
            ],
            "coordinate_separation_checked": True,
        },
    }
    (arguments.output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    schema = {
        "schema_version": 1,
        "columns": {
            column: str(manifests["train"][column].dtype)
            for column in manifests["train"].columns
        },
        "cutout_relpath": (
            "Path relative to the separately obtained stronglens_calibration "
            "data root; no local filesystem paths are stored."
        ),
        "endpoint_role": (
            "model_development, test_negative, terminal_confirmed, or "
            "development_contacted_confirmed"
        ),
    }
    (arguments.output_dir / "schema.json").write_text(
        json.dumps(schema, indent=2) + "\n",
        encoding="utf-8",
    )
    print(arguments.output_dir)


if __name__ == "__main__":
    main()
