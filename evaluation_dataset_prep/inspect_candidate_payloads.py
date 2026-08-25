#!/usr/bin/env python3
"""Download and inspect a deterministic, score-blind candidate pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_npz(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        arrays: dict[str, object] = {}
        for key in data.files:
            value = np.asarray(data[key])
            record: dict[str, object] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            if np.issubdtype(value.dtype, np.number):
                record["finite"] = bool(np.isfinite(value).all())
            if value.ndim == 0:
                record["value"] = value.item()
            arrays[key] = record
        cutout_key = "cutout" if "cutout" in data.files else None
        valid_cutout = False
        if cutout_key is not None:
            cutout = np.asarray(data[cutout_key])
            valid_cutout = (
                cutout.shape in {(101, 101, 3), (3, 101, 101)}
                and np.issubdtype(cutout.dtype, np.number)
                and bool(np.isfinite(cutout).all())
            )
    return {
        "arrays": arrays,
        "valid_three_band_101_cutout": valid_cutout,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    parser.add_argument("--per-bucket", type=int, default=20)
    parser.add_argument(
        "--stage1",
        type=Path,
        default=Path("evaluation_dataset_prep/stage1_antijoin"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_dataset_prep/payload_pilot"),
    )
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    selected: list[tuple[str, str, str]] = []
    for bucket, filename in (
        ("negative", "candidate_negative_keys.parquet"),
        ("positive", "candidate_positive_keys.parquet"),
    ):
        frame = pd.read_parquet(arguments.stage1 / filename)
        frame = frame.sort_values(["galaxy_id", "s3_key"], kind="stable")
        for row in frame.head(arguments.per_bucket).itertuples():
            selected.append((bucket, str(row.galaxy_id), str(row.s3_key)))

    records: list[dict[str, object]] = []
    for bucket, galaxy_id, key in selected:
        destination = arguments.output_dir / bucket / f"{galaxy_id}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = ["aws"]
        if arguments.profile:
            command.extend(["--profile", arguments.profile])
        command.extend(
            [
            "s3",
            "cp",
            "--only-show-errors",
            f"s3://darkhaloscope/{key}",
            str(destination),
            ]
        )
        subprocess.run(command, check=True)
        records.append(
            {
                "bucket": bucket,
                "galaxy_id": galaxy_id,
                "s3_key": key,
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                **inspect_npz(destination),
            }
        )

    report = {
        "schema_version": 1,
        "status": "score_blind_payload_pilot_only",
        "selection": "first identifiers in lexicographic order within each bucket",
        "per_bucket": arguments.per_bucket,
        "detector_scores_used": False,
        "records": records,
    }
    output = arguments.output_dir / "report.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
