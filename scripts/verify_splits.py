#!/usr/bin/env python3
"""Verify train/val/test split disjointness by galaxy_id and cutout_path.

NOTE: This script checks *sample-level* disjointness (no shared galaxy_id or
cutout_path between splits).  For *spatial* (HEALPix) disjointness verification,
see ``scripts/investigate_healpix_nan.py``, which recomputes HEALPix pixel
assignments from RA/Dec and checks for overlapping pixels between splits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Set

from common.manifest_utils import (
    CUTOUT_PATH_COL,
    GALAXY_ID_COL,
    LABEL_COL,
    SPLIT_COL,
    ensure_split_values,
    load_manifest,
    safe_json_dump,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_json", default="split_verification_report.json")
    ap.add_argument("--fail_on_overlap", action="store_true")
    args = ap.parse_args()

    df = load_manifest(args.manifest)
    ensure_split_values(df)

    report: Dict = {
        "manifest": str(Path(args.manifest).resolve()),
        "row_count": int(len(df)),
        "split_counts": df[SPLIT_COL].value_counts(dropna=False).to_dict(),
        "label_counts": df[LABEL_COL].value_counts(dropna=False).to_dict(),
    }
    null_split = df[SPLIT_COL].isna()
    if null_split.any():
        report["null_split_count"] = int(null_split.sum())
        report["note"] = "Overlap checks use only rows with non-null split."

    missing = [c for c in [GALAXY_ID_COL, CUTOUT_PATH_COL] if c not in df.columns]
    if missing:
        report["warnings"] = [f"Missing columns for overlap checks: {missing}"]
        safe_json_dump(report, args.out_json)
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    df2 = df[[SPLIT_COL, GALAXY_ID_COL, CUTOUT_PATH_COL]].copy()
    df2[GALAXY_ID_COL] = df2[GALAXY_ID_COL].astype(str)
    df2[CUTOUT_PATH_COL] = df2[CUTOUT_PATH_COL].astype(str)

    def get_set(split: str, col: str) -> Set[str]:
        return set(df2[df2[SPLIT_COL] == split][col].dropna().tolist())

    overlaps = {}
    for col in [GALAXY_ID_COL, CUTOUT_PATH_COL]:
        train = get_set("train", col)
        val = get_set("val", col)
        test = get_set("test", col)
        overlaps[col] = {
            "train_val": len(train & val),
            "train_test": len(train & test),
            "val_test": len(val & test),
        }
    report["overlaps"] = overlaps

    dupes = {}
    for split in ["train", "val", "test"]:
        part = df2[df2[SPLIT_COL] == split]
        dupes[split] = int(part.duplicated(subset=[GALAXY_ID_COL, CUTOUT_PATH_COL]).sum())
    report["duplicate_rows_within_split"] = dupes

    errors = []
    if any(v > 0 for v in overlaps[GALAXY_ID_COL].values()):
        errors.append("galaxy_id overlaps exist between splits")
    if any(v > 0 for v in overlaps[CUTOUT_PATH_COL].values()):
        errors.append("cutout_path overlaps exist between splits")
    if errors:
        report["errors"] = errors

    safe_json_dump(report, args.out_json)
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.fail_on_overlap and errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
