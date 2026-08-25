#!/usr/bin/env python3
"""Validate that public train, validation, and test manifests are disjoint."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

GROUP_COLUMNS = (
    "galaxy_id",
    "cutout_relpath",
    "healpix_128",
    "physical_group_id",
    "spatial_component_id",
)


def _values(frame: pd.DataFrame, column: str) -> set[object]:
    return set(frame[column].dropna().tolist())


def _unit_vectors(frame: pd.DataFrame) -> np.ndarray:
    ra = np.deg2rad(frame["ra"].to_numpy(dtype=float))
    dec = np.deg2rad(frame["dec"].to_numpy(dtype=float))
    cos_dec = np.cos(dec)
    return np.column_stack(
        (cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec))
    )


def validate(
    manifests: dict[str, pd.DataFrame],
    *,
    minimum_separation_arcsec: float = 38.0,
) -> None:
    required = {*GROUP_COLUMNS, "ra", "dec", "label", "split", "endpoint_role"}
    for name, frame in manifests.items():
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if frame["galaxy_id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate galaxy_id values")
        if frame["cutout_relpath"].astype(str).str.startswith("/").any():
            raise ValueError(f"{name} contains absolute cutout paths")

    for (left_name, left), (right_name, right) in combinations(
        manifests.items(), 2
    ):
        for column in GROUP_COLUMNS:
            overlap = _values(left, column) & _values(right, column)
            if overlap:
                raise ValueError(
                    f"{left_name}/{right_name} overlap in {column}: "
                    f"{len(overlap)} values"
                )

        radius = 2.0 * np.sin(
            np.deg2rad(minimum_separation_arcsec / 3600.0) / 2.0
        )
        neighbours = cKDTree(_unit_vectors(left)).query_ball_tree(
            cKDTree(_unit_vectors(right)),
            radius,
        )
        close_pairs = sum(bool(matches) for matches in neighbours)
        if close_pairs:
            raise ValueError(
                f"{left_name}/{right_name} contain {close_pairs} rows with "
                f"a cross-split neighbour within {minimum_separation_arcsec} arcsec"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--minimum-separation-arcsec", type=float, default=38.0)
    arguments = parser.parse_args()

    manifests = {
        "train": pd.read_parquet(arguments.train),
        "validation": pd.read_parquet(arguments.validation),
        "test": pd.read_parquet(arguments.test),
    }
    validate(
        manifests,
        minimum_separation_arcsec=arguments.minimum_separation_arcsec,
    )
    print(
        "PASS: "
        + ", ".join(f"{name}={len(frame)}" for name, frame in manifests.items())
    )


if __name__ == "__main__":
    main()
