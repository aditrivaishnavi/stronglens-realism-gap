#!/usr/bin/env python3
"""Render all four primary-detector misses in the selected 43-system cohort."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dhs.feature_encoders import lupton_rgb

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = np.float32(1.5717214345932007)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_misses(manifest_path: Path, scores_path: Path) -> pd.DataFrame:
    manifest = pd.read_parquet(manifest_path)
    scores = pd.read_parquet(scores_path)
    keys = [
        "endpoint_id",
        "galaxy_id",
        "global_role",
        "spatial_group_id",
        "physical_group_id",
    ]
    frame = manifest.merge(scores, on=keys, validate="one_to_one")
    missed = frame.loc[~frame["detected_at_frozen_threshold"]].copy()
    if (
        len(frame) != 43
        or int(frame["detected_at_frozen_threshold"].sum()) != 39
        or len(missed) != 4
    ):
        raise RuntimeError("selected-system endpoint is not the final 39/43 result")
    recomputed = missed["raw_logit_float32"].to_numpy(dtype=np.float32) >= THRESHOLD
    if recomputed.any():
        raise RuntimeError("missed-system decision differs from native threshold rule")
    return missed.sort_values("galaxy_id").reset_index(drop=True)


def names_by_id(provenance_path: Path) -> dict[str, str]:
    frame = pd.read_csv(provenance_path, usecols=["galaxy_id", "catalogue_name"])
    return dict(
        zip(
            frame["galaxy_id"].astype(str),
            frame["catalogue_name"].astype(str),
            strict=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "results_v2/evaluation_v2/terminal_confirmed_endpoint/endpoint_manifest.parquet",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=ROOT
        / "results_v2/evaluation_v2/terminal_confirmed_scores/endpoint_scores.parquet",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=ROOT / "data/catalogues/lens_provenance.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "manuscript/figures/appendix_missed_selected_systems",
    )
    arguments = parser.parse_args()

    misses = selected_misses(arguments.manifest, arguments.scores)
    names = names_by_id(arguments.provenance)
    figure, axes = plt.subplots(1, 4, figsize=(7.2, 2.45))
    for axis, row in zip(axes.ravel(), misses.itertuples(index=False), strict=True):
        payload_path = ROOT / str(row.payload_path)
        if sha256(payload_path) != str(row.payload_file_sha256):
            raise RuntimeError(f"payload hash differs: {payload_path}")
        with np.load(payload_path, allow_pickle=False) as archive:
            cutout = np.asarray(archive[str(row.pixel_array_key)], dtype=np.float32)
        if cutout.shape != (101, 101, 3) or not np.isfinite(cutout).all():
            raise RuntimeError(f"unexpected cutout: {payload_path}")
        axis.imshow(lupton_rgb(cutout), origin="lower", interpolation="nearest")
        axis.set_title(
            f"{names[str(row.galaxy_id)]}\n"
            f"logit {float(row.raw_logit_float32):.3f} < {float(THRESHOLD):.3f}",
            fontsize=6.5,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#374151")
    figure.suptitle(
        "All four non-detections in the selected 43-system endpoint",
        fontsize=9.5,
        weight="bold",
    )
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.78, wspace=0.08)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        arguments.output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04
    )
    figure.savefig(
        arguments.output.with_suffix(".png"),
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
