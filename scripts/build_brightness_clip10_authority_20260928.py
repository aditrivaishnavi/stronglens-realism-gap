#!/usr/bin/env python3
"""Build the retained clip-10 brightness/photon authority from existing scores."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results_v2/revision_controls_v1"
OUTPUT = ROOT / "results_v2/revision_controls_v1/retained_clip10"
THRESHOLD = np.float32(1.5717214345932007)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    full_scores_path = SOURCE / "brightness_scores.parquet"
    full_result_path = SOURCE / "BRIGHTNESS_STRESS_RESULT.json"
    design_path = SOURCE / "brightness_design.parquet"
    images_path = SOURCE / "brightness_derived_images.parquet"
    scores = pd.read_parquet(full_scores_path)
    retained = scores[scores["clip_range"].eq(10.0)].reset_index(drop=True)
    if (
        len(retained) != 3096
        or retained["derived_image_id"].nunique() != 3096
        or set(retained["arm"]) != {"expectation", "photon"}
        or not np.array_equal(
            retained["detected_at_fixed_threshold"].to_numpy(dtype=bool),
            retained["raw_logit_float32"].to_numpy(dtype=np.float32)
            >= THRESHOLD,
        )
    ):
        raise RuntimeError("clip-10 brightness support or decisions differ")

    group_columns = [
        "pair_id",
        "target_galaxy_id",
        "target_source_r_mag",
        "arm",
    ]
    target = (
        retained.groupby(group_columns, as_index=False)
        .agg(
            mean_logit=("raw_logit_float32", "mean"),
            detection_rate=("detected_at_fixed_threshold", "mean"),
            draws=("derived_image_id", "size"),
            mean_source_flux=("source_flux_sum", "mean"),
            mean_source_peak=("source_peak", "mean"),
        )
        .sort_values(group_columns)
    )
    cells = (
        target.groupby(["target_source_r_mag", "arm"], as_index=False)
        .agg(
            targets=("target_galaxy_id", "nunique"),
            mean_logit=("mean_logit", "mean"),
            mean_detection_rate=("detection_rate", "mean"),
            target_sd_detection_rate=("detection_rate", "std"),
            mean_source_flux=("mean_source_flux", "mean"),
            mean_source_peak=("mean_source_peak", "mean"),
        )
        .sort_values(["target_source_r_mag", "arm"])
    )
    expectation = target[target["arm"].eq("expectation")].rename(
        columns={
            "mean_logit": "expectation_logit",
            "detection_rate": "expectation_detection_rate",
        }
    )
    photon = target[target["arm"].eq("photon")].rename(
        columns={
            "mean_logit": "photon_mean_logit",
            "detection_rate": "photon_detection_rate",
        }
    )
    keys = ["pair_id", "target_galaxy_id", "target_source_r_mag"]
    paired = expectation[
        keys + ["expectation_logit", "expectation_detection_rate"]
    ].merge(
        photon[keys + ["photon_mean_logit", "photon_detection_rate"]],
        on=keys,
        validate="one_to_one",
    )
    paired["photon_minus_expectation_logit"] = (
        paired["photon_mean_logit"] - paired["expectation_logit"]
    )
    paired["photon_minus_expectation_detection_rate"] = (
        paired["photon_detection_rate"]
        - paired["expectation_detection_rate"]
    )
    contrasts = (
        paired.groupby("target_source_r_mag", as_index=False)
        .agg(
            targets=("target_galaxy_id", "nunique"),
            photon_minus_expectation_logit=(
                "photon_minus_expectation_logit",
                "mean",
            ),
            photon_minus_expectation_detection_rate=(
                "photon_minus_expectation_detection_rate",
                "mean",
            ),
        )
        .sort_values("target_source_r_mag")
    )
    photon_rows = retained[retained["arm"].eq("photon")]
    draw_means = (
        photon_rows.groupby(["target_source_r_mag", "draw_index"], as_index=False)[
            "detected_at_fixed_threshold"
        ]
        .mean()
        .rename(columns={"detected_at_fixed_threshold": "cohort_rate"})
    )
    mcse = draw_means.groupby("target_source_r_mag", as_index=False).agg(
        draw_sd=("cohort_rate", "std"), draws=("draw_index", "nunique")
    )
    mcse["finite_photon_mcse"] = mcse["draw_sd"] / np.sqrt(mcse["draws"])

    score_output = OUTPUT / "brightness_scores_clip10.parquet"
    cell_output = OUTPUT / "brightness_cell_summary_clip10.parquet"
    target_output = OUTPUT / "brightness_target_summary_clip10.parquet"
    atomic_parquet(retained, score_output)
    atomic_parquet(cells, cell_output)
    atomic_parquet(target, target_output)
    result = {
        "schema_version": 1,
        "status": "retained_training_consistent_brightness_photon_response",
        "scope": "clip_10_only",
        "targets": 43,
        "magnitudes": sorted(retained["target_source_r_mag"].unique().tolist()),
        "expectation_images": int(
            retained[retained["arm"].eq("expectation")][
                "derived_image_id"
            ].nunique()
        ),
        "photon_images": int(
            retained[retained["arm"].eq("photon")]["derived_image_id"].nunique()
        ),
        "retained_score_rows": len(retained),
        "photon_draws_per_target_magnitude": 8,
        "gain_electron_per_nmgy": 150.0,
        "clip_range": 10.0,
        "threshold": float(THRESHOLD),
        "threshold_float32_bits": int(THRESHOLD.view(np.uint32)),
        "photon_minus_expectation_contrasts": contrasts.to_dict("records"),
        "finite_photon_mcse": mcse.to_dict("records"),
        "source_bindings": {
            "design_path": str(design_path.relative_to(ROOT)),
            "design_sha256": sha256(design_path),
            "derived_images_path": str(images_path.relative_to(ROOT)),
            "derived_images_sha256": sha256(images_path),
            "full_score_execution_path": str(full_scores_path.relative_to(ROOT)),
            "full_score_execution_sha256": sha256(full_scores_path),
            "full_execution_result_path": str(full_result_path.relative_to(ROOT)),
            "full_execution_result_sha256": sha256(full_result_path),
        },
        "outputs": {
            str(score_output.relative_to(ROOT)): sha256(score_output),
            str(cell_output.relative_to(ROOT)): sha256(cell_output),
            str(target_output.relative_to(ROOT)): sha256(target_output),
        },
        "excluded_from_scientific_evidence": [
            "clip_20_scores",
            "clip_20_minus_clip_10_contrasts",
            "photon_by_clipping_interactions",
            "alternative_annulus_rescoring"
        ],
        "interpretation": (
            "fixed-target source-brightness and photon response under the "
            "trained clip-10 preprocessing; not calibrated visibility or completeness"
        ),
    }
    atomic_json(result, OUTPUT / "BRIGHTNESS_CLIP10_RESULT.json")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
