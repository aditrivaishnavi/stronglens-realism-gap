#!/usr/bin/env python3
"""Versioned serialization correction for the bounded brightness control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow
import run_controls as base
import torch


def seal_path(root: Path) -> Path:
    return (
        root
        / "results_v2/canonical_test_host_repair_v1/revision_controls"
        / "BRIGHTNESS_PROTOCOL_V2_SEAL.json"
    )


def seal(root: Path) -> dict[str, Any]:
    config, config_path = base.load_config(root)
    base.validate_seal(root, config)
    design_path = base.outputs(root)["bright_design"]
    if not design_path.is_file():
        raise base.ControlError("the frozen v1 brightness design is missing")
    source = Path(__file__).resolve()
    payload = {
        "schema_version": 1,
        "status": "brightness_v2_serialization_correction_sealed_pre_score",
        "correction_scope": (
            "serialize the unchanged 128-bit PCG64DXSM seed as hexadecimal text"
        ),
        "v1_runner_path": str(Path(base.__file__).resolve().relative_to(root)),
        "v1_runner_sha256": base.sha256_file(Path(base.__file__).resolve()),
        "v1_protocol_seal_sha256": base.sha256_file(base.outputs(root)["seal"]),
        "v1_design_sha256": base.sha256_file(design_path),
        "v2_runner_path": str(source.relative_to(root)),
        "v2_runner_sha256": base.sha256_file(source),
        "protocol_path": str(config_path.relative_to(root)),
        "protocol_sha256": base.sha256_file(config_path),
        "new_brightness_scores_opened": False,
    }
    base.atomic_json(payload, seal_path(root))
    return payload


def validate_v2_seal(root: Path) -> dict[str, Any]:
    path = seal_path(root)
    if not path.is_file():
        raise base.ControlError("brightness-v2 protocol is not sealed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status")
        != "brightness_v2_serialization_correction_sealed_pre_score"
        or payload.get("v1_runner_sha256")
        != base.sha256_file(Path(base.__file__).resolve())
        or payload.get("v1_protocol_seal_sha256")
        != base.sha256_file(base.outputs(root)["seal"])
        or payload.get("v1_design_sha256")
        != base.sha256_file(base.outputs(root)["bright_design"])
        or payload.get("v2_runner_sha256")
        != base.sha256_file(Path(__file__).resolve())
    ):
        raise base.ControlError("brightness-v2 protocol seal differs")
    return payload


def execute(root: Path) -> dict[str, Any]:
    config, _ = base.load_config(root)
    seal_payload = validate_v2_seal(root)
    paths = base.outputs(root)
    for key in (
        "bright_images",
        "bright_scores",
        "bright_cells",
        "bright_targets",
        "bright_result",
        "bright_figure",
    ):
        if paths[key].exists():
            raise FileExistsError(f"refusing to overwrite {paths[key]}")

    parents = base._matched_source_rows(root, config)
    parents_by_pair = {item["pair_id"]: item for item in parents}
    regenerated_design = base._make_brightness_design(parents, config)
    frozen_design = pd.read_parquet(paths["bright_design"])
    if (
        list(regenerated_design.columns) != list(frozen_design.columns)
        or not regenerated_design.equals(frozen_design)
    ):
        raise base.ControlError("regenerated brightness design differs from v1")

    derived = list(base._iter_derived(parents_by_pair, frozen_design, config))
    records = []
    for record, _ in derived:
        item = dict(record)
        seed = int(item.pop("draw_seed"))
        item["draw_seed_hex"] = (
            "not_applicable_expectation" if seed < 0 else f"{seed:032x}"
        )
        records.append(item)
    image_manifest = pd.DataFrame(records)
    if (
        len(image_manifest) != 3096
        or image_manifest["derived_image_id"].duplicated().any()
        or image_manifest["derived_image_sha256"].duplicated().any()
    ):
        raise base.ControlError("derived brightness-image support differs")
    base.atomic_parquet(image_manifest, paths["bright_images"])

    model, device, batch_contract = base._load_model(root, config)
    threshold = np.asarray(
        config["threshold"]["float32_value"], dtype=np.float32
    )
    score_frames = []
    inference_records = {}
    for clip in config["brightness"]["clip_ranges"]:
        logits, record = base._score_images(
            model,
            device,
            (image for _, image in derived),
            clip_range=float(clip),
            batch_size=int(config["inference"]["batch_size"]),
        )
        frame = image_manifest.copy()
        frame["clip_range"] = float(clip)
        frame["raw_logit_float32"] = logits
        frame["raw_logit_float32_bits"] = logits.view(np.uint32)
        frame["detected_at_fixed_threshold"] = logits >= threshold
        score_frames.append(frame)
        inference_records[f"clip_{float(clip):g}"] = record
    scores = pd.concat(score_frames, ignore_index=True)
    if len(scores) != 6192:
        raise base.ControlError("brightness score support differs")
    base.atomic_parquet(scores, paths["bright_scores"])
    cell_summary, target_summary, result = base._analyze_brightness(
        scores, config, threshold
    )
    base.atomic_parquet(cell_summary, paths["bright_cells"])
    base.atomic_parquet(target_summary, paths["bright_targets"])
    base._plot_brightness(cell_summary, paths["bright_figure"])
    result.update(
        {
            "schema_version": 2,
            "status": "bounded_brightness_photon_clipping_stress_complete",
            "brightness_protocol_v2_seal_sha256": base.sha256_file(
                seal_path(root)
            ),
            "v1_protocol_seal_sha256": base.sha256_file(paths["seal"]),
            "design_sha256": base.sha256_file(paths["bright_design"]),
            "derived_manifest_sha256": base.sha256_file(paths["bright_images"]),
            "scores_sha256": base.sha256_file(paths["bright_scores"]),
            "cell_summary_sha256": base.sha256_file(paths["bright_cells"]),
            "target_summary_sha256": base.sha256_file(paths["bright_targets"]),
            "figure_sha256": base.sha256_file(paths["bright_figure"]),
            "batch_contract": batch_contract,
            "inference_records": inference_records,
            "serialization_correction": (
                "full 128-bit draw seed stored as 32-character hexadecimal text"
            ),
            "runtime": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyarrow": pyarrow.__version__,
                "device": device,
            },
            "v2_seal_status": seal_payload["status"],
        }
    )
    base.atomic_json(result, paths["bright_result"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal", "execute"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    result = seal(root) if args.command == "seal" else execute(root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
