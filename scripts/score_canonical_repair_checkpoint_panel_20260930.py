#!/usr/bin/env python3
"""Score repaired-only G_rec rows across the frozen twelve-checkpoint panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from checkpoint_sensitivity_v2.pipeline import _load_models, _score_images
from grec_pnoise.contracts import load_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel_path = ROOT / "experiments_v2/checkpoint_sensitivity_v2/config/panel.json"
    panel = json.loads(panel_path.read_text())
    panel["__config_path__"] = str(panel_path)
    panel["__config_sha256__"] = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    grec_config = load_config(
        ROOT / "experiments_v2/grec_pnoise/config/production.toml"
    )
    repaired = pd.read_parquet(
        ROOT
        / "results_v2/canonical_test_host_repair_v1/grec/parents/"
        "grec_parent_manifest.parquet"
    )
    old_parent = pd.read_parquet(
        ROOT / "results_v2/grec/parents/grec_parent_manifest.parquet",
        columns=["grec_row_id"],
    )
    old_ids = set(old_parent["grec_row_id"].astype(str))
    new_rows = repaired[
        ~repaired["grec_row_id"].astype(str).isin(old_ids)
    ].copy()
    if len(new_rows) != 12_496:
        raise ValueError("replacement scene support differs from 12,496")

    def images():
        for row in new_rows.itertuples(index=False):
            with np.load(str(row.parent_pixel_path), allow_pickle=False) as payload:
                yield np.asarray(payload["expectation_chw"], dtype=np.float32)

    models, preprocess_kwargs, fixed, preprocess = _load_models(panel, grec_config)
    scores, batch_record = _score_images(
        images(),
        rows=len(new_rows),
        models=models,
        preprocess_kwargs=preprocess_kwargs,
        fixed=fixed,
        preprocess=preprocess,
        device=str(panel["device"]),
        batch_size=int(panel["batch_size"]),
        chunk_rows=2048,
        label="canonical_repair_grec",
    )
    new_score_table = pd.DataFrame({"grec_row_id": new_rows["grec_row_id"]})
    for model_id, values in scores.items():
        new_score_table[f"{model_id}__raw_logit_float32"] = values
        new_score_table[f"{model_id}__raw_logit_float32_bits"] = values.view(
            np.uint32
        )
    old_scores = pd.read_parquet(
        ROOT / "results_v2/checkpoint_sensitivity_v2/grec/panel_logits.parquet"
    )
    retained_ids = set(repaired["grec_row_id"].astype(str)) & set(
        old_scores["grec_row_id"].astype(str)
    )
    retained = old_scores[
        old_scores["grec_row_id"].astype(str).isin(retained_ids)
    ]
    combined = pd.concat(
        [retained, new_score_table], ignore_index=True
    ).sort_values("grec_row_id", ignore_index=True)
    if len(combined) != 49_500 or combined["grec_row_id"].duplicated().any():
        raise ValueError("combined checkpoint score support is invalid")
    scores_path = args.output_dir / "panel_logits.parquet"
    combined.to_parquet(scores_path, index=False)
    thresholds = json.loads(
        (
            ROOT
            / "results_v2/checkpoint_sensitivity_v2/"
            "MODEL_SPECIFIC_THRESHOLDS.json"
        ).read_text()
    )["models"]
    summaries = {}
    for model_id in models:
        values = combined[f"{model_id}__raw_logit_float32"].to_numpy(
            dtype=np.float32
        )
        detected = values >= np.float32(thresholds[model_id]["logit_threshold"])
        summaries[model_id] = {
            "detections": int(detected.sum()),
            "recovery": float(detected.mean()),
        }
    payload = {
        "schema_version": 1,
        "status": "canonical_repair_checkpoint_grec_complete",
        "retained_rows": len(retained),
        "new_rows": len(new_score_table),
        "batch_record": batch_record,
        "models": summaries,
        "scores_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
    }
    (args.output_dir / "CHECKPOINT_GREC_REPAIR_RESULT.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(args.output_dir / "CHECKPOINT_GREC_REPAIR_RESULT.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
