#!/usr/bin/env python3
"""Refresh repaired P_noise and P_src estimates across twelve checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from checkpoint_sensitivity_v2.pipeline import (
    _load_models,
    _photon_images,
    _raw_images,
    _score_images,
)
from grec_pnoise.contracts import load_config

ROOT = Path(__file__).resolve().parents[1]


def score_table(
    frame: pd.DataFrame,
    images,
    identity: pd.DataFrame,
    *,
    models,
    preprocess_kwargs,
    fixed,
    preprocess,
    panel,
    label: str,
) -> pd.DataFrame:
    scores, _ = _score_images(
        images,
        rows=len(identity),
        models=models,
        preprocess_kwargs=preprocess_kwargs,
        fixed=fixed,
        preprocess=preprocess,
        device=str(panel["device"]),
        batch_size=int(panel["batch_size"]),
        chunk_rows=2048,
        label=label,
    )
    result = identity.copy()
    for model_id, values in scores.items():
        result[f"{model_id}__raw_logit_float32"] = values
        result[f"{model_id}__raw_logit_float32_bits"] = values.view(np.uint32)
    return result


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
    models, preprocess_kwargs, fixed, preprocess = _load_models(panel, grec_config)
    thresholds = json.loads(
        (
            ROOT
            / "results_v2/checkpoint_sensitivity_v2/"
            "MODEL_SPECIFIC_THRESHOLDS.json"
        ).read_text()
    )["models"]

    pnoise = pd.read_parquet(
        ROOT
        / "results_v2/canonical_test_host_repair_v1/pnoise/physical/"
        "pnoise_child_manifest.parquet"
    )
    old_parent_ids = set(
        pd.read_parquet(
            ROOT / "results_v2/grec/parents/grec_parent_manifest.parquet",
            columns=["grec_row_id"],
        )["grec_row_id"].astype(str)
    )
    new_pnoise = pnoise[
        ~pnoise["parent_grec_row_id"].astype(str).isin(old_parent_ids)
    ].copy()
    new_pnoise_scores = score_table(
        new_pnoise,
        _raw_images(new_pnoise, "pnoise"),
        new_pnoise[["pnoise_row_id"]],
        models=models,
        preprocess_kwargs=preprocess_kwargs,
        fixed=fixed,
        preprocess=preprocess,
        panel=panel,
        label="canonical_repair_pnoise",
    )
    old_pnoise_scores = pd.read_parquet(
        ROOT / "results_v2/checkpoint_sensitivity_v2/pnoise/panel_logits.parquet"
    )
    retained_child_ids = set(pnoise["pnoise_row_id"].astype(str)) - set(
        new_pnoise["pnoise_row_id"].astype(str)
    )
    combined_pnoise_scores = pd.concat(
        [
            old_pnoise_scores[
                old_pnoise_scores["pnoise_row_id"]
                .astype(str)
                .isin(retained_child_ids)
            ],
            new_pnoise_scores,
        ],
        ignore_index=True,
    ).sort_values("pnoise_row_id", ignore_index=True)
    if len(combined_pnoise_scores) != 396_000:
        raise ValueError("combined P_noise panel support is incomplete")
    combined_pnoise_scores.to_parquet(
        args.output_dir / "pnoise_panel_logits.parquet", index=False
    )
    grec_scores = pd.read_parquet(
        ROOT
        / "results_v2/canonical_test_host_repair_v1/checkpoint_sensitivity/"
        "grec/panel_logits.parquet"
    )
    parent = pd.read_parquet(
        ROOT
        / "results_v2/canonical_test_host_repair_v1/grec/parents/"
        "grec_parent_manifest.parquet"
    )

    psrc = pd.read_parquet(
        ROOT
        / "results_v2/canonical_test_host_repair_v1/psrc/photons/manifest.parquet"
    ).sort_values(["pair_id", "draw_index"], ignore_index=True)
    psrc_root = (
        ROOT / "results_v2/canonical_test_host_repair_v1/psrc/photons"
    )
    psrc_identity = psrc.loc[psrc.index.repeat(2)].reset_index(drop=True)[
        ["pair_id", "rendered_pair_row_id", "draw_index"]
    ]
    psrc_identity["arm"] = np.tile(["structured", "sersic"], len(psrc))
    psrc_scores = score_table(
        psrc,
        _photon_images(psrc, psrc_root),
        psrc_identity,
        models=models,
        preprocess_kwargs=preprocess_kwargs,
        fixed=fixed,
        preprocess=preprocess,
        panel=panel,
        label="canonical_repair_psrc",
    )
    psrc_scores.to_parquet(args.output_dir / "psrc_panel_logits.parquet", index=False)

    results = {}
    for model_id in models:
        threshold = np.float32(thresholds[model_id]["logit_threshold"])
        child = pnoise[["pnoise_row_id", "parent_grec_row_id"]].merge(
            combined_pnoise_scores[
                ["pnoise_row_id", f"{model_id}__raw_logit_float32"]
            ],
            on="pnoise_row_id",
            validate="one_to_one",
        )
        child["decision"] = (
            child[f"{model_id}__raw_logit_float32"].to_numpy(np.float32)
            >= threshold
        )
        physical = child.groupby("parent_grec_row_id")["decision"].mean()
        scene = parent[
            [
                "grec_row_id",
                "host_galaxy_id",
                "psf_bin",
                "depth_bin",
                "theta_e_arcsec",
            ]
        ].merge(
            grec_scores[
                ["grec_row_id", f"{model_id}__raw_logit_float32"]
            ],
            on="grec_row_id",
            validate="one_to_one",
        )
        scene["expectation"] = (
            scene[f"{model_id}__raw_logit_float32"].to_numpy(np.float32)
            >= threshold
        )
        scene["physical"] = scene["grec_row_id"].map(physical)
        pnoise_effect = float(
            (scene["physical"] - scene["expectation"])
            .groupby(scene["host_galaxy_id"])
            .mean()
            .mean()
        )
        scored = psrc_scores.copy()
        scored["decision"] = (
            scored[f"{model_id}__raw_logit_float32"].to_numpy(np.float32)
            >= threshold
        )
        arm = scored.groupby("arm")["decision"].mean()
        results[model_id] = {
            "pnoise_effect": pnoise_effect,
            "psrc_effect": float(arm["sersic"] - arm["structured"]),
            "psrc_structured_detections": int(
                scored.loc[scored["arm"].eq("structured"), "decision"].sum()
            ),
            "psrc_sersic_detections": int(
                scored.loc[scored["arm"].eq("sersic"), "decision"].sum()
            ),
        }
    payload = {
        "schema_version": 1,
        "status": "canonical_repair_intervention_checkpoint_panel_complete",
        "models": results,
    }
    (args.output_dir / "INTERVENTION_PANEL_REPAIR_RESULT.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(args.output_dir / "INTERVENTION_PANEL_REPAIR_RESULT.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
