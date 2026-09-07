#!/usr/bin/env python3
"""Run the sealed secondary attribution of G_domain pixel statistics."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn

from stronglens_probes.gdomain_pixel_attribution import (
    PIXEL_FEATURE_NAMES,
    attribution_suite,
    build_row_plan,
    load_config,
    sha256_file,
    validate_input_bindings,
    validate_specification,
)


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def input_path(root: Path, config: dict[str, Any], key: str) -> Path:
    return root / str(config["inputs"][key]["path"])


def run(root: Path, config_path: Path) -> Path:
    config = load_config(config_path)
    validate_input_bindings(root, config)
    validate_specification(config)
    output = root / "results_v2/probes/gdomain_pixel_attribution_v1"
    if output.exists():
        raise FileExistsError(f"pixel-attribution output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".gdomain-pixel-attribution-v1.", dir=output.parent)
    )
    try:
        manifest = pd.read_parquet(input_path(root, config, "manifest"))
        authoritative_oof = pd.read_parquet(
            input_path(root, config, "authoritative_oof")
        )
        folds = pd.read_parquet(input_path(root, config, "folds"))
        features = np.load(
            input_path(root, config, "feature_matrix"),
            allow_pickle=False,
        )
        final_result = json.loads(
            input_path(root, config, "final_result").read_text(encoding="utf-8")
        )
        plan = build_row_plan(manifest, authoritative_oof, folds)
        results, oof, coefficients = attribution_suite(
            plan,
            features,
            probe_config=config["probe"],
        )
        full_auc = float(
            results.loc[results["analysis"].eq("full_model"), "auc"].iloc[0]
        )
        expected_auc = float(
            final_result["results"]["pixel_statistics"]["primary_oof_auc"]
        )
        if not np.isclose(full_auc, expected_auc, rtol=0.0, atol=1e-12):
            raise RuntimeError("full pixel model does not reproduce final AUC")
        authoritative_columns = [
            "row_id",
            "dependency_component_id",
            "fold",
            "label",
            "analysis_weight",
            "oof_score",
        ]
        full_oof = oof[
            oof["analysis"].eq("full_model")
            & oof["feature_or_band"].eq("all")
        ]
        for column in ("row_id", "dependency_component_id", "fold", "label"):
            if not np.array_equal(
                full_oof[column].to_numpy(),
                authoritative_oof[column].to_numpy(),
            ):
                raise RuntimeError(f"full pixel OOF differs in {column}")
        for column in ("analysis_weight", "oof_score"):
            if not np.allclose(
                full_oof[column].to_numpy(float),
                authoritative_oof[column].to_numpy(float),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(f"full pixel OOF differs in {column}")
        if not set(authoritative_columns) <= set(authoritative_oof):
            raise RuntimeError("authoritative pixel OOF columns are incomplete")

        results.to_parquet(temporary / "pixel_attribution.parquet", index=False)
        oof.to_parquet(temporary / "pixel_attribution_oof.parquet", index=False)
        coefficients.to_parquet(
            temporary / "pixel_coefficient_summary.parquet", index=False
        )
        result_payload = {
            "schema_version": 1,
            "status": "complete",
            "diagnostic_id": config["diagnostic_id"],
            "scope": config["scientific_scope"],
            "full_auc": full_auc,
            "feature_order": list(PIXEL_FEATURE_NAMES),
            "rows": results.to_dict(orient="records"),
        }
        write_json(temporary / "GDOMAIN_PIXEL_ATTRIBUTION_RESULTS.json", result_payload)

        files = sorted(path for path in temporary.iterdir() if path.is_file())
        receipt = {
            "schema_version": 1,
            "status": "complete",
            "diagnostic_id": config["diagnostic_id"],
            "protocol_path": str(config_path.relative_to(root)),
            "protocol_sha256": sha256_file(config_path),
            "source_code_sha256": sha256_file(
                root
                / "experiments_v2/probes/src/stronglens_probes/"
                "gdomain_pixel_attribution.py"
            ),
            "contracts_code_sha256": sha256_file(
                root
                / "experiments_v2/probes/src/stronglens_probes/contracts.py"
            ),
            "runner_sha256": sha256_file(Path(__file__)),
            "runtime_environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyarrow": pyarrow.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "inputs": config["inputs"],
            "counts": {
                "rows": len(plan),
                "dependency_components": int(
                    plan["dependency_component_id"].nunique()
                ),
                "attribution_models": len(results),
            },
            "output_files": {
                path.name: {
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in files
            },
        }
        write_json(temporary / "GDOMAIN_PIXEL_ATTRIBUTION_RECEIPT.json", receipt)
        os.replace(temporary, output)
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "experiments_v2/probes/configs/gdomain_pixel_attribution_v1.yaml"
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = root / config_path
    print(run(root, config_path.resolve()))


if __name__ == "__main__":
    main()
