#!/usr/bin/env python3
"""Verify the committed renderer-reconciliation evidence bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results_v2/grec/renderer_reconciliation_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify() -> dict[str, object]:
    receipt_path = EVIDENCE / "RENDERER_RECONCILIATION_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    bindings = {
        "sample_manifest.parquet": receipt["sample_manifest_sha256"],
        "replay_metrics.parquet": receipt["replay_metrics_sha256"],
        "optical_reference_metrics.parquet": receipt[
            "optical_reference_metrics_sha256"
        ],
        "independent_reference_arcs.npz": receipt["reference_arcs_sha256"],
    }
    for name, expected in bindings.items():
        path = EVIDENCE / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"renderer evidence hash differs: {name}")

    manifest = pd.read_parquet(EVIDENCE / "sample_manifest.parquet")
    if len(manifest) != receipt["sample_rows"] or receipt["sample_rows"] != 142:
        raise RuntimeError("renderer sample count differs")
    for field in (
        "exact_replay_passed",
        "effective_psf_operator_passed",
        "independent_reference_convergence_passed",
        "production_convergence_passed",
    ):
        if receipt[field] is not True:
            raise RuntimeError(f"renderer gate failed: {field}")
    if receipt["production_support_sie"]["cases"] != 142:
        raise RuntimeError("renderer production-support case count differs")

    return {
        "status": "renderer_public_evidence_verified",
        "sample_rows": len(manifest),
        "receipt_sha256": sha256(receipt_path),
        "bindings": bindings,
    }


def main() -> int:
    print(json.dumps(verify(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
