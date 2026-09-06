#!/usr/bin/env python3
"""Render repaired-host P_noise children in parallel and merge retained rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def render_shard(
    index: int,
    rows: list[dict[str, object]],
    config_path: str,
    output_dir: str,
    draw_receipt_path: str,
) -> tuple[int, str, int]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    from grec_pnoise.contracts import load_config
    from grec_pnoise.pnoise import render_photon_children

    config = load_config(config_path)
    receipt = json.loads(Path(draw_receipt_path).read_text())
    manifest, _ = render_photon_children(
        pd.DataFrame(rows),
        config,
        output_dir,
        receipt,
        limit=len(rows),
        canary_label="design_only_canonical_repair_render_shard",
    )
    shard_dir = Path(output_dir) / "repair_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"shard-{index:05d}.parquet"
    temporary = shard_path.with_suffix(".partial.parquet")
    manifest.to_parquet(temporary, index=False)
    temporary.replace(shard_path)
    return index, str(shard_path), len(manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--old-parents", type=Path, required=True)
    parser.add_argument("--old-child-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--draw-receipt", type=Path, required=True)
    parser.add_argument("--threshold-receipt", type=Path, required=True)
    parser.add_argument("--parent-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")

    parents = pd.read_parquet(args.parents)
    old_parents = pd.read_parquet(args.old_parents)
    retained_parent_ids = set(parents["grec_row_id"].astype(str)) & set(
        old_parents["grec_row_id"].astype(str)
    )
    if len(retained_parent_ids) != 37_004:
        raise ValueError("retained parent support differs from 37,004")
    new_parents = parents[
        ~parents["grec_row_id"].astype(str).isin(retained_parent_ids)
    ].copy()
    if len(new_parents) != 12_496:
        raise ValueError("replacement parent support differs from 12,496")

    groups = [
        group.to_dict(orient="records")
        for _, group in new_parents.groupby("host_galaxy_id", sort=True)
    ]
    tasks = [
        sum(groups[start : start + 8], [])
        for start in range(0, len(groups), 8)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"new_parents={len(new_parents)} replacement_hosts={len(groups)} "
        f"tasks={len(tasks)} workers={args.workers}",
        flush=True,
    )
    shard_paths: list[Path] = []
    completed_rows = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = {
            executor.submit(
                render_shard,
                index,
                task,
                str(args.config.resolve()),
                str(args.output_dir.resolve()),
                str(args.draw_receipt.resolve()),
            )
            for index, task in enumerate(tasks)
        }
        completed = 0
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                _, path, rows = future.result()
                shard_paths.append(Path(path))
                completed += 1
                completed_rows += rows
            if completed % 10 == 0 or not pending:
                print(
                    f"completed_tasks={completed}/{len(tasks)} "
                    f"child_rows={completed_rows}/{len(new_parents) * 8}",
                    flush=True,
                )

    new_children = pd.concat(
        [pd.read_parquet(path) for path in sorted(shard_paths)],
        ignore_index=True,
    )
    if len(new_children) != len(new_parents) * 8:
        raise ValueError("replacement child support is incomplete")
    old_children = pd.read_parquet(args.old_child_manifest)
    retained_children = old_children[
        old_children["parent_grec_row_id"].astype(str).isin(retained_parent_ids)
    ].copy()
    if len(retained_children) != len(retained_parent_ids) * 8:
        raise ValueError("retained child support is incomplete")
    combined = pd.concat(
        [retained_children, new_children], ignore_index=True, sort=False
    ).sort_values(["parent_grec_row_id", "draw_index"], ignore_index=True)
    if (
        len(combined) != 396_000
        or combined["pnoise_row_id"].astype(str).duplicated().any()
    ):
        raise ValueError("combined child bank is invalid")
    manifest_path = args.output_dir / "pnoise_child_manifest.parquet"
    temporary = manifest_path.with_suffix(".partial.parquet")
    combined.to_parquet(temporary, index=False)
    temporary.replace(manifest_path)

    old_receipt = json.loads(
        (
            ROOT / "results_v2/pnoise/physical/PNOISE_RENDER_RECEIPT.json"
        ).read_text()
    )
    receipt = {
        **old_receipt,
        "status": "pnoise_physical_bank_frozen_pre_score",
        "rows": len(combined),
        "parent_scenes": int(combined["parent_scene_id"].nunique()),
        "child_manifest_sha256": sha256_file(manifest_path),
        "parent_manifest_sha256": sha256_file(args.parents),
        "parent_receipt_sha256": sha256_file(args.parent_receipt),
        "draw_count_receipt_sha256": sha256_file(args.draw_receipt),
        "threshold_receipt_sha256": sha256_file(args.threshold_receipt),
        "canonical_repair": {
            "retained_child_rows": len(retained_children),
            "new_child_rows": len(new_children),
            "old_child_manifest_sha256": sha256_file(args.old_child_manifest),
            "shards": len(shard_paths),
        },
    }
    atomic_json(args.output_dir / "PNOISE_RENDER_RECEIPT.json", receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
