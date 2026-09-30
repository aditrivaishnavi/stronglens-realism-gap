#!/usr/bin/env python3
"""Render missing canonical-repair G_rec scenes in host-closed processes."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def render_host(
    rows: list[dict[str, object]],
    config_path: str,
    output_dir: str,
    validation_receipt: str,
) -> tuple[str, int]:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    import torch
    from grec_pnoise.contracts import load_config
    from grec_pnoise.rendering import render_parent_bank

    torch.set_num_threads(2)
    table = pd.DataFrame(rows)
    config = load_config(config_path)
    render_parent_bank(
        table,
        config,
        output_dir,
        device="cpu",
        renderer_validation_receipt_path=validation_receipt,
    )
    return str(table.iloc[0]["host_galaxy_id"]), len(table)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--renderer-validation-receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")

    scenes = pd.read_parquet(args.scenes)
    missing_mask = [
        not (
            args.output_dir
            / "grec_parents"
            / str(scene_id)[:2]
            / f"{scene_id}.npz"
        ).is_file()
        for scene_id in scenes["scene_id"]
    ]
    missing = scenes.loc[missing_mask].copy()
    groups = [
        group.to_dict(orient="records")
        for _, group in missing.groupby("host_galaxy_id", sort=True)
    ]
    print(
        f"missing_scenes={len(missing)} missing_hosts={len(groups)} "
        f"workers={args.workers}",
        flush=True,
    )
    if not groups:
        return 0

    completed_hosts = 0
    completed_scenes = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = {
            executor.submit(
                render_host,
                group,
                str(args.config.resolve()),
                str(args.output_dir.resolve()),
                str(args.renderer_validation_receipt.resolve()),
            )
            for group in groups
        }
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                _, count = future.result()
                completed_hosts += 1
                completed_scenes += count
            if completed_hosts % 25 == 0 or not pending:
                print(
                    f"completed_hosts={completed_hosts}/{len(groups)} "
                    f"completed_scenes={completed_scenes}/{len(missing)}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
