#!/usr/bin/env python3
"""Hard-link byte-identical retained G_rec artifacts into the repair tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD_HOSTS = ROOT / "results_v2/grec/hosts/final_grec_hosts.parquet"
OLD_SCENES = ROOT / "results_v2/grec/scenes/grec_scene_table.parquet"
OLD_PARENTS = ROOT / "results_v2/grec/parents"
OLD_PARENT_MANIFEST = OLD_PARENTS / "grec_parent_manifest.parquet"
EXPECTED_RETAINED_HOSTS = 3_364
EXPECTED_RETAINED_SCENES = EXPECTED_RETAINED_HOSTS * 11


class CarryForwardError(RuntimeError):
    """A retained row was not byte-identical and cannot be carried forward."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise CarryForwardError(f"artifact escapes repository: {value}") from error
    if not resolved.is_file():
        raise CarryForwardError(f"artifact is absent: {resolved}")
    return resolved


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != sha256_file(source):
            raise CarryForwardError(f"existing target differs: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-hosts", type=Path, required=True)
    parser.add_argument("--new-scenes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    new_hosts = pd.read_parquet(arguments.new_hosts)
    old_hosts = pd.read_parquet(OLD_HOSTS)
    new_scenes = pd.read_parquet(arguments.new_scenes)
    old_scenes = pd.read_parquet(OLD_SCENES)
    old_parents = pd.read_parquet(OLD_PARENT_MANIFEST)

    retained_ids = set(new_hosts["host_galaxy_id"].astype(str)) & set(
        old_hosts["host_galaxy_id"].astype(str)
    )
    if len(retained_ids) != EXPECTED_RETAINED_HOSTS:
        raise CarryForwardError(
            f"retained-host count is {len(retained_ids)}, "
            f"expected {EXPECTED_RETAINED_HOSTS}"
        )
    host_columns = [
        "host_galaxy_id",
        "host_parent_id",
        "host_payload_sha256",
        "host_pixel_sha256",
        "host_spatial_group_id",
        "psf_bin",
        "depth_bin",
        "psfsize_r",
        "psfdepth_r",
    ]
    old_host_rows = (
        old_hosts[old_hosts["host_galaxy_id"].astype(str).isin(retained_ids)]
        .sort_values("host_galaxy_id")[host_columns]
        .reset_index(drop=True)
    )
    new_host_rows = (
        new_hosts[new_hosts["host_galaxy_id"].astype(str).isin(retained_ids)]
        .sort_values("host_galaxy_id")[host_columns]
        .reset_index(drop=True)
    )
    if not old_host_rows.equals(new_host_rows):
        raise CarryForwardError("retained host identity or metadata changed")

    retained_old_scenes = old_scenes[
        old_scenes["host_galaxy_id"].astype(str).isin(retained_ids)
    ].copy()
    retained_new_scenes = new_scenes[
        new_scenes["host_galaxy_id"].astype(str).isin(retained_ids)
    ].copy()
    if (
        len(retained_old_scenes) != EXPECTED_RETAINED_SCENES
        or len(retained_new_scenes) != EXPECTED_RETAINED_SCENES
    ):
        raise CarryForwardError("retained scene support is incomplete")
    comparable = sorted(set(old_scenes.columns) & set(new_scenes.columns))
    old_compare = retained_old_scenes.sort_values("grec_row_id")[
        comparable
    ].reset_index(drop=True)
    new_compare = retained_new_scenes.sort_values("grec_row_id")[
        comparable
    ].reset_index(drop=True)
    if not old_compare.equals(new_compare):
        raise CarryForwardError("retained scene identities or parameters changed")

    retained_parents = old_parents[
        old_parents["host_galaxy_id"].astype(str).isin(retained_ids)
    ].copy()
    if len(retained_parents) != EXPECTED_RETAINED_SCENES:
        raise CarryForwardError("retained parent manifest support is incomplete")

    output = arguments.output_dir.resolve()
    counts = {"hardlink": 0, "copy": 0, "existing": 0}
    raw_sources: dict[str, Path] = {}
    for row in retained_parents.itertuples(index=False):
        source = resolve_repo_path(str(row.parent_pixel_path))
        relative = source.relative_to(OLD_PARENTS.resolve())
        mode = link_or_copy(source, output / relative)
        counts[mode] += 1
        host_source = resolve_repo_path(str(row.host_store_path))
        raw_sources[str(row.host_pixel_sha256)] = host_source
    for source in raw_sources.values():
        relative = source.relative_to(OLD_PARENTS.resolve())
        mode = link_or_copy(source, output / relative)
        counts[mode] += 1

    receipt = {
        "schema_version": 1,
        "status": "canonical_test_grec_carry_forward_preseeded",
        "retained_hosts": len(retained_ids),
        "retained_scenes": len(retained_parents),
        "retained_raw_hosts": len(raw_sources),
        "new_hosts": len(new_hosts) - len(retained_ids),
        "new_scenes": len(new_scenes) - len(retained_parents),
        "link_modes": counts,
        "old_hosts_sha256": sha256_file(OLD_HOSTS),
        "new_hosts_sha256": sha256_file(arguments.new_hosts),
        "old_scenes_sha256": sha256_file(OLD_SCENES),
        "new_scenes_sha256": sha256_file(arguments.new_scenes),
        "old_parent_manifest_sha256": sha256_file(OLD_PARENT_MANIFEST),
        "output_directory": str(output),
        "retained_scene_ids_sha256": hashlib.sha256(
            "\n".join(sorted(retained_parents["grec_row_id"].astype(str))).encode()
        ).hexdigest(),
    }
    atomic_json(output / "CANONICAL_REPAIR_CARRY_FORWARD_RECEIPT.json", receipt)
    print(output / "CANONICAL_REPAIR_CARRY_FORWARD_RECEIPT.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
