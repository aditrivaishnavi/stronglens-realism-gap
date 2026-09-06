from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from .contracts import atomic_write_json
from .contracts import load_config as load_bound_config


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hex(namespace: str, *parts: object) -> str:
    value = ":".join([namespace, *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_seed(namespace: str, *parts: object) -> int:
    return int(stable_hex(namespace, *parts)[:16], 16) % (2**63 - 1)


def safe_relative(root: Path, relative: str) -> Path:
    parsed = PurePosixPath(str(relative))
    if parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError(f"unsafe relative path: {relative}")
    return root.joinpath(*parsed.parts)


def load_config(path: Path) -> dict[str, Any]:
    return load_bound_config(path)


def reject_outcome_columns(frame: pd.DataFrame, tokens: list[str]) -> None:
    offending = sorted(
        column
        for column in frame.columns
        if any(token.lower() in column.lower() for token in tokens)
    )
    if offending:
        raise ValueError(f"outcome-like columns forbidden in selection: {offending}")


def finite_frame(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} misses columns: {missing}")
    values = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} has non-finite values in {columns}")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_json(path, value)


def dataframe_inventory(frame: pd.DataFrame, key: str) -> str:
    if frame[key].duplicated().any():
        raise ValueError(f"duplicate inventory key: {key}")
    digest = hashlib.sha256()
    for row in frame.sort_values(key, kind="mergesort").to_dict(orient="records"):
        digest.update(
            json.dumps(row, sort_keys=True, default=str, allow_nan=False).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()
