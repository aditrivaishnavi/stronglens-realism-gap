"""Checks that model-development rows are disjoint from the test manifest."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_TEST_MANIFEST = Path("data/manifests/test.parquet")
IDENTIFIER_COLUMN = "galaxy_id"


class SplitOverlapError(RuntimeError):
    """Raised when an operation unexpectedly includes test rows."""


@dataclass(frozen=True)
class Selection:
    """A named frame and the column containing its object identifiers."""

    name: str
    frame: pd.DataFrame
    column: str = IDENTIFIER_COLUMN


@lru_cache(maxsize=8)
def test_identifiers(
    manifest_path: str,
    column: str = IDENTIFIER_COLUMN,
) -> frozenset[str]:
    path = Path(manifest_path)
    if not path.is_file():
        raise SplitOverlapError(f"test manifest does not exist: {path}")
    frame = pd.read_parquet(path, columns=[column])
    return frozenset(frame[column].astype(str))


def count_test_rows(
    frame: pd.DataFrame,
    test_manifest: Path | str = DEFAULT_TEST_MANIFEST,
    column: str = IDENTIFIER_COLUMN,
) -> int:
    """Count rows whose identifiers occur in the separate test manifest."""
    if column not in frame:
        raise SplitOverlapError(f"frame has no {column!r} column")
    identifiers = test_identifiers(str(test_manifest))
    return int(frame[column].astype(str).isin(identifiers).sum())


def check_selections(
    selections: list[Selection],
    *,
    purpose: str,
    allow_test_rows: bool = False,
    test_manifest: Path | str = DEFAULT_TEST_MANIFEST,
) -> dict[str, Any]:
    test_ids = test_identifiers(str(test_manifest))
    by_frame: dict[str, dict[str, int | str]] = {}
    total = 0
    for selection in selections:
        name, frame, column = selection.name, selection.frame, selection.column
        if column not in frame:
            raise SplitOverlapError(f"{name} has no {column!r} column")
        overlap = int(frame[column].astype(str).isin(test_ids).sum())
        total += overlap
        by_frame[name] = {
            "rows": int(len(frame)),
            "identifier_column": column,
            "test_rows": overlap,
        }
    if total and not allow_test_rows:
        raise SplitOverlapError(
            f"{purpose}: {total} selected rows occur in {test_manifest}"
        )
    return {
        "test_rows_selected": total,
        "test_rows_allowed": bool(allow_test_rows),
        "test_manifest": str(test_manifest),
        "by_frame": by_frame,
    }


def check_frames(
    frames: Mapping[str, pd.DataFrame],
    *,
    purpose: str,
    allow_test_rows: bool = False,
    test_manifest: Path | str = DEFAULT_TEST_MANIFEST,
    column: str = IDENTIFIER_COLUMN,
) -> dict[str, Any]:
    return check_selections(
        [Selection(name, frame, column) for name, frame in frames.items()],
        purpose=purpose,
        allow_test_rows=allow_test_rows,
        test_manifest=test_manifest,
    )


def add_arguments(parser: Any) -> None:
    parser.add_argument(
        "--test-manifest",
        default=str(DEFAULT_TEST_MANIFEST),
        help="Test manifest used to check split disjointness.",
    )
    parser.add_argument(
        "--allow-test-rows",
        action="store_true",
        default=False,
        help="Allow test rows for an explicitly requested final evaluation.",
    )
