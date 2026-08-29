"""Strict JSON serialization shared by analysis scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def strict_json_dump(payload: dict[str, Any], path: Path) -> None:
    """Write deterministic standards-compliant JSON with a trailing newline."""
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
