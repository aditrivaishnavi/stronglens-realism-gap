from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation_endpoints_v2.pipeline import (
    _spatial_interval,
    _wilson,
    array_sha256,
    stable_id,
)


def test_float32_array_hash_is_layout_stable() -> None:
    value = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    assert array_sha256(value) == array_sha256(np.ascontiguousarray(value))
    assert array_sha256(value) != array_sha256(value + np.float32(1.0))


def test_endpoint_ids_are_role_typed_and_deterministic() -> None:
    assert stable_id("terminal", "galaxy") == stable_id("terminal", "galaxy")
    assert stable_id("terminal", "galaxy") != stable_id("achfpr", "galaxy")


def test_wilson_interval_contains_observed_fraction() -> None:
    interval = _wilson(20, 20_002, 0.95)
    assert interval[0] < 20 / 20_002 < interval[1]


def test_spatial_multiplier_keeps_rows_in_one_group_together() -> None:
    frame = pd.DataFrame(
        {
            "endpoint_id": ["a", "b", "c", "d"],
            "spatial_group_id": ["shared", "shared", "other", "other"],
            "detected_at_frozen_threshold": [True, True, False, False],
        }
    )
    first = _spatial_interval(frame, 200, 42, 0.95)
    second = _spatial_interval(frame, 200, 42, 0.95)
    assert first == second
    assert first[0] < 0.5 < first[1]
