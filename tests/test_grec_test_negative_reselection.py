from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "evaluation_dataset_prep"
sys.path.insert(0, str(PREP))

from grec_host_selection import select_grec_hosts  # noqa: E402
from reselect_grec_from_test_negative import (  # noqa: E402
    EXPECTED_ALLOWED_GROUPS,
    EXPECTED_ALLOWED_ID_SHA256,
    EXPECTED_ALLOWED_ROWS,
    EXPECTED_REPLACED,
    EXPECTED_RETAINED,
    EXPECTED_SELECTED_ID_SHA256,
    build_allowed_pool,
    id_manifest_sha256,
    load_frozen_inputs,
    verify_selection,
)


def test_shared_selector_reproduces_legacy_proposal() -> None:
    roles = pd.read_parquet(PREP / "role_proposal/proposed_roles.parquet")
    legacy = pd.read_parquet(PREP / "role_proposal/proposed_grec_hosts.parquet")
    selected, _, _ = select_grec_hosts(
        roles[roles["proposed_role"].eq("experiment_host")].copy()
    )
    assert set(selected["galaxy_id"]) == set(legacy["galaxy_id"])
    assert hashlib.sha256(
        "".join(f"{value}\n" for value in sorted(selected["galaxy_id"])).encode()
    ).hexdigest() == hashlib.sha256(
        "".join(f"{value}\n" for value in sorted(legacy["galaxy_id"])).encode()
    ).hexdigest()


def test_canonical_allowed_pool_and_selection_are_frozen() -> None:
    inputs = load_frozen_inputs()
    allowed = build_allowed_pool(inputs)
    assert len(allowed) == EXPECTED_ALLOWED_ROWS
    assert allowed["evaluation_spatial_group_id"].nunique() == EXPECTED_ALLOWED_GROUPS
    assert id_manifest_sha256(allowed["galaxy_id"]) == EXPECTED_ALLOWED_ID_SHA256

    selected, _, _ = select_grec_hosts(allowed)
    continuity = verify_selection(selected, allowed, inputs)
    assert len(selected) == 4_500
    assert id_manifest_sha256(selected["galaxy_id"]) == EXPECTED_SELECTED_ID_SHA256
    assert len(continuity["retained_ids"]) == EXPECTED_RETAINED
    assert len(continuity["removed_ids"]) == EXPECTED_REPLACED
    assert len(continuity["added_ids"]) == EXPECTED_REPLACED


def test_canonical_selection_has_no_frozen_role_contact() -> None:
    inputs = load_frozen_inputs()
    allowed = build_allowed_pool(inputs)
    selected, _, _ = select_grec_hosts(allowed)
    reserved = pd.concat(
        [inputs["threshold_calibration"], inputs["achieved_fpr"]],
        ignore_index=True,
    )
    assert not set(selected["galaxy_id"]) & set(reserved["galaxy_id"])
    assert not set(selected["evaluation_spatial_group_id"]) & set(
        reserved["evaluation_spatial_group_id"]
    )
    test_negative = inputs["test"][
        inputs["test"]["label"].eq(0)
        & inputs["test"]["endpoint_role"].eq("test_negative")
    ]
    assert set(selected["galaxy_id"]) <= set(test_negative["galaxy_id"])
