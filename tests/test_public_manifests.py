from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.validate_manifests import validate

ROOT = Path(__file__).resolve().parents[1]


def frame(name: str, *, ra: float, group: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "galaxy_id": [name],
            "cutout_relpath": [f"stronglens_calibration/{name}.npz"],
            "healpix_128": [group],
            "physical_group_id": [f"physical-{group}"],
            "spatial_component_id": [f"spatial-{group}"],
            "ra": [ra],
            "dec": [0.0],
            "label": [0],
            "split": [name],
            "endpoint_role": ["model_development"],
        }
    )


def test_validator_accepts_disjoint_manifests() -> None:
    validate(
        {
            "train": frame("train", ra=0.0, group=1),
            "validation": frame("validation", ra=1.0, group=2),
            "test": frame("test", ra=2.0, group=3),
        }
    )


def test_validator_rejects_shared_groups() -> None:
    train = frame("train", ra=0.0, group=1)
    validation = frame("validation", ra=1.0, group=1)
    with pytest.raises(ValueError, match="healpix_128"):
        validate(
            {
                "train": train,
                "validation": validation,
                "test": frame("test", ra=2.0, group=3),
            }
        )


def test_final_public_manifests() -> None:
    paths = {
        name: ROOT / f"data/manifests/{name}.parquet"
        for name in ("train", "validation", "test")
    }
    if not all(path.is_file() for path in paths.values()):
        pytest.skip("public manifests have not been generated")

    manifests = {name: pd.read_parquet(path) for name, path in paths.items()}
    validate(manifests)
    assert {name: len(frame) for name, frame in manifests.items()} == {
        "train": 290_312,
        "validation": 62_215,
        "test": 61_498,
    }
    assert int(manifests["train"]["label"].sum()) == 3_633
    assert int(manifests["validation"]["label"].sum()) == 778
    assert int(manifests["test"]["label"].sum()) == 63
    assert (
        manifests["test"]["endpoint_role"].value_counts().to_dict()
        == {
            "test_negative": 61_435,
            "terminal_confirmed": 43,
            "development_contacted_confirmed": 20,
        }
    )
