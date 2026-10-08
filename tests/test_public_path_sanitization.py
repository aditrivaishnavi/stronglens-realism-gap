from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LINEAGE = (
    ROOT
    / "results_v2/canonical_test_host_repair_v1/"
    "PUBLIC_PATH_SANITIZATION.json"
)
PRIVATE_MARKERS = (
    "/Users/" + "balaji",
    "correct-" + "stronglens-realism-gap",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [ROOT / item.decode() for item in output.split(b"\0") if item]


def test_tracked_text_and_parquet_have_no_personal_checkout_paths() -> None:
    for path in tracked_files():
        if not path.is_file():
            continue
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
            for column in frame.columns:
                values = frame[column]
                if not (
                    values.dtype == object
                    or isinstance(values.dtype, pd.StringDtype)
                ):
                    continue
                joined = "\n".join(values.dropna().astype(str).unique())
                assert not any(marker in joined for marker in PRIVATE_MARKERS), path
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        assert not any(marker in text for marker in PRIVATE_MARKERS), path


def test_dual_hash_lineage_binds_every_sanitized_public_artifact() -> None:
    lineage = json.loads(LINEAGE.read_text(encoding="utf-8"))
    assert lineage["history"]["execution_ids_recomputed"] is False
    assert len(lineage["sanitized_artifacts"]) == 67
    for record in lineage["sanitized_artifacts"]:
        path = ROOT / record["path"]
        assert path.is_file(), record["path"]
        assert sha256(path) == record["public_sha256"]
        assert len(record["execution_sha256"]) == 64


def test_public_protocol_hashes_are_current_and_lineage_is_retained() -> None:
    lineage = json.loads(LINEAGE.read_text(encoding="utf-8"))
    for relative, record in lineage["protocol_lineage"].items():
        path = ROOT / relative
        assert path.is_file(), relative
        assert sha256(path) == record["public_protocol_sha256"]
        assert len(record["private_protocol_sha256"]) == 64
        text = path.read_text(encoding="utf-8").lower()
        for forbidden in (
            "five-experiment rerun",
            "user instruction",
            "no final dataset is currently selected",
            "preparation pending",
            "feature access is not authorized",
        ):
            assert forbidden not in text
