#!/usr/bin/env python3
"""Apply score-blind spatial exclusions and group candidate backgrounds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

NSIDE = 128
DEVELOPMENT_SEPARATION_ARCSEC = 38.0
POSITIVE_SEPARATION_ARCSEC = 11.0
EXPECTED_INPUTS = {
    "payload_manifest.parquet": {
        "sha256": "9e340ab44017c4ba9186f75c5b5c5e001eed613a5a1923bb400de1aded7895a3",
        "rows": 209_528,
    },
    "train.parquet": {
        "sha256": "a0caa3824568f96010275e59d89d013a23ac7e6e6ae280de2c636a401fa38e5b",
        "rows": 290_312,
    },
    "validation.parquet": {
        "sha256": "809729e63c9bd6fca96f51b4f8d244aeda604beca4135edcc428f431a9941ee9",
        "rows": 62_215,
    },
    "desi_candidates.csv": {
        "sha256": "b09a372abf71198efa07c246cf6dd8298c4b3af0aae00f62db434b563b98ed19",
        "rows": 5_104,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    cos_dec = np.cos(dec)
    return np.column_stack(
        [cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)]
    )


def chord_radius(arcsec: float) -> float:
    radians = np.deg2rad(arcsec / 3600.0)
    return float(2.0 * np.sin(radians / 2.0))


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, value: int) -> int:
        parent = self.parent
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(self, left: int, right: int) -> None:
        a = self.find(left)
        b = self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def nearest_contact(
    candidates: pd.DataFrame,
    references: pd.DataFrame,
    radius_arcsec: float,
) -> np.ndarray:
    if references.empty:
        return np.zeros(len(candidates), dtype=bool)
    candidate_vectors = unit_vectors(
        candidates["ra"].to_numpy(float),
        candidates["dec"].to_numpy(float),
    )
    reference_vectors = unit_vectors(
        references["ra"].to_numpy(float),
        references["dec"].to_numpy(float),
    )
    distances, _ = cKDTree(reference_vectors).query(
        candidate_vectors,
        k=1,
        distance_upper_bound=chord_radius(radius_arcsec),
        workers=-1,
    )
    return np.isfinite(distances)


def build_components(frame: pd.DataFrame) -> np.ndarray:
    vectors = unit_vectors(
        frame["ra"].to_numpy(float),
        frame["dec"].to_numpy(float),
    )
    union = UnionFind(len(frame))

    for indices in frame.groupby("payload_sha256", sort=False).indices.values():
        indices = np.asarray(indices, dtype=int)
        if len(indices) > 1:
            anchor = int(indices[0])
            for index in indices[1:]:
                union.union(anchor, int(index))

    for indices in frame.groupby("healpix_128", sort=False).indices.values():
        indices = np.asarray(indices, dtype=int)
        if len(indices) > 1:
            anchor = int(indices[0])
            for index in indices[1:]:
                union.union(anchor, int(index))

    pairs = cKDTree(vectors).query_pairs(
        chord_radius(DEVELOPMENT_SEPARATION_ARCSEC),
        output_type="ndarray",
    )
    for left, right in pairs:
        union.union(int(left), int(right))

    roots = np.array([union.find(index) for index in range(len(frame))])
    members: dict[int, list[str]] = {}
    for index, root in enumerate(roots):
        members.setdefault(int(root), []).append(str(frame.iloc[index]["galaxy_id"]))
    stable_ids = {
        root: "spatial-" + hashlib.sha256(
            "\n".join(sorted(identifiers)).encode()
        ).hexdigest()[:20]
        for root, identifiers in members.items()
    }
    return np.array([stable_ids[int(root)] for root in roots], dtype=object)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--payload-manifest",
        type=Path,
        default=Path(
            "evaluation_dataset_prep/candidate_payloads/payload_manifest.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_dataset_prep/spatial_selection"),
    )
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / "SPATIAL_STAGE_COMPLETE.json").unlink(missing_ok=True)

    input_paths = {
        "payload_manifest.parquet": arguments.payload_manifest,
        "train.parquet": arguments.root / "data/manifests/train.parquet",
        "validation.parquet": arguments.root
        / "data/manifests/validation.parquet",
        "desi_candidates.csv": arguments.root / "data/positives/desi_candidates.csv",
    }
    for name, path in input_paths.items():
        expected = EXPECTED_INPUTS[name]
        if not path.is_file() or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"{name} is missing or has the wrong SHA-256")

    payloads = pd.read_parquet(arguments.payload_manifest)
    if len(payloads) != EXPECTED_INPUTS["payload_manifest.parquet"]["rows"]:
        raise ValueError("payload manifest has the wrong row count")
    candidates = payloads[
        payloads["bucket"].eq("negative") & payloads["valid"].eq(1)
    ].copy()
    if len(candidates) != 209_157:
        raise ValueError(f"expected 209157 valid negative candidates, found {len(candidates)}")
    if candidates["galaxy_id"].duplicated().any():
        raise ValueError("negative candidate IDs are not unique")

    train = pd.read_parquet(
        input_paths["train.parquet"],
        columns=["galaxy_id", "ra", "dec", "healpix_128"],
    )
    validation = pd.read_parquet(
        input_paths["validation.parquet"],
        columns=["galaxy_id", "ra", "dec", "healpix_128"],
    )
    if len(train) != EXPECTED_INPUTS["train.parquet"]["rows"]:
        raise ValueError("training manifest has the wrong row count")
    if len(validation) != EXPECTED_INPUTS["validation.parquet"]["rows"]:
        raise ValueError("validation manifest has the wrong row count")
    development = pd.concat([train, validation], ignore_index=True)
    if set(candidates["galaxy_id"].astype(str)) & set(
        development["galaxy_id"].astype(str)
    ):
        raise ValueError("candidate ID overlap with detector development")

    candidates["healpix_128"] = hp.ang2pix(
        NSIDE,
        candidates["ra"].to_numpy(float),
        candidates["dec"].to_numpy(float),
        nest=True,
        lonlat=True,
    )
    candidates["outside_declination_window"] = ~candidates["dec"].between(
        -18.0, 32.0, inclusive="neither"
    )
    candidates["invalid_observing_metadata"] = ~(
        np.isfinite(candidates["psfsize_r"].to_numpy(float))
        & candidates["psfsize_r"].gt(0)
        & np.isfinite(candidates["psfdepth_r"].to_numpy(float))
        & candidates["psfdepth_r"].gt(0)
    )
    candidates["development_healpix_contact"] = candidates["healpix_128"].isin(
        set(development["healpix_128"].astype(int))
    )
    candidates["development_38arcsec_contact"] = nearest_contact(
        candidates,
        development,
        DEVELOPMENT_SEPARATION_ARCSEC,
    )

    positive_catalogue = pd.read_csv(
        input_paths["desi_candidates.csv"],
        usecols=["ra", "dec"],
    ).dropna()
    if len(positive_catalogue) != EXPECTED_INPUTS["desi_candidates.csv"]["rows"]:
        raise ValueError("positive catalogue has the wrong row count")
    candidates["known_positive_11arcsec_contact"] = nearest_contact(
        candidates,
        positive_catalogue,
        POSITIVE_SEPARATION_ARCSEC,
    )

    candidates = candidates.sort_values("galaxy_id").reset_index(drop=True)
    candidates["evaluation_spatial_group_id"] = build_components(candidates)
    raw_contact_counts = {
        column: int(candidates[column].sum())
        for column in (
            "outside_declination_window",
            "invalid_observing_metadata",
            "development_healpix_contact",
            "development_38arcsec_contact",
            "known_positive_11arcsec_contact",
        )
    }
    prohibited_columns = list(raw_contact_counts)
    candidates["directly_prohibited"] = candidates[prohibited_columns].any(axis=1)
    candidates["component_prohibited"] = candidates.groupby(
        "evaluation_spatial_group_id"
    )["directly_prohibited"].transform("any")

    candidates["spatial_eligibility_reason"] = "eligible"
    reasons = [
        ("outside_declination_window", "outside_declination_window"),
        ("invalid_observing_metadata", "invalid_observing_metadata"),
        ("development_healpix_contact", "latest_development_healpix_contact"),
        ("development_38arcsec_contact", "latest_development_38arcsec_contact"),
        ("known_positive_11arcsec_contact", "known_positive_11arcsec_contact"),
    ]
    for column, reason in reasons:
        mask = candidates[column] & candidates["spatial_eligibility_reason"].eq(
            "eligible"
        )
        candidates.loc[mask, "spatial_eligibility_reason"] = reason
    propagated = (
        ~candidates["directly_prohibited"]
        & candidates["component_prohibited"]
        & candidates["spatial_eligibility_reason"].eq("eligible")
    )
    candidates.loc[
        propagated, "spatial_eligibility_reason"
    ] = "component_contains_prohibited_member"

    eligible = candidates[
        candidates["spatial_eligibility_reason"].eq("eligible")
    ].copy()
    eligible = eligible.sort_values("galaxy_id").reset_index(drop=True)
    if eligible["component_prohibited"].any():
        raise AssertionError("eligible output contains a prohibited spatial component")
    if (
        candidates.groupby("healpix_128")["evaluation_spatial_group_id"]
        .nunique()
        .max()
        != 1
    ):
        raise AssertionError("one HEALPix cell crosses spatial components")
    if candidates["evaluation_spatial_group_id"].isna().any():
        raise AssertionError("candidate lacks a spatial component")

    annotated_path = arguments.output_dir / "annotated_negative_candidates.parquet"
    eligible_path = arguments.output_dir / "eligible_negative_candidates.parquet"
    temporary_annotated = arguments.output_dir / "annotated.partial.parquet"
    temporary_eligible = arguments.output_dir / "eligible.partial.parquet"
    candidates.to_parquet(temporary_annotated, index=False)
    eligible.to_parquet(temporary_eligible, index=False)
    temporary_annotated.replace(annotated_path)
    temporary_eligible.replace(eligible_path)

    summary = {
        "schema_version": 1,
        "status": "spatial_grouping_complete_catalogue_join_pending",
        "input_rows": len(candidates),
        "eligible_rows": len(eligible),
        "eligible_spatial_groups": int(
            eligible["evaluation_spatial_group_id"].nunique()
        ),
        "reason_counts": candidates["spatial_eligibility_reason"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "development_id_overlap": 0,
        "raw_contact_counts": raw_contact_counts,
        "grouping_includes": [
            "payload_sha256",
            "healpix_128_nested",
            "centres_within_38_arcsec",
        ],
        "physical_group_status": "pending_Tractor_catalogue_join",
        "detector_scores_used": False,
        "input_hashes": {
            name: sha256_file(path) for name, path in input_paths.items()
        },
        "annotated_manifest_sha256": sha256_file(annotated_path),
        "eligible_manifest_sha256": sha256_file(eligible_path),
    }
    temporary_summary = arguments.output_dir / "summary.partial.json"
    temporary_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path = arguments.output_dir / "summary.json"
    temporary_summary.replace(summary_path)
    completion = arguments.output_dir / "SPATIAL_STAGE_COMPLETE.json"
    temporary_completion = arguments.output_dir / "SPATIAL_STAGE_COMPLETE.partial.json"
    temporary_completion.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "spatial_stage_complete_physical_groups_pending",
                "annotated_manifest_sha256": sha256_file(annotated_path),
                "eligible_manifest_sha256": sha256_file(eligible_path),
                "summary_sha256": sha256_file(summary_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_completion.replace(completion)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
