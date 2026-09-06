from __future__ import annotations

import hashlib
import io
import os
import subprocess
import tarfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .authority import (
    import_predecessor_modules,
    load_predecessor_config,
    validate_inherited_authority,
)
from .compatibility import build_prelaunch_handoff_compatibility
from .contracts import (
    PilotContractError,
    array_bytes_sha256,
    assert_config_integrity,
    atomic_write_json,
    atomic_write_parquet,
    authority_path,
    config_provenance,
    inherited_code_closure,
    parse_passing_junit,
    require_absent,
    sha256_file,
    stable_id,
    successor_code_closure,
)

PSF_NODES = np.array([0.90, 1.05, 1.20, 1.35, 1.50, 1.65, 1.80])
DEPTH_NODES = np.array([22.5, 23.0, 23.5, 24.0, 24.5])
ZERO_OVERLAP_FIELDS = (
    "id_overlap",
    "payload_file_sha256_overlap",
    "raw_array_sha256_overlap",
    "physical_group_overlap",
    "canonical_spatial_component_overlap",
    "healpix_128_overlap",
    "within_38_arcsec_hosts",
)


def nearest_lower_midpoint(values: Sequence[float], nodes: np.ndarray) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    midpoints = (nodes[:-1] + nodes[1:]) / 2.0
    return nodes[np.searchsorted(midpoints, values_array, side="left")]


def depth_magnitude(psfdepth_r: Sequence[float]) -> np.ndarray:
    depth = np.asarray(psfdepth_r, dtype=np.float64)
    if not np.isfinite(depth).all() or (depth <= 0).any():
        raise PilotContractError("historical psfdepth_r must be finite and positive")
    return 22.5 - 2.5 * np.log10(5.0 / np.sqrt(depth))


def cell_key(psf_bin: float, depth_bin: float) -> str:
    return f"{float(psf_bin):.2f}|{float(depth_bin):.1f}"


def deterministic_balanced_selection(
    candidates: pd.DataFrame,
    *,
    cells: Sequence[str],
    hosts_per_cell: int,
    namespace: str,
) -> pd.DataFrame:
    required = {
        "galaxy_id",
        "pilot_cell",
        "physical_group_id",
        "spatial_component_id",
        "cutout_path",
    }
    if required - set(candidates):
        raise PilotContractError("candidate table lacks design-selection identities")
    frame = candidates.copy()
    frame["selection_order_sha256"] = frame.apply(
        lambda row: hashlib.sha256(
            (
                f"{namespace}|{row['pilot_cell']}|{row['galaxy_id']}|"
                f"{row['cutout_path']}"
            ).encode()
        ).hexdigest(),
        axis=1,
    )
    queues = {
        name: frame[frame["pilot_cell"].eq(name)]
        .sort_values(["selection_order_sha256", "galaxy_id"], kind="mergesort")
        .to_dict(orient="records")
        for name in cells
    }
    selected: list[dict[str, Any]] = []
    used_physical: set[str] = set()
    used_spatial: set[str] = set()
    positions = {name: 0 for name in cells}
    counts = {name: 0 for name in cells}
    while any(counts[name] < hosts_per_cell for name in cells):
        progress = False
        for name in cells:
            if counts[name] >= hosts_per_cell:
                continue
            queue = queues[name]
            while positions[name] < len(queue):
                row = queue[positions[name]]
                positions[name] += 1
                physical = str(row["physical_group_id"])
                spatial = str(row["spatial_component_id"])
                if physical in used_physical or spatial in used_spatial:
                    continue
                selected.append(row)
                used_physical.add(physical)
                used_spatial.add(spatial)
                counts[name] += 1
                progress = True
                break
        if not progress:
            raise PilotContractError(
                "insufficient distinct physical/spatial design hosts: "
                f"selected counts {counts}"
            )
    result = pd.DataFrame(selected).sort_values(
        ["pilot_cell", "selection_order_sha256", "galaxy_id"], kind="mergesort"
    )
    observed = result["pilot_cell"].value_counts().to_dict()
    if any(observed.get(name) != hosts_per_cell for name in cells):
        raise AssertionError("balanced selection count drift")
    if (
        result["galaxy_id"].duplicated().any()
        or result["physical_group_id"].astype(str).duplicated().any()
        or result["spatial_component_id"].astype(str).duplicated().any()
    ):
        raise AssertionError("design selection reused an identity unit")
    return result.reset_index(drop=True)


def _historical_candidates(config: Mapping[str, Any]) -> pd.DataFrame:
    metadata = pd.read_parquet(authority_path(config, "historical_metadata"))
    historical = pd.read_parquet(
        authority_path(config, "historical_scoring_manifest")
    )
    required_metadata = {"inj_id", "cutout_path"}
    required_historical = {
        "galaxy_id",
        "cutout_path",
        "ra",
        "dec",
        "split",
        "label",
        "psfsize_r",
        "psfdepth_r",
        "healpix_128",
        "physical_group_id",
        "physical_group_technical_check_status",
        "spatial_component_id",
    }
    if required_metadata - set(metadata) or required_historical - set(historical):
        raise PilotContractError("historical authorities lack identity/physical/spatial data")
    representatives = (
        metadata.sort_values("inj_id", kind="mergesort")
        .drop_duplicates("cutout_path", keep="first")[["inj_id", "cutout_path"]]
        .rename(columns={"cutout_path": "injection_host_path"})
    )
    representatives["galaxy_id"] = representatives["injection_host_path"].map(
        lambda value: Path(str(value)).stem
    )
    manifest = representatives.merge(
        historical,
        on="galaxy_id",
        how="left",
        validate="one_to_one",
    )
    if len(manifest) != 28_625 or manifest["cutout_path"].isna().any():
        raise PilotContractError("v1_val_full host-to-staging join is not complete")
    source = config["historical_source"]
    if (
        not manifest["split"].eq(source["required_split"]).all()
        or not manifest["label"].eq(source["required_label"]).all()
    ):
        raise PilotContractError("historical design candidates are not all validation negatives")
    if not manifest["physical_group_technical_check_status"].eq(
        "automated_technical_check_passed"
    ).all():
        raise PilotContractError("historical physical-group authority is incomplete")
    if (
        manifest["physical_group_id"].isna().any()
        or manifest["spatial_component_id"].isna().any()
        or manifest[["ra", "dec"]].isna().any().any()
    ):
        raise PilotContractError("historical host lacks physical/spatial authority")
    manifest["depth_mag"] = depth_magnitude(manifest["psfdepth_r"])
    manifest["psf_bin"] = nearest_lower_midpoint(manifest["psfsize_r"], PSF_NODES)
    manifest["depth_bin"] = nearest_lower_midpoint(
        manifest["depth_mag"], DEPTH_NODES
    )
    manifest["pilot_cell"] = [
        cell_key(psf, depth)
        for psf, depth in zip(
            manifest["psf_bin"], manifest["depth_bin"], strict=True
        )
    ]
    return manifest


def _s3_key(cutout_path: str) -> str:
    marker = "/stage/"
    if marker not in cutout_path:
        raise PilotContractError(f"historical cutout path lacks stage root: {cutout_path}")
    key = cutout_path.split(marker, 1)[1]
    if not key.startswith("stronglens_calibration/") or ".." in Path(key).parts:
        raise PilotContractError(f"unsafe historical S3 key: {key}")
    return key


def _download_one(
    row: Mapping[str, Any],
    *,
    output: Path,
    bucket: str,
    profile: str,
) -> tuple[str, Path]:
    galaxy_id = str(row["galaxy_id"])
    destination = output / "hosts" / galaxy_id[:4] / f"{galaxy_id}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        # A preparation retry may resume downloaded bytes only because every file
        # is subsequently revalidated against the hash-bound tar ancestry equation.
        return galaxy_id, destination
    temporary = destination.with_name(destination.name + ".partial")
    uri = f"s3://{bucket}/{_s3_key(str(row['cutout_path']))}"
    command = ["aws", "s3", "cp", uri, str(temporary), "--only-show-errors"]
    if profile:
        command.extend(["--profile", profile])
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        temporary.unlink(missing_ok=True)
        raise PilotContractError(
            f"raw historical host acquisition failed for {galaxy_id}: "
            f"{completed.stderr.strip()}"
        )
    os.replace(temporary, destination)
    return galaxy_id, destination


def _stage_and_verify_hosts(
    selected: pd.DataFrame,
    config: Mapping[str, Any],
    output: Path,
    *,
    aws_profile: str,
    workers: int,
) -> pd.DataFrame:
    source = config["historical_source"]
    rows = selected.to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        downloaded = dict(
            executor.map(
                lambda row: _download_one(
                    row,
                    output=output,
                    bucket=str(source["s3_bucket"]),
                    profile=aws_profile,
                ),
                rows,
            )
        )
    tar_path = authority_path(config, "historical_cutout_tar")
    host_rows: list[dict[str, Any]] = []
    expected_shape = tuple(int(value) for value in source["required_shape_hwc"])
    with tarfile.open(tar_path, "r:") as archive:
        archive.getmembers()  # Build one seek index; do not rescan 115,500 headers.
        for row in rows:
            galaxy_id = str(row["galaxy_id"])
            payload_path = downloaded[galaxy_id]
            with np.load(payload_path, allow_pickle=False) as payload:
                if source["raw_host_array"] not in payload.files:
                    raise PilotContractError(f"raw host has no cutout: {payload_path}")
                host = np.array(payload[source["raw_host_array"]], copy=True)
                embedded_id = (
                    str(payload["meta_galaxy_id"].item())
                    if "meta_galaxy_id" in payload.files
                    else galaxy_id
                )
            if (
                embedded_id != galaxy_id
                or host.dtype != np.float32
                or host.shape != expected_shape
                or not np.isfinite(host).all()
            ):
                raise PilotContractError(f"invalid raw historical host: {payload_path}")
            member_name = source["historical_member_template"].format(
                inj_id=row["inj_id"]
            )
            try:
                member = archive.getmember(member_name)
            except KeyError as error:
                raise PilotContractError(
                    f"historical tar lacks representative {member_name}"
                ) from error
            extracted = archive.extractfile(member)
            if extracted is None:
                raise PilotContractError(f"historical member is not readable: {member_name}")
            member_bytes = extracted.read()
            with np.load(io.BytesIO(member_bytes), allow_pickle=False) as historical:
                expectation = np.array(
                    historical[source["historical_expectation_array"]], copy=True
                )
                injection = np.array(
                    historical[source["historical_source_array"]], copy=True
                )
            host_chw = host.transpose(2, 0, 1)
            if (
                expectation.dtype != np.float32
                or injection.dtype != np.float32
                or expectation.shape != host_chw.shape
                or injection.shape != host_chw.shape
                or not np.array_equal(
                    (host_chw + injection).astype(np.float32, copy=False),
                    expectation,
                )
            ):
                raise PilotContractError(
                    "actual historical host bytes fail exact v1_val_full ancestry: "
                    f"{galaxy_id}/{member_name}"
                )
            host_rows.append(
                {
                    "host_galaxy_id": galaxy_id,
                    "host_parent_id": stable_id(
                        "pilot_host",
                        [galaxy_id, str(row["physical_group_id"])],
                    ),
                    "host_pixel_path": str(payload_path),
                    "host_payload_sha256": sha256_file(payload_path),
                    "host_pixel_sha256": array_bytes_sha256(host),
                    "host_spatial_group_id": (
                        f"historical_val_spatial_{row['spatial_component_id']}"
                    ),
                    "host_spatial_coordinate_id": stable_id(
                        "coord",
                        [float(row["ra"]), float(row["dec"])],
                    ),
                    "host_ra_deg": float(row["ra"]),
                    "host_dec_deg": float(row["dec"]),
                    "final_role": source_role(config),
                    "psf_bin": float(row["psf_bin"]),
                    "depth_bin": float(row["depth_bin"]),
                    "depth_mag": float(row["depth_mag"]),
                    "psfsize_r": float(row["psfsize_r"]),
                    "psfdepth_r": float(row["psfdepth_r"]),
                    "psf_fwhm_r_arcsec": float(row["psfsize_r"]),
                    "pixel_scale_arcsec": float(source["pixel_scale_arcsec"]),
                    "host_pixel_unit": str(source["host_pixel_unit"]),
                    "host_flux_calibration_system": str(
                        source["host_flux_calibration_system"]
                    ),
                    "host_calibration_provenance_path": str(
                        authority_path(config, "host_flux_calibration")
                    ),
                    "host_calibration_provenance_sha256": config["authorities"][
                        "host_flux_calibration"
                    ]["sha256"],
                    "physical_group_id": str(row["physical_group_id"]),
                    "historical_spatial_component_id": str(
                        row["spatial_component_id"]
                    ),
                    "historical_healpix_128": int(row["healpix_128"]),
                    "historical_injection_member": member_name,
                    "historical_injection_member_sha256": hashlib.sha256(
                        member_bytes
                    ).hexdigest(),
                    "historical_host_ancestry_exact": True,
                    "selection_order_sha256": str(row["selection_order_sha256"]),
                    "pilot_cell": str(row["pilot_cell"]),
                }
            )
    hosts = pd.DataFrame(host_rows).sort_values(
        ["psf_bin", "depth_bin", "host_galaxy_id"], kind="mergesort"
    )
    if hosts["host_pixel_sha256"].duplicated().any():
        raise PilotContractError("selected historical hosts contain duplicate pixels")
    return hosts.reset_index(drop=True)


def source_role(config: Mapping[str, Any]) -> str:
    return str(config["design"]["global_role"])


def _hash_comparison_role_row(
    row: Mapping[str, Any],
    *,
    payload_root: Path,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    payload_path = (payload_root / str(row["local_relpath"])).resolve()
    if payload_root not in payload_path.parents:
        raise PilotContractError(f"comparison payload escapes root: {payload_path}")
    observed_payload_sha256 = sha256_file(payload_path)
    if observed_payload_sha256 != str(row["payload_sha256"]):
        raise PilotContractError(
            f"comparison payload digest mismatch: {payload_path}"
        )
    with np.load(payload_path, allow_pickle=False) as archive:
        if "cutout" not in archive.files:
            raise PilotContractError(f"comparison payload lacks cutout: {payload_path}")
        pixels = np.array(archive["cutout"], copy=True)
    if (
        pixels.dtype != np.float32
        or pixels.shape != (101, 101, 3)
        or not np.isfinite(pixels).all()
    ):
        raise PilotContractError(f"invalid comparison raw pixels: {payload_path}")
    role = str(row["global_role"])
    galaxy_id = str(row["galaxy_id"])
    return {
        "comparison_row_id": stable_id(
            "comparison_pixel",
            [role, galaxy_id, observed_payload_sha256],
        ),
        "global_role": role,
        "galaxy_id": galaxy_id,
        "source_role_manifest_sha256": source_manifest_sha256,
        "source_payload_path": str(payload_path),
        "source_payload_sha256": observed_payload_sha256,
        "raw_array_key": "cutout",
        "raw_array_dtype": str(pixels.dtype),
        "raw_array_shape": "101x101x3",
        "raw_array_sha256": array_bytes_sha256(pixels),
    }


def build_comparison_role_pixel_hashes(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    assert_config_integrity(config)
    payload_root = Path(config["paths"]["candidate_payload_root"]).resolve()
    if not payload_root.is_dir():
        raise PilotContractError("comparison candidate payload root is unavailable")
    frames: list[pd.DataFrame] = []
    role_sources = (
        ("threshold_calibration", "threshold_calibration_role"),
        ("achieved_fpr", "achieved_fpr_role"),
    )
    for role, authority_name in role_sources:
        source = pd.read_parquet(authority_path(config, authority_name))
        required = {
            "galaxy_id",
            "global_role",
            "local_relpath",
            "payload_sha256",
        }
        if required - set(source):
            raise PilotContractError(f"{role} lacks raw-pixel authority fields")
        if (
            not source["global_role"].eq(role).all()
            or source["galaxy_id"].astype(str).duplicated().any()
        ):
            raise PilotContractError(f"{role} identities are not unique and exact")
        source_sha256 = config["authorities"][authority_name]["sha256"]
        records = source.sort_values("galaxy_id", kind="mergesort").to_dict(
            orient="records"
        )
        workers = int(config["validation"]["comparison_pixel_hash_workers"])
        hash_row = partial(
            _hash_comparison_role_row,
            payload_root=payload_root,
            source_manifest_sha256=source_sha256,
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            hashed = list(executor.map(hash_row, records))
        frames.append(pd.DataFrame(hashed))
    manifest = pd.concat(frames, ignore_index=True)
    if (
        manifest["comparison_row_id"].duplicated().any()
        or manifest[["global_role", "galaxy_id"]].duplicated().any()
    ):
        raise PilotContractError("comparison raw-pixel hash ledger has duplicate rows")
    counts = manifest["global_role"].value_counts().sort_index().to_dict()
    expected_counts = {
        "achieved_fpr": 12_512,
        "threshold_calibration": 20_002,
    }
    if counts != expected_counts:
        raise PilotContractError(
            f"comparison raw-pixel ledger has wrong role counts: {counts}"
        )
    receipt = {
        "schema_version": 1,
        "status": "pnoise_comparison_role_raw_pixel_hashes_complete",
        "rows": len(manifest),
        "role_rows": counts,
        "raw_array_key": "cutout",
        "raw_array_dtype": "float32",
        "raw_array_shape": [101, 101, 3],
        "per_row_payload_sha256_verified": True,
        "per_row_raw_array_sha256_recorded": True,
        "source_role_manifest_sha256": {
            role: config["authorities"][authority_name]["sha256"]
            for role, authority_name in role_sources
        },
        "negative_role_receipt_sha256": config["authorities"][
            "negative_role_receipt"
        ]["sha256"],
        "config": config_provenance(config),
    }
    return manifest, receipt


def _unit_vectors(frame: pd.DataFrame, ra: str, dec: str) -> np.ndarray:
    longitude = np.deg2rad(frame[ra].to_numpy(dtype=float))
    latitude = np.deg2rad(frame[dec].to_numpy(dtype=float))
    cosine = np.cos(latitude)
    return np.column_stack(
        [cosine * np.cos(longitude), cosine * np.sin(longitude), np.sin(latitude)]
    )


def _overlap_record(
    hosts: pd.DataFrame,
    other: pd.DataFrame,
    *,
    role: str,
    id_column: str,
    pixel_column: str | None,
    raw_array_column: str,
    ra_column: str,
    dec_column: str,
    minimum_separation_arcsec: float,
) -> dict[str, Any]:
    ids = set(hosts["host_galaxy_id"].astype(str))
    physical = set(hosts["physical_group_id"].astype(str))
    spatial = set(hosts["historical_spatial_component_id"].astype(str))
    healpix = set(hosts["historical_healpix_128"].astype(str))
    id_overlap = len(ids & set(other[id_column].astype(str)))
    physical_overlap = len(
        physical & set(other["physical_group_id"].dropna().astype(str))
    )
    spatial_overlap = len(
        spatial & set(other["spatial_component_id"].dropna().astype(str))
    )
    healpix_overlap = len(
        healpix & set(other["healpix_128"].dropna().astype(int).astype(str))
    )
    pixel_overlap = 0
    if pixel_column is not None:
        pixel_overlap = len(
            set(hosts["host_payload_sha256"].astype(str))
            & set(other[pixel_column].dropna().astype(str))
        )
    raw_array_overlap = len(
        set(hosts["host_pixel_sha256"].astype(str))
        & set(other[raw_array_column].dropna().astype(str))
    )
    host_vectors = _unit_vectors(hosts, "host_ra_deg", "host_dec_deg")
    other_vectors = _unit_vectors(other, ra_column, dec_column)
    distances, _ = cKDTree(other_vectors).query(host_vectors, k=1)
    angular = np.rad2deg(2.0 * np.arcsin(np.clip(distances / 2.0, 0.0, 1.0))) * 3600
    within = int(np.count_nonzero(angular <= minimum_separation_arcsec))
    return {
        "role": role,
        "id_overlap": id_overlap,
        "payload_file_sha256_overlap": pixel_overlap,
        "raw_array_sha256_overlap": raw_array_overlap,
        "physical_group_overlap": physical_overlap,
        "canonical_spatial_component_overlap": spatial_overlap,
        "healpix_128_overlap": healpix_overlap,
        "within_38_arcsec_hosts": within,
        "minimum_angular_separation_arcsec": float(angular.min()),
    }


def validate_overlap_evidence(
    evidence: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    assert_config_integrity(config)
    gates = evidence.get("gates")
    if not isinstance(gates, list):
        raise PilotContractError("overlap evidence has no numeric gates")
    expected_roles = list(config["overlap"]["comparison_roles"])
    if [gate.get("role") for gate in gates if isinstance(gate, Mapping)] != expected_roles:
        raise PilotContractError("overlap evidence roles/order differ from config")
    overlap = config["overlap"]
    expected_values = {
        "id_overlap": int(overlap["required_id_overlap"]),
        "payload_file_sha256_overlap": int(overlap["required_pixel_overlap"]),
        "raw_array_sha256_overlap": int(overlap["required_pixel_overlap"]),
        "physical_group_overlap": int(
            overlap["required_physical_group_overlap"]
        ),
        "canonical_spatial_component_overlap": int(
            overlap["required_canonical_spatial_component_overlap"]
        ),
        "healpix_128_overlap": int(overlap["required_healpix_128_overlap"]),
        "within_38_arcsec_hosts": int(
            overlap["required_within_38_arcsec_pairs"]
        ),
    }
    minimum = float(overlap["minimum_angular_separation_arcsec"])
    for gate in gates:
        if not isinstance(gate, Mapping):
            raise PilotContractError("overlap gate must be an object")
        for field, expected in expected_values.items():
            value = gate.get(field)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) != expected
            ):
                raise PilotContractError(
                    f"numeric overlap gate failed for {gate.get('role')}/{field}: "
                    f"{value!r} != {expected}"
                )
        separation = gate.get("minimum_angular_separation_arcsec")
        if (
            isinstance(separation, (bool, np.bool_))
            or not isinstance(separation, (int, float, np.integer, np.floating))
            or not np.isfinite(float(separation))
            or float(separation) <= minimum
        ):
            raise PilotContractError(
                f"angular overlap gate failed for {gate.get('role')}: {separation!r}"
            )
    summaries = {
        "final_id_overlap": gates[0]["id_overlap"],
        "final_pixel_array_sha256_overlap": gates[0][
            "raw_array_sha256_overlap"
        ],
        "final_physical_group_overlap": gates[0]["physical_group_overlap"],
        "final_canonical_spatial_component_overlap": gates[0][
            "canonical_spatial_component_overlap"
        ],
        "final_healpix_128_overlap": gates[0]["healpix_128_overlap"],
        "final_within_38_arcsec_hosts": gates[0]["within_38_arcsec_hosts"],
        "threshold_calibration_pixel_array_sha256_overlap": gates[1][
            "raw_array_sha256_overlap"
        ],
        "achieved_fpr_pixel_array_sha256_overlap": gates[2][
            "raw_array_sha256_overlap"
        ],
    }
    mismatches = {
        field: (evidence.get(field), expected)
        for field, expected in summaries.items()
        if evidence.get(field) != expected
    }
    if mismatches:
        raise PilotContractError(f"overlap summary differs from numeric gates: {mismatches}")


def verify_zero_overlap(
    hosts: pd.DataFrame,
    comparison_pixel_hashes: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    assert_config_integrity(config)
    final_selection = pd.read_parquet(authority_path(config, "final_grec_selection"))
    test_identity = pd.read_parquet(
        authority_path(config, "test_manifest"),
        columns=["galaxy_id", "spatial_component_id"],
    )
    final_selection = final_selection.merge(
        test_identity,
        on="galaxy_id",
        how="left",
        validate="one_to_one",
    )
    current_hosts = pd.read_parquet(authority_path(config, "grec_host_manifest"))
    calibration = pd.read_parquet(
        authority_path(config, "threshold_calibration_role")
    )
    achieved = pd.read_parquet(authority_path(config, "achieved_fpr_role"))
    comparison_required = {"global_role", "galaxy_id", "raw_array_sha256"}
    if comparison_required - set(comparison_pixel_hashes):
        raise PilotContractError("comparison raw-pixel hash ledger is incomplete")
    calibration = calibration.merge(
        comparison_pixel_hashes[
            comparison_pixel_hashes["global_role"].eq("threshold_calibration")
        ][["galaxy_id", "raw_array_sha256"]],
        on="galaxy_id",
        how="left",
        validate="one_to_one",
    )
    achieved = achieved.merge(
        comparison_pixel_hashes[
            comparison_pixel_hashes["global_role"].eq("achieved_fpr")
        ][["galaxy_id", "raw_array_sha256"]],
        on="galaxy_id",
        how="left",
        validate="one_to_one",
    )
    final_selection = final_selection.merge(
        current_hosts[["host_galaxy_id", "host_pixel_sha256"]].rename(
            columns={
                "host_galaxy_id": "galaxy_id",
                "host_pixel_sha256": "raw_array_sha256",
            }
        ),
        on="galaxy_id",
        how="left",
        validate="one_to_one",
    )
    if (
        calibration["raw_array_sha256"].isna().any()
        or achieved["raw_array_sha256"].isna().any()
        or final_selection["raw_array_sha256"].isna().any()
    ):
        raise PilotContractError("raw-array hash coverage is incomplete")
    required_final = {
        "galaxy_id",
        "payload_sha256",
        "physical_group_id",
        "spatial_component_id",
        "healpix_128",
        "ra",
        "dec",
    }
    if required_final - set(final_selection):
        raise PilotContractError("final G_rec selection lacks overlap authority fields")
    records = [
        _overlap_record(
            hosts,
            final_selection,
            role="G_rec_final",
            id_column="galaxy_id",
            pixel_column="payload_sha256",
            raw_array_column="raw_array_sha256",
            ra_column="ra",
            dec_column="dec",
            minimum_separation_arcsec=float(
                config["overlap"]["minimum_angular_separation_arcsec"]
            ),
        ),
        _overlap_record(
            hosts,
            calibration,
            role="threshold_calibration",
            id_column="galaxy_id",
            pixel_column="payload_sha256",
            raw_array_column="raw_array_sha256",
            ra_column="ra",
            dec_column="dec",
            minimum_separation_arcsec=float(
                config["overlap"]["minimum_angular_separation_arcsec"]
            ),
        ),
        _overlap_record(
            hosts,
            achieved,
            role="achieved_fpr",
            id_column="galaxy_id",
            pixel_column="payload_sha256",
            raw_array_column="raw_array_sha256",
            ra_column="ra",
            dec_column="dec",
            minimum_separation_arcsec=float(
                config["overlap"]["minimum_angular_separation_arcsec"]
            ),
        ),
    ]
    evidence = {
        "status": "zero_current_role_overlap_verified",
        "gates": records,
        "final_id_overlap": records[0]["id_overlap"],
        "final_pixel_array_sha256_overlap": records[0][
            "raw_array_sha256_overlap"
        ],
        "final_physical_group_overlap": records[0]["physical_group_overlap"],
        "final_canonical_spatial_component_overlap": records[0][
            "canonical_spatial_component_overlap"
        ],
        "final_healpix_128_overlap": records[0]["healpix_128_overlap"],
        "final_within_38_arcsec_hosts": records[0]["within_38_arcsec_hosts"],
        "threshold_calibration_pixel_array_sha256_overlap": records[1][
            "raw_array_sha256_overlap"
        ],
        "achieved_fpr_pixel_array_sha256_overlap": records[2][
            "raw_array_sha256_overlap"
        ],
    }
    validate_overlap_evidence(evidence, config)
    return evidence


def prepare_design(
    config: Mapping[str, Any],
    *,
    junit_path: str | Path,
    aws_profile: str,
    workers: int,
) -> dict[str, Path]:
    assert_config_integrity(config)
    if workers < 1 or workers > 32:
        raise PilotContractError("acquisition workers must be in [1,32]")
    output = Path(config["paths"]["output_root"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if Path(junit_path).resolve() != output / "pytest.xml":
        raise PilotContractError("preparation requires the canonical JUnit report")
    artifact_paths = {
        "host_manifest": output / "design_host_manifest.parquet",
        "host_receipt": output / "DESIGN_HOST_RECEIPT.json",
        "comparison_pixel_hashes": output
        / "comparison_role_pixel_hashes.parquet",
        "comparison_pixel_receipt": output
        / "COMPARISON_ROLE_PIXEL_HASH_RECEIPT.json",
        "handoff_compatibility_receipt": output
        / "PNOISE_PRODUCTION_HANDOFF_COMPATIBILITY_RECEIPT.json",
        "deep_revalidation_receipt": output
        / "PRELAUNCH_DEEP_PIXEL_REVALIDATION_RECEIPT.json",
        "scene_manifest": output / "design_scene_manifest.parquet",
        "scene_receipt": output / "DESIGN_SCENE_RECEIPT.json",
        "ledger": output / "design_role_ledger.parquet",
        "ledger_receipt": output / "DESIGN_ROLE_LEDGER_RECEIPT.json",
        "registration": output / "PNOISE_DESIGN_REGISTRATION.json",
        "test_receipt": output / "DESIGN_TEST_RECEIPT.json",
        "freeze": output / "PNOISE_DESIGN_FREEZE_RECEIPT.json",
        "pilot_scores": output / config["pilot"]["score_artifact"],
        "pilot_receipt": output / config["pilot"]["score_receipt"],
    }
    require_absent(*artifact_paths.values())
    test_record = parse_passing_junit(junit_path, config)
    authority = validate_inherited_authority(config)
    handoff_compatibility, _ = build_prelaunch_handoff_compatibility(
        config,
        output,
    )
    atomic_write_json(
        artifact_paths["handoff_compatibility_receipt"],
        handoff_compatibility,
    )
    candidates = _historical_candidates(config)
    design = config["design"]
    selected = deterministic_balanced_selection(
        candidates,
        cells=design["supported_cells"],
        hosts_per_cell=int(design["hosts_per_cell"]),
        namespace=str(design["selection_namespace"]),
    )
    if len(selected) != int(design["expected_hosts"]):
        raise PilotContractError("selected design host count differs from registration")
    hosts = _stage_and_verify_hosts(
        selected,
        config,
        output,
        aws_profile=aws_profile,
        workers=workers,
    )
    comparison_pixel_hashes, comparison_pixel_receipt = (
        build_comparison_role_pixel_hashes(config)
    )
    atomic_write_parquet(
        artifact_paths["comparison_pixel_hashes"],
        comparison_pixel_hashes,
    )
    comparison_pixel_receipt["comparison_pixel_hash_manifest_sha256"] = (
        sha256_file(artifact_paths["comparison_pixel_hashes"])
    )
    atomic_write_json(
        artifact_paths["comparison_pixel_receipt"],
        comparison_pixel_receipt,
    )
    overlap = verify_zero_overlap(hosts, comparison_pixel_hashes, config)
    atomic_write_parquet(artifact_paths["host_manifest"], hosts)
    host_receipt = {
        "schema_version": 1,
        "status": "pnoise_design_hosts_frozen",
        "global_role": source_role(config),
        "rows": len(hosts),
        "cells": int(hosts["pilot_cell"].nunique()),
        "hosts_per_cell": int(design["hosts_per_cell"]),
        "distinct_physical_groups": int(hosts["physical_group_id"].nunique()),
        "distinct_historical_spatial_components": int(
            hosts["historical_spatial_component_id"].nunique()
        ),
        "historical_validation_negative_rows": int(len(candidates)),
        "historical_host_ancestry_exact": bool(
            hosts["historical_host_ancestry_exact"].all()
        ),
        "historical_cutout_tar_sha256": config["authorities"][
            "historical_cutout_tar"
        ]["sha256"],
        "historical_staging_receipt_sha256": config["authorities"][
            "historical_staging_receipt"
        ]["sha256"],
        "host_manifest_sha256": sha256_file(artifact_paths["host_manifest"]),
        "comparison_pixel_hash_manifest_sha256": sha256_file(
            artifact_paths["comparison_pixel_hashes"]
        ),
        "comparison_pixel_hash_receipt_sha256": sha256_file(
            artifact_paths["comparison_pixel_receipt"]
        ),
        "overlap": overlap,
        "config": config_provenance(config),
    }
    atomic_write_json(artifact_paths["host_receipt"], host_receipt)

    modules = import_predecessor_modules(config)
    predecessor = load_predecessor_config(config, modules)
    scenes = modules["scene_table"].build_scene_table(
        hosts,
        predecessor,
        expected_hosts=int(design["expected_hosts"]),
        expected_cells=int(design["expected_cells"]),
        hosts_per_cell=int(design["hosts_per_cell"]),
        radii_arcsec=design["radii_arcsec"],
        required_role=source_role(config),
    )
    if len(scenes) != int(design["expected_scenes"]):
        raise PilotContractError("design scene count differs from registration")
    atomic_write_parquet(artifact_paths["scene_manifest"], scenes)
    scene_receipt = {
        "schema_version": 1,
        "status": "pnoise_design_scenes_frozen_pre_score",
        "global_role": source_role(config),
        "rows": len(scenes),
        "distinct_hosts": int(scenes["host_galaxy_id"].nunique()),
        "cells": int(scenes[["psf_bin", "depth_bin"]].drop_duplicates().shape[0]),
        "hosts_per_cell": int(design["hosts_per_cell"]),
        "radii_arcsec": [float(value) for value in design["radii_arcsec"]],
        "draw_law": "exact predecessor G_rec scene-table implementation and scientific config",
        "predecessor_config_sha256": config["authorities"]["predecessor_config"][
            "sha256"
        ],
        "predecessor_scene_implementation_sha256": config["inherited_code"][
            "experiments_v2/grec_pnoise/src/grec_pnoise/scene_table.py"
        ],
        "design_host_manifest_sha256": sha256_file(
            artifact_paths["host_manifest"]
        ),
        "design_host_receipt_sha256": sha256_file(artifact_paths["host_receipt"]),
        "comparison_pixel_hash_manifest_sha256": sha256_file(
            artifact_paths["comparison_pixel_hashes"]
        ),
        "comparison_pixel_hash_receipt_sha256": sha256_file(
            artifact_paths["comparison_pixel_receipt"]
        ),
        "design_scene_manifest_sha256": sha256_file(
            artifact_paths["scene_manifest"]
        ),
        "overlap": overlap,
        "detector_outputs_used": False,
        "final_outcomes_used": False,
        "config": config_provenance(config),
    }
    atomic_write_json(artifact_paths["scene_receipt"], scene_receipt)

    ledger = scenes[
        [
            "scene_id",
            "grec_row_id",
            "host_galaxy_id",
            "host_parent_id",
            "host_pixel_sha256",
            "physical_group_id",
            "historical_spatial_component_id",
            "historical_healpix_128",
            "pilot_cell",
            "radius_index",
            "theta_e_arcsec",
            "source_parameter_id",
            "lens_parameter_id",
            "parameter_row_sha256",
        ]
    ].copy()
    ledger.insert(0, "global_role", source_role(config))
    ledger.insert(1, "experiment", "P_noise_design_pilot")
    ledger.insert(2, "parent_experiment_law", "G_rec")
    atomic_write_parquet(artifact_paths["ledger"], ledger)
    ledger_receipt = {
        "schema_version": 1,
        "status": "pnoise_design_role_ledger_frozen",
        "global_role": source_role(config),
        "rows": len(ledger),
        "distinct_hosts": int(ledger["host_galaxy_id"].nunique()),
        "distinct_scenes": int(ledger["scene_id"].nunique()),
        "design_role_ledger_sha256": sha256_file(artifact_paths["ledger"]),
        "design_scene_manifest_sha256": sha256_file(
            artifact_paths["scene_manifest"]
        ),
        "comparison_pixel_hash_manifest_sha256": sha256_file(
            artifact_paths["comparison_pixel_hashes"]
        ),
        "comparison_pixel_hash_receipt_sha256": sha256_file(
            artifact_paths["comparison_pixel_receipt"]
        ),
        "overlap": overlap,
        "final_outcomes_used": False,
        "config": config_provenance(config),
    }
    atomic_write_json(artifact_paths["ledger_receipt"], ledger_receipt)

    test_record.update(
        {
            "junit_path": str(Path(junit_path).resolve()),
            "successor_code_closure": successor_code_closure(config),
            "config": config_provenance(config),
        }
    )
    atomic_write_json(artifact_paths["test_receipt"], test_record)
    estimated_images = int(design["expected_scenes"]) * (
        1 + int(config["pilot"]["maximum_draw_count"])
    )
    raw_pixel_bytes = estimated_images * 3 * 101 * 101 * 4
    registration = {
        "schema_version": 1,
        "status": "pnoise_design_pilot_registered_pre_score",
        "authority": "adopted_successor_execution_authority",
        "scope": "design_role_only",
        "final_outcome_access": "forbidden_and_not_used",
        "predecessor_artifacts_created_under_successor": False,
        "inherited_authority": authority,
        "design": {
            "population": design["source_population"],
            "historical_validation_negative_candidates": len(candidates),
            "hosts": int(design["expected_hosts"]),
            "cells": int(design["expected_cells"]),
            "hosts_per_cell": int(design["hosts_per_cell"]),
            "radii_per_host": len(design["radii_arcsec"]),
            "scenes": int(design["expected_scenes"]),
            "one_host_per_physical_group": True,
            "one_host_per_historical_spatial_component": True,
            "selection_namespace": design["selection_namespace"],
            "overlap": overlap,
        },
        "pilot": {
            "maximum_draw_count": int(config["pilot"]["maximum_draw_count"]),
            "selectable_candidate_draw_counts": config["pilot"][
                "candidate_draw_counts"
            ],
            "diagnostic_nonselectable_draw_counts": config["pilot"][
                "diagnostic_nonselectable_draw_counts"
            ],
            "one_draw_mcse_status": config["inference"][
                "one_draw_mcse_status"
            ],
            "images_to_score": estimated_images,
            "fixed_batch_width": int(config["pilot"]["batch_size"]),
            "pixel_persistence": config["pilot"]["pixel_persistence"],
            "avoided_raw_pixel_bank_bytes": raw_pixel_bytes,
            "feature_extraction": "forbidden",
            "final_rendering_or_scoring": "forbidden",
        },
        "threshold": {
            "receipt_sha256": config["authorities"]["threshold_receipt"]["sha256"],
            "float32_bits": config["scoring"]["threshold_float32_bits"],
            "comparison": config["scoring"]["comparison"],
        },
        "scorer": {
            "checkpoint_sha256": config["scoring"]["checkpoint_sha256"],
            "closure_sha256": config["scoring"][
                "authoritative_scorer_closure_sha256"
            ],
            "backend_profile_sha256": config["scoring"][
                "authoritative_backend_profile_sha256"
            ],
            "batch_size": int(config["pilot"]["batch_size"]),
            "filler_policy": config["pilot"]["filler_policy"],
        },
        "config": config_provenance(config),
        "successor_code_closure": successor_code_closure(config),
        "inherited_code_closure": inherited_code_closure(config),
        "artifacts": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in artifact_paths.items()
            if name
            in {
                "host_manifest",
                "host_receipt",
                "comparison_pixel_hashes",
                "comparison_pixel_receipt",
                "handoff_compatibility_receipt",
                "scene_manifest",
                "scene_receipt",
                "ledger",
                "ledger_receipt",
                "test_receipt",
            }
        },
    }
    atomic_write_json(artifact_paths["registration"], registration)
    freeze = {
        "schema_version": 1,
        "status": "pnoise_design_pilot_review_freeze_complete",
        "safe_to_launch_expensive_pilot": True,
        "tests_passed": True,
        "test_receipt_sha256": sha256_file(artifact_paths["test_receipt"]),
        "registration_sha256": sha256_file(artifact_paths["registration"]),
        "design_scene_manifest_sha256": sha256_file(
            artifact_paths["scene_manifest"]
        ),
        "design_scene_receipt_sha256": sha256_file(
            artifact_paths["scene_receipt"]
        ),
        "design_role_ledger_sha256": sha256_file(artifact_paths["ledger"]),
        "design_role_ledger_receipt_sha256": sha256_file(
            artifact_paths["ledger_receipt"]
        ),
        "design_host_manifest_sha256": sha256_file(
            artifact_paths["host_manifest"]
        ),
        "design_host_receipt_sha256": sha256_file(artifact_paths["host_receipt"]),
        "comparison_pixel_hash_manifest_sha256": sha256_file(
            artifact_paths["comparison_pixel_hashes"]
        ),
        "comparison_pixel_hash_receipt_sha256": sha256_file(
            artifact_paths["comparison_pixel_receipt"]
        ),
        "handoff_compatibility_receipt_sha256": sha256_file(
            artifact_paths["handoff_compatibility_receipt"]
        ),
        "overlap": overlap,
        "final_outcomes_used": False,
        "successor_code_closure": successor_code_closure(config),
        "inherited_code_closure": inherited_code_closure(config),
        "config": config_provenance(config),
    }
    atomic_write_json(artifact_paths["freeze"], freeze)
    return artifact_paths
