from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    ContractError,
    config_provenance,
    load_json,
    source_closure,
)
from .utils import atomic_json, safe_relative, sha256_file


def _centroid(image: np.ndarray, pixel_scale: float) -> tuple[float, float]:
    values = np.asarray(image, dtype=np.float64)
    y, x = np.mgrid[: values.shape[0], : values.shape[1]]
    total = values.sum(dtype=np.float64)
    return (
        float((x * values).sum(dtype=np.float64) / total),
        float((y * values).sum(dtype=np.float64) / total),
    )


def validate_source_package(
    *,
    selection_root: Path,
    source_root: Path,
    config: dict[str, Any],
    output: Path | None,
) -> dict[str, Any]:
    if output is not None and output.exists():
        raise FileExistsError(output)
    selection_receipt = load_json(selection_root / "selection_receipt.json")
    source_receipt = load_json(source_root / "receipt.json")
    ledger_path = selection_root / "source_eligibility.parquet"
    selected_path = selection_root / "selected_sources.parquet"
    manifest_path = source_root / "source_manifest.parquet"
    if (
        selection_receipt.get("source_eligibility_sha256") != sha256_file(ledger_path)
        or selection_receipt.get("selected_sources_sha256")
        != sha256_file(selected_path)
        or selection_receipt.get("config") != config_provenance(config)
        or source_receipt.get("source_manifest_sha256") != sha256_file(manifest_path)
        or source_receipt.get("config") != config_provenance(config)
        or source_receipt.get("source_closure") != source_closure(config)
    ):
        raise ContractError("source selection/construction receipt drift")
    ledger = pd.read_parquet(ledger_path)
    selected = pd.read_parquet(selected_path)
    manifest = pd.read_parquet(manifest_path)
    if (
        len(selected) != config["selection"]["source"]["selected_sources"]
        or set(selected["source_id"]) != set(manifest["source_id"])
        or not selected["construction_preflight_passed"].all()
        or not manifest["structured_is_latent_truth"].eq(False).all()
        or not manifest["structured_contains_residual_measurement_uncertainty"]
        .eq(True)
        .all()
        or manifest["source_image_sha256"].isna().any()
        or not manifest["source_image_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
        or manifest["source_psf_image_sha256"].isna().any()
    ):
        raise ContractError("source package population/truth labels are invalid")
    construction = config["source_construction"]
    pixel_scale = float(construction["source_pixel_scale_arcsec"])
    tolerance = float(construction["centroid_separation_tolerance_arcsec"])
    inventory = hashlib.sha256()
    centroid_separations = []
    integral_errors = []
    correlations = []
    for row in manifest.sort_values("source_id").itertuples(index=False):
        path = safe_relative(source_root, str(row.source_map_relpath))
        observed = sha256_file(path)
        if observed != str(row.source_map_sha256):
            raise ContractError(f"source map drift: {row.source_id}")
        inventory.update(f"{row.source_map_relpath}\t{observed}\n".encode())
        with np.load(path, allow_pickle=False) as payload:
            structured = np.asarray(payload["structured"], dtype=np.float64)
            sersic = np.asarray(payload["sersic"], dtype=np.float64)
            support = np.asarray(payload["common_support"], dtype=bool)
        grid_y, grid_x = np.mgrid[:161, :161]
        expected_support = np.hypot(grid_x - 80, grid_y - 80) * pixel_scale <= float(
            row.common_support_radius_arcsec
        )
        measured_native_radius = (
            construction["native_inscribed_radius_fraction_for_eligibility"]
            * 0.5
            * (min(row.native_height_pixels, row.native_width_pixels) - 1)
            * float(row.native_pixel_scale_arcsec)
        )
        expected_common_radius = min(
            measured_native_radius,
            construction["maximum_common_working_support_radius_arcsec"],
        )
        if (
            structured.shape != (161, 161)
            or sersic.shape != structured.shape
            or support.shape != structured.shape
            or np.any(structured < 0)
            or np.any(sersic < 0)
            or not np.isfinite(structured).all()
            or not np.isfinite(sersic).all()
            or np.any(structured[~support] != 0)
            or np.any(sersic[~support] != 0)
            or not np.array_equal(support, expected_support)
            or not np.isclose(
                row.common_support_radius_arcsec,
                expected_common_radius,
                rtol=0,
                atol=1e-15,
            )
            or float(row.common_support_radius_arcsec)
            > construction["maximum_common_working_support_radius_arcsec"]
        ):
            raise ContractError("saved source support/nonnegative gate failed")
        integrals = np.array(
            [
                structured.sum(dtype=np.float64) * pixel_scale**2,
                sersic.sum(dtype=np.float64) * pixel_scale**2,
            ]
        )
        if np.max(np.abs(integrals - 1.0)) > 2e-7:
            raise ContractError("saved source unit-integral gate failed")
        sc = _centroid(structured, pixel_scale)
        tc = _centroid(sersic, pixel_scale)
        separation = float(np.hypot(sc[0] - tc[0], sc[1] - tc[1]) * pixel_scale)
        if separation > tolerance:
            raise ContractError("saved source centroid gate failed")
        if (
            row.structured_positive_flux_retained_fraction
            < construction["minimum_positive_flux_retained_in_common_support"]
            or row.sersic_positive_flux_retained_fraction
            < construction["minimum_positive_flux_retained_in_common_support"]
            or row.structured_preprojection_negative_to_positive_flux
            > construction["maximum_structured_negative_to_positive_flux"]
            or row.sersic_preprojection_negative_to_positive_flux
            > construction["maximum_sersic_negative_to_positive_flux"]
        ):
            raise ContractError("saved source projection/truncation gate failed")
        centroid_separations.append(separation)
        integral_errors.extend(np.abs(integrals - 1.0))
        correlations.append(
            float(np.corrcoef(structured.ravel(), sersic.ravel())[0, 1])
        )
    if inventory.hexdigest() != source_receipt["source_map_inventory_sha256"]:
        raise ContractError("complete source inventory digest drift")
    result = {
        "schema_version": 2,
        "status": "pass",
        "detector_outcomes_used": False,
        "catalogue_rows": len(ledger),
        "eligible_sources": int(ledger["eligible"].sum()),
        "selected_sources": len(manifest),
        "maximum_centroid_separation_arcsec": float(np.max(centroid_separations)),
        "maximum_integral_absolute_error": float(np.max(integral_errors)),
        "minimum_structured_positive_flux_retained": float(
            manifest["structured_positive_flux_retained_fraction"].min()
        ),
        "minimum_sersic_positive_flux_retained": float(
            manifest["sersic_positive_flux_retained_fraction"].min()
        ),
        "maximum_structured_negative_to_positive_flux": float(
            manifest["structured_preprojection_negative_to_positive_flux"].max()
        ),
        "maximum_sersic_negative_to_positive_flux": float(
            manifest["sersic_preprojection_negative_to_positive_flux"].max()
        ),
        "structured_sersic_correlation": {
            "minimum": float(np.min(correlations)),
            "median": float(np.median(correlations)),
            "maximum": float(np.max(correlations)),
        },
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_map_inventory_sha256": inventory.hexdigest(),
        "config": config_provenance(config),
        "source_closure": source_closure(config),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output, result)
    return result


def _verify_repeat_source_tree(
    root: Path,
    receipt: dict[str, Any],
) -> None:
    manifest_path = root / "source_manifest.parquet"
    if sha256_file(manifest_path) != receipt.get("source_manifest_sha256"):
        raise ContractError("source repeat manifest drift")
    manifest = pd.read_parquet(manifest_path)
    inventory = hashlib.sha256()
    for row in manifest.sort_values("source_id").itertuples(index=False):
        path = safe_relative(root, str(row.source_map_relpath))
        observed = sha256_file(path)
        if observed != str(row.source_map_sha256):
            raise ContractError(f"source repeat map drift: {row.source_id}")
        inventory.update(f"{row.source_map_relpath}\t{observed}\n".encode())
    if inventory.hexdigest() != receipt.get("source_map_inventory_sha256"):
        raise ContractError("source repeat complete map inventory drift")


def validate_source_determinism(
    *,
    primary_source_root: Path,
    repeated_source_root: Path,
    config: dict[str, Any],
    output: Path | None,
) -> dict[str, Any]:
    """Bind byte-identical independent source-construction executions."""
    primary_receipt = load_json(primary_source_root / "receipt.json")
    repeated_receipt = load_json(repeated_source_root / "receipt.json")
    required_status = "restricted_empirical_source_construction_design_only"
    for receipt in (primary_receipt, repeated_receipt):
        if (
            receipt.get("status") != required_status
            or receipt.get("config") != config_provenance(config)
            or receipt.get("source_closure") != source_closure(config)
        ):
            raise ContractError("source determinism receipt is stale")
    _verify_repeat_source_tree(primary_source_root, primary_receipt)
    _verify_repeat_source_tree(repeated_source_root, repeated_receipt)
    fields = [
        "source_manifest_sha256",
        "source_map_inventory_sha256",
        "sources",
        "construction_label",
    ]
    if any(
        primary_receipt.get(field) != repeated_receipt.get(field) for field in fields
    ):
        raise ContractError("independent source executions are not byte-identical")
    if output is not None and output.exists():
        raise FileExistsError(output)
    result = {
        "schema_version": 2,
        "status": "source_construction_byte_determinism_pass",
        "independent_executions": 2,
        "source_manifest_sha256": primary_receipt["source_manifest_sha256"],
        "source_map_inventory_sha256": primary_receipt["source_map_inventory_sha256"],
        "primary_receipt_sha256": sha256_file(primary_source_root / "receipt.json"),
        "repeated_receipt_sha256": sha256_file(repeated_source_root / "receipt.json"),
        "config": config_provenance(config),
        "source_closure": source_closure(config),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output, result)
    return result
