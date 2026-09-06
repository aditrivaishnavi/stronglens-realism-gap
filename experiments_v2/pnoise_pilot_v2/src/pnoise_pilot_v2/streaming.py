from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .authority import (
    import_predecessor_modules,
    load_predecessor_config,
    validate_inherited_authority,
)
from .compatibility import (
    build_prelaunch_handoff_compatibility,
    validate_handoff_compatibility_receipt,
)
from .contracts import (
    PilotContractError,
    array_bytes_sha256,
    assert_config_integrity,
    atomic_write_json,
    atomic_write_npz,
    config_provenance,
    fixed_width_unicode,
    inherited_code_closure,
    load_json,
    require_absent,
    sha256_file,
    successor_code_closure,
    validate_test_receipt,
)
from .deep_validation import deep_revalidate_frozen_pixels
from .prepare import validate_overlap_evidence, verify_zero_overlap


def write_streamed_pilot_scores(
    path: str | Path,
    scenes: pd.DataFrame,
    *,
    draw_count: int,
    expectation_logits: np.ndarray,
    physical_logits: np.ndarray,
    expectation_decisions: np.ndarray,
    physical_decisions: np.ndarray,
    photon_draw_key_bytes: np.ndarray,
    threshold_float32_bits: int,
) -> dict[str, int]:
    """Production NPZ serializer, also used by the bounded integration test."""
    scene_ids = fixed_width_unicode(
        scenes["scene_id"].tolist(),
        field="scene_ids",
    )
    host_ids = fixed_width_unicode(
        scenes["host_galaxy_id"].tolist(),
        field="host_ids",
    )
    cell_ids = fixed_width_unicode(
        [
            f"{float(psf):.2f}|{float(depth):.1f}"
            for psf, depth in zip(
                scenes["psf_bin"],
                scenes["depth_bin"],
                strict=True,
            )
        ],
        field="cell_ids",
    )
    atomic_write_npz(
        path,
        scene_ids=scene_ids,
        host_ids=host_ids,
        cell_ids=cell_ids,
        radius_index=scenes["radius_index"].to_numpy(dtype=np.int16),
        theta_e_arcsec=scenes["theta_e_arcsec"].to_numpy(dtype=np.float32),
        draw_indices=np.arange(draw_count, dtype=np.int16),
        expectation_logits=np.asarray(expectation_logits, dtype=np.float32),
        physical_logits=np.asarray(physical_logits, dtype=np.float32),
        expectation_decisions=np.asarray(expectation_decisions, dtype=np.bool_),
        physical_decisions=np.asarray(physical_decisions, dtype=np.bool_),
        photon_draw_key_bytes=np.asarray(photon_draw_key_bytes, dtype=np.uint8),
        threshold_float32_bits=np.array(
            [threshold_float32_bits],
            dtype=np.uint32,
        ),
    )
    return {
        "scene_id_unicode_width": scene_ids.dtype.itemsize
        // np.dtype("U1").itemsize,
        "host_id_unicode_width": host_ids.dtype.itemsize
        // np.dtype("U1").itemsize,
        "cell_id_unicode_width": cell_ids.dtype.itemsize
        // np.dtype("U1").itemsize,
    }


def validate_design_freeze(config: Mapping[str, Any]) -> dict[str, Any]:
    assert_config_integrity(config)
    output = Path(config["paths"]["output_root"]).resolve()
    freeze_path = output / "PNOISE_DESIGN_FREEZE_RECEIPT.json"
    freeze = load_json(freeze_path)
    expected = {
        "status": "pnoise_design_pilot_review_freeze_complete",
        "safe_to_launch_expensive_pilot": True,
        "tests_passed": True,
        "final_outcomes_used": False,
        "registration_sha256": sha256_file(
            output / "PNOISE_DESIGN_REGISTRATION.json"
        ),
        "design_scene_manifest_sha256": sha256_file(
            output / "design_scene_manifest.parquet"
        ),
        "design_scene_receipt_sha256": sha256_file(
            output / "DESIGN_SCENE_RECEIPT.json"
        ),
        "design_role_ledger_sha256": sha256_file(
            output / "design_role_ledger.parquet"
        ),
        "design_role_ledger_receipt_sha256": sha256_file(
            output / "DESIGN_ROLE_LEDGER_RECEIPT.json"
        ),
        "design_host_manifest_sha256": sha256_file(
            output / "design_host_manifest.parquet"
        ),
        "design_host_receipt_sha256": sha256_file(
            output / "DESIGN_HOST_RECEIPT.json"
        ),
        "comparison_pixel_hash_manifest_sha256": sha256_file(
            output / "comparison_role_pixel_hashes.parquet"
        ),
        "comparison_pixel_hash_receipt_sha256": sha256_file(
            output / "COMPARISON_ROLE_PIXEL_HASH_RECEIPT.json"
        ),
        "handoff_compatibility_receipt_sha256": sha256_file(
            output / "PNOISE_PRODUCTION_HANDOFF_COMPATIBILITY_RECEIPT.json"
        ),
    }
    failures = {
        key: (freeze.get(key), value)
        for key, value in expected.items()
        if freeze.get(key) != value
    }
    if failures:
        raise PilotContractError(f"design freeze is stale or incomplete: {failures}")
    if freeze.get("config") != config_provenance(config):
        raise PilotContractError("design freeze binds another successor config")
    if freeze.get("successor_code_closure") != successor_code_closure(config):
        raise PilotContractError("successor code changed after design review freeze")
    if freeze.get("inherited_code_closure") != inherited_code_closure(config):
        raise PilotContractError("inherited code changed after design review freeze")
    overlap = freeze.get("overlap", {})
    validate_overlap_evidence(overlap, config)
    test_receipt = validate_test_receipt(
        config,
        output,
        expected_receipt_sha256=str(freeze["test_receipt_sha256"]),
    )
    compatibility_path = (
        output / "PNOISE_PRODUCTION_HANDOFF_COMPATIBILITY_RECEIPT.json"
    )
    compatibility = validate_handoff_compatibility_receipt(
        config,
        compatibility_path,
    )
    recomputed_compatibility, _ = build_prelaunch_handoff_compatibility(
        config,
        output,
    )
    if recomputed_compatibility != compatibility:
        raise PilotContractError(
            "actual production handoff compatibility differs from frozen receipt"
        )
    comparison_path = output / "comparison_role_pixel_hashes.parquet"
    comparison_receipt = load_json(
        output / "COMPARISON_ROLE_PIXEL_HASH_RECEIPT.json"
    )
    expected_comparison = {
        "status": "pnoise_comparison_role_raw_pixel_hashes_complete",
        "rows": 32_514,
        "role_rows": {
            "achieved_fpr": 12_512,
            "threshold_calibration": 20_002,
        },
        "raw_array_key": "cutout",
        "raw_array_dtype": "float32",
        "raw_array_shape": [101, 101, 3],
        "per_row_payload_sha256_verified": True,
        "per_row_raw_array_sha256_recorded": True,
        "comparison_pixel_hash_manifest_sha256": sha256_file(comparison_path),
        "negative_role_receipt_sha256": config["authorities"][
            "negative_role_receipt"
        ]["sha256"],
        "config": config_provenance(config),
    }
    comparison_failures = {
        key: (comparison_receipt.get(key), value)
        for key, value in expected_comparison.items()
        if comparison_receipt.get(key) != value
    }
    if comparison_failures:
        raise PilotContractError(
            f"comparison raw-pixel receipt is stale: {comparison_failures}"
        )
    expected_sources = {
        "threshold_calibration": config["authorities"][
            "threshold_calibration_role"
        ]["sha256"],
        "achieved_fpr": config["authorities"]["achieved_fpr_role"]["sha256"],
    }
    if comparison_receipt.get("source_role_manifest_sha256") != expected_sources:
        raise PilotContractError("comparison raw-pixel source bindings changed")
    comparison = pd.read_parquet(comparison_path)
    required_comparison = {
        "comparison_row_id",
        "global_role",
        "galaxy_id",
        "source_role_manifest_sha256",
        "source_payload_sha256",
        "raw_array_sha256",
    }
    if (
        required_comparison - set(comparison)
        or len(comparison) != 32_514
        or comparison["comparison_row_id"].astype(str).duplicated().any()
        or comparison[["global_role", "galaxy_id"]].duplicated().any()
        or comparison[list(required_comparison)].isna().any().any()
        or not comparison["raw_array_sha256"]
        .astype(str)
        .str.fullmatch(r"[0-9a-f]{64}")
        .all()
    ):
        raise PilotContractError("comparison raw-pixel manifest is incomplete")
    observed_counts = (
        comparison["global_role"].value_counts().sort_index().to_dict()
    )
    if observed_counts != expected_comparison["role_rows"]:
        raise PilotContractError("comparison raw-pixel role counts changed")
    for role, expected_sha256 in expected_sources.items():
        role_rows = comparison[comparison["global_role"].eq(role)]
        if not role_rows["source_role_manifest_sha256"].eq(expected_sha256).all():
            raise PilotContractError(f"{role} raw-pixel rows bind another source")
    hosts = pd.read_parquet(output / "design_host_manifest.parquet")
    recomputed_overlap = verify_zero_overlap(hosts, comparison, config)
    if recomputed_overlap != overlap:
        raise PilotContractError("frozen overlap evidence differs from recomputation")
    for receipt_name in (
        "DESIGN_HOST_RECEIPT.json",
        "DESIGN_SCENE_RECEIPT.json",
        "DESIGN_ROLE_LEDGER_RECEIPT.json",
    ):
        receipt = load_json(output / receipt_name)
        if (
            receipt.get("overlap") != overlap
            or receipt.get("config") != config_provenance(config)
        ):
            raise PilotContractError(f"{receipt_name} overlap/config binding is stale")
    registration = load_json(output / "PNOISE_DESIGN_REGISTRATION.json")
    if (
        registration.get("design", {}).get("overlap") != overlap
        or registration.get("config") != config_provenance(config)
        or registration.get("artifacts", {})
        .get("comparison_pixel_hashes", {})
        .get("sha256")
        != sha256_file(comparison_path)
        or registration.get("artifacts", {})
        .get("handoff_compatibility_receipt", {})
        .get("sha256")
        != sha256_file(compatibility_path)
    ):
        raise PilotContractError("registration raw-pixel/overlap binding is stale")
    if test_receipt.get("status") != "pnoise_design_pilot_tests_passed":
        raise PilotContractError("design test receipt status is invalid")
    validated = dict(freeze)
    validated["deep_pixel_revalidation"] = deep_revalidate_frozen_pixels(
        config,
        output,
    )
    return validated


def write_prelaunch_deep_validation_receipt(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    assert_config_integrity(config)
    output = Path(config["paths"]["output_root"]).resolve()
    freeze = validate_design_freeze(config)
    receipt = {
        "schema_version": 1,
        **freeze["deep_pixel_revalidation"],
        "design_freeze_receipt_sha256": sha256_file(
            output / "PNOISE_DESIGN_FREEZE_RECEIPT.json"
        ),
        "design_host_manifest_sha256": sha256_file(
            output / "design_host_manifest.parquet"
        ),
        "comparison_pixel_hash_manifest_sha256": sha256_file(
            output / "comparison_role_pixel_hashes.parquet"
        ),
        "successor_code_closure": successor_code_closure(config),
        "config": config_provenance(config),
        "final_outcomes_used": False,
    }
    path = output / "PRELAUNCH_DEEP_PIXEL_REVALIDATION_RECEIPT.json"
    atomic_write_json(path, receipt)
    return receipt, path


def _extended_render(
    row: Mapping[str, Any],
    host: np.ndarray,
    predecessor: Mapping[str, Any],
    modules: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, float | int]]:
    rendering = modules["rendering"]
    orders = tuple(int(value) for value in predecessor["render"]["quadrature_orders"])
    amendment = (48, 64, 96)
    if orders != (24, 32):
        raise PilotContractError("quadrature amendment is bound to base orders 24,32")
    previous: np.ndarray | None = None
    for order in (*orders, *amendment):
        image, flux_error = rendering.render_source_expectation_at_order(
            row,
            host.shape,
            predecessor,
            order=order,
            device=predecessor["render"]["authoritative_device"],
        )
        if previous is not None:
            difference = np.abs(image.astype(np.float64) - previous.astype(np.float64))
            relative_l1 = float(
                difference.sum()
                / max(np.abs(image.astype(np.float64)).sum(), 1.0e-30)
            )
            relative_max = float(
                difference.max()
                / max(np.abs(image.astype(np.float64)).max(), 1.0e-30)
            )
            if (
                flux_error
                <= float(predecessor["render"]["unlensed_flux_relative_tolerance"])
                and relative_l1
                <= float(predecessor["render"]["lensed_relative_l1_tolerance"])
                and relative_max
                <= float(predecessor["render"]["lensed_relative_max_tolerance"])
            ):
                return image, {
                    "order": order,
                    "unlensed_flux_relative_error": float(flux_error),
                    "previous_order_relative_l1": relative_l1,
                    "previous_order_relative_max": relative_max,
                }
        previous = image
    raise PilotContractError(
        f"design scene {row['scene_id']} failed unchanged convergence through order 96"
    )


def run_streaming_pilot(config: Mapping[str, Any]) -> tuple[Path, Path]:
    """Render one scene at a time and score without retaining detector-input pixels."""
    assert_config_integrity(config)
    output = Path(config["paths"]["output_root"]).resolve()
    scores_path = output / str(config["pilot"]["score_artifact"])
    receipt_path = output / str(config["pilot"]["score_receipt"])
    require_absent(scores_path, receipt_path)
    freeze = validate_design_freeze(config)
    authority = validate_inherited_authority(config)
    modules = import_predecessor_modules(config)
    predecessor = load_predecessor_config(config, modules)
    threshold_path = Path(config["authorities"]["threshold_receipt"]["path"])
    threshold_receipt, threshold = modules["scoring"].validate_scoring_authority(
        predecessor,
        threshold_path,
        device=str(config["pilot"]["authoritative_device"]),
    )
    if (
        threshold_receipt["scorer_closure_sha256"]
        != config["scoring"]["authoritative_scorer_closure_sha256"]
        or threshold_receipt["backend_profile"]["profile_sha256"]
        != config["scoring"]["authoritative_backend_profile_sha256"]
        or int(np.asarray(threshold).view(np.uint32))
        != int(config["scoring"]["threshold_float32_bits"])
    ):
        raise PilotContractError("authoritative threshold/scorer closure changed")

    scenes_path = output / "design_scene_manifest.parquet"
    ledger_path = output / "design_role_ledger.parquet"
    scenes = pd.read_parquet(scenes_path)
    ledger = pd.read_parquet(ledger_path)
    design = config["design"]
    maximum_draws = int(config["pilot"]["maximum_draw_count"])
    if (
        len(scenes) != int(design["expected_scenes"])
        or scenes["scene_id"].duplicated().any()
        or not scenes["final_role"].eq(config["design"]["global_role"]).all()
    ):
        raise PilotContractError("design scene table violates frozen counts/role")
    joined = scenes[
        ["scene_id", "host_galaxy_id", "host_pixel_sha256"]
    ].merge(
        ledger[
            ["scene_id", "host_galaxy_id", "host_pixel_sha256", "global_role"]
        ],
        on=["scene_id", "host_galaxy_id", "host_pixel_sha256"],
        how="left",
        validate="one_to_one",
    )
    if (
        joined["global_role"].isna().any()
        or not joined["global_role"].eq(config["design"]["global_role"]).all()
    ):
        raise PilotContractError("scene table is not exactly covered by design ledger")

    rows = scenes.to_dict(orient="records")
    key_bytes = np.empty((len(rows), maximum_draws, 20), dtype=np.uint8)
    render_records: list[dict[str, float | int]] = []
    gains = modules["pnoise"].gains_from_config(predecessor)
    successor_gains = np.array(
        [
            config["pnoise"]["gain_electron_per_nanomaggy_g"],
            config["pnoise"]["gain_electron_per_nanomaggy_r"],
            config["pnoise"]["gain_electron_per_nanomaggy_z"],
        ],
        dtype=np.float64,
    )
    if not np.array_equal(gains, successor_gains):
        raise PilotContractError("successor photon gain differs from predecessor law")
    if predecessor["pnoise"]["photon_stream_namespace"] != config["pnoise"][
        "photon_stream_namespace"
    ]:
        raise PilotContractError("successor photon stream differs from production law")

    def detector_inputs() -> Iterator[np.ndarray]:
        cached_path: str | None = None
        cached_host: np.ndarray | None = None
        for scene_index, row in enumerate(rows):
            path = str(row["host_pixel_path"])
            if path != cached_path:
                if sha256_file(path) != str(row["host_payload_sha256"]):
                    raise PilotContractError("design host payload changed after freeze")
                with np.load(path, allow_pickle=False) as archive:
                    cached_host = np.array(archive["cutout"], copy=True)
                if (
                    cached_host.dtype != np.float32
                    or cached_host.shape != (101, 101, 3)
                    or not np.isfinite(cached_host).all()
                    or array_bytes_sha256(cached_host)
                    != str(row["host_pixel_sha256"])
                ):
                    raise PilotContractError("invalid frozen design host pixels")
                cached_path = path
            assert cached_host is not None
            modules["scene_table"].verify_scene_identity(row, predecessor)
            arc, record = _extended_render(row, cached_host, predecessor, modules)
            render_records.append(record)
            expectation = (
                cached_host.transpose(2, 0, 1) + arc
            ).astype(np.float32, copy=False)
            yield expectation
            for draw_index in range(maximum_draws):
                photon_arc, photon_key = modules["pnoise"].draw_photon_arc(
                    arc,
                    gains,
                    stream_namespace=config["pnoise"]["photon_stream_namespace"],
                    scene_id=str(row["scene_id"]),
                    draw_index=draw_index,
                )
                digest = bytes.fromhex(photon_key.split("_", 1)[1])
                if len(digest) != 20:
                    raise PilotContractError("photon key is not a 160-bit identity")
                key_bytes[scene_index, draw_index] = np.frombuffer(
                    digest, dtype=np.uint8
                )
                physical = (
                    cached_host.transpose(2, 0, 1) + photon_arc
                ).astype(np.float32, copy=False)
                yield physical

    logits, scoring_record = modules["scoring"].design_only_raw_logits(
        detector_inputs(),
        predecessor,
        device=str(config["pilot"]["authoritative_device"]),
    )
    expected_images = len(rows) * (maximum_draws + 1)
    if logits.shape != (expected_images,) or len(render_records) != len(rows):
        raise PilotContractError("streaming producer did not score the complete panel")
    if (
        scoring_record["scorer_closure_sha256"]
        != threshold_receipt["scorer_closure_sha256"]
        or scoring_record["backend_profile"] != threshold_receipt["backend_profile"]
        or scoring_record["batch"]["inference_batch_size"]
        != int(config["pilot"]["batch_size"])
        or scoring_record["batch"]["filler_policy"]
        != config["pilot"]["filler_policy"]
    ):
        raise PilotContractError("pilot scoring departed from threshold scorer closure")

    matrix = logits.reshape(len(rows), maximum_draws + 1)
    expectation_logits = np.asarray(matrix[:, 0], dtype=np.float32)
    physical_logits = np.asarray(matrix[:, 1:], dtype=np.float32)
    expectation_decisions = expectation_logits >= threshold
    physical_decisions = physical_logits >= threshold
    serialization = write_streamed_pilot_scores(
        scores_path,
        scenes,
        draw_count=maximum_draws,
        expectation_logits=expectation_logits,
        physical_logits=physical_logits,
        expectation_decisions=expectation_decisions,
        physical_decisions=physical_decisions,
        photon_draw_key_bytes=key_bytes,
        threshold_float32_bits=int(config["scoring"]["threshold_float32_bits"]),
    )
    order_counts = Counter(int(record["order"]) for record in render_records)
    receipt = {
        "schema_version": 1,
        "status": "design_only_pnoise_pilot_scores_complete",
        "global_role": config["design"]["global_role"],
        "final_outcomes_used": False,
        "final_outcome_paths_accessed": [],
        "feature_extraction_used": False,
        "final_pnoise_rendering_used": False,
        "pixel_bank_persisted": False,
        "streaming_contract": "one expectation plus 64 physical images per scene; fixed-width batches only",
        "pilot_scores_sha256": sha256_file(scores_path),
        "pilot_scores_arrays": {
            "scenes": len(rows),
            "draws_per_scene": maximum_draws,
            "images_scored": expected_images,
            "photon_keys_recorded": int(key_bytes.shape[0] * key_bytes.shape[1]),
            **serialization,
        },
        "design_freeze_receipt_sha256": sha256_file(
            output / "PNOISE_DESIGN_FREEZE_RECEIPT.json"
        ),
        "design_registration_sha256": sha256_file(
            output / "PNOISE_DESIGN_REGISTRATION.json"
        ),
        "design_scene_manifest_sha256": sha256_file(scenes_path),
        "design_scene_receipt_sha256": sha256_file(
            output / "DESIGN_SCENE_RECEIPT.json"
        ),
        "design_role_ledger_sha256": sha256_file(ledger_path),
        "design_role_ledger_receipt_sha256": sha256_file(
            output / "DESIGN_ROLE_LEDGER_RECEIPT.json"
        ),
        "threshold_receipt_sha256": sha256_file(threshold_path),
        "threshold_float32_bits": int(config["scoring"]["threshold_float32_bits"]),
        "comparison": config["scoring"]["comparison"],
        "checkpoint_sha256": config["scoring"]["checkpoint_sha256"],
        "scorer_closure": scoring_record["scorer_closure"],
        "scorer_closure_sha256": scoring_record["scorer_closure_sha256"],
        "backend_profile": scoring_record["backend_profile"],
        "backend_profile_sha256": scoring_record["backend_profile"]["profile_sha256"],
        "fixed_width": scoring_record["batch"],
        "batch_decoupling": scoring_record["decoupling"],
        "render_authority": {
            "predecessor_config_sha256": config["authorities"][
                "predecessor_config"
            ]["sha256"],
            "renderer_validation_receipt_sha256": config["authorities"][
                "renderer_validation_receipt"
            ]["sha256"],
            "quadrature_amendment_sha256": config["authorities"][
                "quadrature_amendment"
            ]["sha256"],
            "quadrature_order_counts": {
                str(order): count for order, count in sorted(order_counts.items())
            },
            "maximum_unlensed_flux_relative_error": max(
                float(record["unlensed_flux_relative_error"])
                for record in render_records
            ),
            "maximum_previous_order_relative_l1": max(
                float(record["previous_order_relative_l1"])
                for record in render_records
            ),
            "maximum_previous_order_relative_max": max(
                float(record["previous_order_relative_max"])
                for record in render_records
            ),
        },
        "photon_law": config["pnoise"],
        "photon_draw_key_encoding": config["pilot"][
            "photon_draw_key_encoding"
        ],
        "inherited_authority": authority,
        "successor_code_closure": successor_code_closure(config),
        "inherited_code_closure": inherited_code_closure(config),
        "config": config_provenance(config),
        "design_freeze": {
            "status": freeze["status"],
            "zero_overlap_status": freeze["overlap"]["status"],
        },
        "launch_time_deep_pixel_revalidation": freeze[
            "deep_pixel_revalidation"
        ],
    }
    atomic_write_json(receipt_path, receipt)
    return scores_path, receipt_path
