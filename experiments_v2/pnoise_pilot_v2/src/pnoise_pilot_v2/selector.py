from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .authority import import_predecessor_modules, load_predecessor_config
from .compatibility import (
    production_draw_receipt,
    validate_with_actual_production_validator,
)
from .contracts import (
    PilotContractError,
    assert_config_integrity,
    atomic_write_json,
    config_provenance,
    fixed_width_unicode,
    inherited_code_closure,
    load_json,
    require_absent,
    sha256_file,
    stable_id,
    successor_code_closure,
)
from .streaming import validate_design_freeze

PILOT_SCORE_ARRAYS = {
    "scene_ids",
    "host_ids",
    "cell_ids",
    "radius_index",
    "theta_e_arcsec",
    "draw_indices",
    "expectation_logits",
    "physical_logits",
    "expectation_decisions",
    "physical_decisions",
    "photon_draw_key_bytes",
    "threshold_float32_bits",
}


def _expected_photon_key_bytes(
    scene_ids: Sequence[str],
    draw_count: int,
    namespace: str,
) -> np.ndarray:
    expected = np.empty((len(scene_ids), draw_count, 20), dtype=np.uint8)
    for scene_index, scene_id in enumerate(scene_ids):
        for draw_index in range(draw_count):
            key = stable_id(
                "photon",
                [namespace, str(scene_id), int(draw_index)],
                length=40,
            )
            expected[scene_index, draw_index] = np.frombuffer(
                bytes.fromhex(key.split("_", 1)[1]),
                dtype=np.uint8,
            )
    return expected


def validate_pilot_score_payload(
    arrays: Mapping[str, np.ndarray],
    scenes: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    scene_count: int,
    draw_count: int,
    threshold_float32_bits: int,
    photon_stream_namespace: str,
    global_role: str,
) -> dict[str, np.ndarray]:
    if set(arrays) != PILOT_SCORE_ARRAYS:
        raise PilotContractError(
            f"pilot score arrays differ from contract: {sorted(set(arrays))}"
        )
    if len(scenes) != scene_count or len(ledger) != scene_count:
        raise PilotContractError("frozen scene/ledger cardinality changed")
    values = {name: np.asarray(arrays[name]) for name in PILOT_SCORE_ARRAYS}
    expected_shapes = {
        "scene_ids": (scene_count,),
        "host_ids": (scene_count,),
        "cell_ids": (scene_count,),
        "radius_index": (scene_count,),
        "theta_e_arcsec": (scene_count,),
        "draw_indices": (draw_count,),
        "expectation_logits": (scene_count,),
        "physical_logits": (scene_count, draw_count),
        "expectation_decisions": (scene_count,),
        "physical_decisions": (scene_count, draw_count),
        "photon_draw_key_bytes": (scene_count, draw_count, 20),
        "threshold_float32_bits": (1,),
    }
    expected_dtypes = {
        "radius_index": np.dtype(np.int16),
        "theta_e_arcsec": np.dtype(np.float32),
        "draw_indices": np.dtype(np.int16),
        "expectation_logits": np.dtype(np.float32),
        "physical_logits": np.dtype(np.float32),
        "expectation_decisions": np.dtype(np.bool_),
        "physical_decisions": np.dtype(np.bool_),
        "photon_draw_key_bytes": np.dtype(np.uint8),
        "threshold_float32_bits": np.dtype(np.uint32),
    }
    for name, shape in expected_shapes.items():
        if values[name].shape != shape:
            raise PilotContractError(
                f"pilot array {name} shape {values[name].shape} != {shape}"
            )
    for name, dtype in expected_dtypes.items():
        if values[name].dtype != dtype:
            raise PilotContractError(
                f"pilot array {name} dtype {values[name].dtype} != {dtype}"
            )
    expected_scene_ids = fixed_width_unicode(
        scenes["scene_id"].tolist(),
        field="scene_ids",
    )
    expected_host_ids = fixed_width_unicode(
        scenes["host_galaxy_id"].tolist(),
        field="host_ids",
    )
    expected_cell_ids = fixed_width_unicode(
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
    for name, expected in (
        ("scene_ids", expected_scene_ids),
        ("host_ids", expected_host_ids),
        ("cell_ids", expected_cell_ids),
    ):
        if values[name].dtype.kind != "U" or values[name].dtype != expected.dtype:
            raise PilotContractError(
                f"pilot identity array {name} is not exact fixed-width Unicode"
            )
    if (
        not np.array_equal(values["scene_ids"], expected_scene_ids)
        or not np.array_equal(values["host_ids"], expected_host_ids)
        or not np.array_equal(values["cell_ids"], expected_cell_ids)
        or not np.array_equal(
            values["radius_index"],
            scenes["radius_index"].to_numpy(dtype=np.int16),
        )
        or not np.array_equal(
            values["theta_e_arcsec"],
            scenes["theta_e_arcsec"].to_numpy(dtype=np.float32),
        )
        or not np.array_equal(
            values["draw_indices"],
            np.arange(draw_count, dtype=np.int16),
        )
    ):
        raise PilotContractError("pilot ordered scene/host/design identities changed")
    if (
        not ledger["global_role"].eq(global_role).all()
        or not np.array_equal(
            ledger["scene_id"].astype(str).to_numpy(),
            expected_scene_ids,
        )
        or not np.array_equal(
            ledger["host_galaxy_id"].astype(str).to_numpy(),
            expected_host_ids,
        )
    ):
        raise PilotContractError("pilot identities differ from ordered design ledger")

    bits = int(values["threshold_float32_bits"][0])
    if bits != int(threshold_float32_bits):
        raise PilotContractError("pilot threshold bits differ from frozen threshold")
    threshold = np.array([bits], dtype=np.uint32).view(np.float32)[0]
    if (
        not np.isfinite(values["expectation_logits"]).all()
        or not np.isfinite(values["physical_logits"]).all()
    ):
        raise PilotContractError("pilot logits contain non-finite values")
    expected_expectation = values["expectation_logits"] >= threshold
    expected_physical = values["physical_logits"] >= threshold
    if (
        not np.array_equal(values["expectation_decisions"], expected_expectation)
        or not np.array_equal(values["physical_decisions"], expected_physical)
    ):
        raise PilotContractError("pilot decisions do not recompute from logits")
    expected_keys = _expected_photon_key_bytes(
        expected_scene_ids,
        draw_count,
        photon_stream_namespace,
    )
    if not np.array_equal(values["photon_draw_key_bytes"], expected_keys):
        raise PilotContractError("pilot photon keys are incomplete or out of order")
    return values


def validate_frozen_pilot_scores(
    arrays: Mapping[str, np.ndarray],
    scenes: pd.DataFrame,
    ledger: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    assert_config_integrity(config)
    return validate_pilot_score_payload(
        arrays,
        scenes,
        ledger,
        scene_count=int(config["design"]["expected_scenes"]),
        draw_count=int(config["pilot"]["maximum_draw_count"]),
        threshold_float32_bits=int(config["scoring"]["threshold_float32_bits"]),
        photon_stream_namespace=str(
            config["pnoise"]["photon_stream_namespace"]
        ),
        global_role=str(config["design"]["global_role"]),
    )


def evaluate_draw_panel(
    physical_decisions: np.ndarray,
    expectation_decisions: np.ndarray,
    host_ids: Sequence[str],
    cell_ids: Sequence[str],
    candidate_draw_counts: Sequence[int],
    *,
    absolute_mcse_max: float,
    fraction_of_sampling_se_max: float,
) -> dict[str, Any]:
    physical = np.asarray(physical_decisions)
    expectation = np.asarray(expectation_decisions)
    hosts = np.asarray(host_ids, dtype=str)
    cells = np.asarray(cell_ids, dtype=str)
    if (
        physical.ndim != 2
        or physical.shape[1] < 2
        or expectation.shape != (physical.shape[0],)
        or hosts.shape != expectation.shape
        or cells.shape != expectation.shape
    ):
        raise PilotContractError("pilot score arrays have incompatible shapes")
    if not np.isin(physical, [0, 1]).all() or not np.isin(
        expectation, [0, 1]
    ).all():
        raise PilotContractError("draw selector requires binary decisions")
    frame = pd.DataFrame(
        {
            "scene": np.arange(len(hosts)),
            "host": hosts,
            "cell": cells,
        }
    )
    if frame.groupby("host")["cell"].nunique().max() != 1:
        raise PilotContractError("one design host crosses pilot cells")
    host_scene_counts = frame.groupby("host").size()
    if host_scene_counts.nunique() != 1:
        raise PilotContractError("design hosts do not have equal radius support")
    hosts_per_cell = frame[["host", "cell"]].drop_duplicates().groupby("cell").size()
    if hosts_per_cell.nunique() != 1 or len(hosts_per_cell) < 2:
        raise PilotContractError("draw selector requires a balanced multicell pilot")

    unique_cells = sorted(hosts_per_cell.index.astype(str))
    cell_count = len(unique_cells)
    scene_counts = host_scene_counts.to_dict()
    cell_host_counts = hosts_per_cell.to_dict()
    weights = np.array(
        [
            1.0
            / (
                cell_count
                * int(cell_host_counts[cell])
                * int(scene_counts[host])
            )
            for host, cell in zip(hosts, cells, strict=True)
        ],
        dtype=np.float64,
    )
    if not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("pilot scene weights do not sum to one")

    scene_variance = physical.astype(np.float64).var(axis=1, ddof=1)
    maximum_draw_effect = (
        physical.astype(np.float64).mean(axis=1) - expectation.astype(np.float64)
    )
    effects = pd.DataFrame(
        {"host": hosts, "cell": cells, "effect": maximum_draw_effect}
    )
    host_effects = (
        effects.groupby(["cell", "host"], sort=True)["effect"].mean().reset_index()
    )
    sampling_variance = 0.0
    for cell in unique_cells:
        values = host_effects.loc[
            host_effects["cell"].eq(cell), "effect"
        ].to_numpy(dtype=np.float64)
        if values.size < 2:
            raise PilotContractError("sampling SE needs at least two hosts per cell")
        sampling_variance += (
            values.var(ddof=1) / values.size / (cell_count**2)
        )
    sampling_se = math.sqrt(sampling_variance)
    relative_limit = float(fraction_of_sampling_se_max) * sampling_se

    maximum_draws = physical.shape[1]
    candidates = sorted({int(value) for value in candidate_draw_counts})
    if not candidates or candidates[0] < 2 or candidates[-1] > maximum_draws:
        raise PilotContractError("selectable draw panel must be within [2,pilot maximum]")
    records: list[dict[str, Any]] = [
        {
            "draw_count": 1,
            "selectable": False,
            "mcse_status": "unavailable_by_inference_contract",
            "estimated_mcse": None,
            "absolute_pass": False,
            "relative_pass": False,
            "diagnostic_only": True,
        }
    ]
    selected: int | None = None
    for count in candidates:
        estimated_mcse = math.sqrt(
            float(np.sum(np.square(weights) * scene_variance)) / count
        )
        conservative_bernoulli_mcse = math.sqrt(
            float(np.sum(np.square(weights) * 0.25)) / count
        )
        absolute_pass = estimated_mcse <= float(absolute_mcse_max)
        relative_pass = estimated_mcse <= relative_limit
        record = {
            "draw_count": count,
            "selectable": True,
            "mcse_status": "estimated_from_64_draw_design_panel",
            "estimated_mcse": estimated_mcse,
            "conservative_bernoulli_mcse": conservative_bernoulli_mcse,
            "absolute_limit": float(absolute_mcse_max),
            "relative_limit": relative_limit,
            "absolute_pass": absolute_pass,
            "relative_pass": relative_pass,
            "diagnostic_only": False,
        }
        records.append(record)
        if selected is None and absolute_pass and relative_pass:
            selected = count
    if selected is None:
        raise PilotContractError(
            "no selectable draw count passes both frozen MCSE gates; "
            "do not launch final P_noise"
        )
    selected_record = next(
        record for record in records if record["draw_count"] == selected
    )
    return {
        "selected_draw_count": selected,
        "estimated_mcse": selected_record["estimated_mcse"],
        "estimated_sampling_se": sampling_se,
        "relative_mcse_fraction": (
            selected_record["estimated_mcse"] / sampling_se
            if sampling_se > 0
            else None
        ),
        "absolute_mcse_gate_passed": selected_record["absolute_pass"],
        "relative_mcse_gate_passed": selected_record["relative_pass"],
        "candidate_results": records,
        "design_cells": cell_count,
        "design_hosts": int(host_effects["host"].nunique()),
        "design_scenes": int(physical.shape[0]),
        "pilot_draws_per_scene": maximum_draws,
        "weighting": (
            "equal_cell_then_equal_host_after_radius_average; "
            "draws averaged within scene"
        ),
    }


def select_draw_count(config: Mapping[str, Any]) -> Path:
    assert_config_integrity(config)
    output = Path(config["paths"]["output_root"]).resolve()
    draw_receipt_path = output / "PNOISE_DRAW_COUNT_RECEIPT.json"
    require_absent(draw_receipt_path)
    freeze = validate_design_freeze(config)
    scores_path = output / str(config["pilot"]["score_artifact"])
    score_receipt_path = output / str(config["pilot"]["score_receipt"])
    score_receipt = load_json(score_receipt_path)
    expected = {
        "status": "design_only_pnoise_pilot_scores_complete",
        "global_role": config["design"]["global_role"],
        "final_outcomes_used": False,
        "pilot_scores_sha256": sha256_file(scores_path),
        "threshold_receipt_sha256": config["authorities"]["threshold_receipt"][
            "sha256"
        ],
        "scorer_closure_sha256": config["scoring"][
            "authoritative_scorer_closure_sha256"
        ],
        "backend_profile_sha256": config["scoring"][
            "authoritative_backend_profile_sha256"
        ],
    }
    failures = {
        key: (score_receipt.get(key), value)
        for key, value in expected.items()
        if score_receipt.get(key) != value
    }
    if failures:
        raise PilotContractError(f"pilot score receipt is invalid: {failures}")
    scenes = pd.read_parquet(output / "design_scene_manifest.parquet")
    ledger = pd.read_parquet(output / "design_role_ledger.parquet")
    with np.load(scores_path, allow_pickle=False) as archive:
        arrays = validate_frozen_pilot_scores(
            {name: np.array(archive[name], copy=True) for name in archive.files},
            scenes,
            ledger,
            config,
        )
        array_record = score_receipt.get("pilot_scores_arrays", {})
        expected_array_record = {
            "scenes": int(config["design"]["expected_scenes"]),
            "draws_per_scene": int(config["pilot"]["maximum_draw_count"]),
            "images_scored": int(config["design"]["expected_scenes"])
            * (int(config["pilot"]["maximum_draw_count"]) + 1),
            "photon_keys_recorded": int(config["design"]["expected_scenes"])
            * int(config["pilot"]["maximum_draw_count"]),
            "scene_id_unicode_width": arrays["scene_ids"].dtype.itemsize
            // np.dtype("U1").itemsize,
            "host_id_unicode_width": arrays["host_ids"].dtype.itemsize
            // np.dtype("U1").itemsize,
            "cell_id_unicode_width": arrays["cell_ids"].dtype.itemsize
            // np.dtype("U1").itemsize,
        }
        if array_record != expected_array_record:
            raise PilotContractError("pilot score receipt string/array shape record changed")
        result = evaluate_draw_panel(
            arrays["physical_decisions"],
            arrays["expectation_decisions"],
            arrays["host_ids"],
            arrays["cell_ids"],
            config["pilot"]["candidate_draw_counts"],
            absolute_mcse_max=float(config["inference"]["mcse_absolute_max"]),
            fraction_of_sampling_se_max=float(
                config["inference"]["mcse_fraction_of_sampling_se_max"]
            ),
        )
    successor_fields = {
        "one_draw_selectable": False,
        "one_draw_mcse_status": config["inference"]["one_draw_mcse_status"],
        "pilot_scores_sha256": sha256_file(scores_path),
        "pilot_score_receipt_sha256": sha256_file(score_receipt_path),
        "design_freeze_receipt_sha256": sha256_file(
            output / "PNOISE_DESIGN_FREEZE_RECEIPT.json"
        ),
        "design_registration_sha256": sha256_file(
            output / "PNOISE_DESIGN_REGISTRATION.json"
        ),
        "design_scene_manifest_sha256": sha256_file(
            output / "design_scene_manifest.parquet"
        ),
        "design_role_ledger_sha256": sha256_file(
            output / "design_role_ledger.parquet"
        ),
        "threshold_receipt_sha256": config["authorities"]["threshold_receipt"][
            "sha256"
        ],
        "parent_render_authority": {
            name: config["authorities"][name]["sha256"]
            for name in (
                "grec_parent_manifest",
                "grec_parent_receipt",
                "grec_scene_manifest",
                "grec_scene_receipt",
                "renderer_validation_receipt",
                "quadrature_amendment",
                "governance_amendment",
            )
        },
        "checkpoint_sha256": config["scoring"]["checkpoint_sha256"],
        "scorer_closure_sha256": config["scoring"][
            "authoritative_scorer_closure_sha256"
        ],
        "backend_profile_sha256": config["scoring"][
            "authoritative_backend_profile_sha256"
        ],
        "successor_code_closure": successor_code_closure(config),
        "inherited_code_closure": inherited_code_closure(config),
        "config": config_provenance(config),
        "freeze_status": freeze["status"],
        **result,
    }
    modules = import_predecessor_modules(config)
    predecessor = load_predecessor_config(config, modules)
    receipt = production_draw_receipt(
        config,
        predecessor,
        modules,
        successor_fields,
    )
    candidate_path = output / ".PNOISE_DRAW_COUNT_RECEIPT.candidate.json"
    atomic_write_json(candidate_path, receipt)
    try:
        validate_with_actual_production_validator(
            candidate_path,
            config,
            predecessor,
            modules,
        )
    except Exception:
        candidate_path.unlink(missing_ok=True)
        raise
    candidate_path.replace(draw_receipt_path)
    return draw_receipt_path
