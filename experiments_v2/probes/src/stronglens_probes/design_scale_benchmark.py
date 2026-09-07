"""Design-only 49,500 x 2,048 grouped full-refit benchmark."""

from __future__ import annotations

import resource
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .contracts import ContractError, atomic_write_json


def run_design_scale_refit_benchmark(
    *,
    work_path: Path,
    output_path: Path,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Benchmark the actual dense grouped probe shape without final data."""
    if not 1 <= draws <= 10:
        raise ContractError("design-scale benchmark draws must be between 1 and 10")
    references = 43
    hosts = 4_500
    radii = 11
    simulated_rows = hosts * radii
    rows = references + simulated_rows
    dimension = 2_048
    work_path.parent.mkdir(parents=True, exist_ok=True)
    if not work_path.exists():
        values = np.lib.format.open_memmap(
            work_path,
            mode="w+",
            dtype=np.float32,
            shape=(rows, dimension),
        )
        values[:] = 0.0
        rng = np.random.default_rng(seed)
        values[:references, :16] = rng.normal(0.8, 0.7, size=(references, 16)).astype(
            np.float32
        )
        for start in range(0, hosts, 250):
            stop = min(start + 250, hosts)
            base = rng.normal(0.0, 0.8, size=(stop - start, 16))
            expanded = np.repeat(base, radii, axis=0)
            expanded += rng.normal(0.0, 0.05, size=expanded.shape)
            row_start = references + start * radii
            row_stop = references + stop * radii
            values[row_start:row_stop, :16] = expanded.astype(np.float32)
        values.flush()
        del values
    features = np.load(work_path, mmap_mode="r", allow_pickle=False)
    if features.shape != (49_543, 2_048) or features.dtype != np.float32:
        raise ContractError("design-scale feature fixture has the wrong shape")
    labels = np.r_[
        np.ones(references, dtype=np.int8),
        np.zeros(simulated_rows, dtype=np.int8),
    ]
    host_id = np.repeat(np.arange(hosts), radii)
    folds = np.r_[np.arange(references) % 5, host_id % 5]
    point_estimates = []
    iterations = []
    started = time.perf_counter()
    for draw in range(draws):
        rng = np.random.default_rng(np.random.SeedSequence([seed, draw]))
        reference_weight = rng.gamma(1.0, 1.0, size=references)
        host_weight = rng.gamma(1.0, 1.0, size=hosts)
        weights = np.r_[reference_weight, host_weight[host_id] / radii]
        scores = np.empty(rows, dtype=float)
        for fold in range(5):
            evaluate = folds == fold
            train = ~evaluate
            train_x = np.asarray(features[train], dtype=np.float32)
            scaler = StandardScaler().fit(train_x, sample_weight=weights[train])
            transformed = scaler.transform(train_x)
            estimator = LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=5_000,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                estimator.fit(
                    transformed,
                    labels[train],
                    sample_weight=weights[train],
                )
            iterations.append(int(estimator.n_iter_[0]))
            scores[evaluate] = estimator.decision_function(
                scaler.transform(np.asarray(features[evaluate], dtype=np.float32))
            )
            del train_x, transformed, scaler, estimator
        point_estimates.append(
            float(roc_auc_score(labels, scores, sample_weight=weights))
        )
    elapsed = time.perf_counter() - started
    seconds_per_draw = elapsed / draws
    result = {
        "schema_version": 1,
        "status": "complete_design_scale_full_refit_benchmark",
        "contains_final_data": False,
        "shape": [rows, dimension],
        "references": references,
        "G_rec_hosts": hosts,
        "radii_per_host": radii,
        "G_rec_scenes": simulated_rows,
        "folds": 5,
        "draws": draws,
        "fold_fits": draws * 5,
        "elapsed_seconds": elapsed,
        "seconds_per_draw": seconds_per_draw,
        "serial_hours_for_10000_draws": seconds_per_draw * 10_000 / 3_600,
        "ideal_registered_two_worker_hours_for_10000_draws": (
            seconds_per_draw * 10_000 / 3_600 / 2
        ),
        "maximum_logistic_iterations": max(iterations),
        "auc_values": point_estimates,
        "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "interpretation": (
            "Dense design-shape timing fixture with 16 active and 2032 zero "
            "dimensions. It measures memory movement and estimator overhead but may "
            "underestimate iterations for final dense astronomical features."
        ),
    }
    atomic_write_json(result, output_path)
    return result
