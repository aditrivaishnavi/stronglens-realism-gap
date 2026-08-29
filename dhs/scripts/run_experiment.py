from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import traceback
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
import yaml

from dhs import data_splits
from dhs.data import DatasetConfig
from dhs.train import TrainConfig, train_one
from dhs.transforms import AugmentConfig


def _git_commit(cwd: str | None = None) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd or Path(__file__).resolve().parent.parent.parent,
            timeout=5,
        )
        return (r.stdout or "").strip()[:12] if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _append_event(out_dir: str, event: dict) -> None:
    path = Path(out_dir) / "run_events.jsonl"
    with path.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _manifest_summary(path: str, label_col: str) -> dict:
    frame = pd.read_parquet(path, columns=["split", label_col, "cohort"])
    counts = (
        frame.groupby(["split", label_col], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    return {
        "rows": int(len(frame)),
        "split_label_counts": [
            {
                "split": str(row.split),
                "label": int(getattr(row, label_col)),
                "rows": int(row.rows),
            }
            for row in counts.itertuples(index=False)
        ],
        "cohort_counts": {
            str(key): int(value)
            for key, value in frame["cohort"].value_counts().to_dict().items()
        },
        "test_rows": int((frame["split"] == "test").sum()),
    }


def _runtime_info() -> dict:
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
    }


def _write_run_info(
    out_dir: str,
    config_path: str,
    dcfg: DatasetConfig,
    tcfg: TrainConfig,
    *,
    status: str,
    failure: str | None = None,
    best_path: str | None = None,
    best_metric: float | None = None,
    split_check: dict | None = None,
) -> dict:
    """Write a reproducible run record next to checkpoints."""
    os.makedirs(out_dir, exist_ok=True)
    commit = _git_commit()
    config_abs = str(Path(config_path).resolve())
    config_hash = _sha256_file(config_path)
    manifest_summary = _manifest_summary(dcfg.manifest_path, dcfg.label_col)
    if tcfg.mixture_id and manifest_summary["test_rows"] != 0:
        raise ValueError("Mixture training manifests must not contain test rows")
    lineage_hash = None
    if tcfg.lineage_path:
        if not os.path.isfile(tcfg.lineage_path):
            raise FileNotFoundError(f"Missing mixture lineage: {tcfg.lineage_path}")
        lineage_hash = _sha256_file(tcfg.lineage_path)
    source_root = Path(__file__).resolve().parents[2]
    source_files = [
        source_root / "dhs" / "data.py",
        source_root / "dhs" / "train.py",
        source_root / "dhs" / "model.py",
        source_root / "dhs" / "preprocess.py",
        Path(__file__).resolve(),
    ]
    run_info = {
        "git_commit": commit,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "config_path": config_abs,
        "config_sha256": config_hash,
        "command": " ".join(sys.argv),
        "dataset_seed": int(dcfg.seed),
        "mixture_id": tcfg.mixture_id,
        "manifest_path": str(Path(dcfg.manifest_path).resolve()),
        "manifest_sha256": _sha256_file(dcfg.manifest_path),
        "manifest_summary": manifest_summary,
        "split_check": split_check,
        "lineage_path": tcfg.lineage_path or None,
        "lineage_sha256": lineage_hash,
        "cutout_root": dcfg.cutout_root,
        "initialization": {
            "imagenet_pretrained": bool(tcfg.pretrained),
            "init_weights": tcfg.init_weights or None,
            "init_weights_sha256": (
                _sha256_file(tcfg.init_weights) if tcfg.init_weights else None
            ),
        },
        "source_sha256": {
            str(path.relative_to(source_root)): _sha256_file(path)
            for path in source_files
        },
        "runtime": _runtime_info(),
        "failure": failure,
        "best_checkpoint": best_path,
        "best_checkpoint_sha256": (
            _sha256_file(best_path) if best_path and os.path.isfile(best_path) else None
        ),
        "selection_metric": tcfg.selection_metric,
        "best_selection_value": best_metric,
    }
    path = Path(out_dir) / "run_info.json"
    _atomic_json(run_info, path)
    print(f"Wrote {path}")
    return run_info


def _apply_data_root(
    manifest_path: str, out_dir: str, data_root: str
) -> tuple[str, str]:
    """Resolve relative configuration paths below a user-supplied data root."""
    root = Path(data_root)
    manifest = Path(manifest_path)
    output = Path(out_dir)
    return (
        str(manifest if manifest.is_absolute() else root / manifest),
        str(output if output.is_absolute() else root / output),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Resolve relative manifest and output paths below this directory.",
    )
    ap.add_argument(
        "--cutout_root",
        type=str,
        default=None,
        help="Override the shared cutout root used for relative manifest paths.",
    )
    data_splits.add_arguments(ap)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dcfg = DatasetConfig(**cfg["dataset"])
    aug = AugmentConfig(**cfg.get("augment", {}))
    tcfg = TrainConfig(**cfg["train"])

    if args.data_root:
        dcfg.manifest_path, tcfg.out_dir = _apply_data_root(
            dcfg.manifest_path, tcfg.out_dir, args.data_root
        )
        if tcfg.lineage_path:
            lineage = Path(tcfg.lineage_path)
            if not lineage.is_absolute():
                tcfg.lineage_path = str(Path(args.data_root) / lineage)
        print(
            f"Using data_root: {args.data_root} -> manifest_path={dcfg.manifest_path!r} out_dir={tcfg.out_dir!r}"
        )
    if args.cutout_root:
        dcfg.cutout_root = args.cutout_root

    manifest = pd.read_parquet(
        dcfg.manifest_path, columns=["galaxy_id", "split"]
    )
    split_check = data_splits.check_frames(
        {
            "gradient_training_rows": manifest[manifest["split"] == "train"],
            "checkpoint_selection_rows": manifest[manifest["split"] == "val"],
        },
        purpose=f"model fitting from {dcfg.manifest_path}",
        allow_test_rows=args.allow_test_rows,
        test_manifest=args.test_manifest,
    )

    os.makedirs(tcfg.out_dir, exist_ok=True)
    started = {
        "event": "run_started",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "mixture_id": tcfg.mixture_id,
        "seed": int(dcfg.seed),
    }
    _append_event(tcfg.out_dir, started)
    _write_run_info(
        tcfg.out_dir,
        args.config,
        dcfg,
        tcfg,
        status="started",
        split_check=split_check,
    )
    try:
        best_path, best_metric = train_one(tcfg, dcfg, aug)
    except Exception as error:
        failure = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        _append_event(
            tcfg.out_dir,
            {
                "event": "run_failed",
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "error": str(error),
            },
        )
        _write_run_info(
            tcfg.out_dir,
            args.config,
            dcfg,
            tcfg,
            status="failed",
            failure=failure,
            split_check=split_check,
        )
        raise
    _append_event(
        tcfg.out_dir,
        {
            "event": "run_completed",
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "best_checkpoint": best_path,
            "selection_metric": tcfg.selection_metric,
            "best_selection_value": float(best_metric),
        },
    )
    _write_run_info(
        tcfg.out_dir,
        args.config,
        dcfg,
        tcfg,
        status="completed",
        best_path=best_path,
        best_metric=float(best_metric),
        split_check=split_check,
    )
    print(
        f"BEST_{tcfg.selection_metric.upper()}={best_metric:.4f} BEST_CKPT={best_path}"
    )


if __name__ == "__main__":
    main()
