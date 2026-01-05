from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from dhs.data import DatasetConfig
from dhs.transforms import AugmentConfig
from dhs.train import TrainConfig, train_one

# Default Lambda NFS root; replaced by --data_root when provided for portability.
DEFAULT_DATA_ROOT = "/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration"


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


def _write_run_info(
    out_dir: str,
    config_path: str,
    dataset_seed: int,
    config_hash: str | None = None,
) -> None:
    """Write run_info.json next to checkpoints for reproducibility."""
    os.makedirs(out_dir, exist_ok=True)
    commit = _git_commit()
    config_abs = str(Path(config_path).resolve())
    if config_hash is None and os.path.isfile(config_path):
        with open(config_path, "rb") as f:
            config_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    run_info = {
        "git_commit": commit,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "config_path": config_abs,
        "config_sha256_prefix": config_hash,
        "command": " ".join(sys.argv),
        "dataset_seed": dataset_seed,
    }
    path = os.path.join(out_dir, "run_info.json")
    with open(path, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"Wrote {path}")


def _apply_data_root(manifest_path: str, out_dir: str, data_root: str) -> tuple[str, str]:
    """Replace DEFAULT_DATA_ROOT with data_root in paths for portability."""
    new_manifest = manifest_path.replace(DEFAULT_DATA_ROOT, data_root.rstrip("/"), 1)
    new_out = out_dir.replace(DEFAULT_DATA_ROOT, data_root.rstrip("/"), 1)
    return new_manifest, new_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Override data root for manifest_path and out_dir (replaces Lambda NFS path for portability).",
    )
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    dcfg = DatasetConfig(**cfg["dataset"])
    aug = AugmentConfig(**cfg.get("augment", {}))
    tcfg = TrainConfig(**cfg["train"])

    if args.data_root:
        dcfg.manifest_path, tcfg.out_dir = _apply_data_root(
            dcfg.manifest_path, tcfg.out_dir, args.data_root
        )
        print(f"Using data_root: {args.data_root} -> manifest_path={dcfg.manifest_path!r} out_dir={tcfg.out_dir!r}")

    _write_run_info(tcfg.out_dir, args.config, dcfg.seed)
    best_path, best_auc = train_one(tcfg, dcfg, aug)
    print(f"BEST_AUC={best_auc:.4f} BEST_CKPT={best_path}")


if __name__ == "__main__":
    main()
