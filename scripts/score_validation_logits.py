#!/usr/bin/env python
"""Score validation negatives and derive detector operating points.

The primary scoring contract retains raw logits, uses the training batch width
for every forward pass, pads only the final short batch, and discards filler
outputs. Operating points rank the raw float32 logits and classify with
``score >= threshold``. Probabilities are retained for display only.

cuDNN uses deterministic, non-benchmarking mode. The optional comparison input
checks the resulting probabilities against an earlier stored score table.

The split is hard-wired to ``val`` and a manifest carrying any other partition is
refused, as in the script this one is derived from.

Usage:

    python scripts/score_validation_logits.py \\
        --config <materialised train_seed_42.yaml> \\
        --checkpoint <.../epoch_160.pt> \\
        --out-dir outputs/validation_scoring
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dhs import data_splits  # noqa: E402
from dhs.data import DatasetConfig, LensDataset, SplitConfig  # noqa: E402
from dhs.fixed_width_scoring import (  # noqa: E402
    CONTRACT_FILLER,
    FILLER_POLICIES,
    ScoringError,
    assert_no_batch_coupling,
    fixed_width_logits,
    operating_points,
    pin_deterministic_kernels,
)
from dhs.model import build_model  # noqa: E402
from dhs.train import TrainConfig, _collate, _collate_weighted  # noqa: E402
from dhs.transforms import AugmentConfig  # noqa: E402

CARRY = ("galaxy_id", "label", "cohort", "tier")

# The allowances the plan reads operating points at, plus the neighbours needed to
# show whether a rung is bracketed by ties.
ALLOWANCES = (1, 2, 3, 4, 5, 6, 7, 8, 12, 30, 60, 61, 62, 123)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def refuse_any_test_partition(manifest_path: str) -> None:
    splits = set(
        pd.read_parquet(manifest_path, columns=["split"])["split"].astype(str).unique()
    )
    forbidden = splits - {"train", "val"}
    if forbidden:
        raise ScoringError(
            f"{manifest_path} contains partitions {sorted(forbidden)}. This script is "
            "restricted to model-development rows."
        )


def load_model(checkpoint: Path, arch: str, device: str) -> torch.nn.Module:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if "model" not in ckpt:
        raise ScoringError(f"{checkpoint} has no 'model' entry; keys are {sorted(ckpt)}")
    model = build_model(arch)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def logits_in_row_order(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    batch_size: int,
    filler: str = CONTRACT_FILLER,
) -> tuple[np.ndarray, dict[str, object]]:
    """Raw logits in manifest order, through the shared contract.

    Scoring 62,215 rows at batch 64 gives 972 full batches and a terminal batch
    of 7 real rows, so 57 fillers are appended and their outputs dropped.

    The loader yields ``(images, ...)`` tuples, so the images are unwrapped here.
    Everything else is ``dhs.fixed_width_scoring``, the same implementation used
    for injections and confirmed systems.
    """
    return fixed_width_logits(
        model,
        (batch[0] for batch in loader),
        batch_size=batch_size,
        device=device,
        filler=filler,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cutout-root", default=None)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "the contract is the config's train.batch_size. Overriding it changes "
            "the accumulation order and therefore the operating point, so it is "
            "refused unless --allow-variable-batch is also given"
        ),
    )
    ap.add_argument(
        "--allow-variable-batch",
        action="store_true",
        help="required to accept --batch-size; marks the output as a sensitivity run",
    )
    ap.add_argument(
        "--filler-policy",
        default="last",
        choices=FILLER_POLICIES,
        help=(
            "which rows pad the terminal batch. The contract is 'last'. The others "
            "exist only to measure invariance and mark the output as non-governing"
        ),
    )
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    data_splits.add_arguments(ap)
    args = ap.parse_args()

    # Kernel selection must not vary between runs. benchmark=True would let cuDNN
    # pick a different algorithm depending on timing noise, which is precisely the
    # kind of run-to-run difference this contract exists to remove.
    kernels = pin_deterministic_kernels()

    cfg = yaml.safe_load(open(args.config))
    dcfg = DatasetConfig(**cfg["dataset"])
    tcfg = TrainConfig(**cfg["train"])
    if args.cutout_root:
        dcfg.cutout_root = args.cutout_root

    refuse_any_test_partition(dcfg.manifest_path)

    ds = LensDataset(
        dcfg,
        SplitConfig(split_value="val"),
        AugmentConfig(hflip=False, vflip=False, rot90=False),
    )
    # The batch size *is* the contract, so it may not be changed casually. A run
    # that departs from it is still useful for measuring invariance, but it must
    # announce that it is not the governing one.
    contract_batch_size = int(tcfg.batch_size)
    governing = True
    if args.batch_size and args.batch_size != contract_batch_size:
        if not args.allow_variable_batch:
            raise ScoringError(
                f"--batch-size {args.batch_size} departs from the primary contract "
                f"({contract_batch_size} from {args.config}). Batch shape selects "
                "cuDNN kernels and therefore moves the operating point. Pass "
                "--allow-variable-batch to proceed as a sensitivity run."
            )
        governing = False
    if args.filler_policy != "last":
        governing = False
    batch_size = args.batch_size if args.batch_size else contract_batch_size
    is_file_manifest = getattr(dcfg, "mode", None) == "file_manifest"
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate_weighted if is_file_manifest else _collate,
    )

    frame = ds.df.reset_index(drop=True)
    missing = [c for c in CARRY if c not in frame.columns]
    if missing:
        raise ScoringError(f"manifest lacks {missing}")
    split_check = data_splits.check_frames(
        {"validation_rows": frame},
        purpose="unified scoring contract: validation negatives, logit space",
        allow_test_rows=args.allow_test_rows,
        test_manifest=args.test_manifest,
    )
    if split_check["test_rows_selected"] != 0:
        raise ScoringError(
            "the validation selection intersects the test manifest"
        )

    checkpoint = Path(args.checkpoint)
    model = load_model(checkpoint, tcfg.arch, args.device)
    decoupling = assert_no_batch_coupling(model)
    logits, padding = logits_in_row_order(
        model, loader, args.device, batch_size, filler=args.filler_policy
    )
    if logits.size != len(frame):
        raise ScoringError(
            f"scored {logits.size} rows but the validation split has {len(frame)}"
        )

    out = frame[list(CARRY)].copy()
    out["logit"] = logits
    # float32 is what the previous path stored and what a reader sees; float64 is
    # what the operating point is computed from. Both are recorded so the
    # saturation is visible rather than inferred.
    out["probability_float32"] = (
        torch.sigmoid(torch.tensor(logits, dtype=torch.float32)).numpy()
    )
    out["probability_float64"] = 1.0 / (1.0 + np.exp(-logits))

    negatives_mask = out["label"].to_numpy().astype(int) == 0
    negative_logits = logits[negatives_mask]

    os.makedirs(args.out_dir, exist_ok=True)
    scores_path = Path(args.out_dir) / "validation_logits.parquet"
    out.to_parquet(scores_path, index=False)

    saturated32 = int((out.loc[negatives_mask, "probability_float32"] >= 1.0).sum())
    record = {
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "authority": (
            "docs/RASTI_REVISION_STATISTICAL_ANALYSIS_PLAN.md §14 for the threshold "
            "rule and the shared fixed-width scoring contract"
        ),
        "governing": governing,
        "contract": {
            "name": "training_faithful_logit_space",
            "inference_batch_size": batch_size,
            "contract_batch_size": contract_batch_size,
            "batch_shape_fixed": True,
            "final_batch_padding": (
                f"{padding['filler_policy']} row repeated, outputs discarded"
            ),
            **kernels,
            "operating_point_space": "raw logit",
            "operating_point_rule": (
                "smallest float32 strictly greater than s_(k+1), rule score >= t, "
                "per §14"
            ),
            "arch": tcfg.arch,
            "device": args.device,
            "num_workers": args.num_workers,
            **padding,
            "batch_decoupling": decoupling,
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_of(checkpoint),
        "manifest": dcfg.manifest_path,
        "manifest_sha256": sha256_of(Path(dcfg.manifest_path)),
        "rows_scored": int(logits.size),
        "negatives": int(negatives_mask.sum()),
        "positives": int((~negatives_mask).sum()),
        "logit_summary": {
            "max": float(negative_logits.max()),
            "min": float(negative_logits.min()),
            "distinct": int(np.unique(negative_logits).size),
            "tied_at_max": int((negative_logits == negative_logits.max()).sum()),
        },
        "float32_saturation": {
            "negatives_reaching_probability_one": saturated32,
            "note": (
                "these are distinct logits whose float32 sigmoids collide; the "
                "saturation is a property of the display precision, not of the "
                "detector, and is why the operating point is set in logit space"
            ),
        },
        "operating_points": operating_points(negative_logits, ALLOWANCES),
        "split_check": split_check,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
    }

    record_path = Path(args.out_dir) / "unified_scoring_contract.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    sums = "".join(
        f"{sha256_of(p)}  {p.name}\n"
        for p in sorted(Path(args.out_dir).glob("*"))
        if p.name != "SHA256SUMS"
    )
    (Path(args.out_dir) / "SHA256SUMS").write_text(sums)

    print(f"scored        {record['rows_scored']} rows, {record['negatives']} negatives")
    print(
        f"batches       {padding['full_batches']} full + terminal with "
        f"{padding['terminal_batch_real_rows']} real, "
        f"{padding['padded_rows_discarded']} fillers discarded"
    )
    print(f"governing     {governing}")
    print(f"logit max     {record['logit_summary']['max']}")
    print(f"distinct      {record['logit_summary']['distinct']} of {record['negatives']}")
    print(f"float32 sat   {saturated32} negatives reach probability 1.0")
    for rung in record["operating_points"]:
        if rung["allowance"] in (6, 61):
            print(
                f"  allowance {rung['allowance']:>3}  logit {rung['logit_threshold']:.10f}"
                f"  p={rung['probability_threshold_float64']:.16f}"
                f"  identified={rung['identified']}"
            )
    print(f"\nwrote {scores_path} and {record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
