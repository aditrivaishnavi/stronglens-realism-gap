#!/usr/bin/env python3
"""
Evaluation script for checklist 4.5 (calibration) and 4.6 (independent validation).

Computes on test split (disjoint from train by split), and by stratum when manifest has tier:
- AUC, recall @ 0.5, precision @ 0.5
- ECE / MCE and reliability curve (calibration by stratum)
- Tier-A (independent spectroscopic anchor) metrics

Usage (on Lambda or machine with checkpoint + manifest + cutouts):
  python dhs/scripts/run_evaluation.py --config configs/resnet18_baseline_v1.yaml
  python dhs/scripts/run_evaluation.py --config configs/resnet18_baseline_v1.yaml --checkpoint /path/to/best.pt --output results/eval_v1.json

Output: JSON with all metrics + optional --summary path for review markdown.
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
from sklearn.metrics import roc_auc_score

from dhs.calibration import compute_ece, compute_mce, reliability_curve
from dhs.data import DatasetConfig, SplitConfig, LensDataset
from dhs.transforms import AugmentConfig
from dhs.model import build_resnet18


def _read_parquet(path: str) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def _collate_weighted(batch):
    xs, ys, ws = zip(*batch)
    x = torch.from_numpy(np.stack(xs, axis=0)).float()
    y = torch.from_numpy(np.array(ys)).float().view(-1, 1)
    w = torch.from_numpy(np.array(ws)).float().view(-1, 1)
    return x, y, w


def recall_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """Recall (TPR) among positives at given score threshold."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    pos = y_true == 1
    if pos.sum() == 0:
        return float("nan")
    return (y_prob[pos] >= threshold).astype(float).mean()


def precision_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> float:
    """Precision at given score threshold."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    pred_pos = y_prob >= threshold
    if pred_pos.sum() == 0:
        return float("nan")
    return (y_true[pred_pos] == 1).astype(float).mean()


@torch.no_grad()
def run_inference(model, dataset, device, batch_size: int = 256):
    """Run model on dataset; return y_true, y_prob in dataset order."""
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_weighted,
    )
    model.eval()
    ys, ps = [], []
    for x, y, _ in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.sigmoid(logits).cpu().numpy().ravel()
        ys.append(y.numpy().ravel())
        ps.append(p)
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    return y_true, y_prob


def main():
    ap = argparse.ArgumentParser(description="Evaluate trained model (4.5 calibration, 4.6 independent validation)")
    ap.add_argument("--config", required=True, help="Training config YAML (dataset + train.out_dir)")
    ap.add_argument("--checkpoint", default=None, help="Path to best.pt (default: train.out_dir/best.pt)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--output", default=None, help="Output JSON path (default: print to stdout)")
    ap.add_argument("--summary", default=None, help="Output markdown summary path for review")
    ap.add_argument("--n", type=int, default=None, help="Cap samples for quick run (default: all)")
    ap.add_argument("--n_bins", type=int, default=15, help="Bins for ECE/reliability")
    ap.add_argument("--export_predictions", default=None, help="Write CSV with cutout_path,score for bootstrap_eval")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    dcfg = DatasetConfig(**cfg["dataset"])
    out_dir = cfg["train"]["out_dir"]
    checkpoint_path = args.checkpoint or os.path.join(out_dir, "best.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(3).to(device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    no_aug = AugmentConfig(hflip=False, vflip=False, rot90=False)
    dataset = LensDataset(dcfg, SplitConfig(split_value=args.split), no_aug)
    if args.n is not None:
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(min(args.n, len(dataset))))

    y_true, y_prob = run_inference(model, dataset, device)

    # Manifest test subset for tier (same order as dataset; align length to inferred count)
    manifest_df = _read_parquet(dcfg.manifest_path)
    split_col = getattr(SplitConfig(), "split_col", "split")
    df_split = manifest_df[manifest_df[split_col] == args.split].reset_index(drop=True)
    df_split = df_split.iloc[: len(y_true)].reset_index(drop=True)
    tier_col = "tier" if "tier" in df_split.columns else None
    tiers = df_split[tier_col].values if tier_col else np.array([None] * len(df_split))

    n_bins = args.n_bins
    results = {
        "config_path": args.config,
        "checkpoint_path": checkpoint_path,
        "split": args.split,
        "n_samples": int(len(y_true)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall": {
            "auc": float(roc_auc_score(y_true, y_prob)),
            "recall_at_0.5": float(recall_at_threshold(y_true, y_prob, 0.5)),
            "precision_at_0.5": float(precision_at_threshold(y_true, y_prob, 0.5)),
            "ece": float(compute_ece(y_true, y_prob, n_bins=n_bins)),
            "mce": float(compute_mce(y_true, y_prob, n_bins=n_bins)),
            "n_bins": n_bins,
        },
        "reliability_curve": None,
        "by_stratum": {},
    }

    rc = reliability_curve(y_true, y_prob, n_bins=n_bins)
    results["reliability_curve"] = {
        "bin_edges": rc.bin_edges.tolist(),
        "bin_acc": np.nan_to_num(rc.bin_acc, nan=0.0).tolist(),
        "bin_conf": np.nan_to_num(rc.bin_conf, nan=0.0).tolist(),
        "bin_counts": rc.bin_counts.tolist(),
    }

    # Tier-A = held-out Tier-A anchors (checklist 4.6). Do not report ECE/MCE for single-class strata (misleading).
    if tier_col is not None:
        for tier_name in ["A", "B"]:
            mask = tiers == tier_name
            if mask.sum() == 0:
                continue
            yt, yp = y_true[mask], y_prob[mask]
            n_pos = (yt == 1).sum()
            n_neg = (yt == 0).sum()
            single_class = n_pos == 0 or n_neg == 0
            stratum = {
                "n": int(mask.sum()),
                "n_pos": int(n_pos),
                "n_neg": int(n_neg),
                "auc": float(roc_auc_score(yt, yp)) if n_pos > 0 and n_neg > 0 else None,
                "recall_at_0.5": float(recall_at_threshold(yt, yp, 0.5)),
                "precision_at_0.5": float(precision_at_threshold(yt, yp, 0.5)),
                "ece": None if single_class else float(compute_ece(yt, yp, n_bins=n_bins)),
                "mce": None if single_class else float(compute_mce(yt, yp, n_bins=n_bins)),
            }
            if single_class:
                stratum["ece_mce_note"] = "Not reported: stratum is single-class (ECE/MCE not meaningful)."
            results["by_stratum"][f"tier_{tier_name}"] = stratum
        # Held-out Tier-A (confident lenses in test set). Not a separate spectroscopic-search catalog.
        if "tier_A" in results["by_stratum"]:
            results["independent_validation_spectroscopic"] = results["by_stratum"]["tier_A"].copy()
            results["independent_validation_spectroscopic"]["description"] = (
                "Held-out Tier-A (confident lenses with DR10 match, test split only). "
                "Not from a separate spectroscopic-search catalog; same imaging-candidate source, held out from training."
            )

    out_json = args.output
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {out_json}")
    else:
        print(json.dumps(results, indent=2))

    if args.summary:
        write_evaluation_summary(results, args.summary)
        print(f"Wrote summary {args.summary}")

    if args.export_predictions:
        cutout_path_col = getattr(dcfg, "cutout_path_col", "cutout_path")
        pred_df = pd.DataFrame({"cutout_path": df_split[cutout_path_col].astype(str).values, "score": y_prob.astype(np.float64)})
        os.makedirs(os.path.dirname(args.export_predictions) or ".", exist_ok=True)
        pred_df.to_csv(args.export_predictions, index=False)
        print(f"Wrote {args.export_predictions} ({len(pred_df)} rows)")


def write_evaluation_summary(results: dict, path: str) -> None:
    """Write a concise markdown summary for review (checklist 4.5 / 4.6)."""
    lines = [
        "# Evaluation Summary (Checklist 4.5 & 4.6)",
        "",
        f"**Config:** `{results.get('config_path', '')}`  ",
        f"**Checkpoint:** `{results.get('checkpoint_path', '')}`  ",
        f"**Split:** {results.get('split', '')}  ",
        f"**Samples:** {results.get('n_samples', 0):,}  ",
        f"**Time (UTC):** {results.get('timestamp_utc', '')}",
        "",
        "## Overall (test set)",
        "",
        "| Metric | Value |",
        "|--------|--------|",
        f"| AUC | {results['overall']['auc']:.4f} |",
        f"| Recall @ 0.5 | {results['overall']['recall_at_0.5']:.4f} |",
        f"| Precision @ 0.5 | {results['overall']['precision_at_0.5']:.4f} |",
        f"| ECE (n_bins={results['overall']['n_bins']}) | {results['overall']['ece']:.4f} |",
        f"| MCE | {results['overall']['mce']:.4f} |",
        "",
        "## Calibration (4.5)",
        "",
        "ECE/MCE and reliability curve are in the JSON. Overall ECE is dominated by the majority class (negatives); "
        "it does not certify calibration on the positive class. See docs/EVALUATION_HONEST_AUDIT.md.",
        "",
    ]
    if results.get("by_stratum"):
        lines.append("## By stratum (tier)")
        lines.append("")
        for k, v in results["by_stratum"].items():
            if v.get("n_pos", 0) == 0 and v.get("n_neg", 0) == 0:
                continue
            lines.append(f"### {k}")
            lines.append("")
            if v.get("ece_mce_note"):
                lines.append(f"*{v['ece_mce_note']}*")
                lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|--------|")
            lines.append(f"| n | {v.get('n', 0)} |")
            lines.append(f"| n_pos | {v.get('n_pos', 0)} |")
            lines.append(f"| n_neg | {v.get('n_neg', 0)} |")
            if v.get("auc") is not None:
                lines.append(f"| AUC | {v['auc']:.4f} |")
            lines.append(f"| Recall @ 0.5 | {v.get('recall_at_0.5', float('nan')):.4f} |")
            lines.append(f"| Precision @ 0.5 | {v.get('precision_at_0.5', float('nan')):.4f} |")
            if v.get("ece") is not None:
                lines.append(f"| ECE | {v['ece']:.4f} |")
                lines.append(f"| MCE | {v['mce']:.4f} |")
            lines.append("")
    if results.get("independent_validation_spectroscopic"):
        iv = results["independent_validation_spectroscopic"]
        lines.append("## Held-out Tier-A evaluation (4.6)")
        lines.append("")
        lines.append(iv.get("description", ""))
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|--------|")
        lines.append(f"| n | {iv.get('n', 0)} |")
        lines.append(f"| n_pos | {iv.get('n_pos', 0)} |")
        if iv.get("auc") is not None:
            lines.append(f"| AUC | {iv['auc']:.4f} |")
        lines.append(f"| Recall @ 0.5 | {iv.get('recall_at_0.5', float('nan')):.4f} |")
        if iv.get("ece") is not None:
            lines.append(f"| ECE | {iv.get('ece', float('nan')):.4f} |")
        lines.append("")
    lines.append("---")
    lines.append("**Instructions for reviewer:** See docs/EVALUATION_HONEST_AUDIT.md for limitations. "
                 "Check: (1) overall ECE is low but note it is majority-class dominated; (2) Tier-A is held-out, not a separate spectroscopic catalog; (3) ECE/MCE are not reported for single-class strata.")
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
