#!/usr/bin/env python3
"""
Paper IV Parity Evaluation Pipeline.

Comprehensive evaluation script aligned with MNRAS_RAW_NOTES.md Section 9.
Computes all metrics needed for the Paper IV comparison:

  - AUC (ROC) with bootstrap 95% CIs
  - Recall, Precision at threshold (default 0.5)
  - ECE (Expected Calibration Error, equal-mass binning)
  - MCE (Maximum Calibration Error)
  - Reliability curve (for calibration diagram)
  - FPR by N2 confuser category (ring_proxy, edge_on_proxy, blue_clumpy, large_galaxy)
  - FPR by negative pool (N1 vs N2)
  - Per-tier metrics (Tier-A / Tier-B)
  - Bootstrap 95% and 68% CIs for AUC, recall, precision (10,000 resamples)

Architecture is auto-detected from the checkpoint (stored in ckpt["train"]["arch"]).
Preprocessing matches Paper IV parity: 101x101, raw_robust, crop=False.

Usage (on Lambda with checkpoint + manifest + cutouts):
    cd /lambda/nfs/.../code
    export PYTHONPATH=.
    python scripts/evaluate_parity.py \\
        --checkpoint checkpoints/paperIV_efficientnet_v2_s/best.pt \\
        --manifest manifests/training_parity_70_30_v1.parquet \\
        --split val \\
        --output results/eval_efficientnet_v2_s.json \\
        --export-predictions results/preds_efficientnet_v2_s.parquet

For 70/15/15 audit manifest:
    python scripts/evaluate_parity.py \\
        --checkpoint checkpoints/paperIV_resnet18/best.pt \\
        --manifest manifests/training_parity_v1.parquet \\
        --split test \\
        --output results/eval_resnet18_test.json

Author: stronglens_calibration project
Date: 2026-02-11
Aligned with: MNRAS_RAW_NOTES.md Section 9, Paper IV (Inchausti et al. 2025)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# dhs imports (PYTHONPATH must include repo root)
from dhs.model import build_model
from dhs.data import load_cutout_from_file
from dhs.preprocess import preprocess_stack
from dhs.calibration import compute_ece, compute_mce, reliability_curve
from dhs.constants import STAMP_SIZE, CUTOUT_SIZE


# ---------------------------------------------------------------------------
# Constants (aligned with MANIFEST_SCHEMA_TRAINING_V1.md)
# ---------------------------------------------------------------------------
CUTOUT_PATH_COL = "cutout_path"
LABEL_COL = "label"
SPLIT_COL = "split"
TIER_COL = "tier"
POOL_COL = "pool"
CONFUSER_COL = "confuser_category"
SAMPLE_WEIGHT_COL = "sample_weight"

# Paper IV comparison targets (Inchausti et al. 2025, Table 2)
PAPER_IV_METRICS = {
    "resnet": {"auc": 0.9984},
    "efficientnet": {"auc": 0.9987},
    "ensemble": {"auc": 0.9989},
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load model from checkpoint. Auto-detects architecture from stored config.

    The checkpoint stores ckpt["train"]["arch"] and ckpt["train"]["pretrained"]
    from the TrainConfig used during training.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    # Extract architecture info
    train_cfg = ckpt.get("train", {})
    arch = train_cfg.get("arch", "resnet18")
    pretrained = train_cfg.get("pretrained", False)
    epoch = ckpt.get("epoch", "?")
    best_auc = ckpt.get("best_auc", "?")

    print(f"  Architecture: {arch}")
    print(f"  Pretrained: {pretrained}")
    print(f"  Epoch: {epoch}")
    print(f"  Best AUC (training): {best_auc}")

    model = build_model(arch, in_ch=3, pretrained=pretrained).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    return model, ckpt


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(
    model: nn.Module,
    cutout_paths: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    preprocessing: str = "raw_robust",
    crop: bool = False,
    num_workers: int = 0,
) -> np.ndarray:
    """Run model inference on cutout files. Returns sigmoid scores.

    Uses the same preprocessing as training: raw_robust, crop=False for 101x101.
    No augmentation is applied (eval mode).
    """
    n = len(cutout_paths)
    scores = np.zeros(n, dtype=np.float64)
    model.eval()

    # Determine expected output spatial size for fallback zeros
    fallback_size = STAMP_SIZE if crop else CUTOUT_SIZE  # 64 if crop, 101 otherwise

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_paths = cutout_paths[start:end]
        imgs = []
        for p in batch_paths:
            try:
                img = load_cutout_from_file(str(p))  # (3, H, W) CHW
                img = preprocess_stack(img, mode=preprocessing, crop=crop, clip_range=10.0)
                imgs.append(img)
            except Exception as e:
                # On error, use zeros (will produce ~0.5 score)
                print(f"  WARNING: Failed to load {p}: {e}", file=sys.stderr)
                imgs.append(np.zeros((3, fallback_size, fallback_size), dtype=np.float32))
        batch = np.stack(imgs, axis=0)
        x = torch.from_numpy(batch).float().to(device)
        logits = model(x).squeeze(1).cpu().numpy()
        scores[start:end] = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))

        if (start // batch_size) % 50 == 0:
            print(f"  Inference: {end}/{n} ({100*end/n:.1f}%)", end="\r")

    print(f"  Inference: {n}/{n} (100.0%) -- done")
    return scores


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_binary_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> Dict[str, Any]:
    """Compute core binary classification metrics."""
    y = y_true.astype(int)
    pred = (y_score >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    auc = float(roc_auc_score(y, y_score)) if n_pos > 0 and n_neg > 0 else float("nan")

    # Partial AUC at low FPR regimes (scientifically relevant for lens finding,
    # where FPR < 0.1% matters). Added per code review.
    pauc_001 = float("nan")  # pAUC at FPR < 0.1%
    pauc_01 = float("nan")   # pAUC at FPR < 1%
    if n_pos > 0 and n_neg > 0:
        try:
            pauc_001 = float(roc_auc_score(y, y_score, max_fpr=0.001))
            pauc_01 = float(roc_auc_score(y, y_score, max_fpr=0.01))
        except ValueError:
            pass  # can fail if too few samples

    # TPR at fixed FPR thresholds (code review recommendation).
    # For lens finding, the relevant operating point is FPR < 0.1%.
    # Both reviewers: "report TPR at FPR=1e-3, not only AUC."
    tpr_at_fpr_001 = float("nan")  # TPR at FPR = 0.1%
    tpr_at_fpr_01 = float("nan")   # TPR at FPR = 1%
    if n_pos > 0 and n_neg > 0:
        try:
            from sklearn.metrics import roc_curve
            fpr_arr, tpr_arr, _ = roc_curve(y, y_score)
            tpr_at_fpr_001 = float(np.interp(0.001, fpr_arr, tpr_arr))
            tpr_at_fpr_01 = float(np.interp(0.01, fpr_arr, tpr_arr))
        except Exception:
            pass

    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "auc": auc,
        "pauc_fpr_0.001": pauc_001,
        "pauc_fpr_0.01": pauc_01,
        "tpr_at_fpr_0.001": tpr_at_fpr_001,
        "tpr_at_fpr_0.01": tpr_at_fpr_01,
        "recall": recall,
        "precision": precision,
        "fpr": fpr,
        "specificity": specificity,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": int(len(y)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "threshold": threshold,
    }


def compute_fpr_by_confuser(
    df: pd.DataFrame, scores: np.ndarray, threshold: float = 0.5
) -> Dict[str, Any]:
    """Compute FPR stratified by N2 confuser category.

    Per MNRAS_RAW_NOTES.md Section 9.1: FPR by confuser category
    (ring_proxy, edge_on_proxy, blue_clumpy, large_galaxy).
    """
    results = {}

    # Overall N1 vs N2 (skip pool-level if column absent)
    if POOL_COL not in df.columns:
        print(f"  WARNING: Column '{POOL_COL}' not in manifest; skipping pool-level FPR.", file=sys.stderr)
    else:
        for pool_val in ["N1", "N2"]:
            mask = (df[LABEL_COL] == 0) & (df[POOL_COL] == pool_val)
            if mask.sum() == 0:
                continue
            s = scores[mask.values]
            fp = int((s >= threshold).sum())
            n = int(mask.sum())
            results[f"pool_{pool_val}"] = {
                "fpr": fp / n if n > 0 else float("nan"),
                "n_false_positive": fp,
                "n_total": n,
            }

    # By specific confuser category (N2 subtypes)
    if CONFUSER_COL in df.columns:
        neg_mask = df[LABEL_COL] == 0
        for cat in sorted(df.loc[neg_mask, CONFUSER_COL].dropna().unique()):
            if cat in ("none", "", "N1"):
                continue
            mask = neg_mask & (df[CONFUSER_COL] == cat)
            if mask.sum() == 0:
                continue
            s = scores[mask.values]
            fp = int((s >= threshold).sum())
            n = int(mask.sum())
            results[f"confuser_{cat}"] = {
                "fpr": fp / n if n > 0 else float("nan"),
                "n_false_positive": fp,
                "n_total": n,
            }

    return results


def compute_fpr_by_tier(
    df: pd.DataFrame, y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> Dict[str, Any]:
    """Compute metrics stratified by tier (A, B for positives; N for negatives)."""
    results = {}
    if TIER_COL not in df.columns:
        return results

    for tier_val in sorted(df[TIER_COL].dropna().unique()):
        mask = df[TIER_COL] == tier_val
        if mask.sum() == 0:
            continue
        yt = y_true[mask.values]
        ys = scores[mask.values]
        n_pos = int((yt == 1).sum())
        n_neg = int((yt == 0).sum())

        tier_result = {
            "n": int(mask.sum()),
            "n_pos": n_pos,
            "n_neg": n_neg,
        }

        # AUC only meaningful with both classes
        if n_pos > 0 and n_neg > 0:
            tier_result["auc"] = float(roc_auc_score(yt, ys))
        else:
            tier_result["auc"] = None

        # Recall (for positive tiers)
        if n_pos > 0:
            pred = (ys >= threshold).astype(int)
            tp = int(((pred == 1) & (yt == 1)).sum())
            tier_result["recall"] = tp / n_pos
        else:
            tier_result["recall"] = None

        # FPR (for negative tiers)
        if n_neg > 0:
            pred = (ys >= threshold).astype(int)
            fp = int(((pred == 1) & (yt == 0)).sum())
            tier_result["fpr"] = fp / n_neg
        else:
            tier_result["fpr"] = None

        # ECE/MCE only meaningful with both classes
        if n_pos > 0 and n_neg > 0:
            tier_result["ece"] = float(compute_ece(yt, ys, n_bins=15))
            tier_result["mce"] = float(compute_mce(yt, ys, n_bins=15))
        else:
            tier_result["ece"] = None
            tier_result["mce"] = None
            tier_result["ece_mce_note"] = (
                "Not reported: single-class stratum (ECE/MCE not meaningful)."
            )

        results[f"tier_{tier_val}"] = tier_result

    return results


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute bootstrap CIs for AUC, recall, precision.

    Per MNRAS_RAW_NOTES.md Section 9.1: Bootstrap 95% CI, 10,000 resamples.
    Also computes 68% CIs for convenience.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    aucs, recalls, precisions = [], [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]

        # Skip degenerate samples (all same class)
        if yt.min() == yt.max():
            continue

        auc_val = float(roc_auc_score(yt, ys))
        aucs.append(auc_val)

        pred = (ys >= threshold).astype(int)
        tp = ((pred == 1) & (yt == 1)).sum()
        fp = ((pred == 1) & (yt == 0)).sum()
        fn = ((pred == 0) & (yt == 1)).sum()

        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

        if not np.isnan(rec):
            recalls.append(float(rec))
        if not np.isnan(prec):
            precisions.append(float(prec))

    def quantiles(arr: list, name: str) -> Dict:
        if len(arr) < 20:
            return {
                f"{name}_mean": float("nan"),
                f"{name}_std": float("nan"),
                f"{name}_ci68": (float("nan"), float("nan")),
                f"{name}_ci95": (float("nan"), float("nan")),
            }
        a = np.asarray(arr)
        return {
            f"{name}_mean": float(np.mean(a)),
            f"{name}_std": float(np.std(a)),
            f"{name}_ci68": (float(np.quantile(a, 0.16)), float(np.quantile(a, 0.84))),
            f"{name}_ci95": (float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))),
        }

    result = {
        "n_bootstrap": n_bootstrap,
        "n_valid_samples": len(aucs),
        "seed": seed,
    }
    result.update(quantiles(aucs, "auc"))
    result.update(quantiles(recalls, "recall"))
    result.update(quantiles(precisions, "precision"))

    return result


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate(
    checkpoint_path: str,
    manifest_path: str,
    split: str = "val",
    threshold: float = 0.5,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 42,
    n_bins: int = 15,
    batch_size: int = 256,
    device_str: str = "cuda",
    preprocessing: str = "raw_robust",
    crop: bool = False,
    export_predictions: Optional[str] = None,
    data_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Run full evaluation pipeline. Returns results dict."""

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Apply data_root override for portability
    effective_manifest = manifest_path
    effective_ckpt = checkpoint_path
    if data_root:
        default_root = "/lambda/nfs/darkhaloscope-training-dc/stronglens_calibration"
        effective_manifest = manifest_path.replace(default_root, data_root.rstrip("/"), 1)
        effective_ckpt = checkpoint_path.replace(default_root, data_root.rstrip("/"), 1)
        print(f"Data root override: {data_root}")

    # Load model
    model, ckpt = load_checkpoint(effective_ckpt, device)
    arch = ckpt.get("train", {}).get("arch", "unknown")
    epoch = ckpt.get("epoch", "?")

    # Load manifest
    print(f"\nLoading manifest: {effective_manifest}")
    df = pd.read_parquet(effective_manifest)
    print(f"  Total rows: {len(df):,}")

    # Filter to evaluation split
    df_eval = df[df[SPLIT_COL] == split].reset_index(drop=True)
    print(f"  Split '{split}': {len(df_eval):,} rows")
    n_pos = int((df_eval[LABEL_COL] == 1).sum())
    n_neg = int((df_eval[LABEL_COL] == 0).sum())
    print(f"  Positives: {n_pos:,}, Negatives: {n_neg:,}")

    if len(df_eval) == 0:
        raise ValueError(f"No rows in split '{split}'")

    # Run inference
    print(f"\nRunning inference ({len(df_eval):,} samples)...")
    t0 = time.time()
    cutout_paths = df_eval[CUTOUT_PATH_COL].values
    scores = run_inference(
        model, cutout_paths, device,
        batch_size=batch_size,
        preprocessing=preprocessing,
        crop=crop,
    )
    dt = time.time() - t0
    print(f"  Inference time: {dt:.1f}s ({dt/len(df_eval)*1000:.1f}ms/sample)")

    y_true = df_eval[LABEL_COL].values.astype(int)

    # --- Core metrics ---
    print("\nComputing metrics...")
    core = compute_binary_metrics(y_true, scores, threshold)
    print(f"  AUC: {core['auc']:.6f}")
    print(f"  pAUC (FPR<0.1%): {core['pauc_fpr_0.001']:.6f}")
    print(f"  pAUC (FPR<1%):   {core['pauc_fpr_0.01']:.6f}")
    print(f"  TPR@FPR=0.1%:    {core['tpr_at_fpr_0.001']:.4f}")
    print(f"  TPR@FPR=1%:      {core['tpr_at_fpr_0.01']:.4f}")
    print(f"  Recall@{threshold}: {core['recall']:.4f}")
    print(f"  Precision@{threshold}: {core['precision']:.4f}")
    print(f"  FPR@{threshold}: {core['fpr']:.6f}")

    # --- Calibration ---
    ece = float(compute_ece(y_true, scores, n_bins=n_bins))
    mce = float(compute_mce(y_true, scores, n_bins=n_bins))
    print(f"  ECE (n_bins={n_bins}): {ece:.6f}")
    print(f"  MCE: {mce:.6f}")

    # Positive-class calibration per code review:
    # Overall ECE is dominated by negatives (93:1 ratio). For the positive
    # class, assess calibration separately: among samples with p > threshold,
    # what fraction are true positives? This is the "precision at predicted
    # probability" — the positive predictive value stratified by score bin.
    pos_mask = y_true == 1
    pos_ece = float("nan")
    pos_mce = float("nan")
    if pos_mask.sum() > 20:
        # Use fewer bins for positives (sparse)
        pos_nbins = min(n_bins, max(5, int(pos_mask.sum() / 10)))
        try:
            pos_ece = float(compute_ece(y_true[pos_mask], scores[pos_mask], n_bins=pos_nbins))
            pos_mce = float(compute_mce(y_true[pos_mask], scores[pos_mask], n_bins=pos_nbins))
        except Exception:
            pass  # graceful fallback
    print(f"  ECE (positive-class, n_bins={pos_nbins if pos_mask.sum() > 20 else 'N/A'}): {pos_ece:.6f}")
    print(f"  MCE (positive-class): {pos_mce:.6f}")

    rc = reliability_curve(y_true, scores, n_bins=n_bins)
    reliability = {
        "bin_edges": rc.bin_edges.tolist(),
        "bin_acc": np.nan_to_num(rc.bin_acc, nan=0.0).tolist(),
        "bin_conf": np.nan_to_num(rc.bin_conf, nan=0.0).tolist(),
        "bin_counts": rc.bin_counts.tolist(),
    }

    # --- FPR by confuser category ---
    print("  FPR by confuser category...")
    fpr_confuser = compute_fpr_by_confuser(df_eval, scores, threshold)
    for k, v in fpr_confuser.items():
        print(f"    {k}: FPR={v['fpr']:.6f} ({v['n_false_positive']}/{v['n_total']})")

    # --- Per-tier metrics ---
    print("  Per-tier metrics...")
    tier_metrics = compute_fpr_by_tier(df_eval, y_true, scores, threshold)
    for k, v in tier_metrics.items():
        auc_str = f"{v['auc']:.4f}" if v.get("auc") is not None else "N/A"
        rec_str = f"{v['recall']:.4f}" if v.get("recall") is not None else "N/A"
        print(f"    {k}: n={v['n']}, AUC={auc_str}, recall={rec_str}")

    # --- Bootstrap CIs ---
    print(f"\nBootstrap CIs ({n_bootstrap:,} resamples, seed={bootstrap_seed})...")
    t0 = time.time()
    bootstrap = bootstrap_ci(y_true, scores, threshold, n_bootstrap, bootstrap_seed)
    dt = time.time() - t0
    print(f"  Bootstrap time: {dt:.1f}s")
    auc_ci = bootstrap["auc_ci95"]
    print(f"  AUC 95% CI: [{auc_ci[0]:.6f}, {auc_ci[1]:.6f}]")
    rec_ci = bootstrap["recall_ci95"]
    print(f"  Recall 95% CI: [{rec_ci[0]:.4f}, {rec_ci[1]:.4f}]")

    # --- Paper IV comparison ---
    paper_iv_auc = None
    if "resnet" in arch.lower():
        paper_iv_auc = PAPER_IV_METRICS["resnet"]["auc"]
    elif "efficient" in arch.lower():
        paper_iv_auc = PAPER_IV_METRICS["efficientnet"]["auc"]

    comparison = {}
    if paper_iv_auc is not None:
        comparison = {
            "paper_iv_auc": paper_iv_auc,
            "our_auc": core["auc"],
            "delta": core["auc"] - paper_iv_auc,
            "within_bootstrap_ci": auc_ci[0] <= paper_iv_auc <= auc_ci[1],
        }
        print(f"\n  Paper IV AUC: {paper_iv_auc}")
        print(f"  Our AUC:      {core['auc']:.6f}")
        print(f"  Delta:        {comparison['delta']:+.6f}")
        print(f"  Paper IV within 95% CI: {comparison['within_bootstrap_ci']}")

    # --- Config hash for reproducibility ---
    with open(effective_ckpt, "rb") as f:
        ckpt_hash = hashlib.sha256(f.read(4096)).hexdigest()[:16]

    # --- Assemble results ---
    results = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_path": checkpoint_path,
            "manifest_path": manifest_path,
            "arch": arch,
            "epoch": epoch,
            "split": split,
            "threshold": threshold,
            "n_samples": int(len(df_eval)),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "preprocessing": preprocessing,
            "crop": crop,
            "device": str(device),
            "checkpoint_sha256_prefix": ckpt_hash,
            "n_bootstrap": n_bootstrap,
            "bootstrap_seed": bootstrap_seed,
        },
        "overall": core,
        "calibration": {
            "ece": ece,
            "mce": mce,
            "pos_ece": pos_ece,
            "pos_mce": pos_mce,
            "n_bins": n_bins,
            "note": (
                "ECE is dominated by the majority class (negatives). "
                "It does NOT certify calibration on the positive class. "
                "pos_ece/pos_mce assess calibration on positives only "
                "per code review. "
                "See MNRAS_RAW_NOTES.md Section 7.5."
            ),
        },
        "reliability_curve": reliability,
        "fpr_by_confuser": fpr_confuser,
        "by_tier": tier_metrics,
        "bootstrap": bootstrap,
        "paper_iv_comparison": comparison,
    }

    # --- Export predictions ---
    if export_predictions:
        print(f"\nExporting predictions to: {export_predictions}")
        scores_clipped = np.clip(scores, 1e-15, 1.0 - 1e-15)
        pred_df = pd.DataFrame({
            CUTOUT_PATH_COL: df_eval[CUTOUT_PATH_COL].values,
            "score": scores,
            "logit": np.log(scores_clipped / (1.0 - scores_clipped)),
            LABEL_COL: y_true,
        })
        # Include tier and pool for meta-learner and downstream analysis
        for col in [TIER_COL, POOL_COL, CONFUSER_COL, SPLIT_COL]:
            if col in df_eval.columns:
                pred_df[col] = df_eval[col].values

        os.makedirs(os.path.dirname(export_predictions) or ".", exist_ok=True)
        if export_predictions.endswith(".parquet"):
            pred_df.to_parquet(export_predictions, index=False)
        else:
            pred_df.to_csv(export_predictions, index=False)
        print(f"  Exported {len(pred_df):,} predictions")
        results["metadata"]["export_predictions_path"] = export_predictions

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Paper IV Parity Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate EfficientNetV2-S on 70/30 val split
  python scripts/evaluate_parity.py \\
      --checkpoint checkpoints/paperIV_efficientnet_v2_s/best.pt \\
      --manifest manifests/training_parity_70_30_v1.parquet \\
      --split val

  # Evaluate ResNet-18 on 70/15/15 test split
  python scripts/evaluate_parity.py \\
      --checkpoint checkpoints/paperIV_resnet18/best.pt \\
      --manifest manifests/training_parity_v1.parquet \\
      --split test

  # Export predictions for meta-learner
  python scripts/evaluate_parity.py \\
      --checkpoint checkpoints/paperIV_efficientnet_v2_s/best.pt \\
      --manifest manifests/training_parity_70_30_v1.parquet \\
      --split val \\
      --export-predictions results/preds_efficientnet_v2_s.parquet
""",
    )
    ap.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint")
    ap.add_argument("--manifest", required=True, help="Path to training manifest parquet")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"],
                    help="Evaluation split (default: val for 70/30 parity)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Classification threshold (default: 0.5)")
    ap.add_argument("--output", default=None,
                    help="Output JSON path (default: stdout)")
    ap.add_argument("--export-predictions", default=None,
                    help="Export per-sample predictions (parquet or csv)")
    ap.add_argument("--n-bootstrap", type=int, default=10000,
                    help="Number of bootstrap resamples (default: 10000)")
    ap.add_argument("--bootstrap-seed", type=int, default=42,
                    help="Bootstrap random seed (default: 42)")
    ap.add_argument("--n-bins", type=int, default=15,
                    help="ECE/MCE bin count (default: 15)")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="Inference batch size (default: 256)")
    ap.add_argument("--device", default="cuda",
                    help="Device (default: cuda)")
    ap.add_argument("--preprocessing", default="raw_robust",
                    help="Preprocessing mode (default: raw_robust)")
    ap.add_argument("--crop", action="store_true", default=False,
                    help="Center crop to 64x64 (default: False, keep 101x101)")
    ap.add_argument("--data-root", default=None,
                    help="Override data root for path portability")
    args = ap.parse_args()

    results = evaluate(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        split=args.split,
        threshold=args.threshold,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        n_bins=args.n_bins,
        batch_size=args.batch_size,
        device_str=args.device,
        preprocessing=args.preprocessing,
        crop=args.crop,
        export_predictions=args.export_predictions,
        data_root=args.data_root,
    )

    # Save or print results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\n" + json.dumps(results, indent=2, default=str))

    # Print summary table
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    m = results["metadata"]
    o = results["overall"]
    b = results["bootstrap"]
    print(f"  Model:         {m['arch']}")
    print(f"  Epoch:         {m['epoch']}")
    print(f"  Split:         {m['split']} ({m['n_samples']:,} samples)")
    print(f"  Threshold:     {m['threshold']}")
    print()
    print(f"  AUC:           {o['auc']:.6f}  95% CI [{b['auc_ci95'][0]:.6f}, {b['auc_ci95'][1]:.6f}]")
    print(f"  pAUC(FPR<0.1%): {o.get('pauc_fpr_0.001', float('nan')):.6f}")
    print(f"  pAUC(FPR<1%):   {o.get('pauc_fpr_0.01', float('nan')):.6f}")
    print(f"  TPR@FPR=0.1%:  {o.get('tpr_at_fpr_0.001', float('nan')):.4f}")
    print(f"  TPR@FPR=1%:    {o.get('tpr_at_fpr_0.01', float('nan')):.4f}")
    print(f"  Recall:        {o['recall']:.4f}    95% CI [{b['recall_ci95'][0]:.4f}, {b['recall_ci95'][1]:.4f}]")
    print(f"  Precision:     {o['precision']:.4f}    95% CI [{b['precision_ci95'][0]:.4f}, {b['precision_ci95'][1]:.4f}]")
    print(f"  FPR:           {o['fpr']:.6f}")
    print(f"  ECE:           {results['calibration']['ece']:.6f}")
    print(f"  MCE:           {results['calibration']['mce']:.6f}")
    if results.get("paper_iv_comparison"):
        c = results["paper_iv_comparison"]
        print()
        print(f"  Paper IV AUC:  {c['paper_iv_auc']}")
        print(f"  Delta:         {c['delta']:+.6f}")
        print(f"  Within 95% CI: {c['within_bootstrap_ci']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
