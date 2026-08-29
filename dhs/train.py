from __future__ import annotations

import inspect
import json
import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import DatasetConfig, LensDataset, SplitConfig
from .model import build_model, initialization_provenance
from .preprocess_spec import PreprocessSpec
from .transforms import AugmentConfig
from .utils import normalize_outer_annulus

logger = logging.getLogger(__name__)

# Epochs written to epoch_NNN.pt. The decade grid is the pre-registered
# evaluation grid of Amendment 2, whose provenance argument rests on the grid
# having been fixed by code written before the selection question was raised.
# Epoch 19 was added later, by Amendment 3, to match the submitted model's
# Phase-1 depth; it is descriptive only and carries none of that provenance.
PERIODIC_CHECKPOINT_EPOCHS = frozenset({19}) | frozenset(range(10, 161, 10))


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    device: str = "cuda"
    early_stopping_patience: int = 10
    out_dir: str = "./checkpoints"
    mixed_precision: bool = True
    # Paper IV parity additions
    arch: str = "resnet18"  # resnet18, bottlenecked_resnet, efficientnet_v2_s
    effective_batch: int = 0  # 0 = disabled; >0 = gradient accumulation target
    lr_schedule: str = "cosine"  # cosine, step
    lr_step_epoch: int = 80  # epoch at which to halve LR (for step schedule)
    lr_gamma: float = 0.5  # LR decay factor for step schedule
    unweighted_loss: bool = False  # True = ignore sample_weight (Paper IV parity)
    pretrained: bool = False  # For EfficientNetV2-S ImageNet init
    base_ch: int = 16  # Base channel width for bottlenecked_resnet (16=~70K, 27=~195K)
    # Transfer-learning schedule (for pretrained models like EfficientNetV2-S)
    freeze_backbone_epochs: int = (
        0  # Freeze backbone for N epochs, train only classifier head
    )
    warmup_epochs: int = 0  # Linear LR warmup from lr/100 to lr over N epochs
    # Fine-tuning from a prior checkpoint (loads model weights ONLY; optimizer/scheduler start fresh)
    init_weights: str = ""  # Path to .pt checkpoint file; empty = train from scratch
    # Training-mixture configuration.
    balanced_sampling: bool = False
    sampler_positive_fraction: float = 0.5
    examples_per_epoch: int = 0
    mixture_id: str = ""
    lineage_path: str = ""
    selection_metric: str = "pr_auc"


def _collate(batch):
    """Standard collate for paired/unpaired_manifest modes (x, y)."""
    xs, ys = zip(*batch, strict=True)
    x = torch.from_numpy(np.stack(xs, axis=0)).float()
    y = torch.from_numpy(np.array(ys)).float().view(-1, 1)
    return x, y


def _collate_weighted(batch):
    """Weighted collate for file_manifest mode (x, y, weight)."""
    xs, ys, ws = zip(*batch, strict=True)
    x = torch.from_numpy(np.stack(xs, axis=0)).float()
    y = torch.from_numpy(np.array(ys)).float().view(-1, 1)
    w = torch.from_numpy(np.array(ws)).float().view(-1, 1)
    return x, y, w


def negative_only_threshold(
    negatives: np.ndarray, target_fpr: float
) -> tuple[float, int]:
    """Return the score cut fixed by the negative distribution alone.

    The cut is the smallest observed negative score that admits no more than
    ``floor(target_fpr * n_negatives)`` negatives. Positive scores take no part,
    so the operating point cannot move when the positive cohort changes, which
    is what makes the point comparable across training mixtures that share a
    validation negative set but differ in their noisy positives.

    Ties are never split: a group of equal scores is admitted whole or not at
    all. The realised false-positive count may therefore fall below the
    allowance, but it can never exceed it.

    Scores are widened to float64 before the cut is chosen, and callers must
    compare in float64 too. When the whole top tie cannot fit in the allowance
    the cut is the next representable value above the largest score, and under
    NumPy's value-based scalar casting a float64 cut compared against a float32
    score array is demoted back to float32 -- collapsing onto the tie it was
    built to exclude and admitting it, which would overspend the allowance.
    """
    if negatives.size == 0:
        raise ValueError("Threshold calibration requires validation negatives")
    allowance = int(np.floor(target_fpr * negatives.size))
    ordered = np.sort(np.asarray(negatives, dtype=np.float64))
    candidates = np.unique(ordered)
    admitted = ordered.size - np.searchsorted(ordered, candidates, side="left")
    eligible = np.flatnonzero(admitted <= allowance)
    if eligible.size:
        return float(candidates[eligible[0]]), allowance
    # The allowance cannot be met while admitting any negative at all, so the
    # cut is placed just above the largest one.
    return float(np.nextafter(candidates[-1], np.inf)), allowance


@torch.no_grad()
def evaluate(model, loader, device, has_weights=False):
    """Return prevalence-aware validation metrics and operating points."""
    model.eval()
    ys, ps = [], []
    for batch in loader:
        if has_weights:
            x, y, _ = batch  # Ignore weights during eval
        else:
            x, y = batch
        x = x.to(device)
        logits = model(x)
        p = torch.sigmoid(logits).cpu().numpy().ravel()
        ys.append(y.numpy().ravel())
        ps.append(p)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    if set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("Evaluation requires both positive and negative labels")
    # float64 so the threshold comparison below is exact; see
    # negative_only_threshold. Widening is rank-preserving, so it leaves the
    # ranking metrics untouched.
    p = p.astype(np.float64)

    def operating_point(target_fpr: float) -> dict[str, float | int]:
        threshold, allowance = negative_only_threshold(p[y == 0], target_fpr)
        predicted = p >= threshold
        positive = y == 1
        negative = ~positive
        tp = int(np.sum(predicted & positive))
        fp = int(np.sum(predicted & negative))
        tn = int(np.sum((~predicted) & negative))
        fn = int(np.sum((~predicted) & positive))
        return {
            "target_fpr": float(target_fpr),
            "threshold": threshold,
            "false_positive_allowance": allowance,
            "threshold_rule": "negative_only_order_statistic",
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "recall": float(tp / max(tp + fn, 1)),
            "precision": float(tp / max(tp + fp, 1)),
            "observed_fpr": float(fp / max(fp + tn, 1)),
        }

    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    return {
        "rows": int(len(y)),
        "positives": positives,
        "negatives": negatives,
        "prevalence": float(positives / len(y)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "all_negative_baseline": {
            "accuracy": float(negatives / len(y)),
            "recall": 0.0,
            "pr_auc_reference": float(positives / len(y)),
        },
        "operating_points": {
            "fpr_1e-3": operating_point(1e-3),
            "fpr_1e-4": operating_point(1e-4),
        },
    }


def _freeze_backbone(model, arch: str):
    """Freeze all layers except the final classifier head."""
    if arch == "efficientnet_v2_s":
        for param in model.features.parameters():
            param.requires_grad = False
        # classifier head stays trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif arch == "resnet18":
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
    elif arch == "bottlenecked_resnet":
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
    n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    n_total = sum(1 for p in model.parameters())
    print(f"Backbone frozen: {n_frozen}/{n_total} parameter groups frozen")


def _unfreeze_all(model):
    """Unfreeze all layers."""
    for param in model.parameters():
        param.requires_grad = True
    print("All layers unfrozen")


def _warmup_lr(opt, epoch: int, warmup_epochs: int, base_lr: float):
    """Linear LR warmup: ramp from base_lr/100 to base_lr over warmup_epochs."""
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return
    alpha = epoch / warmup_epochs  # 0 at epoch 0, 1 at warmup_epochs
    lr = base_lr * (0.01 + 0.99 * alpha)
    for pg in opt.param_groups:
        pg["lr"] = lr


def build_training_sampler(
    dataset: LensDataset,
    dcfg: DatasetConfig,
    tcfg: TrainConfig,
) -> tuple[WeightedRandomSampler | None, dict]:
    """Build the deterministic class-balanced sampler and summary record."""
    labels = dataset.df[dcfg.label_col].astype(int).to_numpy()
    natural_positive_fraction = float((labels == 1).mean())
    if not tcfg.balanced_sampling:
        return None, {
            "mode": "natural_prevalence_shuffle",
            "examples_per_epoch": int(len(dataset)),
            "positive_fraction": natural_positive_fraction,
        }
    if not 0.0 < tcfg.sampler_positive_fraction < 1.0:
        raise ValueError("sampler_positive_fraction must be between zero and one")
    positives = labels == 1
    negatives = labels == 0
    n_positive = int(positives.sum())
    n_negative = int(negatives.sum())
    if n_positive == 0 or n_negative == 0:
        raise ValueError("Balanced sampling requires both training classes")
    examples = (
        int(tcfg.examples_per_epoch)
        if tcfg.examples_per_epoch > 0
        else int(len(dataset))
    )
    if examples < tcfg.batch_size:
        raise ValueError("examples_per_epoch must be at least one batch")
    sample_weights = np.where(
        positives,
        tcfg.sampler_positive_fraction / n_positive,
        (1.0 - tcfg.sampler_positive_fraction) / n_negative,
    )
    generator = torch.Generator()
    generator.manual_seed(dcfg.seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=examples,
        replacement=True,
        generator=generator,
    )
    return sampler, {
        "mode": "deterministic_class_balanced",
        "examples_per_epoch": examples,
        "positive_fraction": float(tcfg.sampler_positive_fraction),
        "natural_positive_fraction": natural_positive_fraction,
        "unique_positive_rows": n_positive,
        "unique_negative_rows": n_negative,
        "seed": int(dcfg.seed),
    }


def train_one(tcfg: TrainConfig, dcfg: DatasetConfig, aug: AugmentConfig):
    os.makedirs(tcfg.out_dir, exist_ok=True)
    torch.manual_seed(dcfg.seed)
    np.random.seed(dcfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(dcfg.seed)
    if tcfg.mixture_id and tcfg.init_weights:
        raise ValueError(
            "Mixture training forbids historical lens-checkpoint initialization; "
            "use ImageNet pretrained weights with init_weights empty"
        )
    if tcfg.mixture_id and not tcfg.pretrained:
        raise ValueError("Mixture training requires ImageNet initialization")
    ds_tr = LensDataset(dcfg, SplitConfig(split_value="train"), aug)
    ds_va = LensDataset(
        dcfg,
        SplitConfig(split_value="val"),
        AugmentConfig(hflip=False, vflip=False, rot90=False),
    )

    # file_manifest always returns (x, y, w) tuples, so always use weighted collate
    # The unweighted_loss flag controls whether w is used in the loss computation
    is_file_manifest = dcfg.mode == "file_manifest"
    weighted = is_file_manifest and not tcfg.unweighted_loss
    collate_fn = _collate_weighted if is_file_manifest else _collate

    sampler, sampler_info = build_training_sampler(ds_tr, dcfg, tcfg)
    dl_tr = DataLoader(
        ds_tr,
        batch_size=tcfg.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=tcfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=tcfg.batch_size,
        shuffle=False,
        num_workers=tcfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    device = torch.device(tcfg.device if torch.cuda.is_available() else "cpu")

    # Build model using factory (supports resnet18, bottlenecked_resnet, efficientnet_v2_s)
    model = build_model(
        tcfg.arch, in_ch=3, pretrained=tcfg.pretrained, base_ch=tcfg.base_ch
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {tcfg.arch}, params: {n_params:,}")

    # Optionally initialize model weights from a prior checkpoint
    # (loads state_dict only; optimizer and scheduler start fresh with this config's LR)
    if tcfg.init_weights:
        if not os.path.isfile(tcfg.init_weights):
            raise FileNotFoundError(
                f"init_weights checkpoint not found: {tcfg.init_weights}"
            )
        print(f"Loading model weights from: {tcfg.init_weights}")
        ckpt_init = torch.load(
            tcfg.init_weights, map_location=device, weights_only=False
        )
        model.load_state_dict(ckpt_init["model"])
        src_epoch = ckpt_init.get("epoch", "?")
        src_auc = ckpt_init.get("best_auc", "?")
        print(f"  Source checkpoint: epoch={src_epoch}, best_auc={src_auc}")
        print("  Note: Only model weights loaded. Optimizer and scheduler start fresh.")
        del ckpt_init  # Free memory

    # Freeze backbone for pretrained models during initial epochs
    backbone_frozen = False
    if tcfg.freeze_backbone_epochs > 0:
        _freeze_backbone(model, tcfg.arch)
        backbone_frozen = True

    opt = torch.optim.AdamW(
        model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay
    )

    # LR schedule: cosine (default) or step (Paper IV parity)
    if tcfg.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg.epochs)
    elif tcfg.lr_schedule == "step":
        sched = torch.optim.lr_scheduler.StepLR(
            opt, step_size=tcfg.lr_step_epoch, gamma=tcfg.lr_gamma
        )
    else:
        raise ValueError(f"Unknown lr_schedule: {tcfg.lr_schedule}")

    scaler = torch.cuda.amp.GradScaler(
        enabled=(tcfg.mixed_precision and device.type == "cuda")
    )

    # Loss: reduction='none' for weighted, 'mean' for unweighted
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none" if weighted else "mean")

    # Gradient accumulation for effective batch size
    accum_steps = 1
    if tcfg.effective_batch > 0:
        if tcfg.effective_batch % tcfg.batch_size != 0:
            raise ValueError(
                f"effective_batch ({tcfg.effective_batch}) must be divisible by "
                f"batch_size ({tcfg.batch_size})"
            )
        accum_steps = tcfg.effective_batch // tcfg.batch_size
    print(
        f"Gradient accumulation: {accum_steps} steps "
        f"(micro={tcfg.batch_size}, effective={tcfg.batch_size * accum_steps})"
    )

    # Build PreprocessSpec from dataset config — saved in every checkpoint
    # so scoring scripts can auto-load the exact preprocessing used.
    _annulus_r_in = dcfg.annulus_r_in if dcfg.annulus_r_in > 0 else None
    _annulus_r_out = dcfg.annulus_r_out if dcfg.annulus_r_out > 0 else None
    preprocess_spec = PreprocessSpec(
        mode=dcfg.preprocessing,
        crop=dcfg.crop,
        crop_size=dcfg.crop_size,
        clip_range=10.0,
        annulus_r_in=_annulus_r_in,
        annulus_r_out=_annulus_r_out,
    )
    logger.info("PreprocessSpec: %s", preprocess_spec)

    if tcfg.selection_metric not in {"pr_auc", "roc_auc"}:
        raise ValueError("selection_metric must be 'pr_auc' or 'roc_auc'")
    # The annulus config uses a 0.0 sentinel meaning "fall through to
    # normalize_outer_annulus defaults", so the radii that will actually be
    # applied have to be resolved and recorded rather than read off the config.
    annulus_defaults = inspect.signature(normalize_outer_annulus).parameters
    resolved_annulus = {
        "r_in": float(
            _annulus_r_in
            if _annulus_r_in is not None
            else annulus_defaults["r_in"].default
        ),
        "r_out": float(
            _annulus_r_out
            if _annulus_r_out is not None
            else annulus_defaults["r_out"].default
        ),
    }
    micro_batches = len(dl_tr)
    train_labels = ds_tr.df[dcfg.label_col].astype(int).to_numpy()
    train_positives = int((train_labels == 1).sum())
    setup = {
        "mixture_id": tcfg.mixture_id,
        "dataset_seed": int(dcfg.seed),
        "train_rows": int(len(ds_tr)),
        "train_positives": train_positives,
        "validation_rows": int(len(ds_va)),
        "sampler": sampler_info,
        "resolved_annulus": resolved_annulus,
        "initialization": initialization_provenance(tcfg.arch, tcfg.pretrained),
        "periodic_checkpoint_epochs": sorted(
            e for e in PERIODIC_CHECKPOINT_EPOCHS if e <= tcfg.epochs
        ),
        "exposure": {
            "micro_batches_per_epoch": micro_batches,
            "gradient_accumulation_steps": accum_steps,
            "optimizer_steps_per_epoch": -(-micro_batches // accum_steps),
            "examples_per_epoch": int(sampler_info["examples_per_epoch"]),
            "positive_presentations_per_epoch": (
                train_positives
                if sampler is None
                else int(
                    sampler_info["examples_per_epoch"]
                    * sampler_info["positive_fraction"]
                )
            ),
        },
        "selection_metric": tcfg.selection_metric,
        "selection_metric_role": (
            "reported diagnostic only; governs best.pt, which is quarantined "
            "from all test-facing evaluation"
        ),
        "historical_checkpoint_initialization_used": bool(tcfg.init_weights),
        "imagenet_initialization": bool(tcfg.pretrained),
    }
    with open(os.path.join(tcfg.out_dir, "training_setup.json"), "w") as handle:
        json.dump(setup, handle, indent=2, sort_keys=True)
        handle.write("\n")

    best_metric, best_path = -1.0, None
    bad = 0

    for epoch in range(1, tcfg.epochs + 1):
        # Update dataset epoch so augmentation seed varies each epoch
        ds_tr.epoch = epoch

        # Unfreeze backbone after freeze_backbone_epochs
        if backbone_frozen and epoch > tcfg.freeze_backbone_epochs:
            _unfreeze_all(model)
            # Re-create optimizer so all params are in param_groups
            opt = torch.optim.AdamW(
                model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay
            )
            if tcfg.lr_schedule == "cosine":
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=tcfg.epochs
                )
            elif tcfg.lr_schedule == "step":
                sched = torch.optim.lr_scheduler.StepLR(
                    opt, step_size=tcfg.lr_step_epoch, gamma=tcfg.lr_gamma
                )
            # Fast-forward scheduler to current epoch
            for _ in range(epoch - 1):
                sched.step()
            backbone_frozen = False

        # Linear LR warmup (overrides scheduler LR during warmup phase)
        if tcfg.warmup_epochs > 0 and epoch <= tcfg.warmup_epochs:
            _warmup_lr(opt, epoch, tcfg.warmup_epochs, tcfg.lr)

        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0

        opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(dl_tr):
            if is_file_manifest:
                x, y, w = batch
                x, y = x.to(device), y.to(device)
                w = w.to(device) if weighted else None
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                w = None

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(x)
                if weighted and w is not None:
                    per_sample_loss = loss_fn(logits, y)
                    loss = (per_sample_loss * w).mean()
                else:
                    loss = loss_fn(logits, y)

                # Scale loss for gradient accumulation
                if accum_steps > 1:
                    loss = loss / accum_steps

            scaler.scale(loss).backward()

            # Step optimizer every accum_steps
            if (step + 1) % accum_steps == 0 or (step + 1) == len(dl_tr):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            total_loss += loss.item() * (accum_steps if accum_steps > 1 else 1)
            n_batches += 1

        sched.step()

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        validation_metrics = evaluate(
            model, dl_va, device, has_weights=is_file_manifest
        )
        selection_value = float(validation_metrics[tcfg.selection_metric])
        lr_now = opt.param_groups[0]["lr"]

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "best_metric": best_metric,
            "selection_metric": tcfg.selection_metric,
            "validation_metrics": validation_metrics,
            "sampler": sampler_info,
            "dataset": dcfg.__dict__,
            "train": tcfg.__dict__,
            "preprocess_spec": preprocess_spec.to_dict(),
        }
        # Always save last checkpoint
        torch.save(ckpt, os.path.join(tcfg.out_dir, "last.pt"))
        # Save best checkpoint
        if selection_value > best_metric:
            best_metric = selection_value
            ckpt["best_metric"] = best_metric
            best_path = os.path.join(tcfg.out_dir, "best.pt")
            torch.save(ckpt, best_path)
            bad = 0
        else:
            bad += 1
        if epoch in PERIODIC_CHECKPOINT_EPOCHS:
            periodic_path = os.path.join(tcfg.out_dir, f"epoch_{epoch:03d}.pt")
            torch.save(ckpt, periodic_path)

        epoch_record = {
            "epoch": int(epoch),
            "train_loss": float(avg_loss),
            "learning_rate": float(lr_now),
            "elapsed_seconds": float(time.time() - t0),
            "selection_metric": tcfg.selection_metric,
            "selection_value": selection_value,
            "best_selection_value": float(best_metric),
            "validation": validation_metrics,
        }
        with open(os.path.join(tcfg.out_dir, "metrics.jsonl"), "a") as metrics_handle:
            metrics_handle.write(
                json.dumps(epoch_record, sort_keys=True, allow_nan=False) + "\n"
            )
            metrics_handle.flush()
            os.fsync(metrics_handle.fileno())
        print(
            f"[epoch {epoch:03d}] loss={avg_loss:.4f} "
            f"val_roc_auc={validation_metrics['roc_auc']:.4f} "
            f"val_pr_auc={validation_metrics['pr_auc']:.4f} "
            f"best_{tcfg.selection_metric}={best_metric:.4f} "
            f"lr={lr_now:.2e} dt={time.time() - t0:.1f}s"
        )

        # Early stopping (disabled when patience <= 0, e.g. for Paper IV parity 160 epochs)
        if tcfg.early_stopping_patience > 0 and bad >= tcfg.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch} (patience={tcfg.early_stopping_patience})"
            )
            break

    return best_path, best_metric
