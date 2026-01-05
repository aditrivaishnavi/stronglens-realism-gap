from __future__ import annotations
import os, time, logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from .data import LensDataset, DatasetConfig, SplitConfig
from .transforms import AugmentConfig
from .model import build_resnet18, build_model
from .preprocess_spec import PreprocessSpec

logger = logging.getLogger(__name__)

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
    arch: str = "resnet18"            # resnet18, bottlenecked_resnet, efficientnet_v2_s
    effective_batch: int = 0          # 0 = disabled; >0 = gradient accumulation target
    lr_schedule: str = "cosine"       # cosine, step
    lr_step_epoch: int = 80           # epoch at which to halve LR (for step schedule)
    lr_gamma: float = 0.5             # LR decay factor for step schedule
    unweighted_loss: bool = False     # True = ignore sample_weight (Paper IV parity)
    pretrained: bool = False          # For EfficientNetV2-S ImageNet init
    base_ch: int = 16                 # Base channel width for bottlenecked_resnet (16=~70K, 27=~195K)
    # Transfer-learning schedule (for pretrained models like EfficientNetV2-S)
    freeze_backbone_epochs: int = 0   # Freeze backbone for N epochs, train only classifier head
    warmup_epochs: int = 0            # Linear LR warmup from lr/100 to lr over N epochs
    # Fine-tuning from a prior checkpoint (loads model weights ONLY; optimizer/scheduler start fresh)
    init_weights: str = ""            # Path to .pt checkpoint file; empty = train from scratch

def _collate(batch):
    """Standard collate for paired/unpaired_manifest modes (x, y)."""
    xs, ys = zip(*batch)
    x = torch.from_numpy(np.stack(xs, axis=0)).float()
    y = torch.from_numpy(np.array(ys)).float().view(-1,1)
    return x, y


def _collate_weighted(batch):
    """Weighted collate for file_manifest mode (x, y, weight)."""
    xs, ys, ws = zip(*batch)
    x = torch.from_numpy(np.stack(xs, axis=0)).float()
    y = torch.from_numpy(np.array(ys)).float().view(-1,1)
    w = torch.from_numpy(np.array(ws)).float().view(-1,1)
    return x, y, w

@torch.no_grad()
def evaluate(model, loader, device, has_weights=False):
    """Evaluate model. has_weights=True means batch is (x, y, w) tuple."""
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
    return float(roc_auc_score(y, p))

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


def train_one(tcfg: TrainConfig, dcfg: DatasetConfig, aug: AugmentConfig):
    os.makedirs(tcfg.out_dir, exist_ok=True)
    ds_tr = LensDataset(dcfg, SplitConfig(split_value="train"), aug)
    ds_va = LensDataset(dcfg, SplitConfig(split_value="val"), AugmentConfig(hflip=False, vflip=False, rot90=False))

    # file_manifest always returns (x, y, w) tuples, so always use weighted collate
    # The unweighted_loss flag controls whether w is used in the loss computation
    is_file_manifest = dcfg.mode == "file_manifest"
    weighted = is_file_manifest and not tcfg.unweighted_loss
    collate_fn = _collate_weighted if is_file_manifest else _collate

    dl_tr = DataLoader(ds_tr, batch_size=tcfg.batch_size, shuffle=True, num_workers=tcfg.num_workers,
                       pin_memory=True, collate_fn=collate_fn, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=tcfg.batch_size, shuffle=False, num_workers=tcfg.num_workers,
                       pin_memory=True, collate_fn=collate_fn)

    device = torch.device(tcfg.device if torch.cuda.is_available() else "cpu")

    # Build model using factory (supports resnet18, bottlenecked_resnet, efficientnet_v2_s)
    model = build_model(tcfg.arch, in_ch=3, pretrained=tcfg.pretrained,
                        base_ch=tcfg.base_ch).to(device)
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
        ckpt_init = torch.load(tcfg.init_weights, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_init["model"])
        src_epoch = ckpt_init.get("epoch", "?")
        src_auc = ckpt_init.get("best_auc", "?")
        print(f"  Source checkpoint: epoch={src_epoch}, best_auc={src_auc}")
        print(f"  Note: Only model weights loaded. Optimizer and scheduler start fresh.")
        del ckpt_init  # Free memory

    # Freeze backbone for pretrained models during initial epochs
    backbone_frozen = False
    if tcfg.freeze_backbone_epochs > 0:
        _freeze_backbone(model, tcfg.arch)
        backbone_frozen = True

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    # LR schedule: cosine (default) or step (Paper IV parity)
    if tcfg.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg.epochs)
    elif tcfg.lr_schedule == "step":
        sched = torch.optim.lr_scheduler.StepLR(
            opt, step_size=tcfg.lr_step_epoch, gamma=tcfg.lr_gamma
        )
    else:
        raise ValueError(f"Unknown lr_schedule: {tcfg.lr_schedule}")

    scaler = torch.cuda.amp.GradScaler(enabled=(tcfg.mixed_precision and device.type == "cuda"))

    # Loss: reduction='none' for weighted, 'mean' for unweighted
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none' if weighted else 'mean')

    # Gradient accumulation for effective batch size
    accum_steps = 1
    if tcfg.effective_batch > 0:
        if tcfg.effective_batch % tcfg.batch_size != 0:
            raise ValueError(
                f"effective_batch ({tcfg.effective_batch}) must be divisible by "
                f"batch_size ({tcfg.batch_size})"
            )
        accum_steps = tcfg.effective_batch // tcfg.batch_size
    print(f"Gradient accumulation: {accum_steps} steps "
          f"(micro={tcfg.batch_size}, effective={tcfg.batch_size * accum_steps})")

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

    best_auc, best_path = -1.0, None
    bad = 0

    for epoch in range(1, tcfg.epochs + 1):
        # Update dataset epoch so augmentation seed varies each epoch
        ds_tr.epoch = epoch

        # Unfreeze backbone after freeze_backbone_epochs
        if backbone_frozen and epoch > tcfg.freeze_backbone_epochs:
            _unfreeze_all(model)
            # Re-create optimizer so all params are in param_groups
            opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
            if tcfg.lr_schedule == "cosine":
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg.epochs)
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
        auc = evaluate(model, dl_va, device, has_weights=is_file_manifest)
        lr_now = opt.param_groups[0]["lr"]

        ckpt = {
            "epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "best_auc": best_auc,
            "dataset": dcfg.__dict__, "train": tcfg.__dict__,
            "preprocess_spec": preprocess_spec.to_dict(),
        }
        # Always save last checkpoint
        torch.save(ckpt, os.path.join(tcfg.out_dir, "last.pt"))
        # Save best checkpoint
        if auc > best_auc:
            best_auc = auc
            ckpt["best_auc"] = best_auc  # update with new best
            best_path = os.path.join(tcfg.out_dir, "best.pt")
            torch.save(ckpt, best_path)
            bad = 0
        else:
            bad += 1
        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            periodic_path = os.path.join(tcfg.out_dir, f"epoch_{epoch:03d}.pt")
            torch.save(ckpt, periodic_path)

        print(f"[epoch {epoch:03d}] loss={avg_loss:.4f} val_auc={auc:.4f} "
              f"best={best_auc:.4f} lr={lr_now:.2e} dt={time.time()-t0:.1f}s")

        # Early stopping (disabled when patience <= 0, e.g. for Paper IV parity 160 epochs)
        if tcfg.early_stopping_patience > 0 and bad >= tcfg.early_stopping_patience:
            print(f"Early stopping at epoch {epoch} (patience={tcfg.early_stopping_patience})")
            break

    return best_path, best_auc
