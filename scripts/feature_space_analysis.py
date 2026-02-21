#!/usr/bin/env python3
"""
Feature Space Analysis: Compare CNN embeddings of real lenses vs injections.

SCIENTIFIC MOTIVATION
====================
We analyze whether the CNN's penultimate-layer
embeddings can distinguish real lenses from synthetic injections. If a simple
linear probe achieves AUC near 1.0 for "real vs injection", this proves the
CNN encodes an "injection-ness" direction and identifies which feature
dimensions differ.

This script:
  1. Extracts EfficientNetV2-S penultimate-layer (1280-dim) embeddings for:
     (a) Real Tier-A lenses from the val split
     (b) Bright injections at low beta_frac (dramatic arcs)
     (c) Bright injections at high beta_frac (non-arcs)
     (d) Val negatives (non-lenses)
  2. Trains a logistic regression linear probe "real vs injection"
  3. Computes the probe AUC (high AUC = CNN can tell them apart)
  4. Saves embeddings for downstream UMAP/t-SNE visualization

Usage:
    cd stronglens-realism-gap
    export PYTHONPATH=.

    python scripts/feature_space_analysis.py \\
        --checkpoint checkpoints/best.pt \\
        --manifest manifests/training_parity_70_30_v1.parquet \\
        --out-dir results/feature_space \\
        --n-samples 500

Date: 2026-02-13
References:
  - linear probe "real vs injection"
  - Frechet distance in feature space
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dhs.model import build_model
from dhs.preprocess import preprocess_stack
from dhs.scoring_utils import load_model_and_spec
from dhs.injection_engine import (
    sample_lens_params,
    sample_source_params,
    inject_sis_shear,
)


PIXEL_SCALE = 0.262
TIER_COL = "tier"


class EmbeddingExtractor(nn.Module):
    """Wraps an EfficientNetV2-S model to extract penultimate embeddings.

    Also supports multi-layer extraction for per-layer Fréchet distance
    analysis ("compute per-layer mean and covariance, use
    Fréchet distance to quantify distribution gap per layer").
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # EfficientNetV2-S structure: features -> avgpool -> classifier
        # We want the output of avgpool (1280-dim)
        self._embedding = None
        self._hook = None
        self._layer_embeddings: Dict[str, np.ndarray] = {}
        self._layer_hooks: list = []

    def _register_hook(self) -> None:
        """Register a forward hook on avgpool to capture embeddings."""
        if hasattr(self.model, "avgpool"):
            target = self.model.avgpool
        else:
            raise ValueError("Model does not have 'avgpool' attribute. "
                             "Feature extraction not supported for this architecture.")

        def hook_fn(module, input, output):
            self._embedding = output.flatten(1)

        self._hook = target.register_forward_hook(hook_fn)

    def register_layer_hooks(self) -> List[str]:
        """Register hooks on intermediate feature blocks for per-layer analysis.

        For EfficientNetV2-S, features[0] through features[7] are the 8
        main blocks. We hook the output of each block to capture intermediate
        representations at different spatial scales.

        Returns list of layer names that were hooked.
        """
        layer_names = []
        if not hasattr(self.model, "features"):
            return layer_names

        for i, block in enumerate(self.model.features):
            name = f"features_{i}"
            layer_names.append(name)

            def make_hook(layer_name):
                def hook_fn(module, input, output):
                    # Global average pool the spatial dims to get a fixed-size vector
                    if output.dim() == 4:
                        pooled = output.mean(dim=[2, 3])  # (B, C)
                    else:
                        pooled = output.flatten(1)
                    self._layer_embeddings[layer_name] = pooled.cpu().numpy()
                return hook_fn

            h = block.register_forward_hook(make_hook(name))
            self._layer_hooks.append(h)

        return layer_names

    def extract(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Extract embeddings and scores for a batch.

        Returns (embeddings, scores) as numpy arrays.
        """
        if self._hook is None:
            self._register_hook()

        self._layer_embeddings = {}  # clear per-forward

        with torch.no_grad():
            logits = self.model(x).squeeze(-1)
            scores = torch.sigmoid(logits).cpu().numpy()
            embeddings = self._embedding.cpu().numpy()

        return embeddings, scores

    def get_layer_embeddings(self) -> Dict[str, np.ndarray]:
        """Return the most recently captured per-layer embeddings."""
        return dict(self._layer_embeddings)


def extract_embeddings_from_paths(
    extractor: EmbeddingExtractor,
    paths: List[str],
    device: torch.device,
    pp_kwargs: dict,
    batch_size: int = 64,
    collect_layers: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """Extract embeddings from cutout file paths.

    If collect_layers is True, also returns per-layer embeddings dict.
    Returns (penultimate_emb, scores, layer_embs_or_None).
    """
    all_emb = []
    all_scores = []
    all_layer_embs: Dict[str, list] = {}
    failed_paths: List[str] = []

    for start in range(0, len(paths), batch_size):
        end = min(start + batch_size, len(paths))
        batch_paths = paths[start:end]
        imgs = []
        valid_mask = []  # track which paths loaded successfully
        for p in batch_paths:
            try:
                with np.load(p) as z:
                    hwc = z["cutout"].astype(np.float32)
                chw = np.transpose(hwc, (2, 0, 1))
                proc = preprocess_stack(chw, **pp_kwargs)
                imgs.append(proc)
                valid_mask.append(True)
            except Exception as exc:
                failed_paths.append(p)
                print(f"  WARNING: failed to load {p}: {exc}")
                # Still need a placeholder to keep batch alignment for extraction,
                # but we will drop this sample from the output arrays.
                imgs.append(np.zeros((3, 101, 101), dtype=np.float32))
                valid_mask.append(False)

        batch = np.stack(imgs, axis=0)
        x = torch.from_numpy(batch).float().to(device)
        emb, scores = extractor.extract(x)

        # Drop failed samples from output
        valid_idx = [i for i, v in enumerate(valid_mask) if v]
        if valid_idx:
            all_emb.append(emb[valid_idx])
            all_scores.append(scores[valid_idx])

            if collect_layers:
                layer_embs = extractor.get_layer_embeddings()
                for k, v in layer_embs.items():
                    if k not in all_layer_embs:
                        all_layer_embs[k] = []
                    all_layer_embs[k].append(v[valid_idx])

        if (start // batch_size) % 10 == 0:
            print(f"  {end}/{len(paths)}", end="\r")

    n_failed = len(failed_paths)
    n_total = len(paths)
    print(f"\n  Cutout loads: {n_total - n_failed}/{n_total} succeeded, {n_failed} failed")
    if n_failed > 0:
        raise RuntimeError(
            f"{n_failed} of {n_total} cutout loads failed. "
            f"First 5 failed paths: {failed_paths[:5]}"
        )

    layer_result = None
    if collect_layers and all_layer_embs:
        layer_result = {k: np.concatenate(v, axis=0) for k, v in all_layer_embs.items()}

    return np.concatenate(all_emb, axis=0), np.concatenate(all_scores, axis=0), layer_result


def extract_embeddings_from_injections(
    extractor: EmbeddingExtractor,
    hosts_df: pd.DataFrame,
    device: torch.device,
    pp_kwargs: dict,
    rng: np.random.Generator,
    n_samples: int = 500,
    theta_e: float = 1.5,
    target_mag: float = 19.0,
    beta_frac_range: Tuple[float, float] = (0.1, 1.0),
    seed_offset: int = 0,
    collect_layers: bool = False,
    # FIX (2026-02-16): add realism flags that were previously missing
    add_sky_noise: bool = False,
    add_poisson_noise: bool = False,
    gain_e_per_nmgy: float = 150.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """Generate injections and extract embeddings.

    If collect_layers is True, also returns per-layer embeddings dict.
    """
    from dhs.injection_engine import SourceParams
    import dataclasses

    all_emb = []
    all_scores = []
    all_layer_embs: Dict[str, list] = {}
    n_done = 0

    host_indices = rng.choice(len(hosts_df), size=n_samples, replace=True)

    for i, host_idx in enumerate(host_indices):
        host_row = hosts_df.iloc[host_idx]
        try:
            with np.load(str(host_row["cutout_path"])) as z:
                hwc = z["cutout"].astype(np.float32)
        except Exception:
            continue

        host_t = torch.from_numpy(hwc).float()
        host_psf = float(host_row["psfsize_r"])

        lens = sample_lens_params(rng, theta_e)
        source = sample_source_params(rng, theta_e, beta_frac_range=beta_frac_range)

        # Scale to target magnitude
        target_flux = 10.0 ** ((22.5 - target_mag) / 2.5)
        if source.flux_nmgy_r > 0:
            scale = target_flux / source.flux_nmgy_r
        else:
            scale = 1.0
        fields = {f.name: getattr(source, f.name) for f in dataclasses.fields(source)}
        fields["flux_nmgy_r"] = source.flux_nmgy_r * scale
        fields["flux_nmgy_g"] = source.flux_nmgy_g * scale
        fields["flux_nmgy_z"] = source.flux_nmgy_z * scale
        source = SourceParams(**fields)

        # FIX (2026-02-16): pass add_sky_noise and add_poisson_noise
        result = inject_sis_shear(
            host_nmgy_hwc=host_t,
            lens=lens,
            source=source,
            pixel_scale=PIXEL_SCALE,
            psf_fwhm_r_arcsec=host_psf,
            seed=42 + seed_offset + i,
            add_sky_noise=add_sky_noise,
            add_poisson_noise=add_poisson_noise,
            gain_e_per_nmgy=gain_e_per_nmgy,
        )

        inj_chw = result.injected[0].detach().cpu().numpy()
        proc = preprocess_stack(inj_chw, **pp_kwargs)
        x = torch.from_numpy(proc[None]).float().to(device)
        emb, scores = extractor.extract(x)
        all_emb.append(emb)
        all_scores.append(scores)

        if collect_layers:
            layer_embs = extractor.get_layer_embeddings()
            for k, v in layer_embs.items():
                if k not in all_layer_embs:
                    all_layer_embs[k] = []
                all_layer_embs[k].append(v)

        n_done += 1

        if (n_done) % 100 == 0:
            print(f"  {n_done}/{n_samples}", end="\r")

    if not all_emb:
        return np.zeros((0, 1280)), np.zeros(0), None

    layer_result = None
    if collect_layers and all_layer_embs:
        layer_result = {k: np.concatenate(v, axis=0) for k, v in all_layer_embs.items()}

    return np.concatenate(all_emb, axis=0), np.concatenate(all_scores, axis=0), layer_result


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Feature space analysis: real lenses vs injections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=500,
                    help="Samples per category (default: 500)")
    ap.add_argument("--theta-e", type=float, default=1.5)
    ap.add_argument("--target-mag", type=float, default=19.0,
                    help="Target magnitude for bright injections (default: 19.0)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    # Injection realism flags (FIX 2026-02-16: were previously missing)
    ap.add_argument("--add-sky-noise", action="store_true", default=False,
                    help="Add realistic sky noise texture to injected arcs.")
    ap.add_argument("--add-poisson-noise", action="store_true", default=False,
                    help="Add Poisson (shot) noise to injected arcs.")
    ap.add_argument("--gain-e-per-nmgy", type=float, default=150.0,
                    help="Gain in e-/nmgy for Poisson noise (default: 150)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    model, pp_kwargs = load_model_and_spec(args.checkpoint, device)
    extractor = EmbeddingExtractor(model)
    # Register per-layer hooks for multi-layer Fréchet distance analysis
    layer_names = extractor.register_layer_hooks()
    print(f"  Registered hooks on {len(layer_names)} layers: {layer_names}")

    # Load manifest
    print("Loading manifest...")
    df = pd.read_parquet(args.manifest)
    val_df = df[df["split"] == "val"].copy()

    # Category (a): Real Tier-A lenses
    print("\n--- (a) Real Tier-A lenses ---")
    if TIER_COL in val_df.columns:
        tier_a = val_df[(val_df["label"] == 1) & (val_df[TIER_COL] == "A")]
    else:
        tier_a = val_df[val_df["label"] == 1]
        print("  WARNING: No tier column; using all positives as Tier-A proxy")

    n_a = min(args.n_samples, len(tier_a))
    tier_a_sample = tier_a.sample(n=n_a, random_state=args.seed)
    paths_a = tier_a_sample["cutout_path"].astype(str).tolist()
    print(f"  Extracting embeddings for {n_a} Tier-A lenses...")
    emb_a, scores_a, layers_a = extract_embeddings_from_paths(
        extractor, paths_a, device, pp_kwargs, collect_layers=True)
    print(f"  Done: {emb_a.shape[0]} embeddings, median score={np.median(scores_a):.4f}")

    # Category (d): Negatives
    print("\n--- (d) Val negatives ---")
    neg_df = val_df[val_df["label"] == 0].dropna(subset=["psfsize_r", "psfdepth_r"])
    n_d = min(args.n_samples, len(neg_df))
    neg_sample = neg_df.sample(n=n_d, random_state=args.seed)
    paths_d = neg_sample["cutout_path"].astype(str).tolist()
    print(f"  Extracting embeddings for {n_d} negatives...")
    emb_d, scores_d, layers_d = extract_embeddings_from_paths(
        extractor, paths_d, device, pp_kwargs, collect_layers=True)
    print(f"  Done: {emb_d.shape[0]} embeddings, median score={np.median(scores_d):.4f}")

    # Category (b): Bright injections at low beta_frac (dramatic arcs)
    print("\n--- (b) Bright injections, low beta_frac [0.1, 0.3] ---")
    hosts_df = neg_df.sample(n=min(2000, len(neg_df)), random_state=args.seed + 1)
    emb_b, scores_b, layers_b = extract_embeddings_from_injections(
        extractor, hosts_df, device, pp_kwargs, rng,
        n_samples=args.n_samples, theta_e=args.theta_e,
        target_mag=args.target_mag, beta_frac_range=(0.1, 0.3), seed_offset=0,
        collect_layers=True,
        add_sky_noise=args.add_sky_noise,
        add_poisson_noise=args.add_poisson_noise,
        gain_e_per_nmgy=args.gain_e_per_nmgy,
    )
    print(f"  Done: {emb_b.shape[0]} embeddings, median score={np.median(scores_b):.4f}")

    # Category (c): Bright injections at high beta_frac (non-arcs)
    print("\n--- (c) Bright injections, high beta_frac [0.7, 1.0] ---")
    emb_c, scores_c, layers_c = extract_embeddings_from_injections(
        extractor, hosts_df, device, pp_kwargs, rng,
        n_samples=args.n_samples, theta_e=args.theta_e,
        target_mag=args.target_mag, beta_frac_range=(0.7, 1.0), seed_offset=10000,
        collect_layers=True,
        add_sky_noise=args.add_sky_noise,
        add_poisson_noise=args.add_poisson_noise,
        gain_e_per_nmgy=args.gain_e_per_nmgy,
    )
    print(f"  Done: {emb_c.shape[0]} embeddings, median score={np.median(scores_c):.4f}")

    # Linear probe: real (a) vs injection (b)
    print("\n--- Linear probe: real Tier-A vs low-beta_frac injections ---")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score

    X_probe = np.concatenate([emb_a, emb_b], axis=0)
    y_probe = np.concatenate([np.ones(len(emb_a)), np.zeros(len(emb_b))])

    if len(emb_a) >= 10 and len(emb_b) >= 10:
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        cv_scores = cross_val_score(clf, X_probe, y_probe, cv=5, scoring="roc_auc")
        probe_auc_mean = float(np.mean(cv_scores))
        probe_auc_std = float(np.std(cv_scores))
        print(f"  5-fold CV AUC: {probe_auc_mean:.4f} +/- {probe_auc_std:.4f}")
        print(f"  Interpretation: AUC near 1.0 = CNN encodes 'injection-ness'")
    else:
        probe_auc_mean = float("nan")
        probe_auc_std = float("nan")
        print("  Too few samples for linear probe")

    # Frechet distance between real and injection embedding distributions
    print("\n--- Frechet distance ---")

    def frechet_distance(mu1, sigma1, mu2, sigma2):
        """Compute Frechet distance between two multivariate Gaussians."""
        from scipy.linalg import sqrtm
        diff = mu1 - mu2
        covmean = sqrtm(sigma1 @ sigma2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(
            diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        )

    fd_ab = float("nan")
    if emb_a.shape[0] > 10 and emb_b.shape[0] > 10:
        mu_a, sig_a = np.mean(emb_a, axis=0), np.cov(emb_a, rowvar=False)
        mu_b, sig_b = np.mean(emb_b, axis=0), np.cov(emb_b, rowvar=False)
        try:
            fd_ab = frechet_distance(mu_a, sig_a, mu_b, sig_b)
            print(f"  FD(real vs low-bf injection): {fd_ab:.2f}")
        except Exception as e:
            print(f"  FD computation failed: {e}")

    fd_ac = float("nan")
    if emb_a.shape[0] > 10 and emb_c.shape[0] > 10:
        mu_c, sig_c = np.mean(emb_c, axis=0), np.cov(emb_c, rowvar=False)
        try:
            fd_ac = frechet_distance(mu_a, sig_a, mu_c, sig_c)
            print(f"  FD(real vs high-bf injection): {fd_ac:.2f}")
        except Exception as e:
            print(f"  FD computation failed: {e}")

    # Per-layer Frechet distance (identify at which feature scale
    # divergence between real and injected arcs occurs)
    print("\n--- Per-layer Frechet distance  ---")
    per_layer_fd = {}
    if layers_a and layers_b:
        for layer_name in sorted(layers_a.keys()):
            if layer_name in layers_b:
                la = layers_a[layer_name]
                lb = layers_b[layer_name]
                if la.shape[0] > la.shape[1] and lb.shape[0] > lb.shape[1]:
                    try:
                        mu_la = np.mean(la, axis=0)
                        sig_la = np.cov(la, rowvar=False)
                        mu_lb = np.mean(lb, axis=0)
                        sig_lb = np.cov(lb, rowvar=False)
                        fd = frechet_distance(mu_la, sig_la, mu_lb, sig_lb)
                        per_layer_fd[layer_name] = fd
                        print(f"  {layer_name} (dim={la.shape[1]}): FD={fd:.2f}")
                    except Exception as e:
                        per_layer_fd[layer_name] = float("nan")
                        print(f"  {layer_name}: failed ({e})")
                else:
                    per_layer_fd[layer_name] = float("nan")
                    print(f"  {layer_name}: too few samples for cov (n={la.shape[0]}, d={la.shape[1]})")
    else:
        print("  Skipped (layer embeddings not available)")

    # Save embeddings for downstream UMAP/t-SNE
    npz_path = os.path.join(args.out_dir, "embeddings.npz")
    np.savez_compressed(
        npz_path,
        emb_real_tier_a=emb_a, scores_real_tier_a=scores_a,
        emb_inj_low_bf=emb_b, scores_inj_low_bf=scores_b,
        emb_inj_high_bf=emb_c, scores_inj_high_bf=scores_c,
        emb_negatives=emb_d, scores_negatives=scores_d,
    )
    print(f"\nEmbeddings saved: {npz_path}")

    # Save summary results
    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "n_real_tier_a": int(emb_a.shape[0]),
        "n_inj_low_bf": int(emb_b.shape[0]),
        "n_inj_high_bf": int(emb_c.shape[0]),
        "n_negatives": int(emb_d.shape[0]),
        "target_mag": args.target_mag,
        "theta_e": args.theta_e,
        "add_sky_noise": args.add_sky_noise,
        "add_poisson_noise": args.add_poisson_noise,
        "gain_e_per_nmgy": args.gain_e_per_nmgy if args.add_poisson_noise else None,
        "linear_probe": {
            "task": "real_tier_a vs inj_low_bf",
            "cv_auc_mean": probe_auc_mean,
            "cv_auc_std": probe_auc_std,
            "interpretation": "AUC near 1.0 means CNN encodes injection-ness direction",
        },
        "frechet_distance_per_layer": per_layer_fd,
        "frechet_distance": {
            "real_vs_low_bf": fd_ab,
            "real_vs_high_bf": fd_ac,
            "interpretation": "Higher FD = more different in feature space",
        },
        "median_scores": {
            "real_tier_a": float(np.median(scores_a)),
            "inj_low_bf": float(np.median(scores_b)),
            "inj_high_bf": float(np.median(scores_c)),
            "negatives": float(np.median(scores_d)),
        },
    }

    json_path = os.path.join(args.out_dir, "feature_space_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved: {json_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("FEATURE SPACE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Linear probe AUC (real vs inj): {probe_auc_mean:.4f} +/- {probe_auc_std:.4f}")
    print(f"  Frechet distance (real vs low-bf inj): {fd_ab:.2f}")
    print(f"  Frechet distance (real vs high-bf inj): {fd_ac:.2f}")
    print(f"  Median scores: real={np.median(scores_a):.4f}, "
          f"inj_low_bf={np.median(scores_b):.4f}, "
          f"inj_high_bf={np.median(scores_c):.4f}, "
          f"neg={np.median(scores_d):.4f}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
