#!/usr/bin/env python3
"""
Permutation test and bootstrap CI for the linear probe AUC.

Checkpointing: saves state after every permutation iteration to a single
~20KB file (checkpoint.npz). On restart, resumes from the last completed
iteration with RNG state preserved. The observed AUC and CV predictions
are also checkpointed so the 30s initial fit is not repeated.
"""
import os
import sys
import tempfile
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import json
import time
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE = "results/D06_corrected_priors/linear_probe"
EMB_PATH = os.path.join(BASE, "embeddings.npz")
OUT_PATH = os.path.join(BASE, "permutation_bootstrap_results.json")
PROGRESS = os.path.join(BASE, "progress.txt")
CKPT_PATH = os.path.join(BASE, "checkpoint.npz")

N_PERM = 1000
N_BOOT = 5000

def log(msg):
    """Append to progress file with fsync."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(line, file=sys.stderr)

def save_checkpoint(completed_perm, perm_aucs, observed_auc, observed_std,
                    fold_aucs, single_auc, y_prob, rng_state):
    """Atomic write: temp file + rename to avoid corruption."""
    fd, tmp = tempfile.mkstemp(dir=BASE, suffix=".npz")
    os.close(fd)
    np.savez(tmp,
             completed_perm=np.array(completed_perm),
             perm_aucs=perm_aucs,
             observed_auc=np.array(observed_auc),
             observed_std=np.array(observed_std),
             fold_aucs=np.array(fold_aucs),
             single_auc=np.array(single_auc),
             y_prob=y_prob,
             rng_state=np.array(rng_state, dtype=object))
    os.replace(tmp, CKPT_PATH)

def load_checkpoint():
    """Returns dict or None."""
    if not os.path.exists(CKPT_PATH):
        return None
    try:
        ckpt = np.load(CKPT_PATH, allow_pickle=True)
        return {
            "completed_perm": int(ckpt["completed_perm"]),
            "perm_aucs": ckpt["perm_aucs"],
            "observed_auc": float(ckpt["observed_auc"]),
            "observed_std": float(ckpt["observed_std"]),
            "fold_aucs": ckpt["fold_aucs"].tolist(),
            "single_auc": float(ckpt["single_auc"]),
            "y_prob": ckpt["y_prob"],
            "rng_state": tuple(ckpt["rng_state"]),
        }
    except Exception as e:
        log(f"WARNING: corrupt checkpoint, starting fresh: {e}")
        return None


# ── Load data ──
log("Loading embeddings...")
data = np.load(EMB_PATH)
emb_real = data["emb_real_tier_a"]
emb_inj = data["emb_inj_low_bf"]
X = np.vstack([emb_real, emb_inj])
y = np.concatenate([np.ones(len(emb_real)), np.zeros(len(emb_inj))])
log(f"X={X.shape}, n_real={int(y.sum())}, n_inj={int((1-y).sum())}")

# ── Check for checkpoint ──
ckpt = load_checkpoint()

if ckpt is not None:
    log(f"RESUMING from checkpoint: {ckpt['completed_perm']}/{N_PERM} permutations done")
    observed_auc = ckpt["observed_auc"]
    observed_std = ckpt["observed_std"]
    fold_aucs = ckpt["fold_aucs"]
    single_auc = ckpt["single_auc"]
    y_prob = ckpt["y_prob"]
    perm_aucs = ckpt["perm_aucs"]
    start_perm = ckpt["completed_perm"]
    np.random.set_state(ckpt["rng_state"])
    log(f"  observed_auc={observed_auc:.6f}, single_split={single_auc:.6f}")
else:
    log("No checkpoint found, starting fresh")
    np.random.seed(42)

    # ── 1. Observed AUC (5-fold CV) ──
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, solver="lbfgs"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    log("Computing observed AUC (5-fold CV)...")
    t0 = time.time()
    y_prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    observed_auc = roc_auc_score(y, y_prob)
    fold_aucs = []
    for train_idx, test_idx in cv.split(X, y):
        fold_aucs.append(roc_auc_score(y[test_idx], y_prob[test_idx]))
    observed_std = np.std(fold_aucs)
    log(f"AUC={observed_auc:.6f} +/- {observed_std:.6f} ({time.time()-t0:.1f}s)")
    log(f"Per-fold: {[f'{a:.4f}' for a in fold_aucs]}")

    # ── Single-split observed AUC ──
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx_tmp, te_idx_tmp = next(splitter.split(X, y))
    sc_tmp = StandardScaler().fit(X[tr_idx_tmp])
    lr_tmp = LogisticRegression(max_iter=500, solver="lbfgs")
    lr_tmp.fit(sc_tmp.transform(X[tr_idx_tmp]), y[tr_idx_tmp])
    single_auc = roc_auc_score(y[te_idx_tmp], lr_tmp.predict_proba(sc_tmp.transform(X[te_idx_tmp]))[:, 1])
    log(f"Single-split observed AUC: {single_auc:.6f}")

    perm_aucs = np.zeros(N_PERM)
    start_perm = 0

    save_checkpoint(0, perm_aucs, observed_auc, observed_std,
                    fold_aucs, single_auc, y_prob, np.random.get_state())
    log("Initial checkpoint saved")

# ── 2. Permutation test ──
# Recompute the fixed split & scaled features (cheap, <1s)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(splitter.split(X, y))
sc = StandardScaler().fit(X[tr_idx])
X_tr_s = sc.transform(X[tr_idx])
X_te_s = sc.transform(X[te_idx])

if start_perm < N_PERM:
    log(f"--- Permutation test: {start_perm}/{N_PERM} -> {N_PERM} ---")
    t0 = time.time()
    for i in range(start_perm, N_PERM):
        t_iter = time.time()
        y_perm = np.random.permutation(y)
        y_tr_p, y_te_p = y_perm[tr_idx], y_perm[te_idx]
        if len(np.unique(y_tr_p)) < 2 or len(np.unique(y_te_p)) < 2:
            perm_aucs[i] = 0.5
        else:
            lr_p = LogisticRegression(max_iter=200, solver="lbfgs")
            lr_p.fit(X_tr_s, y_tr_p)
            perm_aucs[i] = roc_auc_score(y_te_p, lr_p.predict_proba(X_te_s)[:, 1])
        dt = time.time() - t_iter
        elapsed = time.time() - t0
        cur = perm_aucs[:i+1]

        log(f"perm {i+1:4d}/{N_PERM} | {dt:.2f}s | total {elapsed:.0f}s | "
            f"auc={perm_aucs[i]:.4f} | max={cur.max():.4f} "
            f"n>obs={int(np.sum(cur >= single_auc))}")

        save_checkpoint(i + 1, perm_aucs, observed_auc, observed_std,
                        fold_aucs, single_auc, y_prob, np.random.get_state())

    log("Permutation test complete")
else:
    log("Permutation test already complete in checkpoint")

n_exceeding = int(np.sum(perm_aucs >= single_auc))
perm_p = (n_exceeding + 1) / (N_PERM + 1)
log(f"Permutation result: p={perm_p:.2e} ({n_exceeding}/{N_PERM} >= {single_auc:.4f})")
log(f"  Max perm AUC={perm_aucs.max():.4f}, Mean={perm_aucs.mean():.4f}")

# ── 3. Bootstrap CI (resamples predictions, ~1s total) ──
log(f"--- Bootstrap CI ({N_BOOT} iters, resampling predictions) ---")
t0 = time.time()
boot_aucs = np.zeros(N_BOOT)
n = len(y)
for i in range(N_BOOT):
    idx = np.random.choice(n, size=n, replace=True)
    y_b, p_b = y[idx], y_prob[idx]
    if len(np.unique(y_b)) < 2:
        boot_aucs[i] = np.nan
        continue
    boot_aucs[i] = roc_auc_score(y_b, p_b)

valid_boots = boot_aucs[~np.isnan(boot_aucs)]
boot_ci_lo = float(np.percentile(valid_boots, 2.5))
boot_ci_hi = float(np.percentile(valid_boots, 97.5))
log(f"Bootstrap done in {time.time()-t0:.1f}s")
log(f"  95% CI: [{boot_ci_lo:.4f}, {boot_ci_hi:.4f}]")
log(f"  Mean={valid_boots.mean():.4f}, Std={valid_boots.std():.4f}, N_valid={len(valid_boots)}")

# ── Save final results ──
results = {
    "observed_auc_cv": float(observed_auc),
    "observed_auc_std": float(observed_std),
    "observed_auc_folds": [float(a) for a in fold_aucs],
    "observed_auc_single_split": float(single_auc),
    "permutation_test": {
        "n_permutations": N_PERM,
        "method": "single 80/20 stratified split, max_iter=200",
        "p_value": float(perm_p),
        "n_exceeding": n_exceeding,
        "max_perm_auc": float(perm_aucs.max()),
        "mean_perm_auc": float(perm_aucs.mean()),
    },
    "bootstrap_ci": {
        "n_bootstrap": N_BOOT,
        "method": "resample held-out CV predictions",
        "n_valid": int(len(valid_boots)),
        "ci_95_lower": boot_ci_lo,
        "ci_95_upper": boot_ci_hi,
        "mean": float(valid_boots.mean()),
        "std": float(valid_boots.std()),
    },
}
with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
log(f"Results saved to {OUT_PATH}")

log("========== SUMMARY ==========")
log(f"Observed AUC (5-fold CV):  {observed_auc:.4f} +/- {observed_std:.4f}")
log(f"Permutation p-value:       {perm_p:.2e}  ({n_exceeding}/{N_PERM})")
log(f"Bootstrap 95% CI:          [{boot_ci_lo:.4f}, {boot_ci_hi:.4f}]")
log("==============================")
