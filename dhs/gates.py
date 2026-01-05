from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .constants import CORE_BOX
from .utils import center_slices, azimuthal_median_profile

@dataclass
class GateResults:
    core_lr_auc: float
    radial_profile_auc: float

def _core_feat(x3: np.ndarray) -> np.ndarray:
    _, H, W = x3.shape
    ys, xs = center_slices(H, W, CORE_BOX)
    return x3[:, ys, xs].reshape(-1)

def core_lr_auc(xs: np.ndarray, ys: np.ndarray) -> float:
    X = np.stack([_core_feat(x) for x in xs], axis=0)
    y = ys.astype(int)
    clf = LogisticRegression(max_iter=500, solver="liblinear")
    clf.fit(X, y)
    p = clf.predict_proba(X)[:,1]
    return float(roc_auc_score(y, p))

def radial_lr_auc(xs: np.ndarray, ys: np.ndarray, rmax: int = 32) -> float:
    feats = []
    for x in xs:
        v = [azimuthal_median_profile(x[b], r_max=rmax) for b in range(3)]
        feats.append(np.concatenate(v, axis=0))
    X = np.stack(feats, axis=0)
    y = ys.astype(int)
    clf = LogisticRegression(max_iter=500, solver="liblinear")
    clf.fit(X, y)
    p = clf.predict_proba(X)[:,1]
    return float(roc_auc_score(y, p))

def run_shortcut_gates(xs: np.ndarray, ys: np.ndarray) -> GateResults:
    return GateResults(core_lr_auc=core_lr_auc(xs, ys), radial_profile_auc=radial_lr_auc(xs, ys))
