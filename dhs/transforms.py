from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .utils import radial_rmap

@dataclass
class AugmentConfig:
    hflip: bool = True
    vflip: bool = True
    rot90: bool = True
    core_dropout_prob: float = 0.0
    core_radius: float = 5.0
    az_shuffle_prob: float = 0.0
    az_shuffle_rmax: int = 32

def random_augment(img3: np.ndarray, seed: int, cfg: AugmentConfig) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = img3
    if cfg.hflip and rng.random() < 0.5:
        x = x[..., :, ::-1]
    if cfg.vflip and rng.random() < 0.5:
        x = x[..., ::-1, :]
    if cfg.rot90:
        k = int(rng.integers(0, 4))
        x = np.rot90(x, k=k, axes=(-2, -1))
    if cfg.core_dropout_prob > 0 and rng.random() < cfg.core_dropout_prob:
        H, W = x.shape[-2], x.shape[-1]
        r = radial_rmap(H, W)
        m = r < cfg.core_radius
        x = x.copy()
        x[:, m] = 0.0
    if cfg.az_shuffle_prob > 0 and rng.random() < cfg.az_shuffle_prob:
        x = azimuthal_shuffle(x, seed=int(rng.integers(0, 2**31-1)), rmax=cfg.az_shuffle_rmax)
    return x.astype(np.float32)

def azimuthal_shuffle(img3: np.ndarray, seed: int, rmax: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H, W = img3.shape[-2], img3.shape[-1]
    r = radial_rmap(H, W)
    out = img3.copy()
    for ri in range(rmax):
        m = (r >= ri) & (r < ri + 1)
        idx = np.argwhere(m)
        if len(idx) < 4:
            continue
        perm = rng.permutation(len(idx))
        for c in range(out.shape[0]):
            vals = out[c][m].copy()
            out[c][m] = vals[perm]
    return out.astype(np.float32)
