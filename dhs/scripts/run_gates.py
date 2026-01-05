from __future__ import annotations
import argparse, yaml
import numpy as np
from torch.utils.data import DataLoader
from dhs.data import DatasetConfig, SplitConfig, LensDataset
from dhs.transforms import AugmentConfig
from dhs.gates import run_shortcut_gates

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--n", type=int, default=2048)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    dcfg = DatasetConfig(**cfg["dataset"])
    aug = AugmentConfig(hflip=False, vflip=False, rot90=False, core_dropout_prob=0.0, az_shuffle_prob=0.0)
    ds = LensDataset(dcfg, SplitConfig(split_value=args.split), aug)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, collate_fn=lambda b: tuple(zip(*b)))
    xs, ys = [], []
    for x, y in dl:
        xs.extend(list(x))
        ys.extend(list(y))
        if len(ys) >= args.n:
            break
    xs = np.stack(xs[:args.n], axis=0)
    ys = np.array(ys[:args.n]).astype(int)
    res = run_shortcut_gates(xs, ys)
    print(f"core_lr_auc={res.core_lr_auc:.4f} radial_profile_auc={res.radial_profile_auc:.4f}")

if __name__ == "__main__":
    main()
