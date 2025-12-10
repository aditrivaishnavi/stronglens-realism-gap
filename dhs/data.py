from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .constants import BANDS, SEED_DEFAULT
from .utils import decode_npz_blob
from .preprocess import preprocess_stack
from .transforms import AugmentConfig, random_augment

def _read_parquet_any(path: str) -> pd.DataFrame:
    table = pq.read_table(path)
    return table.to_pandas()

def stack_from_npz(decoded: dict[str, np.ndarray]) -> np.ndarray:
    imgs = []
    for b in BANDS:
        k = f"image_{b}"
        if k not in decoded:
            raise KeyError(f"Missing {k} in npz")
        imgs.append(decoded[k].astype(np.float32))
    return np.stack(imgs, axis=0)

@dataclass
class DatasetConfig:
    parquet_path: str
    mode: str
    preprocessing: str
    label_col: str = "label"
    seed: int = SEED_DEFAULT
    manifest_path: Optional[str] = None
    cutout_path_col: str = "cutout_path"  # For file_manifest mode
    sample_weight_col: Optional[str] = "sample_weight"  # For weighted loss
    crop: bool = True         # False = keep 101x101 for Paper IV parity
    crop_size: int = 0        # 0 = default STAMP_SIZE; >0 = custom crop size
    # Annulus override for corrected normalization (see KNOWN ISSUE in dhs/utils.py).
    # Q1.8: Sentinel value 0.0 means "use normalize_outer_annulus defaults (20, 32)",
    # which are locked to all existing trained models. Negative values also default.
    # For retraining with corrected annulus, set BOTH to default_annulus_radii output:
    #   annulus_r_in: 32.5   annulus_r_out: 45.0  (for 101x101 stamps)
    # Both must be set together or both left at 0.0; partial setting will raise ValueError.
    annulus_r_in: float = 0.0   # 0.0 = use default (20 px)
    annulus_r_out: float = 0.0  # 0.0 = use default (32 px)

@dataclass
class SplitConfig:
    split_col: str = "split"
    split_value: str = "train"

def load_cutout_from_file(path: str) -> np.ndarray:
    """Load cutout from local .npz file. Returns (H, W, 3) array in CHW format after transpose."""
    with np.load(path, allow_pickle=True) as data:
        cutout = data["cutout"]  # Shape: (101, 101, 3) in HWC format
        # Transpose to CHW for PyTorch
        return cutout.transpose(2, 0, 1).astype(np.float32)


class LensDataset:
    def __init__(self, dcfg: DatasetConfig, scfg: SplitConfig, aug: AugmentConfig):
        self.dcfg, self.scfg, self.aug = dcfg, scfg, aug
        self.epoch = 0  # Updated by training loop each epoch for augmentation diversity
        if dcfg.mode == "paired":
            df = _read_parquet_any(dcfg.parquet_path)
        elif dcfg.mode == "unpaired_manifest":
            if not dcfg.manifest_path:
                raise ValueError("manifest_path required for unpaired_manifest")
            df = _read_parquet_any(dcfg.manifest_path)
        elif dcfg.mode == "file_manifest":
            # New mode: load from local file paths
            if not dcfg.manifest_path:
                raise ValueError("manifest_path required for file_manifest mode")
            df = _read_parquet_any(dcfg.manifest_path)
        else:
            raise ValueError(f"Unknown mode {dcfg.mode}")
        if scfg.split_col in df.columns:
            df = df[df[scfg.split_col] == scfg.split_value].reset_index(drop=True)
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _get_blob_and_label(self, i: int):
        row = self.df.iloc[i]
        if self.dcfg.mode == "paired":
            if (i % 2) == 0:
                return row["stamp_npz"], 1
            return row["ctrl_stamp_npz"], 0
        return row["blob"], int(row[self.dcfg.label_col])

    def _get_file_and_label(self, i: int):
        """For file_manifest mode: return file path and label."""
        row = self.df.iloc[i]
        path = row[self.dcfg.cutout_path_col]
        label = int(row[self.dcfg.label_col])
        # Q4.3 fix: explicit None check avoids semantically odd row.get(None, 1.0)
        if self.dcfg.sample_weight_col is not None:
            weight = float(row.get(self.dcfg.sample_weight_col, 1.0))
        else:
            weight = 1.0
        return path, label, weight

    def __getitem__(self, i: int):
        # Determine crop and annulus settings from config
        crop_kwargs = {}
        if hasattr(self.dcfg, 'crop'):
            crop_kwargs['crop'] = self.dcfg.crop
        if hasattr(self.dcfg, 'crop_size') and self.dcfg.crop_size > 0:
            crop_kwargs['crop_size'] = self.dcfg.crop_size
        # Pass annulus radii if explicitly set (> 0); otherwise let
        # normalize_outer_annulus use its hardcoded defaults (20, 32).
        if self.dcfg.annulus_r_in > 0:
            crop_kwargs['annulus_r_in'] = self.dcfg.annulus_r_in
        if self.dcfg.annulus_r_out > 0:
            crop_kwargs['annulus_r_out'] = self.dcfg.annulus_r_out

        # Augmentation seed includes epoch so the model sees different
        # augmentations each epoch while remaining deterministic/reproducible.
        # Without epoch, every sample gets the same augmentation forever and
        # the model memorises the fixed augmented training set.
        seed = (self.dcfg.seed * 1000003 + i + self.epoch * len(self)) & 0x7fffffff

        if self.dcfg.mode == "file_manifest":
            # Load from local file
            path, y, weight = self._get_file_and_label(i)
            img3 = load_cutout_from_file(path)  # Already CHW
            img3 = preprocess_stack(img3, mode=self.dcfg.preprocessing, **crop_kwargs)
            img3 = random_augment(img3, seed=seed, cfg=self.aug)
            return img3, np.int64(y), np.float32(weight)
        else:
            blob, y = self._get_blob_and_label(i)
            dec = decode_npz_blob(blob)
            img3 = stack_from_npz(dec)
            img3 = preprocess_stack(img3, mode=self.dcfg.preprocessing, **crop_kwargs)
            img3 = random_augment(img3, seed=seed, cfg=self.aug)
            return img3, np.int64(y)

def build_unpaired_manifest(
    parquet_path: str,
    out_path: str,
    bins: Sequence[str],
    seed: int = SEED_DEFAULT,
    pos_blob_col: str = "stamp_npz",
    neg_blob_col: str = "ctrl_stamp_npz",
    ra_col: str = "ra",
    dec_col: str = "dec",
    split_col: str = "split",
) -> str:
    df = _read_parquet_any(parquet_path).copy()
    rng = np.random.default_rng(seed)
    if split_col not in df.columns:
        raise ValueError(f"Expected split_col={split_col} in parquet")

    df["_lrg_id"] = list(zip(df[ra_col].astype(float).round(6), df[dec_col].astype(float).round(6)))

    outs = []
    for split in ["train","val","test"]:
        d = df[df[split_col] == split].reset_index(drop=True)
        lrgs = d["_lrg_id"].unique()
        rng.shuffle(lrgs)
        mid = len(lrgs) // 2
        pos_lrgs, neg_lrgs = set(lrgs[:mid]), set(lrgs[mid:])

        pos = d[d["_lrg_id"].isin(pos_lrgs)].copy()
        neg = d[d["_lrg_id"].isin(neg_lrgs)].copy()

        pos_rows = pos[[pos_blob_col, "_lrg_id", *bins]].copy()
        pos_rows.rename(columns={pos_blob_col: "blob"}, inplace=True)
        pos_rows["label"] = 1
        pos_rows["split"] = split

        neg_rows = neg[[neg_blob_col, "_lrg_id", *bins]].copy()
        neg_rows.rename(columns={neg_blob_col: "blob"}, inplace=True)
        neg_rows["label"] = 0
        neg_rows["split"] = split

        # quantile bin edges from positives
        bpos, bneg = pos_rows.copy(), neg_rows.copy()
        for col in bins:
            qs = pos_rows[col].quantile([0.0,0.2,0.4,0.6,0.8,1.0]).values
            qs = np.unique(qs)
            if len(qs) < 3:
                bpos[col+"_bin"] = 0
                bneg[col+"_bin"] = 0
            else:
                bpos[col+"_bin"] = np.digitize(pos_rows[col].values, qs[1:-1], right=True)
                bneg[col+"_bin"] = np.digitize(neg_rows[col].values, qs[1:-1], right=True)

        bin_cols = [c+"_bin" for c in bins]
        pos_counts = bpos.groupby(bin_cols).size().reset_index(name="n_pos")
        bneg = bneg.merge(pos_counts, on=bin_cols, how="left")
        bneg["n_pos"] = bneg["n_pos"].fillna(0).astype(int)

        sampled = []
        for _, grp in bneg.groupby(bin_cols, dropna=False):
            n = int(grp["n_pos"].iloc[0])
            if n <= 0:
                continue
            take = grp.sample(n=min(n, len(grp)), replace=(n>len(grp)),
                              random_state=int(rng.integers(0, 2**31-1)))
            sampled.append(take.drop(columns=["n_pos"]))
        sampled_negs = pd.concat(sampled, ignore_index=True) if sampled else bneg.head(0).drop(columns=["n_pos"])

        outs.append(pd.concat([pos_rows, sampled_negs.drop(columns=bin_cols, errors="ignore")], ignore_index=True))

    manifest = pd.concat(outs, ignore_index=True)

    import pyarrow as pa
    table = pa.Table.from_pandas(manifest)
    pq.write_table(table, out_path, compression="zstd")
    return out_path
