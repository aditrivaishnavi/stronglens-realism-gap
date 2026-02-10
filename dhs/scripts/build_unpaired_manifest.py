from __future__ import annotations
import argparse
from dhs.data import build_unpaired_manifest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bins", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--pos_blob_col", default="stamp_npz")
    ap.add_argument("--neg_blob_col", default="ctrl_stamp_npz")
    ap.add_argument("--ra_col", default="ra")
    ap.add_argument("--dec_col", default="dec")
    ap.add_argument("--split_col", default="split")
    a = ap.parse_args()
    bins = [b.strip() for b in a.bins.split(",") if b.strip()]
    print(build_unpaired_manifest(
        parquet_path=a.parquet, out_path=a.out, bins=bins, seed=a.seed,
        pos_blob_col=a.pos_blob_col, neg_blob_col=a.neg_blob_col,
        ra_col=a.ra_col, dec_col=a.dec_col, split_col=a.split_col,
    ))

if __name__ == "__main__":
    main()
