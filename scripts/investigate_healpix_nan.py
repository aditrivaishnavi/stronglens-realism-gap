#!/usr/bin/env python3
"""Investigate NaN healpix_128 for positive samples in the training manifest.

ALL 4,788 positives have NaN in healpix_128.
This script determines:
  1. Whether ra/dec columns exist with valid values for positives
  2. If so, recomputes healpix_128 and checks spatial overlap between splits
  3. Assesses spatial leakage risk for the train/val split

Output: JSON report + console summary.
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    import healpy as hp
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False


def compute_healpix(ra, dec, nside=128):
    """Compute HEALPix index from ra/dec (degrees) using RING scheme."""
    if not HAS_HEALPY:
        return np.full(len(ra), np.nan)
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    return hp.ang2pix(nside, theta, phi)


def main():
    ap = argparse.ArgumentParser(description="Investigate NaN healpix in manifest")
    ap.add_argument("--manifest", required=True, help="Path to training manifest parquet")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--nside", type=int, default=128, help="HEALPix NSIDE (default: 128)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading manifest...")
    df = pd.read_parquet(args.manifest)
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    pos = df[df["label"] == 1].copy()
    neg = df[df["label"] == 0].copy()
    print(f"  Positives: {len(pos)}, Negatives: {len(neg)}")

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_rows": len(df),
        "n_positives": len(pos),
        "n_negatives": len(neg),
        "columns": list(df.columns),
    }

    # --- 1. Check ra/dec columns ---
    has_ra = "ra" in df.columns
    has_dec = "dec" in df.columns
    has_healpix = "healpix_128" in df.columns
    report["has_ra"] = has_ra
    report["has_dec"] = has_dec
    report["has_healpix_128"] = has_healpix

    print(f"\n=== 1. Column presence ===")
    print(f"  ra: {has_ra}, dec: {has_dec}, healpix_128: {has_healpix}")

    if has_ra and has_dec:
        # Check NaN counts for positives vs negatives
        pos_ra_nan = pos["ra"].isna().sum()
        pos_dec_nan = pos["dec"].isna().sum()
        neg_ra_nan = neg["ra"].isna().sum()
        neg_dec_nan = neg["dec"].isna().sum()

        print(f"\n=== 2. NaN counts in ra/dec ===")
        print(f"  Positives: ra NaN={pos_ra_nan}/{len(pos)}, dec NaN={pos_dec_nan}/{len(pos)}")
        print(f"  Negatives: ra NaN={neg_ra_nan}/{len(neg)}, dec NaN={neg_dec_nan}/{len(neg)}")

        report["positive_ra_nan"] = int(pos_ra_nan)
        report["positive_dec_nan"] = int(pos_dec_nan)
        report["negative_ra_nan"] = int(neg_ra_nan)
        report["negative_dec_nan"] = int(neg_dec_nan)

        pos_valid = pos.dropna(subset=["ra", "dec"])
        print(f"  Positives with valid ra/dec: {len(pos_valid)}/{len(pos)}")
        report["positives_with_valid_radec"] = len(pos_valid)

        if len(pos_valid) > 0:
            print(f"\n  RA range: [{pos_valid['ra'].min():.4f}, {pos_valid['ra'].max():.4f}]")
            print(f"  Dec range: [{pos_valid['dec'].min():.4f}, {pos_valid['dec'].max():.4f}]")
            report["positive_ra_range"] = [float(pos_valid["ra"].min()), float(pos_valid["ra"].max())]
            report["positive_dec_range"] = [float(pos_valid["dec"].min()), float(pos_valid["dec"].max())]

            # --- 3. Check healpix_128 for positives ---
            if has_healpix:
                pos_hp_nan = pos["healpix_128"].isna().sum()
                neg_hp_nan = neg["healpix_128"].isna().sum()
                print(f"\n=== 3. HEALPix NaN counts ===")
                print(f"  Positives: healpix_128 NaN={pos_hp_nan}/{len(pos)}")
                print(f"  Negatives: healpix_128 NaN={neg_hp_nan}/{len(neg)}")
                report["positive_healpix_nan"] = int(pos_hp_nan)
                report["negative_healpix_nan"] = int(neg_hp_nan)

            # --- 4. Recompute healpix and check spatial overlap ---
            if HAS_HEALPY and len(pos_valid) > 0:
                print(f"\n=== 4. Recomputing HEALPix (NSIDE={args.nside}) ===")
                hp_idx = compute_healpix(
                    pos_valid["ra"].values, pos_valid["dec"].values, args.nside
                )
                pos_valid = pos_valid.copy()
                pos_valid["hp_recomputed"] = hp_idx

                # Split-wise analysis
                for split_name in sorted(pos_valid["split"].unique()):
                    split_pos = pos_valid[pos_valid["split"] == split_name]
                    unique_hp = split_pos["hp_recomputed"].nunique()
                    print(f"  {split_name}: {len(split_pos)} positives in {unique_hp} unique HEALPix pixels")

                # Check overlap between train and val
                train_pos = pos_valid[pos_valid["split"] == "train"]
                val_pos = pos_valid[pos_valid["split"] == "val"]

                if len(train_pos) > 0 and len(val_pos) > 0:
                    train_hp = set(train_pos["hp_recomputed"].unique())
                    val_hp = set(val_pos["hp_recomputed"].unique())
                    overlap = train_hp & val_hp
                    print(f"\n  Train HEALPix pixels: {len(train_hp)}")
                    print(f"  Val HEALPix pixels: {len(val_hp)}")
                    print(f"  Overlapping pixels: {len(overlap)}")

                    # Count positives in overlapping pixels
                    if len(overlap) > 0:
                        train_in_overlap = train_pos[train_pos["hp_recomputed"].isin(overlap)]
                        val_in_overlap = val_pos[val_pos["hp_recomputed"].isin(overlap)]
                        print(f"  Train positives in overlapping pixels: {len(train_in_overlap)}")
                        print(f"  Val positives in overlapping pixels: {len(val_in_overlap)}")

                        # Angular separation between overlapping train/val positives
                        # (minimum angular sep between any train-val pair in same pixel)
                        print(f"\n  WARNING: {len(overlap)} HEALPix pixels contain BOTH train and val positives.")
                        print(f"  This means spatial leakage is POSSIBLE.")

                    report["spatial_overlap"] = {
                        "train_unique_pixels": len(train_hp),
                        "val_unique_pixels": len(val_hp),
                        "overlapping_pixels": len(overlap),
                        "train_in_overlap": len(train_in_overlap) if len(overlap) > 0 else 0,
                        "val_in_overlap": len(val_in_overlap) if len(overlap) > 0 else 0,
                    }

                    # Per-tier analysis
                    if "tier" in pos_valid.columns:
                        print(f"\n=== 5. Tier-wise spatial analysis ===")
                        for tier_name in sorted(pos_valid["tier"].dropna().unique()):
                            tier_pos = pos_valid[pos_valid["tier"] == tier_name]
                            tier_train = tier_pos[tier_pos["split"] == "train"]
                            tier_val = tier_pos[tier_pos["split"] == "val"]
                            if len(tier_train) > 0 and len(tier_val) > 0:
                                t_hp = set(tier_train["hp_recomputed"].unique())
                                v_hp = set(tier_val["hp_recomputed"].unique())
                                t_v_overlap = t_hp & v_hp
                                print(f"  Tier {tier_name}: train={len(tier_train)} ({len(t_hp)} pixels), "
                                      f"val={len(tier_val)} ({len(v_hp)} pixels), "
                                      f"overlap={len(t_v_overlap)} pixels")
                                report[f"tier_{tier_name}_overlap"] = {
                                    "train_count": len(tier_train),
                                    "val_count": len(tier_val),
                                    "train_pixels": len(t_hp),
                                    "val_pixels": len(v_hp),
                                    "overlapping_pixels": len(t_v_overlap),
                                }

            elif not HAS_HEALPY:
                print("\n  WARNING: healpy not installed, cannot recompute HEALPix.")
                print("  Install with: pip install healpy")
                report["healpy_available"] = False

        else:
            print("\n  ALL positives have NaN ra/dec — cannot assess spatial distribution.")
            report["all_positives_missing_radec"] = True

    else:
        print("\n  ra/dec columns missing from manifest!")
        report["radec_columns_missing"] = True

    # --- 6. Check how split was assigned ---
    print(f"\n=== 6. Split assignment analysis ===")
    if has_healpix:
        # For negatives, check if split correlates with healpix
        neg_valid = neg.dropna(subset=["healpix_128"])
        if len(neg_valid) > 0:
            # Sample check: are all negatives with same healpix in same split?
            hp_split_map = neg_valid.groupby("healpix_128")["split"].nunique()
            mixed_hp = (hp_split_map > 1).sum()
            print(f"  Negatives: {len(neg_valid)} with valid healpix")
            print(f"  HEALPix pixels with negatives in BOTH splits: {mixed_hp}/{len(hp_split_map)}")
            report["negative_mixed_healpix_pixels"] = int(mixed_hp)
            report["negative_total_healpix_pixels"] = len(hp_split_map)

            if mixed_hp == 0:
                print("  => Negatives appear spatially split (each pixel in one split only)")
                report["negative_split_mechanism"] = "spatial"
            else:
                pct_mixed = mixed_hp / len(hp_split_map) * 100
                print(f"  => {pct_mixed:.1f}% of pixels have negatives in both splits")
                report["negative_split_mechanism"] = "mixed_or_random"

    # Save
    out_path = os.path.join(args.out_dir, "healpix_investigation.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
