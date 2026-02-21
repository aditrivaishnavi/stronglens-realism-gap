#!/usr/bin/env python3
"""
Masked Pixel Diagnostic: Check NaN/zero/non-finite pixel fractions in cutouts.

"Preprocessing replaces non-finite values with zero.
That is acceptable only if masked pixels are rare and random. Store and
propagate a mask or drop stamps with high masked fraction."

This script samples cutouts from the manifest and reports:
  - Fraction of non-finite (NaN/Inf) pixels per stamp
  - Fraction of exactly-zero pixels per stamp
  - Flags stamps with >5% non-finite pixels
  - Summary statistics

Usage:
    cd stronglens-realism-gap
    export PYTHONPATH=.

    python scripts/masked_pixel_diagnostic.py \\
        --manifest manifests/training_parity_70_30_v1.parquet \\
        --out-dir results/masked_pixels \\
        --n-samples 2000

Date: 2026-02-13
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Masked pixel diagnostic: check NaN/zero fractions in cutouts",
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="Flag cutouts with non-finite fraction above this (default: 0.05)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading manifest...")
    df = pd.read_parquet(args.manifest)
    n = min(args.n_samples, len(df))
    sample = df.sample(n=n, random_state=args.seed)
    print(f"Sampling {n} cutouts...")

    nonfinite_fracs = []
    zero_fracs = []
    flagged_paths = []
    errors = 0

    for i, (_, row) in enumerate(sample.iterrows()):
        try:
            with np.load(str(row["cutout_path"])) as z:
                hwc = z["cutout"].astype(np.float32)
        except Exception:
            errors += 1
            continue

        total_pix = hwc.size
        n_nonfinite = int(np.sum(~np.isfinite(hwc)))
        n_zero = int(np.sum(hwc == 0.0))

        frac_nf = n_nonfinite / total_pix
        frac_zero = n_zero / total_pix

        nonfinite_fracs.append(frac_nf)
        zero_fracs.append(frac_zero)

        if frac_nf > args.threshold:
            flagged_paths.append({
                "path": str(row["cutout_path"]),
                "nonfinite_frac": frac_nf,
                "label": int(row.get("label", -1)),
                "split": str(row.get("split", "?")),
            })

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}", end="\r")

    nf = np.array(nonfinite_fracs)
    zf = np.array(zero_fracs)

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_sampled": n,
        "n_loaded": len(nf),
        "n_errors": errors,
        "threshold": args.threshold,
        "nonfinite_pixels": {
            "mean_frac": float(np.mean(nf)) if len(nf) > 0 else float("nan"),
            "median_frac": float(np.median(nf)) if len(nf) > 0 else float("nan"),
            "max_frac": float(np.max(nf)) if len(nf) > 0 else float("nan"),
            "pct_above_threshold": float((nf > args.threshold).mean() * 100) if len(nf) > 0 else float("nan"),
            "n_above_threshold": int((nf > args.threshold).sum()) if len(nf) > 0 else 0,
        },
        "zero_pixels": {
            "mean_frac": float(np.mean(zf)) if len(zf) > 0 else float("nan"),
            "median_frac": float(np.median(zf)) if len(zf) > 0 else float("nan"),
            "max_frac": float(np.max(zf)) if len(zf) > 0 else float("nan"),
        },
        "flagged_cutouts": flagged_paths[:50],  # top 50
        "interpretation": {
            "if_most_zero": "Most masked pixels are at survey edges or bad columns.",
            "if_many_flagged": (
                "Stamps with >5% masked pixels may have biased normalization. "
                "Consider dropping or tracking a mask channel."
            ),
        },
    }

    json_path = os.path.join(args.out_dir, "masked_pixel_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {json_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("MASKED PIXEL DIAGNOSTIC")
    print("=" * 60)
    print(f"  Cutouts sampled: {len(nf)} (errors: {errors})")
    print(f"  Non-finite pixels:")
    print(f"    Mean fraction:   {np.mean(nf):.6f}")
    print(f"    Max fraction:    {np.max(nf):.6f}")
    print(f"    >{args.threshold*100:.0f}% non-finite: {int((nf > args.threshold).sum())} cutouts "
          f"({(nf > args.threshold).mean()*100:.1f}%)")
    print(f"  Zero pixels:")
    print(f"    Mean fraction:   {np.mean(zf):.6f}")
    print(f"    Max fraction:    {np.max(zf):.6f}")
    if flagged_paths:
        print(f"\n  Flagged cutouts (>{args.threshold*100:.0f}% non-finite):")
        for fp in flagged_paths[:10]:
            print(f"    {fp['path']}: {fp['nonfinite_frac']*100:.1f}% (label={fp['label']})")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
