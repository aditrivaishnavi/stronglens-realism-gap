"""
McNemar test on bright-arc paired injection data.

For each of 8 source-magnitude bins (200 paired injections per bin),
each injection is scored under two conditions: baseline (no Poisson)
and Poisson noise (gain=150). The detection threshold is p > 0.3.

The data provides per-bin counts of:
  - "gained": detected with Poisson but NOT without (discordant pair type b)
  - "lost":   detected without Poisson but NOT with   (discordant pair type c)

McNemar's chi-squared = (b - c)^2 / (b + c), df=1.

We compute:
  1. Per-bin McNemar tests (where b + c > 0)
  2. Aggregate over the three bins where Poisson helps (source mag 20-23)
  3. Full aggregate over all 8 bins
"""

import json
import sys
from pathlib import Path
from scipy.stats import chi2

DATA = Path(__file__).resolve().parent.parent / (
    "results/D06_corrected_priors/poisson_diagnostics/"
    "d06_poisson_diagnostics.json"
)
OUT = DATA.parent / "mcnemar_results.json"


def mcnemar_chi2(b, c):
    """Return (chi2_stat, p_value) for McNemar's test with continuity correction."""
    if b + c == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)  # Edwards continuity correction
    p = chi2.sf(stat, df=1)
    return float(stat), float(p)


def mcnemar_exact(b, c):
    """Return exact (mid-p) McNemar p-value via binomial test."""
    from scipy.stats import binom
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_exact = 2 * binom.cdf(k, n, 0.5)
    return float(min(p_exact, 1.0))


def main():
    with open(DATA) as f:
        diag = json.load(f)

    bins = diag["paired_delta"]["results_by_bin"]

    results = {"per_bin": {}, "aggregate_20_23": {}, "aggregate_all": {}}

    total_gained = 0
    total_lost = 0
    bright_gained = 0
    bright_lost = 0

    for mag_bin in sorted(bins.keys()):
        d = bins[mag_bin]
        b = d["n_gained_above_03"]
        c = d["n_lost_below_03"]

        n = d["n_paired"]
        det_base = round(n * d["baseline_detection_rate"])
        det_pois = round(n * d["poisson_detection_rate"])
        both_det = det_base - c
        both_undet = n - both_det - b - c

        chi2_stat, p_val = mcnemar_chi2(b, c)
        p_exact = mcnemar_exact(b, c)

        results["per_bin"][mag_bin] = {
            "n_paired": n,
            "gained_b": b,
            "lost_c": c,
            "both_detected": both_det,
            "both_undetected": both_undet,
            "mcnemar_chi2": round(chi2_stat, 4),
            "mcnemar_p_chi2": p_val,
            "mcnemar_p_exact": p_exact,
            "baseline_det_rate": d["baseline_detection_rate"],
            "poisson_det_rate": d["poisson_detection_rate"],
        }

        total_gained += b
        total_lost += c

        lo = int(mag_bin.split("-")[0])
        if 20 <= lo <= 22:
            bright_gained += b
            bright_lost += c

    chi2_bright, p_bright = mcnemar_chi2(bright_gained, bright_lost)
    p_bright_exact = mcnemar_exact(bright_gained, bright_lost)
    results["aggregate_20_23"] = {
        "bins_included": ["20-21", "21-22", "22-23"],
        "total_gained": bright_gained,
        "total_lost": bright_lost,
        "mcnemar_chi2": round(chi2_bright, 4),
        "mcnemar_p_chi2": p_bright,
        "mcnemar_p_exact": p_bright_exact,
    }

    chi2_all, p_all = mcnemar_chi2(total_gained, total_lost)
    p_all_exact = mcnemar_exact(total_gained, total_lost)
    results["aggregate_all"] = {
        "bins_included": sorted(bins.keys()),
        "total_gained": total_gained,
        "total_lost": total_lost,
        "mcnemar_chi2": round(chi2_all, 4),
        "mcnemar_p_chi2": p_all,
        "mcnemar_p_exact": p_all_exact,
    }

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results written to {OUT}")
    print()

    print("=== Per-bin McNemar results ===")
    print(f"{'Bin':>8s}  {'Gained':>6s}  {'Lost':>4s}  {'chi2':>8s}  {'p(chi2)':>10s}  {'p(exact)':>10s}")
    for mag_bin in sorted(results["per_bin"].keys()):
        r = results["per_bin"][mag_bin]
        print(f"{mag_bin:>8s}  {r['gained_b']:6d}  {r['lost_c']:4d}  "
              f"{r['mcnemar_chi2']:8.3f}  {r['mcnemar_p_chi2']:10.6f}  "
              f"{r['mcnemar_p_exact']:10.6f}")

    print()
    a = results["aggregate_20_23"]
    print(f"=== Aggregate source mag 20-23 ===")
    print(f"  Gained={a['total_gained']}, Lost={a['total_lost']}")
    print(f"  chi2={a['mcnemar_chi2']:.3f}, p(chi2)={a['mcnemar_p_chi2']:.6f}, "
          f"p(exact)={a['mcnemar_p_exact']:.6f}")

    print()
    a = results["aggregate_all"]
    print(f"=== Aggregate all bins ===")
    print(f"  Gained={a['total_gained']}, Lost={a['total_lost']}")
    print(f"  chi2={a['mcnemar_chi2']:.3f}, p(chi2)={a['mcnemar_p_chi2']:.6f}, "
          f"p(exact)={a['mcnemar_p_exact']:.6f}")


if __name__ == "__main__":
    main()
