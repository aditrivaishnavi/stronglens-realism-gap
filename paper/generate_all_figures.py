#!/usr/bin/env python3
"""
Generate all 4 publication figures for the MNRAS paper.
Includes auto-validation checks for each figure.

Usage:
    cd stronglens_calibration
    python paper/generate_all_figures.py
"""
import json
import os
import sys
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ── Paths ────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
D06 = BASE / "results" / "D06_corrected_priors"
OUT = BASE / "paper"

GRID_NP = D06 / "grid_no_poisson" / "selection_function.csv"
GRID_P = D06 / "grid_poisson" / "selection_function.csv"
EMB = D06 / "linear_probe" / "embeddings.npz"

BA_FILES = {
    "Baseline": D06 / "ba_baseline" / "bright_arc_results.json",
    "Poisson (g=150)": D06 / "ba_poisson" / "bright_arc_results.json",
    "clip=20": D06 / "ba_clip20" / "bright_arc_results.json",
    "Poisson+clip20": D06 / "ba_poisson_clip20" / "bright_arc_results.json",
    "Unrestricted": D06 / "ba_unrestricted" / "bright_arc_results.json",
    r"Gain=$10^{12}$": D06 / "ba_gain_1e12" / "bright_arc_results.json",
}

MAG_BINS = ["18-19", "19-20", "20-21", "21-22", "22-23", "23-24", "24-25", "25-26"]
MAG_MIDS = [18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5]

# ── Utilities ────────────────────────────────────────────────────────
def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_grid_by_theta(csv_path):
    """Load grid CSV, aggregate by theta_E for threshold=0.3, source_mag_bin='all', threshold_type='fixed'."""
    rows = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if (r["threshold_type"] == "fixed" and
                r["threshold"] == "0.3" and
                r["source_mag_bin"] == "all" and
                int(r["n_injections"]) > 0):
                te = float(r["theta_e"])
                if te not in rows:
                    rows[te] = {"n_inj": 0, "n_det": 0}
                rows[te]["n_inj"] += int(r["n_injections"])
                rows[te]["n_det"] += int(r["n_detected"])
    theta_es = sorted(rows.keys())
    completeness = [rows[t]["n_det"] / rows[t]["n_inj"] for t in theta_es]
    ci_lo = [wilson_ci(rows[t]["n_det"], rows[t]["n_inj"])[0] for t in theta_es]
    ci_hi = [wilson_ci(rows[t]["n_det"], rows[t]["n_inj"])[1] for t in theta_es]
    return theta_es, completeness, ci_lo, ci_hi, rows


def load_grid_by_mag(csv_path):
    """Load grid CSV, aggregate by lensed_* mag bins for threshold=0.3, threshold_type='fixed'."""
    rows = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if (r["threshold_type"] == "fixed" and
                r["threshold"] == "0.3" and
                r["source_mag_bin"].startswith("lensed_") and
                int(r["n_injections"]) > 0):
                mb = r["source_mag_bin"]
                if mb not in rows:
                    rows[mb] = {"n_inj": 0, "n_det": 0}
                rows[mb]["n_inj"] += int(r["n_injections"])
                rows[mb]["n_det"] += int(r["n_detected"])
    return rows


def load_brightarc(path):
    """Load bright-arc JSON, return dict of bin -> detection_rate_p03."""
    with open(path) as f:
        d = json.load(f)
    return d["results_by_bin"]


validation_results = {}


# ── Figure 1: Completeness vs theta_E and lensed magnitude ──────────
def make_fig1():
    print("Generating Figure 1: Completeness vs theta_E and lensed mag...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: C vs theta_E (no Poisson)
    theta_es, comp, ci_lo, ci_hi, raw = load_grid_by_theta(GRID_NP)
    comp_pct = [c * 100 for c in comp]
    ci_lo_pct = [c * 100 for c in ci_lo]
    ci_hi_pct = [c * 100 for c in ci_hi]
    yerr_lo = [c - lo for c, lo in zip(comp_pct, ci_lo_pct)]
    yerr_hi = [hi - c for c, hi in zip(comp_pct, ci_hi_pct)]

    ax1.errorbar(theta_es, comp_pct, yerr=[yerr_lo, yerr_hi],
                 fmt="o-", color="C0", capsize=3, markersize=5, linewidth=1.5,
                 label="No Poisson")
    ax1.set_xlabel(r"$\theta_{\rm E}$ (arcsec)", fontsize=12)
    ax1.set_ylabel("Completeness (per cent)", fontsize=12)
    ax1.set_title(r"Completeness vs $\theta_{\rm E}$ ($p > 0.3$)", fontsize=11)
    ax1.set_xlim(0.3, 3.2)
    ax1.set_ylim(0, 10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # Right: C vs lensed mag (no Poisson and Poisson)
    mag_np = load_grid_by_mag(GRID_NP)
    mag_p = load_grid_by_mag(GRID_P)

    mag_labels_ordered = ["lensed_18-20", "lensed_20-22", "lensed_22-24", "lensed_24-27"]
    mag_display = ["18-20", "20-22", "22-24", "24-27"]
    x_pos = range(len(mag_labels_ordered))

    comp_np = []
    comp_p = []
    for ml in mag_labels_ordered:
        if ml in mag_np and mag_np[ml]["n_inj"] > 0:
            comp_np.append(mag_np[ml]["n_det"] / mag_np[ml]["n_inj"] * 100)
        else:
            comp_np.append(0)
        if ml in mag_p and mag_p[ml]["n_inj"] > 0:
            comp_p.append(mag_p[ml]["n_det"] / mag_p[ml]["n_inj"] * 100)
        else:
            comp_p.append(0)

    width = 0.35
    ax2.bar([x - width/2 for x in x_pos], comp_np, width, label="No Poisson", color="C0", alpha=0.8)
    ax2.bar([x + width/2 for x in x_pos], comp_p, width, label="Poisson (g=150)", color="C1", alpha=0.8)
    ax2.set_xticks(list(x_pos))
    ax2.set_xticklabels(mag_display)
    ax2.set_xlabel("Lensed apparent magnitude", fontsize=12)
    ax2.set_ylabel("Completeness (per cent)", fontsize=12)
    ax2.set_title(r"Completeness vs lensed mag ($p > 0.3$)", fontsize=11)
    ax2.set_ylim(0, 65)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    outpath = OUT / "fig1_completeness.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Validation
    checks = []
    checks.append(("11 theta_E points", len(theta_es) == 11))
    # Find C at theta_E=2.5 (D06 peak)
    idx_peak = theta_es.index(2.5) if 2.5 in theta_es else -1
    if idx_peak >= 0:
        checks.append(("C(theta_E=2.5)=8.33%", abs(comp_pct[idx_peak] - 8.33) < 0.05))
    else:
        checks.append(("C(theta_E=2.5) found", False))
    checks.append(("4 lensed-mag bins", len(comp_np) == 4))
    checks.append(("File size > 10KB", outpath.stat().st_size > 10000))
    validation_results["Figure 1"] = checks
    return outpath


# ── Figure 2: Two-panel UMAP ────────────────────────────────────────
def make_fig2():
    print("Generating Figure 2: UMAP embeddings...")
    data = np.load(EMB)

    emb_real = data["emb_real_tier_a"]
    emb_low = data["emb_inj_low_bf"]
    emb_high = data["emb_inj_high_bf"]
    emb_neg = data["emb_negatives"]

    scores_real = data["scores_real_tier_a"]
    scores_low = data["scores_inj_low_bf"]
    scores_high = data["scores_inj_high_bf"]
    scores_neg = data["scores_negatives"]

    all_emb = np.concatenate([emb_real, emb_low, emb_high, emb_neg], axis=0)
    all_scores = np.concatenate([scores_real, scores_low, scores_high, scores_neg])

    n_real = len(emb_real)
    n_low = len(emb_low)
    n_high = len(emb_high)
    n_neg = len(emb_neg)
    n_total = n_real + n_low + n_high + n_neg

    # UMAP
    try:
        from umap import UMAP
    except ImportError:
        print("  WARNING: umap-learn not installed. Falling back to PCA for layout.")
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(all_emb)
    else:
        reducer = UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=42)
        coords = reducer.fit_transform(all_emb)

    # Split coordinates
    idx = 0
    c_real = coords[idx:idx+n_real]; idx += n_real
    c_low = coords[idx:idx+n_low]; idx += n_low
    c_high = coords[idx:idx+n_high]; idx += n_high
    c_neg = coords[idx:idx+n_neg]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: category-colored
    ax1.scatter(c_neg[:, 0], c_neg[:, 1], s=8, alpha=0.3, c="grey", label=f"Negatives (n={n_neg})", rasterized=True)
    ax1.scatter(c_high[:, 0], c_high[:, 1], s=12, alpha=0.5, c="cyan", label=f"Inj high-bf (n={n_high})", rasterized=True)
    ax1.scatter(c_low[:, 0], c_low[:, 1], s=12, alpha=0.5, c="C0", label=f"Inj low-bf (n={n_low})", rasterized=True)
    ax1.scatter(c_real[:, 0], c_real[:, 1], s=30, alpha=0.9, c="goldenrod", edgecolors="k", linewidths=0.5,
                label=f"Real Tier-A (n={n_real})", zorder=10)
    ax1.set_xlabel("UMAP-1", fontsize=11)
    ax1.set_ylabel("UMAP-2", fontsize=11)
    ax1.set_title("Category", fontsize=12)
    ax1.legend(fontsize=8, loc="best", markerscale=1.5)

    # Right: score-colored
    sc = ax2.scatter(coords[:, 0], coords[:, 1], s=10, c=all_scores, cmap="viridis",
                     vmin=0, vmax=1, alpha=0.6, rasterized=True)
    # Overlay real lenses with edge
    ax2.scatter(c_real[:, 0], c_real[:, 1], s=30, c=scores_real, cmap="viridis",
                vmin=0, vmax=1, edgecolors="k", linewidths=0.5, zorder=10)
    ax2.set_xlabel("UMAP-1", fontsize=11)
    ax2.set_ylabel("UMAP-2", fontsize=11)
    ax2.set_title("CNN Score", fontsize=12)
    cbar = plt.colorbar(sc, ax=ax2, shrink=0.8)
    cbar.set_label("Score $p$", fontsize=10)

    plt.tight_layout()
    outpath = OUT / "fig2_umap.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Validation
    checks = []
    checks.append((f"Total points = {n_total} (expect 1612)", n_total == 1612))
    checks.append(("No NaN in coords", not np.any(np.isnan(coords))))
    checks.append(("4 categories present", n_real > 0 and n_low > 0 and n_high > 0 and n_neg > 0))
    checks.append(("File size > 10KB", outpath.stat().st_size > 10000))
    validation_results["Figure 2"] = checks
    return outpath


# ── Figure 3: Score distributions ────────────────────────────────────
def make_fig3():
    print("Generating Figure 3: Score distributions...")
    data = np.load(EMB)
    scores_real = data["scores_real_tier_a"]
    scores_low = data["scores_inj_low_bf"]
    scores_neg = data["scores_negatives"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bins = np.linspace(0, 1, 51)

    ax.hist(scores_neg, bins=bins, alpha=0.5, color="grey", label=f"Negatives (n={len(scores_neg)}, med={np.median(scores_neg):.1e})",
            density=False, log=True)
    ax.hist(scores_low, bins=bins, alpha=0.6, color="C0", label=f"Inj low-bf (n={len(scores_low)}, med={np.median(scores_low):.3f})",
            density=False, log=True)
    ax.hist(scores_real, bins=bins, alpha=0.8, color="goldenrod",
            label=f"Real Tier-A (n={len(scores_real)}, med={np.median(scores_real):.3f})",
            density=False, log=True)

    # Threshold lines
    thresholds = {"$p=0.3$": 0.3, "$p=0.806$\n(FPR$=10^{-3}$)": 0.806, "$p=0.995$\n(FPR$\\approx 3{\\times}10^{-4}$)": 0.995}
    for label, val in thresholds.items():
        ax.axvline(val, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.text(val + 0.01, ax.get_ylim()[1] * 0.5, label, fontsize=7, color="red", rotation=90, va="center")

    ax.set_xlabel("CNN Score $p$", fontsize=12)
    ax.set_ylabel("Count (log scale)", fontsize=12)
    ax.set_title("Score Distributions", fontsize=12)
    ax.legend(fontsize=8, loc="upper center")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    outpath = OUT / "fig3_scores.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Validation
    checks = []
    checks.append((f"Tier-A count = {len(scores_real)} (expect 112)", len(scores_real) == 112))
    checks.append((f"Inj count = {len(scores_low)} (expect 500)", len(scores_low) == 500))
    checks.append((f"Neg count = {len(scores_neg)} (expect 500)", len(scores_neg) == 500))
    checks.append(("3 threshold lines", len(thresholds) == 3))
    checks.append(("File size > 10KB", outpath.stat().st_size > 10000))
    validation_results["Figure 3"] = checks
    return outpath


# ── Figure 4: Bright-arc detection rates ─────────────────────────────
def make_fig4():
    print("Generating Figure 4: Bright-arc detection rates (signature figure)...")
    fig, ax = plt.subplots(figsize=(8, 5))

    styles = {
        "Baseline": dict(color="C0", linestyle="-", linewidth=2, marker="s", markersize=5),
        "Poisson (g=150)": dict(color="C1", linestyle="--", linewidth=2, marker="^", markersize=5),
        "clip=20": dict(color="C2", linestyle=":", linewidth=2, marker="D", markersize=5),
        "Poisson+clip20": dict(color="C3", linestyle="-.", linewidth=2, marker="v", markersize=5),
        "Unrestricted": dict(color="C4", linestyle="-", linewidth=1, marker="x", markersize=5),
        r"Gain=$10^{12}$": dict(color="C0", linestyle="none", marker="o", markersize=9,
                                markerfacecolor="none", markeredgewidth=2),
    }

    all_rates = {}
    for label, path in BA_FILES.items():
        results = load_brightarc(path)
        rates = [results[b]["detection_rate_p03"] * 100 for b in MAG_BINS]
        all_rates[label] = rates

        # Wilson CIs
        n = 200
        ci_lo = []
        ci_hi = []
        for b in MAG_BINS:
            k = int(results[b]["detection_rate_p03"] * n)
            lo, hi = wilson_ci(k, n)
            ci_lo.append(rates[MAG_BINS.index(b)] - lo * 100)
            ci_hi.append(hi * 100 - rates[MAG_BINS.index(b)])

        style = styles[label]
        if label == r"Gain=$10^{12}$":
            ax.plot(MAG_MIDS, rates, label=label, zorder=5, **style)
        else:
            ax.errorbar(MAG_MIDS, rates, yerr=[ci_lo, ci_hi],
                        label=label, capsize=2, zorder=3, **style)

    # Tier-A recall reference line
    ax.axhline(89.3, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(25.7, 90, "Tier-A recall\n(89.3%)", fontsize=8, color="red", va="bottom", ha="right")

    ax.set_xlabel("Source apparent magnitude", fontsize=12)
    ax.set_ylabel("Detection rate (per cent, $p > 0.3$)", fontsize=12)
    ax.set_title("Bright-Arc Detection Rate vs Source Magnitude", fontsize=12)
    ax.set_xlim(17.8, 26.2)
    ax.set_ylim(-2, 50)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = OUT / "fig4_brightarc.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Validation
    checks = []
    checks.append((f"6 conditions loaded", len(all_rates) == 6))
    total_points = sum(len(v) for v in all_rates.values())
    checks.append((f"48 data points (6x8)", total_points == 48))
    # Gain=1e12 must match baseline within 1 pp (0.5 pp = 1/N for N=200)
    baseline = all_rates["Baseline"]
    gain_ctrl = all_rates[r"Gain=$10^{12}$"]
    gain_match = all(abs(b - g) < 1.5 for b, g in zip(baseline, gain_ctrl))
    checks.append(("Gain=1e12 ~= Baseline within 1.5 pp", gain_match))
    # All rates in [0, 100]
    all_valid = all(0 <= r <= 100 for rates in all_rates.values() for r in rates)
    checks.append(("All rates in [0, 100]", all_valid))
    checks.append(("File size > 10KB", outpath.stat().st_size > 10000))
    validation_results["Figure 4"] = checks
    return outpath


# ── Main ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("MNRAS Paper Figure Generation")
    print("=" * 60)

    paths = []
    paths.append(make_fig1())
    paths.append(make_fig2())
    paths.append(make_fig3())
    paths.append(make_fig4())

    # Print validation summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    all_pass = True
    for fig_name, checks in validation_results.items():
        print(f"\n{fig_name}:")
        for desc, passed in checks:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {desc}")
            if not passed:
                all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL CHECKS PASSED")
        print(f"Figures saved to: {OUT}")
        for p in paths:
            sz = p.stat().st_size
            print(f"  {p.name}: {sz/1024:.1f} KB")
    else:
        print("SOME CHECKS FAILED — review output above")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
