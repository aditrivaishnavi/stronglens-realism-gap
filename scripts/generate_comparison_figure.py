#!/usr/bin/env python3
"""
Generate Figure 5: Hand-picked galleries of real Tier-A lenses and
parametric Sérsic injections (D06 grid, no-Poisson baseline).

Layout: 2 panels, each 2 rows x 4 cols = 8 cutouts per panel.
  Panel (a): 8 real Tier-A lenses sorted by r-band magnitude.
  Panel (b): 8 parametric injections across three theta_E regimes
             (3 compact, 3 medium, 2 extended), sorted by theta_E then r-mag.

Cutouts are loaded from the gallery thumbnail PNGs (pre-rendered grz composites)
so the script runs locally without needing lambda3 or the model.

Usage:
    python scripts/generate_comparison_figure.py \
        --gallery-dir results/D06_corrected_priors/gallery \
        --out-dir paper

Date: 2026-02-18  (v5 — hand-picked 8+8, local-only)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

REAL_PICKS = [
    "REAL_070",  # p=0.49, r=16.30
    "REAL_075",  # p=1.00, r=16.56
    "REAL_051",  # p=1.00, r=18.17
    "REAL_017",  # p=0.93, r=18.49
    "REAL_086",  # p=0.15, r=18.88  (missed)
    "REAL_063",  # p=0.99, r=19.05
    "REAL_040",  # p=0.02, r=19.89  (missed)
    "REAL_107",  # p=0.47, r=21.12
]

INJ_PICKS = [
    "INJ_075_033",  # p=0.0002, r=16.46, theta_E=0.75
    "INJ_075_032",  # p=0.0002, r=18.63, theta_E=0.75
    "INJ_075_041",  # p=0.0001, r=21.30, theta_E=0.75
    "INJ_15_004",   # p=0.0003, r=18.04, theta_E=1.50
    "INJ_15_020",   # p=0.0002, r=19.77, theta_E=1.50
    "INJ_15_002",   # p=0.0002, r=20.45, theta_E=1.50
    "INJ_25_046",   # p=0.0473, r=17.93, theta_E=2.50
    "INJ_25_023",   # p=0.6913, r=19.27, theta_E=2.50 (detected)
]

N_COLS = 4


def load_gallery_meta(gallery_dir):
    with open(gallery_dir / "gallery_data.json") as f:
        data = json.load(f)
    uid_map = {}
    for item in data.get("real_lenses", []):
        uid_map[item["uid"]] = item
    for te_key, inj_list in data.get("injections", {}).items():
        for item in inj_list:
            uid_map[item["uid"]] = item
    return uid_map


def load_thumb(gallery_dir, uid_map, uid):
    meta = uid_map[uid]
    thumb_path = gallery_dir / meta["thumb"]
    img = np.array(Image.open(thumb_path).convert("RGB"))
    return img, meta


def format_score(score):
    if score >= 0.1:
        return f"{score:.2f}"
    if score >= 0.001:
        return f"{score:.3f}"
    return f"{score:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery-dir", type=str,
                        default="results/D06_corrected_priors/gallery")
    parser.add_argument("--out-dir", type=str, default="paper")
    args = parser.parse_args()

    gallery_dir = Path(args.gallery_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    uid_map = load_gallery_meta(gallery_dir)

    fig = plt.figure(figsize=(7.0, 7.0))
    gs = GridSpec(5, N_COLS, figure=fig,
                  height_ratios=[1, 1, 0.25, 1, 1],
                  hspace=0.35, wspace=0.10)

    # Panel (a): Real Tier-A lenses
    real_axes = {}
    for idx, uid in enumerate(REAL_PICKS):
        r = idx // N_COLS
        c = idx % N_COLS
        ax = fig.add_subplot(gs[r, c])
        real_axes[(r, c)] = ax
        img, meta = load_thumb(gallery_dir, uid_map, uid)
        ax.imshow(img, origin="lower", aspect="equal")
        sc = meta["score"]
        rmag = meta["r_mag"]
        ax.set_title(f"p={format_score(sc)},  r={rmag:.1f}",
                     fontsize=9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Panel (a) title
    fig.text(0.5, 0.93, "(a)  Real Tier-A strong lenses",
             fontsize=11, fontweight="bold", ha="center", va="bottom")

    # Spacer row
    for c in range(N_COLS):
        ax_sp = fig.add_subplot(gs[2, c])
        ax_sp.axis("off")

    # Panel (b): Injections
    for idx, uid in enumerate(INJ_PICKS):
        r = 3 + idx // N_COLS
        c = idx % N_COLS
        ax = fig.add_subplot(gs[r, c])
        img, meta = load_thumb(gallery_dir, uid_map, uid)
        ax.imshow(img, origin="lower", aspect="equal")
        sc = meta["score"]
        rmag = meta["r_mag"]
        te = meta.get("theta_e", "?")
        ax.set_title(f"p={format_score(sc)},  r={rmag:.1f}",
                     fontsize=9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(f"$\\theta_E={te}''$", fontsize=9, labelpad=2)

    # Panel (b) title
    fig.text(0.5, 0.48, "(b)  Parametric injections (D06 grid, no Poisson)",
             fontsize=11, fontweight="bold", ha="center", va="bottom")

    pdf_path = out_dir / "fig5_comparison.pdf"
    png_path = out_dir / "fig5_comparison.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(png_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Figure saved to: {pdf_path}")

    audit = {
        "version": "v5_handpicked_8x8",
        "description": (
            "Hand-picked 8 real Tier-A lenses + 8 parametric injections. "
            "Real lenses sorted by r-mag, include 2 missed (p<0.3) and 2 moderate. "
            "Injections span 3 theta_E regimes (0.75, 1.50, 2.50) with 1 detected. "
            "No pairing or brightness matching."
        ),
        "real_picks": REAL_PICKS,
        "injection_picks": INJ_PICKS,
    }
    audit_path = out_dir / "fig5_audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"Audit saved to: {audit_path}")


if __name__ == "__main__":
    main()
