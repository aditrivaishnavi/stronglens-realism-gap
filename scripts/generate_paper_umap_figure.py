#!/usr/bin/env python3
"""Generate publication-quality UMAP figure for the RASTI paper.

Produces a two-panel figure:
  Panel A: UMAP colored by category (Real Tier-A, low-bf injection, high-bf injection, negative)
  Panel B: UMAP colored by CNN score

Loads pre-computed UMAP coordinates (or recomputes from embeddings).

Usage:
    # From embeddings (recomputes UMAP):
    python scripts/generate_paper_umap_figure.py \
        --embeddings results/D06_corrected_priors/linear_probe/embeddings.npz \
        --out-dir paper/
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def load_embeddings_and_scores(embeddings_path):
    """Load embeddings and scores, handling both naming conventions."""
    data = np.load(embeddings_path, allow_pickle=True)
    keys = list(data.keys())

    if "emb_real" in keys:
        emb = {
            "real": data["emb_real"],
            "inj_low": data["emb_inj_low"],
            "inj_high": data["emb_inj_high"],
            "neg": data["emb_neg"],
        }
        scores = {
            "real": data.get("scores_real", None),
            "inj_low": data.get("scores_inj_low", None),
            "inj_high": data.get("scores_inj_high", None),
            "neg": data.get("scores_neg", None),
        }
    elif "emb_real_tier_a" in keys:
        emb = {
            "real": data["emb_real_tier_a"],
            "inj_low": data["emb_inj_low_bf"],
            "inj_high": data["emb_inj_high_bf"],
            "neg": data["emb_negatives"],
        }
        scores = {
            "real": data.get("scores_real_tier_a", None),
            "inj_low": data.get("scores_inj_low_bf", None),
            "inj_high": data.get("scores_inj_high_bf", None),
            "neg": data.get("scores_negatives", None),
        }
    else:
        raise KeyError(f"Unrecognized embedding keys: {keys}")

    return emb, scores


def main():
    ap = argparse.ArgumentParser(description="Generate paper UMAP figure")
    ap.add_argument("--umap-npz", default=None,
                    help="Path to pre-computed umap_coordinates.npz")
    ap.add_argument("--embeddings", required=True,
                    help="Path to embeddings.npz from feature_space_analysis.py")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load embeddings and scores
    emb, scores = load_embeddings_and_scores(args.embeddings)
    n_real = len(emb["real"])
    n_low = len(emb["inj_low"])
    n_high = len(emb["inj_high"])
    n_neg = len(emb["neg"])

    print(f"Loaded: {n_real} real, {n_low} low-bf, {n_high} high-bf, {n_neg} neg")

    # Get UMAP coordinates
    if args.umap_npz and os.path.exists(args.umap_npz):
        print(f"Loading pre-computed UMAP from: {args.umap_npz}")
        umap_data = np.load(args.umap_npz)
        coords_real = umap_data["coords_real"]
        coords_low = umap_data["coords_inj_low"]
        coords_high = umap_data["coords_inj_high"]
        coords_neg = umap_data["coords_neg"]
    else:
        if not HAS_UMAP:
            print("ERROR: umap-learn not installed and no pre-computed coordinates.")
            sys.exit(1)
        print("Computing UMAP from embeddings...")
        all_emb = np.vstack([emb["real"], emb["inj_low"], emb["inj_high"], emb["neg"]])
        reducer = umap.UMAP(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            random_state=args.seed,
            n_components=2,
        )
        umap_coords = reducer.fit_transform(all_emb)
        idx = 0
        coords_real = umap_coords[idx:idx + n_real]; idx += n_real
        coords_low = umap_coords[idx:idx + n_low]; idx += n_low
        coords_high = umap_coords[idx:idx + n_high]; idx += n_high
        coords_neg = umap_coords[idx:idx + n_neg]

    # Combine for score-colored plot
    all_coords = np.vstack([coords_real, coords_low, coords_high, coords_neg])
    all_scores = None
    if all(v is not None for v in scores.values()):
        all_scores = np.concatenate([
            scores["real"], scores["inj_low"], scores["inj_high"], scores["neg"]
        ])

    # =========================================================================
    # Two-panel figure
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Panel A: Category-colored ---
    ax1.scatter(
        coords_neg[:, 0], coords_neg[:, 1],
        c="#BBBBBB", s=6, alpha=0.25,
        label=f"Negatives ($n$={n_neg})",
        zorder=1, edgecolors="none", rasterized=True,
    )
    ax1.scatter(
        coords_high[:, 0], coords_high[:, 1],
        c="#00ACC1", s=20, alpha=0.55, marker="^",
        label=r"Injections, $\beta_{frac}$" + f" [0.7,1.0] ($n$={n_high})",
        zorder=2, edgecolors="none",
    )
    ax1.scatter(
        coords_low[:, 0], coords_low[:, 1],
        c="#1976D2", s=20, alpha=0.65, marker="o",
        label=r"Injections, $\beta_{frac}$" + f" [0.1,0.3] ($n$={n_low})",
        zorder=3, edgecolors="none",
    )
    ax1.scatter(
        coords_real[:, 0], coords_real[:, 1],
        c="#E65100", s=70, alpha=0.9, marker="*",
        label=f"Real Tier-A lenses ($n$={n_real})",
        zorder=4, edgecolors="black", linewidths=0.3,
    )

    ax1.set_xlabel("UMAP-1", fontsize=12)
    ax1.set_ylabel("UMAP-2", fontsize=12)
    ax1.set_title("(a) Feature space by category", fontsize=13, fontweight="bold")
    ax1.legend(loc="best", fontsize=9, framealpha=0.85, fancybox=True)
    ax1.tick_params(labelsize=10)

    # --- Panel B: Score-colored ---
    if all_scores is not None:
        sc = ax2.scatter(
            all_coords[:, 0], all_coords[:, 1],
            c=all_scores, cmap="RdYlGn", s=10, alpha=0.6,
            vmin=0, vmax=1, edgecolors="none", rasterized=True,
        )
        cbar = plt.colorbar(sc, ax=ax2, shrink=0.85, pad=0.02)
        cbar.set_label("CNN score $p$(lens)", fontsize=11)
        cbar.ax.tick_params(labelsize=9)
    else:
        ax2.text(0.5, 0.5, "Scores not available",
                 transform=ax2.transAxes, ha="center", va="center", fontsize=14)

    ax2.set_xlabel("UMAP-1", fontsize=12)
    ax2.set_ylabel("UMAP-2", fontsize=12)
    ax2.set_title("(b) Feature space by CNN score", fontsize=13, fontweight="bold")
    ax2.tick_params(labelsize=10)

    # Suptitle
    fig.suptitle(
        "UMAP projection of EfficientNetV2-S penultimate embeddings (1280-D)",
        fontsize=14, y=1.01,
    )

    fig.tight_layout()

    # Save
    png_path = os.path.join(args.out_dir, "paper_umap_two_panel.png")
    pdf_path = os.path.join(args.out_dir, "paper_umap_two_panel.pdf")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nTwo-panel figure saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")

    # =========================================================================
    # Also generate individual high-res panels for flexibility
    # =========================================================================

    # Panel A standalone
    fig_a, ax_a = plt.subplots(1, 1, figsize=(8, 7))
    ax_a.scatter(coords_neg[:, 0], coords_neg[:, 1], c="#BBBBBB", s=6, alpha=0.25,
                 label=f"Negatives ($n$={n_neg})", zorder=1, edgecolors="none", rasterized=True)
    ax_a.scatter(coords_high[:, 0], coords_high[:, 1], c="#00ACC1", s=20, alpha=0.55,
                 marker="^", label=r"Injections, $\beta_{frac}$" + f" [0.7,1.0] ($n$={n_high})",
                 zorder=2, edgecolors="none")
    ax_a.scatter(coords_low[:, 0], coords_low[:, 1], c="#1976D2", s=20, alpha=0.65,
                 marker="o", label=r"Injections, $\beta_{frac}$" + f" [0.1,0.3] ($n$={n_low})",
                 zorder=3, edgecolors="none")
    ax_a.scatter(coords_real[:, 0], coords_real[:, 1], c="#E65100", s=70, alpha=0.9,
                 marker="*", label=f"Real Tier-A lenses ($n$={n_real})",
                 zorder=4, edgecolors="black", linewidths=0.3)
    ax_a.set_xlabel("UMAP-1", fontsize=13)
    ax_a.set_ylabel("UMAP-2", fontsize=13)
    ax_a.set_title("CNN Feature Space: Real Lenses vs Parametric Injections", fontsize=14)
    ax_a.legend(loc="best", fontsize=10, framealpha=0.85)
    ax_a.tick_params(labelsize=11)
    fig_a.tight_layout()
    fig_a.savefig(os.path.join(args.out_dir, "paper_umap_category.pdf"), bbox_inches="tight")
    plt.close(fig_a)

    # Panel B standalone
    if all_scores is not None:
        fig_b, ax_b = plt.subplots(1, 1, figsize=(9, 7))
        sc_b = ax_b.scatter(all_coords[:, 0], all_coords[:, 1], c=all_scores,
                            cmap="RdYlGn", s=10, alpha=0.6, vmin=0, vmax=1,
                            edgecolors="none", rasterized=True)
        cbar_b = plt.colorbar(sc_b, ax=ax_b, shrink=0.85)
        cbar_b.set_label("CNN score $p$(lens)", fontsize=12)
        ax_b.set_xlabel("UMAP-1", fontsize=13)
        ax_b.set_ylabel("UMAP-2", fontsize=13)
        ax_b.set_title("CNN Score Distribution in Feature Space", fontsize=14)
        ax_b.tick_params(labelsize=11)
        fig_b.tight_layout()
        fig_b.savefig(os.path.join(args.out_dir, "paper_umap_scores.pdf"), bbox_inches="tight")
        plt.close(fig_b)

    print("\nAll paper figures generated.")


if __name__ == "__main__":
    main()
