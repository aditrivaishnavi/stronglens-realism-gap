#!/usr/bin/env python3
"""Generate the manuscript cohort-flow figure from committed summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "manuscript/figures/section2_cohort_flow.pdf"
INK = "black"
LIGHT_GREY = "0.93"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "pdf.fonttype": 42,
    }
)


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def node(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "white",
    linestyle: str | tuple[float, tuple[float, ...]] = "solid",
    fontsize: float = 7.2,
) -> None:
    axis.add_patch(
        Rectangle(
            xy,
            width,
            height,
            linewidth=0.8,
            edgecolor=INK,
            facecolor=facecolor,
            linestyle=linestyle,
            joinstyle="miter",
        )
    )
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        linespacing=1.12,
    )


def arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": INK,
            "linewidth": 0.75,
            "mutation_scale": 7,
            "shrinkA": 1.5,
            "shrinkB": 1.5,
        },
    )


def main() -> None:
    flow = load("results/populations/sample_flow.json")
    provenance = load("results/populations/lens_provenance_summary.json")
    split = load("data/manifests/split_summary.json")
    lenscat = flow["lenscat"]
    status = lenscat["tier_a_scientific_status"]
    if status != provenance["scientific_status"]:
        raise ValueError("sample-flow and provenance status counts differ")
    clean = flow["clean_split"]
    roles = split["splits"]["test"]["endpoint_roles"]
    train = pd.read_parquet(ROOT / "data/manifests/train.parquet")
    validation = pd.read_parquet(ROOT / "data/manifests/validation.parquet")
    positive_train = train.loc[train["label"].eq(1)]
    positive_validation = validation.loc[validation["label"].eq(1)]
    candidate_train = int(positive_train["tier"].eq("A").sum())
    candidate_validation = int(positive_validation["tier"].eq("A").sum())
    tier_b_train = int(positive_train["tier"].eq("B").sum())
    tier_b_validation = int(positive_validation["tier"].eq("B").sum())
    candidate_labels = candidate_train + candidate_validation
    tier_b_labels = tier_b_train + tier_b_validation

    figure, axis = plt.subplots(figsize=(7.1, 6.0))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.01,
        0.975,
        r"$\bf{(a)}$ Catalogue provenance and object-level status",
        ha="left",
        va="top",
        fontsize=8.5,
    )
    node(axis, (0.02, 0.835), 0.13, 0.075, f"Lenscat entries\n{lenscat['desi_named_entries']:,}")
    node(
        axis,
        (0.195, 0.835),
        0.15,
        0.075,
        f"DR10 matches\n{lenscat['dr10_tractor_matches_within_5_arcsec']:,}",
    )
    node(axis, (0.39, 0.875), 0.12, 0.065, f"Tier-A grade\n{lenscat['tier_a']:,}")
    node(axis, (0.39, 0.745), 0.12, 0.065, f"Tier-B grade\n{lenscat['tier_b']:,}")
    arrow(axis, (0.15, 0.8725), (0.195, 0.8725))
    arrow(axis, (0.345, 0.882), (0.39, 0.9075))
    arrow(axis, (0.345, 0.863), (0.39, 0.7775))
    node(
        axis,
        (0.57, 0.86),
        0.40,
        0.095,
        "Object-level status\n"
        f"{status['confirmed_lens']} confirmed  |  {status['candidate']} candidate\n"
        f"{status['confirmed_nonlens']} non-lens  |  {status['uncertain']} uncertain",
        fontsize=6.8,
    )
    arrow(axis, (0.51, 0.9075), (0.57, 0.9075))
    node(
        axis,
        (0.57, 0.735),
        0.29,
        0.065,
        f"Unreviewed Tier-B\ncatalogue labels: {lenscat['tier_b']:,}",
        linestyle=(0, (3, 2)),
        fontsize=6.9,
    )
    arrow(axis, (0.51, 0.7775), (0.57, 0.7675))
    axis.plot([0.01, 0.99], [0.685, 0.685], color=INK, linewidth=0.45)

    axis.text(
        0.01,
        0.655,
        r"$\bf{(b)}$ Positive-label roles",
        ha="left",
        va="top",
        fontsize=8.5,
    )
    node(
        axis,
        (0.02, 0.535),
        0.16,
        0.07,
        f"Confirmed lenses\n{status['confirmed_lens']}",
        facecolor=LIGHT_GREY,
    )
    node(
        axis,
        (0.245, 0.525),
        0.25,
        0.09,
        "Excluded from development of\nthe reported primary detector\n"
        f"{status['confirmed_lens']}",
        facecolor=LIGHT_GREY,
        fontsize=6.3,
    )
    node(
        axis,
        (0.62, 0.57),
        0.30,
        0.065,
        f"Primary evaluation: {roles['terminal_confirmed']}",
        fontsize=6.8,
    )
    node(
        axis,
        (0.62, 0.48),
        0.30,
        0.065,
        f"Exploratory cohort: {roles['development_contacted_confirmed']}",
        fontsize=6.8,
    )
    arrow(axis, (0.18, 0.57), (0.245, 0.57))
    axis.plot([0.495, 0.56], [0.57, 0.57], color=INK, linewidth=0.75)
    axis.plot([0.56, 0.56], [0.5125, 0.6025], color=INK, linewidth=0.75)
    arrow(axis, (0.56, 0.6025), (0.62, 0.6025))
    arrow(axis, (0.56, 0.5125), (0.62, 0.5125))
    node(
        axis,
        (0.02, 0.39),
        0.22,
        0.085,
        f"Eligible noisy labels\n{candidate_labels:,} candidates\n"
        f"{tier_b_labels:,} Tier-B",
        linestyle=(0, (3, 2)),
        fontsize=6.9,
    )
    node(
        axis,
        (0.34, 0.39),
        0.59,
        0.085,
        "Primary-detector mixtures\n"
        f"Candidate-only: {candidate_train:,} train; "
        f"{candidate_validation:,} validation ({candidate_labels:,} total)\n"
        f"Candidate + Tier-B: {len(positive_train):,} train; "
        f"{len(positive_validation):,} validation "
        f"({len(positive_train) + len(positive_validation):,} total)",
        linestyle=(0, (3, 2)),
        fontsize=6.6,
    )
    arrow(axis, (0.24, 0.4325), (0.34, 0.4325))
    axis.text(
        0.34,
        0.365,
        "Dashed outlines denote catalogue labels used for fitting, not confirmed lenses.",
        ha="left",
        va="top",
        fontsize=6.6,
    )
    axis.plot([0.01, 0.99], [0.325, 0.325], color=INK, linewidth=0.45)

    axis.text(
        0.01,
        0.295,
        r"$\bf{(c)}$ Analysis dataset",
        ha="left",
        va="top",
        fontsize=8.5,
    )
    node(axis, (0.02, 0.145), 0.17, 0.08, f"Negatives\n{flow['negative_sample']['clean_negatives']:,}")
    node(axis, (0.29, 0.125), 0.18, 0.085, f"Training\n{clean['train_negatives']:,} negatives")
    node(
        axis,
        (0.54, 0.125),
        0.18,
        0.085,
        f"Validation\n{clean['validation_negatives']:,} negatives",
    )
    test_total = clean["test_negatives"] + clean["test_confirmed_lenses"]
    node(
        axis,
        (0.79, 0.11),
        0.19,
        0.11,
        f"Test\n{clean['test_negatives']:,} negatives\n"
        f"+ {clean['test_confirmed_lenses']} confirmed lenses\n{test_total:,} total",
        facecolor=LIGHT_GREY,
        fontsize=6.9,
    )
    axis.plot([0.19, 0.245], [0.185, 0.185], color=INK, linewidth=0.75)
    axis.plot([0.245, 0.245], [0.185, 0.255], color=INK, linewidth=0.75)
    axis.plot([0.245, 0.885], [0.255, 0.255], color=INK, linewidth=0.75)
    arrow(axis, (0.38, 0.255), (0.38, 0.21))
    arrow(axis, (0.63, 0.255), (0.63, 0.21))
    arrow(axis, (0.885, 0.255), (0.885, 0.22))
    axis.text(
        0.02,
        0.075,
        "No confirmed lens is used to develop the reported primary detector.",
        ha="left",
        va="center",
        fontsize=6.8,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
