#!/usr/bin/env python3
"""Generate compact monochrome process diagrams for the manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

INK = "#111111"
RULE = "#555555"


def canvas(height: float) -> tuple[plt.Figure, plt.Axes]:
    figure, axis = plt.subplots(figsize=(7.2, height))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    return figure, axis


def node(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    fontsize: float = 6.8,
    linewidth: float = 0.9,
) -> None:
    axis.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor="white",
            edgecolor=INK,
            linewidth=linewidth,
        )
    )
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        linespacing=1.05,
    )


def arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    stop: tuple[float, float],
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            stop,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.1,
            color=INK,
            shrinkA=1,
            shrinkB=1,
        )
    )


def save(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    figure.savefig(
        output.with_suffix(".png"),
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)


def data_training_flow(output: Path) -> None:
    figure, axis = canvas(3.65)
    axis.text(
        0.01,
        0.97,
        "(a) Catalogue preparation and grouped roles",
        fontsize=8,
        weight="bold",
    )

    axis.text(0.005, 0.82, "Positive", fontsize=6.5, weight="bold", va="center")
    node(axis, 0.10, 0.76, 0.12, 0.12, "Lenscat\n4,788")
    node(axis, 0.29, 0.76, 0.17, 0.12, "Literature review\n389")
    node(axis, 0.53, 0.76, 0.12, 0.12, "Confirmed\n63")
    node(axis, 0.76, 0.84, 0.19, 0.08, "Selected endpoint: 43")
    node(axis, 0.76, 0.69, 0.19, 0.10, "Other confirmed\nrole: 20")
    arrow(axis, (0.22, 0.82), (0.29, 0.82))
    arrow(axis, (0.46, 0.82), (0.53, 0.82))
    arrow(axis, (0.65, 0.82), (0.76, 0.88))
    arrow(axis, (0.65, 0.82), (0.76, 0.74))

    axis.text(0.005, 0.58, "Negative", fontsize=6.5, weight="bold", va="center")
    node(axis, 0.10, 0.52, 0.12, 0.12, "Sampled\n453,100")
    node(axis, 0.29, 0.52, 0.17, 0.12, "Usable cutouts\n446,893")
    node(axis, 0.53, 0.52, 0.12, 0.12, "Final pool\n409,551")
    node(axis, 0.76, 0.50, 0.19, 0.16, "Train 286,679\nValidation 61,437\nTest 61,435")
    arrow(axis, (0.22, 0.58), (0.29, 0.58))
    arrow(axis, (0.46, 0.58), (0.53, 0.58))
    arrow(axis, (0.65, 0.58), (0.76, 0.58))
    axis.text(0.255, 0.61, "$-6,207$", fontsize=6.2, ha="center")
    axis.text(0.495, 0.61, "$-37,342$", fontsize=6.2, ha="center")

    axis.plot([0.01, 0.98], [0.43, 0.43], color=RULE, linewidth=0.7)
    axis.text(
        0.01,
        0.39,
        "(b) Detector protocols and operating points",
        fontsize=8,
        weight="bold",
    )

    axis.text(0.005, 0.28, "Primary", fontsize=6.5, weight="bold", va="center")
    node(axis, 0.12, 0.23, 0.14, 0.10, "Both candidate\nsources\nseed 42", fontsize=6.0)
    node(axis, 0.31, 0.23, 0.17, 0.10, "Terminal\nepoch 160")
    node(axis, 0.55, 0.23, 0.16, 0.10, "Threshold\n20 / 20,002")

    axis.text(0.005, 0.11, "Sensitivity", fontsize=6.5, weight="bold", va="center")
    node(axis, 0.12, 0.06, 0.14, 0.10, "12 runs")
    node(axis, 0.31, 0.06, 0.17, 0.10, "Best validation\nepochs 17--103")
    node(axis, 0.55, 0.06, 0.16, 0.10, "12 thresholds\n20 / 20,002 each")

    node(
        axis,
        0.79,
        0.10,
        0.20,
        0.19,
        "Same frozen populations\n12,512 negatives\n43 selected systems\n$G_{rec}$, $P_{noise}$, $P_{src}$",
        fontsize=6.1,
    )
    for y in (0.28, 0.11):
        arrow(axis, (0.26, y), (0.31, y))
        arrow(axis, (0.48, y), (0.55, y))
    arrow(axis, (0.71, 0.28), (0.79, 0.235))
    arrow(axis, (0.71, 0.11), (0.79, 0.155))
    save(figure, output)


def experiment_flow(output: Path) -> None:
    figure, axis = canvas(3.0)
    axis.text(0.02, 0.96, "Observed input", fontsize=7, weight="bold")
    axis.text(0.28, 0.96, "Controlled input", fontsize=7, weight="bold")
    axis.text(0.77, 0.96, "Scoring path", fontsize=7, weight="bold")

    centres = (0.82, 0.64, 0.46, 0.28, 0.10)
    inputs = (
        "$G_{domain}$ injection subset",
        "$G_{match}$ bare + injected",
        "$G_{rec}$ expectation pixels",
        "$P_{noise}$ photon-draw realizations",
        "$P_{src}$\nfixed target + paired source maps",
    )
    for centre, input_text in zip(centres, inputs, strict=True):
        node(axis, 0.28, centre - 0.055, 0.38, 0.11, input_text)

    for centre in centres[:2]:
        node(axis, 0.02, centre - 0.055, 0.18, 0.11, "43 systems")
        axis.text(0.24, centre, "+", ha="center", va="center", fontsize=10)
    for centre in centres[2:]:
        axis.text(0.11, centre, "—", ha="center", va="center", fontsize=10)

    node(axis, 0.77, 0.57, 0.20, 0.31, "Frozen encoders\n\nAUC\nmovement")
    node(axis, 0.77, 0.05, 0.20, 0.47, "Primary detector\n\nrecovery\nrisk differences")
    arrow(axis, (0.66, 0.82), (0.77, 0.80))
    arrow(axis, (0.66, 0.64), (0.77, 0.66))
    arrow(axis, (0.66, 0.46), (0.77, 0.43))
    arrow(axis, (0.66, 0.28), (0.77, 0.29))
    arrow(axis, (0.66, 0.10), (0.77, 0.15))
    save(figure, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "manuscript/figures",
    )
    arguments = parser.parse_args()
    data_training_flow(arguments.output_dir / "section2_data_training_flow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
