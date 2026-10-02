#!/usr/bin/env python3
"""Generate the manuscript recovery figure from the final G_rec receipt."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "results_v2/canonical_test_host_repair_v1/grec/GREC_ANALYSIS.json"
)
OUTPUT = ROOT / "manuscript/figures/section5_grouped_recovery.pdf"


def main() -> None:
    result = json.loads(SOURCE.read_text(encoding="utf-8"))["grec"]
    radius = result["radius"]
    x = np.asarray([row["theta_e_arcsec"] for row in radius])
    y = 100.0 * np.asarray([row["estimate"] for row in radius])
    low = 100.0 * np.asarray([row["simultaneous_interval"][0] for row in radius])
    high = 100.0 * np.asarray([row["simultaneous_interval"][1] for row in radius])

    plt.rcParams.update({"font.size": 8.5})
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    ax.fill_between(
        x,
        low,
        high,
        color="#6baed6",
        alpha=0.28,
        linewidth=0,
        label="Simultaneous 95% band",
    )
    ax.plot(
        x,
        y,
        color="#08519c",
        marker="o",
        linewidth=1.6,
        markersize=3.8,
        label="Conditional recovery",
    )
    ax.axhline(0.0, color="0.65", linewidth=0.8)
    ax.set(
        xlabel=r"Einstein radius $\theta_{\rm E}$ (arcsec)",
        ylabel="Conditional recovery (per cent)",
        xlim=(0.45, 3.05),
    )
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
    )
    ax.legend(frameon=False, fontsize=7.2, loc="upper left")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
