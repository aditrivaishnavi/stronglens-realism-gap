#!/usr/bin/env python3
"""Generate final manuscript figures for the bounded revision repair."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "manuscript/figures"


def cell_radius_figure() -> None:
    table = pd.read_parquet(
        ROOT
        / "results_v2/canonical_test_host_repair_v1/revision_controls/"
        "grec_cell_radius_response.parquet"
    )
    cells = (
        table[["psf_bin", "depth_bin"]]
        .drop_duplicates()
        .sort_values(["psf_bin", "depth_bin"])
        .reset_index(drop=True)
    )
    fig, axes = plt.subplots(3, 3, figsize=(10.0, 8.0), sharex=True, sharey=True)
    for axis, row in zip(axes.flat, cells.itertuples(index=False), strict=True):
        frame = table[
            table["psf_bin"].eq(row.psf_bin)
            & table["depth_bin"].eq(row.depth_bin)
        ]
        axis.plot(frame["theta_e_arcsec"], 100 * frame["recovery"], marker="o")
        axis.set_title(f"PSF {row.psf_bin:.2f}, depth {row.depth_bin:.1f}")
        axis.grid(alpha=0.25)
    for index, axis in enumerate(axes.flat):
        if index // 3 == 2:
            axis.set_xlabel(r"$\theta_{\rm E}$ (arcsec)")
        if index % 3 == 0:
            axis.set_ylabel("Recovery (%)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "appendix_grec_cell_radius_response.pdf")
    plt.close(fig)


def brightness_figure() -> None:
    cells = pd.read_parquet(
        ROOT
        / "results_v2/canonical_test_host_repair_v1/revision_controls/"
        "brightness_cell_summary_clip10.parquet"
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True)
    for arm, linestyle in (("expectation", "-"), ("photon", "--")):
        frame = cells[cells["arm"].eq(arm)]
        axes[0].plot(
            frame["target_source_r_mag"],
            100 * frame["mean_detection_rate"],
            marker="o",
            linestyle=linestyle,
            label=arm,
        )
        axes[1].plot(
            frame["target_source_r_mag"],
            frame["mean_logit"],
            marker="o",
            linestyle=linestyle,
            label=arm,
        )
    axes[0].set_ylabel("Detection rate (%)")
    axes[1].set_ylabel("Mean native logit")
    for axis in axes:
        axis.set_xlabel(r"Intrinsic source $r$ magnitude")
        axis.invert_xaxis()
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / "appendix_brightness_response.pdf")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cell_radius_figure()
    brightness_figure()


if __name__ == "__main__":
    main()
