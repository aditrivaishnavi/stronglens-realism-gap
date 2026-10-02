#!/usr/bin/env python3
"""Generate a score-blind illustration from the final P_src artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTATION_ROOT = (
    ROOT / "results_v2/canonical_test_host_repair_v1/psrc/expectations"
)
PHOTON_ROOT = ROOT / "results_v2/canonical_test_host_repair_v1/psrc/photons"
OUTPUT_PDF = ROOT / "manuscript/figures/section3_final_psrc_construction.pdf"
OUTPUT_PNG = ROOT / "manuscript/figures/section3_final_psrc_construction.png"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rgb_pair(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([first.reshape(-1, 3), second.reshape(-1, 3)])
    ordered = stacked[:, [2, 1, 0]]
    floor = float(np.nanpercentile(ordered, 1.0))
    ceiling = float(np.nanpercentile(ordered, 99.8))
    scale = max(ceiling - floor, 1e-8)

    def transform(image: np.ndarray) -> np.ndarray:
        value = np.maximum((image[:, :, [2, 1, 0]] - floor) / scale, 0.0)
        return np.clip(np.arcsinh(8.0 * value) / np.arcsinh(8.0), 0.0, 1.0)

    return transform(first), transform(second)


def main() -> None:
    expectations = pd.read_parquet(EXPECTATION_ROOT / "manifest.parquet")
    expectations["illustration_flux"] = expectations[
        ["structured_lensed_flux_r", "sersic_lensed_flux_r"]
    ].min(axis=1)
    selected = expectations.sort_values(
        ["illustration_flux", "pair_id"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    expectation_path = EXPECTATION_ROOT / str(selected["expectation_relpath"])
    if sha256(expectation_path) != str(selected["expectation_sha256"]):
        raise RuntimeError("selected expectation artifact hash differs")

    photons = pd.read_parquet(PHOTON_ROOT / "manifest.parquet")
    photon_row = photons.loc[
        photons["pair_id"].eq(selected["pair_id"]) & photons["draw_index"].eq(0)
    ]
    if len(photon_row) != 1:
        raise RuntimeError("selected pair does not have exactly one draw-zero row")
    photon_row = photon_row.iloc[0]
    photon_path = PHOTON_ROOT / str(photon_row["photon_pair_relpath"])
    if sha256(photon_path) != str(photon_row["photon_pair_sha256"]):
        raise RuntimeError("selected photon artifact hash differs")

    with np.load(expectation_path, allow_pickle=False) as archive:
        structured_arc = np.asarray(archive["structured_expectation"])
        sersic_arc = np.asarray(archive["sersic_expectation"])
        structured_expectation = np.asarray(archive["structured_host_plus_expectation"])
        sersic_expectation = np.asarray(archive["sersic_host_plus_expectation"])
    with np.load(photon_path, allow_pickle=False) as archive:
        structured_photon = np.asarray(archive["structured_cutouts"])[0]
        sersic_photon = np.asarray(archive["sersic_cutouts"])[0]

    crop = np.s_[17:84, 17:84]
    arc_max = float(
        max(
            np.nanpercentile(structured_arc[crop][:, :, 1], 99.8),
            np.nanpercentile(sersic_arc[crop][:, :, 1], 99.8),
        )
    )
    expectation_rgb = rgb_pair(structured_expectation[crop], sersic_expectation[crop])
    photon_rgb = rgb_pair(structured_photon[crop], sersic_photon[crop])

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    figure, axes = plt.subplots(2, 3, figsize=(7.15, 4.55))
    rows = [
        (
            "Structured",
            structured_arc[crop][:, :, 1],
            expectation_rgb[0],
            photon_rgb[0],
        ),
        (
            "Single-Sérsic twin",
            sersic_arc[crop][:, :, 1],
            expectation_rgb[1],
            photon_rgb[1],
        ),
    ]
    for row_index, (label, arc, expectation, photon) in enumerate(rows):
        axes[row_index, 0].imshow(
            arc,
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=arc_max,
        )
        axes[row_index, 1].imshow(expectation, origin="lower")
        axes[row_index, 2].imshow(photon, origin="lower")
        axes[row_index, 0].set_ylabel(label, fontweight="bold")
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
    for axis, title in zip(
        axes[0],
        [
            r"Lensed source expectation ($r$ band)",
            r"Target + source expectation ($z/r/g$)",
            r"Final detector input, draw 0 ($z/r/g$)",
        ],
        strict=True,
    ):
        axis.set_title(title, fontsize=8.3)
    figure.suptitle(
        "Final paired source construction: same injection target, lens, PSF, and intrinsic flux",
        fontsize=10.2,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        (
            "Score-blind display selection: maximum of the smaller pre-score "
            "lensed r-band fluxes; central 67×67-pixel crop"
        ),
        ha="center",
        fontsize=7.4,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.075,
        top=0.89,
        wspace=0.035,
        hspace=0.045,
    )
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    figure.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(
        {
            "pair_id": str(selected["pair_id"]),
            "source_id": str(selected["source_id"]),
            "expectation_sha256": sha256(expectation_path),
            "photon_sha256": sha256(photon_path),
            "output_pdf_sha256": sha256(OUTPUT_PDF),
            "output_png_sha256": sha256(OUTPUT_PNG),
        }
    )


if __name__ == "__main__":
    main()
