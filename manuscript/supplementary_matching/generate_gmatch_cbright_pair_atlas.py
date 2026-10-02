#!/usr/bin/env python3
"""Generate a provenance-bound atlas of the 43 observed/matched-target pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "results_v2").is_dir() and (candidate / "manuscript").is_dir():
            return candidate
    raise RuntimeError("could not locate repository root")


ROOT = repository_root()
GMATCH = (
    ROOT
    / "results_v2/canonical_test_host_repair_v1/gmatch/preparation"
)
BRIGHTNESS = ROOT / "results_v2/canonical_test_host_repair_v1/revision_controls"
PROVENANCE = ROOT / "manuscript/tables/confirmed_43_provenance.csv"

MATCHES_SHA256 = "b16127143b0c9c1e62ddd7829209f2dd47a5d6e04435cf71e025c146913373ee"
REFERENCE_ADAPTER_SHA256 = (
    "43ead962c7371d42c6a9ad3f4f75861b60821c815017f79c507e7de2c253a1fb"
)
PAIR_PLAN_SHA256 = "39f3b19e232f09aef40e8d6a107bf17ec80e655f410983cb40d7707e95218e2e"
BRIGHTNESS_DESIGN_SHA256 = (
    "c09bf29230064e4a6a306bb6a6aac015c70e7e0fcf889314e58c3fb72c53b7ef"
)
PROVENANCE_SHA256 = (
    "99b898da0be117b1102e923dabec02e11bf2624296a76242573f86e908843bcf"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {observed} != {expected}")


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def lupton_rgb(image_grz: np.ndarray, *, m: float = 0.03, q: float = 20.0) -> np.ndarray:
    """Apply the fixed Legacy Survey/Stein dr2_rgb channel mapping."""
    image = np.asarray(image_grz, dtype=np.float32).copy()
    if image.shape != (101, 101, 3):
        raise RuntimeError(f"unexpected cutout shape: {image.shape}")
    scales_grz = np.array([6.0, 3.4, 2.2], dtype=np.float32)
    affine_grz = image * scales_grz + m
    intensity = np.maximum(0.0, affine_grz[:, :, 0])
    intensity += np.maximum(0.0, affine_grz[:, :, 1])
    intensity += np.maximum(0.0, affine_grz[:, :, 2])
    intensity /= 3.0
    mapped = np.arcsinh(q * intensity) / np.sqrt(q)
    safe = intensity + (intensity == 0.0) * 1e-6
    rgb = affine_grz[:, :, [2, 1, 0]] * (mapped / safe)[:, :, None]
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def read_cutout(
    *,
    relpath: str,
    array_key: str,
    expected_file_sha256: str,
    expected_array_sha256: str,
) -> np.ndarray:
    path = ROOT / relpath
    require_sha256(path, expected_file_sha256)
    with np.load(path, allow_pickle=False) as archive:
        if array_key not in archive.files:
            raise RuntimeError(f"{path} lacks array {array_key}")
        image = np.asarray(archive[array_key], dtype=np.float32)
    if image.shape != (101, 101, 3) or not np.isfinite(image).all():
        raise RuntimeError(f"invalid cutout in {path}: {image.shape}")
    if array_sha256(image) != expected_array_sha256:
        raise RuntimeError(f"pixel-array SHA-256 mismatch: {path}:{array_key}")
    return image


def load_pairs() -> pd.DataFrame:
    matches_path = GMATCH / "matches.parquet"
    reference_path = GMATCH / "reference_adapter.parquet"
    pair_plan_path = GMATCH / "pair_plan.parquet"
    brightness_path = BRIGHTNESS / "brightness_design.parquet"
    require_sha256(matches_path, MATCHES_SHA256)
    require_sha256(reference_path, REFERENCE_ADAPTER_SHA256)
    require_sha256(pair_plan_path, PAIR_PLAN_SHA256)
    require_sha256(brightness_path, BRIGHTNESS_DESIGN_SHA256)
    require_sha256(PROVENANCE, PROVENANCE_SHA256)

    matches = pd.read_parquet(matches_path)
    references = pd.read_parquet(reference_path).rename(
        columns={
            "galaxy_id": "reference_id",
            "pixel_relpath": "reference_pixel_relpath",
            "pixel_file_sha256": "reference_pixel_file_sha256",
            "pixel_sha256": "reference_pixel_sha256",
            "pixel_array_key": "reference_pixel_array_key",
        }
    )
    bare = (
        pd.read_parquet(pair_plan_path)
        .loc[lambda frame: frame["arm"].eq("bare")]
        .rename(
            columns={
                "pixel_relpath": "target_pixel_relpath",
                "pixel_file_sha256": "target_pixel_file_sha256",
                "pixel_sha256": "target_pixel_sha256",
                "pixel_array_key": "target_pixel_array_key",
            }
        )
    )
    brightness = pd.read_parquet(brightness_path)
    expected_magnitudes = {18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5}
    brightness_pairs = brightness[
        ["pair_id", "reference_id", "target_galaxy_id"]
    ].drop_duplicates()
    grouped_magnitudes = brightness.groupby("pair_id")["target_source_r_mag"]
    counts = grouped_magnitudes.size()
    magnitude_sets = grouped_magnitudes.apply(lambda values: set(map(float, values)))
    if (
        len(brightness) != 344
        or len(brightness_pairs) != 43
        or not counts.eq(8).all()
        or any(values != expected_magnitudes for values in magnitude_sets)
    ):
        raise RuntimeError("brightness design is not the final 43 x 8 design")

    reference_columns = [
        "reference_id",
        "reference_pixel_relpath",
        "reference_pixel_file_sha256",
        "reference_pixel_sha256",
        "reference_pixel_array_key",
    ]
    bare_columns = [
        "pair_id",
        "reference_id",
        "target_galaxy_id",
        "theta_e_arcsec",
        "target_pixel_relpath",
        "target_pixel_file_sha256",
        "target_pixel_sha256",
        "target_pixel_array_key",
    ]
    pairs = matches.merge(
        references[reference_columns],
        on="reference_id",
        validate="one_to_one",
    ).merge(
        bare[bare_columns],
        on=["pair_id", "reference_id", "target_galaxy_id"],
        validate="one_to_one",
    )
    pairs = pairs.merge(
        brightness_pairs.assign(in_brightness_design=True),
        on=["pair_id", "reference_id", "target_galaxy_id"],
        validate="one_to_one",
    )

    names = pd.read_csv(PROVENANCE, usecols=["galaxy_id", "catalogue_name"]).rename(
        columns={"galaxy_id": "reference_id"}
    )
    pairs = pairs.merge(names, on="reference_id", validate="one_to_one")
    if (
        len(pairs) != 43
        or pairs["reference_id"].nunique() != 43
        or pairs["target_galaxy_id"].nunique() != 43
        or not pairs["in_brightness_design"].all()
        or not np.all(pairs["theta_e_arcsec"].to_numpy(dtype=float) == 1.5)
    ):
        raise RuntimeError("pair atlas does not bind exactly 43 final C_bright targets")
    return pairs.sort_values(["catalogue_name", "reference_id"]).reset_index(drop=True)


def cover_page(pdf: PdfPages, pairs: pd.DataFrame) -> None:
    figure = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    figure.text(
        0.07,
        0.89,
        "Observed strong lenses and their matched C_bright targets",
        fontsize=22,
        weight="bold",
        color="#111827",
    )
    figure.text(
        0.07,
        0.83,
        "Complete 43-pair atlas • exact retained pixel payloads",
        fontsize=13,
        color="#374151",
    )
    body = (
        "Each following page places the exact observed-reference cutout used in the "
        "final authoritative analysis beside the exact bare catalogue-negative "
        "target selected for that reference. The 43 targets are the final targets "
        "inherited by C_bright.\n\n"
        "The assignment is score-blind, without replacement, and globally optimized "
        "for maximum cardinality then minimum total robust-IQR distance. It is not 43 "
        "independent nearest-neighbour choices. Matching uses observed z magnitude, "
        "catalogue Sérsic index, axis ratio, r-band PSF FWHM, and r-band depth.\n\n"
        "All panels show the complete 101×101-pixel (≈26.5 arcsec) g/r/z cutout with "
        "the same fixed Legacy Survey/Stein dr2_rgb mapping (m=0.03, Q=20). There is "
        "no panel-specific crop or intensity stretch. The target is shown before "
        "source insertion; C_bright subsequently rescales the retained θE=1.5-arcsec "
        "source component on this target.\n\n"
        "Visual similarity is not a matching criterion. The atlas is a diagnostic of "
        "what the five-covariate matching does—and does not—control."
    )
    figure.text(
        0.07,
        0.72,
        body,
        fontsize=11.5,
        color="#1F2937",
        va="top",
        linespacing=1.5,
        wrap=True,
    )
    source_text = (
        f"Pairs: {len(pairs)}\n"
        f"matches.parquet SHA-256: {MATCHES_SHA256}\n"
        f"reference_adapter.parquet SHA-256: {REFERENCE_ADAPTER_SHA256}\n"
        f"pair_plan.parquet SHA-256: {PAIR_PLAN_SHA256}\n"
        f"brightness_design.parquet SHA-256: {BRIGHTNESS_DESIGN_SHA256}"
    )
    figure.text(
        0.07,
        0.24,
        source_text,
        fontsize=8.5,
        family="monospace",
        color="#4B5563",
        va="top",
        linespacing=1.45,
    )
    figure.text(
        0.07,
        0.07,
        "Generated directly from final authority bindings and retained pixel payloads.",
        fontsize=9,
        color="#6B7280",
    )
    pdf.savefig(figure)
    plt.close(figure)


def pair_page(pdf: PdfPages, row: object, index: int, total: int) -> None:
    observed = read_cutout(
        relpath=str(row.reference_pixel_relpath),
        array_key=str(row.reference_pixel_array_key),
        expected_file_sha256=str(row.reference_pixel_file_sha256),
        expected_array_sha256=str(row.reference_pixel_sha256),
    )
    target = read_cutout(
        relpath=str(row.target_pixel_relpath),
        array_key=str(row.target_pixel_array_key),
        expected_file_sha256=str(row.target_pixel_file_sha256),
        expected_array_sha256=str(row.target_pixel_sha256),
    )

    figure = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    grid = figure.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[4.7, 1.3],
        left=0.055,
        right=0.945,
        top=0.84,
        bottom=0.075,
        wspace=0.06,
        hspace=0.16,
    )
    observed_axis = figure.add_subplot(grid[0, 0])
    target_axis = figure.add_subplot(grid[0, 1])
    for axis, image in ((observed_axis, observed), (target_axis, target)):
        axis.imshow(lupton_rgb(image), origin="lower", interpolation="nearest")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#374151")
            spine.set_linewidth(1.0)

    observed_axis.set_title(
        f"Observed confirmed lens\n{row.catalogue_name}  •  {row.reference_id}",
        fontsize=11.5,
        weight="bold",
        pad=10,
    )
    target_axis.set_title(
        "Matched bare target inherited by C_bright\n"
        f"{row.target_galaxy_id}  •  before source insertion",
        fontsize=11.5,
        weight="bold",
        pad=10,
    )
    figure.text(
        0.055,
        0.94,
        f"Pair {index:02d} of {total}: exact observed and matched-target pixels",
        fontsize=17,
        weight="bold",
        color="#111827",
    )
    figure.text(
        0.945,
        0.945,
        f"global match distance = {float(row.distance):.3f}",
        fontsize=10,
        ha="right",
        color="#374151",
    )
    figure.text(
        0.945,
        0.915,
        f"admissible targets for this reference = {int(row.admissible_host_count):,}",
        fontsize=9,
        ha="right",
        color="#6B7280",
    )

    labels = [
        ("Observed z magnitude", "observed_zmag", ""),
        ("Sérsic index", "sersic_n", ""),
        ("Axis ratio", "axis_ratio_q", ""),
        ("r-band PSF FWHM", "psf_fwhm_r_arcsec", " arcsec"),
        ("r-band 5σ depth", "depth5_r_mag", " mag"),
    ]
    lines = [
        "Matched covariates"
        "                              observed       target    Δ(ref−target)/IQR"
    ]
    for label, stem, suffix in labels:
        reference = float(getattr(row, f"reference__{stem}"))
        matched = float(getattr(row, f"target__{stem}"))
        delta = float(getattr(row, f"delta_standardised__{stem}"))
        lines.append(
            f"{label:<29} {reference:>8.3f}{suffix:<7} "
            f"{matched:>8.3f}{suffix:<7} {delta:>+9.3f}"
        )
    metadata_axis = figure.add_subplot(grid[1, :])
    metadata_axis.axis("off")
    metadata_axis.text(
        0.0,
        1.0,
        "\n".join(lines),
        family="monospace",
        fontsize=8.8,
        va="top",
        color="#1F2937",
        linespacing=1.3,
    )
    metadata_axis.text(
        1.0,
        1.0,
        "Score-blind global assignment • no replacement\n"
        "Bare target shown; C_bright uses its retained θE=1.5″ parent scene\n"
        "Full 101×101 field • fixed g/r/z mapping • no panel-specific stretch",
        fontsize=8.8,
        ha="right",
        va="top",
        color="#4B5563",
        linespacing=1.5,
    )
    pdf.savefig(figure)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    script_directory = Path(__file__).resolve().parent
    default_output = (
        script_directory / "canonical_gmatch_cbright_confirmed_target_atlas.pdf"
        if script_directory.name == "supplementary_matching"
        else ROOT
        / "presentation/inspection/canonical_gmatch_cbright_confirmed_target_atlas.pdf"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
    )
    arguments = parser.parse_args()
    pairs = load_pairs()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(
        arguments.output,
        metadata={
            "Title": "Observed strong lenses and matched C_bright targets",
            "Subject": "Complete 43-pair retained-payload atlas",
            "Author": "Aditrivaishnavi Balaji",
        },
    ) as pdf:
        cover_page(pdf, pairs)
        for index, row in enumerate(pairs.itertuples(index=False), start=1):
            pair_page(pdf, row, index, len(pairs))

    receipt = {
        "status": "complete_final_43_pair_atlas",
        "pairs": 43,
        "pdf_pages": 44,
        "observed_payload_files_verified": 43,
        "matched_target_payload_files_verified": 43,
        "observed_pixel_arrays_verified": 43,
        "matched_target_pixel_arrays_verified": 43,
        "output_pdf": str(arguments.output.relative_to(ROOT)),
        "output_pdf_sha256": sha256_file(arguments.output),
        "display": {
            "mapping": "Legacy Survey/Stein dr2_rgb",
            "m": 0.03,
            "q": 20.0,
            "full_field": True,
            "panel_specific_crop": False,
            "panel_specific_stretch": False,
        },
        "authorities": {
            "matches.parquet": MATCHES_SHA256,
            "reference_adapter.parquet": REFERENCE_ADAPTER_SHA256,
            "pair_plan.parquet": PAIR_PLAN_SHA256,
            "brightness_design.parquet": BRIGHTNESS_DESIGN_SHA256,
        },
    }
    receipt_path = arguments.output.with_suffix(".json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {arguments.output} ({receipt['pdf_pages']} pages)")
    print(f"Wrote {receipt_path}")
    print(f"SHA-256 {receipt['output_pdf_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
