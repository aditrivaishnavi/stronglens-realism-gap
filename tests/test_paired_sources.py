"""Guards on paired COSMOS and fitted-Sersic source rendering.

The COSMOS sample is a large external download, so the tests that need it skip
when it is absent rather than fail.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

COSMOS_DIR = Path("data/downloads/cosmos_25_2/COSMOS_25.2_training_sample")
SOURCE_MANIFEST = Path(
    "data/derived/paired_source_noisy_diagnostics.parquet"
)

needs_cosmos = pytest.mark.skipif(
    not COSMOS_DIR.is_dir() or not SOURCE_MANIFEST.is_file(),
    reason="COSMOS sample is not present",
)


def test_support_mask_preserves_area_and_handles_rectangles() -> None:
    """248 of 265 catalogue stamps are rectangular, so squareness cannot be assumed."""
    from dhs.paired_sources import support_mask

    native = np.zeros((21, 31))
    native[2:19, 3:28] = 1.0  # a rectangle, deliberately not square
    expected = int((native != 0).sum())

    unrotated = support_mask(native, stamp_size=81, angle_rad=0.0)
    assert unrotated.sum() == expected
    assert unrotated[40, 40]
    assert not unrotated[40, 0]

    for angle in (np.pi / 6, np.pi / 4, 1.7, 4.3):
        turned = support_mask(native, stamp_size=81, angle_rad=angle)
        # Rotation moves the support without creating or destroying it.
        assert turned.sum() == pytest.approx(expected, rel=0.03)

    quarter = support_mask(native, stamp_size=81, angle_rad=np.pi / 2)
    assert quarter.sum() == pytest.approx(expected, rel=0.03)
    # A quarter turn of a 21x31 rectangle is 31 tall and 21 wide, not the original.
    assert not np.array_equal(quarter, unrotated)


def test_support_mask_keeps_interior_holes_and_masked_corners() -> None:
    """Some catalogue stamps carry masked corners, so a bounding box is wrong."""
    from dhs.paired_sources import support_mask

    native = np.ones((25, 25))
    native[:4, :4] = 0.0
    native[12, 12] = 0.0
    mask = support_mask(native, stamp_size=61, angle_rad=0.0)
    assert mask.sum() == int((native != 0).sum())
    assert not mask[30, 30]  # the interior hole survives


@needs_cosmos
def test_repaired_real_arm_puts_its_flux_where_the_galaxy_is() -> None:
    """The defect and its repair, stated as a number."""
    import galsim
    import pandas as pd

    from dhs.paired_sources import draw, flux_within, normalize, render_arms

    manifest = pd.read_parquet(SOURCE_MANIFEST).head(4)
    catalog = galsim.COSMOSCatalog(sample="25.2", dir=str(COSMOS_DIR))

    for record in manifest.itertuples():
        hlr = float(record.sersic_circularized_hlr_arcsec)
        published = normalize(
            draw(
                catalog.makeGalaxy(
                    int(record.cosmos_catalog_index),
                    gal_type="real",
                    noise_pad_size=0,
                ).rotate(float(record.rotation_rad) * galsim.radians),
                stamp_size=int(record.source_stamp_size),
                flux=1.0,
            )
        )
        real_map, _, _ = render_arms(
            record=record,
            catalog=catalog,
            twin_noise="catalogue",
            support_matched=True,
            noise_seed=1,
        )
        assert flux_within(published, hlr) < 0.10
        assert flux_within(real_map, hlr) > 0.15
        assert flux_within(real_map, hlr) > 4 * flux_within(published, hlr)


@needs_cosmos
def test_twin_noise_matches_the_sky_actually_present_in_the_stamp() -> None:
    """Twin noise matches sky variation measured inside the catalogue stamp."""
    import galsim
    import pandas as pd

    from dhs.paired_sources import in_footprint_sky_rms, native_image, render_arms

    manifest = pd.read_parquet(SOURCE_MANIFEST).head(6)
    catalog = galsim.COSMOSCatalog(sample="25.2", dir=str(COSMOS_DIR))

    for record in manifest.itertuples():
        _, _, matched = render_arms(
            record=record,
            catalog=catalog,
            twin_noise="catalogue",
            support_matched=True,
            noise_seed=1,
        )
        _, _, noiseless = render_arms(
            record=record,
            catalog=catalog,
            twin_noise="none",
            support_matched=True,
            noise_seed=1,
        )
        truth = in_footprint_sky_rms(
            native_image(catalog, int(record.cosmos_original_index))
        )
        assert truth > 1e-4, "the independent sky measurement must be non-trivial"
        assert 0.6 < matched["twin_added_noise_rms"] / truth < 1.4
        assert noiseless["twin_added_noise_rms"] == 0.0

        # The catalogue model's own variance sits about ten per cent below the
        # sky the stamp carries, which is why the amplitude is measured rather
        # than taken from the model. Assert the gap so that silently reverting
        # to the model's variance fails here.
        assert matched["measured_sky_sigma"] > matched["catalogue_noise_rms"]
        assert matched["twin_added_noise_rms"] == pytest.approx(
            matched["measured_sky_sigma"], rel=0.1
        )


@needs_cosmos
def test_null_control_removes_the_morphology_difference_only() -> None:
    """Both arms become the same profile, so the pipeline must find nothing.

    Any separation the experiment reports here is machinery rather than
    morphology, which makes this the calibration for every headline number.
    """
    import galsim
    import pandas as pd

    from dhs.paired_sources import render_arms

    manifest = pd.read_parquet(SOURCE_MANIFEST).head(4)
    catalog = galsim.COSMOSCatalog(sample="25.2", dir=str(COSMOS_DIR))

    for record in manifest.itertuples():
        real_map, twin_map, diagnostics = render_arms(
            record=record,
            catalog=catalog,
            twin_noise="catalogue",
            support_matched=True,
            noise_seed=20260822 + int(record.pair_index),
            arms="twin_vs_twin",
        )
        assert diagnostics["arms"] == "twin_vs_twin"
        # Same profile, independent noise: neither identical nor systematically
        # ordered in concentration.
        assert not np.array_equal(real_map, twin_map)
        assert diagnostics["real_flux_within_half_light_radius"] == pytest.approx(
            diagnostics["twin_flux_within_half_light_radius"], abs=0.06
        )

    with pytest.raises(ValueError, match="null control"):
        render_arms(
            record=next(manifest.itertuples()),
            catalog=catalog,
            twin_noise="none",
            support_matched=True,
            noise_seed=1,
            arms="twin_vs_twin",
        )


@needs_cosmos
def test_support_matching_is_what_keeps_the_clip_pedestal_symmetric() -> None:
    """Matched support and matched variance are only sufficient together.

    Stated per pair rather than in aggregate, and against the unmatched arm
    rather than an absolute threshold, because a single pair's pedestal
    difference is dominated by its own noise realisation. Whether the aggregate
    difference exceeds what the machinery invents is settled by the null
    control, not here.
    """
    import galsim
    import pandas as pd

    from dhs.paired_sources import render_arms

    manifest = pd.read_parquet(SOURCE_MANIFEST).head(6)
    catalog = galsim.COSMOSCatalog(sample="25.2", dir=str(COSMOS_DIR))

    for record in manifest.itertuples():
        seed = 20260822 + int(record.pair_index)
        _, _, matched = render_arms(
            record=record,
            catalog=catalog,
            twin_noise="catalogue",
            support_matched=True,
            noise_seed=seed,
        )
        # Dropping the support match spreads the twin's noise over the whole
        # frame against the real arm's footprint, so the pedestal it carries
        # can only grow. This is why support matching is not an optional arm.
        _, _, unmatched = render_arms(
            record=record,
            catalog=catalog,
            twin_noise="catalogue",
            support_matched=False,
            noise_seed=seed,
        )
        assert unmatched["twin_pedestal_fraction"] > matched["twin_pedestal_fraction"]
        assert unmatched["twin_pedestal_fraction"] > unmatched["real_pedestal_fraction"]


@needs_cosmos
def test_real_arm_reproduces_the_catalogue_stamp_exactly() -> None:
    """Reconvolution is algebraically equivalent but numerically inexact.

    Both routes are compared against the catalogue image in units of that
    stamp's own sky noise, which is the only scale on which the difference
    means anything. ``original_gal`` agrees to float32 round-trip precision;
    ``Convolve(real, original_psf)`` lands four orders of magnitude further
    away, so there is no accuracy cost to preferring the direct route.
    """
    import galsim
    import pandas as pd

    from dhs.paired_sources import in_footprint_sky_rms, native_image

    manifest = pd.read_parquet(SOURCE_MANIFEST).head(4)
    catalog = galsim.COSMOSCatalog(sample="25.2", dir=str(COSMOS_DIR))

    for record in manifest.itertuples():
        truth = native_image(catalog, int(record.cosmos_original_index))
        sigma = in_footprint_sky_rms(truth)
        base = catalog.makeGalaxy(
            int(record.cosmos_catalog_index), gal_type="real", noise_pad_size=0
        )
        height, width = truth.shape

        def render(
            profile: galsim.GSObject, *, nx: int = width, ny: int = height
        ) -> np.ndarray:
            return np.asarray(
                profile.drawImage(nx=nx, ny=ny, scale=0.03, method="no_pixel").array,
                dtype=np.float64,
            )

        direct = np.abs(render(base.original_gal) - truth).max() / sigma
        reconvolved = (
            np.abs(render(galsim.Convolve(base, base.original_psf)) - truth).max()
            / sigma
        )
        assert direct < 1e-4
        assert reconvolved > 100 * direct


@needs_cosmos
def test_render_arms_is_deterministic_given_a_seed() -> None:
    import galsim
    import pandas as pd

    from dhs.paired_sources import render_arms

    record = next(pd.read_parquet(SOURCE_MANIFEST).head(1).itertuples())
    catalog = galsim.COSMOSCatalog(sample="25.2", dir=str(COSMOS_DIR))
    first, second = (
        render_arms(
            record=record,
            catalog=catalog,
            twin_noise="catalogue",
            support_matched=True,
            noise_seed=7,
        )
        for _ in range(2)
    )
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
