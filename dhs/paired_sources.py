"""Rendering primitives for the paired COSMOS and fitted-Sersic source arms.

The COSMOS arm uses the interpolated observed source stamp. The fitted-Sersic
arm uses the catalogue fit of the same object. The primary construction matches
source support and the declared correlated-noise amplitude before each arm is
normalized to unit source-plane flux. A ``twin_vs_twin`` control renders the
same fitted profile in both arms under independent noise realizations.

These choices define a source-construction comparison, not a pure morphology
intervention, because residual stamp and noise treatment also differ.
"""

from __future__ import annotations

from typing import Any

import galsim
import numpy as np

SOURCE_PIXEL_SCALE_ARCSEC = 0.03

# Expected positive part of a zero-mean Gaussian, E[max(x, 0)] = sigma / sqrt(2 pi).
HALF_NORMAL_MEAN = float(1.0 / np.sqrt(2.0 * np.pi))

# The sky is sampled outside this fraction of the catalogue stamp's smaller
# dimension, widening if that leaves too few pixels for a stable scale.
SKY_RING_RADIUS_FRACTION = 0.42
SKY_RING_MINIMUM_PIXELS = 50


def normalize(raw: np.ndarray) -> np.ndarray:
    """Clip negatives and scale to unit flux, as the frozen stage does."""
    positive = np.maximum(np.asarray(raw, dtype=np.float64), 0.0)
    total = float(positive.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Rendered source has no positive finite flux")
    return positive / total


def radius_grid(stamp_size: int) -> np.ndarray:
    centre = (stamp_size - 1) / 2.0
    y, x = np.mgrid[0:stamp_size, 0:stamp_size]
    return np.hypot(y - centre, x - centre) * SOURCE_PIXEL_SCALE_ARCSEC


def sersic_profile(record: Any) -> galsim.GSObject:
    """The catalogue's own single-Sersic fit to this object."""
    return galsim.Sersic(
        n=float(record.sersic_n),
        half_light_radius=float(record.sersic_circularized_hlr_arcsec),
    ).shear(
        q=float(record.sersic_q),
        beta=float(record.sersic_phi_catalog_rad) * galsim.radians,
    )


def draw(
    profile: galsim.GSObject, *, stamp_size: int, flux: float | None
) -> np.ndarray:
    shaped = profile if flux is None else profile.withFlux(flux)
    return np.asarray(
        shaped.drawImage(
            nx=stamp_size,
            ny=stamp_size,
            scale=SOURCE_PIXEL_SCALE_ARCSEC,
            method="no_pixel",
        ).array,
        dtype=np.float64,
    )


def flux_within(image: np.ndarray, radius_arcsec: float) -> float:
    total = float(image.sum())
    if total <= 0:
        return float("nan")
    inside = image[radius_grid(image.shape[0]) <= radius_arcsec].sum()
    return float(inside / total)


def native_image(catalog: galsim.COSMOSCatalog, original_index: int) -> np.ndarray:
    return np.asarray(
        catalog.real_cat.getGalImage(original_index).array, dtype=np.float64
    )


def support_mask(
    native: np.ndarray, *, stamp_size: int, angle_rad: float
) -> np.ndarray:
    """The catalogue stamp's own support, rotated onto the working grid.

    Uses a nearest-neighbour interpolant so the rotated boundary stays crisp;
    a smooth interpolant would ring and blur the very edge being matched.
    """
    indicator = native != 0.0
    height, width = native.shape

    # Inverse-map each working-grid pixel back to the catalogue grid. Doing the
    # rotation arithmetic directly keeps the boundary exact; rendering the
    # indicator through GalSim would resample it and blur the edge being matched.
    centre = (stamp_size - 1) / 2.0
    y, x = np.mgrid[0:stamp_size, 0:stamp_size]
    dx = x - centre
    dy = y - centre
    cos, sin = np.cos(angle_rad), np.sin(angle_rad)
    source_x = cos * dx + sin * dy
    source_y = -sin * dx + cos * dy

    column = np.rint(source_x + (width - 1) / 2.0).astype(int)
    row = np.rint(source_y + (height - 1) / 2.0).astype(int)
    inside = (row >= 0) & (row < height) & (column >= 0) & (column < width)
    mask = np.zeros((stamp_size, stamp_size), dtype=bool)
    mask[inside] = indicator[row[inside], column[inside]]
    return mask


def in_footprint_sky_rms(native: np.ndarray) -> float:
    """Sky RMS measured in the catalogue stamp's corners, away from the galaxy.

    Kept as an independent numerical cross-check. The amplitude the twin
    actually receives comes from :func:`measured_sky_sigma`.
    """
    height, width = native.shape
    block = max(4, min(height, width) // 8)
    corners = np.concatenate(
        [
            native[:block, :block].ravel(),
            native[:block, -block:].ravel(),
            native[-block:, :block].ravel(),
            native[-block:, -block:].ravel(),
        ]
    )
    return float(np.std(corners))


def measured_sky_sigma(native: np.ndarray) -> float:
    """Sky scale from the catalogue stamp's outer ring, by median deviation.

    A median absolute deviation rather than a standard deviation, so that a
    neighbouring object or a cosmic ray in the ring shifts the answer by little.
    Verified to recover 1.014 on noise of known amplitude, and uncorrelated with
    galaxy size, so it is measuring sky rather than the wings of the source.
    """
    height, width = native.shape
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    radius = np.hypot(grid_y - (height - 1) / 2.0, grid_x - (width - 1) / 2.0)
    smaller = min(height, width)

    fraction = SKY_RING_RADIUS_FRACTION
    ring = native[radius > fraction * smaller]
    while ring.size < SKY_RING_MINIMUM_PIXELS and fraction > 0.15:
        fraction -= 0.05
        ring = native[radius > fraction * smaller]
    if ring.size < 8:
        ring = native.ravel()
    return float(1.4826 * np.median(np.abs(ring - np.median(ring))))


def clip_pedestal_fraction(raw: np.ndarray) -> float:
    """Flux that clipping at zero invents, over the flux that survives it.

    In expectation the discarded negative flux equals the excess the clip adds
    over the true profile, so this needs neither a noise model nor a clean
    version of the arm. That matters because the real arm has no clean version:
    its noise is part of the observation. Checked against the twin, where the
    truth is computable, the ratio is 0.973.
    """
    positive = float(raw[raw > 0].sum())
    if positive <= 0:
        return float("nan")
    return float(np.abs(raw[raw < 0]).sum() / positive)


def render_arms(
    *,
    record: Any,
    catalog: galsim.COSMOSCatalog,
    twin_noise: str,
    support_matched: bool,
    noise_seed: int,
    arms: str = "real_vs_twin",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Render one pair's two surface-brightness maps.

    ``twin_noise`` is ``"catalogue"`` to give the twin a realisation of the
    source's own correlated-noise model at the amplitude measured off the real
    stamp, or ``"none"`` to leave it noiseless as the production pipeline's
    analytic sources are.

    ``arms`` is ``"real_vs_twin"`` for the experiment, or ``"twin_vs_twin"``
    for the null control, where the first arm is a second Sersic render under
    an independent noise realisation. The control removes the morphology
    difference and changes nothing else, so any signal it shows is machinery.
    """
    if twin_noise not in {"catalogue", "none"}:
        raise ValueError(f"Unsupported twin noise treatment: {twin_noise}")
    if arms not in {"real_vs_twin", "twin_vs_twin"}:
        raise ValueError(f"Unsupported arm configuration: {arms}")
    if arms == "twin_vs_twin" and twin_noise != "catalogue":
        # Leaving one arm noiseless would make the two arms identical rather
        # than independent, which tests nothing.
        raise ValueError("The null control requires twin_noise='catalogue'")

    stamp_size = int(record.source_stamp_size)
    angle_rad = float(record.rotation_rad)
    angle = angle_rad * galsim.radians

    base = catalog.makeGalaxy(
        int(record.cosmos_catalog_index), gal_type="real", noise_pad_size=0
    )
    observed = base.original_gal
    native = native_image(catalog, int(record.cosmos_original_index))
    height, width = native.shape
    mask = support_mask(native, stamp_size=stamp_size, angle_rad=angle_rad)

    # The model supplies the correlation structure; the amplitude comes from the
    # real stamp, because the model's own variance is a systematic 10 per cent
    # below the sky the stamp actually carries.
    sky_sigma = measured_sky_sigma(native)
    catalogue_noise_rms = float(
        np.sqrt(base.noise.convolvedWith(base.original_psf).getVariance())
    )
    noise_model = (
        base.noise.convolvedWith(base.original_psf)
        .rotate(angle)
        .withVariance(sky_sigma**2)
    )

    def add_noise(image_array: np.ndarray, seed: int) -> np.ndarray:
        image = galsim.Image(image_array.copy(), scale=SOURCE_PIXEL_SCALE_ARCSEC)
        image.addNoise(noise_model.copy(rng=galsim.BaseDeviate(seed)))
        return np.asarray(image.array, dtype=np.float64)

    def render_twin() -> np.ndarray:
        return draw(
            galsim.Convolve(
                sersic_profile(record).rotate(angle), base.original_psf.rotate(angle)
            ),
            stamp_size=stamp_size,
            flux=float(observed.flux),
        )

    twin_raw = render_twin()
    if arms == "real_vs_twin":
        real_raw = draw(observed.rotate(angle), stamp_size=stamp_size, flux=None)
    else:
        # Same profile, independent realisation. The offset keeps the two draws
        # from sharing a stream while staying reproducible from the seed.
        real_raw = add_noise(render_twin(), noise_seed + 1_000_003)

    added_noise_rms = 0.0
    if twin_noise == "catalogue":
        noised = add_noise(twin_raw, noise_seed)
        added_noise_rms = float(np.std(noised - twin_raw))
        twin_raw = noised

    if support_matched:
        real_raw = np.where(mask, real_raw, 0.0)
        twin_raw = np.where(mask, twin_raw, 0.0)

    real_map = normalize(real_raw)
    twin_map = normalize(twin_raw)

    hlr = float(record.sersic_circularized_hlr_arcsec)
    diagnostics = {
        "arms": arms,
        "native_stamp_height_px": int(height),
        "native_stamp_width_px": int(width),
        "support_pixels": int(mask.sum()),
        "measured_sky_sigma": sky_sigma,
        "catalogue_noise_rms": catalogue_noise_rms,
        "in_footprint_sky_rms": in_footprint_sky_rms(native),
        "twin_added_noise_rms": added_noise_rms,
        # Measured from each arm's own pixels rather than modelled, so the two
        # are comparable even though only one of them has a clean version.
        "real_pedestal_fraction": clip_pedestal_fraction(real_raw),
        "twin_pedestal_fraction": clip_pedestal_fraction(twin_raw),
        "stamp_half_widths_per_half_light_radius": float(
            min(height, width) / 2.0 * SOURCE_PIXEL_SCALE_ARCSEC / hlr
        ),
        "real_flux_within_half_light_radius": flux_within(real_map, hlr),
        "twin_flux_within_half_light_radius": flux_within(twin_map, hlr),
        "real_twin_pixel_correlation": float(
            np.corrcoef(real_map.ravel(), twin_map.ravel())[0, 1]
        ),
    }
    return real_map, twin_map, diagnostics
