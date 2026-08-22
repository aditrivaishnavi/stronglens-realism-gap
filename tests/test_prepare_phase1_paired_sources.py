from __future__ import annotations

import galsim
import numpy as np

from scripts.prepare_phase1_paired_sources import (
    candidate_indices,
    normalized_map,
)


def test_candidate_indices_enforce_frozen_source_ranges() -> None:
    dtype = [
        ("SERSICFIT", "f8", (8,)),
        ("FIT_STATUS", "i4", (5,)),
        ("VIABLE_SERSIC", "i4"),
        ("MAG_AUTO", "f8"),
    ]
    catalog = np.zeros(3, dtype=dtype)
    catalog["SERSICFIT"][:, 1] = 10.0
    catalog["SERSICFIT"][:, 2] = 1.0
    catalog["SERSICFIT"][:, 3] = 0.7
    catalog["FIT_STATUS"][:, 4] = 1
    catalog["VIABLE_SERSIC"] = 1
    catalog["MAG_AUTO"] = [24.0, 22.0, 24.0]
    catalog["SERSICFIT"][2, 2] = 4.0

    assert candidate_indices(catalog).tolist() == [0]


def test_normalized_source_map_has_unit_sum() -> None:
    profile = galsim.Sersic(n=1.0, half_light_radius=0.25)

    source, diagnostics = normalized_map(
        profile,
        stamp_size=101,
        pixel_scale=0.03,
    )

    assert source.shape == (101, 101)
    assert np.isclose(source.sum(), 1.0)
    assert diagnostics["raw_negative_pixel_fraction"] == 0.0
