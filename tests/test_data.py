"""Tests for `dhs.data`.

Covers `stack_from_npz`, which converts a decoded NPZ into the channel-first
array the model consumes. Its band ordering is the part that matters: every
cutout on S3 stores bands as separate named arrays, so the order in the stacked
result comes from `BANDS` rather than from the file, and a silent reordering
here would mistrain a model without failing anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from dhs.constants import BANDS
from dhs.data import stack_from_npz


def _decoded(**values: float) -> dict[str, np.ndarray]:
    """Build a decoded NPZ mapping, one constant-valued plane per band."""
    return {
        f"image_{band}": np.full((8, 8), value, dtype=np.float32)
        for band, value in values.items()
    }


def test_bands_are_stacked_in_the_canonical_order() -> None:
    """Channel *i* must hold `BANDS[i]`, whatever order the dict happens to be in."""
    per_band = {band: float(index + 1) for index, band in enumerate(BANDS)}
    decoded = dict(reversed(list(_decoded(**per_band).items())))

    stacked = stack_from_npz(decoded)

    assert stacked.shape == (len(BANDS), 8, 8)
    for index, band in enumerate(BANDS):
        assert stacked[index].min() == stacked[index].max() == per_band[band]


def test_result_is_float32_even_when_the_file_is_not() -> None:
    """Cutouts on S3 are not uniformly typed; the stack must be."""
    decoded = {f"image_{band}": np.ones((4, 4), dtype=np.float64) for band in BANDS}

    assert stack_from_npz(decoded).dtype == np.float32


def test_a_missing_band_names_itself() -> None:
    """A partial cutout must fail loudly rather than stack a wrong-sized array."""
    decoded = _decoded(**{band: 1.0 for band in BANDS})
    absent = f"image_{BANDS[-1]}"
    del decoded[absent]

    with pytest.raises(KeyError, match=absent):
        stack_from_npz(decoded)
