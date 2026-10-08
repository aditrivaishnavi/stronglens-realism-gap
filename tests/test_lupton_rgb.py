from __future__ import annotations

import numpy as np

from dhs.feature_encoders import lupton_rgb


def reference_dr2_rgb(
    image_grz: np.ndarray, *, m: float = 0.03, q: float = 20.0
) -> np.ndarray:
    image = np.asarray(image_grz, dtype=np.float32)
    affine_grz = image * np.asarray([6.0, 3.4, 2.2], dtype=np.float32) + m
    intensity = np.maximum(0.0, affine_grz[:, :, 0])
    intensity += np.maximum(0.0, affine_grz[:, :, 1])
    intensity += np.maximum(0.0, affine_grz[:, :, 2])
    intensity /= 3.0
    mapped = np.arcsinh(q * intensity) / np.sqrt(q)
    safe = intensity + (intensity == 0.0) * 1e-6
    affine_zrg = affine_grz[:, :, [2, 1, 0]]
    return np.clip(affine_zrg * (mapped / safe)[:, :, None], 0.0, 1.0).astype(
        np.float32
    )


def test_lupton_rgb_matches_authoritative_dr2_mapping() -> None:
    image = np.asarray(
        [
            [[-0.01, 0.00, 0.02], [0.10, 0.03, -0.02]],
            [[0.00, 0.00, 0.00], [0.01, 0.02, 0.03]],
        ],
        dtype=np.float32,
    )
    observed = lupton_rgb(image)
    expected = reference_dr2_rgb(image)
    assert np.array_equal(observed, expected)
    np.testing.assert_allclose(
        observed,
        np.asarray(
            [
                [
                    [0.30891627, 0.12523633, 0.0],
                    [0.0, 0.27052644, 1.0],
                ],
                [
                    [0.12719311, 0.12719311, 0.12719311],
                    [0.31629974, 0.32288933, 0.29653100],
                ],
            ],
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=2 * np.finfo(np.float32).eps,
    )
    assert np.all(observed[1, 0] > 0.0)


def test_lupton_rgb_q_controls_asinh_stretch_when_m_is_zero() -> None:
    image = np.full((2, 2, 3), 0.01, dtype=np.float32)
    low_q = lupton_rgb(image, m=0.0, q=5.0)
    high_q = lupton_rgb(image, m=0.0, q=20.0)
    assert not np.array_equal(low_q, high_q)


def test_lupton_rgb_preserves_pinned_zero_intensity_bits() -> None:
    image = np.full((1, 1, 3), -1.0, dtype=np.float32)
    observed = lupton_rgb(image)
    assert observed.view(np.uint32).reshape(-1).tolist() == [
        2147483648,
        2147483648,
        2147483648,
    ]


def test_lupton_rgb_preserves_random_fixture_mapping() -> None:
    rng = np.random.default_rng(9172026)
    image = rng.normal(0.0, 0.05, size=(17, 19, 3)).astype(np.float32)
    observed = lupton_rgb(image)
    expected = reference_dr2_rgb(image)
    assert np.array_equal(observed, expected)
    np.testing.assert_allclose(
        [observed.min(), observed.max(), observed.mean(), observed.std()],
        [-0.0, 1.0, 0.25861278, 0.31492627],
        rtol=0.0,
        atol=2 * np.finfo(np.float32).eps,
    )
