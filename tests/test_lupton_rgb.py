from __future__ import annotations

import hashlib

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
    assert observed.view(np.uint32).reshape(-1).tolist() == [
        1050552902,
        1040203252,
        0,
        0,
        1049264753,
        1065353216,
        1040334569,
        1040334569,
        1040334569,
        1050800650,
        1051021760,
        1050137321,
    ]
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


def test_lupton_rgb_preserves_pinned_random_fixture_hash() -> None:
    rng = np.random.default_rng(9172026)
    image = rng.normal(0.0, 0.05, size=(17, 19, 3)).astype(np.float32)
    observed = lupton_rgb(image)
    assert hashlib.sha256(observed.tobytes()).hexdigest() == (
        "9396d3f1f1943beebe50ec6bb99994a5baf253d895b5c8e9237bf576327337ef"
    )
