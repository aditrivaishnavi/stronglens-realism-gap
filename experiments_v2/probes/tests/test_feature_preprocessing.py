from __future__ import annotations

import hashlib

import numpy as np

from stronglens_probes.feature_extraction import lupton_rgb


def _reference(image_grz: np.ndarray) -> np.ndarray:
    image = np.asarray(image_grz, dtype=np.float32)
    affine_grz = image * np.asarray([6.0, 3.4, 2.2], dtype=np.float32) + 0.03
    intensity = np.maximum(0.0, affine_grz[:, :, 0])
    intensity += np.maximum(0.0, affine_grz[:, :, 1])
    intensity += np.maximum(0.0, affine_grz[:, :, 2])
    intensity /= 3.0
    mapped = np.arcsinh(20.0 * intensity) / np.sqrt(20.0)
    safe = intensity + (intensity == 0.0) * 1e-6
    return np.clip(
        affine_grz[:, :, [2, 1, 0]] * (mapped / safe)[:, :, None],
        0.0,
        1.0,
    ).astype(np.float32)


def test_feature_extraction_uses_authoritative_dr2_rgb_mapping() -> None:
    image = np.linspace(-0.02, 0.12, 60, dtype=np.float32).reshape(4, 5, 3)
    assert np.array_equal(lupton_rgb(image), _reference(image))


def test_feature_extraction_preserves_zero_intensity_sign_bits() -> None:
    image = np.full((1, 1, 3), -1.0, dtype=np.float32)
    assert lupton_rgb(image).view(np.uint32).reshape(-1).tolist() == [
        2147483648,
        2147483648,
        2147483648,
    ]


def test_feature_extraction_preserves_pinned_random_fixture_hash() -> None:
    rng = np.random.default_rng(9172026)
    image = rng.normal(0.0, 0.05, size=(17, 19, 3)).astype(np.float32)
    observed = lupton_rgb(image)
    assert hashlib.sha256(observed.tobytes()).hexdigest() == (
        "9396d3f1f1943beebe50ec6bb99994a5baf253d895b5c8e9237bf576327337ef"
    )
