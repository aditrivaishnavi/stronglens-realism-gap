from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from grec_pnoise.scoring import _import_fixed_modules

from checkpoint_sensitivity_v2.pipeline import _score_images
from checkpoint_sensitivity_v2.threshold_variation import _threshold

fixed_width_scoring, _, _ = _import_fixed_modules(
    {"scoring": {"repo_root": str(Path(__file__).resolve().parents[3])}}
)


class MeanModel(torch.nn.Module):
    def __init__(self, multiplier: float) -> None:
        super().__init__()
        self.multiplier = multiplier

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.mean(dim=(1, 2, 3)) * self.multiplier


def test_batch_major_panel_scoring_preserves_order_and_terminal_padding() -> None:
    images = [np.full((3, 5, 5), index, dtype=np.float32) for index in range(70)]
    preprocess = SimpleNamespace(
        preprocess_stack=lambda image, **_: np.asarray(image, dtype=np.float32)
    )
    models = {
        "one": MeanModel(1.0).eval(),
        "two": MeanModel(2.0).eval(),
    }

    scores, receipt = _score_images(
        iter(images),
        rows=70,
        models=models,
        preprocess_kwargs={},
        fixed=fixed_width_scoring,
        preprocess=preprocess,
        device="cpu",
        batch_size=64,
        chunk_rows=64,
        label="test",
    )

    expected = np.arange(70, dtype=np.float32)
    np.testing.assert_array_equal(scores["one"], expected)
    np.testing.assert_array_equal(scores["two"], 2.0 * expected)
    assert receipt["terminal_batch_real_rows"] == 6
    assert receipt["raw_images_read_once_for_complete_panel"] is True
    assert receipt["preprocessing_applied_once_per_image"] is True
    assert receipt["model_forward_passes_per_image"] == 2


def test_batch_major_scoring_rejects_incomplete_input_stream() -> None:
    preprocess = SimpleNamespace(
        preprocess_stack=lambda image, **_: np.asarray(image, dtype=np.float32)
    )
    model = MeanModel(1.0).eval()
    images = [np.zeros((3, 5, 5), dtype=np.float32)]

    try:
        _score_images(
            iter(images),
            rows=2,
            models={"one": model},
            preprocess_kwargs={},
            fixed=fixed_width_scoring,
            preprocess=preprocess,
            device="cpu",
            batch_size=64,
            chunk_rows=64,
            label="test",
        )
    except RuntimeError as error:
        assert "yielded 1 images, expected 2" in str(error)
    else:
        raise AssertionError("incomplete input stream was accepted")


def test_threshold_variation_uses_next_float_above_k_plus_one() -> None:
    values = np.arange(1001, dtype=np.float32)
    threshold = _threshold(values, 0.02)
    ordered = np.sort(values)[::-1]
    assert threshold == np.nextafter(ordered[20], np.float32(np.inf), dtype=np.float32)
    assert np.count_nonzero(values >= threshold) == 20
