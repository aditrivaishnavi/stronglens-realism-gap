"""Fail-closed tests for fixed-width detector inference."""

from __future__ import annotations

import pytest
import torch

from dhs.fixed_width_scoring import ScoringError, fixed_width_logits


class NonFiniteModel(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (len(values), 1),
            torch.nan,
            dtype=values.dtype,
            device=values.device,
        )


class WrongWidthModel(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (len(values), 2),
            dtype=values.dtype,
            device=values.device,
        )


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (NonFiniteModel(), "non-finite logits"),
        (WrongWidthModel(), "logits for"),
    ],
)
def test_fixed_width_scoring_rejects_invalid_model_output(
    model: torch.nn.Module,
    message: str,
) -> None:
    batch = torch.zeros((2, 3, 4, 4), dtype=torch.float32)
    with pytest.raises(ScoringError, match=message):
        fixed_width_logits(
            model.eval(),
            [batch],
            batch_size=2,
            device="cpu",
        )
