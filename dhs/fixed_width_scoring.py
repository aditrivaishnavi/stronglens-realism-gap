"""Fixed-width inference shared by all reported detector populations.

**Batch shape.** Batch size selects cuDNN kernels, which fixes the order of
floating-point accumulation, so a short terminal batch puts its rows through a
different kernel from every other batch. The remedy is to pad every batch to the
contract width and discard the filler outputs. That is only sound because
evaluation-mode inference decouples rows, which :func:`assert_no_batch_coupling`
establishes at run time rather than assuming.

**Threshold placement.** For false-positive allowance k,
the smallest representable value strictly greater than s_(k+1), classifying by
``score >= t``, places the boundary above the complete allowed negative set.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import torch

FILLER_POLICIES = ("last", "first", "zeros", "reversed")
CONTRACT_FILLER = "last"


class ScoringError(RuntimeError):
    """A departure from the fixed-width scoring contract."""


def assert_no_batch_coupling(model: torch.nn.Module) -> dict[str, object]:
    """Establish that a row's logit cannot depend on what shares its batch.

    Padding is harmless only if examples do not interact. In training mode they
    do: BatchNorm normalises using the statistics of whatever batch it is handed,
    so a filler row would perturb every real row beside it. In eval mode
    BatchNorm uses its stored running statistics and dropout is the identity,
    which severs the coupling.
    """
    if model.training:
        raise ScoringError(
            "model is in training mode; batch statistics would couple rows and the "
            "filler rows would change the logits of the real ones"
        )

    norm_layers: list[str] = []
    dropout_layers: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if module.training:
                raise ScoringError(f"{name} is in training mode")
            if module.track_running_stats and module.running_mean is None:
                raise ScoringError(
                    f"{name} tracks running statistics but has none stored, so it "
                    "would fall back to batch statistics"
                )
            norm_layers.append(name)
        elif isinstance(module, torch.nn.modules.dropout._DropoutNd):
            if module.training:
                raise ScoringError(f"{name} is active; dropout must be the identity")
            dropout_layers.append(name)

    return {
        "model_training_mode": False,
        "batchnorm_layers": len(norm_layers),
        "batchnorm_all_using_running_statistics": True,
        "dropout_layers": len(dropout_layers),
        "dropout_all_inactive": True,
        "why_padding_is_safe": (
            "in eval mode normalisation uses stored running statistics and dropout "
            "is the identity, so no row's output depends on the other rows sharing "
            "its batch; filler outputs are discarded and cannot affect real ones"
        ),
    }


def pin_deterministic_kernels() -> dict[str, object]:
    """Remove run-to-run kernel variation, which the contract cannot tolerate.

    ``benchmark=True`` lets cuDNN choose an algorithm from timing measurements,
    so the same code on the same hardware can accumulate in a different order on
    a different day. That is exactly the variation this contract exists to
    eliminate.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {"cudnn_deterministic": True, "cudnn_benchmark": False}


def _filler_block(x: torch.Tensor, width: int, policy: str) -> torch.Tensor:
    if policy == "last":
        return x[-1:].expand(width, *x.shape[1:])
    if policy == "first":
        return x[:1].expand(width, *x.shape[1:])
    if policy == "zeros":
        return torch.zeros((width, *x.shape[1:]), dtype=x.dtype, device=x.device)
    if policy == "reversed":
        return x.flip(0)[:1].expand(width, *x.shape[1:])
    raise ScoringError(
        f"unknown filler policy {policy!r}; expected one of {FILLER_POLICIES}"
    )


@torch.no_grad()
def fixed_width_logits(
    model: torch.nn.Module,
    batches: Iterable[torch.Tensor],
    *,
    batch_size: int,
    device: str,
    filler: str = CONTRACT_FILLER,
) -> tuple[np.ndarray, dict[str, object]]:
    """Raw logits in input order, every forward pass at exactly ``batch_size``.

    ``batches`` yields tensors whose leading dimension is the batch. Any batch
    narrower than the contract width is padded and its filler outputs dropped, so
    a caller cannot accidentally send a short batch through a different kernel.

    ``filler`` is ``last`` under the contract. The alternatives exist so that
    invariance to the filler choice can be measured, which is the empirical
    counterpart of the decoupling argument.
    """
    if filler not in FILLER_POLICIES:
        raise ScoringError(f"unknown filler policy {filler!r}")

    collected: list[np.ndarray] = []
    padded_rows = 0
    terminal_real_rows: int | None = None
    seen = 0
    n_batches = 0

    for batch in batches:
        x = batch.to(device, non_blocking=True)
        present = int(x.shape[0])
        seen += present
        n_batches += 1
        if present > batch_size:
            raise ScoringError(
                f"batch of {present} exceeds the contract width {batch_size}"
            )
        if present < batch_size:
            if terminal_real_rows is not None:
                raise ScoringError(
                    "a second short batch appeared; only the terminal batch may be "
                    "short, otherwise the caller is not iterating in order"
                )
            terminal_real_rows = present
            width = batch_size - present
            x = torch.cat([x, _filler_block(x, width, filler)], dim=0)
            padded_rows += width
        output = model(x).reshape(-1)
        if output.numel() != x.shape[0]:
            raise ScoringError(
                f"model returned {output.numel()} logits for {x.shape[0]} rows"
            )
        non_finite = int(torch.count_nonzero(~torch.isfinite(output)).item())
        if non_finite:
            raise ScoringError(
                f"model returned {non_finite} non-finite logits in a batch"
            )
        collected.append(output[:present].cpu().double().numpy())

    record = {
        "inference_batch_size": batch_size,
        "batch_shape_fixed": True,
        "filler_policy": filler,
        "final_batch_padding": f"{filler} row repeated, outputs discarded",
        "batches": n_batches,
        "full_batches": n_batches - (1 if terminal_real_rows is not None else 0),
        "terminal_batch_real_rows": terminal_real_rows,
        "padded_rows_discarded": padded_rows,
        "rows_scored": seen,
        "governing_filler": filler == CONTRACT_FILLER,
    }
    return np.concatenate(collected) if collected else np.empty(0), record


def array_batches(
    images: np.ndarray | Iterable[np.ndarray], batch_size: int
) -> Iterator[torch.Tensor]:
    """Group preprocessed arrays into contract-width batches, in order."""
    buffer: list[np.ndarray] = []
    for image in images:
        buffer.append(np.asarray(image, dtype=np.float32))
        if len(buffer) == batch_size:
            yield torch.from_numpy(np.stack(buffer))
            buffer = []
    if buffer:
        yield torch.from_numpy(np.stack(buffer))


def operating_points(
    negative_logits: np.ndarray, allowances: Iterable[int]
) -> list[dict]:
    """Thresholds exactly as §14 of the plan defines them.

    For allowance k, with the validation negatives sorted descending as
    s_(1) >= s_(2) >= ..., the threshold is the smallest float32 strictly greater
    than **s_(k+1)**, and the rule is ``score >= t``.

    "Smallest representable" is evaluated in float32, the native precision of the
    model's output. Stepping in float64 would put the threshold inside a gap no
    float32 score can occupy, quietly turning ``>=`` into ``>``.

    Ties resolve above the whole tied group, so the realised count falls below k
    rather than exceeding it. The plan requires the looser adjacent threshold be
    reported alongside as an interval, so every rung carries it.
    """
    values = np.asarray(negative_logits, dtype=np.float32)
    if values.ndim != 1:
        raise ScoringError(
            f"negative logits must be one-dimensional, got shape {values.shape}"
        )
    non_finite = int(np.count_nonzero(~np.isfinite(values)))
    if non_finite:
        # NaN sorts unpredictably and never compares true against a threshold,
        # so a single NaN would silently shift every order statistic below it.
        raise ScoringError(
            f"negative logits contain {non_finite} non-finite values; the "
            "operating point cannot be constructed"
        )
    ordered = np.sort(values)[::-1]
    total = ordered.size
    inf32 = np.float32(np.inf)
    rungs: list[dict] = []

    for allowance in allowances:
        if allowance < 1 or allowance + 1 > total:
            continue
        loose = ordered[allowance]  # s_(k+1), zero-indexed
        threshold = np.nextafter(loose, inf32, dtype=np.float32)
        at_or_above = int((ordered >= threshold).sum())
        at_or_above_loose = int((ordered >= loose).sum())
        rungs.append(
            {
                "allowance": allowance,
                "rule": "smallest float32 strictly greater than s_(k+1); score >= t",
                "logit_threshold": float(threshold),
                "logit_threshold_float32_bits": int(threshold.view(np.uint32)),
                "s_k": float(ordered[allowance - 1]),
                "s_k_plus_1": float(loose),
                "probability_threshold_float64": float(
                    1.0 / (1.0 + np.exp(-np.longdouble(threshold)))
                ),
                "negatives_at_or_above": at_or_above,
                "realised_fpr": at_or_above / total,
                "identified": at_or_above == allowance,
                "tied_at_s_k_plus_1": int((ordered == loose).sum()),
                "loose_negatives_at_or_above": at_or_above_loose,
                "loose_realised_fpr": at_or_above_loose / total,
            }
        )
    return rungs
