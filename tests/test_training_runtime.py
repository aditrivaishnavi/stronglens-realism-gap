"""Tests for portable mixture loading, balanced sampling, and metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dhs.data import DatasetConfig, LensDataset, SplitConfig
from dhs.model import initialization_provenance
from dhs.scripts.run_experiment import _write_run_info
from dhs.train import (
    PERIODIC_CHECKPOINT_EPOCHS,
    TrainConfig,
    build_training_sampler,
    evaluate,
    negative_only_threshold,
)
from dhs.transforms import AugmentConfig


def dataset_config(manifest: Path, root: Path, *, seed: int = 42) -> DatasetConfig:
    return DatasetConfig(
        parquet_path="",
        mode="file_manifest",
        preprocessing="raw_robust",
        manifest_path=str(manifest),
        cutout_path_col="cutout_relpath",
        cutout_root=str(root),
        sample_weight_col="sample_weight",
        seed=seed,
        crop=False,
        annulus_r_in=32.5,
        annulus_r_out=45.0,
    )


def test_relative_cutout_paths_resolve_under_shared_root(tmp_path: Path) -> None:
    relative = Path("stronglens_calibration/cutouts/positives/a.npz")
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    np.savez(target, cutout=np.ones((101, 101, 3), dtype=np.float32))
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame(
        {
            "cutout_relpath": [str(relative)],
            "label": [1],
            "sample_weight": [1.0],
            "split": ["train"],
            "cohort": ["clean_confirmed_lens"],
        }
    ).to_parquet(manifest, index=False)
    dataset = LensDataset(
        dataset_config(manifest, tmp_path),
        SplitConfig(split_value="train"),
        AugmentConfig(hflip=False, vflip=False, rot90=False),
    )

    image, label, weight = dataset[0]
    assert image.shape == (3, 101, 101)
    assert label == 1
    assert weight == 1.0


def test_balanced_sampler_is_seeded_and_records_natural_prevalence(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame(
        {
            "cutout_relpath": [f"cutouts/{index}.npz" for index in range(10)],
            "label": [1, 1] + [0] * 8,
            "sample_weight": [1.0] * 10,
            "split": ["train"] * 10,
            "cohort": ["clean_confirmed_lens"] * 2 + ["clean_negative"] * 8,
        }
    ).to_parquet(manifest, index=False)
    dcfg = dataset_config(manifest, tmp_path, seed=123)
    dataset = LensDataset(
        dcfg,
        SplitConfig(split_value="train"),
        AugmentConfig(),
    )
    tcfg = TrainConfig(
        batch_size=4,
        balanced_sampling=True,
        sampler_positive_fraction=0.5,
        examples_per_epoch=1000,
    )
    first, info = build_training_sampler(dataset, dcfg, tcfg)
    second, _ = build_training_sampler(dataset, dcfg, tcfg)
    first_indices = list(first)
    second_indices = list(second)

    assert first_indices == second_indices
    sampled_positive_fraction = float(dataset.df.iloc[first_indices]["label"].mean())
    assert abs(sampled_positive_fraction - 0.5) < 0.05
    assert info["natural_positive_fraction"] == 0.2
    assert info["unique_positive_rows"] == 2
    assert info["unique_negative_rows"] == 8


class IdentityLogit(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1]


def test_evaluation_reports_standard_metrics() -> None:
    logits = torch.tensor([[-4.0], [-2.0], [2.0], [4.0]])
    labels = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)

    metrics = evaluate(IdentityLogit(), loader, torch.device("cpu"))

    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["all_negative_baseline"]["accuracy"] == 0.5
    assert set(metrics["operating_points"]) == {"fpr_1e-3", "fpr_1e-4"}
    assert metrics["operating_points"]["fpr_1e-3"]["tp"] == 2


def test_the_operating_point_never_exceeds_its_false_positive_allowance() -> None:
    """The realised false-positive count respects floor(target * n_negatives).

    With 1000 negatives and a 1e-3 target the allowance is exactly one, so the
    cut must admit at most one negative.
    """
    generator = np.random.default_rng(20260825)
    negatives = generator.normal(-3.0, 1.0, size=1000)

    for target, expected_allowance in ((1e-3, 1), (1e-4, 0)):
        threshold, allowance = negative_only_threshold(negatives, target)
        assert allowance == expected_allowance
        assert int(np.sum(negatives >= threshold)) <= allowance


def test_the_threshold_ignores_the_positive_cohort_entirely() -> None:
    """Two mixtures sharing validation negatives must share the operating point.

    This is the property the previous rule lacked: it maximised true-positive
    rate, so adding noisy positives moved the cut and made the arms
    incomparable.
    """
    negatives = np.linspace(0.0, 0.5, 500)
    lean = np.concatenate([negatives, np.array([0.9])])
    rich = np.concatenate([negatives, np.linspace(0.6, 0.99, 400)])

    labels_lean = np.concatenate([np.zeros(500), np.ones(1)])
    labels_rich = np.concatenate([np.zeros(500), np.ones(400)])

    lean_cut, _ = negative_only_threshold(lean[labels_lean == 0], 1e-2)
    rich_cut, _ = negative_only_threshold(rich[labels_rich == 0], 1e-2)
    assert lean_cut == rich_cut


def test_a_tie_group_is_admitted_whole_or_not_at_all() -> None:
    """Ties are never split, so the allowance may bind more tightly than asked.

    Ten negatives share the top score while the allowance is five, so the cut
    must sit above the tie rather than admit half of it.
    """
    negatives = np.concatenate([np.zeros(90), np.full(10, 0.8)])

    threshold, allowance = negative_only_threshold(negatives, 0.05)

    assert allowance == 5
    assert int(np.sum(negatives >= threshold)) == 0
    assert threshold > 0.8


def test_a_saturated_float32_tie_cannot_overspend_the_allowance() -> None:
    """The float32 case that the gen6 fits actually hit.

    Sigmoid saturates to exactly 1.0 in float32, so the top tie can outnumber a
    strict allowance and the cut has to sit above 1.0. Seven negatives against
    an allowance of six is the shape of the real overspend. The cut must be
    chosen and compared in float64: no float32 value above 1.0 is reachable by
    widening the next float64 step, so a float32 comparison cannot express this
    cut at all.
    """
    negatives = np.concatenate(
        [np.zeros(61430, dtype=np.float32), np.full(7, 1.0, dtype=np.float32)]
    )

    threshold, allowance = negative_only_threshold(negatives, 1e-4)

    assert allowance == 6
    # Widened explicitly, because that is the contract the caller must honour:
    # comparing the float32 array directly is the demotion this guards against
    # and would itself admit the tie under NumPy 1.x.
    assert int(np.sum(negatives.astype(np.float64) >= threshold)) == 0
    assert threshold > 1.0
    # Why the comparison dtype is load-bearing rather than incidental: narrowed
    # to float32 this cut is exactly 1.0 and admits the whole tie.
    assert np.float32(threshold) == np.float32(1.0)


def test_the_operating_point_never_exceeds_its_allowance_on_saturated_scores(
    tmp_path: Path,
) -> None:
    """End to end through evaluate(), which owns the comparison.

    A dtype-safe threshold is not enough if evaluate() compares it against
    float32 predictions, so the guarantee is asserted where it is consumed.

    Discriminating only under NumPy 1.x value-based casting, which is what the
    training instances run: there a float64 cut compared against a float32 score
    array is demoted to float32 and admits the tie. Under NEP 50 (NumPy 2) the
    array is widened instead and the overspend cannot arise, so this passes
    either way and is a guard rather than a reproduction.
    """

    class _Saturated(torch.utils.data.Dataset):
        """Scores saturate at 1.0 for a tie of negatives wider than any allowance."""

        def __init__(self) -> None:
            # 40 negatives, of which 8 saturate, against a 1e-3 allowance of 0
            # and a 1e-2 allowance of 0 -- both bind below the tie width.
            self.labels = np.concatenate([np.zeros(40), np.ones(4)])
            self.saturated = set(range(8))

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, i: int):
            # A single feature the stub model turns into a logit.
            big = 40.0 if (i in self.saturated or self.labels[i] == 1.0) else -40.0
            return torch.tensor([big], dtype=torch.float32), torch.tensor(
                self.labels[i], dtype=torch.float32
            )

    class _Identity(torch.nn.Module):
        def forward(self, x):  # noqa: D102
            return x

    loader = torch.utils.data.DataLoader(_Saturated(), batch_size=8)
    metrics = evaluate(_Identity(), loader, torch.device("cpu"))

    for name, point in metrics["operating_points"].items():
        assert point["fp"] <= point["false_positive_allowance"], (
            f"{name} admitted {point['fp']} negatives on an allowance of "
            f"{point['false_positive_allowance']}"
        )


def test_calibration_refuses_an_empty_negative_set() -> None:
    with pytest.raises(ValueError, match="requires validation negatives"):
        negative_only_threshold(np.array([]), 1e-3)


def test_run_info_binds_manifest_lineage_code_and_environment(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame(
        {
            "cutout_relpath": ["a.npz", "b.npz"],
            "label": [1, 0],
            "sample_weight": [1.0, 1.0],
            "split": ["train", "val"],
            "cohort": ["clean_confirmed_lens", "clean_negative"],
        }
    ).to_parquet(manifest, index=False)
    lineage = tmp_path / "lineage.json"
    lineage.write_text('{"status":"bound"}\n')
    config = tmp_path / "config.yaml"
    config.write_text("config: test\n")
    out_dir = tmp_path / "run"
    dcfg = dataset_config(manifest, tmp_path)
    tcfg = TrainConfig(
        out_dir=str(out_dir),
        mixture_id="confirmed_only",
        lineage_path=str(lineage),
        pretrained=True,
    )

    record = _write_run_info(
        str(out_dir),
        str(config),
        dcfg,
        tcfg,
        status="started",
    )

    assert record["status"] == "started"
    assert record["manifest_summary"]["test_rows"] == 0
    assert len(record["manifest_sha256"]) == 64
    assert len(record["lineage_sha256"]) == 64
    assert record["initialization"]["imagenet_pretrained"] is True
    assert record["initialization"]["init_weights"] is None
    assert "dhs/train.py" in record["source_sha256"]
    assert (out_dir / "run_info.json").is_file()


# ---------------------------------------------------------------------------
# Amendment 3 invariants: natural-prevalence training and the checkpoint grid.
# ---------------------------------------------------------------------------


def test_natural_prevalence_sampler_takes_one_full_pass() -> None:
    """No sampler, and the epoch is the manifest, not a chosen example count."""
    frame = pd.DataFrame(
        {"label": [1] * 7 + [0] * 193, "sample_weight": [1.0] * 200}
    )

    class _Stub:
        df = frame

        def __len__(self) -> int:
            return len(self.df)

    dcfg = DatasetConfig(
        parquet_path="", mode="file_manifest", preprocessing="raw_robust",
        label_col="label", seed=42,
    )
    tcfg = TrainConfig(balanced_sampling=False, examples_per_epoch=316100)

    sampler, info = build_training_sampler(_Stub(), dcfg, tcfg)

    assert sampler is None
    assert info["mode"] == "natural_prevalence_shuffle"
    # examples_per_epoch on the config is inert once balancing is off.
    assert info["examples_per_epoch"] == 200
    assert info["positive_fraction"] == pytest.approx(0.035)


def test_periodic_grid_is_the_decades_plus_epoch_19() -> None:
    """Amendment 2's decade grid, plus the epoch-19 Phase-1 depth bridge."""
    assert PERIODIC_CHECKPOINT_EPOCHS == frozenset(
        {19, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160}
    )
    assert 19 in PERIODIC_CHECKPOINT_EPOCHS
    assert 160 in PERIODIC_CHECKPOINT_EPOCHS
    # The confirmatory epoch must be on the grid; stray epochs must not be.
    assert 159 not in PERIODIC_CHECKPOINT_EPOCHS
    assert 1 not in PERIODIC_CHECKPOINT_EPOCHS


def test_best_pt_is_never_a_test_facing_checkpoint() -> None:
    """best.pt is validation-selected and quarantined from test evaluation."""
    grid = {f"epoch_{e:03d}.pt" for e in PERIODIC_CHECKPOINT_EPOCHS}
    assert "best.pt" not in grid
    assert "last.pt" not in grid
    # The 96-model transaction enumerates decade checkpoints only.
    confirmatory = sorted(e for e in PERIODIC_CHECKPOINT_EPOCHS if e % 10 == 0)
    assert len(confirmatory) == 16
    assert max(confirmatory) == 160


def test_pretrained_weights_enum_is_pinned_not_default() -> None:
    """DEFAULT would silently retarget under a torchvision bump."""
    record = initialization_provenance("efficientnet_v2_s", pretrained=True)
    assert record["weights_enum"] == "IMAGENET1K_V1"
    assert record["torchvision_version"]
    assert initialization_provenance("efficientnet_v2_s", pretrained=False).get(
        "weights_enum"
    ) is None
