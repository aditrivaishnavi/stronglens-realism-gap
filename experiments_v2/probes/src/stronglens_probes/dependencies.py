"""Typed dependency components and deterministic pre-feature fold allocation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import ContractError, require_columns, stable_digest


@dataclass
class _UnionFind:
    parent: np.ndarray
    rank: np.ndarray

    @classmethod
    def create(cls, size: int) -> _UnionFind:
        return cls(np.arange(size, dtype=np.int64), np.zeros(size, dtype=np.int8))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = int(self.parent[index])
        return index

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _valid_dependency(value: object) -> bool:
    if pd.isna(value):
        return False
    normalized = str(value).strip()
    return normalized not in {"", "None", "nan"} and not normalized.startswith(
        "not_applicable:"
    )


def geographic_spatial_key(healpix_128: object) -> str:
    """Return one typed NSIDE=128 nested HEALPix dependency namespace."""
    if pd.isna(healpix_128):
        raise ContractError("geographic spatial key cannot be null")
    try:
        numeric = float(healpix_128)
        integer = int(numeric)
    except (TypeError, ValueError, OverflowError) as error:
        raise ContractError("geographic spatial key must be an integer") from error
    if numeric != integer or not 0 <= integer < 12 * 128**2:
        raise ContractError("geographic spatial key is outside NSIDE=128")
    return f"healpix128:{integer}"


def build_dependency_components(
    frame: pd.DataFrame,
    *,
    row_id_column: str,
    dependency_columns: Iterable[str],
    output_column: str = "dependency_component_id",
) -> pd.DataFrame:
    """Union rows sharing any registered dependency of the same declared type."""
    require_columns(frame, {row_id_column}, context="dependency input")
    if (
        frame[row_id_column].isna().any()
        or frame[row_id_column].astype(str).duplicated().any()
    ):
        raise ContractError("dependency row identifiers must be non-null and unique")
    columns = list(dependency_columns)
    require_columns(frame, columns, context="dependency input")
    if not columns:
        raise ContractError("no registered dependency column is present")
    if frame[columns].isna().any().any():
        raise ContractError("dependency input contains null dependency values")
    result = frame.copy().reset_index(drop=True)
    union = _UnionFind.create(len(result))
    for column in columns:
        groups: dict[str, list[int]] = {}
        for index, value in enumerate(result[column]):
            if _valid_dependency(value):
                groups.setdefault(str(value), []).append(index)
        for indices in groups.values():
            anchor = indices[0]
            for index in indices[1:]:
                union.union(anchor, index)
    members: dict[int, list[str]] = {}
    for index, row_id in enumerate(result[row_id_column].astype(str)):
        members.setdefault(union.find(index), []).append(row_id)
    component_ids = {
        root: f"dep-{stable_digest(*sorted(row_ids))[:24]}"
        for root, row_ids in members.items()
    }
    result[output_column] = [
        component_ids[union.find(index)] for index in range(len(result))
    ]
    return result


def assign_dependency_folds(
    frame: pd.DataFrame,
    *,
    component_column: str,
    label_column: str,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    """Greedily balance class rows while never splitting a dependency component."""
    if n_splits < 2:
        raise ContractError("at least two folds are required")
    require_columns(
        frame,
        {component_column, label_column},
        context="fold allocation input",
    )
    labels = sorted(pd.unique(frame[label_column]).tolist())
    if labels != [0, 1]:
        raise ContractError("fold labels must be exactly binary integers 0 and 1")
    counts = (
        frame.groupby([component_column, label_column])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=labels, fill_value=0)
    )
    for label in labels:
        if int((counts[label] > 0).sum()) < n_splits:
            raise ContractError(
                f"class {label} has fewer dependency components than folds"
            )
    target = counts.sum(axis=0).to_numpy(dtype=float) / n_splits
    normalized_size = (counts.to_numpy(dtype=float) / np.maximum(target, 1.0)).max(
        axis=1
    )
    ordered = sorted(
        range(len(counts)),
        key=lambda index: (
            -float(normalized_size[index]),
            -int(counts.iloc[index].sum()),
            stable_digest(seed, counts.index[index]),
        ),
    )
    state = np.zeros((n_splits, len(labels)), dtype=float)
    group_counts = np.zeros(n_splits, dtype=int)
    assignment: dict[str, int] = {}
    for index in ordered:
        component = str(counts.index[index])
        contribution = counts.iloc[index].to_numpy(dtype=float)
        candidates: list[tuple[float, int, str, int]] = []
        for fold in range(n_splits):
            trial = state.copy()
            trial[fold] += contribution
            class_loss = float(
                np.square((trial - target[None, :]) / np.maximum(target, 1.0)).sum()
            )
            candidates.append(
                (
                    class_loss,
                    int(group_counts[fold]),
                    stable_digest(seed, component, fold),
                    fold,
                )
            )
        fold = min(candidates)[-1]
        state[fold] += contribution
        group_counts[fold] += 1
        assignment[component] = int(fold)
    output = pd.DataFrame(
        {
            component_column: list(assignment),
            "fold": [assignment[component] for component in assignment],
        }
    ).sort_values(component_column, ignore_index=True)
    allocated = frame[[component_column, label_column]].merge(
        output, on=component_column, how="left", validate="many_to_one"
    )
    if allocated["fold"].isna().any():
        raise ContractError("fold allocation omitted a dependency component")
    support = allocated.groupby(["fold", label_column]).size().unstack(fill_value=0)
    if set(support.index) != set(range(n_splits)) or (support == 0).any().any():
        raise ContractError("at least one fold lacks one class")
    return output


def assert_dependency_closed_folds(
    frame: pd.DataFrame,
    fold_map: pd.DataFrame,
    *,
    component_column: str,
    dependency_columns: Iterable[str],
) -> None:
    require_columns(fold_map, {component_column, "fold"}, context="fold map")
    if fold_map[component_column].duplicated().any():
        raise ContractError("fold map contains duplicate components")
    joined = frame.merge(
        fold_map, on=component_column, how="left", validate="many_to_one"
    )
    if joined["fold"].isna().any():
        raise ContractError("fold map does not cover all rows")
    for column in dependency_columns:
        if column not in joined:
            continue
        valid = joined[column].map(_valid_dependency)
        if not valid.any():
            continue
        crossings = joined.loc[valid].groupby(column)["fold"].nunique()
        if (crossings > 1).any():
            example = str(crossings[crossings > 1].index[0])
            raise ContractError(f"dependency {column}={example} crosses folds")
