"""Exact manifest authorities carried through every downstream probe stage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import (
    ContractError,
    assert_score_blind,
    canonical_json_sha256,
    normalized_protocol_config_sha256,
    ordered_mapping_sha256,
    require_columns,
    require_nonempty_columns,
    require_sha256,
    require_sha256_column,
    sha256_file,
    validate_config,
    validate_reference_payloads,
    validate_reference_roster,
)


@dataclass(frozen=True)
class AuthorityContext:
    """Immutable content and file identities for references and host universe."""

    config_sha256: str
    reference_manifest_sha256: str
    reference_payload_manifest_sha256: str
    experiment_host_universe_sha256: str
    reference_payload_adapter_manifest_sha256: str
    experiment_host_adapter_manifest_sha256: str
    reference_ids: tuple[str, ...]
    reference_payload_items: tuple[tuple[str, str], ...]
    reference_physical_items: tuple[tuple[str, str], ...]
    reference_spatial_items: tuple[tuple[str, str], ...]
    reference_psf_items: tuple[tuple[str, str], ...]
    reference_render_items: tuple[tuple[str, str], ...]
    host_payload_items: tuple[tuple[str, str], ...]
    host_parent_items: tuple[tuple[str, str], ...]
    host_physical_items: tuple[tuple[str, str], ...]
    host_spatial_items: tuple[tuple[str, str], ...]
    authority_digest: str
    reference_manifest_path: Path
    reference_payload_manifest_path: Path
    experiment_host_universe_path: Path
    reference_payload_source_path: Path
    experiment_host_universe_source_path: Path
    reference_manifest_stat: tuple[int, int, int]
    reference_payload_manifest_stat: tuple[int, int, int]
    experiment_host_universe_stat: tuple[int, int, int]
    reference_payload_source_stat: tuple[int, int, int]
    experiment_host_universe_source_stat: tuple[int, int, int]

    @property
    def reference_payloads(self) -> dict[str, str]:
        return dict(self.reference_payload_items)

    @property
    def host_payloads(self) -> dict[str, str]:
        return dict(self.host_payload_items)

    @property
    def host_parents(self) -> dict[str, str]:
        return dict(self.host_parent_items)

    @property
    def host_physicals(self) -> dict[str, str]:
        return dict(self.host_physical_items)

    @property
    def host_spatials(self) -> dict[str, str]:
        return dict(self.host_spatial_items)

    @property
    def reference_psfs(self) -> dict[str, str]:
        return dict(self.reference_psf_items)

    @property
    def reference_renders(self) -> dict[str, str]:
        return dict(self.reference_render_items)

    def assert_config(self, config: Mapping[str, Any]) -> None:
        validate_config(config)
        current_hashes = {
            "reference_manifest_sha256": sha256_file(self.reference_manifest_path),
            "reference_payload_adapter_manifest_sha256": sha256_file(
                self.reference_payload_manifest_path
            ),
            "experiment_host_adapter_manifest_sha256": sha256_file(
                self.experiment_host_universe_path
            ),
            "reference_payload_manifest_sha256": sha256_file(
                self.reference_payload_source_path
            ),
            "experiment_host_universe_sha256": sha256_file(
                self.experiment_host_universe_source_path
            ),
        }
        if current_hashes != {
            "reference_payload_adapter_manifest_sha256": (
                self.reference_payload_adapter_manifest_sha256
            ),
            "experiment_host_adapter_manifest_sha256": (
                self.experiment_host_adapter_manifest_sha256
            ),
            "reference_manifest_sha256": self.reference_manifest_sha256,
            "reference_payload_manifest_sha256": (
                self.reference_payload_manifest_sha256
            ),
            "experiment_host_universe_sha256": (self.experiment_host_universe_sha256),
        }:
            raise ContractError("an authority file changed after context creation")
        if normalized_protocol_config_sha256(config) != self.config_sha256:
            raise ContractError(
                "downstream immutable protocol differs from authority context"
            )
        expected = config["authorities"]
        observed = {
            "reference_manifest_sha256": self.reference_manifest_sha256,
            "reference_payload_manifest_sha256": (
                self.reference_payload_manifest_sha256
            ),
            "experiment_host_universe_sha256": (self.experiment_host_universe_sha256),
            "reference_payload_adapter_manifest_sha256": (
                self.reference_payload_adapter_manifest_sha256
            ),
            "experiment_host_adapter_manifest_sha256": (
                self.experiment_host_adapter_manifest_sha256
            ),
        }
        for field, value in observed.items():
            if expected[field] != value:
                raise ContractError(f"authority context has wrong {field}")

    def assert_exact_references(
        self,
        frame: pd.DataFrame,
        *,
        id_column: str,
        context: str,
    ) -> None:
        require_nonempty_columns(frame, {id_column}, context=context)
        observed = tuple(sorted(frame[id_column].astype(str).unique()))
        if observed != self.reference_ids or len(frame) != 43:
            raise ContractError(
                f"{context} is not the exact authority reference roster"
            )

    def assert_reference_subset(
        self,
        frame: pd.DataFrame,
        *,
        id_column: str,
        context: str,
    ) -> None:
        require_nonempty_columns(frame, {id_column}, context=context)
        unknown = set(frame[id_column].astype(str)).difference(self.reference_ids)
        if unknown:
            raise ContractError(f"{context} contains non-authority references")

    def assert_reference_payloads(
        self,
        frame: pd.DataFrame,
        *,
        id_column: str,
        payload_column: str,
        exact: bool,
        context: str,
    ) -> None:
        if exact:
            self.assert_exact_references(frame, id_column=id_column, context=context)
        else:
            self.assert_reference_subset(frame, id_column=id_column, context=context)
        require_sha256_column(frame, payload_column, context=context)
        expected = self.reference_payloads
        for row in frame[[id_column, payload_column]].itertuples(index=False):
            if expected.get(str(row[0])) != str(row[1]):
                raise ContractError(f"{context} has a non-authority reference payload")

    def assert_reference_metadata(
        self,
        frame: pd.DataFrame,
        *,
        id_column: str,
        physical_column: str,
        spatial_column: str,
        exact: bool,
        context: str,
    ) -> None:
        if exact:
            self.assert_exact_references(frame, id_column=id_column, context=context)
        else:
            self.assert_reference_subset(frame, id_column=id_column, context=context)
        require_nonempty_columns(
            frame,
            {id_column, physical_column, spatial_column},
            context=context,
        )
        physicals = dict(self.reference_physical_items)
        spatials = dict(self.reference_spatial_items)
        for row in frame[[id_column, physical_column, spatial_column]].itertuples(
            index=False
        ):
            identifier, physical, spatial = map(str, row)
            if (
                physicals.get(identifier) != physical
                or spatials.get(identifier) != spatial
            ):
                raise ContractError(f"{context} has non-authority reference metadata")

    def assert_reference_render_metadata(
        self,
        frame: pd.DataFrame,
        *,
        id_column: str,
        psf_column: str,
        render_column: str,
        exact: bool,
        context: str,
    ) -> None:
        if exact:
            self.assert_exact_references(frame, id_column=id_column, context=context)
        else:
            self.assert_reference_subset(frame, id_column=id_column, context=context)
        require_nonempty_columns(
            frame,
            {id_column, psf_column, render_column},
            context=context,
        )
        if not self.reference_psfs or not self.reference_renders:
            raise ContractError("reference payload PSF/render authority is unavailable")
        for row in frame[[id_column, psf_column, render_column]].itertuples(
            index=False
        ):
            identifier, psf, render = map(str, row)
            if (
                self.reference_psfs.get(identifier) != psf
                or self.reference_renders.get(identifier) != render
            ):
                raise ContractError(
                    f"{context} has non-authority reference PSF/render metadata"
                )

    def assert_exact_host_universe(
        self,
        frame: pd.DataFrame,
        *,
        id_column: str,
        payload_column: str,
        context: str,
    ) -> None:
        require_nonempty_columns(frame, {id_column}, context=context)
        require_sha256_column(frame, payload_column, context=context)
        observed = {
            str(row[0]): str(row[1])
            for row in frame[[id_column, payload_column]].itertuples(index=False)
        }
        if observed != self.host_payloads or len(frame) != len(observed):
            raise ContractError(f"{context} is not the exact authority host universe")

    def assert_host_subset(
        self,
        frame: pd.DataFrame,
        *,
        id_column: str,
        payload_column: str,
        parent_column: str,
        physical_column: str | None = None,
        spatial_column: str | None = None,
        context: str,
    ) -> None:
        columns = [id_column, payload_column, parent_column]
        if physical_column is not None:
            columns.append(physical_column)
        if spatial_column is not None:
            columns.append(spatial_column)
        require_nonempty_columns(frame, columns, context=context)
        require_sha256_column(frame, payload_column, context=context)
        payloads = self.host_payloads
        parents = self.host_parents
        physicals = self.host_physicals
        spatials = self.host_spatials
        if physical_column is not None and not physicals:
            raise ContractError("final host physical-group authority is unavailable")
        for row in frame[columns].itertuples(index=False):
            values = list(map(str, row))
            identifier, payload, parent = values[:3]
            if payloads.get(identifier) != payload or parents.get(identifier) != parent:
                raise ContractError(f"{context} has non-authority host ancestry")
            position = 3
            if physical_column is not None:
                if physicals.get(identifier) != values[position]:
                    raise ContractError(f"{context} has wrong host physical group")
                position += 1
            if (
                spatial_column is not None
                and spatials.get(identifier) != values[position]
            ):
                raise ContractError(f"{context} has wrong host spatial group")


def load_authority_context(
    *,
    config: Mapping[str, Any],
    references_path: Path,
    reference_payloads_path: Path,
    host_universe_path: Path,
    reference_payload_adapter_path: Path | None = None,
    host_adapter_path: Path | None = None,
    development_ids: Iterable[str] = (),
) -> AuthorityContext:
    """Load exact physical files and seal their selected content in one context."""
    validate_config(config)
    expected = config["authorities"]
    observed_hashes = {
        "reference_manifest_sha256": sha256_file(references_path),
        "reference_payload_manifest_sha256": sha256_file(reference_payloads_path),
        "experiment_host_universe_sha256": sha256_file(host_universe_path),
    }
    for field, observed in observed_hashes.items():
        require_sha256(observed, context=field)
        if observed != expected[field]:
            raise ContractError(f"{field} differs from frozen authority")

    reference_payload_authority_path = (
        reference_payload_adapter_path or reference_payloads_path
    )
    host_authority_path = host_adapter_path or host_universe_path
    adapter_hashes = {
        "reference_payload_adapter_manifest_sha256": sha256_file(
            reference_payload_authority_path
        ),
        "experiment_host_adapter_manifest_sha256": sha256_file(host_authority_path),
    }
    for field, observed in adapter_hashes.items():
        configured = expected.get(field, observed)
        require_sha256(configured, context=field)
        if observed != configured:
            raise ContractError(f"{field} differs from frozen adapter authority")

    references = validate_reference_roster(
        pd.read_parquet(references_path),
        config=config,
        development_ids=development_ids,
    )
    reference_payload_frame = pd.read_parquet(reference_payload_authority_path)
    validate_reference_roster(
        reference_payload_frame,
        config=config,
        development_ids=development_ids,
    )
    payloads = validate_reference_payloads(references, reference_payload_frame)
    host_source_frame = pd.read_parquet(host_universe_path)
    host_frame = pd.read_parquet(host_authority_path)
    assert_score_blind(host_source_frame, context="source authority host universe")
    assert_score_blind(host_frame, context="authority host universe")
    role_column = config["G_match"]["host_universe_role_column"]
    role_value = config["G_match"]["host_universe_role_value"]
    parent_column = expected["experiment_host_parent_id_column"]
    payload_column = expected["experiment_host_payload_hash_column"]
    require_columns(
        host_frame,
        {
            "galaxy_id",
            role_column,
            payload_column,
            "evaluation_spatial_group_id",
        },
        context="authority host universe",
    )
    require_columns(
        host_source_frame,
        {"galaxy_id", role_column, "payload_sha256", "evaluation_spatial_group_id"},
        context="source authority host universe",
    )
    selected_hosts = host_frame[host_frame[role_column].eq(role_value)].copy()
    source_hosts = host_source_frame[
        host_source_frame[role_column].eq(role_value)
    ].copy()
    source_lookup = source_hosts.set_index(source_hosts["galaxy_id"].astype(str))
    if not set(selected_hosts["galaxy_id"].astype(str)).issubset(source_lookup.index):
        raise ContractError("host adapter is not a subset of the authority universe")
    for row in selected_hosts[
        ["galaxy_id", payload_column, "evaluation_spatial_group_id"]
    ].itertuples(index=False):
        source = source_lookup.loc[str(row[0])]
        if (
            str(source["payload_sha256"]) != str(row[1])
            or str(source["evaluation_spatial_group_id"]) != str(row[2])
        ):
            raise ContractError("host adapter changed source payload or spatial group")
    if parent_column == "galaxy_id":
        selected_hosts["_authority_parent_id"] = selected_hosts["galaxy_id"].astype(str)
        parent_column = "_authority_parent_id"
    require_nonempty_columns(
        selected_hosts,
        {"galaxy_id", parent_column, "evaluation_spatial_group_id"},
        context="authority host universe",
    )
    require_sha256_column(
        selected_hosts,
        payload_column,
        context="authority host universe",
        unique=True,
    )
    if selected_hosts["galaxy_id"].astype(str).duplicated().any():
        raise ContractError("authority host universe contains duplicate IDs")

    reference_ids = tuple(sorted(references["galaxy_id"].astype(str)))
    reference_payload_map = {
        str(row.galaxy_id): str(row.payload_sha256)
        for row in payloads[["galaxy_id", "payload_sha256"]].itertuples(index=False)
    }
    reference_physical_map = {
        str(row.galaxy_id): str(row.physical_group_id)
        for row in references[["galaxy_id", "physical_group_id"]].itertuples(
            index=False
        )
    }
    reference_spatial_map = {
        str(row.galaxy_id): str(row.spatial_component_id)
        for row in references[["galaxy_id", "spatial_component_id"]].itertuples(
            index=False
        )
    }
    reference_psf_column = expected["reference_psf_id_column"]
    reference_render_column = expected["reference_render_parameter_id_column"]
    if (
        reference_psf_column in reference_payload_frame
        or reference_render_column in reference_payload_frame
    ):
        require_nonempty_columns(
            reference_payload_frame,
            {reference_psf_column, reference_render_column},
            context="reference payload authority",
        )
    reference_psf_map = (
        dict(
            zip(
                reference_payload_frame["galaxy_id"].astype(str),
                reference_payload_frame[reference_psf_column].astype(str),
                strict=True,
            )
        )
        if reference_psf_column in reference_payload_frame
        else {}
    )
    reference_render_map = (
        dict(
            zip(
                reference_payload_frame["galaxy_id"].astype(str),
                reference_payload_frame[reference_render_column].astype(str),
                strict=True,
            )
        )
        if reference_render_column in reference_payload_frame
        else {}
    )
    host_payload_map = {
        str(row[0]): str(row[1])
        for row in selected_hosts[["galaxy_id", payload_column]].itertuples(index=False)
    }
    host_parent_map = {
        str(row[0]): str(row[1])
        for row in selected_hosts[["galaxy_id", parent_column]].itertuples(index=False)
    }
    physical_column = expected["experiment_host_physical_group_column"]
    spatial_column = expected["experiment_host_spatial_group_column"]
    if physical_column in selected_hosts:
        require_nonempty_columns(
            selected_hosts,
            {physical_column},
            context="authority host universe",
        )
    host_physical_map = (
        {
            str(row[0]): str(row[1])
            for row in selected_hosts[["galaxy_id", physical_column]].itertuples(
                index=False
            )
        }
        if physical_column in selected_hosts
        else {}
    )
    host_spatial_map = {
        str(row[0]): str(row[1])
        for row in selected_hosts[["galaxy_id", spatial_column]].itertuples(index=False)
    }
    config_digest = normalized_protocol_config_sha256(config)
    authority_payload = {
        "config_sha256": config_digest,
        **observed_hashes,
        **adapter_hashes,
        "reference_ids_sha256": ordered_mapping_sha256(
            {identifier: identifier for identifier in reference_ids}
        ),
        "reference_payloads_sha256": ordered_mapping_sha256(reference_payload_map),
        "reference_physicals_sha256": ordered_mapping_sha256(reference_physical_map),
        "reference_spatials_sha256": ordered_mapping_sha256(reference_spatial_map),
        "reference_psfs_sha256": ordered_mapping_sha256(reference_psf_map),
        "reference_renders_sha256": ordered_mapping_sha256(reference_render_map),
        "host_payloads_sha256": ordered_mapping_sha256(host_payload_map),
        "host_parents_sha256": ordered_mapping_sha256(host_parent_map),
        "host_physicals_sha256": ordered_mapping_sha256(host_physical_map),
        "host_spatials_sha256": ordered_mapping_sha256(host_spatial_map),
    }
    return AuthorityContext(
        config_sha256=config_digest,
        reference_manifest_sha256=observed_hashes["reference_manifest_sha256"],
        reference_payload_manifest_sha256=observed_hashes[
            "reference_payload_manifest_sha256"
        ],
        experiment_host_universe_sha256=observed_hashes[
            "experiment_host_universe_sha256"
        ],
        reference_payload_adapter_manifest_sha256=adapter_hashes[
            "reference_payload_adapter_manifest_sha256"
        ],
        experiment_host_adapter_manifest_sha256=adapter_hashes[
            "experiment_host_adapter_manifest_sha256"
        ],
        reference_ids=reference_ids,
        reference_payload_items=tuple(sorted(reference_payload_map.items())),
        reference_physical_items=tuple(sorted(reference_physical_map.items())),
        reference_spatial_items=tuple(sorted(reference_spatial_map.items())),
        reference_psf_items=tuple(sorted(reference_psf_map.items())),
        reference_render_items=tuple(sorted(reference_render_map.items())),
        host_payload_items=tuple(sorted(host_payload_map.items())),
        host_parent_items=tuple(sorted(host_parent_map.items())),
        host_physical_items=tuple(sorted(host_physical_map.items())),
        host_spatial_items=tuple(sorted(host_spatial_map.items())),
        authority_digest=canonical_json_sha256(authority_payload),
        reference_manifest_path=references_path.resolve(),
        reference_payload_manifest_path=reference_payload_authority_path.resolve(),
        experiment_host_universe_path=host_authority_path.resolve(),
        reference_payload_source_path=reference_payloads_path.resolve(),
        experiment_host_universe_source_path=host_universe_path.resolve(),
        reference_manifest_stat=_stat_identity(references_path),
        reference_payload_manifest_stat=_stat_identity(
            reference_payload_authority_path
        ),
        experiment_host_universe_stat=_stat_identity(host_authority_path),
        reference_payload_source_stat=_stat_identity(reference_payloads_path),
        experiment_host_universe_source_stat=_stat_identity(host_universe_path),
    )


def _stat_identity(path: Path) -> tuple[int, int, int]:
    status = path.stat()
    return (int(status.st_ino), int(status.st_size), int(status.st_mtime_ns))
