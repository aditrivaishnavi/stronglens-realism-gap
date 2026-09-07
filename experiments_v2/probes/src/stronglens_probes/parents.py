"""Hash-bound raw Tractor and G_rec parent manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .authorities import AuthorityContext
from .contracts import (
    ContractError,
    assert_score_blind,
    require_columns,
    require_nonempty_columns,
    require_sha256,
    require_sha256_column,
    sha256_file,
)

RAW_TRACTOR_COLUMNS = (
    "galaxy_id",
    "flux_z",
    "sersic",
    "shape_e1",
    "shape_e2",
    "psfsize_r",
    "psfdepth_r",
)

PARENT_IDENTITY_COLUMNS = (
    "grec_row_id",
    "scene_id",
    "host_galaxy_id",
    "host_parent_id",
    "host_pixel_sha256",
    "host_physical_group_id",
    "host_spatial_group_id",
    "source_parameter_id",
    "lens_parameter_id",
    "psf_id",
    "rendering_id",
    "render_parameter_id",
    "random_draw_id",
    "theta_e_arcsec",
    "intrinsic_flux_g_nmgy",
    "intrinsic_flux_r_nmgy",
    "intrinsic_flux_z_nmgy",
    "source_flux_parameter_sha256",
    "final_image_sha256",
)


@dataclass(frozen=True)
class BoundTractorRows:
    frame: pd.DataFrame
    path: Path
    sha256: str
    authority_field: str

    def revalidate(self, config: dict[str, Any]) -> None:
        expected = config["authorities"].get(self.authority_field)
        require_sha256(expected, context=self.authority_field)
        if sha256_file(self.path) != self.sha256 or self.sha256 != expected:
            raise ContractError(f"{self.authority_field} changed after loading")
        try:
            pd.testing.assert_frame_equal(
                self.frame.reset_index(drop=True),
                pd.read_parquet(self.path).reset_index(drop=True),
                check_exact=True,
            )
        except AssertionError as error:
            raise ContractError(
                f"{self.authority_field} mutated after loading"
            ) from error


@dataclass(frozen=True)
class GRecArtifacts:
    scenes: pd.DataFrame
    hosts: pd.DataFrame
    tractor: BoundTractorRows
    scene_path: Path
    host_path: Path
    scene_sha256: str
    host_sha256: str

    def revalidate(
        self,
        *,
        config: dict[str, Any],
        authority: AuthorityContext,
    ) -> None:
        authority.assert_config(config)
        expected = config["authorities"]
        if (
            sha256_file(self.scene_path) != self.scene_sha256
            or self.scene_sha256 != expected.get("grec_scene_manifest_sha256")
            or sha256_file(self.host_path) != self.host_sha256
            or self.host_sha256 != expected.get("grec_host_manifest_sha256")
        ):
            raise ContractError("G_rec parent manifests changed after loading")
        self.tractor.revalidate(config)
        try:
            pd.testing.assert_frame_equal(
                self.scenes.reset_index(drop=True),
                pd.read_parquet(self.scene_path).reset_index(drop=True),
                check_exact=True,
            )
            pd.testing.assert_frame_equal(
                self.hosts.reset_index(drop=True),
                pd.read_parquet(self.host_path).reset_index(drop=True),
                check_exact=True,
            )
        except AssertionError as error:
            raise ContractError(
                "G_rec in-memory parents mutated after loading"
            ) from error


def load_bound_tractor_rows(
    *,
    path: Path,
    authority_field: str,
    config: dict[str, Any],
    authority: AuthorityContext,
    population: str,
) -> BoundTractorRows:
    """Load raw, un-derived Tractor rows from one immutable authority file."""
    authority.assert_config(config)
    observed_sha256 = sha256_file(path)
    expected_sha256 = config["authorities"].get(authority_field)
    require_sha256(expected_sha256, context=authority_field)
    if observed_sha256 != expected_sha256:
        raise ContractError(f"{authority_field} differs from immutable config")
    frame = pd.read_parquet(path)
    assert_score_blind(frame, context=authority_field)
    require_nonempty_columns(frame, {"galaxy_id"}, context=authority_field)
    require_columns(
        frame,
        {*RAW_TRACTOR_COLUMNS, "payload_sha256"},
        context=authority_field,
    )
    require_sha256_column(frame, "payload_sha256", context=authority_field)
    if frame["galaxy_id"].astype(str).duplicated().any():
        raise ContractError(f"{authority_field} has duplicate galaxy IDs")
    if population == "reference":
        authority.assert_exact_references(
            frame,
            id_column="galaxy_id",
            context=authority_field,
        )
        authority.assert_reference_payloads(
            frame,
            id_column="galaxy_id",
            payload_column="payload_sha256",
            exact=True,
            context=authority_field,
        )
    elif population == "host":
        authority.assert_exact_host_universe(
            frame,
            id_column="galaxy_id",
            payload_column="payload_sha256",
            context=authority_field,
        )
    else:
        raise ContractError(f"unknown Tractor population {population}")
    return BoundTractorRows(
        frame=frame,
        path=path.resolve(),
        sha256=observed_sha256,
        authority_field=authority_field,
    )


def load_grec_artifacts(
    *,
    scene_path: Path,
    host_path: Path,
    host_tractor_path: Path,
    config: dict[str, Any],
    authority: AuthorityContext,
) -> GRecArtifacts:
    """Load and cross-check every G_rec scene against host and Tractor parents."""
    authority.assert_config(config)
    expected = config["authorities"]
    scene_sha256 = sha256_file(scene_path)
    host_sha256 = sha256_file(host_path)
    for field, observed in (
        ("grec_scene_manifest_sha256", scene_sha256),
        ("grec_host_manifest_sha256", host_sha256),
    ):
        require_sha256(expected.get(field), context=field)
        if expected[field] != observed:
            raise ContractError(f"{field} differs from immutable config")
    scenes = pd.read_parquet(scene_path)
    hosts = pd.read_parquet(host_path)
    assert_score_blind(scenes, context="G_rec scene manifest")
    assert_score_blind(hosts, context="G_rec host manifest")
    require_nonempty_columns(
        scenes, PARENT_IDENTITY_COLUMNS, context="G_rec scene manifest"
    )
    require_sha256_column(scenes, "host_pixel_sha256", context="G_rec scene manifest")
    require_sha256_column(scenes, "final_image_sha256", context="G_rec scene manifest")
    require_sha256_column(
        scenes,
        "source_flux_parameter_sha256",
        context="G_rec scene manifest",
    )
    if scenes["grec_row_id"].astype(str).duplicated().any():
        raise ContractError("G_rec scene rows are duplicated")
    require_nonempty_columns(
        hosts,
        {
            "galaxy_id",
            "host_parent_id",
            "host_physical_group_id",
            "host_spatial_group_id",
            "psf_bin",
            "depth_bin",
        },
        context="G_rec host manifest",
    )
    require_sha256_column(hosts, "payload_sha256", context="G_rec host manifest")
    if hosts["galaxy_id"].astype(str).duplicated().any():
        raise ContractError("G_rec host rows are duplicated")
    contract = config["G_domain"]["G_rec_contract"]
    reduced_fixture = bool(
        config["execution"].get("allow_reduced_grec_design_fixture", False)
    )
    if not reduced_fixture:
        if len(hosts) != int(contract["hosts"]) or len(scenes) != int(
            contract["scenes"]
        ):
            raise ContractError("G_rec host/scene totals violate the 4500x11 contract")
        cell_counts = hosts.groupby(["psf_bin", "depth_bin"]).size()
        if (
            len(cell_counts) != int(contract["condition_cells"])
            or not cell_counts.eq(int(contract["hosts_per_cell"])).all()
        ):
            raise ContractError("G_rec condition cells are not nine by 500 hosts")
    theta = sorted(pd.to_numeric(scenes["theta_e_arcsec"]).unique().tolist())
    expected_theta = [float(value) for value in contract["theta_e_arcsec"]]
    if theta != expected_theta:
        raise ContractError("G_rec scenes do not contain the exact eleven radii")
    per_host = scenes.groupby("host_galaxy_id")["theta_e_arcsec"]
    if not per_host.nunique().eq(len(expected_theta)).all():
        raise ContractError("G_rec host lacks one complete eleven-radius sequence")
    if not scenes.groupby(["host_galaxy_id", "theta_e_arcsec"]).size().eq(1).all():
        raise ContractError("G_rec host/radius scene is duplicated")
    for column in contract["shared_nuisance_columns"]:
        if (
            not scenes.groupby("host_galaxy_id")[column]
            .nunique(dropna=False)
            .eq(1)
            .all()
        ):
            raise ContractError(f"G_rec nuisance {column} changes across radii")
    authority.assert_host_subset(
        hosts,
        id_column="galaxy_id",
        payload_column="payload_sha256",
        parent_column="host_parent_id",
        physical_column="host_physical_group_id",
        spatial_column="host_spatial_group_id",
        context="G_rec host manifest",
    )
    host_lookup = hosts.set_index(hosts["galaxy_id"].astype(str))
    for scene in scenes.itertuples(index=False):
        host_id = str(scene.host_galaxy_id)
        if host_id not in host_lookup.index:
            raise ContractError(f"G_rec scene has unknown host {host_id}")
        host = host_lookup.loc[host_id]
        comparisons = {
            "host_parent_id": host["host_parent_id"],
            "host_physical_group_id": host["host_physical_group_id"],
            "host_spatial_group_id": host["host_spatial_group_id"],
        }
        if "host_payload_sha256" in scenes:
            comparisons["host_payload_sha256"] = host["payload_sha256"]
        else:
            comparisons["host_pixel_sha256"] = host["payload_sha256"]
        if "source_host_pixel_sha256" in hosts:
            comparisons["host_pixel_sha256"] = host["source_host_pixel_sha256"]
        for column, expected_value in comparisons.items():
            if str(getattr(scene, column)) != str(expected_value):
                raise ContractError(f"G_rec scene changed host field {column}")
    tractor = load_bound_tractor_rows(
        path=host_tractor_path,
        authority_field="experiment_host_tractor_manifest_sha256",
        config=config,
        authority=authority,
        population="host",
    )
    missing_tractor = set(hosts["galaxy_id"].astype(str)).difference(
        tractor.frame["galaxy_id"].astype(str)
    )
    if missing_tractor:
        raise ContractError("G_rec hosts lack bound raw Tractor rows")
    return GRecArtifacts(
        scenes=scenes,
        hosts=hosts,
        tractor=tractor,
        scene_path=scene_path.resolve(),
        host_path=host_path.resolve(),
        scene_sha256=scene_sha256,
        host_sha256=host_sha256,
    )
