from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stronglens_probes.authorities import AuthorityContext, load_authority_context
from stronglens_probes.contracts import load_config, sha256_file
from stronglens_probes.execution import _source_tree_sha256
from stronglens_probes.parents import (
    BoundTractorRows,
    GRecArtifacts,
    load_bound_tractor_rows,
    load_grec_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config() -> dict:
    return load_config(ROOT / "configs/probes.yaml")


@pytest.fixture
def references() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "galaxy_id": [f"ref-{index:02d}" for index in range(43)],
            "endpoint_role": ["terminal_confirmed"] * 43,
            "label": np.ones(43, dtype=int),
            "lensing_status": ["confirmed_lens"] * 43,
            "final_lensing_status": ["confirmed_lens"] * 43,
            "split": ["test"] * 43,
            "healpix_128": np.arange(43, dtype=np.int64),
            "physical_group_id": [f"ref-physical-{index:02d}" for index in range(43)],
            "spatial_component_id": [f"ref-spatial-{index:02d}" for index in range(43)],
        }
    )


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _psfdepth(depth: np.ndarray) -> np.ndarray:
    return np.square(5.0 / np.power(10.0, (22.5 - depth) / 2.5))


@dataclass
class AuthorityBundle:
    config: dict
    context: AuthorityContext
    references: pd.DataFrame
    reference_payloads: pd.DataFrame
    hosts: pd.DataFrame
    references_path: Path
    reference_payloads_path: Path
    hosts_path: Path
    reference_tractor: BoundTractorRows
    host_tractor: BoundTractorRows
    grec: GRecArtifacts
    checkpoint_paths: dict[str, Path]


@pytest.fixture
def authority_bundle(
    tmp_path: Path,
    config: dict,
    references: pd.DataFrame,
) -> AuthorityBundle:
    frozen = copy.deepcopy(config)
    frozen["execution"]["source_tree_sha256"] = _source_tree_sha256()
    frozen["execution"]["dependency_lock_sha256"] = sha256_file(ROOT / "uv.lock")
    frozen["execution"]["allow_reduced_grec_design_fixture"] = True
    reference_payloads = references.copy()
    reference_payloads["payload_sha256"] = [
        digest(f"reference-pixel-{index}") for index in range(43)
    ]
    reference_payloads["valid"] = 1
    reference_payloads["bucket"] = "positive"
    reference_payloads["psf_id"] = [f"reference-psf-{index}" for index in range(43)]
    reference_payloads["render_parameter_id"] = [
        f"reference-render-{index}" for index in range(43)
    ]
    hosts = pd.DataFrame(
        {
            "galaxy_id": [f"host-{index:03d}" for index in range(80)],
            "payload_sha256": [digest(f"host-pixel-{index}") for index in range(80)],
            "proposed_role": ["experiment_host"] * 80,
            "evaluation_spatial_group_id": [
                f"host-spatial-{index}" for index in range(80)
            ],
            "host_healpix_128": np.arange(100, 180, dtype=np.int64),
            "host_physical_group_id": [f"host-physical-{index}" for index in range(80)],
            "psf_bin": [1.0 + 0.1 * (index % 4) for index in range(80)],
            "depth_bin": [24.0 + 0.5 * ((index // 4) % 2) for index in range(80)],
        }
    )

    reference_coordinate = np.linspace(-1.0, 1.0, 43)
    host_coordinate = np.linspace(-1.1, 1.1, 80)
    reference_tractor_frame = pd.DataFrame(
        {
            "galaxy_id": reference_payloads["galaxy_id"],
            "payload_sha256": reference_payloads["payload_sha256"],
            "flux_z": np.power(
                10.0, (22.5 - (18.5 + 0.3 * reference_coordinate)) / 2.5
            ),
            "sersic": 4.5 + 0.2 * reference_coordinate,
            "shape_e1": (1.0 - (0.75 + 0.05 * reference_coordinate))
            / (1.0 + (0.75 + 0.05 * reference_coordinate)),
            "shape_e2": np.zeros(43),
            "psfsize_r": 1.3 + 0.1 * reference_coordinate,
            "psfdepth_r": _psfdepth(24.0 + 0.2 * reference_coordinate),
        }
    )
    host_tractor_frame = hosts.copy()
    host_tractor_frame["flux_z"] = np.power(
        10.0, (22.5 - (18.5 + 0.3 * host_coordinate)) / 2.5
    )
    host_tractor_frame["sersic"] = 4.5 + 0.2 * host_coordinate
    host_q = 0.75 + 0.05 * host_coordinate
    host_tractor_frame["shape_e1"] = (1.0 - host_q) / (1.0 + host_q)
    host_tractor_frame["shape_e2"] = 0.0
    host_tractor_frame["psfsize_r"] = 1.3 + 0.1 * host_coordinate
    host_tractor_frame["psfdepth_r"] = _psfdepth(24.0 + 0.2 * host_coordinate)
    boundary_q = np.asarray([0.6, 0.8, 0.8, 0.5999])
    boundary_mag = np.asarray([19.5, 19.5001, 19.0, 19.0])
    host_tractor_frame.loc[:3, "flux_z"] = np.power(10.0, (22.5 - boundary_mag) / 2.5)
    host_tractor_frame.loc[:3, "sersic"] = [4.0, 5.0, 3.9999, 5.0]
    host_tractor_frame.loc[:3, "shape_e1"] = (1.0 - boundary_q) / (1.0 + boundary_q)

    grec_hosts = hosts.rename(
        columns={"evaluation_spatial_group_id": "host_spatial_group_id"}
    ).copy()
    grec_hosts["host_parent_id"] = grec_hosts["galaxy_id"]
    grec_scene_rows = []
    for index, host in grec_hosts.iterrows():
        for radius in frozen["G_domain"]["G_rec_contract"]["theta_e_arcsec"]:
            grec_scene_rows.append(
                {
                    "grec_row_id": f"grec-{index:03d}-{radius:.2f}",
                    "scene_id": f"scene-{index:03d}-{radius:.2f}",
                    "host_galaxy_id": host["galaxy_id"],
                    "host_parent_id": host["host_parent_id"],
                    "host_pixel_sha256": host["payload_sha256"],
                    "host_physical_group_id": host["host_physical_group_id"],
                    "host_spatial_group_id": host["host_spatial_group_id"],
                    "source_parameter_id": f"source-{index:03d}",
                    "lens_parameter_id": f"lens-{index:03d}",
                    "psf_id": f"psf-{index:03d}",
                    "rendering_id": f"rendering-{index:03d}",
                    "render_parameter_id": f"render-parameter-{index:03d}",
                    "random_draw_id": f"random-{index:03d}",
                    "theta_e_arcsec": float(radius),
                    "intrinsic_flux_g_nmgy": 1.0,
                    "intrinsic_flux_r_nmgy": 2.0,
                    "intrinsic_flux_z_nmgy": 3.0,
                    "source_flux_parameter_sha256": digest(f"flux-{index}"),
                    "final_image_sha256": digest(f"final-{index}-{radius}"),
                }
            )
    grec_scene_frame = pd.DataFrame(grec_scene_rows)

    paths = {
        "references": tmp_path / "references.parquet",
        "reference_payloads": tmp_path / "reference_payloads.parquet",
        "hosts": tmp_path / "host_authority.parquet",
        "reference_tractor": tmp_path / "reference_tractor.parquet",
        "host_tractor": tmp_path / "host_tractor.parquet",
        "grec_hosts": tmp_path / "grec_hosts.parquet",
        "grec_scenes": tmp_path / "grec_scenes.parquet",
        "stein_checkpoint": tmp_path / "stein.ckpt",
        "imagenet_checkpoint": tmp_path / "imagenet.pth",
    }
    for frame, name in (
        (references, "references"),
        (reference_payloads, "reference_payloads"),
        (hosts, "hosts"),
        (reference_tractor_frame, "reference_tractor"),
        (host_tractor_frame, "host_tractor"),
        (grec_hosts, "grec_hosts"),
        (grec_scene_frame, "grec_scenes"),
    ):
        frame.to_parquet(paths[name], index=False)
    paths["stein_checkpoint"].write_bytes(b"design-only-stein-checkpoint")
    paths["imagenet_checkpoint"].write_bytes(b"design-only-imagenet-checkpoint")
    frozen["representations"]["primary"]["checkpoint_sha256"] = sha256_file(
        paths["stein_checkpoint"]
    )
    frozen["representations"]["required_diagnostics"][0]["checkpoint_sha256"] = (
        sha256_file(paths["imagenet_checkpoint"])
    )

    authority_fields = {
        "reference_manifest_sha256": "references",
        "reference_payload_manifest_sha256": "reference_payloads",
        "reference_payload_adapter_manifest_sha256": "reference_payloads",
        "experiment_host_universe_sha256": "hosts",
        "experiment_host_adapter_manifest_sha256": "hosts",
        "reference_tractor_manifest_sha256": "reference_tractor",
        "experiment_host_tractor_manifest_sha256": "host_tractor",
        "grec_host_manifest_sha256": "grec_hosts",
        "grec_scene_manifest_sha256": "grec_scenes",
    }
    for field, name in authority_fields.items():
        frozen["authorities"][field] = sha256_file(paths[name])
    frozen["authorities"]["experiment_host_parent_id_column"] = "galaxy_id"
    frozen["authorities"]["experiment_host_spatial_group_column"] = (
        "evaluation_spatial_group_id"
    )

    context = load_authority_context(
        config=frozen,
        references_path=paths["references"],
        reference_payloads_path=paths["reference_payloads"],
        host_universe_path=paths["hosts"],
    )
    reference_tractor = load_bound_tractor_rows(
        path=paths["reference_tractor"],
        authority_field="reference_tractor_manifest_sha256",
        config=frozen,
        authority=context,
        population="reference",
    )
    host_tractor = load_bound_tractor_rows(
        path=paths["host_tractor"],
        authority_field="experiment_host_tractor_manifest_sha256",
        config=frozen,
        authority=context,
        population="host",
    )
    grec = load_grec_artifacts(
        scene_path=paths["grec_scenes"],
        host_path=paths["grec_hosts"],
        host_tractor_path=paths["host_tractor"],
        config=frozen,
        authority=context,
    )
    return AuthorityBundle(
        config=frozen,
        context=context,
        references=references,
        reference_payloads=reference_payloads,
        hosts=grec_hosts,
        references_path=paths["references"],
        reference_payloads_path=paths["reference_payloads"],
        hosts_path=paths["hosts"],
        reference_tractor=reference_tractor,
        host_tractor=host_tractor,
        grec=grec,
        checkpoint_paths={
            "stein_ssl_resnet50": paths["stein_checkpoint"],
            "imagenet_effnet_v2_s": paths["imagenet_checkpoint"],
        },
    )
