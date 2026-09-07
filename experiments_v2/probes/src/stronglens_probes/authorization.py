"""Stage- and manifest-specific design/final execution authorization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .authorities import AuthorityContext
from .contracts import (
    ContractError,
    normalized_protocol_config_sha256,
    require_sha256,
    sha256_file,
)


def protocol_config_sha256(config: dict[str, Any]) -> str:
    """Digest protocol content without circular receipt-hash fields."""
    return normalized_protocol_config_sha256(config)


@dataclass(frozen=True)
class ExecutionAuthorization:
    authorization_kind: str
    data_role: str
    allowed_stages: tuple[str, ...]
    manifest_sha256s: tuple[str, ...]
    receipt: dict[str, Any]
    receipt_path: Path
    receipt_sha256: str

    def revalidate(
        self,
        *,
        config: dict[str, Any],
        authority: AuthorityContext,
        stage: str,
        manifest_sha256: str,
        feature_suite_authority_sha256: str | None = None,
    ) -> None:
        authority.assert_config(config)
        if (
            sha256_file(self.receipt_path) != self.receipt_sha256
            or json.loads(self.receipt_path.read_text(encoding="utf-8")) != self.receipt
        ):
            raise ContractError("execution authorization changed after loading")
        if stage not in self.allowed_stages:
            raise ContractError(f"authorization does not permit stage {stage}")
        if manifest_sha256 not in self.manifest_sha256s:
            raise ContractError("authorization does not permit this manifest")
        if feature_suite_authority_sha256 is not None:
            allowed = self.receipt.get("feature_suite_authority_sha256s", [])
            if feature_suite_authority_sha256 not in allowed:
                raise ContractError(
                    "authorization does not permit this feature-suite authority"
                )
        extraction_stages = {"feature_extract", "feature_load"}
        if stage in extraction_stages and self.authorization_kind != "extraction":
            raise ContractError("feature stage requires extraction authorization")
        if stage not in extraction_stages and self.authorization_kind != "analysis":
            raise ContractError("analysis stage requires analysis authorization")
        if self.receipt.get("protocol_config_sha256") != protocol_config_sha256(config):
            raise ContractError("authorization has the wrong protocol config")
        if (
            self.receipt.get("reference_manifest_sha256")
            != authority.reference_manifest_sha256
            or self.receipt.get("reference_payload_manifest_sha256")
            != authority.reference_payload_manifest_sha256
            or self.receipt.get("host_universe_manifest_sha256")
            != authority.experiment_host_universe_sha256
        ):
            raise ContractError("authorization has the wrong data authorities")
        if self.data_role == "design":
            expected_status = (
                "authorized_design_extraction"
                if self.authorization_kind == "extraction"
                else "authorized_design_analysis"
            )
            if self.receipt.get("status") != expected_status:
                raise ContractError("design authorization status is invalid")
        elif self.data_role == "final":
            field = (
                "extraction_authorization_receipt_sha256"
                if self.authorization_kind == "extraction"
                else "analysis_authorization_receipt_sha256"
            )
            expected = config["execution"].get(field)
            require_sha256(expected, context="final analysis authorization")
            if (
                (
                    self.authorization_kind == "extraction"
                    and config["execution"].get("feature_extraction_authorized")
                    is not True
                )
                or (
                    self.authorization_kind == "analysis"
                    and config["execution"].get("final_analysis_authorized") is not True
                )
                or expected != self.receipt_sha256
                or self.receipt.get("status")
                != (
                    "authorized_final_extraction"
                    if self.authorization_kind == "extraction"
                    else "authorized_final_analysis"
                )
            ):
                raise ContractError("final stage authorization is invalid")
        else:
            raise ContractError(f"unknown authorization role {self.data_role}")


def load_execution_authorization(
    *,
    path: Path,
    config: dict[str, Any],
    authority: AuthorityContext,
) -> ExecutionAuthorization:
    """Load a receipt; every use rechecks its bytes and allowed parent manifests."""
    authority.assert_config(config)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    data_role = receipt.get("data_role")
    authorization_kind = receipt.get("authorization_kind")
    stages = receipt.get("allowed_stages")
    manifests = receipt.get("manifest_sha256s")
    if data_role not in {"design", "final"}:
        raise ContractError("authorization data_role must be design or final")
    if authorization_kind not in {"extraction", "analysis"}:
        raise ContractError("authorization_kind must be extraction or analysis")
    if (
        not isinstance(stages, list)
        or not stages
        or not all(isinstance(stage, str) and stage for stage in stages)
    ):
        raise ContractError("authorization stages are invalid")
    if (
        not isinstance(manifests, list)
        or not manifests
        or not all(isinstance(value, str) for value in manifests)
    ):
        raise ContractError("authorization manifests are invalid")
    for value in manifests:
        require_sha256(value, context="authorization manifest")
    authorization = ExecutionAuthorization(
        authorization_kind=str(authorization_kind),
        data_role=str(data_role),
        allowed_stages=tuple(stages),
        manifest_sha256s=tuple(manifests),
        receipt=receipt,
        receipt_path=path.resolve(),
        receipt_sha256=sha256_file(path),
    )
    # Check static authority/config fields immediately. A specific stage and
    # manifest are checked by the caller.
    if receipt.get("protocol_config_sha256") != protocol_config_sha256(config):
        raise ContractError("authorization has the wrong protocol config")
    if (
        receipt.get("reference_manifest_sha256") != authority.reference_manifest_sha256
        or receipt.get("reference_payload_manifest_sha256")
        != authority.reference_payload_manifest_sha256
        or receipt.get("host_universe_manifest_sha256")
        != authority.experiment_host_universe_sha256
    ):
        raise ContractError("authorization has the wrong data authorities")
    if authorization.authorization_kind == "extraction":
        representation_entries = [
            config["representations"]["primary"],
            *config["representations"]["required_diagnostics"],
        ]
        expected_bindings = {
            entry["id"]: {
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "extractor_code_sha256": entry["extractor_code_sha256"],
                "preprocessing_code_sha256": entry["preprocessing_code_sha256"],
                "preprocessing_contract_sha256": entry["preprocessing_contract_sha256"],
                "extraction_contract_sha256": entry["extraction_contract_sha256"],
            }
            for entry in representation_entries
        }
        if receipt.get("representation_bindings") != expected_bindings:
            raise ContractError(
                "extraction authorization does not pin exact checkpoints/code"
            )
    else:
        suites = receipt.get("feature_suite_authority_sha256s")
        result_contracts = receipt.get("result_contract_sha256s")
        feature_stages = {"probe_analysis", "joint_inference", "bootstrap"}
        if set(authorization.allowed_stages) & feature_stages and (
            not isinstance(suites, list) or not suites
        ):
            raise ContractError(
                "analysis authorization lacks feature-suite authorities"
            )
        if not isinstance(suites, list):
            suites = []
        if not isinstance(result_contracts, list) or not result_contracts:
            raise ContractError("analysis authorization lacks result contracts")
        for value in [*suites, *result_contracts]:
            require_sha256(value, context="analysis authorization binding")
    return authorization
