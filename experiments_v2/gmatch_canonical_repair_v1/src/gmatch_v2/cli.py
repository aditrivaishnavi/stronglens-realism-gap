"""Command-line entry points for G_match v2 preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .preparation import prepare, write_config_seal
from .validator import independent_validate


def _defaults() -> tuple[Path, Path, Path]:
    package_root = Path(__file__).resolve().parents[2]
    repository_root = package_root.parents[1]
    return (
        package_root,
        package_root / "config/production.toml",
        repository_root / "docs/G_MATCH_V2_AUTHORITY.md",
    )


def main() -> int:
    package_root, default_config, default_documentation = _defaults()
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "prepare", "validate"))
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument(
        "--authority-document", type=Path, default=default_documentation
    )
    arguments = parser.parse_args()
    if arguments.command == "seal":
        result = write_config_seal(
            package_root=package_root,
            config_path=arguments.config.resolve(),
            documentation_path=arguments.authority_document.resolve(),
        )
    elif arguments.command == "prepare":
        result = prepare(
            package_root=package_root,
            config_path=arguments.config.resolve(),
            documentation_path=arguments.authority_document.resolve(),
        )
    else:
        result = independent_validate(
            package_root=package_root,
            config_path=arguments.config.resolve(),
            documentation_path=arguments.authority_document.resolve(),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
