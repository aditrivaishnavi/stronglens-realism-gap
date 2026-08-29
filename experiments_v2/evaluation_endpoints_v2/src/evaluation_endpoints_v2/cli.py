from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import (
    prepare_manifest,
    score_endpoint,
    verify_endpoint,
    write_seal,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Frozen-threshold achieved-FPR and 43-system endpoints"
    )
    result.add_argument("--config", type=Path, required=True)
    commands = result.add_subparsers(dest="command", required=True)

    commands.add_parser("seal")

    prepare = commands.add_parser("prepare")
    prepare.add_argument(
        "--role",
        choices=["achieved_fpr", "terminal_confirmed"],
        required=True,
    )
    prepare.add_argument("--output", type=Path, required=True)

    score = commands.add_parser("score")
    score.add_argument(
        "--role",
        choices=["achieved_fpr", "terminal_confirmed"],
        required=True,
    )
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--manifest-receipt", type=Path, required=True)
    score.add_argument("--device", required=True)
    score.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify")
    verify.add_argument(
        "--role",
        choices=["achieved_fpr", "terminal_confirmed"],
        required=True,
    )
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--manifest-receipt", type=Path, required=True)
    verify.add_argument("--scores", type=Path, required=True)
    verify.add_argument("--score-receipt", type=Path, required=True)
    verify.add_argument("--result", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    arguments = parser().parse_args()
    if arguments.command == "seal":
        output = write_seal(arguments.config)
    elif arguments.command == "prepare":
        output = prepare_manifest(arguments.config, arguments.role, arguments.output)
    elif arguments.command == "score":
        output = score_endpoint(
            arguments.config,
            arguments.role,
            arguments.manifest,
            arguments.manifest_receipt,
            arguments.output,
            device=arguments.device,
        )
    else:
        output = verify_endpoint(
            arguments.config,
            arguments.role,
            arguments.manifest,
            arguments.manifest_receipt,
            arguments.scores,
            arguments.score_receipt,
            arguments.result,
            arguments.output,
        )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
