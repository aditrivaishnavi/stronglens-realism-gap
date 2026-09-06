from __future__ import annotations

import argparse
import json

from .contracts import load_config
from .prepare import prepare_design
from .selector import select_draw_count
from .streaming import (
    run_streaming_pilot,
    write_prelaunch_deep_validation_receipt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed successor P_noise design-pilot workflow"
    )
    parser.add_argument("--config", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-design",
        help="stage actual historical hosts and freeze score-blind design artifacts",
    )
    prepare.add_argument("--junit", required=True)
    prepare.add_argument("--aws-profile")
    prepare.add_argument("--workers", type=int, default=8)

    subparsers.add_parser(
        "check-freeze",
        help="verify the review freeze without rendering or scoring",
    )
    subparsers.add_parser(
        "run-pilot",
        help="stream 64 source-photon draws through fixed-width MPS scoring",
    )
    subparsers.add_parser(
        "select-draw-count",
        help="freeze the smallest selectable count passing both MCSE gates",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "prepare-design":
        profile = (
            args.aws_profile
            if args.aws_profile is not None
            else config["historical_source"]["aws_profile_default"]
        )
        artifacts = prepare_design(
            config,
            junit_path=args.junit,
            aws_profile=str(profile),
            workers=args.workers,
        )
        print(
            json.dumps(
                {name: str(path) for name, path in artifacts.items()},
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "check-freeze":
        receipt, path = write_prelaunch_deep_validation_receipt(config)
        print(
            json.dumps(
                {"receipt_path": str(path), **receipt},
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "run-pilot":
        scores, receipt = run_streaming_pilot(config)
        print(json.dumps({"pilot_scores": str(scores), "receipt": str(receipt)}))
    elif args.command == "select-draw-count":
        receipt = select_draw_count(config)
        print(json.dumps({"draw_count_receipt": str(receipt)}))
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
