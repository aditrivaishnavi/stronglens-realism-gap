from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import (
    POPULATIONS,
    analyze_panel,
    load_panel,
    score_population,
    validate_panel,
    verify_outputs,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Rescore final frozen pixels across the complete checkpoint panel"
    )
    result.add_argument("--config", type=Path, required=True)
    commands = result.add_subparsers(dest="command", required=True)

    commands.add_parser("validate")

    score = commands.add_parser("score")
    score.add_argument("--population", choices=POPULATIONS, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--chunk-rows", type=int, default=1024)

    analyze = commands.add_parser("analyze")
    analyze.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    arguments = parser().parse_args()
    if arguments.command == "validate":
        panel, _ = load_panel(arguments.config)
        output = validate_panel(panel)
    elif arguments.command == "score":
        output = score_population(
            arguments.config,
            arguments.population,
            arguments.output,
            chunk_rows=arguments.chunk_rows,
        )
    elif arguments.command == "analyze":
        output = analyze_panel(arguments.config, arguments.output)
    else:
        output = verify_outputs(arguments.config, arguments.output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
