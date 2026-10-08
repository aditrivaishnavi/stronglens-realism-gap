#!/usr/bin/env python3
"""Build the compiled manuscript PDF."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(command: list[str], *, cwd: Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def build_manuscript() -> None:
    run(["tectonic", "-X", "compile", "main.tex", "--outdir", ".", "--keep-logs"])
    (ROOT / "main.pdf").replace(ROOT / "manuscript.pdf")
    build_dir = ROOT / "build"
    build_dir.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "manuscript.pdf", build_dir / "main.pdf")


def main() -> None:
    build_manuscript()
    print(ROOT / "manuscript.pdf")


if __name__ == "__main__":
    main()
