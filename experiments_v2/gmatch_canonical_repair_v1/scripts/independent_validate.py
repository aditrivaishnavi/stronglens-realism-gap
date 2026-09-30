#!/usr/bin/env python3
"""Run the independent raw-authority G_match v2 reconstruction."""

import sys

from gmatch_v2.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "validate")
    raise SystemExit(main())
