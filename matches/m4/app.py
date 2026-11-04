"""Harness entrypoint: python3 /app/app.py --question "..." --url https://... [--id ID]."""

import sys

from amdleague.m4.cli import main

if __name__ == "__main__":
    sys.exit(main())
