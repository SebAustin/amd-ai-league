#!/usr/bin/env python3
"""Match 2 (OCR) entrypoint the Harness execs once per image:
``python3 /app/app.py --input-image /app/input/image_01.png``. Logic lives in amdleague.m2."""

from amdleague.m2.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
