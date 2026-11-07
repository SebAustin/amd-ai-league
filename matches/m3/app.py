#!/usr/bin/env python3
"""Match 3 (RAG) entrypoint the Harness execs: ``--index /app/corpus`` once, then one
``--corpus /app/corpus --query-id ID --query "..."`` per Question. Logic lives in amdleague.m3."""

from amdleague.m3.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
