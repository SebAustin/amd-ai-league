"""Score the Match 2 reader on the brief's 10 samples (``python eval/m2/fetch_samples.py`` first).

Usage:
    python eval/m2/run_eval.py          # no model: checks the plumbing, every answer is ""
    python eval/m2/run_eval.py --llm    # resident vision model (GPU)

Graded like the League: uppercase, whitespace and ``- . · _`` removed, exact match, 20 points each.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

from amdleague.common.deadline import Deadline
from amdleague.common.model_client import ModelClient
from amdleague.m2.images import prepare
from amdleague.m2.read import read_image

HERE = Path(__file__).parent
POINTS = 20


def league_norm(text: str) -> str:
    return re.sub(r"[\s\-.·_]+", "", text.upper())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--samples", default=str(HERE / "samples"))
    ap.add_argument("--budget", type=float, default=25.0)
    args = ap.parse_args()
    root = Path(args.samples)
    cases = json.loads((root / "samples.json").read_text())
    llm = ModelClient() if args.llm else None
    if llm is not None:
        llm.wait_ready(600)
    scratch = Path(tempfile.mkdtemp(prefix="m2-eval-"))
    score = 0
    for case in cases:
        start = time.monotonic()
        reading = read_image(prepare(root / case["image"], scratch), llm, Deadline(args.budget))
        ok = league_norm(reading.text) == league_norm(case["expected"])
        score += POINTS * ok
        print(f"{'OK' if ok else 'XX'}  {case['image']:<14} {reading.text!r:<22} "
              f"want {case['expected']!r:<20} conf {reading.confidence:.2f} "
              f"{time.monotonic() - start:.1f}s  ({case['category']})")
    print(f"\nscore {score}/{POINTS * len(cases)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
