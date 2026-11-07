"""Score the Match 3 pipeline on the official sample kit, graded like the Harness.

Usage:
    sh eval/m3/fetch_kit.sh               # once: kit -> eval/m3/kit (git-ignored)
    python eval/m3/run_eval.py            # rules only (no GPU): images unread
    python eval/m3/run_eval.py --llm      # with the resident model server (GPU)

A Question scores 20 only when the answer matches (uppercase, whitespace and - . · _ removed,
units/degree signs ignored via the kit's aliases) AND the citations are exactly the expected set.
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
from amdleague.m3.answer import answer_question
from amdleague.m3.cli import make_transcriber
from amdleague.m3.index import build_index

HERE = Path(__file__).parent
DEFAULT_KIT = HERE / "kit" / "mc3-starter-kit"
POINTS = 20
_STRIP = re.compile(r"[\s\-.·_°]+")


def league_norm(text: str) -> str:
    return _STRIP.sub("", text.upper())


def grade(case: dict, answer: str, citations: list[str]) -> tuple[bool, bool]:
    expected = [case["expected_answer"], *case.get("answer_aliases", [])]
    if not case["expected_answer"]:
        answer_ok = answer == ""
    else:
        answer_ok = league_norm(answer) in {league_norm(e) for e in expected if e}
    cites_ok = set(citations) == set(case["expected_citations"])
    return answer_ok, cites_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", type=Path, default=DEFAULT_KIT)
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--budget", type=float, default=25.0)
    args = ap.parse_args()
    questions = json.loads((args.kit / "sample-questions.json").read_text())
    corpus = args.kit / questions["corpus"]
    llm = ModelClient() if args.llm else None

    start = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        transcribe = None
        if llm is not None and llm.wait_ready(600):
            transcribe = make_transcriber(llm, Path(tmp) / "images")
        index = build_index(corpus, Path(tmp), transcribe=transcribe, deadline=Deadline(480))
    print(f"indexed {len(index.documents)} documents in {time.monotonic() - start:.1f}s "
          f"(skipped {len(index.skipped)})\n")

    total, rows = 0, []
    for case in questions["queries"]:
        t0 = time.monotonic()
        res = answer_question(case["query"], index.documents, llm, Deadline(args.budget))
        answer_ok, cites_ok = grade(case, res.answer, list(res.citations))
        score = POINTS if answer_ok and cites_ok else 0
        total += score
        mark = "OK " if score else ("A✓C✗" if answer_ok else ("A✗C✓" if cites_ok else "XX "))
        print(f"{mark:5} Q{case['n']:<2} {res.answer!r:<18} cites={list(res.citations)} "
              f"want {case['expected_answer']!r} {case['expected_citations']} "
              f"[{res.method}, {time.monotonic() - t0:.1f}s]")
        rows.append({"n": case["n"], "answer": res.answer, "citations": list(res.citations),
                     "method": res.method, "score": score})
    print(f"\nscore {total}/{POINTS * len(questions['queries'])}")
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out / f"{stamp}{'-llm' if args.llm else ''}.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
