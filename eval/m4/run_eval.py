"""Score the Match 4 agent on eval/m4/requests.jsonl.

Usage:
    python eval/m4/run_eval.py                 # rules only (no GPU)
    python eval/m4/run_eval.py --llm           # use the resident model server
    python eval/m4/run_eval.py --only quotes   # ids containing "quotes"
    python eval/m4/run_eval.py --record pages  # save every fetched page (needs internet)
    python eval/m4/run_eval.py --llm --replay pages   # same pages, offline (GPU notebook)

Grading follows the League: answers are compared after uppercasing and removing whitespace and
``- . · _``; a case with ``"expected": []`` is unanswerable and only an empty answer scores.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from amdleague.common.deadline import Deadline
from amdleague.common.model_client import ModelClient
from amdleague.m4.agent import answer_request
from amdleague.m4.answer import MIN_CONFIDENCE
from amdleague.m4.fetch import Fetcher, HttpFetcher
from amdleague.m4.request import make_request

sys.path.insert(0, str(Path(__file__).parent))
from page_cache import RecordingFetcher, ReplayFetcher

HERE = Path(__file__).parent


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" .,:;\"'")


_LEAGUE_STRIP = re.compile(r"[\s\-.·_]+")


def league_norm(text: str) -> str:
    """The League graders' normalisation: uppercase, drop whitespace and - . · _"""
    return _LEAGUE_STRIP.sub("", text.upper())


def grade(answer: str, expected: list[str]) -> tuple[bool, bool]:
    """(correct, contains). ``expected == []`` marks an unanswerable question: only "" is right."""
    if not expected:
        return answer.strip() == "", answer.strip() == ""
    got = league_norm(answer)
    exact = bool(got) and any(got == league_norm(e) for e in expected)
    loose = norm(answer)
    contains = bool(loose) and any(norm(e) in loose or loose in norm(e) for e in expected)
    return exact, contains


def make_fetcher(record: str, replay: str) -> Fetcher | None:
    if replay:
        return ReplayFetcher(Path(replay))
    if record:
        return RecordingFetcher(HttpFetcher(), Path(record))
    return None  # the agent's own live fetcher


def run_case(case: dict, llm: ModelClient | None, budget: float, min_conf: float,
             fetcher: Fetcher | None = None) -> dict:
    req = make_request(case["question"], case.get("url"), case.get("corpus"), case["id"],
                       base_dir=HERE)
    start = time.monotonic()
    ans = answer_request(req, llm, Deadline(budget), fetcher=fetcher, min_confidence=min_conf)
    exact, contains = grade(ans.value, case["expected"])
    return {"id": case["id"], "answer": ans.value, "expected": case["expected"],
            "citations": list(ans.citations), "method": ans.method,
            "confidence": ans.confidence, "exact": exact, "contains": contains,
            "seconds": round(time.monotonic() - start, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--budget", type=float, default=25.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    ap.add_argument("--record", default="", help="save every fetched page into this folder")
    ap.add_argument("--replay", default="", help="serve pages only from a --record folder")
    args = ap.parse_args()
    cases = [json.loads(line) for line in (HERE / "requests.jsonl").read_text().splitlines()
             if line.strip()]
    cases = [c for c in cases if args.only in c["id"]]
    llm = ModelClient() if args.llm else None
    workers = 1 if args.llm else args.workers  # one GPU server: keep calls sequential
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda c: run_case(c, llm, args.budget, args.min_confidence,
                                                   make_fetcher(args.record, args.replay)),
                                cases))
    for r in results:
        mark = "OK " if r["exact"] else ("~  " if r["contains"] else "XX ")
        want = r["expected"][0] if r["expected"] else "<empty>"
        print(f"{mark}{r['id']:<28} {r['answer'][:50]!r:<54} want {want!r} "
              f"[{r['method']}, {r['seconds']}s]")
    n = len(results) or 1
    exact = sum(r["exact"] for r in results)
    contains = sum(r["contains"] for r in results)
    answerable = [r for r in results if r["expected"]]
    unanswerable = [r for r in results if not r["expected"]]
    refused = sum(1 for r in answerable if not r["answer"])
    print(f"\nscore {exact}/{len(results)} ({exact / n:.0%})  contains {contains}/{len(results)}  "
          f"max {max((r['seconds'] for r in results), default=0)}s")
    print(f"answerable {sum(r['exact'] for r in answerable)}/{len(answerable)} "
          f"(refused {refused})  unanswerable "
          f"{sum(r['exact'] for r in unanswerable)}/{len(unanswerable)} correctly empty")
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out / f"{stamp}{'-llm' if args.llm else ''}.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
