"""Score the Match 4 agent on eval/m4/requests.jsonl.

Usage:
    python eval/m4/run_eval.py                 # rules only (no GPU)
    python eval/m4/run_eval.py --llm           # use the resident model server
    python eval/m4/run_eval.py --only quotes   # ids containing "quotes"
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
from amdleague.m4.request import make_request

HERE = Path(__file__).parent


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" .,:;\"'")


def grade(answer: str, expected: list[str]) -> tuple[bool, bool]:
    got = norm(answer)
    exact = any(got == norm(e) for e in expected)
    contains = bool(got) and any(norm(e) in got or got in norm(e) for e in expected)
    return exact, contains


def run_case(case: dict, llm: ModelClient | None, budget: float) -> dict:
    req = make_request(case["question"], case.get("url"), case.get("corpus"), case["id"],
                       base_dir=HERE)
    start = time.monotonic()
    ans = answer_request(req, llm, Deadline(budget))
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
    args = ap.parse_args()
    cases = [json.loads(line) for line in (HERE / "requests.jsonl").read_text().splitlines()
             if line.strip()]
    cases = [c for c in cases if args.only in c["id"]]
    llm = ModelClient() if args.llm else None
    workers = 1 if args.llm else args.workers  # one GPU server: keep calls sequential
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda c: run_case(c, llm, args.budget), cases))
    for r in results:
        mark = "OK " if r["exact"] else ("~  " if r["contains"] else "XX ")
        print(f"{mark}{r['id']:<28} {r['answer'][:50]!r:<54} want {r['expected'][0]!r} "
              f"[{r['method']}, {r['seconds']}s]")
    n = len(results) or 1
    exact = sum(r["exact"] for r in results)
    contains = sum(r["contains"] for r in results)
    print(f"\nexact {exact}/{len(results)} ({exact / n:.0%})  contains {contains}/{len(results)}  "
          f"max {max((r['seconds'] for r in results), default=0)}s")
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out / f"{stamp}{'-llm' if args.llm else ''}.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
