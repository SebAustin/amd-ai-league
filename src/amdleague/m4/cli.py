"""Entrypoint: ``python3 /app/app.py --question ... --url ... [--id ...]``.

Contract we hold ourselves to: a valid ``<id>_output.json`` exists before any network call and is
replaced atomically with the final answer; the process never crashes with a traceback; the
time budget is counted from process start, not from when we got round to the request.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from amdleague.common.deadline import Deadline
from amdleague.common.model_client import ModelClient
from amdleague.common.output import write_json
from amdleague.m4.agent import answer_request
from amdleague.m4.answer import NO_ANSWER, Answer
from amdleague.m4.render import make_renderer
from amdleague.m4.request import Request, RequestError, parse_args, parse_batch

log = logging.getLogger("m4")

PROCESS_START = time.monotonic()  # module import ~ interpreter start
DEFAULT_OUTPUT_DIR = "/app/output"
DEFAULT_BUDGET_SECONDS = 25.0
SAFETY_MARGIN_SECONDS = 1.5  # final write + interpreter exit


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def _budget() -> float:
    raw = os.environ.get("M4_TIME_BUDGET", "")
    try:
        value = float(raw) if raw else DEFAULT_BUDGET_SECONDS
    except ValueError:
        log.warning("ignoring invalid M4_TIME_BUDGET=%r", raw)
        value = DEFAULT_BUDGET_SECONDS
    return max(1.0, value - SAFETY_MARGIN_SECONDS)


def build_payload(req_id: str, question: str, answer: Answer, req: Request | None = None,
                  error: str | None = None) -> dict[str, Any]:
    citations = list(answer.citations)
    payload: dict[str, Any] = {
        "id": req_id,
        "question": question,
        "answer": answer.value,
        "citations": citations,
        "sources": citations,
        "source_url": citations[0] if citations else "",
        "confidence": round(answer.confidence, 3),
        "method": answer.method,
    }
    if req is not None and req.url:
        payload["start_url"] = req.url
    if error:
        payload["error"] = error
    return payload


class Sink:
    """Where results go: ``<dir>/<id>_output.json``, plus an exact ``--output`` file if given."""

    def __init__(self, argv: list[str]) -> None:
        try:
            args = parse_args(argv)
            output_dir, output_file = args.output_dir, args.output_file
        except RequestError:
            output_dir, output_file = None, None
        if output_file and not output_file.endswith(".json"):
            output_dir, output_file = output_dir or output_file, None
        self.dir = Path(output_dir or os.environ.get("M4_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
        self.file = Path(output_file) if output_file else None

    def emit(self, payload: dict[str, Any]) -> None:
        targets = [self.dir / f"{payload['id']}_output.json"]
        if self.file is not None:
            targets.append(self.file)
        for path in targets:
            try:
                write_json(path, payload)
            except OSError as exc:
                log.error("cannot write %s: %s", path, exc)


def run_one(req: Request, sink: Sink, llm: ModelClient | None, renderer: Any,
            deadline: Deadline) -> None:
    sink.emit(build_payload(req.id, req.question, NO_ANSWER, req))  # placeholder first
    try:
        answer = answer_request(req, llm, deadline, renderer=renderer)
        payload = build_payload(req.id, req.question, answer, req)
    except Exception as exc:
        log.exception("request %s failed", req.id)
        payload = build_payload(req.id, req.question, NO_ANSWER, req, error=str(exc))
    sink.emit(payload)
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=os.environ.get("M4_LOG_LEVEL", "INFO"), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # A real CLI run counts from interpreter start; an in-process call (tests, eval) from now.
    started = PROCESS_START if argv is None else time.monotonic()
    argv = list(sys.argv[1:] if argv is None else argv)
    sink = Sink(argv)
    try:
        items = parse_batch(argv)
    except RequestError as exc:
        log.error("invalid request: %s", exc)
        sink.emit(build_payload(exc.request_id or "invalid", "", NO_ANSWER, error=str(exc)))
        return 2
    budget = _budget()
    llm = None if _env_flag("M4_DISABLE_LLM") else ModelClient()
    renderer = None if _env_flag("M4_DISABLE_BROWSER") else make_renderer()
    first = True
    try:
        for item in items:
            if isinstance(item, RequestError):
                sink.emit(build_payload(item.request_id or "invalid", "", NO_ANSWER,
                                        error=str(item)))
                continue
            # The first request also pays for interpreter start-up; later ones get a fresh budget.
            deadline = Deadline(budget, start=started) if first else Deadline(budget)
            first = False
            run_one(item, sink, llm, renderer, deadline)
    finally:
        if renderer is not None:
            renderer.close()
    return 2 if all(isinstance(i, RequestError) for i in items) and items else 0
