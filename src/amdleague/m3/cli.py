"""Match 3 entrypoint, invoked by the Harness in two shapes:

    python3 /app/app.py --index /app/corpus
    python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."

Every Question writes ``/app/output/<query-id>_output.json`` with both ``answer`` and
``citations`` — a Refusal placeholder first, replaced atomically by the result — and the process
never exits with a traceback.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

from amdleague.common.deadline import Deadline
from amdleague.common.model_client import ModelClient
from amdleague.common.output import write_json
from amdleague.m3.answer import MIN_CONFIDENCE, REFUSAL, Result, answer_question
from amdleague.m3.corpus import walk
from amdleague.m3.index import Transcriber, build_index, load_index
from amdleague.m4.request import safe_id

log = logging.getLogger("m3")

PROCESS_START = time.monotonic()
DEFAULT_OUTPUT_DIR = "/app/output"
DEFAULT_INDEX_DIR = "/app/index"
QUERY_BUDGET_SECONDS = 25.0  # of the 30 s per Question
INDEX_BUDGET_SECONDS = 480.0  # of the 10 min startup, which also loads the model
STARTUP_SECONDS = 600.0  # container start + model load + --index, as the Harness counts it
STARTUP_MARGIN = 30.0
MIN_INDEX_BUDGET = 30.0
MIN_INDEX_SECONDS = 60.0  # kept for parsing + transcription after waiting for the model
TRANSCRIBE_MAX_TOKENS = 512
TRANSCRIBE_TIMEOUT = 120.0
MAX_IMAGE_SIDE = 1600
EXIF_ORIENTATION = 0x0112
TRANSCRIBE_PROMPT = (
    "Transcribe ALL text visible in this image exactly as written: labels, part numbers, pin "
    "names, revision codes, table cells. Keep each label next to its value on one line "
    "(e.g. 'B14: THERM_ALERT#'). Then add one line describing what the image shows. "
    "Do not guess text you cannot read."
)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        log.warning("ignoring invalid %s", name)
        return default


def _output_dir() -> Path:
    return Path(os.environ.get("M3_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))


def _index_dir() -> Path:
    return Path(os.environ.get("M3_INDEX_DIR", DEFAULT_INDEX_DIR))


def container_age() -> float:
    """Seconds since the container's PID 1 started (0 when unknown, e.g. outside Linux)."""
    try:
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        fields = Path("/proc/1/stat").read_text().rsplit(")", 1)[1].split()
        started = int(fields[19]) / os.sysconf("SC_CLK_TCK")  # field 22: starttime
    except (OSError, ValueError, IndexError):
        return 0.0
    return max(0.0, uptime - started)


def index_budget() -> float:
    """What is left of the Harness's startup budget, which began when the container started."""
    left = STARTUP_SECONDS - STARTUP_MARGIN - container_age()
    configured = _env_float("M3_INDEX_BUDGET", INDEX_BUDGET_SECONDS)
    return max(MIN_INDEX_BUDGET, min(configured, left))


def _llm() -> ModelClient | None:
    return None if _env_flag("M3_DISABLE_LLM") else ModelClient()


def _shrunk(image: Path, scratch: Path) -> Path:
    """Upright and at most MAX_IMAGE_SIDE: phone photos carry their rotation in EXIF, and
    large images cost the vision model seconds per megapixel."""
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return image
    try:
        with Image.open(image) as img:
            rotated = img.getexif().get(EXIF_ORIENTATION, 1) != 1
            if not rotated and max(img.size) <= MAX_IMAGE_SIDE:
                return image
            fixed = ImageOps.exif_transpose(img)
            fixed.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            scratch.mkdir(parents=True, exist_ok=True)
            out = scratch / f"{hashlib.sha1(str(image).encode()).hexdigest()[:16]}.png"
            fixed.convert("RGB").save(out)
            return out
    except Exception:  # noqa: BLE001 - an odd image is still worth sending as is
        log.warning("cannot normalise %s; sending as is", image)
        return image


def make_transcriber(llm: ModelClient, scratch: Path, deadline: Deadline | None = None
                     ) -> Transcriber:
    def transcribe(image: Path) -> str | None:
        timeout = TRANSCRIBE_TIMEOUT
        if deadline is not None:
            timeout = min(timeout, deadline.remaining())
        content = [{"type": "image", "path": str(_shrunk(image, scratch))},
                   {"type": "text", "text": TRANSCRIBE_PROMPT}]
        return llm.complete([{"role": "user", "content": content}], TRANSCRIBE_MAX_TOKENS,
                            timeout=timeout)
    return transcribe


def run_index(corpus: Path, started: float = PROCESS_START) -> int:
    deadline = Deadline(index_budget(), start=started)
    llm = _llm()
    transcriber = None
    if llm is not None:
        # The entrypoint starts loading the model at container start; wait for it once here
        # rather than spending a transcription timeout per image.
        wait = max(0.0, deadline.remaining() - MIN_INDEX_SECONDS)
        if llm.wait_ready(wait):
            transcriber = make_transcriber(llm, _index_dir() / "images", deadline)
        else:
            log.error("model not ready after %.0fs: indexing without reading images", wait)
    try:
        build_index(corpus, _index_dir(), transcribe=transcriber, deadline=deadline)
    except Exception:
        log.exception("indexing failed; questions will read the corpus directly")
    return 0


def _payload(result: Result) -> dict:
    return {"answer": result.answer, "citations": list(result.citations),
            "confidence": round(result.confidence, 3)}


def run_query(corpus: Path, query_id: str, question: str, started: float = PROCESS_START) -> int:
    out = _output_dir() / f"{query_id}_output.json"
    _refuse(query_id)  # a valid answer exists before any work
    deadline = Deadline(_env_float("M3_TIME_BUDGET", QUERY_BUDGET_SECONDS), start=started)
    try:
        index = load_index(_index_dir())
        docs = index.documents if index is not None else tuple(walk(corpus)[0])
        min_conf = min(1.0, max(0.0, _env_float("M3_MIN_CONFIDENCE", MIN_CONFIDENCE)))
        result = answer_question(question, docs, _llm(), deadline, min_confidence=min_conf)
    except Exception:
        log.exception("question %s failed", query_id)
        result = REFUSAL
    write_json(out, _payload(result))
    return 0


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--index", type=Path)
    ap.add_argument("--corpus", type=Path, default=Path("/app/corpus"))
    ap.add_argument("--query-id")
    ap.add_argument("--query")
    return ap


_VALUE_FLAGS = ("--index", "--corpus", "--query-id", "--query")


def _join_values(argv: list[str]) -> list[str]:
    """``--query "-5 V rail?"`` -> ``--query=-5 V rail?`` so a value starting with '-' is never
    mistaken for a flag."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] in _VALUE_FLAGS and i + 1 < len(argv):
            out.append(f"{argv[i]}={argv[i + 1]}")
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def _query_id_in(argv: list[str]) -> str | None:
    for i, arg in enumerate(argv):
        if arg.startswith("--query-id="):
            return arg.split("=", 1)[1]
        if arg == "--query-id" and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _refuse(query_id: str) -> None:
    try:
        write_json(_output_dir() / f"{query_id}_output.json", _payload(REFUSAL))
    except OSError:
        log.exception("cannot write the answer file for %s", query_id)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=os.environ.get("M3_LOG_LEVEL", "INFO"), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # A real CLI run counts from interpreter start; an in-process call (tests, eval) from now.
    started = PROCESS_START if argv is None else time.monotonic()
    raw = list(sys.argv[1:] if argv is None else argv)
    try:
        args, unknown = _parser().parse_known_args(_join_values(raw))
    except SystemExit:
        query_id = _query_id_in(raw)
        if query_id:
            _refuse(safe_id(query_id))
        return 2
    if unknown:
        log.warning("ignoring unknown arguments %s", unknown)
    if args.index is not None:
        return run_index(args.index, started)
    if not args.query_id:
        log.error("a question needs --query-id")
        return 2
    query_id = safe_id(args.query_id)
    if not args.query:
        _refuse(query_id)
        log.error("a question needs --query")
        return 2
    return run_query(args.corpus, query_id, args.query, started)
