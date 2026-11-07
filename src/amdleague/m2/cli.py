"""``python3 /app/app.py --input-image /app/input/image_01.png`` -> /app/output/image_01_output.json

An empty answer is written before any work and replaced atomically, so the Harness always finds
valid JSON; the process never ends with a traceback and stops inside its time budget.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from amdleague.common.deadline import Deadline
from amdleague.common.model_client import ModelClient
from amdleague.common.output import write_json
from amdleague.m2.images import prepare
from amdleague.m2.read import NOTHING, Reading, read_image

log = logging.getLogger(__name__)

PROCESS_START = time.monotonic()
DEFAULT_OUTPUT_DIR = "/app/output"
DEFAULT_SCRATCH_DIR = "/tmp/m2-views"
TIME_BUDGET_SECONDS = 25.0  # of the 30 s per image


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def _budget() -> float:
    try:
        return float(os.environ.get("M2_TIME_BUDGET", "") or TIME_BUDGET_SECONDS)
    except ValueError:
        return TIME_BUDGET_SECONDS


def _llm() -> ModelClient | None:
    return None if _env_flag("M2_DISABLE_LLM") else ModelClient()


def _payload(reading: Reading) -> dict:
    return {"text": reading.text, "confidence": round(reading.confidence, 3)}


def output_path(image: Path) -> Path:
    """image_01.png -> <output>/image_01_output.json (only the last extension is replaced)."""
    out_dir = Path(os.environ.get("M2_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    return out_dir / f"{image.stem}_output.json"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read the text in one image.")
    p.add_argument("--input-image", "--image", "--input", "-i", dest="image")
    p.add_argument("positional", nargs="?")
    return p


def run(image: Path, started: float) -> int:
    target = output_path(image)
    write_json(target, _payload(NOTHING))
    deadline = Deadline(_budget() - (time.monotonic() - started))
    try:
        scratch = Path(os.environ.get("M2_SCRATCH_DIR", DEFAULT_SCRATCH_DIR))
        reading = read_image(prepare(image, scratch), _llm(), deadline)
    except Exception:  # the answer file must stay valid whatever happens
        log.exception("reading %s failed", image)
        reading = NOTHING
    write_json(target, _payload(reading))
    log.info("%s -> %r (%.2f) in %.1fs", image.name, reading.text, reading.confidence,
             time.monotonic() - started)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=os.environ.get("M2_LOG_LEVEL", "INFO"), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    started = PROCESS_START if argv is None else time.monotonic()
    args, unknown = _parser().parse_known_args(sys.argv[1:] if argv is None else argv)
    if unknown:
        log.warning("ignoring unknown arguments %s", unknown)
    image = args.image or args.positional
    if not image:
        log.error("usage: app.py --input-image PATH")
        return 2
    try:
        return run(Path(image), started)
    except OSError:
        log.exception("cannot write the answer for %s", image)
        return 1
