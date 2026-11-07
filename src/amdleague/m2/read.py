"""Ask the vision model what an image says, then apply the League's transcription rules in code.

The model returns the text lines and whether the image is a plate or a sign. The code then
decides the answer: on a plate only lines holding the registration (they carry digits) are kept,
so banners ("CALIFORNIA"), slogans and websites drop out while "京A·12345" stays whole; a sign
keeps every line, joined top to bottom with single spaces. Two views (original and
contrast-boosted) are read; agreement raises confidence, the original wins a disagreement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from amdleague.common.deadline import Deadline

MAX_NEW_TOKENS = 96
MIN_FIRST_SECONDS = 1.0  # time needed to try the first view at all
MIN_SECOND_SECONDS = 3.0  # time needed before a second view is worth reading
MAX_CALL_SECONDS = 15.0
AGREED_CONFIDENCE = 0.95
SINGLE_CONFIDENCE = 0.7
DISAGREED_CONFIDENCE = 0.6
FALLBACK_CONFIDENCE = 0.5

PROMPT = (
    "Read the text in this image. It is usually a vehicle licence plate or a road sign (stop, "
    "speed limit, advisory speed, work zone, warning).\n"
    'Reply with JSON only: {"kind": "plate" | "sign" | "other", "lines": ["...", "..."]}\n'
    "- lines: each line of text exactly as printed, top to bottom, in its own script (keep "
    "Chinese characters such as 京 or 沪).\n"
    "- For a plate, give the registration number exactly: keep a leading Chinese province "
    "character and letter, and leave out state names, slogans, websites and stickers.\n"
    "- Never add units, words or punctuation that are not printed. If nothing is readable, "
    'reply {"kind": "other", "lines": []}.'
)

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_LABEL = re.compile(r"^\s*(?:the\s+)?(?:text|plate|licen[cs]e\s+plate|answer|number|sign|"
                    r"reading|registration)(?:\s+(?:reads|says|is))?\s*[:：]\s*", re.IGNORECASE)
_QUOTES = "\"'`“”‘’「」"
_SPACES = re.compile(r"\s+")
_PLATE_PART = re.compile(r"^[A-Z]{1,4}$")  # "JHT" printed on its own line beside "2951"
_SHORT_STATES = frozenset({"IOWA", "OHIO", "UTAH"})  # banners short enough to look like parts


class LLM(Protocol):
    def complete(self, messages: list[dict[str, Any]], max_new_tokens: int,
                 timeout: float) -> str | None: ...


@dataclass(frozen=True)
class Reading:
    text: str
    confidence: float


NOTHING = Reading("", 0.0)


def _json_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            data, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def parse_reply(raw: str | None) -> tuple[str, list[str]] | None:
    """(kind, lines) from the model's reply: JSON when it complied, plain lines otherwise."""
    if not raw or not raw.strip():
        return None
    text = _FENCE.sub("", raw.strip())
    data = _json_object(text)
    if data is None:
        return "other", [line for line in text.splitlines() if line.strip()]
    kind = str(data.get("kind") or "other").strip().lower()
    lines = data.get("lines")
    if not isinstance(lines, list):
        lines = str(data.get("text") or "").splitlines()
    return kind, [str(line) for line in lines if str(line).strip()]


def _clean(line: str) -> str:
    line = _LABEL.sub("", line.strip()).strip().strip(_QUOTES).strip()
    return _SPACES.sub(" ", line)


def compose(kind: str, lines: list[str]) -> str:
    """The answer under the League's convention for this kind of image."""
    cleaned = [c for c in (_clean(line) for line in lines) if c]
    if kind == "plate":
        cleaned = _registration(cleaned)
    return " ".join(cleaned)


def _registration(lines: list[str]) -> list[str]:
    """Lines with digits, plus a short letters-only line right next to one; the rest
    (state names, slogans, websites) is printed around the number. No digit at all: a vanity
    plate, keep everything."""
    numbered = [any(ch.isdigit() for ch in line) for line in lines]
    if not any(numbered):
        return lines

    def part(i: int) -> bool:
        line = lines[i]
        beside = (i > 0 and numbered[i - 1]) or (i + 1 < len(lines) and numbered[i + 1])
        return beside and bool(_PLATE_PART.match(line)) and line not in _SHORT_STATES

    return [line for i, line in enumerate(lines) if numbered[i] or part(i)]


def _key(text: str) -> str:
    return re.sub(r"[\s\-.·_]+", "", text.upper())


def _read_view(view: Path, llm: LLM, deadline: Deadline) -> str:
    content = [{"type": "image", "path": str(view)}, {"type": "text", "text": PROMPT}]
    timeout = min(MAX_CALL_SECONDS, deadline.remaining() - 0.5)
    parsed = parse_reply(llm.complete([{"role": "user", "content": content}], MAX_NEW_TOKENS,
                                      timeout=timeout))
    return compose(*parsed) if parsed else ""


def read_image(views: list[Path], llm: LLM | None, deadline: Deadline) -> Reading:
    if llm is None or not views or deadline.remaining() < MIN_FIRST_SECONDS:
        return NOTHING
    first = _read_view(views[0], llm, deadline)
    if len(views) < 2 or deadline.remaining() < MIN_SECOND_SECONDS:
        return Reading(first, SINGLE_CONFIDENCE) if first else NOTHING
    second = _read_view(views[1], llm, deadline)
    if first and _key(first) == _key(second):
        return Reading(first, AGREED_CONFIDENCE)
    if first:
        return Reading(first, DISAGREED_CONFIDENCE)
    return Reading(second, FALLBACK_CONFIDENCE) if second else NOTHING
