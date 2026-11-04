"""Deterministic answer-type detection and value extraction (the fallback when the LLM can't)."""

from __future__ import annotations

import re
from enum import Enum


class AnswerType(Enum):
    EMAIL = "email"
    PHONE = "phone"
    PRICE = "price"
    YEAR = "year"
    DATE = "date"
    VERSION = "version"
    NUMBER = "number"
    TEXT = "text"


# Order matters: the first matching cue wins ("how many ... last year" is a NUMBER,
# "what year" beats "when", price beats "how many").
_TYPE_CUES: tuple[tuple[AnswerType, re.Pattern[str]], ...] = (
    (AnswerType.EMAIL, re.compile(r"\be-?mail\b")),
    (AnswerType.PHONE, re.compile(r"\b(phone|telephone|tel|call|fax|hotline)\b")),
    (AnswerType.PRICE, re.compile(r"\b(price|prices|cost|costs|how much|fee|fees|pay|charge)\b")),
    (AnswerType.NUMBER, re.compile(r"\b(how many|number of|count|how long|how big|how tall)\b")),
    (AnswerType.YEAR, re.compile(r"\b(what|which|in what) year\b|\byear\b")),
    (AnswerType.DATE, re.compile(r"\b(when|date|day|deadline)\b")),
    (AnswerType.VERSION, re.compile(r"\bversion\b|\brelease number\b")),
)

_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
          r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_CURRENCY_CODES = r"(?:USD|EUR|GBP|JPY|CAD|AUD|CHF|INR|CNY|SGD|AED)"
_AMOUNT = r"\d[\d,]*(?:\.\d+)?"

_PATTERNS: dict[AnswerType, re.Pattern[str]] = {
    AnswerType.EMAIL: re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    AnswerType.PHONE: re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d[\d\s.-]{5,}\d"),
    AnswerType.PRICE: re.compile(
        rf"[$€£¥₹]\s?{_AMOUNT}|{_CURRENCY_CODES}\s?{_AMOUNT}|{_AMOUNT}\s?{_CURRENCY_CODES}\b"
    ),
    AnswerType.YEAR: re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b"),
    AnswerType.DATE: re.compile(
        rf"\b{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"
        rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}\.?,?\s+\d{{4}}\b"
        r"|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
        rf"|\b{_MONTH}\s+\d{{4}}\b"
    ),
    AnswerType.VERSION: re.compile(r"\bv?\d+(?:\.\d+){1,3}(?:[-+][\w.]+)?\b"),
    AnswerType.NUMBER: re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w.]*\d)"),
}

# Words that, right before a value, signal it is the requested kind of value.
TYPE_KEYWORDS: dict[AnswerType, tuple[str, ...]] = {
    AnswerType.EMAIL: ("email", "e-mail", "mail", "contact"),
    AnswerType.PHONE: ("phone", "call", "tel", "telephone", "fax"),
    AnswerType.PRICE: ("price", "cost", "now", "only", "total"),
    AnswerType.YEAR: ("founded", "since", "established", "in", "year", "built", "born"),
    AnswerType.DATE: ("date", "released", "published", "on", "updated", "due"),
    AnswerType.VERSION: ("version", "release", "v", "firmware", "latest"),
    AnswerType.NUMBER: (),
    AnswerType.TEXT: (),
}


def detect_answer_type(question: str) -> AnswerType:
    q = question.lower()
    for answer_type, cue in _TYPE_CUES:
        if cue.search(q):
            return answer_type
    return AnswerType.TEXT


def find_candidates(text: str, answer_type: AnswerType) -> list[tuple[str, int]]:
    """``(value, start offset)`` pairs in text order, de-duplicated by value."""
    pattern = _PATTERNS.get(answer_type)
    if pattern is None:
        return []
    if answer_type is AnswerType.DATE:
        dates = _find(text, pattern, answer_type)
        # "When was Acme founded?" is often answered by a bare year.
        return dates or _find(text, _PATTERNS[AnswerType.YEAR], AnswerType.YEAR)
    return _find(text, pattern, answer_type)


def _find(text: str, pattern: re.Pattern[str], answer_type: AnswerType) -> list[tuple[str, int]]:
    found: dict[str, int] = {}
    for match in pattern.finditer(text):
        value = match.group(0).strip().rstrip(".,;:")
        if answer_type is AnswerType.PHONE and sum(ch.isdigit() for ch in value) < 7:
            continue
        found.setdefault(value, match.start())
    return list(found.items())


def extract_candidates(text: str, answer_type: AnswerType) -> list[str]:
    return [value for value, _ in find_candidates(text, answer_type)]
