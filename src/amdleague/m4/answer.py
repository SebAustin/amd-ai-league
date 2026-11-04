"""Choose the single Answer: LLM over ranked passages, then rules, then a best guess.

Policy (ADR 0002, supersedes 0001): answer only when confident. League graders score an empty
answer as correct on an unanswerable question and a guess as wrong, so any candidate whose
confidence is below ``min_confidence`` is replaced by an empty answer.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Protocol

from amdleague.common.deadline import Deadline
from amdleague.m4.rank import Passage, tokenize
from amdleague.m4.rules import TYPE_KEYWORDS, AnswerType, detect_answer_type, find_candidates

log = logging.getLogger(__name__)

LLM_MAX_NEW_TOKENS = 64
LLM_PASSAGES = 10
LLM_PASSAGE_CHARS = 500
KEYWORD_WINDOW = 25
BEST_GUESS_CHARS = 200
GROUNDED_CONFIDENCE = 0.8
UNGROUNDED_CONFIDENCE = 0.45  # value not found in any passage: likely made up
MIN_CONFIDENCE = 0.5  # below this we abstain (empty answer)
LLM_DISSENT_PENALTY = 0.25  # the model read the passages and said the answer is not there
UNCUED_NUMBER_CONFIDENCE = 0.4  # any number matches a "how many" question: need its noun nearby
COUNTED_NOUN_WINDOW = 40
_COUNTED_NOUN = re.compile(r"\bhow many ([a-z]+)", re.IGNORECASE)
_FIELD = re.compile(r"^([^:]{1,40}?):\s+(.+)$")
FIELD_MIN_KEY_MATCH = 0.5
# Question wording -> the label a page is likely to use. Applied to both sides after stemming.
SYNONYMS = {
    "birth": "born", "birthday": "born", "rated": "rating", "star": "rating",
    "cost": "price", "costs": "price", "priced": "price", "available": "availability",
    "stock": "availability", "inhabitant": "population", "people": "population",
    "kilometer": "km", "kilometre": "km", "situated": "located", "location": "located",
}
_GENERIC_TERMS = frozenset({"many", "number", "city", "copie", "copy", "unit", "square", "km",
                            "listed", "shown", "page", "site", "website", "value", "code"})

SYSTEM_PROMPT = (
    "You read passages taken from web pages and extract the ONE specific value that answers the "
    "question: a price, date, name, number, version, email, phone number, title or short phrase. "
    "Copy it exactly as written in the passage — no explanation, no full sentence, no units or "
    "words that are not part of the value. Reply with JSON only: "
    '{"answer": "<value>", "passage": <passage number>}. '
    'If no passage contains the answer, reply {"answer": "", "passage": 0}.'
)


class LLM(Protocol):
    def complete(
        self, messages: list[dict[str, str]], max_new_tokens: int, timeout: float
    ) -> str | None: ...


@dataclass(frozen=True)
class Answer:
    value: str
    citations: tuple[str, ...]
    confidence: float
    method: str


NO_ANSWER = Answer(value="", citations=(), confidence=0.0, method="none")


def normalise(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", folded).strip()


def _build_messages(question: str, passages: list[Passage]) -> list[dict[str, str]]:
    lines = [
        f"[{i}] (source: {p.url} | {p.title}) {p.text[:LLM_PASSAGE_CHARS]}"
        for i, p in enumerate(passages, start=1)
    ]
    user = f"Question: {question}\n\nPassages:\n" + "\n".join(lines)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _parse_reply(reply: str) -> tuple[str, int] | None:
    """First JSON object in the reply (values may contain braces); null/non-text = no answer."""
    decoder = json.JSONDecoder()
    data = None
    for start in (i for i, ch in enumerate(reply) if ch == "{"):
        try:
            data, _ = decoder.raw_decode(reply, start)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(data, dict):
        return None
    raw = data.get("answer")
    if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
        return None
    value = str(raw).strip().strip('"')
    try:
        index = int(data.get("passage", 0))
    except (TypeError, ValueError):
        index = 0
    return (value, index) if value else None


def _is_refusal(reply: str) -> bool:
    """The model replied with valid JSON whose answer is empty/null: "not in these passages"."""
    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(reply) if ch == "{"):
        try:
            data, _ = decoder.raw_decode(reply, start)
        except json.JSONDecodeError:
            continue
        return isinstance(data, dict) and data.get("answer") in ("", None)
    return False


REFUSAL = Answer(value="", citations=(), confidence=0.0, method="llm_refusal")


def llm_answer(question: str, passages: list[Passage], llm: LLM, deadline: Deadline) -> Answer | None:
    """The model's Answer; ``REFUSAL`` if it says the passages lack it; None if the call failed."""
    shown = passages[:LLM_PASSAGES]
    if not shown or deadline.remaining() < 1.5:
        return None
    reply = llm.complete(_build_messages(question, shown), LLM_MAX_NEW_TOKENS,
                         timeout=deadline.remaining() - 0.5)
    parsed = _parse_reply(reply) if reply else None
    if parsed is None:
        log.info("LLM gave no usable answer: %r", reply)
        return REFUSAL if reply and _is_refusal(reply) else None
    value, index = parsed
    wanted = normalise(value)
    grounded = [p for p in shown if wanted in normalise(p.text)]
    chosen = shown[index - 1] if 1 <= index <= len(shown) else None
    if chosen is not None and chosen in grounded:
        return Answer(value, (chosen.url,), 0.85, "llm")
    if grounded:
        return Answer(value, (grounded[0].url,), 0.8, "llm")
    cite = chosen or shown[0]
    return Answer(value, (cite.url,), UNGROUNDED_CONFIDENCE, "llm")


def _keyword_bonus(text_lower: str, offset: int, answer_type: AnswerType) -> float:
    window = text_lower[max(0, offset - KEYWORD_WINDOW):offset]
    words = set(re.findall(r"[a-z]+", window))
    return 0.6 if words & set(TYPE_KEYWORDS[answer_type]) else 0.0


def _proximity(text_lower: str, offset: int, terms: list[str]) -> float:
    positions = [m.start() for t in terms for m in re.finditer(re.escape(t), text_lower)]
    if not positions:
        return 0.0
    distance = min(abs(offset - pos) for pos in positions)
    return 1.0 / (1.0 + distance / 20.0)


def rules_answer(question: str, passages: list[Passage]) -> Answer | None:
    answer_type = detect_answer_type(question)
    if answer_type is AnswerType.TEXT:
        return None
    terms = [t for t in tokenize(question) if len(t) > 2]
    best: tuple[float, str, str] | None = None
    for rank, passage in enumerate(passages):
        lower = passage.text.lower()
        for value, offset in find_candidates(passage.text, answer_type):
            score = (1.0 / (1 + rank) + _proximity(lower, offset, terms)
                     + _keyword_bonus(lower, offset, answer_type))
            if best is None or score > best[0]:
                best = (score, value, passage.url)
    if best is None:
        return None
    confidence = min(0.75, 0.35 + best[0] / 5)
    if answer_type is AnswerType.NUMBER and not _counted_noun_near(question, passages, best):
        confidence = min(confidence, UNCUED_NUMBER_CONFIDENCE)
    return Answer(best[1], (best[2],), confidence, "rules")


def _counted_noun_near(question: str, passages: list[Passage],
                       best: tuple[float, str, str]) -> bool:
    """For "how many pages ...", is the word "pages" written next to the chosen number?"""
    match = _COUNTED_NOUN.search(question)
    noun = set(tokenize(match.group(1))) if match else set()
    if not noun:
        return False
    _, value, url = best
    for passage in passages:
        if passage.url != url:
            continue
        for found in re.finditer(re.escape(value), passage.text):
            lo, hi = found.start() - COUNTED_NOUN_WINDOW, found.end() + COUNTED_NOUN_WINDOW
            if noun & set(tokenize(passage.text[max(0, lo):hi])):
                return True
    return False


def _canon(tokens: list[str]) -> set[str]:
    return {SYNONYMS.get(t, t) for t in tokens}


def _fields(text: str) -> list[tuple[str, str]]:
    """``Label: value`` pairs from a passage ("A › B › Key: value | Key2: value2")."""
    pairs = []
    for segment in re.split(r" › | \| ", text):
        match = _FIELD.match(segment.strip())
        if match:
            pairs.append((match.group(1).strip(), match.group(2).strip()))
    return pairs


def _value_for_type(value: str, question: str, answer_type: AnswerType) -> tuple[str, bool]:
    """The value to report, and whether it has the kind of value the question asks for."""
    if answer_type is not AnswerType.TEXT:
        found = find_candidates(value, answer_type)
        if found:
            return found[0][0], True
        return value.strip(" ,;"), False
    if re.match(r"\s*where\b", question, re.IGNORECASE):  # "Born: March 14, 1879 in Ulm, Germany"
        for date, _ in find_candidates(value, AnswerType.DATE):
            value = value.replace(date, " ")
        value = re.sub(r"^\s*(in|at)\s+", "", value.strip(), flags=re.IGNORECASE)
    return value.strip(" ,;"), True


def _label_terms(key: str) -> set[str]:
    """Label words that carry meaning: drop units and numbers such as 'km', '2'."""
    return _canon([t for t in tokenize(key) if len(t) > 2 and not t.isdigit()])


def field_answer(question: str, passages: list[Passage]) -> Answer | None:
    """Match question words against ``Label: value`` labels; the subject must be nearby."""
    answer_type = detect_answer_type(question)
    query = _canon(tokenize(question))
    question_norm = normalise(question)
    best: tuple[float, str, str] | None = None
    for rank, passage in enumerate(passages):
        context = _canon(tokenize(f"{passage.text} {passage.title}"))
        for key, value in _fields(passage.text):
            key_terms = _label_terms(key)
            if not key_terms or not value or normalise(value) in question_norm:
                continue  # an answer is never something the question already says
            key_match = len(key_terms & query) / len(key_terms)
            if key_match < FIELD_MIN_KEY_MATCH:
                continue
            subject = query - key_terms - _GENERIC_TERMS
            coverage = len(subject & context) / len(subject) if subject else 1.0
            if subject and coverage == 0:
                continue
            reported, typed = _value_for_type(value, question, answer_type)
            score = key_match + 1.5 * coverage + 0.5 / (1 + rank) - (0.0 if typed else 1.0)
            if reported and (best is None or score > best[0]):
                best = (score, reported, passage.url)
    if best is None or not best[1]:
        return None
    return Answer(best[1], (best[2],), min(0.8, 0.3 + best[0] / 5), "field")


def best_guess(passages: list[Passage]) -> Answer:
    top = passages[0]
    first_sentence = re.split(r"(?<=[.!?])\s", top.text, maxsplit=1)[0]
    return Answer(first_sentence[:BEST_GUESS_CHARS], (top.url,), 0.15, "best_guess")


def _gate(answer: Answer, min_confidence: float) -> Answer:
    if answer.value and answer.confidence >= min_confidence:
        return answer
    log.info("abstaining: %r at confidence %.2f (%s)", answer.value, answer.confidence,
             answer.method)
    return Answer("", (), answer.confidence, "abstain")


def choose_answer(
    question: str,
    passages: list[Passage],
    llm: LLM | None,
    deadline: Deadline,
    min_confidence: float = MIN_CONFIDENCE,
) -> Answer:
    if not passages:
        return NO_ANSWER
    model = llm_answer(question, passages, llm, deadline) if llm is not None else None
    if model is not None and model.value and model.confidence >= GROUNDED_CONFIDENCE:
        return _gate(model, min_confidence)  # the value appears verbatim in a passage
    # An ungrounded model value may be a hallucination: prefer grounded deterministic answers.
    model_value = model if model is not None and model.value else None
    candidate = (field_answer(question, passages) or rules_answer(question, passages)
                 or model_value or best_guess(passages))
    if model is REFUSAL:
        candidate = replace(candidate,
                            confidence=max(0.0, candidate.confidence - LLM_DISSENT_PENALTY))
    return _gate(candidate, min_confidence)
