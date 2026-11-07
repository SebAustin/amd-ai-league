"""Answer one Question from the indexed Corpus, citing exactly the files the Answer needs.

1. Retrieve passages (BM25, superseded files ranked down) and show them to the model, each
   labelled with its file.
2. Identifiers named in those passages (a ticket id in a log) are looked up up front, and their
   records are shown with the first call — the second hop of a two-file chain. The model can
   still ask for ONE further lookup of an identifier it saw.
3. Citations are then decided by the code, not trusted from the model: a file is kept only if
   it contains the Answer, or it is the link of the chain that supplied the identifier leading
   to it. A value found in no readable file is a Refusal (recalled, hallucinated, or read from a
   file we could not open).
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

from amdleague.common.deadline import Deadline
from amdleague.m3.corpus import Document
from amdleague.m3.retrieve import asks_old, retrieve, superseded_paths
from amdleague.m4.answer import field_answer, rules_answer
from amdleague.m4.rank import Passage

log = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.5
RETRIEVE_K = 8
LOOKUP_K = 4
PASSAGE_CHARS = 700
MAX_NEW_TOKENS = 96
MIN_LLM_SECONDS = 2.0
CITED_CONFIDENCE = 0.9
UNCITED_CONFIDENCE = 0.7
MAX_CITATIONS = 3
MAX_PREFETCH = 4  # records fetched up front for identifiers in the first hop
_IDENTIFIER = re.compile(r"\b(?=[A-Z0-9_-]*\d)(?=[A-Z0-9_-]*[A-Z])[A-Z0-9]+(?:[-_][A-Z0-9]+)+\b|"
                         r"\b[A-Z]{1,4}\d{3,}\b")
# Only these start a chain: tickets and error codes carry long numbers (ORR-1847, E7731);
# product and revision names (TQ-40, REV-C2) do not, and are mentioned everywhere.
_CHAIN_NUMBER = re.compile(r"\d{3,}")

SYSTEM_PROMPT = (
    "You answer questions about a private document collection using ONLY the numbered passages "
    "given. Each passage starts with the file it comes from. The products are fictional: never "
    "use outside knowledge. Reply with one JSON object and nothing else.\n"
    '- If the passages contain the answer: {"answer": "<value>", "files": ["<file>", ...]}. '
    "The value only — a number, part number, version, quarter or code — complete with its "
    "qualifier (\"Q3 FY27\", not \"Q3\"; \"REV-C2\", not \"C2\"), no sentence, no unit words. "
    "List only the files without which the answer would be impossible.\n"
    "- Follow identifiers: when the passage matching the question names a ticket, error code "
    "or part number, the answer is in that identifier's own record (passages marked \"record "
    "for <id>\"), which outranks a similar-sounding description elsewhere.\n"
    "- If a passage names an identifier whose details you need but cannot see: "
    '{"lookup": "<identifier>"}.\n'
    '- Otherwise: {"answer": "", "files": []}. Do not guess. A withdrawn or superseded '
    "document is not the answer unless the question asks about it."
)
NO_MORE_LOOKUPS = "\nYou already used your one lookup: answer now, or reply with an empty answer."


class LLM(Protocol):
    def complete(self, messages: list[dict[str, str]], max_new_tokens: int,
                 timeout: float) -> str | None: ...


@dataclass(frozen=True)
class Result:
    answer: str
    citations: tuple[str, ...]
    confidence: float
    method: str


REFUSAL = Result("", (), 0.0, "refusal")


def squash(text: str) -> str:
    """Letters and digits only, uppercased."""
    return re.sub(r"[^0-9A-Z]", "", text.upper())


def _norm(text: str) -> str:
    """Lowercase, one space between words, thousands separators dropped (10,000 -> 10000)."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return re.sub(r"\s+", " ", text).strip()


_NUMBER = r"[-+]?\d[\d.]*"
_UNIT = r"(?:°\s*[a-z]?|%|[a-z]{1,4})"


def _number_pattern(number: str) -> str:
    # 94 matches "94 C" and "94C" but not "88194" or "94.5"; 4.3.2 never matches "14:32".
    return rf"(?<![\d.]){re.escape(number)}(?!\d|\.\d)"


def _answer_patterns(answer: str) -> list[re.Pattern[str]]:
    value = _norm(answer)
    if not value:
        return []
    patterns = []
    measured = re.fullmatch(rf"({_NUMBER})\s*{_UNIT}?", value)
    if measured:  # a number, possibly with a short unit: "94", "94°C", "180 s", "56 GiB"
        patterns.append(_number_pattern(measured.group(1)))
    words = r"\s*".join(re.escape(t) for t in value.split(" "))
    patterns.append(rf"(?<![0-9a-z]){words}(?![0-9a-z])")
    return [re.compile(p) for p in patterns]


def grounded(answer: str, text: str) -> bool:
    """Is the answer written in the text as a whole value (not inside a longer number/word)?"""
    flat = _norm(text)
    if any(p.search(flat) for p in _answer_patterns(answer)):
        return True
    # Codes written with different separators ("ORR FAN 2214 B"): only for long codes with
    # letters, where squashing cannot create a false match from unrelated digits.
    key = squash(answer)
    return len(key) >= 5 and any(c.isalpha() for c in key) and key in squash(text)


def _messages(question: str, passages: list[Passage], extra: str = "",
              records: dict[tuple[str, str], str] | None = None) -> list[dict[str, str]]:
    records = records or {}

    def label(p: Passage) -> str:
        ident = records.get((p.url, p.text))
        return f"file: {p.url} (record for {ident})" if ident else f"file: {p.url}"

    body = "\n\n".join(f"[{i}] {label(p)}\n{p.text[:PASSAGE_CHARS]}"
                       for i, p in enumerate(passages, start=1))
    user = f"Question: {question}\n\nPassages:\n{body}{extra}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _parse(reply: str | None) -> dict | None:
    if not reply:
        return None
    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(reply) if ch == "{"):
        try:
            data, _ = decoder.raw_decode(reply, start)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _ask(llm: LLM, messages: list[dict[str, str]], deadline: Deadline) -> dict | None:
    if deadline.remaining() < MIN_LLM_SECONDS:
        return None
    return _parse(llm.complete(messages, MAX_NEW_TOKENS, timeout=deadline.remaining() - 0.5))


def _identifiers(text: str) -> set[str]:
    return set(_IDENTIFIER.findall(text))


def _chain_identifiers(text: str) -> set[str]:
    return {i for i in _identifiers(text) if _CHAIN_NUMBER.search(i)}


def _clean_path(raw: object) -> str:
    path = str(raw).strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path.lstrip("/").removeprefix("app/corpus/")


def _cite(question: str, answer: str, claimed: list[str], first_hop: list[Passage],
          shown: list[Passage], docs: dict[str, Document], lookup: str | None) -> tuple[str, ...]:
    """Exactly the files the answer needs: those holding the value, plus — for a chain — the
    one file that supplied the identifier the value was found under."""
    answer_files = [f for f in claimed
                    if f in docs and any(grounded(answer, p) for p in docs[f].passages)]
    if not answer_files:  # model cited nothing usable: best-ranked shown file holding the value
        answer_files = [p.url for p in shown if grounded(answer, p.text)][:1]
    if not answer_files:
        return ()
    old = superseded_paths(tuple(docs.values()))
    wanted = [f for f in answer_files if (f in old) == asks_old(question)]
    cited = list(dict.fromkeys(wanted or answer_files))
    # Identifiers written next to the value (the ORR-1847 in "ORR-1847 | fixed_in: 4.3.2"),
    # minus those the question already gives: only these can link a second file.
    bridges: set[str] = set()
    for f in cited:
        for passage in docs[f].passages:
            if grounded(answer, passage):
                bridges |= _identifiers(passage)
    bridges -= _identifiers(question)
    # A link supplies the identifier, never the value: a file that also holds the value is a
    # second source (e.g. the other revision), not a step of the chain.
    def holds_value(path: str) -> bool:
        return any(grounded(answer, p) for p in docs[path].passages)

    if lookup:
        link = next((p.url for p in first_hop if p.url not in cited and lookup in bridges
                     and lookup in p.text and not holds_value(p.url)), None)
    else:
        link = next((f for f in claimed if f in docs and f not in cited and not holds_value(f)
                     and any(b in p for b in bridges for p in docs[f].passages)), None)
    if link:
        cited.append(link)
    return tuple(cited[:MAX_CITATIONS])


def _prefetch(question: str, docs: tuple[Document, ...],
              shown: list[Passage]) -> list[tuple[Passage, str]]:
    """For each identifier a first-hop passage names but the question does not, its record in
    another file — taken from the first hop when already there, else fetched — in first-hop
    order, at most MAX_PREFETCH records."""
    asked = _identifiers(question)
    old = set() if asks_old(question) else superseded_paths(docs)  # a revision is no chain
    tagged: set[tuple[str, str]] = set()
    records: list[tuple[Passage, str]] = []
    for passage in (p for p in shown if p.url not in old):
        for ident in sorted(_chain_identifiers(passage.text) - asked, key=passage.text.index):
            def is_record(h: Passage, ident: str = ident, src: str = passage.url) -> bool:
                return (h.url != src and h.url not in old and ident in h.text
                        and (h.url, h.text) not in tagged)

            hit = next((h for h in shown if is_record(h)), None) or next(
                (h for h in retrieve(ident, docs, k=LOOKUP_K) if is_record(h)), None)
            if hit is None:
                continue
            tagged.add((hit.url, hit.text))
            records.append((hit, ident))
            if len(records) >= MAX_PREFETCH:
                return records
    return records


def _llm_answer(question: str, docs: tuple[Document, ...], llm: LLM, deadline: Deadline,
                shown: list[Passage]) -> Result | None:
    """A Result, REFUSAL when the model declines, or None when the model could not be used."""
    first_hop = shown
    fetched = _prefetch(question, docs, shown)
    records = {(p.url, p.text): ident for p, ident in fetched}
    in_first_hop = {(p.url, p.text) for p in shown}
    shown = shown + [p for p, _ in fetched if (p.url, p.text) not in in_first_hop]
    reply = _ask(llm, _messages(question, shown, records=records), deadline)
    if reply is None:
        return None
    lookup = None
    if "lookup" in reply and "answer" not in reply:
        lookup = str(reply.get("lookup") or "").strip()
        found = [p for p in retrieve(lookup, docs, k=LOOKUP_K) if lookup in p.text] if lookup else []
        seen = {(p.url, p.text) for p in shown}
        shown = shown + [p for p in found if (p.url, p.text) not in seen]
        reply = _ask(llm, _messages(question, shown, NO_MORE_LOOKUPS, records), deadline)
        if reply is None or "answer" not in reply:
            return REFUSAL
    answer = reply.get("answer")
    if not isinstance(answer, (str, int, float)) or isinstance(answer, bool):
        return REFUSAL
    answer = str(answer).strip()
    if not answer:
        return REFUSAL
    files = reply.get("files") if isinstance(reply.get("files"), list) else []
    claimed = [_clean_path(f) for f in files]
    by_path = {d.path: d for d in docs}
    if lookup is None:  # value read from a prefetched record: its identifier was the hop
        lookup = next((ident for (url, text), ident in records.items()
                       if grounded(answer, text)), None)
    citations = _cite(question, answer, claimed, first_hop, shown, by_path, lookup)
    if not citations:
        log.info("refusing %r: found in no readable file", answer)
        return REFUSAL
    cited_by_model = any(c in claimed for c in citations)
    confidence = CITED_CONFIDENCE if cited_by_model else UNCITED_CONFIDENCE
    return Result(answer, citations, confidence, "llm+lookup" if lookup else "llm")


def _rules_answer(question: str, shown: list[Passage]) -> Result | None:
    found = field_answer(question, shown) or rules_answer(question, shown)
    if found is None:
        return None
    return Result(found.value, tuple(found.citations[:1]), found.confidence, found.method)


def answer_question(question: str, docs: tuple[Document, ...] | list[Document], llm: LLM | None,
                    deadline: Deadline, min_confidence: float = MIN_CONFIDENCE) -> Result:
    docs = tuple(docs)
    shown = retrieve(question, docs, k=RETRIEVE_K)
    if not shown:
        return REFUSAL
    result = _llm_answer(question, docs, llm, deadline, shown) if llm is not None else None
    if result is None:  # no model, or it failed: grounded rules only
        result = _rules_answer(question, shown) or REFUSAL
    if not result.answer or result.confidence < min_confidence:
        return Result("", (), result.confidence, "refusal")
    return result
