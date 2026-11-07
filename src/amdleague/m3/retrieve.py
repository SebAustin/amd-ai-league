"""BM25 over every Corpus passage, with superseded documents ranked down.

A withdrawn datasheet holds a real number for the same question, so it retrieves just as well
as the current one; ranking it down (not dropping it) keeps it answerable when a Question asks
about the old revision explicitly.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from pathlib import PurePosixPath

from amdleague.m3.corpus import Document
from amdleague.m4.html_text import Page
from amdleague.m4.rank import Passage, rank_passages

SUPERSEDED_FACTOR = 0.3
# Letter boundaries, not \b: in "datasheet_r1_WITHDRAWN" the underscore is a word character.
_MARKER = re.compile(r"(?<![a-z])(withdrawn|superseded|obsolete|deprecated)(?![a-z])",
                     re.IGNORECASE)
# The document's own status line: "revision 1 - WITHDRAWN", "Status: obsolete",
# "superseded by revision 2" — not a mention such as "replaces withdrawn Rev 1".
_STATUS = re.compile(
    r"(?:[-–—,(]\s*|\bstatus\s*:\s*)(?:withdrawn|superseded|obsolete|deprecated)\b"
    r"|\bsuperseded\s+by\b"
    r"|\bthis\s+(?:document|datasheet|revision|version|spec\w*)\s+is\s+"
    r"(?:withdrawn|superseded|obsolete|deprecated)\b",
    re.IGNORECASE)
_STATUS_CAPS = re.compile(r"\b(?:WITHDRAWN|SUPERSEDED|OBSOLETE|DEPRECATED)\b")  # a stamp
_REVISION = re.compile(r"(?:^|[_\-\s.])(?:rev(?:ision)?|r|v)[_\-]?(\d+(?:\.\d+)*)(?=$|[_\-\s])",
                       re.IGNORECASE)
_ASKS_OLD = re.compile(r"\b(withdrawn|superseded|obsolete|deprecated|old|older|previous|"
                       r"original|earlier|preliminary)\b", re.IGNORECASE)
_NAMES_REVISION = re.compile(r"\b(revision\s*\d+|rev\s*\d+|r\d+|v\d+(?:\.\d+)*)\b", re.IGNORECASE)
HEADER_LINES = 2


def asks_old(question: str) -> bool:
    return bool(_ASKS_OLD.search(question))


def _family(path: str) -> tuple[str, tuple[int, ...]] | None:
    """('specs/tq40_datasheet', 2) for specs/tq40_datasheet_r2.pdf; None without a revision."""
    pure = PurePosixPath(path)
    stem = _MARKER.sub("", pure.stem)
    match = None
    for match in _REVISION.finditer(stem):
        pass  # the last revision-looking token is the revision
    if match is None:
        return None
    base = (stem[:match.start()] + stem[match.end():]).strip("_- .").lower()
    version = tuple(int(n) for n in match.group(1).split("."))
    return f"{pure.parent}/{base}{pure.suffix.lower()}", version


def superseded_paths(docs: list[Document] | tuple[Document, ...]) -> set[str]:
    marked = set()
    families: dict[str, list[tuple[tuple[int, ...], str]]] = defaultdict(list)
    for doc in docs:
        header = "\n".join("\n".join(doc.passages[:1]).splitlines()[:HEADER_LINES])
        if _MARKER.search(PurePosixPath(doc.path).name) or _STATUS.search(header) \
                or _STATUS_CAPS.search(header):
            marked.add(doc.path)
        family = _family(doc.path)
        if family is not None:
            families[family[0]].append((family[1], doc.path))
    for members in families.values():
        if len(members) > 1:
            newest = max(rev for rev, _ in members)
            marked.update(path for rev, path in members if rev < newest)
    return marked


def retrieve(question: str, docs: list[Document] | tuple[Document, ...], k: int = 12) -> list[Passage]:
    pages = [Page(url=d.path, title=d.path.replace("/", " ").replace("_", " "),
                  passages=d.passages, links=()) for d in docs if d.passages]
    ranked = rank_passages(question, pages, k=max(k * 3, 30))
    old = superseded_paths(docs)
    # Asked about the old revision: the superseded file is the source, not the current one.
    if asks_old(question):
        factor = 1 / SUPERSEDED_FACTOR
    elif _NAMES_REVISION.search(question):
        factor = 1.0  # a named revision: let the text decide which file it is
    else:
        factor = SUPERSEDED_FACTOR
    ranked = [replace(p, score=p.score * factor) if p.url in old else p for p in ranked]
    ranked.sort(key=lambda p: -p.score)
    return ranked[:k]
