"""BM25 passage ranking across every page the agent has read."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from amdleague.m4.html_text import Page

K1 = 1.5
B = 0.75
_TOKEN = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "me", "much", "my", "of", "on", "or", "our", "please", "tell", "that", "the", "their", "there", "this", "to", "us", "was", "we", "what", "when", "where", "which", "who", "whom", "whose", "why", "will", "with", "would", "you", "your", "find", "give", "get", "show", "list", "current", "currently", "does", "do", "about"]
)


@dataclass(frozen=True)
class Passage:
    text: str
    url: str
    title: str
    score: float


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


_POSSESSIVE = re.compile(r"['’]s\b")


def tokenize(text: str) -> list[str]:
    text = _POSSESSIVE.sub("", text.lower().replace("’", "'"))
    return [_stem(t) for t in _TOKEN.findall(text) if t not in STOPWORDS]


def rank_passages(question: str, pages: list[Page], k: int = 10) -> list[Passage]:
    docs: list[tuple[str, str, str, list[str]]] = []
    for page in pages:
        title_tokens = tokenize(page.title)
        for text in page.passages:
            docs.append((text, page.url, page.title, tokenize(text) + title_tokens))
    if not docs:
        return []
    query = list(dict.fromkeys(tokenize(question)))
    n = len(docs)
    avg_len = sum(len(d[3]) for d in docs) / n or 1.0
    df = Counter(t for d in docs for t in set(d[3]))
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in query}

    scored = []
    for order, (text, url, title, tokens) in enumerate(docs):
        tf = Counter(tokens)
        norm = K1 * (1 - B + B * len(tokens) / avg_len)
        score = sum(idf[t] * tf[t] * (K1 + 1) / (tf[t] + norm) for t in query if tf[t])
        scored.append((score, -order, Passage(text, url, title, score)))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [p for _, _, p in scored[:k]]
