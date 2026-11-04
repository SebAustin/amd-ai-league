"""Best-first, same-site crawl that follows the links whose text best matches the question."""

from __future__ import annotations

import heapq
import itertools
import logging
from typing import Protocol
from urllib.parse import urlparse

from amdleague.common.deadline import Deadline
from amdleague.m4.fetch import Fetcher, is_same_site
from amdleague.m4.html_text import Link, Page, parse_page
from amdleague.m4.rank import tokenize
from amdleague.m4.rules import AnswerType, detect_answer_type

log = logging.getLogger(__name__)

DEPTH_PENALTY = 0.35
MIN_FETCH_SECONDS = 0.4
THIN_PAGE_CHARS = 200
_BINARY_EXT = (".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".zip", ".gz", ".mp4",
               ".mp3", ".css", ".js", ".ico", ".woff", ".woff2", ".exe", ".dmg")
# Generic "where would this kind of answer live" hints.
_TYPE_HINTS: dict[AnswerType, tuple[str, ...]] = {
    AnswerType.EMAIL: ("contact", "support", "about", "help"),
    AnswerType.PHONE: ("contact", "support", "about", "help", "location"),
    AnswerType.PRICE: ("price", "pricing", "product", "shop", "store", "plan", "buy"),
    AnswerType.YEAR: ("about", "history", "company", "story"),
    AnswerType.DATE: ("news", "release", "changelog", "event", "blog"),
    AnswerType.VERSION: ("download", "release", "changelog", "doc", "news"),
    AnswerType.NUMBER: (),
    AnswerType.TEXT: ("about",),
}


class Renderer(Protocol):
    def render(self, url: str, timeout: float) -> str | None: ...


def _normalise(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.netloc.lower().removeprefix('www.')}{path}?{parsed.query}"


def link_score(link: Link, query_terms: set[str], hints: tuple[str, ...]) -> float:
    path_words = urlparse(link.url).path.replace("-", " ").replace("_", " ").replace("/", " ")
    terms = set(tokenize(f"{link.text} {path_words}"))
    if not terms:
        return 0.0
    overlap = len(terms & query_terms)
    hint = 0.5 if any(h in t for t in terms for h in hints) else 0.0
    return overlap / (1 + 0.1 * len(terms)) + hint


def _maybe_render(page: Page, url: str, renderer: Renderer | None, deadline: Deadline) -> Page:
    if renderer is None or sum(len(p) for p in page.passages) >= THIN_PAGE_CHARS:
        return page
    if deadline.remaining() < 3:
        return page
    html = renderer.render(url, timeout=min(8.0, deadline.remaining() - 1))
    return parse_page(html, url, citation=page.url) if html else page


def crawl(
    start_url: str,
    question: str,
    fetcher: Fetcher,
    deadline: Deadline,
    max_pages: int = 12,
    max_depth: int = 3,
    renderer: Renderer | None = None,
) -> list[Page]:
    query_terms = set(tokenize(question))
    hints = _TYPE_HINTS[detect_answer_type(question)]
    counter = itertools.count()
    frontier: list[tuple[float, int, str, int]] = [(-1e9, next(counter), start_url, 0)]
    seen: set[str] = {_normalise(start_url)}
    pages: list[Page] = []

    site_roots = [start_url]  # plus wherever the start URL redirected to

    while frontier and len(pages) < max_pages and deadline.remaining() > MIN_FETCH_SECONDS:
        _, _, url, depth = heapq.heappop(frontier)
        doc = fetcher.fetch(url, timeout=deadline.remaining() - 0.2)
        if doc is None:
            continue
        if not pages and not is_same_site(doc.url, start_url):
            site_roots.append(doc.url)
        try:  # one broken page must never cost the pages already read
            page = _maybe_render(parse_page(doc.html, doc.url, citation=doc.citation), doc.url,
                                 renderer, deadline)
        except Exception:
            log.exception("skipping unparseable page %s", doc.url)
            continue
        pages.append(page)
        seen.add(_normalise(doc.url))
        if depth < max_depth:
            for link, priority in _next_links(page, seen, site_roots, query_terms, hints, depth):
                heapq.heappush(frontier, (-priority, next(counter), link.url, depth + 1))
    log.info("crawled %d pages", len(pages))
    return pages


def _next_links(
    page: Page, seen: set[str], site_roots: list[str], query_terms: set[str],
    hints: tuple[str, ...], depth: int,
) -> list[tuple[Link, float]]:
    found = []
    for link in page.links:
        try:
            key = _normalise(link.url)
            if key in seen or not any(is_same_site(link.url, root) for root in site_roots):
                continue
            if urlparse(link.url).path.lower().endswith(_BINARY_EXT):
                continue
            priority = link_score(link, query_terms, hints) - DEPTH_PENALTY * (depth + 1)
        except ValueError:
            continue
        seen.add(key)
        found.append((link, priority))
    return found
