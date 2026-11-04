"""HTML -> Page: clean text passages (tables kept as header-labelled rows) and outgoing links."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urldefrag, urljoin

from bs4 import BeautifulSoup, Tag

log = logging.getLogger(__name__)

MAX_PASSAGE_CHARS = 400
MAX_LINKS = 400
_DROP_TAGS = ("script", "style", "noscript", "template", "svg", "iframe", "head")
_SKIP_SCHEMES = ("mailto:", "javascript:", "tel:", "data:")
MAX_HEADING_CHARS = 80
_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_BLOCK_TAGS = ("p", "div", "li", "ul", "ol", "dl", "dt", "dd", "section", "article", "header",
               "footer", "main", "aside", "nav", "form", "blockquote", "pre", "figure",
               "figcaption", "address", "tr", "caption", "label", "option", "summary", "details")
_BLOCK_MARK = " "  # never appears in real text after _clean
_HEADING_MARK = "⁣"
_RATING_CLASS = re.compile(r"rating|stars?\b", re.IGNORECASE)
_WS = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Link:
    text: str
    url: str


@dataclass(frozen=True)
class Page:
    url: str
    title: str
    passages: tuple[str, ...]
    links: tuple[Link, ...]


def _clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _flatten_json_ld(node: Any, out: list[str], depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(node, list):
        for item in node:
            _flatten_json_ld(item, out, depth + 1)
    elif isinstance(node, dict):
        for key, value in node.items():
            if key == "@context":
                continue
            if isinstance(value, (dict, list)):
                _flatten_json_ld(value, out, depth + 1)
            elif value not in (None, ""):
                out.append(f"{key.lstrip('@')}: {_clean(str(value))}")


def _json_ld_passages(soup: BeautifulSoup) -> list[str]:
    passages = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        fields: list[str] = []
        _flatten_json_ld(data, fields)
        if fields:
            passages.append(" | ".join(fields))
    return passages


def _cells(row: Tag) -> list[str]:
    return [_clean(c.get_text(" ")) for c in row.find_all(["th", "td"])]


def _table_passages(table: Tag) -> list[str]:
    rows = table.find_all("tr")
    if not rows:
        return []
    header: list[str] = []
    first = rows[0]
    if first.find("th") and not first.find("td"):
        header, rows = _cells(first), rows[1:]
    passages = []
    for row in rows:
        cells = _cells(row)
        if not any(cells):
            continue
        if header and len(header) == len(cells):
            passages.append(" | ".join(f"{h}: {c}" for h, c in zip(header, cells, strict=True)))
        elif len(cells) == 2 and cells[0] and len(cells[0]) <= 60:
            passages.append(f"{cells[0].rstrip(':')}: {cells[1]}")  # key/value table
        else:
            passages.append(" | ".join(c for c in cells if c))
    return passages


def _rating_value(tag: Tag) -> str | None:
    """Rating widgets often carry the value only in attributes or class names."""
    for attr in ("aria-label", "title", "data-rating", "data-score"):
        value = tag.get(attr)
        if value and re.search(r"\d|one|two|three|four|five", str(value), re.IGNORECASE):
            return _clean(str(value))
    classes = [c for c in tag.get("class") or [] if not _RATING_CLASS.search(c)]
    return " ".join(classes) if classes and len(classes) <= 2 else None


def _inline_ratings(body: Tag) -> None:
    for tag in body.find_all(class_=_RATING_CLASS):
        if _clean(tag.get_text(" ")):
            continue
        value = _rating_value(tag)
        if value:
            tag.append(f"{_BLOCK_MARK}Rating: {value}{_BLOCK_MARK}")


def _mark_blocks(body: Tag) -> None:
    """Newline markers only at block boundaries, so inline label/value pairs stay together."""
    for br in body.find_all("br"):
        br.replace_with(_BLOCK_MARK)
    for tag in body.find_all(_HEADING_TAGS):
        text = _clean(tag.get_text(" "))[:MAX_HEADING_CHARS]
        tag.replace_with(f"{_BLOCK_MARK}{_HEADING_MARK}{tag.name[1]}{text}{_BLOCK_MARK}")
    for tag in body.find_all(_BLOCK_TAGS):
        tag.insert_before(_BLOCK_MARK)
        tag.insert_after(_BLOCK_MARK)


def _body_passages(body: Tag) -> list[str]:
    headings: dict[int, str] = {}
    passages: list[str] = []
    for raw in body.get_text(" ").split(_BLOCK_MARK):
        text = _clean(raw)
        if not text:
            continue
        if text.startswith(_HEADING_MARK) and len(text) > 1 and text[1].isdigit():
            level, heading = int(text[1]), text[2:].strip()
            headings = {lvl: h for lvl, h in headings.items() if lvl < level}
            if heading:
                headings[level] = heading
                passages.append(heading)
            continue
        context = " › ".join(h for _, h in sorted(headings.items()) if h not in text)
        for chunk in _split_long(text):
            passages.append(f"{context} › {chunk}" if context else chunk)
    return passages


def _split_long(text: str) -> list[str]:
    if len(text) <= MAX_PASSAGE_CHARS:
        return [text]
    chunks, current = [], ""
    for sentence in _SENTENCE_END.split(text):
        if current and len(current) + len(sentence) > MAX_PASSAGE_CHARS:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def _links(soup: BeautifulSoup, base_url: str) -> tuple[Link, ...]:
    links, seen = [], set()
    for a in soup.find_all("a", href=True):
        href = str(a["href"]).strip()
        if not href or href.startswith("#") or href.lower().startswith(_SKIP_SCHEMES):
            continue
        try:
            url = urldefrag(urljoin(base_url, href)).url
        except ValueError:  # e.g. "https://[YOUR-DOMAIN]/" placeholders
            continue
        text = _clean(a.get_text(" ")) or _clean(str(a.get("title", "")))
        if (url, text) in seen:
            continue
        seen.add((url, text))
        links.append(Link(text=text, url=url))
        if len(links) >= MAX_LINKS:
            break
    return tuple(links)


def parse_page(html: str, url: str, citation: str | None = None) -> Page:
    """Parse ``html`` fetched from ``url``; ``citation`` overrides the Page's url (corpus paths)."""
    # Our internal markers must never come from the page itself.
    html = html.replace(_HEADING_MARK, "").replace(_BLOCK_MARK, " ")
    soup = BeautifulSoup(html, "lxml")
    title = _clean(soup.title.get_text()) if soup.title else ""
    passages = _json_ld_passages(soup)
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        passages.append(_clean(str(meta["content"])))
    links = _links(soup, url)
    for tag in soup.find_all(_DROP_TAGS):
        tag.decompose()
    body = soup.body or soup
    _inline_ratings(body)
    for table in body.find_all("table"):
        rows = _table_passages(table)
        table.replace_with(_BLOCK_MARK + _BLOCK_MARK.join(rows) + _BLOCK_MARK)
    _mark_blocks(body)
    passages.extend(_body_passages(body))
    unique = tuple(dict.fromkeys(p for p in passages if p))
    return Page(url=citation or url, title=title, passages=unique, links=links)
