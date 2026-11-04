"""Page sources: live HTTP, or a saved copy of a site on disk (the "corpus")."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse

import httpx
from bs4 import UnicodeDammit

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 6.0
MAX_BYTES = 3_000_000
MAX_CORPUS_FILES = 3000
MAX_REDIRECTS = 5
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0 Safari/537.36 amdleague-m4"
)
_TEXT_TYPES = ("text/html", "application/xhtml", "text/plain", "application/xml", "text/xml",
               "application/json")
CORPUS_SUFFIXES = (".html", ".htm", ".xhtml", ".txt", ".md")


@dataclass(frozen=True)
class Document:
    url: str  # final URL after redirects (or the requested URL for a corpus page)
    citation: str  # what we report back: the URL, or a corpus-relative path
    html: str


class Fetcher(Protocol):
    def fetch(self, url: str, timeout: float | None = None) -> Document | None: ...


def decode_html(body: bytes, declared: str | None) -> str:
    """Use the HTTP charset if given, else sniff BOM / <meta charset>, else utf-8 / cp1252."""
    if declared:
        try:
            return body.decode(declared, errors="replace")
        except LookupError:
            pass
    dammit = UnicodeDammit(body, is_html=True)  # BOM, then <meta charset>, then sniffing
    return dammit.unicode_markup or body.decode("utf-8", errors="replace")


def host_key(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    return host.removeprefix("www.")


def is_same_site(url: str, start_url: str) -> bool:
    return host_key(url) == host_key(start_url) and host_key(url) != ""


class HttpFetcher:
    """GETs one page with tight timeouts and a size cap. Never raises; failures return None."""

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        max_bytes: int = MAX_BYTES,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self._client = httpx.Client(
            follow_redirects=True,
            max_redirects=MAX_REDIRECTS,
            transport=transport,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*;q=0.8"},
        )

    def fetch(self, url: str, timeout: float | None = None) -> Document | None:
        try:
            if urlparse(url).scheme not in ("http", "https"):
                return None
        except ValueError:
            return None
        limit = max(0.5, min(self.timeout, timeout) if timeout else self.timeout)
        end = time.monotonic() + limit  # wall clock for the whole page, not per read
        try:
            with self._client.stream("GET", url, timeout=limit) as resp:
                if resp.status_code >= 400:
                    log.info("HTTP %s for %s", resp.status_code, url)
                    return None
                ctype = resp.headers.get("content-type", "text/html").lower()
                if not ctype.startswith(_TEXT_TYPES):
                    return None
                body = bytearray()
                for chunk in resp.iter_bytes():
                    body.extend(chunk)
                    if len(body) >= self.max_bytes or time.monotonic() >= end:
                        break
                text = decode_html(bytes(body), resp.charset_encoding)
                final = str(resp.url)
        except (httpx.HTTPError, httpx.InvalidURL, ValueError, UnicodeError, LookupError,
                OSError) as exc:
            log.info("fetch failed for %s: %s", url, exc)
            return None
        return Document(url=final, citation=final, html=text)


class CorpusFetcher:
    """Serves pages from a saved copy of a site; citations are paths relative to the root."""

    def __init__(self, root: Path, base_url: str | None = None) -> None:
        self.root = root.resolve()
        self.base_url = base_url

    def _read(self, path: Path) -> Document | None:
        try:
            resolved = path.resolve()
            if not resolved.is_relative_to(self.root) or not resolved.is_file():
                return None
            text = decode_html(resolved.read_bytes()[:MAX_BYTES], None)
        except OSError as exc:
            log.info("cannot read %s: %s", path, exc)
            return None
        rel = resolved.relative_to(self.root).as_posix()
        return Document(url=rel, citation=rel, html=text)

    def iter_documents(self) -> Iterator[Document]:
        """Lazily, so callers can stop when their deadline runs out."""
        files = sorted(
            p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in CORPUS_SUFFIXES
        )
        for path in files[:MAX_CORPUS_FILES]:
            doc = self._read(path)
            if doc is not None:
                yield doc

    def all_documents(self) -> list[Document]:
        return list(self.iter_documents())

    def fetch(self, url: str, timeout: float | None = None) -> Document | None:
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https"):
            if self.base_url and not is_same_site(url, self.base_url):
                return None
            rel = unquote(parsed.path).lstrip("/")
        else:
            rel = unquote(url).lstrip("/")
        base = self.root / rel if rel else self.root
        for candidate in (base, base / "index.html", base.with_name(base.name + ".html")):
            doc = self._read(candidate)
            if doc is not None:
                return doc
        return None
