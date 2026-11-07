"""Record the pages the agent fetches live, and replay them offline.

GPU notebooks reach only Hugging Face and pip, so the live eval is recorded on a machine with
internet (``run_eval.py --record DIR``) and replayed next to the GPU (``--replay DIR``): the model
then reads exactly the pages the crawler saw.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from amdleague.m4.fetch import Document, Fetcher


def _key(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest() + ".json"


class RecordingFetcher:
    def __init__(self, inner: Fetcher, root: Path) -> None:
        self.inner, self.root = inner, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def fetch(self, url: str, timeout: float | None = None) -> Document | None:
        doc = self.inner.fetch(url, timeout)
        record = None if doc is None else {"url": doc.url, "citation": doc.citation,
                                           "html": doc.html}
        (self.root / _key(url)).write_text(json.dumps({"requested": url, "doc": record}))
        return doc


class ReplayFetcher:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def fetch(self, url: str, timeout: float | None = None) -> Document | None:
        path = self.root / _key(url)
        if not path.is_file():
            return None
        record = json.loads(path.read_text())["doc"]
        return None if record is None else Document(**record)
