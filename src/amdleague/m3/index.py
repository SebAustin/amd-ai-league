"""Build the Corpus index once (``--index``) and persist it for the per-Question processes.

Each Question runs in a new process, so everything expensive — parsing PDFs, spreadsheets and
reading images with the vision model — happens here and is saved as one JSON file.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from amdleague.common.deadline import Deadline
from amdleague.common.output import write_json
from amdleague.m3.corpus import Document, Skipped, walk
from amdleague.m3.parsers import windows

log = logging.getLogger(__name__)

INDEX_VERSION = 1
INDEX_FILE = "index.json"
MIN_TRANSCRIBE_SECONDS = 5.0

Transcriber = Callable[[Path], "str | None"]


@dataclass(frozen=True)
class Index:
    documents: tuple[Document, ...]
    skipped: tuple[Skipped, ...]


def _transcribe_images(root: Path, docs: list[Document], transcribe: Transcriber | None,
                       deadline: Deadline) -> list[Document]:
    out: list[Document] = []
    for doc in docs:
        if doc.kind != "image" or transcribe is None:
            out.append(doc)
            continue
        if deadline.remaining() < MIN_TRANSCRIBE_SECONDS:
            log.warning("no time left to read %s", doc.path)
            out.append(doc)
            continue
        try:
            text = transcribe(root / doc.path)
        except Exception:
            log.exception("transcribing %s failed", doc.path)
            text = None
        passages = tuple(windows(text.splitlines())) if text else ()
        out.append(Document(doc.path, doc.kind, passages))
    return out


def build_index(corpus: Path, out_dir: Path, transcribe: Transcriber | None,
                deadline: Deadline) -> Index:
    docs, skipped = walk(Path(corpus))
    docs = _transcribe_images(Path(corpus), docs, transcribe, deadline)
    index = Index(tuple(docs), tuple(skipped))
    payload = {
        "version": INDEX_VERSION,
        "documents": [{"path": d.path, "kind": d.kind, "passages": list(d.passages)}
                      for d in index.documents],
        "skipped": [{"path": s.path, "reason": s.reason} for s in index.skipped],
    }
    write_json(Path(out_dir) / INDEX_FILE, payload)
    log.info("indexed %d documents (%d images read) into %s", len(docs),
             sum(1 for d in docs if d.kind == "image" and d.passages), out_dir)
    return index


def load_index(out_dir: Path) -> Index | None:
    path = Path(out_dir) / INDEX_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != INDEX_VERSION:
            raise ValueError(f"index version {data.get('version')!r}")
        docs = tuple(Document(d["path"], d["kind"], tuple(d["passages"]))
                     for d in data["documents"])
        skipped = tuple(Skipped(s["path"], s["reason"]) for s in data.get("skipped", []))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        log.warning("ignoring unusable index %s: %s", path, exc)
        return None
    return Index(docs, skipped)
