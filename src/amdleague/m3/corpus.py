"""Walk a Corpus defensively: one bad file never costs the files after it.

The brief's four traps (empty directory, unknown type, unreadable file, encrypted PDF) all end
up in ``skipped`` with a reason; everything else becomes a ``Document``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from amdleague.m3.parsers import Encrypted, kind_of, parse

log = logging.getLogger(__name__)

MAX_FILE_BYTES = 200 * 1024 * 1024


@dataclass(frozen=True)
class Document:
    path: str  # corpus-relative, forward slashes: what a Citation reports
    kind: str  # text | csv | xlsx | docx | pdf | image
    passages: tuple[str, ...]


@dataclass(frozen=True)
class Skipped:
    path: str
    reason: str


def _files(root: Path) -> list[Path]:
    found: list[Path] = []

    def on_error(exc: OSError) -> None:
        log.warning("cannot list %s: %s", getattr(exc, "filename", "?"), exc)

    for directory, subdirs, names in os.walk(root, onerror=on_error):
        subdirs.sort()
        found.extend(Path(directory) / n for n in sorted(names))
    return found


def _inside(path: Path, real_root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(real_root)
    except (OSError, RuntimeError):  # dangling link or loop
        return False


def _read_one(path: Path, rel: str) -> Document | Skipped:
    try:
        if not os.access(path, os.R_OK):
            return Skipped(rel, "unreadable")
        kind = kind_of(path)
        if kind is None:
            return Skipped(rel, "unknown type")
        if path.stat().st_size > MAX_FILE_BYTES:
            return Skipped(rel, "too large")
        if kind == "image":
            with path.open("rb") as handle:  # prove it opens; the model reads it later
                handle.read(1)
            return Document(rel, kind, ())
        passages = tuple(p for p in parse(path, kind) if p.strip())
    except Encrypted:
        return Skipped(rel, "encrypted")
    except PermissionError:
        return Skipped(rel, "unreadable")
    except Exception as exc:  # noqa: BLE001 - any parser failure: record it, keep walking
        log.warning("cannot parse %s: %s", rel, exc)
        return Skipped(rel, f"parse error: {type(exc).__name__}")
    return Document(rel, kind, passages)


def walk(root: Path) -> tuple[list[Document], list[Skipped]]:
    root = Path(root)
    docs: list[Document] = []
    skipped: list[Skipped] = []
    if not root.is_dir():
        log.error("corpus %s is not a directory", root)
        return docs, skipped
    real_root = root.resolve()
    for path in _files(root):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink() and not _inside(path, real_root):
            skipped.append(Skipped(rel, "link outside corpus"))
            continue
        if not path.is_file():
            skipped.append(Skipped(rel, "not a regular file"))
            continue
        result = _read_one(path, rel)
        (docs if isinstance(result, Document) else skipped).append(result)
    log.info("corpus: %d documents, %d skipped", len(docs), len(skipped))
    return docs, skipped
