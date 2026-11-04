"""Turn harness arguments (in whatever spelling the harness uses) into validated Requests.

The Match 4 harness contract was not public when this was written, so the parser accepts the
spellings used by earlier Matches and by other entrants, plus a JSON / JSONL ``--input`` file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

QUESTION_KEYS = ("question", "query", "prompt", "task", "request")
URL_KEYS = ("url", "start_url", "website", "start", "link", "site")
ID_KEYS = ("id", "query_id", "request_id", "qid", "task_id")
CORPUS_KEYS = ("corpus", "pages", "site_dir", "corpus_dir")


class RequestError(ValueError):
    def __init__(self, message: str, request_id: str | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id


@dataclass(frozen=True)
class Request:
    id: str
    question: str
    url: str | None
    corpus: Path | None


def build_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False: unknown spellings like "--output" must be ignored, not "ambiguous".
    p = argparse.ArgumentParser(description="Match 4 web information retrieval agent",
                                allow_abbrev=False)
    p.add_argument("--question", "--query", "-q", "--prompt", "--task", dest="question")
    p.add_argument("--url", "--start-url", "--start_url", "--website", "--site", dest="url")
    p.add_argument("--corpus", "--pages", "--site-dir", "--corpus-dir", dest="corpus")
    p.add_argument("--id", "--query-id", "--query_id", "--request-id", "--qid", dest="id")
    p.add_argument("--input", "--input-file", "--requests", dest="input")
    p.add_argument("--output-dir", "--output_dir", "--out-dir", dest="output_dir")
    p.add_argument("--output", "--output-file", "--out", dest="output_file")
    return p


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Never exits: argparse usage errors become RequestError."""
    try:
        args, _unknown = build_parser().parse_known_args(argv)
    except SystemExit as exc:
        raise RequestError(f"cannot parse arguments: {argv!r}", _id_from_argv(argv)) from exc
    return args


def _id_from_argv(argv: list[str]) -> str | None:
    for i, token in enumerate(argv):
        if token in ("--id", "--query-id", "--query_id", "--request-id", "--qid"):
            return safe_id(argv[i + 1]) if i + 1 < len(argv) else None
    return None


def safe_id(raw: str) -> str:
    """Keep harness ids verbatim except for anything that could escape the output folder."""
    cleaned = re.sub(r"[/\\\x00]", "_", raw.strip()).replace("..", "_").lstrip(".")
    return cleaned[:160] or "request"


def _derived_id(question: str, url: str | None) -> str:
    return hashlib.sha1(f"{question}|{url}".encode()).hexdigest()[:12]


def _first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def make_request(
    question: Any, url: Any, corpus: Any, raw_id: Any, base_dir: Path | None = None
) -> Request:
    rid = safe_id(str(raw_id)) if raw_id not in (None, "") else None
    if not isinstance(question, str) or not question.strip():
        raise RequestError("a question is required", rid)
    question = question.strip()
    url_str = str(url).strip() if url else None
    if url_str and urlparse(url_str).scheme not in ("http", "https"):
        raise RequestError(f"url must be http or https: {url_str!r}", rid)
    corpus_path = None
    if corpus:
        corpus_path = Path(corpus)
        if base_dir is not None and not corpus_path.is_absolute():
            corpus_path = base_dir / corpus_path
        if not corpus_path.is_dir():
            raise RequestError(f"corpus folder not found: {corpus_path}", rid)
    if not url_str and corpus_path is None:
        raise RequestError("need a url or corpus to search", rid)
    return Request(rid or _derived_id(question, url_str), question, url_str, corpus_path)


def _load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] in "[{":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        if isinstance(data, dict):
            nested = _first(data, ("requests", "queries", "tasks", "items"))
            return list(nested) if isinstance(nested, list) else [data]
    records = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append(None)  # reported as an invalid record, not a lost batch
    return records


def _record_request(record: Any, args: argparse.Namespace, base_dir: Path, index: int,
                    ) -> Request | RequestError:
    if not isinstance(record, dict):
        return RequestError(f"input record {index} is not an object", f"invalid-{index}")
    try:
        return make_request(
            _first(record, QUESTION_KEYS),
            _first(record, URL_KEYS) or args.url,
            _first(record, CORPUS_KEYS) or args.corpus,
            _first(record, ID_KEYS),
            base_dir=base_dir,
        )
    except RequestError as exc:
        exc.request_id = exc.request_id or f"invalid-{index}"
        return exc


def parse_batch(argv: list[str]) -> list[Request | RequestError]:
    """Every request, or the error for it, so one bad record never loses the others."""
    args = parse_args(argv)
    if not args.input:
        return [make_request(args.question, args.url, args.corpus, args.id)]
    path = Path(args.input)
    try:
        records = _load_records(path)
    except (OSError, ValueError) as exc:
        rid = safe_id(args.id) if args.id else None
        raise RequestError(f"cannot read input file {path}: {exc}", rid) from exc
    return [_record_request(r, args, path.parent, i) for i, r in enumerate(records)]


def parse_requests(argv: list[str]) -> list[Request]:
    """Strict form: raise the first error."""
    results = parse_batch(argv)
    for item in results:
        if isinstance(item, RequestError):
            raise item
    return [r for r in results if isinstance(r, Request)]
