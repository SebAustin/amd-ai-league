"""Turn one Corpus file into retrievable passages. Each parser may raise; the walk catches it.

Tables (csv, xlsx, docx tables) become one passage per row, every cell labelled with its column
header, so a row like ``ticket: ORR-1847 | fixed_in: 4.3.2`` is found by either value.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable
from pathlib import Path

WINDOW_CHARS = 600
OVERLAP_LINES = 2
LONG_LINE_OVERLAP = 120  # chars repeated between the pieces of one over-long line
SNIFF_BYTES = 8192
MAX_ROWS = 20_000
CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


class Encrypted(Exception):
    """The file opens but its content needs a password we are not given."""


def _pieces(line: str, limit: int) -> list[str]:
    """Split an over-long line into overlapping pieces instead of cutting off its end."""
    if len(line) <= limit:
        return [line]
    step = limit - LONG_LINE_OVERLAP
    return [line[i:i + limit] for i in range(0, len(line) - LONG_LINE_OVERLAP, step)]


def windows(lines: Iterable[str], limit: int = WINDOW_CHARS) -> list[str]:
    """Group lines into passages of at most ``limit`` chars, overlapping by a couple of lines
    so a fact split across a window edge still appears whole in one passage."""
    return _group((piece for raw in lines for piece in _pieces(raw.rstrip(), limit)), limit)


def _group(lines: Iterable[str], limit: int) -> list[str]:
    passages: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if not line.strip():
            continue
        if current and size + len(line) + 1 > limit:
            passages.append("\n".join(current))
            current = current[-OVERLAP_LINES:]
            size = sum(len(x) + 1 for x in current)
            while current and size + len(line) + 1 > limit:
                size -= len(current.pop(0)) + 1
        current.append(line)
        size += len(line) + 1
    if current:
        passages.append("\n".join(current))
    return passages


def _decode(data: bytes) -> str:
    for encoding in CSV_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _label_rows(rows: Iterable[list[str]], prefix: str = "") -> list[str]:
    """First non-empty row is the header; every later row becomes ``h1: v1 | h2: v2``."""
    passages: list[str] = []
    header: list[str] | None = None
    for count, row in enumerate(rows):
        if count >= MAX_ROWS:
            break
        cells = [c.strip() for c in row]
        if not any(cells):
            continue
        if header is None and len(cells) > 1:
            header = [h or f"column {i + 1}" for i, h in enumerate(cells)]
            passages.append(prefix + " | ".join(header))
            continue
        if header is None or len(cells) == 1:
            passages.append(prefix + " ".join(c for c in cells if c))
            continue
        pairs = [f"{header[i] if i < len(header) else f'column {i + 1}'}: {v}"
                 for i, v in enumerate(cells) if v]
        passages.extend(_row_chunks(prefix, pairs))
    return passages


def _row_chunks(prefix: str, pairs: list[str], limit: int = WINDOW_CHARS) -> list[str]:
    """A wide row becomes several passages, each repeating the row's first cell (its key) and
    the last cell, so every value stays next to what identifies the row."""
    whole = prefix + " | ".join(pairs)
    if len(whole) <= limit or len(pairs) < 3:
        return [whole]
    key, last, middle = pairs[0], pairs[-1], pairs[1:-1]
    chunks, current = [], []
    budget = limit - len(prefix) - len(key) - len(last) - 6
    for pair in middle:
        if current and sum(len(x) + 3 for x in current) + len(pair) > budget:
            chunks.append(current)
            current = []
        current.append(pair[:max(budget, 40)])
    if current:
        chunks.append(current)
    return [prefix + " | ".join([key, *chunk, last]) for chunk in chunks]


def parse_text(path: Path) -> list[str]:
    return windows(_decode(path.read_bytes()).splitlines())


def parse_csv(path: Path) -> list[str]:
    text = _decode(path.read_bytes())
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    return _label_rows(csv.reader(io.StringIO(text), dialect))


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def parse_xlsx(path: Path) -> list[str]:
    import openpyxl

    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        passages: list[str] = []
        for sheet in book.worksheets:
            rows = ([_cell(v) for v in row] for row in sheet.iter_rows(values_only=True))
            passages.extend(_label_rows(rows, prefix=f"{sheet.title} › "))
        return passages
    finally:
        book.close()


def parse_docx(path: Path) -> list[str]:
    import docx

    document = docx.Document(str(path))
    passages = windows(p.text for p in document.paragraphs)
    for table in document.tables:
        rows = ([cell.text for cell in row.cells] for row in table.rows)
        passages.extend(_label_rows(rows))
    return passages


def parse_pdf(path: Path) -> list[str]:
    from pypdf import PdfReader

    with path.open("rb") as handle:
        head = handle.read(1 << 20)
    if b"/Encrypt" in head:  # do not even try: opening may attempt decryption
        raise Encrypted(path.name)
    reader = PdfReader(path)
    if reader.is_encrypted:
        raise Encrypted(path.name)
    lines: list[str] = []
    for page in reader.pages:
        lines.extend((page.extract_text() or "").splitlines())
    return windows(lines)


TEXT_SUFFIXES = frozenset({
    ".txt", ".log", ".py", ".md", ".rst", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".conf", ".xml", ".html", ".htm", ".sh", ".js", ".ts", ".c", ".h", ".cpp", ".java", ".go",
    ".rs", ".sql", ".env", ".properties",
})
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"})
PARSERS = {".csv": parse_csv, ".tsv": parse_csv, ".xlsx": parse_xlsx, ".xlsm": parse_xlsx,
           ".docx": parse_docx, ".pdf": parse_pdf}


def looks_like_text(path: Path) -> bool:
    """No NUL byte and valid UTF-8 in the first 8 KB: README, .jsonl, .tex, extensionless..."""
    with path.open("rb") as handle:
        head = handle.read(SNIFF_BYTES)
    if not head or b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as exc:
        return exc.start >= len(head) - 4  # only a multi-byte char cut at the sniff boundary
    return True


def kind_of(path: Path) -> str | None:
    """Raises OSError if the file must be sniffed and cannot be opened."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in TEXT_SUFFIXES:
        return "text"
    if suffix in PARSERS:
        return suffix.lstrip(".")
    return "text" if looks_like_text(path) else None


def parse(path: Path, kind: str) -> list[str]:
    if kind == "text":
        return parse_text(path)
    return PARSERS[path.suffix.lower()](path)
