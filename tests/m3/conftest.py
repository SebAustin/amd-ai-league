"""A synthetic Match 3 Corpus with every file type and all four traps from the brief:
an empty directory, an unknown binary type, an unreadable file and an encrypted PDF."""

from __future__ import annotations

import csv
import os
import struct
import zlib
from pathlib import Path

import docx
import openpyxl
import pytest
from pypdf import PdfReader, PdfWriter


def make_pdf(path: Path, lines: list[str]) -> None:
    """A minimal one-page PDF whose text pypdf can extract."""
    ops = ["BT", "/F1 11 Tf", "14 TL", "50 780 Td"]
    for line in lines:
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        ops.append(f"({safe}) Tj T*")
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
         b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1, xref)
    path.write_bytes(bytes(out))


def make_encrypted_pdf(path: Path, plain: Path) -> None:
    writer = PdfWriter(clone_from=PdfReader(plain))
    writer.encrypt(user_password="s3cret", owner_password="s3cret", algorithm="RC4-128")
    with path.open("wb") as handle:
        writer.write(handle)


def make_png(path: Path) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    raw = b"\x00\xff\xff\xff"  # one white RGB pixel
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    for sub in ("archive", "engineering", "logs", "planning", "specs", "support", "vendor"):
        (root / sub).mkdir(parents=True)

    (root / "engineering" / "ingest.py").write_text(
        '"""Ingest service."""\n\n# Raised from 60 after the backlog incident.\n'
        "DEFAULT_BATCH_TIMEOUT_S = 180\nMAX_INFLIGHT = 12\n")
    (root / "engineering" / "release_notes.txt").write_text(
        "Firmware release notes\n\n4.3.1  Fixes a rare doorbell hang.\n"
        "4.3.3  Packaging only.\n\nThe bug database is authoritative for fixes.\n")
    (root / "logs" / "prod.log").write_text(
        "2026-09-02T03:14:22Z node: WARN thermal: die 0 at 91C\n"
        "2026-09-02T03:14:41Z node: ERROR E7731: thermal throttle engaged on die 0\n"
        "2026-09-02T03:19:03Z node: INFO thermal: throttle released\n"
        "2026-09-02T03:19:03Z node: INFO incident logged against ORR-1847\n")

    roadmap = docx.Document()
    roadmap.add_heading("Accelerator Roadmap", level=1)
    roadmap.add_paragraph("TQ-60 enters customer sampling in Q3 FY27.")
    table = roadmap.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "part", "status"
    table.rows[1].cells[0].text, table.rows[1].cells[1].text = "TQ-40", "volume production"
    roadmap.save(root / "planning" / "roadmap.docx")

    make_pdf(root / "specs" / "tq40_datasheet_r2.pdf",
             ["TQ-40 Datasheet, revision 2 - supersedes revision 1",
              "Maximum junction temperature .......... 94 C"])
    make_pdf(root / "specs" / "tq40_datasheet_r1_WITHDRAWN.pdf",
             ["TQ-40 Datasheet, revision 1 - WITHDRAWN",
              "Maximum junction temperature .......... 105 C"])
    make_png(root / "specs" / "pinout.png")
    make_png(root / "support" / "label.jpg")

    with (root / "support" / "bugs.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ticket", "summary", "fixed_in"])
        writer.writerow(["ORR-1847", "Die temperature sensor reads high", "4.3.2"])
        writer.writerow(["ORR-1802", "Doorbell hang", "4.3.1"])

    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "RMA Parts"
    sheet.append(["part_number", "description", "compatible_with"])
    sheet.append(["ORR-FAN-2214-B", "Fan assembly, field replaceable", "TQ-40"])
    sheet.append(["ORR-FAN-1990-A", "Fan assembly, legacy", "TQ-20"])
    book.create_sheet("Notes").append(["Lead times exclude customs."])
    book.save(root / "support" / "parts.xlsx")

    make_encrypted_pdf(root / "vendor" / "supplier_ENCRYPTED.pdf",
                       root / "specs" / "tq40_datasheet_r2.pdf")
    (root / "vendor" / "capture.dat").write_bytes(bytes(range(256)) * 8)
    locked = root / "vendor" / "aaa_audit.txt"  # sorts first: must not stop the walk
    locked.write_text("audit secrets")
    locked.chmod(0)
    yield root
    locked.chmod(0o600)


@pytest.fixture
def is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0
