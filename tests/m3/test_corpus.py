from amdleague.m3.corpus import walk


def _texts(docs, path):
    doc = next(d for d in docs if d.path == path)
    return "\n".join(doc.passages)


def test_every_readable_type_is_parsed_with_relative_paths(corpus):
    docs, _ = walk(corpus)
    paths = {d.path for d in docs}
    assert paths >= {
        "engineering/ingest.py", "engineering/release_notes.txt", "logs/prod.log",
        "planning/roadmap.docx", "specs/tq40_datasheet_r2.pdf",
        "specs/tq40_datasheet_r1_WITHDRAWN.pdf", "support/bugs.csv", "support/parts.xlsx",
        "specs/pinout.png", "support/label.jpg",
    }
    assert all("\\" not in p and not p.startswith("/") for p in paths)


def test_traps_are_skipped_without_stopping_the_walk(corpus, is_root):
    docs, skipped = walk(corpus)
    reasons = {s.path: s.reason for s in skipped}
    assert reasons["vendor/supplier_ENCRYPTED.pdf"] == "encrypted"
    assert reasons["vendor/capture.dat"] == "unknown type"
    if not is_root:  # root ignores mode bits; the harness drops DAC_OVERRIDE instead
        assert reasons["vendor/aaa_audit.txt"] == "unreadable"
    assert not any(d.path.startswith("vendor/") for d in docs)
    assert len(docs) == 10  # nothing after the locked file was lost


def test_python_constant_and_comment_are_both_kept(corpus):
    text = _texts(walk(corpus)[0], "engineering/ingest.py")
    assert "DEFAULT_BATCH_TIMEOUT_S = 180" in text
    assert "Raised from 60" in text


def test_csv_rows_carry_their_headers(corpus):
    docs, _ = walk(corpus)
    rows = next(d for d in docs if d.path == "support/bugs.csv").passages
    assert any("ticket: ORR-1847" in r and "fixed_in: 4.3.2" in r for r in rows)


def test_xlsx_rows_carry_sheet_and_headers(corpus):
    text = _texts(walk(corpus)[0], "support/parts.xlsx")
    assert "RMA Parts › part_number: ORR-FAN-2214-B | description: Fan assembly" in text
    assert "Lead times exclude customs." in text


def test_docx_paragraphs_and_tables(corpus):
    text = _texts(walk(corpus)[0], "planning/roadmap.docx")
    assert "TQ-60 enters customer sampling in Q3 FY27." in text
    assert "part: TQ-40 | status: volume production" in text


def test_pdf_text_is_extracted(corpus):
    text = _texts(walk(corpus)[0], "specs/tq40_datasheet_r2.pdf")
    assert "94 C" in text


def test_images_wait_for_transcription(corpus):
    docs, _ = walk(corpus)
    image = next(d for d in docs if d.path == "specs/pinout.png")
    assert image.kind == "image"
    assert image.passages == ()


def test_long_text_is_split_into_overlapping_windows(tmp_path):
    root = tmp_path / "c"
    root.mkdir()
    lines = [f"line {i:03d} value V{i}" for i in range(200)]
    (root / "big.log").write_text("\n".join(lines))
    doc = walk(root)[0][0]
    assert len(doc.passages) > 3
    assert all(len(p) <= 700 for p in doc.passages)
    joined = "\n".join(doc.passages)
    assert all(f"V{i}" in joined for i in range(200))


def test_empty_and_missing_corpus(tmp_path):
    (tmp_path / "empty").mkdir()
    assert walk(tmp_path / "empty") == ([], [])
    assert walk(tmp_path / "nope") == ([], [])
