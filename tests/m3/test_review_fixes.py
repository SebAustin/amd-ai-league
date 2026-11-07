"""Regressions for the Match 3 code review (numbers match the review findings)."""

import json

from amdleague.common.deadline import Deadline
from amdleague.m3 import cli
from amdleague.m3.answer import answer_question, grounded
from amdleague.m3.corpus import Document, walk
from amdleague.m3.parsers import windows
from amdleague.m3.retrieve import superseded_paths


class LLM:
    def __init__(self, *replies):
        self.replies = list(replies)

    def complete(self, messages, max_new_tokens, timeout):
        return json.dumps(self.replies.pop(0)) if self.replies else None


# 1 -------------------------------------------------------------------------------------------
def test_1_long_lines_are_chunked_not_truncated():
    passages = windows(["intro " + "x " * 400 + "The fixed version is 4.3.2"])
    assert any("4.3.2" in p for p in passages)
    assert all(len(p) <= 600 for p in passages)


def test_1_wide_table_rows_keep_their_key_in_every_chunk(tmp_path):
    root = tmp_path / "c"
    root.mkdir()
    header = ["part"] + [f"col{i}" for i in range(60)] + ["lead_time"]
    row = ["ORR-X-1"] + [f"value number {i}" for i in range(60)] + ["21"]
    (root / "wide.csv").write_text(",".join(header) + "\n" + ",".join(row) + "\n")
    passages = walk(root)[0][0].passages
    assert all(len(p) <= 700 for p in passages)
    assert any("part: ORR-X-1" in p and "lead_time: 21" in p for p in passages)


# 2 -------------------------------------------------------------------------------------------
def test_2_version_does_not_ground_in_a_timestamp():
    assert not grounded("4.3.2", "2026-09-02T14:32:01 node ok")
    assert grounded("4.3.2", "fixed_in: 4.3.2")


def test_2_numbers_need_boundaries_and_units_are_tolerated():
    assert grounded("94", "Maximum junction temperature .... 94 C")
    assert grounded("94°C", "Maximum junction temperature .... 94 C")
    assert grounded("94", "max 94C")
    assert not grounded("94", "batch 88194 accepted")
    assert not grounded("3 weeks", "3 units shipped")
    assert grounded("10,000", "volume 10000 units")
    assert grounded("Q3 FY27", "sampling in Q3  FY27.")
    assert grounded("REV-C2", "Board rev-c2")


# 3 -------------------------------------------------------------------------------------------
DOCS = (
    Document("a/datasheet.txt", "text", ("TQ40-A2 board power 410 W",)),
    Document("a/errata.txt", "text", ("TQ40-A2 errata: fan noise at idle",)),
    Document("logs/prod.log", "text", ("ERROR E7731 thermal throttle\nincident logged against ORR-1847",)),
    Document("support/bugs.csv", "csv", ("ticket: ORR-1847 | fixed_in: 4.3.2",)),
    Document("eng/notes.txt", "text", ("Release notes: ORR-1847 mentioned in passing",)),
)


def test_3_over_claimed_file_sharing_only_a_part_number_is_dropped():
    res = answer_question("What is the board power of the TQ40-A2?", DOCS,
                          LLM({"answer": "410", "files": ["a/datasheet.txt", "a/errata.txt"]}),
                          Deadline(10))
    assert res.citations == ("a/datasheet.txt",)


def test_3_lookup_cites_the_file_that_named_the_identifier_not_another_mention():
    res = answer_question("The log shows a thermal throttle. Which firmware fixed the defect?", DOCS,
                          LLM({"lookup": "ORR-1847"}, {"answer": "4.3.2", "files": ["support/bugs.csv"]}),
                          Deadline(10))
    assert set(res.citations) == {"logs/prod.log", "support/bugs.csv"}


# 4 -------------------------------------------------------------------------------------------
def test_4_header_mentioning_a_withdrawn_predecessor_is_not_superseded():
    docs = [Document("d_r2.txt", "text", ("Datasheet Rev 2 replaces withdrawn Rev 1\nmax 94 C",)),
            Document("api.txt", "text", ("Changelog: deprecated API removed in 2.0",)),
            Document("old.txt", "text", ("Datasheet, revision 1 - WITHDRAWN, SUPERSEDED BY REVISION 2",))]
    assert superseded_paths(docs) == {"old.txt"}


def test_4_dotted_versions_group_into_one_family():
    docs = [Document("spec_v1.2.pdf", "pdf", ("a",)), Document("spec_v1.10.pdf", "pdf", ("b",))]
    assert superseded_paths(docs) == {"spec_v1.2.pdf"}


# 5 -------------------------------------------------------------------------------------------
def test_5_question_about_the_withdrawn_revision_cites_it():
    docs = (Document("s/brief_v1.txt", "text", ("TQ-40 board power 410 W",)),
            Document("s/brief_v2.txt", "text", ("TQ-40 board power 410 W",)))
    res = answer_question("What board power did the withdrawn v1 brief list?", docs,
                          LLM({"answer": "410", "files": ["s/brief_v1.txt", "s/brief_v2.txt"]}),
                          Deadline(10))
    assert res.citations == ("s/brief_v1.txt",)


# 6 -------------------------------------------------------------------------------------------
def test_6_index_budget_shrinks_with_container_age(monkeypatch):
    monkeypatch.setattr(cli, "container_age", lambda: 200.0)
    assert cli.index_budget() <= cli.STARTUP_SECONDS - cli.STARTUP_MARGIN - 200.0
    monkeypatch.setattr(cli, "container_age", lambda: 0.0)
    assert cli.index_budget() == cli.INDEX_BUDGET_SECONDS


# 7 -------------------------------------------------------------------------------------------
def test_7_text_files_with_unknown_suffix_are_read_binary_ones_skipped(tmp_path):
    root = tmp_path / "c"
    root.mkdir()
    (root / "README").write_text("Default port is 8443")
    (root / "notes.jsonl").write_text('{"k": "v"}\n')
    (root / "blob.bin").write_bytes(b"\x00\x01\x02" * 100)
    docs, skipped = walk(root)
    assert {d.path for d in docs} == {"README", "notes.jsonl"}
    assert [s.path for s in skipped] == ["blob.bin"]


# 8 -------------------------------------------------------------------------------------------
def test_8_symlink_inside_the_corpus_is_followed_outside_is_not(tmp_path):
    root = tmp_path / "c"
    root.mkdir()
    (root / "real.txt").write_text("value 7")
    (root / "link.txt").symlink_to(root / "real.txt")
    (tmp_path / "secret.txt").write_text("outside")
    (root / "escape.txt").symlink_to(tmp_path / "secret.txt")
    docs, skipped = walk(root)
    assert {d.path for d in docs} == {"real.txt", "link.txt"}
    assert [s.path for s in skipped] == ["escape.txt"]


# 9 -------------------------------------------------------------------------------------------
def test_9_unparseable_arguments_still_write_a_refusal(tmp_path, monkeypatch):
    monkeypatch.setenv("M3_OUTPUT_DIR", str(tmp_path))
    assert cli.main(["--query-id", "q5", "--query", "-5 V rail?", "--bogus"]) in (0, 2)
    data = json.loads((tmp_path / "q5_output.json").read_text())
    assert data["answer"] == "" and data["citations"] == []


# LOW -----------------------------------------------------------------------------------------
def test_dot_slash_citations_are_normalised():
    res = answer_question("Which firmware version fixed ticket ORR-1847?", DOCS,
                          LLM({"answer": "4.3.2", "files": ["./support/bugs.csv"]}), Deadline(10))
    assert res.citations == ("support/bugs.csv",)
    assert res.confidence >= 0.9


def test_4_marker_in_filename_alone_marks_superseded():
    docs = [Document("specs/tq40_datasheet_r1_WITHDRAWN.pdf", "pdf", ("TQ-40 Datasheet\nmax 105 C",))]
    assert superseded_paths(docs) == {"specs/tq40_datasheet_r1_WITHDRAWN.pdf"}
