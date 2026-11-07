import json

from amdleague.common.deadline import Deadline
from amdleague.m3.corpus import Document
from amdleague.m3.index import build_index, load_index
from amdleague.m3.retrieve import retrieve, superseded_paths


def test_index_round_trip_with_image_transcription(corpus, tmp_path):
    seen = []

    def transcribe(path):
        seen.append(path.name)
        return "THERM_ALERT# B14" if path.name == "pinout.png" else None

    out = tmp_path / "index"
    built = build_index(corpus, out, transcribe=transcribe, deadline=Deadline(60))
    loaded = load_index(out)
    assert loaded is not None
    assert [d.path for d in loaded.documents] == [d.path for d in built.documents]
    pinout = next(d for d in loaded.documents if d.path == "specs/pinout.png")
    assert "B14" in "\n".join(pinout.passages)
    label = next(d for d in loaded.documents if d.path == "support/label.jpg")
    assert label.passages == ()  # transcription failed: kept, but nothing to retrieve
    assert sorted(seen) == ["label.jpg", "pinout.png"]
    assert any(s.reason == "encrypted" for s in loaded.skipped)


def test_transcription_stops_at_the_deadline(corpus, tmp_path):
    calls = []
    build_index(corpus, tmp_path / "i", transcribe=lambda p: calls.append(p) or "x",
                deadline=Deadline(0))
    assert calls == []


def test_transcriber_errors_never_abort_indexing(corpus, tmp_path):
    def boom(path):
        raise RuntimeError("gpu fell over")

    index = build_index(corpus, tmp_path / "i", transcribe=boom, deadline=Deadline(60))
    assert len(index.documents) == 10


def test_missing_or_corrupt_index_loads_as_none(tmp_path):
    assert load_index(tmp_path / "none") is None
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "index.json").write_text("{not json")
    assert load_index(tmp_path / "bad") is None
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "index.json").write_text(json.dumps({"version": -1}))
    assert load_index(tmp_path / "old") is None


DOCS = [
    Document("specs/tq40_datasheet_r1_WITHDRAWN.pdf", "pdf",
             ("TQ-40 Datasheet, revision 1 - WITHDRAWN\nMaximum junction temperature 105 C",)),
    Document("specs/tq40_datasheet_r2.pdf", "pdf",
             (("TQ-40 Datasheet, revision 2 - supersedes revision 1\n"
               "Maximum junction temperature 94 C\nRevision 1 is withdrawn."),)),
    Document("docs/manual_v1.txt", "text", ("Manual v1: fan speed 3000 rpm",)),
    Document("docs/manual_v2.txt", "text", ("Manual v2: fan speed 2800 rpm",)),
]


def test_withdrawn_and_older_revisions_are_marked_superseded():
    assert superseded_paths(DOCS) == {"specs/tq40_datasheet_r1_WITHDRAWN.pdf",
                                      "docs/manual_v1.txt"}


def test_current_revision_outranks_withdrawn_one():
    top = retrieve("What is the maximum junction temperature of the TQ-40?", DOCS, k=4)
    assert top[0].url == "specs/tq40_datasheet_r2.pdf"


def test_asking_about_the_old_revision_lifts_the_penalty():
    top = retrieve("What junction temperature did the withdrawn revision 1 datasheet list?",
                   DOCS, k=4)
    assert top[0].url == "specs/tq40_datasheet_r1_WITHDRAWN.pdf"


def test_identifier_lookup_finds_exact_token():
    docs = [Document("support/bugs.csv", "csv",
                     ("ticket: ORR-1847 | summary: sensor reads high | fixed_in: 4.3.2",
                      "ticket: ORR-1802 | summary: hang | fixed_in: 4.3.1"))]
    top = retrieve("ORR-1847", docs, k=1)
    assert "4.3.2" in top[0].text


def test_naming_the_current_revision_does_not_boost_the_old_one():
    top = retrieve("What is the maximum junction temperature in revision 2 of the TQ-40 datasheet?",
                   DOCS, k=4)
    assert top[0].url == "specs/tq40_datasheet_r2.pdf"
