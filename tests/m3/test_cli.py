import json

import pytest

from amdleague.m3 import cli


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("M3_DISABLE_LLM", "1")
    monkeypatch.setenv("M3_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("M3_OUTPUT_DIR", str(tmp_path / "output"))


def _read(tmp_path, qid):
    return json.loads((tmp_path / "output" / f"{qid}_output.json").read_text())


def test_index_then_query_writes_contract_json(corpus, tmp_path):
    assert cli.main(["--index", str(corpus)]) == 0
    assert (tmp_path / "index" / "index.json").exists()
    code = cli.main(["--corpus", str(corpus), "--query-id", "query_04",
                     "--query", "Which firmware version fixed ticket ORR-1847?"])
    assert code == 0
    data = _read(tmp_path, "query_04")
    assert data["answer"] == "4.3.2"
    assert data["citations"] == ["support/bugs.csv"]
    assert 0 <= data["confidence"] <= 1


def test_query_without_index_walks_the_corpus(corpus, tmp_path):
    cli.main(["--corpus", str(corpus), "--query-id", "q1",
              "--query", "Which firmware version fixed ticket ORR-1847?"])
    assert _read(tmp_path, "q1")["answer"] == "4.3.2"


def test_unanswerable_question_is_a_refusal_with_both_keys(corpus, tmp_path):
    cli.main(["--index", str(corpus)])
    cli.main(["--corpus", str(corpus), "--query-id", "q10",
              "--query", "What is the unit price of the TQ-40 at 10,000 unit volume?"])
    data = _read(tmp_path, "q10")
    assert data["answer"] == "" and data["citations"] == []


def test_missing_query_still_writes_a_refusal(tmp_path):
    assert cli.main(["--query-id", "q2"]) == 2
    assert _read(tmp_path, "q2") == {"answer": "", "citations": [], "confidence": 0.0}


def test_query_id_is_used_verbatim_as_the_stem(corpus, tmp_path):
    cli.main(["--corpus", str(corpus), "--query-id", "query_07", "--query", "anything?"])
    assert (tmp_path / "output" / "query_07_output.json").exists()


def test_index_survives_missing_corpus(tmp_path):
    assert cli.main(["--index", str(tmp_path / "nope")]) == 0


def test_transcriber_downscales_and_asks_the_model(corpus, tmp_path):
    seen = {}

    class FakeLLM:
        def complete(self, messages, max_new_tokens, timeout):
            seen["content"] = messages[0]["content"]
            return "PIN B14 THERM_ALERT#"

    text = cli.make_transcriber(FakeLLM(), tmp_path / "scratch")(corpus / "specs" / "pinout.png")
    assert text == "PIN B14 THERM_ALERT#"
    kinds = [part["type"] for part in seen["content"]]
    assert kinds == ["image", "text"]


def test_index_without_ready_model_still_indexes_text(corpus, tmp_path, monkeypatch):
    class NeverReady:
        def wait_ready(self, timeout):
            return False

    monkeypatch.delenv("M3_DISABLE_LLM")
    monkeypatch.setattr(cli, "_llm", lambda: NeverReady())
    assert cli.main(["--index", str(corpus)]) == 0
    assert (tmp_path / "index" / "index.json").exists()
