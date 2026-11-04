import json

import pytest

from amdleague.common.deadline import Deadline
from amdleague.m4.agent import answer_request
from amdleague.m4.cli import main
from amdleague.m4.request import Request


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def complete(self, messages, max_new_tokens, timeout):
        self.calls += 1
        return self.reply


@pytest.mark.parametrize(
    ("question", "expected", "citation"),
    [
        ("How much does the Cordless Drill Pro cost?", "$149.00", "products/index.html"),
        ("What is the support email address?", "support@acme-tools.example", "about/contact.html"),
        ("What year was Acme Tools founded?", "1987", "about/history.html"),
        ("What firmware version does the Cordless Drill Pro have?", "2.3.1",
         "products/drill-pro.html"),
    ],
)
def test_rules_only_answers_from_corpus(site_dir, question, expected, citation):
    req = Request(id="t", question=question, url=None, corpus=site_dir)
    ans = answer_request(req, llm=None, deadline=Deadline(10))
    assert ans.value == expected
    assert ans.citations[0] == citation


def test_llm_answer_is_used_and_grounded(site_dir):
    llm = FakeLLM('{"answer": "Jane Doe", "passage": 1}')
    req = Request(id="t", question="Who founded Acme Tools?", url=None, corpus=site_dir)
    ans = answer_request(req, llm=llm, deadline=Deadline(10))
    assert ans.value == "Jane Doe"
    assert ans.citations == ("about/history.html",)
    assert ans.method == "llm"
    assert ans.confidence >= 0.7


def test_bad_llm_reply_falls_back_to_rules(site_dir):
    llm = FakeLLM("I am not json")
    req = Request(id="t", question="What is the support email address?", url=None, corpus=site_dir)
    ans = answer_request(req, llm=llm, deadline=Deadline(10))
    assert ans.value == "support@acme-tools.example"
    assert ans.method == "rules"


def test_cli_writes_output_json(site_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("M4_DISABLE_LLM", "1")
    code = main(["--id", "q1", "-q", "What is the support email address?",
                 "--corpus", str(site_dir), "--output-dir", str(tmp_path)])
    assert code == 0
    data = json.loads((tmp_path / "q1_output.json").read_text())
    assert data["answer"] == "support@acme-tools.example"
    assert data["citations"] == ["about/contact.html"]
    assert data["source_url"] == "about/contact.html"
    assert 0 <= data["confidence"] <= 1


def test_cli_writes_valid_json_even_on_bad_input(tmp_path, monkeypatch):
    monkeypatch.setenv("M4_DISABLE_LLM", "1")
    code = main(["--id", "bad", "-q", "x?", "--url", "ftp://nope", "--output-dir", str(tmp_path)])
    assert code == 2
    data = json.loads((tmp_path / "bad_output.json").read_text())
    assert data["answer"] == ""
    assert "error" in data
