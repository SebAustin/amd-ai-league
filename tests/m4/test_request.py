import json

import pytest

from amdleague.m4.request import parse_requests


def test_accepts_query_and_url_spellings():
    (req,) = parse_requests(["--question", "What is the price?", "--url", "https://a.example/"])
    assert req.question == "What is the price?"
    assert req.url == "https://a.example/"
    assert req.corpus is None
    assert req.id  # derived id is never empty


@pytest.mark.parametrize("flag", ["--query", "--question", "-q", "--prompt"])
def test_question_aliases(flag):
    (req,) = parse_requests([flag, "Who founded it?", "--url", "https://a.example"])
    assert req.question == "Who founded it?"


def test_explicit_id_is_kept():
    (req,) = parse_requests(["--query-id", "q42", "-q", "x?", "--url", "https://a.example"])
    assert req.id == "q42"


def test_corpus_mode(site_dir):
    (req,) = parse_requests(["-q", "x?", "--corpus", str(site_dir)])
    assert req.corpus == site_dir
    assert req.url is None


def test_json_input_file_with_many_requests(tmp_path):
    path = tmp_path / "in.json"
    path.write_text(json.dumps([
        {"id": "a", "question": "Q1?", "url": "https://a.example"},
        {"query_id": "b", "query": "Q2?", "start_url": "https://b.example"},
    ]))
    reqs = parse_requests(["--input", str(path)])
    assert [r.id for r in reqs] == ["a", "b"]
    assert reqs[1].url == "https://b.example"


def test_jsonl_input_file(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text('{"id": "a", "question": "Q1?", "url": "https://a.example"}\n')
    assert parse_requests(["--input", str(path)])[0].question == "Q1?"


def test_rejects_missing_question():
    with pytest.raises(ValueError, match="question"):
        parse_requests(["--url", "https://a.example"])


def test_rejects_non_http_url():
    with pytest.raises(ValueError, match="http"):
        parse_requests(["-q", "x?", "--url", "file:///etc/passwd"])


def test_rejects_missing_source():
    with pytest.raises(ValueError, match="url or corpus"):
        parse_requests(["-q", "x?"])


def test_unsafe_id_is_sanitised():
    (req,) = parse_requests(["--id", "../../etc/x", "-q", "x?", "--url", "https://a.example"])
    assert "/" not in req.id and ".." not in req.id
