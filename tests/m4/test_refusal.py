"""ADR 0002: answer when confident, otherwise return an empty answer (League graders score an
empty answer as correct on unanswerable questions and a guess as wrong)."""

import json

import pytest

from amdleague.common.deadline import Deadline
from amdleague.m4.agent import answer_request
from amdleague.m4.answer import MIN_CONFIDENCE, choose_answer
from amdleague.m4.cli import main
from amdleague.m4.rank import Passage
from amdleague.m4.request import Request


class _LLM:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, messages, max_new_tokens, timeout):
        return self.reply


def test_low_confidence_guess_is_refused_by_default(site_dir):
    req = Request(id="t", question="Who founded Acme Tools?", url=None, corpus=site_dir)
    ans = answer_request(req, llm=None, deadline=Deadline(10))
    assert ans.value == ""
    assert ans.citations == ()
    assert ans.method == "refusal"
    assert ans.confidence < MIN_CONFIDENCE


def test_zero_threshold_restores_always_answer(site_dir):
    req = Request(id="t", question="Who founded Acme Tools?", url=None, corpus=site_dir)
    ans = answer_request(req, llm=None, deadline=Deadline(10), min_confidence=0.0)
    assert ans.value
    assert ans.method == "best_guess"


def test_confident_rule_answer_is_still_given():
    passages = [Passage("Contact: Email: hi@acme.example", "u", "Contact", 1.0)]
    ans = choose_answer("What is the email address?", passages, None, Deadline(5))
    assert ans.value == "hi@acme.example"


def test_ungrounded_llm_value_alone_is_refused():
    passages = [Passage("Acme sells drills and saws.", "u", "Home", 1.0)]
    ans = choose_answer("Who is the CEO of Acme?", passages,
                        _LLM('{"answer": "John Smith", "passage": 1}'), Deadline(5))
    assert ans.value == ""


def test_grounded_llm_value_is_given():
    passages = [Passage("Our CEO is Jane Doe.", "u", "About", 1.0)]
    ans = choose_answer("Who is the CEO of Acme?", passages,
                        _LLM('{"answer": "Jane Doe", "passage": 1}'), Deadline(5))
    assert ans.value == "Jane Doe"


@pytest.mark.parametrize("reply", ['{"answer": "", "passage": 0}', '{"answer": null}'])
def test_llm_saying_not_found_lowers_fallback_confidence(reply):
    passages = [Passage("Office hours: Monday 9am. Call 555-0100 for sales.", "u", "Home", 1.0)]
    question = "What is the phone number of the warehouse?"
    alone = choose_answer(question, passages, None, Deadline(5), min_confidence=0.0)
    dissent = choose_answer(question, passages, _LLM(reply), Deadline(5), min_confidence=0.0)
    assert alone.value == dissent.value == "555-0100"
    assert dissent.confidence < alone.confidence


def test_failed_llm_call_is_not_dissent():
    passages = [Passage("Office hours: Monday 9am. Call 555-0100 for sales.", "u", "Home", 1.0)]
    question = "What is the phone number of the warehouse?"
    alone = choose_answer(question, passages, None, Deadline(5), min_confidence=0.0)
    failed = choose_answer(question, passages, _LLM(None), Deadline(5), min_confidence=0.0)
    assert failed.confidence == alone.confidence


@pytest.mark.parametrize(("env", "expect_empty"), [("", True), ("0", False), ("oops", True)])
def test_cli_threshold_from_env(site_dir, tmp_path, monkeypatch, env, expect_empty):
    monkeypatch.setenv("M4_DISABLE_LLM", "1")
    monkeypatch.setenv("M4_MIN_CONFIDENCE", env)
    main(["--id", "q", "-q", "Who founded Acme Tools?", "--corpus", str(site_dir),
          "--output-dir", str(tmp_path)])
    data = json.loads((tmp_path / "q_output.json").read_text())
    assert (data["answer"] == "") is expect_empty
    assert isinstance(data["citations"], list)


def test_bare_number_without_cue_is_refused():
    passages = [Passage("Tipping the Velvet £53.74 In stock Add to basket", "u", "Velvet", 1.0)]
    ans = choose_answer("How many pages does Tipping the Velvet have?", passages, None, Deadline(5))
    assert ans.value == ""


def test_number_next_to_its_cue_is_given():
    passages = [Passage("The library holds a total of 250 volumes.", "u", "About", 1.0)]
    ans = choose_answer("How many volumes does the library hold in total?", passages, None,
                        Deadline(5))
    assert ans.value == "250"
