import json

import pytest

from amdleague.common.deadline import Deadline
from amdleague.m3.answer import answer_question
from amdleague.m3.corpus import Document

DOCS = (
    Document("specs/tq40_datasheet_r1_WITHDRAWN.pdf", "pdf",
             ("TQ-40 Datasheet, revision 1 - WITHDRAWN\nMaximum junction temperature 105 C",)),
    Document("specs/tq40_datasheet_r2.pdf", "pdf",
             ("TQ-40 Datasheet, revision 2\nMaximum junction temperature .......... 94 C",)),
    Document("logs/prod.log", "text",
             (("03:14:41 ERROR E7731: thermal throttle engaged on die 0\n"
              "03:19:03 INFO incident logged against ORR-1847"),)),
    Document("support/bugs.csv", "csv",
             ("ticket | summary | fixed_in",
              "ticket: ORR-1847 | summary: sensor reads high under load | fixed_in: 4.3.2",
              "ticket: ORR-1802 | summary: doorbell hang | fixed_in: 4.3.1")),
    Document("engineering/release_notes.txt", "text",
             ("Firmware release notes\n4.3.1 Fixes a rare doorbell hang.\n4.3.3 Packaging only.",)),
    Document("planning/roadmap.docx", "docx", ("TQ-60 enters customer sampling in Q3 FY27.",)),
)


class ScriptedLLM:
    """Replies in order; records what it was shown."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, messages, max_new_tokens, timeout):
        self.prompts.append(messages[-1]["content"])
        reply = self.replies.pop(0) if self.replies else None
        return json.dumps(reply) if isinstance(reply, dict) else reply


def ask(question, *replies, docs=DOCS):
    llm = ScriptedLLM(*replies)
    return answer_question(question, docs, llm, Deadline(20)), llm


def test_single_hop_answer_cites_its_file():
    result, _ = ask("What is the maximum junction temperature of the TQ-40?",
                    {"answer": "94", "files": ["specs/tq40_datasheet_r2.pdf"]})
    assert (result.answer, result.citations) == ("94", ("specs/tq40_datasheet_r2.pdf",))


def test_units_in_the_answer_still_ground():
    result, _ = ask("What is the maximum junction temperature of the TQ-40?",
                    {"answer": "94°C", "files": ["specs/tq40_datasheet_r2.pdf"]})
    assert result.citations == ("specs/tq40_datasheet_r2.pdf",)


def test_files_that_do_not_support_the_answer_are_pruned():
    result, _ = ask("Which firmware version fixed ticket ORR-1847?",
                    {"answer": "4.3.2", "files": ["support/bugs.csv", "engineering/release_notes.txt",
                                                  "logs/prod.log"]})
    assert result.citations == ("support/bugs.csv",)  # the log only repeats the question's id


def test_lookup_hop_cites_both_links_of_the_chain():
    result, llm = ask(
        "The production log shows a thermal throttle incident. Which firmware release fixed "
        "the underlying defect?",
        {"lookup": "ORR-1847"},
        {"answer": "4.3.2", "files": ["support/bugs.csv"]},
    )
    assert result.answer == "4.3.2"
    assert set(result.citations) == {"logs/prod.log", "support/bugs.csv"}
    assert "fixed_in: 4.3.2" in llm.prompts[1]  # the second hop saw the looked-up row


def test_model_citing_both_chain_files_in_one_hop_keeps_both():
    result, _ = ask(
        "The production log shows a thermal throttle incident. Which firmware release fixed "
        "the underlying defect?",
        {"answer": "4.3.2", "files": ["logs/prod.log", "support/bugs.csv"]},
    )
    assert set(result.citations) == {"logs/prod.log", "support/bugs.csv"}


def test_value_found_in_no_file_is_refused():
    result, _ = ask("What is the unit price of the TQ-40 at 10,000 units?",
                    {"answer": "6412", "files": ["specs/tq40_datasheet_r2.pdf"]})
    assert (result.answer, result.citations, result.method) == ("", (), "refusal")


def test_model_refusal_is_a_refusal():
    result, _ = ask("What is the unit price of the TQ-40 at 10,000 units?",
                    {"answer": "", "files": []})
    assert (result.answer, result.citations) == ("", ())


def test_uncited_but_grounded_answer_cites_the_best_ranked_file():
    result, _ = ask("In which quarter does the TQ-60 enter customer sampling?",
                    {"answer": "Q3 FY27", "files": []})
    assert result.citations == ("planning/roadmap.docx",)


def test_superseded_file_is_dropped_when_current_one_also_supports():
    docs = DOCS + (Document("specs/tq40_brief_v1.txt", "text", ("TQ-40 board power 410 W",)),
                   Document("specs/tq40_brief_v2.txt", "text", ("TQ-40 board power 410 W",)))
    result, _ = ask("What is the board power of the TQ-40?",
                    {"answer": "410", "files": ["specs/tq40_brief_v1.txt",
                                                "specs/tq40_brief_v2.txt"]}, docs=docs)
    assert result.citations == ("specs/tq40_brief_v2.txt",)


@pytest.mark.parametrize("reply", [None, "not json at all"])
def test_failed_model_call_falls_back_to_rules(reply):
    result, _ = ask("In which quarter does the TQ-60 enter customer sampling?", reply)
    # rules cannot read quarters reliably: either a grounded answer or a refusal, never junk
    assert result.answer in ("", "Q3 FY27")
    assert (result.answer == "") == (result.citations == ())


def test_no_model_uses_rules_with_citations():
    result = answer_question("Which firmware version fixed ticket ORR-1847?", DOCS, None,
                             Deadline(5), min_confidence=0.0)
    assert result.answer == "4.3.2"
    assert result.citations == ("support/bugs.csv",)


def test_empty_corpus_is_a_refusal():
    result = answer_question("Anything?", (), ScriptedLLM(), Deadline(5))
    assert (result.answer, result.citations) == ("", ())


def test_only_one_lookup_is_allowed():
    result, llm = ask("The production log shows a thermal throttle incident. Which firmware "
                      "release fixed the underlying defect?",
                      {"lookup": "ORR-1847"}, {"lookup": "ORR-1802"})
    assert result.answer == ""
    assert len(llm.prompts) == 2
