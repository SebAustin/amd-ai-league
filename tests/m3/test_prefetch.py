"""The second hop is fetched up front: a record named by an identifier in the first-hop passages
is shown to the model with the first call, so it never has to remember to ask for it."""

from amdleague.common.deadline import Deadline
from amdleague.m3.answer import answer_question
from amdleague.m3.corpus import Document

from .test_answer import DOCS, ScriptedLLM

CHAIN_Q = ("The production log shows a thermal throttle incident. Which firmware release fixed "
           "the underlying defect?")


def test_record_named_in_the_log_is_shown_in_the_first_call():
    llm = ScriptedLLM({"answer": "4.3.2", "files": ["support/bugs.csv"]})
    answer_question(CHAIN_Q, DOCS, llm, Deadline(20))
    assert len(llm.prompts) == 1
    assert "fixed_in: 4.3.2" in llm.prompts[0]
    assert "record for ORR-1847" in llm.prompts[0]


def test_answer_from_a_prefetched_record_cites_the_file_that_named_it():
    llm = ScriptedLLM({"answer": "4.3.2", "files": ["support/bugs.csv"]})
    result = answer_question(CHAIN_Q, DOCS, llm, Deadline(20))
    assert result.answer == "4.3.2"
    assert set(result.citations) == {"logs/prod.log", "support/bugs.csv"}


def test_answer_found_in_the_first_hop_does_not_gain_a_link():
    llm = ScriptedLLM({"answer": "4.3.2", "files": ["support/bugs.csv"]})
    result = answer_question("Which firmware version fixed ticket ORR-1847?", DOCS, llm,
                             Deadline(20))
    assert result.citations == ("support/bugs.csv",)


def test_identifiers_in_the_question_are_not_prefetched():
    llm = ScriptedLLM({"answer": "4.3.2", "files": ["support/bugs.csv"]})
    answer_question("Which firmware version fixed ticket ORR-1847?", DOCS, llm, Deadline(20))
    assert "record for ORR-1847" not in llm.prompts[0]


def test_a_withdrawn_revision_is_never_a_chain_link():
    llm = ScriptedLLM({"answer": "94", "files": ["specs/tq40_datasheet_r2.pdf"]})
    result = answer_question("What is the maximum junction temperature in the datasheet?", DOCS,
                             llm, Deadline(20))
    assert result.citations == ("specs/tq40_datasheet_r2.pdf",)


def test_prefetch_is_bounded():
    many = tuple(Document(f"t/{i}.txt", "text", (f"ticket ABC-{i:03d} fixed_in 9.{i}",))
                 for i in range(20))
    log = Document("logs/x.log", "text",
                   (" ".join(f"incident ABC-{i:03d}" for i in range(20)) + " thermal throttle",))
    llm = ScriptedLLM({"answer": "", "files": []})
    answer_question(CHAIN_Q, (log, *many), llm, Deadline(20))
    assert llm.prompts[0].count("record for") <= 4


def test_product_names_are_not_chain_identifiers():
    docs = (Document("specs/datasheet.txt", "text", ("TQ-40 datasheet: board power 410 W",)),
            Document("support/asset_label.txt", "text", ("Asset label: TQ-40 board REV-C2",)))
    llm = ScriptedLLM({"answer": "REV-C2", "files": ["support/asset_label.txt"]})
    result = answer_question("What board revision is printed on the unit?", docs, llm,
                             Deadline(20))
    assert result.citations == ("support/asset_label.txt",)
    assert "record for" not in llm.prompts[0]


def test_chain_link_does_not_depend_on_the_model_listing_the_record():
    llm = ScriptedLLM({"answer": "4.3.2", "files": []})
    result = answer_question(CHAIN_Q, DOCS, llm, Deadline(20))
    assert set(result.citations) == {"logs/prod.log", "support/bugs.csv"}
