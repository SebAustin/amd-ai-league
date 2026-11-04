import pytest

from amdleague.m4.rules import AnswerType, detect_answer_type, extract_candidates


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How much does the Cordless Drill Pro cost?", AnswerType.PRICE),
        ("What is the price of the hammer?", AnswerType.PRICE),
        ("When was firmware 2.3.1 released?", AnswerType.DATE),
        ("What year was Acme founded?", AnswerType.YEAR),
        ("What is the latest firmware version?", AnswerType.VERSION),
        ("What is the support email address?", AnswerType.EMAIL),
        ("What phone number can I call?", AnswerType.PHONE),
        ("How many drills are in stock?", AnswerType.NUMBER),
        ("Who founded Acme Tools?", AnswerType.TEXT),
    ],
)
def test_detect_answer_type(question, expected):
    assert detect_answer_type(question) is expected


def test_extracts_prices():
    assert extract_candidates("Now $1,299.00 was $1,499", AnswerType.PRICE)[0] == "$1,299.00"
    assert extract_candidates("Price: £51.77", AnswerType.PRICE) == ["£51.77"]
    assert extract_candidates("costs 20 EUR", AnswerType.PRICE) == ["20 EUR"]


def test_extracts_dates():
    text = "released on March 14, 2026 and updated 2026-04-01"
    assert extract_candidates(text, AnswerType.DATE) == ["March 14, 2026", "2026-04-01"]


def test_extracts_versions_emails_phones_years():
    assert extract_candidates("Firmware version 2.3.1 now", AnswerType.VERSION) == ["2.3.1"]
    assert extract_candidates("mail support@acme.example.", AnswerType.EMAIL) == [
        "support@acme.example"
    ]
    assert extract_candidates("call +1 (555) 010-7788 today", AnswerType.PHONE) == [
        "+1 (555) 010-7788"
    ]
    assert extract_candidates("founded in 1987 by", AnswerType.YEAR) == ["1987"]


def test_text_type_has_no_rule_candidates():
    assert extract_candidates("anything", AnswerType.TEXT) == []
