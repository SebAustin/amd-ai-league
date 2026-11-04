"""Generic extraction behaviours found while scoring the live eval set."""

from amdleague.m4.answer import field_answer
from amdleague.m4.html_text import parse_page
from amdleague.m4.rank import rank_passages, tokenize


def _answer(html: str, question: str):
    page = parse_page(html, "https://x.example/")
    return field_answer(question, rank_passages(question, [page], k=10))


CARDS = """<html><body><h1>Countries</h1>
<div class="country"><h3>Andorra</h3><div>
<strong>Capital:</strong> <span>Andorra la Vella</span><br>
<strong>Population:</strong> <span>84000</span><br>
<strong>Area (km<sup>2</sup>):</strong> <span>468.0</span><br></div></div>
<div class="country"><h3>Iceland</h3><div>
<strong>Capital:</strong> <span>Reykjavik</span><br>
<strong>Population:</strong> <span>308910</span><br></div></div></body></html>"""


def test_inline_label_and_value_stay_in_one_passage():
    page = parse_page(CARDS, "https://x.example/")
    assert any(p.endswith("Population: 84000") and "Andorra" in p for p in page.passages)


def test_field_answer_uses_heading_as_subject():
    assert _answer(CARDS, "What is Iceland's population?").value == "308910"
    assert _answer(CARDS, "What is the capital of Andorra?").value == "Andorra la Vella"


def test_units_in_labels_do_not_block_match():
    assert _answer(CARDS, "What is the area of Andorra?").value == "468.0"


def test_value_already_in_question_is_never_the_answer():
    html = "<h1>Sapiens: A Brief History</h1><table><tr><th>UPC</th><td>abc123</td></tr></table>"
    assert _answer(html, "What is the UPC of Sapiens: A Brief History?").value == "abc123"


def test_rating_from_class_name():
    html = '<h1>Book</h1><p class="star-rating Four"><i></i></p>'
    assert _answer(html, "What is the star rating of Book?").value == "Four"


def test_where_question_strips_date_from_born_field():
    html = "<h3>Ada</h3><p><strong>Born:</strong> <span>December 10, 1815</span> " \
           "<span>in London, England</span></p>"
    assert _answer(html, "Where was Ada born?").value == "London, England"
    assert _answer(html, "When was Ada born?").value == "December 10, 1815"


def test_possessive_is_stripped():
    assert tokenize("Iceland's population") == ["iceland", "population"]
