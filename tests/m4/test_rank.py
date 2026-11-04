from amdleague.m4.html_text import Page
from amdleague.m4.rank import rank_passages, tokenize


def test_tokenize_drops_stopwords_and_lowercases():
    assert tokenize("What IS the Price of the Hammer?") == ["price", "hammer"]


def test_ranks_relevant_passage_first():
    pages = [
        Page(url="u1", title="Home", passages=("We sell hammers.", "Welcome friends"), links=()),
        Page(url="u2", title="Contact", passages=("Support email is a@b.example",), links=()),
    ]
    ranked = rank_passages("What is the support email?", pages, k=2)
    assert ranked[0].url == "u2"
    assert ranked[0].text == "Support email is a@b.example"


def test_title_terms_help_ranking():
    pages = [
        Page(url="u1", title="Cordless Drill Pro", passages=("Battery 4.0 Ah",), links=()),
        Page(url="u2", title="Hammer", passages=("Battery not included",), links=()),
    ]
    ranked = rank_passages("Cordless Drill Pro battery capacity", pages, k=1)
    assert ranked[0].url == "u1"


def test_empty_pages_give_empty_ranking():
    assert rank_passages("anything", [], k=3) == []
