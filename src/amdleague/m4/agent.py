"""One Request in, one Answer out: gather pages, rank passages, pick the value."""

from __future__ import annotations

import logging

from amdleague.common.deadline import Deadline
from amdleague.m4.answer import LLM, NO_ANSWER, Answer, choose_answer
from amdleague.m4.fetch import CorpusFetcher, Fetcher, HttpFetcher
from amdleague.m4.html_text import Page, parse_page
from amdleague.m4.navigate import Renderer, crawl
from amdleague.m4.rank import rank_passages
from amdleague.m4.request import Request

log = logging.getLogger(__name__)

LLM_RESERVE_SECONDS = 8.0
RULES_RESERVE_SECONDS = 0.5
RANK_K = 12


def _corpus_pages(req: Request, deadline: Deadline) -> list[Page]:
    """A saved copy of the site: read every page and let ranking do the navigating."""
    assert req.corpus is not None
    pages = []
    for doc in CorpusFetcher(req.corpus, base_url=req.url).iter_documents():
        if deadline.expired():
            log.warning("deadline reached after %d corpus pages", len(pages))
            break
        try:
            pages.append(parse_page(doc.html, req.url or doc.url, citation=doc.citation))
        except Exception:
            log.exception("skipping unparseable corpus file %s", doc.citation)
    return pages


def gather_pages(
    req: Request,
    deadline: Deadline,
    fetcher: Fetcher | None = None,
    renderer: Renderer | None = None,
) -> list[Page]:
    if req.corpus is not None:
        return _corpus_pages(req, deadline)
    assert req.url is not None
    return crawl(req.url, req.question, fetcher or HttpFetcher(), deadline, renderer=renderer)


def answer_request(
    req: Request,
    llm: LLM | None,
    deadline: Deadline,
    fetcher: Fetcher | None = None,
    renderer: Renderer | None = None,
) -> Answer:
    reserve = LLM_RESERVE_SECONDS if llm is not None else RULES_RESERVE_SECONDS
    gather_deadline = deadline.child(max(deadline.remaining() - reserve,
                                         deadline.remaining() * 0.5))
    pages = gather_pages(req, gather_deadline, fetcher, renderer)
    if not pages:
        log.warning("no pages read for request %s", req.id)
        return NO_ANSWER
    passages = rank_passages(req.question, pages, k=RANK_K)
    return choose_answer(req.question, passages, llm, deadline)
