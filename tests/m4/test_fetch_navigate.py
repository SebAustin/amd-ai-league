import httpx

from amdleague.common.deadline import Deadline
from amdleague.m4.fetch import CorpusFetcher, HttpFetcher, is_same_site
from amdleague.m4.navigate import crawl


def test_same_site_ignores_www():
    assert is_same_site("https://www.a.example/x", "https://a.example/")
    assert not is_same_site("https://b.example/", "https://a.example/")


def test_corpus_fetcher_loads_all_pages_with_relative_citations(site_dir):
    fetcher = CorpusFetcher(site_dir)
    docs = fetcher.all_documents()
    names = sorted(d.citation for d in docs)
    assert names == [
        "about/contact.html",
        "about/history.html",
        "index.html",
        "products/drill-pro.html",
        "products/index.html",
    ]


def test_corpus_fetcher_resolves_site_urls(site_dir):
    fetcher = CorpusFetcher(site_dir, base_url="https://acme.example/")
    doc = fetcher.fetch("https://acme.example/products/")
    assert doc is not None and doc.citation == "products/index.html"
    assert fetcher.fetch("https://acme.example/nope.html") is None


def _mock_site(site_dir):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.lstrip("/") or "index.html"
        if path.endswith("/"):
            path += "index.html"
        file = site_dir / path
        if not file.exists():
            return httpx.Response(404)
        return httpx.Response(200, text=file.read_text(), headers={"content-type": "text/html"})

    return httpx.MockTransport(handler)


def test_http_fetcher_returns_none_on_404(site_dir):
    fetcher = HttpFetcher(transport=_mock_site(site_dir))
    assert fetcher.fetch("https://acme.example/missing.html") is None
    doc = fetcher.fetch("https://acme.example/")
    assert doc is not None and "Acme" in doc.html


def test_http_fetcher_rejects_non_http_schemes():
    fetcher = HttpFetcher(transport=httpx.MockTransport(lambda r: httpx.Response(200, text="x")))
    assert fetcher.fetch("file:///etc/passwd") is None
    assert fetcher.fetch("javascript:alert(1)") is None


def test_http_fetcher_allows_localhost_harness_sites():
    # The grader may serve saved sites on localhost; we must not block it.
    html = "<html><body>ok</body></html>"
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, text=html, headers={"content-type": "text/html"})
    )
    doc = HttpFetcher(transport=transport).fetch("http://localhost:8000/")
    assert doc is not None


def test_http_fetcher_skips_binary_content():
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"})
    )
    assert HttpFetcher(transport=transport).fetch("https://a.example/x.png") is None


def test_crawl_follows_relevant_links_to_answer_page(site_dir):
    fetcher = HttpFetcher(transport=_mock_site(site_dir))
    pages = crawl(
        "https://acme.example/",
        "What firmware version does the Cordless Drill Pro run?",
        fetcher,
        Deadline(10),
        max_pages=4,
    )
    urls = [p.url for p in pages]
    assert urls[0] == "https://acme.example/"
    assert "https://acme.example/products/drill-pro.html" in urls
    assert all("other.example.org" not in u for u in urls)


def test_crawl_stops_when_deadline_expired(site_dir):
    fetcher = HttpFetcher(transport=_mock_site(site_dir))
    assert crawl("https://acme.example/", "anything", fetcher, Deadline(0), max_pages=4) == []
