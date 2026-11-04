"""Regression tests for the robustness review (H = high, M = medium, L = low severity)."""

import http.server
import json
import socket
import threading
import time

import httpx
import pytest

from amdleague.common import model_server
from amdleague.common.deadline import Deadline
from amdleague.m4.answer import _parse_reply, choose_answer
from amdleague.m4.cli import main
from amdleague.m4.fetch import HttpFetcher, is_same_site
from amdleague.m4.html_text import parse_page
from amdleague.m4.navigate import crawl
from amdleague.m4.rank import Passage
from amdleague.m4.rules import AnswerType, detect_answer_type

BAD_LINKS = """<html><body><h1>Home</h1>
<a href="https://[YOUR-DOMAIN]/x">bad ipv6</a> <a href="http://localhost:${PORT}/">bad port</a>
<a href="/contact.html">Contact</a><p>Hello ⁣x world</p></body></html>"""


def _transport(pages: dict[str, str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = pages.get(request.url.path)
        if body is None:
            return httpx.Response(404)
        return httpx.Response(200, text=body, headers={"content-type": "text/html"})
    return httpx.MockTransport(handler)


# H1 ---------------------------------------------------------------------------------------
def test_h1_malformed_links_do_not_break_parsing():
    page = parse_page(BAD_LINKS, "https://a.example/")
    assert any(link.url == "https://a.example/contact.html" for link in page.links)


def test_h1_same_site_tolerates_bad_urls():
    assert not is_same_site("https://[bad/", "https://a.example/")


def test_h1_fetch_never_raises_on_invalid_url():
    fetcher = HttpFetcher(transport=_transport({}))
    assert fetcher.fetch("http://localhost:${PORT}/") is None


def test_h1_crawl_keeps_pages_after_a_bad_one():
    pages = {"/": BAD_LINKS, "/contact.html": "<p>Email: hi@a.example</p>"}
    got = crawl("https://a.example/", "contact email", HttpFetcher(transport=_transport(pages)),
                Deadline(5), max_pages=5)
    assert [p.url for p in got] == ["https://a.example/", "https://a.example/contact.html"]


# H2 ---------------------------------------------------------------------------------------
class _Trickle(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("content-type", "text/html")
        self.end_headers()
        try:
            for _ in range(20):
                self.wfile.write(b"<p>x</p>" * 10)
                self.wfile.flush()
                time.sleep(0.3)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


def test_h2_slow_trickle_respects_wall_clock_timeout():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Trickle)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        start = time.monotonic()
        HttpFetcher(timeout=1.0).fetch(f"http://127.0.0.1:{server.server_port}/", timeout=1.0)
        assert time.monotonic() - start < 2.0
    finally:
        server.shutdown()


# H3 ---------------------------------------------------------------------------------------
def test_h3_server_survives_client_that_hung_up():
    server_side, client_side = socket.socketpair()
    client_side.sendall(b'{"messages": [], "max_new_tokens": 4}\n')
    client_side.close()  # the client gave up before the reply

    def engine_generate(*_args, **_kwargs):
        return "ok"

    model_server._handle(server_side, ("t", "tok", "m"), generate=engine_generate)  # no raise


# H4 ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("flag", ["--output=/nonexistent/x.json", "--bogus", "--q=hi"])
def test_h4_unexpected_flags_still_write_output(tmp_path, monkeypatch, flag, site_dir):
    monkeypatch.setenv("M4_DISABLE_LLM", "1")
    monkeypatch.setenv("M4_OUTPUT_DIR", str(tmp_path))
    main(["--id", "q1", "-q", "What is the support email address?", "--corpus", str(site_dir),
          flag])
    assert (tmp_path / "q1_output.json").exists()


def test_h4_output_file_flag_writes_that_exact_path(tmp_path, monkeypatch, site_dir):
    monkeypatch.setenv("M4_DISABLE_LLM", "1")
    target = tmp_path / "res" / "answer.json"
    main(["--id", "q1", "-q", "What is the support email address?", "--corpus", str(site_dir),
          "--output", str(target)])
    assert json.loads(target.read_text())["answer"] == "support@acme-tools.example"


# M2 ---------------------------------------------------------------------------------------
def test_m2_how_many_beats_year_cue():
    assert detect_answer_type("How many copies were sold last year?") is AnswerType.NUMBER


def test_m2_when_founded_falls_back_to_year():
    passages = [Passage("Acme was founded in 1998 in Ohio.", "u", "About", 1.0)]
    assert choose_answer("When was Acme founded?", passages, None, Deadline(5)).value == "1998"


# M3 ---------------------------------------------------------------------------------------
def test_m3_null_answer_is_no_answer():
    assert _parse_reply('{"answer": null, "passage": 1}') is None


def test_m3_braces_inside_value_parse():
    assert _parse_reply('noise {"answer": "a}b", "passage": 2} tail') == ("a}b", 2)


class _LLM:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, messages, max_new_tokens, timeout):
        return self.reply


def test_m3_grounded_rules_beat_ungrounded_llm():
    passages = [Passage("Contact: Email: hi@acme.example", "u", "Contact", 1.0)]
    ans = choose_answer("What is the email address?", passages,
                        _LLM('{"answer": "made-up@x.example", "passage": 1}'), Deadline(5))
    assert ans.value == "hi@acme.example"


# M4 ---------------------------------------------------------------------------------------
def test_m4_meta_charset_is_honoured():
    html = '<html><head><meta charset="windows-1252"></head><body>Prix 12,50 \x80 Caf\xe9</body>'
    body = html.encode("latin-1")
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, content=body, headers={"content-type": "text/html"})
    )
    doc = HttpFetcher(transport=transport).fetch("https://a.example/")
    assert doc is not None and "€" in doc.html and "Café" in doc.html


# M7 ---------------------------------------------------------------------------------------
def test_m7_redirected_start_host_is_crawled():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a.example":
            return httpx.Response(301, headers={"location": "https://shop.a.example/"})
        if request.url.path == "/":
            return httpx.Response(200, text='<a href="/price.html">Price</a>',
                                  headers={"content-type": "text/html"})
        return httpx.Response(200, text="<p>Price: $5</p>", headers={"content-type": "text/html"})

    got = crawl("https://a.example/", "price", HttpFetcher(transport=httpx.MockTransport(handler)),
                Deadline(5), max_pages=3)
    assert "https://shop.a.example/price.html" in [p.url for p in got]


# M8 / L2 / L3 -----------------------------------------------------------------------------
def test_m8_one_bad_record_does_not_lose_the_batch(tmp_path, monkeypatch, site_dir):
    monkeypatch.setenv("M4_DISABLE_LLM", "1")
    path = tmp_path / "in.jsonl"
    path.write_text("\n".join([
        json.dumps({"id": "good", "question": "What is the support email address?",
                    "corpus": str(site_dir)}),
        json.dumps({"id": "noq", "corpus": str(site_dir)}),
        "[1]",
    ]))
    out = tmp_path / "out"
    main(["--input", str(path), "--output-dir", str(out)])
    assert json.loads((out / "good_output.json").read_text())["answer"]
    assert "error" in json.loads((out / "noq_output.json").read_text())


def test_m8_bad_input_file_id_is_sanitised(tmp_path, monkeypatch):
    out = tmp_path / "out"
    main(["--input", str(tmp_path / "missing.json"), "--id", "../escaped", "--output-dir", str(out)])
    assert not (tmp_path / "escaped_output.json").exists()
    assert len(list(out.glob("*_output.json"))) == 1


def test_l2_bad_budget_env_does_not_crash(tmp_path, monkeypatch, site_dir):
    monkeypatch.setenv("M4_DISABLE_LLM", "1")
    monkeypatch.setenv("M4_TIME_BUDGET", "fast")
    code = main(["--id", "b", "-q", "What is the support email address?", "--corpus",
                 str(site_dir), "--output-dir", str(tmp_path)])
    assert code == 0
    assert json.loads((tmp_path / "b_output.json").read_text())["answer"]


def test_l3_harmless_ids_are_kept_verbatim(tmp_path, monkeypatch, site_dir):
    monkeypatch.setenv("M4_DISABLE_LLM", "1")
    main(["--id", "q 1-é", "-q", "What is the support email address?", "--corpus",
          str(site_dir), "--output-dir", str(tmp_path)])
    assert json.loads((tmp_path / "q 1-é_output.json").read_text())["id"] == "q 1-é"
