"""Record pages fetched live, then replay them where the web is unreachable (GPU notebooks)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval" / "m4"))

from page_cache import RecordingFetcher, ReplayFetcher

from amdleague.m4.fetch import Document


class Live:
    def __init__(self):
        self.calls = 0

    def fetch(self, url, timeout=None):
        self.calls += 1
        if "missing" in url:
            return None
        return Document(url=url + "/final", citation=url + "/final", html=f"<p>{url}</p>")


def test_recorded_pages_replay_identically(tmp_path):
    live = Live()
    rec = RecordingFetcher(live, tmp_path)
    first = rec.fetch("https://a.test/x", timeout=3)
    assert rec.fetch("https://a.test/missing") is None
    replay = ReplayFetcher(tmp_path)
    assert replay.fetch("https://a.test/x") == first
    assert replay.fetch("https://a.test/missing") is None
    assert replay.fetch("https://a.test/never-seen") is None
    assert live.calls == 2
