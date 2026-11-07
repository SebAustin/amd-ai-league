import json

import pytest
from PIL import Image

from amdleague.m2 import cli


class LLM:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, messages, max_new_tokens, timeout):
        return json.dumps(self.reply)


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("M2_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("M2_SCRATCH_DIR", str(tmp_path / "scratch"))


def _out(tmp_path, stem):
    return json.loads((tmp_path / "output" / f"{stem}_output.json").read_text())


def test_writes_text_and_confidence_named_after_the_input(tmp_path, monkeypatch):
    Image.new("RGB", (300, 150), "white").save(tmp_path / "image_01.png")
    monkeypatch.setattr(cli, "_llm", lambda: LLM({"kind": "plate", "lines": ["CALIFORNIA",
                                                                            "7ABC123"]}))
    assert cli.main(["--input-image", str(tmp_path / "image_01.png")]) == 0
    data = _out(tmp_path, "image_01")
    assert data["text"] == "7ABC123"
    assert 0.0 <= data["confidence"] <= 1.0


def test_tiff_with_dots_in_its_name(tmp_path, monkeypatch):
    Image.new("RGB", (300, 150), "white").save(tmp_path / "sign.v2.tiff")
    monkeypatch.setattr(cli, "_llm", lambda: LLM({"kind": "sign", "lines": ["STOP"]}))
    cli.main(["--input-image", str(tmp_path / "sign.v2.tiff")])
    assert _out(tmp_path, "sign.v2")["text"] == "STOP"


def test_unreadable_image_still_writes_an_empty_answer(tmp_path, monkeypatch):
    (tmp_path / "broken.jpg").write_bytes(b"\xff\xd8 not really")
    monkeypatch.setattr(cli, "_llm", lambda: LLM({"kind": "sign", "lines": ["STOP"]}))
    assert cli.main(["--input-image", str(tmp_path / "broken.jpg")]) == 0
    assert _out(tmp_path, "broken") == {"text": "", "confidence": 0.0}


def test_model_disabled_writes_an_empty_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("M2_DISABLE_LLM", "1")
    Image.new("RGB", (300, 150), "white").save(tmp_path / "x.png")
    assert cli.main(["--input-image", str(tmp_path / "x.png")]) == 0
    assert _out(tmp_path, "x")["text"] == ""


def test_crash_inside_reading_still_leaves_valid_json(tmp_path, monkeypatch):
    Image.new("RGB", (300, 150), "white").save(tmp_path / "x.png")

    def boom():
        raise RuntimeError("socket gone")

    monkeypatch.setattr(cli, "_llm", boom)
    assert cli.main(["--input-image", str(tmp_path / "x.png")]) == 0
    assert _out(tmp_path, "x")["text"] == ""


def test_alternative_spellings_and_missing_argument(tmp_path, monkeypatch):
    monkeypatch.setenv("M2_DISABLE_LLM", "1")
    Image.new("RGB", (300, 150), "white").save(tmp_path / "y.png")
    cli.main(["--image", str(tmp_path / "y.png")])
    cli.main([str(tmp_path / "y.png"), "--unknown-flag"])
    assert _out(tmp_path, "y")["text"] == ""
    assert cli.main([]) == 2
