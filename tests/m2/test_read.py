import json

from amdleague.common.deadline import Deadline
from amdleague.m2.read import compose, parse_reply, read_image


class LLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, max_new_tokens, timeout):
        self.calls.append(messages)
        reply = self.replies.pop(0) if self.replies else None
        return json.dumps(reply, ensure_ascii=False) if isinstance(reply, dict) else reply


def test_plate_keeps_the_number_and_drops_banner_and_slogan():
    assert compose("plate", ["CALIFORNIA", "7ABC123", "dmv.ca.gov"]) == "7ABC123"
    assert compose("plate", ["TEXAS", "ABC-1234", "THE LONE STAR STATE"]) == "ABC-1234"


def test_chinese_plate_keeps_province_character_and_letter():
    assert compose("plate", ["京A·12345"]) == "京A·12345"


def test_plate_split_over_two_lines_with_digits_is_joined():
    assert compose("plate", ["NEW YORK", "JHT", "2951", "EMPIRE STATE"]) == "JHT 2951"


def test_plate_without_any_digit_keeps_all_lines():
    assert compose("plate", ["GO FAST"]) == "GO FAST"


def test_signs_join_lines_top_to_bottom():
    assert compose("sign", ["ROAD", "WORK", "AHEAD"]) == "ROAD WORK AHEAD"
    assert compose("sign", ["SPEED", "LIMIT", "65"]) == "SPEED LIMIT 65"
    assert compose("sign", ["35"]) == "35"


def test_labels_quotes_and_added_units_are_stripped():
    assert compose("sign", ['"STOP"']) == "STOP"
    assert compose("plate", ["Plate: 7ABC123"]) == "7ABC123"
    assert compose("sign", ["Text: SPEED LIMIT 65"]) == "SPEED LIMIT 65"


def test_parse_reply_accepts_json_fences_and_plain_text():
    assert parse_reply('```json\n{"kind": "sign", "lines": ["STOP"]}\n```') == ("sign", ["STOP"])
    assert parse_reply('{"kind": "plate", "text": "7ABC123"}') == ("plate", ["7ABC123"])
    assert parse_reply("ROAD\nWORK\nAHEAD") == ("other", ["ROAD", "WORK", "AHEAD"])
    assert parse_reply(None) is None
    assert parse_reply("   ") is None


def test_two_agreeing_views_give_high_confidence(tmp_path):
    llm = LLM({"kind": "sign", "lines": ["STOP"]}, {"kind": "sign", "lines": ["stop"]})
    result = read_image([tmp_path / "a.png", tmp_path / "b.png"], llm, Deadline(20))
    assert result.text == "STOP" and result.confidence >= 0.9
    assert llm.calls[0][-1]["content"][0] == {"type": "image", "path": str(tmp_path / "a.png")}


def test_disagreeing_views_keep_the_original_with_lower_confidence(tmp_path):
    llm = LLM({"kind": "plate", "lines": ["5XYZ891"]}, {"kind": "plate", "lines": ["5XYZ881"]})
    result = read_image([tmp_path / "a.png", tmp_path / "b.png"], llm, Deadline(20))
    assert result.text == "5XYZ891" and result.confidence < 0.9


def test_empty_original_falls_back_to_the_enhanced_view(tmp_path):
    llm = LLM({"kind": "other", "lines": []}, {"kind": "plate", "lines": ["沪B·88888"]})
    assert read_image([tmp_path / "a", tmp_path / "b"], llm, Deadline(20)).text == "沪B·88888"


def test_no_time_left_skips_the_second_view(tmp_path):
    llm = LLM({"kind": "sign", "lines": ["STOP"]})
    result = read_image([tmp_path / "a", tmp_path / "b"], llm, Deadline(2.5))
    assert result.text == "STOP" and len(llm.calls) == 1


def test_no_model_or_no_views_reads_nothing(tmp_path):
    assert read_image([tmp_path / "a"], None, Deadline(20)).text == ""
    assert read_image([], LLM(), Deadline(20)).text == ""
    assert read_image([tmp_path / "a"], LLM(None), Deadline(20)).text == ""


def test_short_state_banner_beside_the_number_is_still_dropped():
    assert compose("plate", ["OHIO", "ABC 1234"]) == "ABC 1234"
