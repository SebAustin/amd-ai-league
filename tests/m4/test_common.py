import json
import socket
import tempfile
import threading
from pathlib import Path

from amdleague.common.deadline import Deadline
from amdleague.common.model_client import ModelClient
from amdleague.common.output import write_json


def test_deadline_counts_down():
    d = Deadline(5)
    assert 4 < d.remaining() <= 5
    assert not d.expired()
    assert Deadline(0).expired()


def test_child_deadline_is_capped_by_parent():
    assert Deadline(1).child(10).remaining() <= 1


def test_write_json_is_atomic_and_overwrites(tmp_path):
    path = tmp_path / "x.json"
    write_json(path, {"a": 1})
    write_json(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 2}
    assert list(tmp_path.iterdir()) == [path]


def test_model_client_returns_none_without_server(tmp_path):
    client = ModelClient(tmp_path / "missing.sock", autostart=False)
    assert client.complete([{"role": "user", "content": "hi"}], 8, timeout=0.5) is None


def test_model_client_round_trip():
    # Unix socket paths are capped at ~104 chars on macOS; pytest tmp_path can exceed it.
    sock_path = Path(tempfile.mkdtemp(prefix="m4", dir="/tmp")) / "m.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(1)

    def serve():
        conn, _ = server.accept()
        with conn:
            line = conn.makefile("rb").readline()
            req = json.loads(line)
            reply = {"text": "echo:" + req["messages"][0]["content"]}
            conn.sendall((json.dumps(reply) + "\n").encode())

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    client = ModelClient(sock_path, autostart=False)
    assert client.complete([{"role": "user", "content": "hi"}], 8, timeout=2) == "echo:hi"
    server.close()


def test_vision_model_detection_and_message_parts(tmp_path):
    from amdleague.common.model_server import as_parts, is_vision_model

    assert not is_vision_model(tmp_path)
    (tmp_path / "preprocessor_config.json").write_text("{}")
    assert is_vision_model(tmp_path)
    parts = as_parts([{"role": "system", "content": "hi"},
                      {"role": "user", "content": [{"type": "image", "path": "/x.png"}]}])
    assert parts[0]["content"] == [{"type": "text", "text": "hi"}]
    assert parts[1]["content"][0]["type"] == "image"


def test_wait_ready_gives_up_when_loading_failed(tmp_path):
    from amdleague.common.model_client import ModelClient

    sock = tmp_path / "m.sock"
    sock.with_suffix(".failed").write_text("load failed")
    assert ModelClient(sock).wait_ready(5) is False
    assert ModelClient(sock, autostart=False).wait_ready(0) is False
