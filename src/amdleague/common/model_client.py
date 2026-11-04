"""Thin client for the resident GPU model server (see ``model_server``).

Every failure returns ``None`` so callers can fall back to rule-based extraction.

Files next to the socket coordinate start-up between processes:
``.starting`` (a server is loading; holds its PID), ``.failed`` (loading failed: do not wait).
"""

from __future__ import annotations

import errno
import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_SOCKET = Path(os.environ.get("M4_MODEL_SOCKET", "/tmp/amdleague-model.sock"))
SERVER_LOG = Path(os.environ.get("M4_MODEL_LOG", "/tmp/model_server.log"))
STALE_LOCK_SECONDS = 600.0  # longer than any plausible model load
_POLL_SECONDS = 0.25


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class ModelClient:
    def __init__(self, socket_path: Path = DEFAULT_SOCKET, autostart: bool = True) -> None:
        self.socket_path = Path(socket_path)
        self.autostart = autostart
        self.lock = self.socket_path.with_suffix(".starting")
        self.failed = self.socket_path.with_suffix(".failed")

    def _lock_is_live(self) -> bool:
        """A start lock counts only while its owner runs and it is not absurdly old."""
        try:
            age = time.time() - self.lock.stat().st_mtime
            content = self.lock.read_text().strip()
        except OSError:
            return False
        if age > STALE_LOCK_SECONDS:
            return False
        return _pid_alive(int(content)) if content.isdigit() else True

    def ensure_started(self) -> None:
        """Spawn the model server in the background if nothing is listening or loading."""
        if not self.autostart or self.socket_path.exists() or self.failed.exists():
            return
        if self.lock.exists() and not self._lock_is_live():
            log.warning("removing stale model-server lock %s", self.lock)
            self.lock.unlink(missing_ok=True)
        try:
            fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return  # another process is already starting it
        log.info("starting model server for %s", self.socket_path)
        with SERVER_LOG.open("ab") as log_file:  # never inherit the harness's stdout/stderr
            proc = subprocess.Popen(
                [sys.executable, "-m", "amdleague.common.model_server"],
                env={**os.environ, "M4_MODEL_SOCKET": str(self.socket_path)},
                stdin=subprocess.DEVNULL, stdout=log_file, stderr=log_file,
                start_new_session=True,
            )
        os.write(fd, str(proc.pid).encode())
        os.close(fd)

    def _wait_for_socket(self, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while not self.socket_path.exists():
            if self.failed.exists() or not self._lock_is_live() or time.monotonic() >= end:
                return False  # failed, nobody loading, or out of time
            time.sleep(_POLL_SECONDS)
        return True

    def complete(
        self, messages: list[dict[str, str]], max_new_tokens: int, timeout: float
    ) -> str | None:
        start = time.monotonic()
        self.ensure_started()
        if timeout <= 0 or not self._wait_for_socket(timeout):
            log.warning("model server not available within %.1fs", timeout)
            return None
        left = max(0.1, timeout - (time.monotonic() - start))
        request: dict[str, Any] = {"messages": messages, "max_new_tokens": max_new_tokens}
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                conn.settimeout(left)
                conn.connect(str(self.socket_path))
                conn.sendall((json.dumps(request) + "\n").encode("utf-8"))
                line = conn.makefile("rb").readline()
            reply = json.loads(line)
        except ConnectionRefusedError:
            log.warning("model server socket is stale; removing it so it can restart")
            self.socket_path.unlink(missing_ok=True)
            return None
        except (OSError, ValueError) as exc:
            if getattr(exc, "errno", None) == errno.ENOENT:
                self.socket_path.unlink(missing_ok=True)
            log.warning("model call failed: %s", exc)
            return None
        text = reply.get("text") if isinstance(reply, dict) else None
        return text if isinstance(text, str) else None
