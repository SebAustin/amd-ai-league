"""Resident model server: loads the LLM once on the AMD GPU and answers over a Unix socket.

Protocol: one JSON line in ``{"messages": [...], "max_new_tokens": n}``,
one JSON line out ``{"text": "..."}``. Started by the container entrypoint, or lazily by
``ModelClient.ensure_started``. Readiness is signalled by atomically renaming the socket into
place, so clients never connect to a half-initialised server.
"""

from __future__ import annotations

import json
import logging
import os
import socket
from pathlib import Path
from typing import Any

log = logging.getLogger("model_server")

MODEL_DIR = Path(os.environ.get("M4_MODEL_DIR", "/models/llm"))
SOCKET_PATH = Path(os.environ.get("M4_MODEL_SOCKET", "/tmp/amdleague-model.sock"))
MAX_NEW_TOKENS_CAP = 256
CONN_TIMEOUT_SECONDS = 60.0


def _load():  # pragma: no cover - needs a GPU
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("no GPU visible to torch; refusing to serve on CPU")
    log.info("torch %s hip=%s device=%s", torch.__version__, torch.version.hip,
             torch.cuda.get_device_name(0))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    return torch, tokenizer, model


def _generate(torch, tokenizer, model, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=min(max_new_tokens, MAX_NEW_TOKENS_CAP),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _handle(conn: socket.socket, engine: tuple[Any, Any, Any], generate=None) -> None:
    """Serve one request. Never raises: a client that hung up must not kill the server."""
    generate = generate or _generate
    with conn:
        try:
            conn.settimeout(CONN_TIMEOUT_SECONDS)
            line = conn.makefile("rb").readline()
            if not line:
                return
            try:
                req = json.loads(line)
                text = generate(*engine, req["messages"], int(req.get("max_new_tokens", 64)))
                reply: dict[str, Any] = {"text": text}
            except Exception as exc:
                log.exception("generation failed")
                reply = {"error": str(exc)}
            conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))
        except OSError as exc:
            log.warning("client connection dropped: %s", exc)


def main() -> None:  # pragma: no cover - needs a GPU
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    try:
        engine = _load()
        _generate(*engine, [{"role": "user", "content": "Say OK."}], 4)  # warm-up kernels
    except Exception:
        # Mark the failure so clients stop waiting and stop respawning; they use rules instead.
        SOCKET_PATH.with_suffix(".failed").write_text("load failed\n")
        SOCKET_PATH.with_suffix(".starting").unlink(missing_ok=True)
        log.exception("model failed to load")
        raise
    staging = SOCKET_PATH.with_suffix(".staging")
    staging.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(staging))
    server.listen(8)
    os.replace(staging, SOCKET_PATH)  # readiness signal
    SOCKET_PATH.with_suffix(".starting").unlink(missing_ok=True)
    log.info("serving on %s", SOCKET_PATH)
    while True:
        try:
            conn, _ = server.accept()
        except OSError as exc:
            log.warning("accept failed: %s", exc)
            continue
        _handle(conn, engine)


if __name__ == "__main__":  # pragma: no cover
    main()
