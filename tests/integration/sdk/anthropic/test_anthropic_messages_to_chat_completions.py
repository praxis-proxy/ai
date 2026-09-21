#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "anthropic>=0.40",
#     "pytest>=8.0",
# ]
# ///
"""
Anthropic SDK tests for request-field handling in the Messages to Chat
Completions translation.

Starts Praxis with the shipped `messages-to-openai` example, retargeted at a
local stub backend that records the translated request, and verifies through
the official Anthropic Python SDK that unmapped fields reach the backend, that
`metadata.user_id` becomes a hashed `safety_identifier`, that `thinking` is dropped,
and that fields the translation cannot honor are rejected before any backend
call.

Usage:
    cargo build -p praxis-ai-proxy
    uv run tests/integration/sdk/anthropic/test_anthropic_messages_to_chat_completions.py -s -v
"""

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from anthropic import Anthropic, BadRequestError

CONFIG_PATH = "examples/configs/anthropic/messages-to-openai.yaml"
MODEL = "stub-model"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _find_binary() -> str:
    configured = os.environ.get("PRAXIS_AI_BIN")
    if configured:
        if os.path.isfile(configured):
            return configured
        raise FileNotFoundError(f"PRAXIS_AI_BIN={configured!r} not found")
    for candidate in ["target/debug/praxis-ai", "target/release/praxis-ai"]:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "praxis-ai binary not found — run `cargo build -p praxis-ai-proxy` first"
    )


def _wait_for_proxy(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"proxy did not start within {timeout}s on port {port}")


class RecordingBackend(BaseHTTPRequestHandler):
    """Chat Completions stub that keeps the last request body it received."""

    bodies: list[dict] = []

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        RecordingBackend.bodies.append(json.loads(self.rfile.read(length)))
        reply = json.dumps(
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": 1_700_000_000,
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "4"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13},
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, format, *args):
        pass


def _write_config(proxy_port: int, backend_port: int) -> str:
    with open(CONFIG_PATH) as f:
        config = f.read()
    for old, new in [
        ("127.0.0.1:8080", f"127.0.0.1:{proxy_port}"),
        ("127.0.0.1:8000", f"127.0.0.1:{backend_port}"),
    ]:
        replaced = config.replace(old, new)
        assert replaced != config, f"example drift: {old} not found in {CONFIG_PATH}"
        config = replaced
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(config)
    return path


@pytest.fixture(scope="module")
def anthropic_client():
    backend_port = _free_port()
    backend = HTTPServer(("127.0.0.1", backend_port), RecordingBackend)
    threading.Thread(target=backend.serve_forever, daemon=True).start()

    proxy_port = _free_port()
    config_path = _write_config(proxy_port, backend_port)
    proc = subprocess.Popen(
        [_find_binary(), "-c", config_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_proxy(proxy_port)
        yield Anthropic(
            base_url=f"http://127.0.0.1:{proxy_port}",
            api_key="not-needed",
            max_retries=0,
            timeout=10.0,
        )
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        backend.shutdown()
        os.unlink(config_path)


class TestRequestFieldHandling:
    def test_unmapped_fields_reach_the_backend(self, anthropic_client):
        RecordingBackend.bodies.clear()

        response = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=64,
            metadata={"user_id": "user-1"},
            thinking={"type": "enabled", "budget_tokens": 1024},
            extra_body={"top_k": 40},
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

        assert response.content[0].text == "4"
        [upstream] = RecordingBackend.bodies
        assert upstream["top_k"] == 40
        assert upstream["safety_identifier"] == hashlib.sha256(b"user-1").hexdigest()
        assert "metadata" not in upstream
        assert "thinking" not in upstream

    def test_unrepresentable_field_is_rejected_before_the_backend(self, anthropic_client):
        RecordingBackend.bodies.clear()

        with pytest.raises(BadRequestError) as excinfo:
            anthropic_client.messages.create(
                model=MODEL,
                max_tokens=64,
                service_tier="standard_only",
                messages=[{"role": "user", "content": "What is 2+2?"}],
            )

        assert "`service_tier` is not supported" in str(excinfo.value)
        assert RecordingBackend.bodies == []

    def test_chat_completions_field_whose_output_is_discarded_is_rejected(
        self, anthropic_client
    ):
        RecordingBackend.bodies.clear()

        with pytest.raises(BadRequestError) as excinfo:
            anthropic_client.messages.create(
                model=MODEL,
                max_tokens=64,
                extra_body={"moderation": {"input": True, "output": True}},
                messages=[{"role": "user", "content": "What is 2+2?"}],
            )

        assert "`moderation` is not supported" in str(excinfo.value)
        assert RecordingBackend.bodies == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
