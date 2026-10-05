#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "openai>=2.0",
#     "pytest>=8.0",
# ]
# ///
"""OpenAI SDK compatibility tests for Azure Chat Completions translation.

Boots Praxis with the azure/chat-completions-to-openai example in front of
a local fake Azure backend. Asserts the rewritten deployment path,
api-version, stripped request body, and that the official OpenAI Python
SDK can complete non-streaming, streaming, and error calls.

Usage:
    cargo build -p praxis-ai-proxy
    uv run tests/integration/sdk/openai/test_azure_chat_completions.py -v
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from openai import NotFoundError, OpenAI

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = REPO_ROOT / "examples/configs/azure/chat-completions-to-openai.yaml"

NON_STREAM_RESPONSE = {
    "id": "chatcmpl-abc",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Paris"},
            "finish_reason": "stop",
            "content_filter_results": {"hate": {"filtered": False, "severity": "safe"}},
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    "prompt_filter_results": [{"prompt_index": 0, "content_filter_results": {}}],
}

STREAM_BODY = (
    'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1,'
    '"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},'
    '"finish_reason":null,"content_filter_results":{}}]}\n\n'
    'data: {"id":"","object":"","created":0,"model":"","choices":[{"index":0,'
    '"finish_reason":null,"content_filter_results":{"hate":{"filtered":false}},'
    '"content_filter_offsets":{"check_offset":2,"start_offset":0,"end_offset":2}}]}\n\n'
    'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1,'
    '"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)

AZURE_ERROR = {
    "error": {
        "message": "The API deployment for this resource does not exist.",
        "type": None,
        "code": "DeploymentNotFound",
        "innererror": {"code": "DeploymentNotFound"},
    }
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _find_binary() -> str:
    configured = os.environ.get("PRAXIS_AI_BIN")
    if configured:
        if os.path.isfile(configured):
            return configured
        raise FileNotFoundError(f"PRAXIS_AI_BIN={configured!r} not found")
    for candidate in ("target/debug/praxis-ai", "target/release/praxis-ai"):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("praxis-ai binary not found — run `cargo build -p praxis-ai-proxy` first")


def _wait_for_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"port {port} did not accept connections within {timeout}s")


class _FakeAzure:
    """Local Azure stand-in: scripts POST responses and records forwarded requests."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.posts: list[tuple[str, bytes]] = []
        self.scripts: deque[tuple[int, str, bytes]] = deque()

    def reset(self) -> None:
        with self.lock:
            self.posts.clear()
            self.scripts.clear()

    def script(self, status: int, content_type: str, body: str | bytes) -> None:
        payload = body if isinstance(body, bytes) else body.encode()
        with self.lock:
            self.scripts.append((status, content_type, payload))

    def take_script(self) -> tuple[int, str, bytes]:
        with self.lock:
            if self.scripts:
                return self.scripts.popleft()
        return (500, "text/plain", b"exhausted")

    def record_post(self, path: str, body: bytes) -> None:
        with self.lock:
            self.posts.append((path, body))


_STATE = _FakeAzure()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args: Any) -> None:  # silence access logs
        pass

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        _STATE.record_post(self.path, body)
        status, content_type, payload = _STATE.take_script()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)


def _patched_config(listener_port: int, backend_port: int) -> str:
    text = CONFIG_PATH.read_text()
    text = text.replace("127.0.0.1:8080", f"127.0.0.1:{listener_port}")
    text = text.replace("my-resource.openai.azure.com:443", f"127.0.0.1:{backend_port}")
    return text


@pytest.fixture(scope="session")
def praxis_proxy():
    backend_port = _free_port()
    listener_port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", backend_port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    fd, config_path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as handle:
        handle.write(_patched_config(listener_port, backend_port))

    proc = subprocess.Popen(
        [_find_binary(), "-c", config_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_port(listener_port)
        yield listener_port
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        server.shutdown()
        os.unlink(config_path)


@pytest.fixture(scope="session")
def openai_client(praxis_proxy: int) -> OpenAI:
    return OpenAI(
        api_key="not-needed",
        base_url=f"http://127.0.0.1:{praxis_proxy}/v1",
        max_retries=0,
        timeout=10.0,
    )


@pytest.fixture(autouse=True)
def _reset_fake_azure() -> None:
    _STATE.reset()


def _last_post() -> tuple[str, dict[str, Any]]:
    assert _STATE.posts, "fake Azure received no POST"
    path, raw = _STATE.posts[-1]
    return path, json.loads(raw)


def _assert_azure_upstream(
    path: str,
    body: dict[str, Any],
    *,
    messages: list[dict[str, Any]],
    extras: dict[str, Any] | None = None,
) -> None:
    parsed = urlparse(path)
    assert parsed.path == "/openai/deployments/gpt-4o/chat/completions", path
    query = parse_qs(parsed.query, keep_blank_values=True)
    assert query.get("api-version") == ["2024-10-21"], query
    assert "model" not in body, body
    assert body["messages"] == messages, body
    for key, value in (extras or {}).items():
        assert body[key] == value, (key, body)


class TestAzureChatCompletionsSdk:
    def test_non_streaming(self, openai_client: OpenAI) -> None:
        _STATE.script(200, "application/json", json.dumps(NON_STREAM_RESPONSE))
        messages = [{"role": "user", "content": "What is the capital of France?"}]

        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2,
            user="sdk-test",
        )

        assert completion.choices[0].message.content == "Paris"
        assert completion.choices[0].finish_reason == "stop"
        path, body = _last_post()
        _assert_azure_upstream(
            path,
            body,
            messages=messages,
            extras={"temperature": 0.2, "user": "sdk-test"},
        )

    def test_streaming(self, openai_client: OpenAI) -> None:
        _STATE.script(200, "text/event-stream", STREAM_BODY)
        messages = [{"role": "user", "content": "hi"}]

        stream = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True,
        )
        parts: list[str] = []
        finish = None
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is not None and delta.content:
                parts.append(delta.content)
            if chunk.choices[0].finish_reason:
                finish = chunk.choices[0].finish_reason

        assert "".join(parts) == "Hi"
        assert finish == "stop"
        path, body = _last_post()
        _assert_azure_upstream(path, body, messages=messages, extras={"stream": True})

    def test_streaming_annotation_with_usage_is_preserved(self, openai_client: OpenAI) -> None:
        content_frame = {
            "id": "chatcmpl-abc",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": "Hi"},
                "finish_reason": "stop",
            }],
        }
        usage_frame = {
            "id": "chatcmpl-abc",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{"index": 0, "finish_reason": None, "content_filter_results": {}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }
        body = (
            "data: " + json.dumps(content_frame) + "\n\n"
            + "data: " + json.dumps(usage_frame) + "\n\n"
            + "data: [DONE]\n\n"
        )
        _STATE.script(200, "text/event-stream", body)

        chunks = list(
            openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
                stream_options={"include_usage": True},
            )
        )

        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "Hi"
        assert chunks[1].choices == []
        assert chunks[1].usage is not None
        assert chunks[1].usage.total_tokens == 3

        _STATE.script(200, "text/event-stream", body)
        with openai_client.chat.completions.stream(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            stream_options={"include_usage": True},
        ) as stream:
            list(stream)
            completion = stream.get_final_completion()

        assert completion.choices[0].message.content == "Hi"
        assert completion.usage is not None
        assert completion.usage.total_tokens == 3

        metadata_frame = {
            **usage_frame,
            "choices": [{**usage_frame["choices"][0], "logprobs": None}],
            "usage": None,
            "provider_data": {"source": "azure"},
            "service_tier": "default",
        }
        metadata_body = (
            "data: " + json.dumps(content_frame) + "\n\n"
            + "data: " + json.dumps(metadata_frame) + "\n\n"
            + "data: [DONE]\n\n"
        )
        _STATE.script(200, "text/event-stream", metadata_body)
        with openai_client.chat.completions.stream(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        ) as stream:
            events = list(stream)
            completion = stream.get_final_completion()

        assert completion.choices[0].message.content == "Hi"
        metadata_chunk = next(
            event.chunk
            for event in events
            if event.type == "chunk" and event.chunk.choices == []
        )
        assert metadata_chunk.service_tier == "default"
        assert metadata_chunk.model_extra["provider_data"] == {"source": "azure"}

    def test_error(self, openai_client: OpenAI) -> None:
        _STATE.script(404, "application/json", json.dumps(AZURE_ERROR))
        messages = [{"role": "user", "content": "hi"}]

        with pytest.raises(NotFoundError) as exc_info:
            openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )

        assert exc_info.value.status_code == 404, "missing deployment must return 404"
        assert "does not exist" in str(exc_info.value), "upstream deployment-not-found text must reach the client"
        path, body = _last_post()
        _assert_azure_upstream(path, body, messages=messages)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
