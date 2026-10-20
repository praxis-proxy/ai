#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=2.0,<4", "pytest>=8.0"]
# ///
"""Official OpenAI-client coverage for selected Anthropic Messages providers."""

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

import pytest
from openai import APIStatusError, OpenAI

ROOT = Path(__file__).resolve().parents[4]
OVERLAY = ROOT / "tests/integration/fixtures/inference/routing/openai-to-anthropic.json"
MODEL = "fixture-openai-model"


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _binary() -> str:
    configured = os.environ.get("PRAXIS_AI_BIN")
    if configured:
        return configured
    for candidate in ("target/debug/praxis-ai", "target/release/praxis-ai"):
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("build praxis-ai-proxy or set PRAXIS_AI_BIN")


class _BackendState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.scripted: deque[tuple[int, str, bytes]] = deque()
        self.requests: list[tuple[str, dict]] = []

    def script(self, status: int, content_type: str, body: dict | str) -> None:
        encoded = json.dumps(body).encode() if isinstance(body, dict) else body.encode()
        with self.lock:
            self.scripted.append((status, content_type, encoded))

    def take(self, path: str, body: bytes) -> tuple[int, str, bytes]:
        with self.lock:
            self.requests.append((path, json.loads(body)))
            return self.scripted.popleft()


STATE = _BackendState()


@pytest.fixture(autouse=True)
def reset_backend_state():
    with STATE.lock:
        STATE.scripted.clear()
        STATE.requests.clear()


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers["content-length"])
        status, content_type, body = STATE.take(self.path, self.rfile.read(length))
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


def _message(content: list[dict], usage: dict, stop_reason: str = "end_turn") -> dict:
    return {
        "type": "message",
        "id": "msg_sdk",
        "role": "assistant",
        "model": "claude-fixture-model",
        "content": content,
        "stop_reason": stop_reason,
        "usage": usage,
    }


def _config(proxy_port: int, backend_port: int, api_translation_enabled: bool = True) -> dict:
    return {
        "listeners": [{"name": "proxy", "address": f"127.0.0.1:{proxy_port}", "filter_chains": ["main"]}],
        "filter_chains": [{
            "name": "main",
            "filters": [
                {"filter": "model_to_header", "header": "X-Model"},
                {"filter": "intelligent_route", "overlay_file": str(OVERLAY), "model_header": "X-Model",
                 "clusters": ["anthropic-backend"], "reload": {"enabled": False}},
                {"filter": "openai_chat_completions_to_anthropic_stream", "max_partial_event_bytes": 1048576,
                 "max_tool_blocks": 256},
                {"filter": "openai_chat_completions_to_anthropic", "api_translation_enabled": api_translation_enabled,
                 "max_body_bytes": 1048576},
                {"filter": "credential_inject", "credentials": [{
                    "name": "anthropic-fixture-secret", "namespace": "grid-fixture", "key": "api-key",
                    "strategy": "apikey", "header": "x-api-key", "value": "fixture-only-dummy-token"}]},
                {"filter": "load_balancer", "clusters": [{
                    "name": "anthropic-backend", "endpoints": [f"127.0.0.1:{backend_port}"]}]},
            ],
        }],
        "admin": {},
        "shutdown_timeout_secs": 5,
        "insecure_options": {"allow_private_endpoints": True},
    }


@pytest.fixture(scope="module")
def client():
    proxy_port, backend_port = _port(), _port()
    backend = ThreadingHTTPServer(("127.0.0.1", backend_port), _Handler)
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    with tempfile.TemporaryDirectory() as directory:
        config = Path(directory) / "praxis.yaml"
        config.write_text(json.dumps(_config(proxy_port, backend_port)))
        proc = subprocess.Popen([_binary(), "-c", str(config)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection(("127.0.0.1", proxy_port), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                raise TimeoutError("Praxis did not start")
            with OpenAI(api_key="caller-only", base_url=f"http://127.0.0.1:{proxy_port}/v1",
                        max_retries=0, timeout=10) as sdk:
                yield sdk
        finally:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    backend.shutdown()
    thread.join(timeout=5)


@pytest.fixture
def translation_disabled_client():
    """OpenAI client for verifying the disabled server-side translation policy."""
    proxy_port, backend_port = _port(), _port()
    with tempfile.TemporaryDirectory() as directory:
        config = Path(directory) / "praxis.yaml"
        config.write_text(json.dumps(_config(proxy_port, backend_port, api_translation_enabled=False)))
        proc = subprocess.Popen([_binary(), "-c", str(config)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection(("127.0.0.1", proxy_port), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                raise TimeoutError("Praxis did not start")
            with OpenAI(api_key="caller-only", base_url=f"http://127.0.0.1:{proxy_port}/v1",
                        max_retries=0, timeout=10) as sdk:
                yield sdk
        finally:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def test_json_and_cached_usage(client: OpenAI) -> None:
    STATE.script(200, "application/json", _message(
        [{"type": "text", "text": "hello"}],
        {"input_tokens": 10, "cache_read_input_tokens": 1000,
         "cache_creation_input_tokens": 50, "output_tokens": 4},
    ))
    response = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=16)
    assert response.choices[0].message.content == "hello", "Anthropic text must reach the OpenAI message"
    assert response.usage.prompt_tokens == 1060, "cache tokens must count toward prompt usage"
    assert response.usage.completion_tokens == 4, "Anthropic output tokens must become completion tokens"
    assert response.usage.total_tokens == 1064, "total tokens must include cache and output tokens"
    path, upstream = STATE.requests[-1]
    assert path == "/v1/messages", "selected Anthropic requests must use the Messages path"
    assert upstream["model"] == "claude-fixture-model", "selected target model must reach Anthropic"


def test_null_cache_usage(client: OpenAI) -> None:
    STATE.script(200, "application/json", _message(
        [{"type": "text", "text": "hello"}],
        {"input_tokens": 10, "cache_read_input_tokens": None,
         "cache_creation_input_tokens": None, "output_tokens": 4},
    ))
    response = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=16)
    assert response.choices[0].message.content == "hello", "nullable Anthropic usage must not discard translated text"
    assert response.usage.prompt_tokens == 10, "null cache counters must add no prompt tokens"
    assert response.usage.total_tokens == 14, "null cache counters must add no total tokens"


def test_streaming_and_terminal_usage(client: OpenAI) -> None:
    events = (
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_sdk",'
        '"usage":{"input_tokens":10,"cache_read_input_tokens":1000,'
        '"cache_creation_input_tokens":50}}}\n\n'
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"text","text":""}}\n\n'
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":"hello"}}\n\n'
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        '"usage":{"output_tokens":4}}\n\n'
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )
    STATE.script(200, "text/event-stream", events)
    chunks = list(client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "hi"}], stream=True,
        stream_options={"include_usage": True}, max_tokens=16,
    ))
    assert any(chunk.choices and chunk.choices[0].delta.content == "hello" for chunk in chunks), "Anthropic SSE text must reach OpenAI chunks"
    assert chunks[-1].usage.prompt_tokens == 1060, "SSE cache tokens must count toward prompt usage"
    assert chunks[-1].usage.total_tokens == 1064, "SSE total usage must include cache and output tokens"


def test_tool_call_and_result_round_trip(client: OpenAI) -> None:
    STATE.script(200, "application/json", _message(
        [{"type": "tool_use", "id": "toolu_1", "name": "weather", "input": {"city": "Oslo"}}],
        {"input_tokens": 3, "output_tokens": 2}, "tool_use",
    ))
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]
    first = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "weather?"}], tools=tools, max_tokens=16,
    )
    call = first.choices[0].message.tool_calls[0]
    assert call.function.name == "weather", "Anthropic tool use must become an OpenAI function call"
    assert json.loads(call.function.arguments) == {"city": "Oslo"}, "tool arguments must retain the provider input"
    STATE.script(200, "application/json", _message(
        [{"type": "text", "text": "sunny"}], {"input_tokens": 4, "output_tokens": 2},
    ))
    second = client.chat.completions.create(model=MODEL, messages=[
        {"role": "user", "content": "weather?"},
        first.choices[0].message.model_dump(exclude_none=True),
        {"role": "tool", "tool_call_id": call.id, "content": "sunny"},
    ], tools=tools, max_tokens=16)
    assert second.choices[0].message.content == "sunny", "tool-result round trip must return translated text"
    assert STATE.requests[-1][1]["messages"][-1]["content"][0]["type"] == "tool_result", "OpenAI tool results must become Anthropic tool_result blocks"


@pytest.mark.parametrize("status", [429, 529])
def test_provider_error_preserves_status(client: OpenAI, status: int) -> None:
    STATE.script(status, "application/json", {"type": "error", "error": {"message": "busy"}})
    with pytest.raises(APIStatusError) as caught:
        client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=16)
    assert caught.value.status_code == status, "provider HTTP status must survive translation"
    body = caught.value.response.json()
    assert body["error"]["code"] == status, "OpenAI error code must retain provider status"
    assert body["error"]["type"] == ("rate_limit_error" if status == 429 else "server_error"), "OpenAI error type must match provider status"


def test_disabled_translation_names_platform_administrator(translation_disabled_client: OpenAI) -> None:
    with pytest.raises(APIStatusError) as caught:
        translation_disabled_client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=16,
        )

    assert caught.value.status_code == 503
    error = caught.value.response.json()["error"]
    assert error["code"] == "api_translation_disabled"
    assert "platform administrator has forbidden translation" in error["message"]
    assert not STATE.requests, "disabled translation must reject before contacting the provider"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
