#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "anthropic>=0.40",
#     "pytest>=8.0",
# ]
# ///
"""
Anthropic Messages hosted web-search integration tests through the official SDK.

Drives the shipped `anthropic/full-flow-agentic.yaml` example — the server-owned
web-search loop over Praxis core's iterative_request_router — with fully local
stub backends (a native Anthropic Messages model and a Tavily-shaped search
provider). The single example config has no streaming opt-in: the same pipeline
serves BOTH transports, selected per request from the client's `stream` flag.

    * `stream=False` -> the loop runs buffered and the SDK receives one final
      Anthropic Messages JSON object.
    * `stream=True`  -> the loop streams the terminal answer as one coherent
      client-visible Messages SSE lifecycle; the managed `WebSearch` tool-use
      block stays internal and never reaches the client.

The model stub branches purely on the request `stream` flag, so both transports
run against the exact same example config with no per-mode knob.

Usage:
    cargo build -p praxis-ai-proxy
    uv run tests/integration/sdk/anthropic/test_anthropic_web_search.py -s -v

Environment variables:
    PRAXIS_AI_BIN  path to praxis-ai binary (auto-detected if unset)
"""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from anthropic import Anthropic, AuthenticationError

CONFIG_PATH = "examples/configs/anthropic/full-flow-agentic.yaml"
SCOPED_CONFIG_PATH = "examples/configs/anthropic/web-search-scoped-credentials.yaml"
TOOL_USE_ID = "toolu_web_search_01"
FINAL_TEXT = "Potato is a starchy tuber native to the Americas."
LARGE_ASSISTANT_TEXT = "Searching first. " + "x" * (256 * 1024)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _find_binary() -> str:
    override = os.environ.get("PRAXIS_AI_BIN")
    if override:
        if os.path.isfile(override):
            return override
        raise FileNotFoundError(f"PRAXIS_AI_BIN={override!r} not found")
    for candidate in ["target/debug/praxis-ai", "target/release/praxis-ai"]:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "praxis-ai binary not found — run `cargo build -p praxis-ai-proxy` first"
    )


def _wait_for_proxy(port: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"Praxis did not start within {timeout}s")


def _sse(event_type: str, data: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


def _search_round_sse() -> bytes:
    """One managed-search round as native Anthropic Messages SSE.

    Mirrors the Rust functional fixture: message_start -> WebSearch tool_use
    block -> message_delta(tool_use) -> message_stop. The proxy suppresses the
    tool_use block, so the client never sees these frames.
    """
    body = _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_search_1",
                "type": "message",
                "role": "assistant",
                "model": "openai/gpt-oss-20b",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 20, "output_tokens": 0},
            },
        },
    )
    body += _sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": TOOL_USE_ID,
                "name": "WebSearch",
                "input": {},
            },
        },
    )
    body += _sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"query": "potato"}'},
        },
    )
    body += _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    body += _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 8},
        },
    )
    body += _sse("message_stop", {"type": "message_stop"})
    return body


def _answer_round_sse() -> bytes:
    """The terminal round as native Anthropic Messages SSE."""
    body = _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_answer_1",
                "type": "message",
                "role": "assistant",
                "model": "openai/gpt-oss-20b",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 74, "output_tokens": 0},
            },
        },
    )
    body += _sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )
    body += _sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": FINAL_TEXT},
        },
    )
    body += _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    body += _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 18},
        },
    )
    body += _sse("message_stop", {"type": "message_stop"})
    return body


def _search_round_json(
    large_assistant_content: bool = False, sequence_tool_block: bool = False
) -> bytes:
    content = []
    if large_assistant_content:
        content.append({"type": "text", "text": LARGE_ASSISTANT_TEXT})
    if sequence_tool_block:
        content.append(["tool_use", "WebSearch", TOOL_USE_ID, ["potato"]])
    else:
        content.append(
            {
                "type": "tool_use",
                "id": TOOL_USE_ID,
                "name": "WebSearch",
                "input": {"query": "potato"},
            }
        )
    return json.dumps(
        {
            "id": "msg_search_1",
            "type": "message",
            "role": "assistant",
            "model": "openai/gpt-oss-20b",
            "content": content,
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 20, "output_tokens": 8},
        }
    ).encode()


def _answer_round_json() -> bytes:
    return json.dumps(
        {
            "id": "msg_answer_1",
            "type": "message",
            "role": "assistant",
            "model": "openai/gpt-oss-20b",
            "content": [{"type": "text", "text": FINAL_TEXT}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 74, "output_tokens": 18},
        }
    ).encode()


def _has_tool_result(request: dict) -> bool:
    for message in request.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and block.get("tool_use_id") == TOOL_USE_ID
            ):
                return True
    return False


# ---------------------------------------------------------------------------
# Local stub backends
# ---------------------------------------------------------------------------


class _StubServer:
    """A threaded HTTP server exposing its bound port and captured requests."""

    def __init__(self, handler_cls):
        self.port = _free_port()
        self.requests: list[dict] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), handler_cls)
        self._server.captured = self.requests
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> "_StubServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class _ModelHandler(BaseHTTPRequestHandler):
    """Native Anthropic Messages model: search round, then terminal answer.

    Branches on the request `stream` flag so a single backend serves both the
    buffered (JSON) and streaming (SSE) transports the proxy selects per client.
    """

    protocol_version = "HTTP/1.1"

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            request = {}
        self.server.captured.append(request)

        streaming = request.get("stream") is True
        answering = _has_tool_result(request)
        if streaming:
            body = _answer_round_sse() if answering else _search_round_sse()
            content_type = "text/event-stream"
        else:
            large_assistant_content = any(
                isinstance(message.get("content"), str)
                and "large assistant content" in message["content"]
                for message in request.get("messages", [])
            )
            sequence_tool_block = any(
                isinstance(message.get("content"), str)
                and "sequence-shaped tool block" in message["content"]
                for message in request.get("messages", [])
            )
            body = (
                _answer_round_json()
                if answering
                else _search_round_json(large_assistant_content, sequence_tool_block)
            )
            content_type = "application/json"

        # Close after each response so the proxy never pools a keep-alive
        # connection to this stub across the buffered and streaming test cases;
        # a reused stale socket would drop a later callout before it is captured.
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class _SearchHandler(BaseHTTPRequestHandler):
    """Tavily-shaped header-authenticated search provider returning one web result.

    Tavily authenticates with an ``Authorization: Bearer`` header (issue #1389),
    so the handler captures both that header and the parsed request body as
    ``{"authorization": <str|None>, "body": {...}}`` and returns the Tavily
    response shape (``{"results": [{"title", "url", "content"}]}``).
    """

    protocol_version = "HTTP/1.1"

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        authorization = self.headers.get("Authorization")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {}
        self.server.captured.append({"authorization": authorization, "body": body})
        body = json.dumps(
            {
                "results": [
                    {
                        "title": "Potato - Wikipedia",
                        "url": "https://en.wikipedia.org/wiki/Potato",
                        "content": "The potato is a starchy tuber native to the Americas.",
                    }
                ]
            }
        ).encode()
        # See _ModelHandler: close so no pooled connection is reused across cases.
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def _write_config(proxy_port: int, model_port: int, search_port: int) -> str:
    """Retarget the shipped example to the local stubs.

    Mirrors the Rust functional loader: rewrite the listener and backend
    addresses, point the provider at the local search stub, and add the
    `allow_private_upstreams` opt-in the executor requires for a loopback
    provider callout.
    """
    with open(CONFIG_PATH) as f:
        config = f.read()

    replaced = config.replace("127.0.0.1:8080", f"127.0.0.1:{proxy_port}")
    assert replaced != config, "example drift: listener address not found"
    config = replaced

    replaced = config.replace('endpoints: ["127.0.0.1:8000"]', f'endpoints: ["127.0.0.1:{model_port}"]')
    assert replaced != config, "example drift: model backend endpoint not found"
    config = replaced

    replaced = config.replace(
        "api_key: ${WEB_SEARCH_API_KEY}",
        f"api_key: test-key\n                base_url: http://127.0.0.1:{search_port}",
    )
    assert replaced != config, "example drift: provider api_key not found"
    config = replaced

    replaced = config.replace(
        "allow_private_endpoints: true",
        "allow_private_endpoints: true\n  allow_private_upstreams: true",
    )
    assert replaced != config, "example drift: insecure_options not found"
    config = replaced

    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(config)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def web_search_stack(request):
    """Start the search + model stubs and a proxy on the shipped example."""
    try:
        binary = _find_binary()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    model = _StubServer(_ModelHandler).start()
    search = _StubServer(_SearchHandler).start()
    proxy_port = _free_port()
    config_path = _write_config(proxy_port, model.port, search.port)

    log_fd, log_path = tempfile.mkstemp(suffix=".log")
    log_file = os.fdopen(log_fd, "w")
    started = False
    proc = subprocess.Popen(
        [binary, "-c", config_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        # VLLM_API_KEY is resolved at build time by the chat backend's
        # credential_injection even though this native path never dials that
        # cluster, so it must be present for the proxy to start.
        env={**os.environ, "WEB_SEARCH_API_KEY": "test-key", "VLLM_API_KEY": "test-vllm-key"},
    )
    try:
        _wait_for_proxy(proxy_port)
        started = True
        yield {"proxy_port": proxy_port, "model": model, "search": search}
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        model.stop()
        search.stop()
        if not started or request.session.testsfailed > 0:
            with open(log_path) as f:
                print(f"\n=== Praxis logs ===\n{f.read()}", file=sys.stderr)
        os.unlink(config_path)
        os.unlink(log_path)


@pytest.fixture
def anthropic_client(web_search_stack):
    return Anthropic(
        base_url=f"http://127.0.0.1:{web_search_stack['proxy_port']}",
        api_key="test-anthropic-key",
        max_retries=0,
        timeout=60,
    )


@pytest.fixture(scope="module")
def scoped_credential_client(web_search_stack, request):
    """Run the shipped per-user credential example through the Anthropic SDK."""
    proxy_port = _free_port()
    model_port = web_search_stack["model"].port
    search_port = web_search_stack["search"].port
    with open(SCOPED_CONFIG_PATH) as f:
        config = f.read()
    for old, new in [
        ("127.0.0.1:8080", f"127.0.0.1:{proxy_port}"),
        ('endpoints: ["127.0.0.1:8000"]', f'endpoints: ["127.0.0.1:{model_port}"]'),
        (
            "api_key: ${WEB_SEARCH_API_KEY}",
            f"api_key: test-key\n                base_url: http://127.0.0.1:{search_port}",
        ),
        (
            "allow_private_endpoints: true",
            "allow_private_endpoints: true\n  allow_private_upstreams: true",
        ),
    ]:
        assert old in config, f"scoped example drift: {old} not found"
        config = config.replace(old, new)

    config_fd, config_path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(config_fd, "w") as f:
        f.write(config)
    log_fd, log_path = tempfile.mkstemp(suffix=".log")
    log_file = os.fdopen(log_fd, "w")
    proc = subprocess.Popen(
        [_find_binary(), "-c", config_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env={**os.environ, "WEB_SEARCH_API_KEY": "test-key"},
    )
    try:
        _wait_for_proxy(proxy_port)
        yield Anthropic(
            base_url=f"http://127.0.0.1:{proxy_port}",
            api_key="test-anthropic-key",
            max_retries=0,
            timeout=10,
        )
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        if request.session.testsfailed > 0:
            with open(log_path) as f:
                print(f"\n=== Scoped Praxis logs ===\n{f.read()}", file=sys.stderr)
        os.unlink(config_path)
        os.unlink(log_path)


def _messages_kwargs() -> dict:
    return {
        "model": "openai/gpt-oss-20b",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Use web search to look up potato, then summarize."}
        ],
        "tools": [
            {
                "name": "WebSearch",
                "description": "Search the web",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAnthropicWebSearch:
    """The one example config serves buffered and streaming web search."""

    def test_buffered_web_search_loop(self, anthropic_client, web_search_stack):
        model = web_search_stack["model"]
        search = web_search_stack["search"]
        model.requests.clear()
        search.requests.clear()

        response = anthropic_client.messages.create(**_messages_kwargs())

        # The client receives one final JSON answer; the managed WebSearch
        # tool-use block is resolved server-side and never surfaced.
        assert response.type == "message"
        assert response.stop_reason == "end_turn"
        assert [block.type for block in response.content] == ["text"]
        assert response.content[0].text == FINAL_TEXT
        assert all(block.type != "tool_use" for block in response.content)

        # The loop re-enters the model once with one managed search in between.
        assert len(model.requests) == 2, "buffered loop re-enters the model"
        assert len(search.requests) == 1, "buffered loop dispatches one search"
        assert model.requests[0].get("stream") is not True
        # Issue #1389: Tavily is header-authenticated. The configured key travels
        # in the Authorization bearer header injected at the pinned provider, and
        # never in the request body the outbound chain can read.
        assert search.requests[0]["authorization"] == "Bearer test-key", search.requests[0]
        assert "api_key" not in search.requests[0]["body"], search.requests[0]

    def test_buffered_large_assistant_content_reenters_complete(
        self, anthropic_client, web_search_stack
    ):
        model = web_search_stack["model"]
        search = web_search_stack["search"]
        model.requests.clear()
        search.requests.clear()
        kwargs = _messages_kwargs()
        kwargs["messages"][0]["content"] = "Use web search with large assistant content."

        response = anthropic_client.messages.create(**kwargs)

        assert response.content[0].text == FINAL_TEXT
        assert len(model.requests) == 2
        assert len(search.requests) == 1
        assistant_turns = [
            message for message in model.requests[1]["messages"] if message["role"] == "assistant"
        ]
        assert len(assistant_turns) == 1
        assert assistant_turns[0]["content"] == [
            {"type": "text", "text": LARGE_ASSISTANT_TEXT},
            {
                "type": "tool_use",
                "id": TOOL_USE_ID,
                "name": "WebSearch",
                "input": {"query": "potato"},
            },
        ]

    def test_buffered_sequence_tool_block_reenters_complete(
        self, anthropic_client, web_search_stack
    ):
        model = web_search_stack["model"]
        search = web_search_stack["search"]
        model.requests.clear()
        search.requests.clear()
        kwargs = _messages_kwargs()
        kwargs["messages"][0]["content"] = "Use web search with a sequence-shaped tool block."

        response = anthropic_client.messages.create(**kwargs)

        assert response.content[0].text == FINAL_TEXT
        assert len(model.requests) == 2
        assert len(search.requests) == 1
        assert search.requests[0]["body"]["query"] == "potato"
        assistant_turns = [
            message for message in model.requests[1]["messages"] if message["role"] == "assistant"
        ]
        assert len(assistant_turns) == 1
        assert assistant_turns[0]["content"] == [
            ["tool_use", "WebSearch", TOOL_USE_ID, ["potato"]]
        ]

    def test_streaming_web_search_loop(self, anthropic_client, web_search_stack):
        model = web_search_stack["model"]
        search = web_search_stack["search"]
        model.requests.clear()
        search.requests.clear()

        event_types: list[str] = []
        collected = ""
        block_types: list[str] = []
        with anthropic_client.messages.stream(**_messages_kwargs()) as stream:
            for event in stream:
                event_types.append(event.type)
                if event.type == "content_block_start":
                    block_types.append(event.content_block.type)
            collected = stream.get_final_message().content[0].text

        assert event_types, "stream produced no events"
        # Exactly one client-visible lifecycle wraps the terminal answer: open
        # with message_start, close with message_stop, exactly one of each so
        # the internal search round is not leaked as a second lifecycle.
        assert event_types[0] == "message_start", f"stream must open with message_start: {event_types}"
        assert event_types[-1] == "message_stop", f"stream must close with message_stop: {event_types}"
        assert event_types.count("message_start") == 1, (
            f"the internal search round must not leak a second message_start: {event_types}"
        )
        assert event_types.count("message_stop") == 1, (
            f"the internal search round must not leak a second message_stop: {event_types}"
        )

        # A text delta arrives strictly between open and close.
        assert "content_block_delta" in event_types, f"the terminal answer must stream a text delta: {event_types}"
        first_delta = event_types.index("content_block_delta")
        assert 0 < first_delta < len(event_types) - 1, (
            f"the text delta must fall strictly between message_start and message_stop: {event_types}"
        )

        # The managed WebSearch tool_use block stays internal to the loop.
        assert block_types == ["text"], f"only the terminal text block is visible: {block_types}"
        assert collected == FINAL_TEXT, f"the streamed answer must equal the terminal text: {collected!r}"

        assert len(model.requests) == 2, "streaming loop re-enters the model"
        assert len(search.requests) == 1, "streaming loop dispatches one search"
        assert model.requests[0].get("stream") is True, "the first model round must request streaming transport"


class TestAnthropicWebSearchCredential:
    def test_missing_user_key_surfaces_authentication_error(
        self, scoped_credential_client, web_search_stack
    ):
        model = web_search_stack["model"]
        search = web_search_stack["search"]
        model.requests.clear()
        search.requests.clear()

        with pytest.raises(AuthenticationError) as exc_info:
            scoped_credential_client.messages.create(
                **_messages_kwargs(),
                extra_headers={"x-auth-tenant": "acme", "x-auth-user": "alice"},
            )

        error = exc_info.value
        assert error.status_code == 401, "missing credential must return 401"
        assert error.body["type"] == "error", "error envelope type must be error"
        assert error.body["error"]["type"] == "authentication_error", (
            "error type must be authentication_error"
        )
        assert "brave_search" in error.body["error"]["message"], (
            "error message must name the missing brave_search credential"
        )
        assert model.requests == [], "missing credential must fail before inference"
        assert search.requests == [], "missing credential must not reach the provider"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
