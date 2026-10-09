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
`metadata.user_id` becomes a hashed `safety_identifier`, that `thinking` is rejected,
that fields the translation cannot honor are rejected before any backend call,
and that malformed streamed tool calls fail closed at the client boundary.

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
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from anthropic import APIConnectionError, APIStatusError, Anthropic, BadRequestError

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
    response_content_type = "application/json"
    send_tool_reply_once = False
    untranslatable_reply_once: str | None = None
    error_response_once = False

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length))
        RecordingBackend.bodies.append(body)
        if RecordingBackend.error_response_once:
            RecordingBackend.error_response_once = False
            reply = json.dumps(
                {
                    "type": "error",
                    "error": {
                        "message": "stubbed upstream failure",
                        "type": "future_error",
                    }
                }
            ).encode()
            self.send_response(500)
            self.send_header("content-type", self.response_content_type)
            self.send_header("content-length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)
            return
        if body.get("stream"):
            self._send_invalid_tool_id_stream()
            return

        choice = {
            "index": 0,
            "message": {"role": "assistant", "content": "4"},
            "finish_reason": "stop",
        }
        if RecordingBackend.send_tool_reply_once:
            RecordingBackend.send_tool_reply_once = False
            choice["message"] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }],
            }
            choice["finish_reason"] = "tool_calls"
        if RecordingBackend.untranslatable_reply_once:
            kind = RecordingBackend.untranslatable_reply_once
            RecordingBackend.untranslatable_reply_once = None
            if kind == "finish_reason":
                choice["finish_reason"] = "content_filter"
            else:
                choice["message"]["refusal"] = "blocked"
        if body.get("stop"):
            # vLLM reports the matched stop string in a choice-level
            # `stop_reason`; pretend the first sequence was generated.
            choice["stop_reason"] = body["stop"][0]
        reply = json.dumps(
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": 1_700_000_000,
                "model": MODEL,
                "choices": [choice],
                "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13},
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", self.response_content_type)
        self.send_header("content-length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def _send_invalid_tool_id_stream(self):
        valid_prefix = {
            "id": "chatcmpl-stub",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "prefix"},
                    "finish_reason": None,
                }
            ],
        }
        invalid_tool_call = {
            "id": "chatcmpl-stub",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call.bad",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{}",
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        chunks = [
            f"data: {json.dumps(valid_prefix)}\n\n".encode(),
            f"data: {json.dumps(invalid_tool_call)}\n\n".encode(),
        ]
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("content-length", str(sum(map(len, chunks))))
        self.end_headers()

        # Flush a valid event first so the invalid tool call is encountered
        # after the downstream streaming response has begun.
        self.wfile.write(chunks[0])
        self.wfile.flush()
        time.sleep(0.1)
        self.wfile.write(chunks[1])
        self.wfile.flush()

    def log_message(self, format, *args):
        pass


def _write_config(proxy_port: int, backend_port: int, allow_lossy: bool = False) -> str:
    with open(CONFIG_PATH) as f:
        config = f.read()
    for old, new in [
        ("127.0.0.1:8080", f"127.0.0.1:{proxy_port}"),
        ("127.0.0.1:8000", f"127.0.0.1:{backend_port}"),
    ]:
        replaced = config.replace(old, new)
        assert replaced != config, f"example drift: {old} not found in {CONFIG_PATH}"
        config = replaced
    if allow_lossy:
        # Opt the strict example into operator-approved degradation by adding the
        # allowlist to the translation filter block.
        marker = "      - filter: anthropic_messages_to_chat_completions\n        max_body_bytes: 1048576\n"
        replacement = (
            marker
            + "        allow_lossy_features:\n"
            + "          - prompt_caching\n"
            + "          - extended_thinking\n"
        )
        replaced = config.replace(marker, replacement)
        assert replaced != config, f"example drift: translation filter block not found in {CONFIG_PATH}"
        config = replaced
    # Rejection evidence must be flushed before the client observes the abort.
    config += "\nruntime:\n  logging:\n    non_blocking: false\n"
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(config)
    return path


def _start_client(log_path: Path, *, allow_lossy: bool) -> Iterator[Anthropic]:
    """Start a backend + proxy and yield an Anthropic client plus the proxy port.

    `allow_lossy` selects the strict example config (the default) or the same
    config with operator-approved degradation enabled on the translation filter.
    The proxy's stdout and stderr go to `log_path`.
    """
    backend_port = _free_port()
    backend = HTTPServer(("127.0.0.1", backend_port), RecordingBackend)
    threading.Thread(target=backend.serve_forever, daemon=True).start()

    proxy_port = _free_port()
    config_path = _write_config(proxy_port, backend_port, allow_lossy=allow_lossy)
    with log_path.open("wb") as output:
        proc = subprocess.Popen(
            [_find_binary(), "-c", config_path],
            stdout=output,
            stderr=subprocess.STDOUT,
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


@pytest.fixture(scope="module")
def proxy_log(tmp_path_factory):
    return tmp_path_factory.mktemp("messages-to-chat").joinpath("proxy.log")


@pytest.fixture(scope="module")
def anthropic_client(proxy_log):
    yield from _start_client(proxy_log, allow_lossy=False)


@pytest.fixture(scope="module")
def degrading_client(tmp_path_factory):
    log_path = tmp_path_factory.mktemp("messages-to-chat-degrading").joinpath("proxy.log")
    yield from _start_client(log_path, allow_lossy=True)


class TestRequestFieldHandling:
    def test_tool_input_presence_is_preserved(self, anthropic_client):
        RecordingBackend.bodies.clear()

        anthropic_client.messages.create(
            model=MODEL,
            max_tokens=64,
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call_missing", "name": "f"},
                        {"type": "tool_use", "id": "call_null", "name": "g", "input": None},
                        {"type": "tool_use", "id": "call_object", "name": "h", "input": {}},
                        {"type": "tool_use", "id": "call_array", "name": "i", "input": [1, 2]},
                    ],
                }
            ],
        )

        [upstream] = RecordingBackend.bodies
        arguments = [call["function"]["arguments"] for call in upstream["messages"][0]["tool_calls"]]
        assert arguments == ["{}", "null", "{}", "[1,2]"]

    def test_mixed_case_vendor_json_response_is_transformed(self, anthropic_client):
        RecordingBackend.bodies.clear()
        RecordingBackend.response_content_type = "Application/Problem+JsOn; charset=utf-8"
        try:
            response = anthropic_client.messages.create(
                model=MODEL,
                max_tokens=64,
                messages=[{"role": "user", "content": "What is 2+2?"}],
            )
        finally:
            RecordingBackend.response_content_type = "application/json"

        assert response.content[0].text == "4"
        assert len(RecordingBackend.bodies) == 1

    def test_unmapped_fields_reach_the_backend(self, anthropic_client):
        RecordingBackend.bodies.clear()

        response = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=64,
            metadata={"user_id": "user-1"},
            extra_body={"top_k": 40},
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

        assert response.content[0].text == "4"
        [upstream] = RecordingBackend.bodies
        assert upstream["top_k"] == 40
        assert upstream["safety_identifier"] == hashlib.sha256(b"user-1").hexdigest()
        assert "metadata" not in upstream

    def test_thinking_cannot_be_silently_dropped(self, anthropic_client):
        RecordingBackend.bodies.clear()

        with pytest.raises(BadRequestError) as excinfo:
            anthropic_client.messages.create(
                model=MODEL,
                max_tokens=2048,
                thinking={"type": "enabled", "budget_tokens": 1024},
                messages=[{"role": "user", "content": "What is 2+2?"}],
            )

        assert excinfo.value.status_code == 400
        assert "thinking" in str(excinfo.value)
        assert RecordingBackend.bodies == []

    def test_default_valued_rejected_field_is_dropped(self, anthropic_client):
        RecordingBackend.bodies.clear()

        response = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=64,
            service_tier="auto",
            extra_body={"n": 1},
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

        assert response.content[0].text == "4"
        [upstream] = RecordingBackend.bodies
        assert "n" not in upstream, "default-valued n must be dropped, not forwarded"
        assert "service_tier" not in upstream, "default-valued service_tier must be dropped, not forwarded"

    def test_serialized_tool_history_and_empty_result_continue(self, anthropic_client):
        RecordingBackend.bodies.clear()
        RecordingBackend.send_tool_reply_once = True

        first = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": "Call lookup"}],
        )
        tool_use = next(block for block in first.content if block.type == "tool_use")
        history_block = tool_use.model_dump()
        history_block.setdefault("toolset_name", None)

        second = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=64,
            messages=[
                {"role": "user", "content": "Call lookup"},
                {"role": "assistant", "content": [history_block]},
                {"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "toolset_name": None,
                }]},
            ],
        )

        assert second.content[0].text == "4"
        assert len(RecordingBackend.bodies) == 2
        upstream = RecordingBackend.bodies[-1]
        assert upstream["messages"][1]["tool_calls"][0]["function"]["name"] == "lookup"
        assert upstream["messages"][2]["content"] == ""

    def test_empty_citations_in_history_reach_the_backend(self, anthropic_client):
        RecordingBackend.bodies.clear()

        response = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=64,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "Hi", "citations": []}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Hello", "citations": []}]},
                {"role": "user", "content": "Continue"},
            ],
        )

        assert response.content[0].text == "4"
        [upstream] = RecordingBackend.bodies
        assert [message["content"] for message in upstream["messages"]] == ["Hi", "Hello", "Continue"]

    def test_unrepresentable_field_is_rejected_before_the_backend(self, anthropic_client):
        RecordingBackend.bodies.clear()

        with pytest.raises(BadRequestError) as excinfo:
            anthropic_client.messages.create(
                model=MODEL,
                max_tokens=64,
                service_tier="standard_only",
                messages=[{"role": "user", "content": "What is 2+2?"}],
            )

        assert "`service_tier` is not supported" in str(excinfo.value), "error message must name the unsupported service_tier field"
        assert RecordingBackend.bodies == [], "rejected request must not reach the backend"

    def test_matched_stop_sequence_is_reported(self, anthropic_client):
        RecordingBackend.bodies.clear()

        response = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=64,
            stop_sequences=[","],
            messages=[{"role": "user", "content": "Count: 1, 2, 3"}],
        )

        [upstream] = RecordingBackend.bodies
        assert upstream["stop"] == [","]
        assert response.stop_reason == "stop_sequence"
        assert response.stop_sequence == ","

    def test_stop_without_matched_sequence_is_end_turn(self, anthropic_client):
        RecordingBackend.bodies.clear()

        response = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

        assert response.stop_reason == "end_turn"
        assert response.stop_sequence is None

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

        assert "`moderation` is not supported" in str(excinfo.value), "error message must name the unsupported moderation field"
        assert RecordingBackend.bodies == [], "rejected request must not reach the backend"


class TestResponseUsage:
    def test_usage_carries_null_output_tokens_details(self, anthropic_client):
        response = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

        # The pinned Messages schema requires the key; the SDK defaults a missing
        # key to None too, so check the wire payload actually carried it.
        assert "output_tokens_details" in response.usage.model_fields_set
        assert response.usage.output_tokens_details is None


class TestResponseValidation:
    def test_unrecognized_upstream_error_type_is_normalized(self, anthropic_client):
        RecordingBackend.error_response_once = True
        try:
            with pytest.raises(APIStatusError) as excinfo:
                anthropic_client.messages.create(
                    model=MODEL,
                    max_tokens=64,
                    messages=[{"role": "user", "content": "Hi"}],
                )
        finally:
            RecordingBackend.error_response_once = False

        error = excinfo.value
        assert error.status_code == 500, "an unrecognized upstream error must surface as HTTP 500"
        assert error.body["error"]["type"] == "api_error", "the Anthropic error type must normalize to api_error"

    @pytest.mark.parametrize("kind, expected_error", [
        ("finish_reason", "Chat finish_reason cannot be translated to Anthropic Messages"),
        ("refusal", "Chat response field cannot be translated to Anthropic Messages"),
    ])
    def test_untranslatable_success_fails_closed(
        self, anthropic_client, proxy_log, kind, expected_error
    ):
        RecordingBackend.bodies.clear()
        log_offset = proxy_log.stat().st_size
        RecordingBackend.untranslatable_reply_once = kind
        try:
            with pytest.raises((APIStatusError, json.JSONDecodeError, APIConnectionError)) as excinfo:
                anthropic_client.messages.create(
                    model=MODEL,
                    max_tokens=64,
                    messages=[{"role": "user", "content": "Hi"}],
                )
        finally:
            RecordingBackend.untranslatable_reply_once = None

        error = excinfo.value
        if isinstance(error, APIStatusError):
            assert error.status_code == 500
            assert error.body["type"] == "error"
            assert error.body["error"]["type"] == "api_error"
        elif isinstance(error, json.JSONDecodeError):
            # A response-body rejection can abort after 200 headers have been
            # sent. The SDK then fails to parse the empty, aborted body.
            assert error.doc == ""
        else:
            logs = proxy_log.read_bytes()[log_offset:].decode("utf-8")
            assert expected_error in logs, "connection failure must follow the controlled response rejection"
            assert "failed to transform Chat Completions-compatible response" in logs
        assert len(RecordingBackend.bodies) == 1, "the controlled response must reach the proxy exactly once"


class TestStreamingResponseValidation:
    def test_invalid_tool_id_aborts_incomplete_stream(self, anthropic_client):
        RecordingBackend.bodies.clear()
        events = []

        with pytest.raises(APIStatusError) as excinfo:
            with anthropic_client.messages.stream(
                model=MODEL,
                max_tokens=64,
                messages=[{"role": "user", "content": "Use the weather tool"}],
            ) as stream:
                events.extend(stream)

        error = excinfo.value
        assert error.body.get("type") == "error", "error envelope type must be error"
        assert error.body.get("error", {}).get("type") == "api_error", "error envelope error type must be api_error"
        assert (
            error.body.get("error", {}).get("message")
            == "upstream response could not be transformed"
        ), "error message must report the failed transformation"
        assert not any(event.type == "message_stop" for event in events), "aborted stream must not emit message_stop"
        assert not any(
            event.type == "content_block_start"
            and event.content_block.type == "tool_use"
            for event in events
        ), "aborted stream must not start a tool_use block"
        [upstream] = RecordingBackend.bodies
        assert upstream["stream"] is True, "forwarded request must have been a stream"


DEGRADED_HEADER = "x-degraded-features"


class TestOperatorApprovedDegradation:
    """`allow_lossy_features` degrades the listed features instead of a 400."""

    def test_prompt_caching_is_degraded_and_reported(self, degrading_client):
        RecordingBackend.bodies.clear()

        raw = degrading_client.messages.with_raw_response.create(
            model=MODEL,
            max_tokens=64,
            system=[{"type": "text", "text": "Be brief", "cache_control": {"type": "ephemeral"}}],
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": "What is 2+2?", "cache_control": {"type": "ephemeral"}}],
            }],
        )

        assert raw.http_response.status_code == 200
        assert "prompt_caching" in raw.http_response.headers.get(DEGRADED_HEADER, "")
        message = raw.parse()
        assert message.content[0].text == "4"

        [upstream] = RecordingBackend.bodies
        serialized = json.dumps(upstream)
        assert "cache_control" not in serialized, f"cache_control must be stripped: {serialized}"
        # The prompt content survives the degradation.
        assert any(m["role"] == "system" and m["content"] == "Be brief" for m in upstream["messages"])

    def test_extended_thinking_is_degraded_and_reported(self, degrading_client):
        RecordingBackend.bodies.clear()

        raw = degrading_client.messages.with_raw_response.create(
            model=MODEL,
            max_tokens=2048,
            thinking={"type": "enabled", "budget_tokens": 1024},
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

        assert raw.http_response.status_code == 200
        assert "extended_thinking" in raw.http_response.headers.get(DEGRADED_HEADER, "")
        message = raw.parse()
        assert message.content[0].text == "4"

        [upstream] = RecordingBackend.bodies
        assert "thinking" not in upstream, f"thinking must be stripped: {upstream}"

    def test_both_features_reported_together(self, degrading_client):
        RecordingBackend.bodies.clear()

        raw = degrading_client.messages.with_raw_response.create(
            model=MODEL,
            max_tokens=2048,
            thinking={"type": "enabled", "budget_tokens": 1024},
            system=[{"type": "text", "text": "Be brief", "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

        assert raw.http_response.status_code == 200
        reported = raw.http_response.headers.get(DEGRADED_HEADER, "")
        assert "prompt_caching" in reported and "extended_thinking" in reported, reported

    def test_streaming_request_is_degraded_and_reported(self, degrading_client):
        RecordingBackend.bodies.clear()

        # The degraded-feature header is set in the response-header phase, before
        # the body streams, so it is observable on a streaming response even
        # though this backend's stream aborts mid-body.
        with degrading_client.messages.with_streaming_response.create(
            model=MODEL,
            # `budget_tokens` must stay below `max_tokens` per the Anthropic schema.
            max_tokens=2048,
            thinking={"type": "enabled", "budget_tokens": 1024},
            messages=[{"role": "user", "content": "Use the weather tool"}],
            stream=True,
        ) as response:
            assert response.status_code == 200
            assert "extended_thinking" in response.headers.get(DEGRADED_HEADER, "")

        [upstream] = RecordingBackend.bodies
        assert upstream["stream"] is True, "forwarded request must have been a stream"
        assert "thinking" not in upstream, f"thinking must be stripped from the streamed request: {upstream}"

    def test_malformed_cache_control_is_still_rejected(self, degrading_client):
        RecordingBackend.bodies.clear()

        with pytest.raises(BadRequestError) as excinfo:
            degrading_client.messages.create(
                model=MODEL,
                max_tokens=64,
                system=[{"type": "text", "text": "Be brief", "cache_control": {"type": "persistent"}}],
                messages=[{"role": "user", "content": "What is 2+2?"}],
            )

        assert excinfo.value.status_code == 400
        assert RecordingBackend.bodies == [], "a malformed marker must not reach the backend"

    def test_unlisted_feature_is_still_rejected(self, degrading_client):
        RecordingBackend.bodies.clear()

        with pytest.raises(BadRequestError) as excinfo:
            degrading_client.messages.create(
                model=MODEL,
                max_tokens=64,
                service_tier="standard_only",
                messages=[{"role": "user", "content": "What is 2+2?"}],
            )

        assert "`service_tier` is not supported" in str(excinfo.value)
        assert RecordingBackend.bodies == [], "an unlisted feature must still fail closed"

    def test_requests_without_lossy_features_report_no_header(self, degrading_client):
        RecordingBackend.bodies.clear()

        raw = degrading_client.messages.with_raw_response.create(
            model=MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

        assert raw.http_response.status_code == 200
        assert DEGRADED_HEADER not in raw.http_response.headers, "a clean request must not be flagged as degraded"

    def test_empty_context_management_is_served_as_a_no_op(self, degrading_client):
        RecordingBackend.bodies.clear()

        # `edits` is optional in the Anthropic schema, so an empty
        # `context_management` is a valid no-op: it is stripped and served, not
        # rejected, and removing a no-op is not itself a degradation.
        raw = degrading_client.messages.with_raw_response.create(
            model=MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            extra_body={"context_management": {}},
        )

        assert raw.http_response.status_code == 200
        assert DEGRADED_HEADER not in raw.http_response.headers, "an empty context_management is a no-op, not a degradation"
        [upstream] = RecordingBackend.bodies
        assert "context_management" not in upstream, f"empty context_management must be stripped: {upstream}"

    def test_malformed_clear_thinking_keep_is_still_rejected(self, degrading_client):
        RecordingBackend.bodies.clear()

        # The `keep` union carries no null, so an explicit null is malformed and
        # must fail closed rather than be stripped as a clean degradation.
        with pytest.raises(BadRequestError) as excinfo:
            degrading_client.messages.create(
                model=MODEL,
                max_tokens=64,
                messages=[{"role": "user", "content": "What is 2+2?"}],
                extra_body={"context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": None}]}},
            )

        assert excinfo.value.status_code == 400
        assert "context_management" in str(excinfo.value)
        assert RecordingBackend.bodies == [], "a malformed edit must not reach the backend"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
