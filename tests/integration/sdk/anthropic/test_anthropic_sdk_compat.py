#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "anthropic",
#     "pytest>=8.0",
# ]
# ///
"""Anthropic SDK v1/v2 compatibility matrix for Messages routes.

Exercises both Anthropic SDK major-version lanes against local recording
backends across the two routes where the Anthropic SDK is the client —
native Messages passthrough and Anthropic Messages -> Chat Completions
translation — the only routes whose outbound wire headers differ between
SDK v1 (`anthropic-version: 2023-06-01`, `x-api-key`) and SDK v2
(`anthropic-version: 2026-10-21`, `Authorization: Bearer`).

Native request guardrails are also checked through `nemo-guardrails.yaml`,
using recording NeMo and model backends for both SDK lanes.

The third translation direction (OpenAI -> Anthropic Messages) is driven
by the OpenAI SDK as the client, so it does not vary on this axis; its
backend wire headers are asserted in
``tests/integration/sdk/openai/test_selected_anthropic_chat_completions.py``.

Everything runs against local stub backends: no Anthropic credentials and
no external provider calls. Select the lane explicitly with
``--anthropic-sdk-version=1.x`` or ``--anthropic-sdk-version=2.x`` (see the
conftest), e.g.:

    cargo build -p praxis-ai-proxy
    uv run --with "anthropic==1.17.0" tests/integration/sdk/anthropic/test_anthropic_sdk_compat.py -s -v --anthropic-sdk-version=1.x
    uv run --with "anthropic==2.3.0" tests/integration/sdk/anthropic/test_anthropic_sdk_compat.py -s -v --anthropic-sdk-version=2.x
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from collections import deque
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from anthropic import APIStatusError, Anthropic, BadRequestError

from conftest import find_binary as _find_binary
from conftest import free_port as _free_port
from conftest import wait_for_proxy as _wait_for_proxy

NATIVE_CONFIG_PATH = "examples/configs/anthropic/messages-protocol.yaml"
TRANSLATED_CONFIG_PATH = "examples/configs/anthropic/messages-to-openai.yaml"
MODEL = "stub-model"


class _RecordingState:
    """Shared recording state for one stub backend.

    Requests are recorded as ``(path, headers, body)`` with header names
    lower-cased. Scripted responses are consumed first-in-first-out; when
    the script is empty the backend serves its default reply.
    """

    def __init__(self, default_reply: bytes, default_content_type: str) -> None:
        self.lock = threading.Lock()
        self.requests: list[tuple[str, dict[str, str], dict]] = []
        self.scripted: deque[tuple[int, str, bytes]] = deque()
        self.default_reply = default_reply
        self.default_content_type = default_content_type

    def script(self, status: int, content_type: str, payload: dict | str | bytes) -> None:
        if isinstance(payload, bytes):
            encoded = payload
        elif isinstance(payload, dict):
            encoded = json.dumps(payload).encode()
        else:
            encoded = payload.encode()
        with self.lock:
            self.scripted.append((status, content_type, encoded))

    def reset(self) -> None:
        with self.lock:
            self.requests.clear()
            self.scripted.clear()

    def take(self, path: str, headers: dict[str, str], body: bytes) -> tuple[int, str, bytes]:
        with self.lock:
            self.requests.append((path, headers, json.loads(body)))
            if self.scripted:
                return self.scripted.popleft()
            return (200, self.default_content_type, self.default_reply)


def _anthropic_message(content: list[dict], stop_reason: str = "end_turn") -> bytes:
    return json.dumps({
        "type": "message",
        "id": "msg_stub",
        "role": "assistant",
        "model": MODEL,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }).encode()


def _anthropic_sse(text: str = "hello") -> bytes:
    events = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_stub", "type": "message", "role": "assistant", "model": MODEL,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 0},
        }}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                           "usage": {"output_tokens": 4}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    wire = "".join(
        f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events
    )
    return wire.encode()


def _chat_completion(content: str | None = None, tool_calls: list[dict] | None = None,
                     finish_reason: str = "stop") -> bytes:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        finish_reason = "tool_calls"
    return json.dumps({
        "id": "chatcmpl-stub",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": MODEL,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13},
    }).encode()


def _chat_sse(text: str = "hello") -> bytes:
    chunks = []
    deltas = [({"role": "assistant", "content": ""}, None), ({"content": text}, None), ({}, "stop")]
    for delta, finish_reason in deltas:
        chunks.append(json.dumps({
            "id": "chatcmpl-stub",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": MODEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }))
    wire = "".join(f"data: {chunk}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return wire.encode()


NATIVE_STATE = _RecordingState(_anthropic_message([{"type": "text", "text": "hello"}]),
                               "application/json")
TRANSLATED_STATE = _RecordingState(_chat_completion(content="hello"), "application/json")
NEMO_STATE = _RecordingState(
    json.dumps({"status": "passed", "content": "Hello"}).encode(), "application/json"
)


class _RecordingHandler(BaseHTTPRequestHandler):
    """POST handler bound to a module-level `_RecordingState`."""

    state: _RecordingState

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        headers = {key.lower(): value for key, value in self.headers.items()}
        status, content_type, reply = self.state.take(self.path, headers, body)
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, format, *args):
        pass


class _NativeHandler(_RecordingHandler):
    state = NATIVE_STATE


class _TranslatedHandler(_RecordingHandler):
    state = TRANSLATED_STATE


class _NemoHandler(_RecordingHandler):
    state = NEMO_STATE


@pytest.fixture(autouse=True)
def _reset_recording_states():
    NATIVE_STATE.reset()
    TRANSLATED_STATE.reset()
    NEMO_STATE.reset()


def _retarget_example(config_path: str, replacements: list[tuple[str, str]]) -> str:
    with open(config_path) as f:
        config = f.read()
    for old, new in replacements:
        replaced = config.replace(old, new)
        assert replaced != config, f"example drift: {old} not found in {config_path}"
        config = replaced
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(config)
    return path


def _start_proxy(config_path: str, port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [_find_binary(), "-c", config_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_proxy(port)
    except BaseException:
        _stop_proxy(proc)
        raise
    return proc


def _start_backend(state: _RecordingState, handler_class) -> tuple[ThreadingHTTPServer, threading.Thread]:
    backend_port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", backend_port), handler_class)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_backend(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _stop_proxy(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture(scope="module")
def native_client():
    """Anthropic SDK client against native Messages passthrough (example config)."""
    # ExitStack registers cleanup as each resource is created, so a failure
    # partway through startup (backend, config, proxy) still releases the
    # resources that were already created.
    with ExitStack() as stack:
        server, thread = _start_backend(NATIVE_STATE, _NativeHandler)
        stack.callback(_stop_backend, server, thread)
        proxy_port = _free_port()
        config_path = _retarget_example(NATIVE_CONFIG_PATH, [
            ("127.0.0.1:8080", f"127.0.0.1:{proxy_port}"),
            ("127.0.0.1:3001", f"127.0.0.1:{server.server_address[1]}"),
        ])
        stack.callback(os.unlink, config_path)
        proc = _start_proxy(config_path, proxy_port)
        stack.callback(_stop_proxy, proc)
        yield Anthropic(
            base_url=f"http://127.0.0.1:{proxy_port}",
            api_key="test-key",
            max_retries=0,
            timeout=10.0,
        )


@pytest.fixture(scope="module")
def translated_client():
    """Anthropic SDK client against Messages -> Chat Completions translation."""
    with ExitStack() as stack:
        server, thread = _start_backend(TRANSLATED_STATE, _TranslatedHandler)
        stack.callback(_stop_backend, server, thread)
        proxy_port = _free_port()
        config_path = _retarget_example(TRANSLATED_CONFIG_PATH, [
            ("127.0.0.1:8080", f"127.0.0.1:{proxy_port}"),
            ("127.0.0.1:8000", f"127.0.0.1:{server.server_address[1]}"),
        ])
        stack.callback(os.unlink, config_path)
        proc = _start_proxy(config_path, proxy_port)
        stack.callback(_stop_proxy, proc)
        yield Anthropic(
            base_url=f"http://127.0.0.1:{proxy_port}",
            api_key="test-key",
            max_retries=0,
            timeout=10.0,
        )


@pytest.fixture()
def native_guardrails_client():
    """Exercise the checked-in request guardrails example with recording providers."""
    with ExitStack() as stack:
        backend, backend_thread = _start_backend(NATIVE_STATE, _NativeHandler)
        stack.callback(_stop_backend, backend, backend_thread)
        nemo, nemo_thread = _start_backend(NEMO_STATE, _NemoHandler)
        stack.callback(_stop_backend, nemo, nemo_thread)
        port = _free_port()
        config_path = _retarget_example("examples/configs/nemo-guardrails.yaml", [
            ("0.0.0.0:8080", f"127.0.0.1:{port}"),
            ("127.0.0.1:3000", f"127.0.0.1:{backend.server_address[1]}"),
            ("127.0.0.1:3001", f"127.0.0.1:{nemo.server_address[1]}"),
        ])
        stack.callback(os.unlink, config_path)
        proc = _start_proxy(config_path, port)
        stack.callback(_stop_proxy, proc)
        yield stack.enter_context(Anthropic(
            base_url=f"http://127.0.0.1:{port}", api_key="test-key",
            max_retries=0, timeout=10.0,
        ))


def _native_expected_headers(lane: str) -> dict[str, str]:
    """Wire headers each SDK lane must send on the native route."""
    if lane == "2.x":
        return {"anthropic-version": "2026-10-21", "authorization": "Bearer test-key"}
    return {"anthropic-version": "2023-06-01", "x-api-key": "test-key"}


def _native_forbidden_headers(lane: str) -> list[str]:
    return ["x-api-key"] if lane == "2.x" else ["authorization"]


class TestNativeGuardrails:
    """Native Messages must be screened before forwarding through either SDK lane."""

    @pytest.mark.parametrize("content", ["Hello", [{"type": "text", "text": "Hello"}]],
                             ids=["string-content", "text-blocks"])
    def test_canonical_checks_preserve_native_input(self, native_guardrails_client, anthropic_sdk_info, content):
        body = {
            "model": MODEL, "max_tokens": 64,
            "system": [{"type": "text", "text": "Follow policy."}],
            "messages": [{"role": "user", "content": content}],
        }
        response = native_guardrails_client.messages.create(**body)
        assert response.content[0].text == "hello", "the native response must parse through the SDK"
        assert len(NEMO_STATE.requests) == 1, "the SDK request must actually be screened"
        check_path, check_headers, check = NEMO_STATE.requests[0]
        assert check_path == "/v1/checks", "guardrails must use the NeMo checks endpoint"
        assert check["messages"] == [
            {"role": "system", "content": "Follow policy."},
            {"role": "user", "content": "Hello"},
        ], "NeMo must receive canonical system and user text"
        assert check["guardrails"] == {"rail_types": ["input"], "config_ids": ["your-config"]}, (
            "the example must retain its guardrail configuration and input phase"
        )
        assert "authorization" not in check_headers and "x-api-key" not in check_headers, (
            "caller credentials must not be projected onto NeMo"
        )
        assert len(NATIVE_STATE.requests) == 1, "a passed verdict must forward exactly once"
        path, headers, forwarded = NATIVE_STATE.requests[0]
        assert path == "/v1/messages", "the native Messages route must be preserved"
        assert forwarded == body, "guardrails must preserve all original Anthropic fields"
        _, lane = anthropic_sdk_info
        for name, expected in _native_expected_headers(lane).items():
            assert headers.get(name) == expected, f"guardrails must preserve SDK {lane} header {name}"

    @pytest.mark.parametrize("verdict, content, expected_status", [
        ("blocked", "Contact alice@example.com", 403),
        ("modified", [{"type": "text", "text": "Contact alice@"},
                      {"type": "text", "text": "example.com"}], 502),
    ])
    def test_unsafe_verdicts_prevent_model_calls(self, native_guardrails_client, verdict, content, expected_status):
        NEMO_STATE.script(200, "application/json", {
            "status": verdict, "content": "masked", "rail": "pii",
        })
        with pytest.raises(APIStatusError) as error:
            native_guardrails_client.messages.create(
                model=MODEL, max_tokens=64, system="Follow policy.",
                messages=[{"role": "user", "content": content}],
            )
        assert error.value.status_code == expected_status, "the SDK must observe the guardrail rejection"
        assert len(NEMO_STATE.requests) == 1, "the rejection must follow a real provider call"
        assert not NATIVE_STATE.requests, "blocked or unsupported input must never reach the model"

    def test_modified_text_reaches_model_masked(self, native_guardrails_client):
        NEMO_STATE.script(200, "application/json", {"status": "modified", "content": "Contact <EMAIL_ADDRESS>"})
        response = native_guardrails_client.messages.create(
            model=MODEL, max_tokens=64, system="Follow policy.",
            messages=[{"role": "user", "content": "Contact alice@example.com"}],
        )
        assert response.content[0].text == "hello", "the model response must parse through the SDK"
        assert len(NEMO_STATE.requests) == 1, "the original input must be screened"
        assert len(NATIVE_STATE.requests) == 1, "the masked input must reach the model once"
        _, _, forwarded = NATIVE_STATE.requests[0]
        assert forwarded["messages"][0]["content"] == "Contact <EMAIL_ADDRESS>", "only masked text may reach the model"

    def test_unchanged_modification_keeps_native_body(self, native_guardrails_client):
        NEMO_STATE.script(200, "application/json", {"status": "modified", "content": "Hello"})
        body = {"model": MODEL, "max_tokens": 64, "messages": [{"role": "user", "content": "Hello"}]}
        response = native_guardrails_client.messages.create(**body)
        assert response.content[0].text == "hello", "an unchanged modification must pass through the SDK"
        assert len(NEMO_STATE.requests) == 1, "unchanged input must still be evaluated"
        assert len(NATIVE_STATE.requests) == 1, "an unchanged verdict must reach the model once"
        assert NATIVE_STATE.requests[0][2] == body, "a no-op modification must preserve the native request"


class TestNativePassthrough:
    """Route 1: native `/v1/messages` passthrough (`messages-protocol.yaml`)."""

    def test_wire_headers_match_the_sdk_lane(self, native_client, anthropic_sdk_info):
        _version, lane = anthropic_sdk_info
        assert lane in ("1.x", "2.x"), f"unexpected Anthropic SDK lane {lane!r}"

        native_client.messages.create(
            model=MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": "Hi"}],
        )

        [path, headers, _body] = NATIVE_STATE.requests[-1]
        assert path == "/v1/messages", "native route must keep the /v1/messages path"
        for name, expected in _native_expected_headers(lane).items():
            assert headers.get(name) == expected, (
                f"SDK {lane} must send {name}: {expected!r} on the native route, "
                f"got {headers.get(name)!r}"
            )
        for name in _native_forbidden_headers(lane):
            assert name not in headers, (
                f"SDK {lane} must not send {name} alongside its own auth scheme"
            )

    @pytest.mark.parametrize("version", ["2023-06-01", "2026-10-21"])
    @pytest.mark.parametrize("field,value", [("temperature", 0.5), ("top_p", 0.9), ("top_k", 40)])
    def test_native_sampling_and_version_reach_backend_unchanged(self, native_client, version, field, value):
        native_client.messages.create(
            model=MODEL, max_tokens=64,
            messages=[{"role": "user", "content": "Hi"}],
            extra_headers={"anthropic-version": version},
            extra_body={field: value},
        )
        path, headers, body = NATIVE_STATE.requests[-1]
        assert path == "/v1/messages"
        assert headers["anthropic-version"] == version, "native caller-selected version must be preserved"
        assert body[field] == value, "native passthrough leaves version restrictions to the backend"

    def test_json_message_round_trip(self, native_client):
        response = native_client.messages.create(
            model=MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert response.role == "assistant"
        assert response.content[0].type == "text"
        assert response.content[0].text == "hello"
        assert response.stop_reason == "end_turn"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 4

    def test_streaming_event_lifecycle(self, native_client):
        NATIVE_STATE.script(200, "text/event-stream", _anthropic_sse("hello"))

        events = []
        with native_client.messages.stream(
            model=MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": "Hi"}],
        ) as stream:
            events.extend(stream)
            text = stream.get_final_message().content[0].text

        assert text == "hello"
        kinds = [event.type for event in events]
        for expected in ["message_start", "content_block_start", "content_block_delta",
                         "message_delta", "message_stop"]:
            assert expected in kinds, f"native SSE lifecycle must include {expected}: {kinds}"
        assert kinds.index("message_start") < kinds.index("content_block_delta"), (
            f"message_start must precede content_block_delta: {kinds}"
        )
        assert kinds.index("content_block_delta") < kinds.index("message_stop"), (
            f"content_block_delta must precede message_stop: {kinds}"
        )

    def test_tool_use_round_trip(self, native_client):
        NATIVE_STATE.script(200, "application/json", _anthropic_message(
            [{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"k": 1}}],
            stop_reason="tool_use",
        ))

        first = native_client.messages.create(
            model=MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": "Call lookup"}],
            tools=[{
                "name": "lookup",
                "description": "Look up a value",
                "input_schema": {"type": "object"},
            }],
        )

        tool_use = next(block for block in first.content if block.type == "tool_use")
        assert tool_use.name == "lookup"
        assert tool_use.input == {"k": 1}
        assert first.stop_reason == "tool_use"

        second = native_client.messages.create(
            model=MODEL,
            max_tokens=64,
            messages=[
                {"role": "user", "content": "Call lookup"},
                {"role": "assistant", "content": [tool_use.model_dump()]},
                {"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": "found",
                }]},
            ],
        )

        assert second.content[0].text == "hello"
        _path, _headers, body = NATIVE_STATE.requests[-1]
        blocks = body["messages"][-1]["content"]
        assert any(
            block.get("type") == "tool_result" and block.get("tool_use_id") == tool_use.id
            for block in blocks
        ), "tool_result block must reach the native backend unchanged"


class TestTranslatedToChatCompletions:
    """Route 2: Anthropic Messages -> Chat Completions (`messages-to-openai.yaml`)."""

    def test_wire_headers_target_chat_completions(self, translated_client, anthropic_sdk_info):
        _version, lane = anthropic_sdk_info

        translated_client.messages.create(
            model=MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": "Hi"}],
        )

        [path, headers, body] = TRANSLATED_STATE.requests[-1]
        assert path == "/v1/chat/completions", (
            "translated route must rewrite the path to Chat Completions"
        )
        assert body["model"] == MODEL
        assert "anthropic-version" not in headers, (
            f"the Anthropic protocol header must not leak to a Chat Completions "
            f"backend (SDK {lane})"
        )
        assert "x-api-key" not in headers, (
            f"the Anthropic credential header must not leak to a Chat Completions "
            f"backend (SDK {lane})"
        )
        assert "authorization" not in headers, (
            f"the caller's credential must not leak to a Chat Completions backend "
            f"in any SDK lane — SDK v2 sends it as 'authorization: Bearer' "
            f"(SDK {lane})"
        )
        assert headers.get("content-type") == "application/json"

    def test_json_message_translation(self, translated_client):
        response = translated_client.messages.create(
            model=MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert response.content[0].text == "hello"
        assert response.stop_reason == "end_turn"
        _path, _headers, body = TRANSLATED_STATE.requests[-1]
        assert body["messages"] == [{"role": "user", "content": "Hi"}], (
            "the Anthropic message must be translated to a Chat Completions message"
        )

    def test_streaming_translation(self, translated_client):
        TRANSLATED_STATE.script(200, "text/event-stream", _chat_sse("hello"))

        with translated_client.messages.stream(
            model=MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": "Hi"}],
        ) as stream:
            text = stream.get_final_message().content[0].text

        assert text == "hello", "Chat Completions chunks must be translated to Anthropic SSE"
        _path, _headers, body = TRANSLATED_STATE.requests[-1]
        assert body["stream"] is True, "streaming must be requested on the translated route"

    def test_tool_round_trip(self, translated_client):
        TRANSLATED_STATE.script(200, "application/json", _chat_completion(tool_calls=[{
            "id": "call_1", "type": "function",
            "function": {"name": "lookup", "arguments": "{\"k\": 1}"},
        }]))

        first = translated_client.messages.create(
            model=MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": "Call lookup"}],
            tools=[{
                "name": "lookup",
                "description": "Look up a value",
                "input_schema": {"type": "object"},
            }],
        )

        tool_use = next(block for block in first.content if block.type == "tool_use")
        assert tool_use.name == "lookup"
        assert tool_use.input == {"k": 1}
        assert first.stop_reason == "tool_use"

        second = translated_client.messages.create(
            model=MODEL,
            max_tokens=64,
            messages=[
                {"role": "user", "content": "Call lookup"},
                {"role": "assistant", "content": [tool_use.model_dump()]},
                {"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": "found",
                }]},
            ],
        )

        assert second.content[0].text == "hello"
        _path, _headers, body = TRANSLATED_STATE.requests[-1]
        translated_history = body["messages"]
        assert translated_history[1]["tool_calls"][0]["function"]["name"] == "lookup", (
            "the assistant tool_use block must become a Chat Completions tool call"
        )
        assert translated_history[-1]["role"] == "tool", (
            "the tool_result block must become a Chat Completions tool message"
        )
        assert translated_history[-1]["tool_call_id"] == tool_use.id

    @pytest.mark.parametrize("version", ["2023-06-01", "2026-10-21"])
    @pytest.mark.parametrize("temperature,top_p", [(1, 1), (0.5, 0.9)])
    def test_sampling_parameters_are_translated(self, translated_client, version, temperature, top_p):
        # Sampling parameters are no longer typed SDK fields; they travel as
        # request-body fields via extra_body on both lanes.
        translated_client.messages.create(
            model=MODEL,
            max_tokens=64,
            stop_sequences=["\n"],
            extra_body={"temperature": temperature, "top_p": top_p, "top_k": 40},
            extra_headers={"anthropic-version": version},
            messages=[{"role": "user", "content": "Hi"}],
        )

        _path, headers, body = TRANSLATED_STATE.requests[-1]
        assert "anthropic-version" not in headers, "caller version must not reach a Chat Completions backend"
        assert body["temperature"] == temperature
        assert body["top_p"] == top_p
        assert body["stop"] == ["\n"], "stop_sequences must become the Chat Completions stop"
        assert body["top_k"] == 40, "top_k must be forwarded as an unmapped field"

    def test_empty_citations_are_accepted(self, translated_client):
        response = translated_client.messages.create(
            model=MODEL,
            max_tokens=64,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "Hi", "citations": []}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Hello", "citations": None}]},
                {"role": "user", "content": "Continue"},
            ],
        )

        assert response.content[0].text == "hello"
        _path, _headers, body = TRANSLATED_STATE.requests[-1]
        assert [message["content"] for message in body["messages"]] == ["Hi", "Hello", "Continue"]

    def test_active_citations_are_rejected(self, translated_client):
        with pytest.raises(BadRequestError) as excinfo:
            translated_client.messages.create(
                model=MODEL,
                max_tokens=64,
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "Hi",
                        "citations": [{"type": "web_search_result_location",
                                       "url": "https://example.test", "cited_text": "Hi"}],
                    }],
                }],
            )

        assert excinfo.value.status_code == 400
        assert "unsupported Anthropic content block" in str(excinfo.value)
        assert TRANSLATED_STATE.requests == [], "rejected request must not reach the backend"

    def test_document_block_is_rejected(self, translated_client):
        with pytest.raises(BadRequestError) as excinfo:
            translated_client.messages.create(
                model=MODEL,
                max_tokens=64,
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "document",
                        "source": {"type": "text", "media_type": "text/plain", "data": "doc"},
                    }],
                }],
            )

        assert excinfo.value.status_code == 400
        assert "unsupported Anthropic content block" in str(excinfo.value)
        assert TRANSLATED_STATE.requests == []

    def test_thinking_block_is_rejected(self, translated_client):
        with pytest.raises(BadRequestError) as excinfo:
            translated_client.messages.create(
                model=MODEL,
                max_tokens=2048,
                messages=[{
                    "role": "assistant",
                    "content": [{
                        "type": "thinking",
                        "thinking": "prior reasoning",
                        "signature": "sig",
                    }],
                }],
            )

        assert excinfo.value.status_code == 400
        assert "unsupported Anthropic content block" in str(excinfo.value)
        assert TRANSLATED_STATE.requests == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
