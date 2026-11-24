#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "anthropic==1.17.0",
#     "pytest>=8.0",
# ]
# ///
"""
Anthropic Messages MCP connector integration tests through the official SDK.

Drives the shipped `anthropic/mcp-connector.yaml` example with local stubs: a
model serving native Messages and Chat Completions, and an MCP server. The SDK
sends the connector exactly as it would to Anthropic (`client.beta.messages.create` with
`mcp_servers`, an `mcp_toolset` tool, and the `mcp-client-2025-11-20` beta);
Praxis lists the server's tools, runs the model's MCP calls, and returns
`mcp_tool_use` / `mcp_tool_result` blocks that the SDK parses into its own
types.

Usage:
    cargo build -p praxis-ai-proxy --features anthropic-mcp-connector-filter
    uv run tests/integration/sdk/anthropic/test_anthropic_mcp_connector.py -s -v

Environment variables:
    PRAXIS_AI_BIN  path to praxis-ai binary (auto-detected if unset)
"""

import gzip
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.client import HTTPConnection, IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from anthropic import APIStatusError, Anthropic, BadRequestError, InternalServerError

from mcp_stub import McpStub, weather_result

CONFIG_PATH = "examples/configs/anthropic/mcp-connector.yaml"
EXAMPLE_MCP_URL = "http://127.0.0.1:9100/mcp"
MODEL = "openai/gpt-oss-20b"
BETA = "mcp-client-2025-11-20"
ENCODED_TOOL = "weather__get_weather"
FINAL_TEXT = "It is sunny in Paris."

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
        "praxis-ai binary not found — run "
        "`cargo build -p praxis-ai-proxy --features anthropic-mcp-connector-filter` first"
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


def _has_tool_result(request: dict) -> bool:
    for message in request.get("messages", []):
        content = message.get("content")
        if isinstance(content, list) and any(block.get("type") == "tool_result" for block in content):
            return True
    return False


def _message(content: list[dict], stop_reason: str, usage: tuple[int, int]) -> bytes:
    return json.dumps(
        {
            "id": "msg_stub",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
        }
    ).encode()


# ---------------------------------------------------------------------------
# Native Messages and Chat Completions model stub
# ---------------------------------------------------------------------------


class _ModelHandler(BaseHTTPRequestHandler):
    """Emit model rounds for both routes and wait for the streaming client at the gate."""

    protocol_version = "HTTP/1.1"

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        self.server.captured.append(
            {"body": request, "headers": {key.lower(): value for key, value in self.headers.items()}}
        )
        prompt = request["messages"][0]["content"]
        if prompt in ("truncated HTTP error", "oversized HTTP error"):
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self._chunk('{"type":"error",')
            self.server.release_first_round.wait(timeout=10)
            if prompt == "oversized HTTP error":
                self._chunk('"error":{"type":"api_error","message":"' + "x" * 8000 + '"}}')
                self.wfile.write(b"0\r\n\r\n")
            self.close_connection = True
            return
        if prompt == "compressed model response":
            frame = 'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","name":"weather__get_weather","id":"call","input":{}}}\n\n'
            payload = gzip.compress(frame.encode())
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if request.get("stream"):
            if urlsplit(self.path).path == "/v1/chat/completions":
                self._chat_stream(request)
            else:
                self._stream(request)
            return
        tool_names = [tool.get("name") for tool in request.get("tools", [])]
        if _has_tool_result(request) or ENCODED_TOOL not in tool_names:
            body = _message([{"type": "text", "text": FINAL_TEXT}], "end_turn", (20, 7))
        else:
            call = {"type": "tool_use", "id": "toolu_stub_1", "name": ENCODED_TOOL, "input": {"city": "Paris"}}
            body = _message([{"type": "text", "text": "Let me check."}, call], "tool_use", (10, 5))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self, request):
        round_number = sum(
            1 for m in request["messages"]
            if isinstance(m["content"], list) and any(b["type"] == "tool_result" for b in m["content"])
        )
        prompt = request["messages"][0]["content"]
        self._start_sse(large_headers=prompt == "large round headers")

        def event(kind, **payload):
            data = json.dumps({"type": kind, **payload})
            delimiter = "\r" if prompt == "hold after message stop" else "\n"
            self._chunk(delimiter.join((f"event: {kind}", f"data: {data}", "", "")))

        answer = round_number == 2
        message = json.loads(_message([], None, (10, 0)))
        message["id"] = f"msg_{round_number}"
        event("message_start", message=message)
        event("content_block_start", index=0, content_block={"type": "text", "text": ""})
        event("content_block_delta", index=0, delta={"type": "text_delta", "text": FINAL_TEXT if answer else "Checking."})
        if prompt == "hold after error":
            event("error", error={"type": "api_error", "message": "model interrupted"})
            self.server.release_first_round.wait(timeout=10)
            self.close_connection = True
            return
        if prompt == "large round headers":
            event("content_block_delta", index=0, delta={"type": "text_delta", "text": "x" * 3500})
        if round_number == 0:
            # The client releases this gate only after observing the first text.
            # A buffering proxy cannot finish this backend round to unblock itself.
            if not self.server.release_first_round.wait(timeout=10):
                event("error", error={"type": "api_error", "message": "client did not receive the first text delta"})
                self.wfile.write(b"0\r\n\r\n")
                return
        event("content_block_stop", index=0)
        cities = [["Paris", "Lyon"], ["Nice"], []][round_number]
        for index, city in enumerate(cities, start=1):
            event("content_block_start", index=index, content_block={
                "type": "tool_use", "id": f"toolu_{round_number}_{index}", "name": ENCODED_TOOL, "input": {}
            })
            for part in ['{"city":', json.dumps(city) + "}"]:
                event("content_block_delta", index=index, delta={"type": "input_json_delta", "partial_json": part})
            event("content_block_stop", index=index)
        if round_number == 0:
            event("content_block_start", index=3, content_block={"type": "compaction", "content": None})
            event("content_block_delta", index=3, delta={
                "type": "compaction_delta", "content": "Weather checks in progress.", "encrypted_content": "opaque"
            })
            event("content_block_stop", index=3)
            if any(tool["name"] == "local" for tool in request["tools"]):
                event("content_block_start", index=4, content_block={
                    "type": "tool_use", "id": "client_1", "name": "local", "input": {}
                })
                event("content_block_stop", index=4)
        metadata = {"context_management": {"applied_edits": []}, "input_transformations": []} if answer else {}
        usage = {"output_tokens": 5}
        if answer:
            usage["iterations"] = [{"type": "message", "input_tokens": 10, "output_tokens": 5}]
        event("message_delta", delta={"stop_reason": "end_turn" if answer else "tool_use", "stop_sequence": None},
              usage=usage, **metadata)
        event("message_stop")
        if prompt == "hold after message stop":
            self.server.release_message_eof.wait(timeout=10)
            self.close_connection = True
            return
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _start_sse(self, large_headers=False, identity=False):
        self.send_response(200)
        if identity:
            self.send_header("Content-Encoding", "identity")
        if large_headers:
            self.send_header("x-model-metadata", "h" * 4500)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _chunk(self, text):
        frame = text.encode()
        self.wfile.write(f"{len(frame):x}\r\n".encode() + frame + b"\r\n")
        self.wfile.flush()

    def _chat_stream(self, request):
        self._start_sse(identity=True)
        if any(message["role"] == "tool" for message in request["messages"]):
            final = {"id": "chatcmpl_final", "model": request["model"],
                     "choices": [{"index": 0, "delta": {"content": FINAL_TEXT}, "finish_reason": "stop"}]}
            self._chunk(f"data: {json.dumps(final)}\n\n")
        else:
            path = Path(__file__).resolve().parents[2] / "fixtures/inference/recordings/synthetic/messages/mcp-model-tool-stream.json"
            frames = json.loads(path.read_text())["turns"][0]["upstream"]["response"]["body"]["frames"]
            for index, frame in enumerate(frames):
                self._chunk(f"data: {frame['data']}\n\n")
                if index == 0 and not self.server.release_first_round.wait(timeout=10):
                    break
        self._chunk("data: [DONE]\n\n")
        if request["messages"][0]["content"] == "hold after message stop":
            self.server.release_message_eof.wait(timeout=10)
            self.close_connection = True
            return
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def log_message(self, *_args):
        pass


class _ModelStub:
    def __init__(self):
        self.port = _free_port()
        self.requests: list[dict] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), _ModelHandler)
        self.release_first_round = threading.Event()
        self._server.release_first_round = self.release_first_round
        self.release_message_eof = threading.Event()
        self._server.release_message_eof = self.release_message_eof
        self._server.captured = self.requests
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> "_ModelStub":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _write_config(proxy_port: int, model_port: int, mcp_url: str, router_settings: dict | None = None) -> str:
    """Retarget the shipped example to the local stubs."""
    with open(CONFIG_PATH) as f:
        config = f.read()
    for old, new in [
        ("127.0.0.1:8080", f"127.0.0.1:{proxy_port}"),
        ('endpoints: ["127.0.0.1:8000"]', f'endpoints: ["127.0.0.1:{model_port}"]'),
        ('endpoints: ["127.0.0.1:8001"]', f'endpoints: ["127.0.0.1:{model_port}"]'),
        (EXAMPLE_MCP_URL, mcp_url),
    ]:
        replaced = config.replace(old, new)
        assert replaced != config, f"example drift: {old!r} not found"
        config = replaced
    if router_settings:
        router_settings = dict(router_settings)
        if limit := router_settings.pop("connector_max_body_bytes", None):
            config = config.replace("                timeout_ms: 30000",
                                    f"                timeout_ms: 30000\n                max_body_bytes: {limit}")
        settings = "".join(f"\n        {key}: {value}" for key, value in router_settings.items())
        config = config.replace("        initial_step: inference", "        initial_step: inference" + settings)
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(config)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stack(request):
    """Start the model and MCP stubs and a proxy on the shipped example."""
    try:
        binary = _find_binary()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    model = _ModelStub().start()
    mcp = McpStub().start()
    proxy_port = _free_port()
    config_path = _write_config(proxy_port, model.port, mcp.url, getattr(request, "param", None))
    log_fd, log_path = tempfile.mkstemp(suffix=".log")
    log_file = os.fdopen(log_fd, "w")
    started = False
    proc = subprocess.Popen(
        [binary, "-c", config_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        # The Chat Completions cluster's credential is resolved at build time.
        env={**os.environ, "VLLM_API_KEY": "test-vllm-key"},
    )
    try:
        _wait_for_proxy(proxy_port)
        started = True
        yield {"proxy_port": proxy_port, "model": model, "mcp": mcp}
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        model.stop()
        mcp.stop()
        if not started or request.session.testsfailed > 0:
            with open(log_path) as f:
                print(f"\n=== Praxis logs ===\n{f.read()}", file=sys.stderr)
        os.unlink(config_path)
        os.unlink(log_path)


@pytest.fixture
def client(stack):
    stack["model"].requests.clear()
    stack["mcp"].requests.clear()
    return Anthropic(
        base_url=f"http://127.0.0.1:{stack['proxy_port']}",
        api_key="test-anthropic-key",
        max_retries=0,
        timeout=60,
    )


def _connector_kwargs(stack, toolset: dict | None = None) -> dict:
    return {
        "model": MODEL,
        "max_tokens": 512,
        "betas": [BETA],
        "mcp_servers": [{"type": "url", "url": stack["mcp"].url, "name": "weather"}],
        "tools": [toolset or {"type": "mcp_toolset", "mcp_server_name": "weather"}],
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAnthropicMcpConnector:
    def test_sdk_receives_mcp_tool_blocks(self, client, stack):
        response = client.beta.messages.create(**_connector_kwargs(stack))

        assert [block.type for block in response.content] == [
            "text",
            "mcp_tool_use",
            "mcp_tool_result",
            "text",
        ]
        tool_use, tool_result = response.content[1], response.content[2]
        assert tool_use.name == "get_weather"
        assert tool_use.server_name == "weather"
        assert tool_use.input == {"city": "Paris"}
        assert tool_result.tool_use_id == tool_use.id
        assert tool_result.is_error is False
        assert tool_result.content[0].text == weather_result({"city": "Paris"})
        assert response.content[-1].text == FINAL_TEXT
        assert response.stop_reason == "end_turn"
        assert response.usage.input_tokens == 30
        assert response.usage.output_tokens == 12

        calls = stack["mcp"].calls("tools/call")
        assert len(calls) == 1
        assert calls[0]["params"]["name"] == "get_weather"
        assert calls[0]["params"]["arguments"] == {"city": "Paris"}

    def test_backend_never_sees_the_connector(self, client, stack):
        client.beta.messages.create(**_connector_kwargs(stack))

        first = stack["model"].requests[0]
        assert "mcp_servers" not in first["body"]
        assert [tool["name"] for tool in first["body"]["tools"]] == ["weather__get_weather", "weather__delete_city"]
        assert "mcp-client" not in first["headers"].get("anthropic-beta", "")
        second = stack["model"].requests[1]["body"]
        assert second["messages"][-1]["content"][0]["type"] == "tool_result"

    def test_toolset_allowlist_limits_the_tools_sent(self, client, stack):
        toolset = {
            "type": "mcp_toolset",
            "mcp_server_name": "weather",
            "default_config": {"enabled": False},
            "configs": {"get_weather": {"enabled": True}},
        }
        client.beta.messages.create(**_connector_kwargs(stack, toolset))

        tools = stack["model"].requests[0]["body"]["tools"]
        assert [tool["name"] for tool in tools] == ["weather__get_weather"]
        assert tools[0]["input_schema"]["required"] == ["city"]

    def test_replayed_mcp_blocks_continue_the_conversation(self, client, stack):
        kwargs = _connector_kwargs(stack)
        first = client.beta.messages.create(**kwargs)
        stack["model"].requests.clear()

        history = [block.model_dump(exclude_none=True) for block in first.content]
        kwargs["messages"] = [
            *kwargs["messages"],
            {"role": "assistant", "content": history},
            {"role": "user", "content": "Thanks! Anything else?"},
        ]
        second = client.beta.messages.create(**kwargs)

        assert second.content[-1].text == FINAL_TEXT
        sent = stack["model"].requests[0]["body"]
        assert [message["role"] for message in sent["messages"]] == [
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
        ]
        assert sent["messages"][1]["content"][-1]["type"] == "tool_use"
        assert sent["messages"][2]["content"][0]["type"] == "tool_result"
        assert "mcp_tool_" not in json.dumps(sent)

    def test_unconfigured_server_is_rejected(self, client, stack):
        kwargs = _connector_kwargs(stack)
        kwargs["mcp_servers"] = [{"type": "url", "url": "https://unknown.example/mcp", "name": "weather"}]

        with pytest.raises(BadRequestError) as excinfo:
            client.beta.messages.create(**kwargs)

        assert "not an MCP server configured" in str(excinfo.value)
        assert stack["model"].requests == []

    def test_dangling_toolset_is_rejected(self, client, stack):
        kwargs = _connector_kwargs(stack)
        kwargs["tools"] = [{"type": "mcp_toolset", "mcp_server_name": "missing"}]

        with pytest.raises(BadRequestError) as excinfo:
            client.beta.messages.create(**kwargs)

        assert "does not match any mcp_servers entry" in str(excinfo.value)

    def test_sdk_streams_before_eof_and_composes_three_rounds(self, client, stack):
        model = stack["model"]
        model.release_first_round.clear()
        try:
            params = _connector_kwargs(stack)
            params["betas"] += ["compact-2026-01-12", "task-budgets-2026-03-13"]
            with client.beta.messages.stream(**params) as stream:
                events = []
                for event in stream:
                    events.append(event)
                    if event.type == "text" and not model.release_first_round.is_set():
                        assert event.text == "Checking."
                        assert stack["mcp"].calls("tools/call") == []
                        model.release_first_round.set()
                message = stream.get_final_message()
        finally:
            model.release_first_round.set()

        kinds = [event.type for event in events]
        assert kinds.count("message_start") == kinds.count("message_delta") == kinds.count("message_stop") == 1
        starts = [event.index for event in events if event.type == "content_block_start"]
        assert starts == list(range(10))
        assert message.id == "msg_0"
        assert message.stop_reason == "end_turn"
        assert [b.type for b in message.content] == [
            "text", "mcp_tool_use", "mcp_tool_use", "compaction", "mcp_tool_result", "mcp_tool_result",
            "text", "mcp_tool_use", "mcp_tool_result", "text"
        ]
        for call_idx, result_idx, city in [(1, 4, "Paris"), (2, 5, "Lyon"), (7, 8, "Nice")]:
            call, result = message.content[call_idx], message.content[result_idx]
            assert call.name == "get_weather" and call.server_name == "weather"
            assert call.input == {"city": city}
            assert result.tool_use_id == call.id and result.is_error is False
            assert result.content[0].text == weather_result(call.input)
        compaction = message.content[3]
        assert (compaction.content, compaction.encrypted_content) == ("Weather checks in progress.", "opaque")
        history = model.requests[1]["body"]["messages"][-2]["content"]
        assert history[-1] == compaction.model_dump(exclude_none=True)
        assert message.context_management.applied_edits == []
        assert message.input_transformations == []
        assert message.usage.iterations[0].type == "message"
        assert message.content[-1].text == FINAL_TEXT
        assert message.usage.input_tokens == 30
        assert message.usage.output_tokens == 15
        assert len(model.requests) == 3 and all(req["body"]["stream"] for req in model.requests)
        calls = stack["mcp"].calls("tools/call")
        assert sorted(call["params"]["arguments"]["city"] for call in calls) == ["Lyon", "Nice", "Paris"]

    def test_sdk_receives_translated_call_before_arguments_finish(self, client, stack):
        model = stack["model"]
        model.release_first_round.clear()
        params = {**_connector_kwargs(stack), "model": "Qwen/Qwen3-8B"}
        try:
            with client.beta.messages.stream(**params) as stream:
                for event in stream:
                    if event.type == "content_block_start" and event.content_block.type == "mcp_tool_use":
                        assert stack["mcp"].calls("tools/call") == []
                        model.release_first_round.set()
                message = stream.get_final_message()
        finally:
            model.release_first_round.set()
        call, result, text = message.content
        assert (call.server_name, call.name, call.input) == ("weather", "get_weather", {"city": "Paris"})
        assert result.tool_use_id == call.id
        assert result.content[0].text == weather_result(call.input)
        assert text.text == FINAL_TEXT and message.stop_reason == "end_turn"
        assert len(model.requests) == 2 and len(stack["mcp"].calls("tools/call")) == 1

    def test_encoded_first_model_round_returns_a_messages_http_error(self, client, stack):
        params = _connector_kwargs(stack)
        params["messages"][0]["content"] = "compressed model response"
        with pytest.raises(InternalServerError) as failure:
            with client.beta.messages.stream(**params) as stream:
                list(stream)
        assert failure.value.status_code == 502
        assert failure.value.response.headers["content-type"] == "application/json"
        assert failure.value.response.json()["error"]["type"] == "api_error"
        assert stack["model"].requests[0]["headers"]["accept-encoding"] == "identity"
        assert stack["mcp"].calls("tools/call") == []

    @pytest.mark.parametrize("stack", [{"max_state_bytes": 8000}], indirect=True)
    def test_oversize_mixed_results_end_the_message_with_an_error(self, client, stack, monkeypatch):
        monkeypatch.setattr("mcp_stub.weather_result", lambda _arguments: "x" * 6000)
        stack["model"].release_first_round.set()
        params = _connector_kwargs(stack)
        params["tools"].append({"name": "local", "input_schema": {"type": "object"}})
        blocks = []
        with pytest.raises(APIStatusError, match="retained-state limit"):
            with client.beta.messages.stream(**params) as stream:
                for event in stream:
                    if event.type == "content_block_start":
                        blocks.append(event.content_block.type)
        assert blocks.count("mcp_tool_use") == 2 and "tool_use" in blocks
        assert len(stack["mcp"].calls("tools/call")) == 2
        assert len(stack["model"].requests) == 1

    @pytest.mark.parametrize("stack", [{"max_state_bytes": 8000}], indirect=True)
    def test_current_headers_and_round_history_share_the_router_budget(self, client, stack):
        stack["model"].release_first_round.set()
        params = _connector_kwargs(stack)
        params["messages"][0]["content"] = "large round headers"
        seen = []
        with pytest.raises(APIStatusError, match="retained-state limit") as failure:
            with client.beta.messages.stream(**params) as stream:
                seen.extend(event.type for event in stream)
        assert failure.value.status_code == 200 and seen.count("message_start") == 1
        assert len(stack["model"].requests) == 1 and not stack["mcp"].calls("tools/call")

    @pytest.mark.parametrize("stack", [{"connector_max_body_bytes": 2048}], indirect=True)
    @pytest.mark.parametrize("prompt", ["truncated HTTP error", "oversized HTTP error"])
    def test_partial_http_error_ends_as_a_transport_failure(self, stack, prompt):
        params = _connector_kwargs(stack)
        params.pop("betas")
        params["stream"] = True
        params["messages"][0]["content"] = prompt
        gate = stack["model"].release_first_round
        gate.clear()
        conn = HTTPConnection("127.0.0.1", stack["proxy_port"], timeout=4)
        try:
            conn.request("POST", "/v1/messages", json.dumps(params), {
                "content-type": "application/json", "anthropic-beta": BETA,
            })
            response = conn.getresponse()
            assert response.status == 503
            prefix = b'{"type":"error",'
            assert response.read(len(prefix)) == prefix
            gate.set()
            with pytest.raises(IncompleteRead) as failure:
                response.read()
            assert failure.value.partial == b""
        finally:
            gate.set()
            conn.close()

    @pytest.mark.parametrize("model_name,rounds,calls", [(MODEL, 3, 3), ("Qwen/Qwen3-8B", 2, 1)])
    def test_message_stop_completes_each_round_while_backend_bodies_remain_open(
        self, stack, client, model_name, rounds, calls
    ):
        params = {**_connector_kwargs(stack), "model": model_name}
        params["messages"][0]["content"] = "hold after message stop"
        model = stack["model"]
        model.release_first_round.set()
        model.release_message_eof.clear()
        try:
            with client.with_options(timeout=4).beta.messages.stream(**params) as stream:
                message = stream.get_final_message()
            assert message.stop_reason == "end_turn" and message.content[-1].text == FINAL_TEXT
            assert len(model.requests) == rounds and len(stack["mcp"].calls("tools/call")) == calls
        finally:
            model.release_message_eof.set()

    def test_provider_error_closes_client_response_before_upstream_eof(self, stack):
        params = _connector_kwargs(stack)
        params.pop("betas")
        params["stream"] = True
        params["messages"][0]["content"] = "hold after error"
        gate = stack["model"].release_first_round
        gate.clear()
        conn = HTTPConnection("127.0.0.1", stack["proxy_port"], timeout=4)
        try:
            conn.request("POST", "/v1/messages", json.dumps(params), {
                "content-type": "application/json", "anthropic-beta": BETA,
            })
            response = conn.getresponse()
            assert response.status == 200
            body = response.read()
            assert body.count(b"event: error") == 1 and b"model interrupted" in body
            assert b"event: message_start" in body and b"event: message_stop" not in body
        finally:
            gate.set()
            conn.close()

    @pytest.mark.parametrize("stack", [{"step_timeout_ms": 1500}], indirect=True)
    def test_step_timeout_during_mcp_call_terminates_the_open_message(self, client, stack, monkeypatch):
        gate = threading.Event()

        def held_result(arguments):
            gate.wait(timeout=10)
            return weather_result(arguments)

        monkeypatch.setattr("mcp_stub.weather_result", held_result)
        stack["model"].release_first_round.set()
        seen = []
        try:
            with pytest.raises(APIStatusError, match="MCP tool execution") as error:
                with client.beta.messages.stream(**_connector_kwargs(stack)) as stream:
                    seen.extend(event.type for event in stream)
            assert error.value.status_code == 200
        finally:
            gate.set()
        assert seen.count("message_start") == 1 and "message_stop" not in seen
        assert stack["mcp"].calls("tools/call")
        assert len(stack["model"].requests) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
