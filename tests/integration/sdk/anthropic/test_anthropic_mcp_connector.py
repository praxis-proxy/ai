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
native Anthropic Messages model and an MCP server. The SDK sends the connector
exactly as it would to Anthropic (`client.beta.messages.create` with
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
from anthropic import Anthropic, BadRequestError

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
# Native Messages model stub
# ---------------------------------------------------------------------------


class _ModelHandler(BaseHTTPRequestHandler):
    """Call the encoded MCP tool once, then answer from its result."""

    protocol_version = "HTTP/1.1"

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        self.server.captured.append(
            {"body": request, "headers": {key.lower(): value for key, value in self.headers.items()}}
        )
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

    def log_message(self, *_args):
        pass


class _ModelStub:
    def __init__(self):
        self.port = _free_port()
        self.requests: list[dict] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), _ModelHandler)
        self._server.captured = self.requests
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> "_ModelStub":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _write_config(proxy_port: int, model_port: int, mcp_url: str) -> str:
    """Retarget the shipped example to the local stubs."""
    with open(CONFIG_PATH) as f:
        config = f.read()
    for old, new in [
        ("127.0.0.1:8080", f"127.0.0.1:{proxy_port}"),
        ('endpoints: ["127.0.0.1:8000"]', f'endpoints: ["127.0.0.1:{model_port}"]'),
        (EXAMPLE_MCP_URL, mcp_url),
    ]:
        replaced = config.replace(old, new)
        assert replaced != config, f"example drift: {old!r} not found"
        config = replaced
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
    config_path = _write_config(proxy_port, model.port, mcp.url)
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

    def test_streaming_connector_request_is_rejected(self, client, stack):
        with pytest.raises(BadRequestError) as excinfo:
            client.beta.messages.create(**_connector_kwargs(stack), stream=True)

        assert "streaming" in str(excinfo.value)
        assert stack["mcp"].requests == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
