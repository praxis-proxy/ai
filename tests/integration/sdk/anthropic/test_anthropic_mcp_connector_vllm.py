#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "anthropic==1.17.0",
#     "pytest>=8.0",
# ]
# ///
"""Official Anthropic SDK -> Praxis -> real vLLM MCP connector acceptance.

Drives the Chat Completions route of the shipped
``anthropic/mcp-connector.yaml`` example against a live model and a local MCP
stub server. The SDK sends the connector exactly as it would to Anthropic;
Praxis lists the stub's tools, the live model calls one, Praxis runs it over
MCP and re-enters inference, and the SDK receives ``mcp_tool_use`` /
``mcp_tool_result`` blocks followed by the model's answer.

Determinism comes from forcing, not prompting: ``tool_choice={"type": "any"}``
lowers to a required Chat Completions tool call, which vLLM enforces through
constrained decoding even on a small model. The toolset allowlists only
``get_weather``, so the forced call can only target it. On re-entry Praxis
relaxes ``tool_choice`` to ``auto`` so the model can answer and the loop ends.

Required: PRAXIS_TEST_VLLM_BASE_URL (local http URL), PRAXIS_TEST_VLLM_MODEL,
VLLM_API_KEY, and a praxis-ai binary built with
``--features anthropic-mcp-connector-filter`` (or PRAXIS_AI_BIN). Set
PRAXIS_TEST_REQUIRE_LIVE=1 in CI so missing infrastructure fails instead of
reporting a misleading green run.
"""

import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlparse

import pytest
from anthropic import Anthropic

from mcp_stub import McpStub, weather_result

ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = ROOT / "examples/configs/anthropic/mcp-connector.yaml"
EXAMPLE_MCP_URL = "http://127.0.0.1:9100/mcp"
# The example routes this model to the Chat Completions backend; the live model
# is substituted onto that route so the translation path is exercised.
CHAT_MODEL_ROUTE = "Qwen/Qwen3-8B"
BETA = "mcp-client-2025-11-20"


def _required_live() -> bool:
    return os.environ.get("PRAXIS_TEST_REQUIRE_LIVE", "").lower() in {"1", "true"}


def _live_config() -> tuple[str, str]:
    required = ("PRAXIS_TEST_VLLM_BASE_URL", "PRAXIS_TEST_VLLM_MODEL", "VLLM_API_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        message = f"live vLLM MCP connector SDK test requires {', '.join(missing)}"
        if _required_live():
            pytest.fail(message)
        pytest.skip(message)
    parsed = urlparse(os.environ["PRAXIS_TEST_VLLM_BASE_URL"])
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port or parsed.path not in {"", "/"}:
        pytest.fail("PRAXIS_TEST_VLLM_BASE_URL must be a local http://host:port URL")
    return f"{parsed.hostname}:{parsed.port}", os.environ["PRAXIS_TEST_VLLM_MODEL"]


def _binary() -> str:
    configured = os.environ.get("PRAXIS_AI_BIN")
    candidates = [Path(configured)] if configured else [
        ROOT / "target/debug/praxis-ai",
        ROOT / "target/release/praxis-ai",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    pytest.fail("praxis-ai binary is missing; build praxis-ai-proxy or set PRAXIS_AI_BIN")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_proxy(port: int, process: subprocess.Popen, log_path: Path) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"Praxis exited during startup:\n{log_path.read_text()}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    pytest.fail(f"Praxis did not listen on {port}:\n{log_path.read_text()}")


def _write_config(proxy_port: int, authority: str, model: str, mcp_url: str) -> str:
    """Retarget the shipped example to live vLLM and the local MCP stub."""
    config = CONFIG_PATH.read_text()
    for old, new in [
        ('address: "127.0.0.1:8080"', f'address: "127.0.0.1:{proxy_port}"'),
        (f'x-praxis-ai-model: "{CHAT_MODEL_ROUTE}"', f'x-praxis-ai-model: "{model}"'),
        ('"127.0.0.1:8001"', f'"{authority}"'),
        # Anchor on the key: the header comment's curl example repeats the URL.
        (f"server_url: {EXAMPLE_MCP_URL}", f"server_url: {mcp_url}"),
    ]:
        assert config.count(old) == 1, f"example drift: {old!r}"
        config = config.replace(old, new)
    # The ephemeral GPU runner runs as root; opt in only in this generated config.
    if os.geteuid() == 0 and "allow_root:" not in config:
        assert config.count("insecure_options:\n") == 1, CONFIG_PATH
        config = config.replace("insecure_options:\n", "insecure_options:\n  allow_root: true\n", 1)
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(config)
    return path


@pytest.fixture(scope="module")
def live_stack(request):
    """Live vLLM model + local MCP stub behind the shipped example."""
    authority, model = _live_config()
    mcp = McpStub().start()
    proxy_port = _free_port()
    config_path = _write_config(proxy_port, authority, model, mcp.url)
    fd, log_name = tempfile.mkstemp(suffix=".log")
    log_file = os.fdopen(fd, "w")
    log_path = Path(log_name)
    proc = subprocess.Popen([_binary(), "-c", config_path], stdout=log_file, stderr=subprocess.STDOUT)
    try:
        _wait_for_proxy(proxy_port, proc, log_path)
        yield {"proxy_port": proxy_port, "model": model, "mcp": mcp}
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        mcp.stop()
        if request.session.testsfailed > 0:
            print(f"\n=== Praxis logs ===\n{log_path.read_text()}", file=sys.stderr)
        os.unlink(config_path)
        log_path.unlink()


@pytest.mark.parametrize("streaming", [False, True], ids=["buffered", "streaming"])
def test_live_model_calls_mcp_tool_through_the_connector(live_stack, streaming):
    client = Anthropic(
        base_url=f"http://127.0.0.1:{live_stack['proxy_port']}",
        api_key="unused-by-the-chat-route",
        max_retries=0,
        timeout=180,
    )
    mcp = live_stack["mcp"]

    mcp.requests.clear()
    params = dict(
        model=live_stack["model"],
        max_tokens=1024,
        betas=[BETA],
        mcp_servers=[{"type": "url", "url": mcp.url, "name": "weather"}],
        tools=[
            {
                "type": "mcp_toolset",
                "mcp_server_name": "weather",
                "default_config": {"enabled": False},
                "configs": {"get_weather": {"enabled": True}},
            }
        ],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": "What is the weather in Paris? Use the weather tool. /no_think"}],
    )

    if streaming:
        with client.beta.messages.stream(**params) as stream:
            response = stream.get_final_message()
    else:
        response = client.beta.messages.create(**params)
    assert response.type == "message", response
    types = [block.type for block in response.content]
    assert "mcp_tool_use" in types and "mcp_tool_result" in types, response.model_dump_json()
    tool_use = next(block for block in response.content if block.type == "mcp_tool_use")
    tool_result = next(block for block in response.content if block.type == "mcp_tool_result")
    assert tool_use.name == "get_weather"
    assert tool_use.server_name == "weather"
    assert isinstance(tool_use.input.get("city"), str) and tool_use.input["city"].strip()
    assert tool_result.tool_use_id == tool_use.id
    assert tool_result.is_error is False
    assert tool_result.content[0].text == weather_result(tool_use.input)
    assert all(block.type != "tool_use" for block in response.content), response.model_dump_json()

    calls = mcp.calls("tools/call")
    assert calls, "the MCP server must receive the tool call"
    assert all(call["params"]["name"] == "get_weather" for call in calls)
    assert mcp.calls("tools/list"), "Praxis lists the server's tools first"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
