#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "anthropic==1.9.0",
#     "pytest>=8.0",
# ]
# ///
"""Official Anthropic SDK -> Praxis -> real vLLM Messages acceptance.

The native and translated production examples are both exercised against the
same keyed vLLM server. Qwen3-8B runs the text matrix; the image case requires
a separately served vision model and is selected with ``-k image``.

Required: PRAXIS_TEST_VLLM_BASE_URL (local HTTP URL), PRAXIS_TEST_VLLM_MODEL,
VLLM_API_KEY, and a built praxis-ai binary (or PRAXIS_AI_BIN). Set
PRAXIS_TEST_REQUIRE_LIVE=1 in CI so missing infrastructure fails instead of
reporting a misleading green run.
"""

import base64
import os
from pathlib import Path
import secrets
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlparse
import zlib

import pytest
from anthropic import Anthropic


ROOT = Path(__file__).resolve().parents[4]
EXAMPLES = {
    "native": ROOT / "examples/configs/anthropic/messages-native-vllm.yaml",
    "translated": ROOT / "examples/configs/anthropic/messages-to-openai-vllm.yaml",
}
ROUTES = tuple(EXAMPLES)


def _required_live() -> bool:
    return os.environ.get("PRAXIS_TEST_REQUIRE_LIVE", "").lower() in {"1", "true"}


def _live_config() -> tuple[str, str, str]:
    required = ("PRAXIS_TEST_VLLM_BASE_URL", "PRAXIS_TEST_VLLM_MODEL", "VLLM_API_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        message = f"live vLLM SDK test requires {', '.join(missing)}"
        if _required_live():
            pytest.fail(message)
        pytest.skip(message)

    parsed = urlparse(os.environ["PRAXIS_TEST_VLLM_BASE_URL"])
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port or parsed.path not in {"", "/"}:
        pytest.fail("PRAXIS_TEST_VLLM_BASE_URL must be a local http://host:port URL")
    return f"{parsed.hostname}:{parsed.port}", os.environ["PRAXIS_TEST_VLLM_MODEL"], os.environ["VLLM_API_KEY"]


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


@pytest.fixture(scope="module")
def live_clients():
    authority, model, backend_key = _live_config()
    binary = _binary()
    password = secrets.token_hex(24)
    gateway_auth = "Basic " + base64.b64encode(f"gateway:{password}".encode()).decode()
    clients = {}
    processes = []

    with tempfile.TemporaryDirectory(prefix="praxis-anthropic-sdk-") as temporary:
        directory = Path(temporary)
        try:
            for route, example in EXAMPLES.items():
                port = _free_port()
                config = example.read_text()
                assert config.count('address: "127.0.0.1:8080"') == 1, example
                assert config.count('"127.0.0.1:8000"') == 1, example
                config = config.replace('address: "127.0.0.1:8080"', f'address: "127.0.0.1:{port}"')
                config = config.replace('"127.0.0.1:8000"', f'"{authority}"')
                # The ephemeral GPU runner runs as root. Match the existing
                # OpenAI SDK harness by opting in only in this generated config.
                if os.geteuid() == 0:
                    assert config.count("insecure_options:\n") == 1, example
                    config = config.replace("insecure_options:\n", "insecure_options:\n  allow_root: true\n", 1)
                config_path = directory / f"{route}.yaml"
                config_path.write_text(config)
                log_path = directory / f"{route}.log"
                environment = os.environ.copy()
                environment["GATEWAY_AUTH_PASSWORD"] = password
                environment["VLLM_API_KEY"] = backend_key
                log = log_path.open("w")
                try:
                    process = subprocess.Popen(
                        [binary, "-c", str(config_path)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=environment,
                    )
                finally:
                    log.close()
                processes.append(process)
                _wait_for_proxy(port, process, log_path)
                clients[route] = Anthropic(
                    base_url=f"http://127.0.0.1:{port}",
                    api_key="sdk-client-key-must-not-reach-vllm",
                    default_headers={"Authorization": gateway_auth},
                    max_retries=0,
                    timeout=180,
                )
            yield clients, model
        finally:
            for client in clients.values():
                client.close()
            for process in processes:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


@pytest.mark.parametrize("route", ROUTES)
def test_basic_message_and_usage(live_clients, route):
    clients, model = live_clients
    response = clients[route].messages.create(
        model=model,
        max_tokens=64,
        system="Reply briefly.",
        messages=[{"role": "user", "content": "Say hello. /no_think"}],
    )
    assert response.type == "message"
    assert response.role == "assistant"
    assert response.model == model
    assert any(block.type == "text" and block.text.strip() for block in response.content)
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0


@pytest.mark.parametrize("route", ROUTES)
def test_streaming_text(live_clients, route):
    clients, model = live_clients
    event_types = []
    with clients[route].messages.stream(
        model=model,
        max_tokens=64,
        messages=[{"role": "user", "content": "Say hello. /no_think"}],
    ) as stream:
        for event in stream:
            event_types.append(event.type)
        message = stream.get_final_message()
    assert event_types[0] == "message_start"
    assert event_types[-1] == "message_stop"
    assert "content_block_delta" in event_types[1:-1]
    assert any(block.type == "text" and block.text.strip() for block in message.content)


@pytest.mark.parametrize("route", ROUTES)
def test_count_tokens(live_clients, route):
    # The translated example leaves this path intact. It succeeds here because
    # the shared vLLM instance also serves native /v1/messages/count_tokens.
    clients, model = live_clients
    count = clients[route].messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": "Count these input tokens."}],
    )
    assert count.input_tokens > 0


@pytest.mark.parametrize("route", ROUTES)
def test_tool_round_trip(live_clients, route):
    clients, model = live_clients
    tool = {
        "name": "get_weather",
        "description": "Return a city's temperature",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
    user = {
        "role": "user",
        "content": "Use get_weather for Paris before answering. Do not guess the temperature. /no_think",
    }
    first = clients[route].messages.create(
        model=model,
        max_tokens=128,
        messages=[user],
        tools=[tool],
        tool_choice={"type": "auto"},
    )
    calls = [block for block in first.content if block.type == "tool_use"]
    assert first.stop_reason == "tool_use", first.model_dump_json()
    assert len(calls) == 1, first.model_dump_json()
    call = calls[0]
    assert call.name == "get_weather"
    assert call.id
    assert isinstance(call.input, dict)
    assert "paris" in str(call.input.get("city", "")).lower()

    assistant_content = [block.model_dump(exclude_none=True) for block in first.content]
    second = clients[route].messages.create(
        model=model,
        max_tokens=128,
        messages=[
            user,
            {"role": "assistant", "content": assistant_content},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": call.id, "content": "17 degrees Celsius"}
                ],
            },
        ],
        tools=[tool],
        tool_choice={"type": "none"},
    )
    final_text = " ".join(block.text for block in second.content if block.type == "text")
    assert "17" in final_text, second.model_dump_json()
    assert all(block.type != "tool_use" for block in second.content)


@pytest.mark.parametrize("route", ROUTES)
def test_named_tool_choice_stop_reason(live_clients, route):
    clients, model = live_clients
    first = clients[route].messages.create(
        model=model,
        max_tokens=128,
        messages=[{"role": "user", "content": "Call get_weather for Paris. /no_think"}],
        tools=[
            {
                "name": "get_weather",
                "description": "Return a city's temperature",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        tool_choice={"type": "tool", "name": "get_weather"},
    )
    calls = [block for block in first.content if block.type == "tool_use"]
    assert len(calls) == 1, first.model_dump_json()
    if first.stop_reason == "end_turn":
        pytest.xfail("pinned vLLM v0.29.0 maps a named tool call to end_turn")
    assert first.stop_reason == "tool_use", first.model_dump_json()


def _solid_png(rgb: tuple[int, int, int]) -> bytes:
    """Generate a tiny deterministic image without a checked-in binary fixture."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload))

    width = height = 224
    row = b"\x00" + bytes(rgb) * width
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * height))
        + chunk(b"IEND", b"")
    )


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("color,rgb", [("red", (255, 0, 0)), ("blue", (0, 0, 255))])
def test_image_message(live_clients, route, color, rgb):
    # Run with -k image only against a vLLM-served vision model. The default
    # Qwen3-8B text matrix must never be represented as multimodal coverage.
    clients, model = live_clients
    image = base64.b64encode(_solid_png(rgb)).decode("ascii")
    response = clients[route].messages.create(
        model=model,
        max_tokens=64,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image}},
                    {"type": "text", "text": "What is the dominant color? Answer in one word."},
                ],
            }
        ],
    )
    text = " ".join(block.text for block in response.content if block.type == "text")
    assert color in text.lower()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
