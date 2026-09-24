#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.27,<1",
#     "openai>=2.0,<3",
#     "pytest>=8.0,<9",
# ]
# ///
"""OpenAI Python SDK tests for the Bedrock Converse translation example.

The test starts a strict local Bedrock Runtime simulator and Praxis with
``examples/configs/bedrock/chat-completions-to-converse.yaml``. It verifies
that official OpenAI SDK calls work for finite and streaming completions.

Usage:
    cargo build -p praxis-ai-proxy
    uv run tests/integration/sdk/openai/test_bedrock_converse_chat_completions.py -s
"""

import json
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from openai import OpenAI

MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
CONFIG_PATH = "examples/configs/bedrock/chat-completions-to-converse.yaml"


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
    for candidate in ["target/debug/praxis-ai", "target/release/praxis-ai"]:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "praxis-ai binary not found — run `cargo build -p praxis-ai-proxy` first"
    )


def _eventstream_frame(event_type: str, payload: dict) -> bytes:
    headers = b"".join(
        _eventstream_string_header(name, value)
        for name, value in [
            (":message-type", "event"),
            (":event-type", event_type),
            (":content-type", "application/json"),
        ]
    )
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode()
    total_length = 16 + len(headers) + len(payload_bytes)
    prelude = struct.pack(">II", total_length, len(headers))
    prelude_with_crc = prelude + struct.pack(">I", zlib.crc32(prelude))
    message = prelude_with_crc + headers + payload_bytes
    return message + struct.pack(">I", zlib.crc32(message))


def _eventstream_string_header(name: str, value: str) -> bytes:
    name_bytes = name.encode()
    value_bytes = value.encode()
    return (
        bytes([len(name_bytes)])
        + name_bytes
        + b"\x07"
        + struct.pack(">H", len(value_bytes))
        + value_bytes
    )


def _stream_body() -> bytes:
    events = [
        ("messageStart", {"role": "assistant"}),
        ("contentBlockStart", {"contentBlockIndex": 0, "start": {}}),
        (
            "contentBlockDelta",
            {
                "contentBlockIndex": 0,
                "delta": {"text": "Paris is the capital of France."},
            },
        ),
        ("contentBlockStop", {"contentBlockIndex": 0}),
        ("messageStop", {"stopReason": "end_turn"}),
        (
            "metadata",
            {
                "usage": {"inputTokens": 12, "outputTokens": 7, "totalTokens": 19},
                "metrics": {"latencyMs": 1},
            },
        ),
    ]
    return b"".join(_eventstream_frame(event_type, payload) for event_type, payload in events)


class BedrockSimulatorHandler(BaseHTTPRequestHandler):
    server_version = "StrictBedrockSimulator/1.0"

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        try:
            request = json.loads(body)
            self._validate_request(request)
        except (AssertionError, json.JSONDecodeError) as exc:
            self._send_json(400, {"message": f"invalid Converse request: {exc}"})
            return

        if self.path.endswith("/converse-stream"):
            response = _stream_body()
            self.send_response(200)
            self.send_header("Content-Type", "application/vnd.amazon.eventstream")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            for offset in range(0, len(response), 17):
                self.wfile.write(response[offset : offset + 17])
                self.wfile.flush()
            return

        assert self.path.endswith("/converse"), self.path
        self._send_json(
            200,
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "Paris is the capital of France."}],
                    }
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 12, "outputTokens": 7, "totalTokens": 19},
                "metrics": {"latencyMs": 1},
            },
        )

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _validate_request(self, request: dict) -> None:
        expected_base = f"/model/{MODEL}/"
        assert self.path in {
            f"{expected_base}converse",
            f"{expected_base}converse-stream",
        }, self.path
        assert self.headers.get("Authorization", "").startswith("AWS4-HMAC-SHA256 ")
        assert self.headers.get("X-Amz-Date")
        assert self.headers.get("X-Amz-Content-Sha256")
        assert "model" not in request
        assert request["messages"] == [
            {
                "role": "user",
                "content": [{"text": "What is the capital of France?"}],
            }
        ]


def _write_config(proxy_port: int, backend_port: int) -> str:
    with open(CONFIG_PATH, encoding="utf-8") as config_file:
        config = config_file.read()
    config = config.replace("0.0.0.0:8080", f"127.0.0.1:{proxy_port}")
    config = config.replace("127.0.0.1:3000", f"127.0.0.1:{backend_port}")
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as config_file:
        config_file.write(config)
    return path


def _wait_for_proxy(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"proxy did not start within {timeout}s")


@pytest.fixture(scope="session")
def bedrock_gateway():
    backend = ThreadingHTTPServer(("127.0.0.1", 0), BedrockSimulatorHandler)
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()

    proxy_port = _free_port()
    config_path = _write_config(proxy_port, backend.server_port)
    log_file = tempfile.NamedTemporaryFile(prefix="praxis-bedrock-", suffix=".log")
    env = os.environ.copy()
    env["AWS_ACCESS_KEY_ID"] = "TESTACCESSKEY"
    env["AWS_SECRET_ACCESS_KEY"] = "test-secret-key"
    proc = subprocess.Popen(
        [_find_binary(), "-c", config_path],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_proxy(proxy_port)
        yield proxy_port
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        backend.shutdown()
        backend.server_close()
        backend_thread.join(timeout=5)
        os.unlink(config_path)
        log_file.close()


@pytest.fixture(scope="session")
def openai_client(bedrock_gateway):
    return OpenAI(
        api_key="client-key-must-not-reach-bedrock",
        base_url=f"http://127.0.0.1:{bedrock_gateway}/v1",
        max_retries=0,
        timeout=10.0,
    )


def test_finite_chat_completion(openai_client: OpenAI) -> None:
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    assert response.object == "chat.completion"
    assert response.model == MODEL
    assert response.choices[0].message.content == "Paris is the capital of France."
    assert response.choices[0].finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.total_tokens == 19


def test_streaming_chat_completion(openai_client: OpenAI) -> None:
    stream = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = list(stream)

    text = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    finish_reasons = [chunk.choices[0].finish_reason for chunk in chunks if chunk.choices]
    usage = [chunk.usage for chunk in chunks if chunk.usage is not None]
    assert text == "Paris is the capital of France."
    assert "stop" in finish_reasons
    assert usage[-1].total_tokens == 19


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
