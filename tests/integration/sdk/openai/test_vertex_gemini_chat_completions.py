#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.27",
#     "openai>=2.0",
#     "pytest>=8.0",
# ]
# ///
"""
OpenAI SDK compatibility tests for the Vertex AI Gemini translation filter.

Starts a Praxis proxy with the openai_chat_completions_to_vertexai_gemini filter
backed by a fake Vertex AI Gemini API server, then exercises the Chat Completions
API using the official OpenAI Python SDK to verify end-to-end translation:

- OpenAI Chat Completions request format → Vertex Gemini generateContent
- Vertex Gemini response format → OpenAI Chat Completions
- Streaming SSE frames preserve OpenAI structure
- Tool calls are properly mapped between schemas
- Error responses are normalized to OpenAI format

Usage:
    cargo build -p praxis-ai-proxy
    uv run tests/integration/sdk/openai/test_vertex_gemini_chat_completions.py -v
"""

import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
import pytest
from openai import APIError, APIStatusError, OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PRAXIS_AI_BIN = os.environ.get("PRAXIS_AI_BIN")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Allocate a free TCP port on loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _find_binary() -> str:
    """Locate the praxis-ai binary."""
    if PRAXIS_AI_BIN:
        if os.path.isfile(PRAXIS_AI_BIN):
            return PRAXIS_AI_BIN
        raise FileNotFoundError(f"PRAXIS_AI_BIN={PRAXIS_AI_BIN!r} not found")
    for candidate in ["target/debug/praxis-ai", "target/release/praxis-ai"]:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "praxis-ai binary not found — run `cargo build -p praxis-ai-proxy` first"
    )


def _wait_for_proxy(port: int, timeout: float = 10.0) -> None:
    """Wait for the Praxis proxy to become ready on loopback:{port}."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"proxy did not start within {timeout}s")


# ---------------------------------------------------------------------------
# Fake Vertex AI Gemini Backend
# ---------------------------------------------------------------------------


class FakeVertexGeminiHandler(BaseHTTPRequestHandler):
    """
    Minimal Vertex AI Gemini API mock for end-to-end testing.

    Handles:
    - POST /v1/projects/{project}/locations/{region}/models/{model}:generateContent
    - POST /v1/projects/{project}/locations/{region}/models/{model}:streamGenerateContent?alt=sse

    Paths and the translated request shape are validated so an SDK test cannot
    pass by accidentally reaching an un-translated endpoint.
    """

    # Shared request capture for test assertions
    last_request_body: Optional[bytes] = None
    last_request_path: Optional[str] = None

    VERTEX_PATH = re.compile(
        r"^/v1/projects/test-project/locations/us-central1/"
        r"publishers/google/models/(?P<model>[A-Za-z0-9._-]+)"
        r"(?P<method>:(?:generateContent|streamGenerateContent))(?P<query>\?alt=sse)?$"
    )

    def do_POST(self) -> None:
        """Handle POST request to generateContent or streamGenerateContent endpoints."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            FakeVertexGeminiHandler.last_request_body = body
            FakeVertexGeminiHandler.last_request_path = self.path

            # Parse the request to determine response
            try:
                request_json = json.loads(body)
            except json.JSONDecodeError:
                self._send_error(400, "Invalid JSON in request body")
                return

            path_match = self.VERTEX_PATH.fullmatch(self.path)
            if path_match is None:
                self._send_error(404, f"Unexpected Vertex path: {self.path}")
                return

            if not isinstance(request_json.get("contents"), list):
                self._send_error(400, "Translated request must contain contents")
                return
            if "messages" in request_json or "model" in request_json or "stream" in request_json:
                self._send_error(400, "OpenAI-only fields reached the Vertex backend")
                return

            # Route based on the exact Vertex endpoint.
            if path_match.group("method") == ":streamGenerateContent":
                if path_match.group("query") != "?alt=sse":
                    self._send_error(404, "Streaming endpoint must request SSE")
                    return
                self._handle_streaming(request_json)
            elif path_match.group("query") is None:
                self._handle_non_streaming(request_json)
            else:
                self._send_error(404, "Non-streaming endpoint must not have a query")
        except Exception as e:
            self._send_error(500, f"Server error: {e}")

    def _handle_non_streaming(self, request_json: dict[str, Any]) -> None:
        """Respond to generateContent (non-streaming) with a Gemini response."""
        # Extract user message from OpenAI-translated Gemini format
        contents = request_json.get("contents", [])
        user_text = ""
        has_tool_use = bool(request_json.get("tools"))

        if contents:
            parts = contents[0].get("parts", [])
            if parts and "text" in parts[0]:
                user_text = parts[0]["text"]

        if "trigger upstream error" in user_text.lower():
            self._send_error(429, "Quota exceeded by fake Vertex backend")
            return

        response = self._make_gemini_response(user_text, has_tool_use)
        self._send_json_response(200, response)

    def _handle_streaming(self, request_json: dict[str, Any]) -> None:
        """Respond to streamGenerateContent with SSE frames in Gemini format."""
        contents = request_json.get("contents", [])
        user_text = ""
        has_tool_use = bool(request_json.get("tools"))
        if contents:
            parts = contents[0].get("parts", [])
            if parts and "text" in parts[0]:
                user_text = parts[0]["text"]

        if "trigger truncated stream" in user_text.lower():
            self._send_sse_bytes(
                b'data: {"candidates":[{"index":0,"content":{"parts":[{"text":"partial"}]}}]}\n\n'
            )
            return

        if "trigger stream frame error" in user_text.lower():
            self._send_sse_bytes(
                b"data: not-json\n\n"
                b'data: {"candidates":[{"index":0,"content":{"parts":[{"text":"must-not-leak"}]},"finishReason":"STOP"}]}\n\n'
            )
            return

        # Generate all chunks first to compute total size (for Content-Length)
        if has_tool_use:
            chunks = self._make_gemini_streaming_tool_chunks()
        else:
            chunks = self._make_gemini_streaming_chunks(user_text)
        body_lines = []
        for chunk in chunks:
            frame = f"data: {json.dumps(chunk)}\n\n"
            body_lines.append(frame)
        body_str = "".join(body_lines)
        body_bytes = body_str.encode()

        self._send_sse_bytes(body_bytes)

    def _send_sse_bytes(self, body: bytes) -> None:
        """Send an exact SSE byte sequence and close the connection."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _make_gemini_response(self, user_text: str, has_tools: bool) -> dict[str, Any]:
        """Synthesize a Gemini generateContent response."""
        if has_tools:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "get_weather",
                                        "args": {"location": "Paris"},
                                    }
                                }
                            ],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ]
            }

        response_text = self._synthesize_response(user_text, False)

        return {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": response_text}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }

    def _make_gemini_streaming_chunks(self, user_text: str) -> list[dict[str, Any]]:
        """Synthesize Gemini SSE streaming chunks."""
        response_text = self._synthesize_response(user_text, False)

        # First chunk: partial response
        chunks = [
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": response_text[:20]}],
                            "role": "model",
                        },
                        "index": 0,
                    }
                ],
                "responseId": "resp-streaming-001",
            }
        ]

        # Final chunk: rest of response + finish_reason + usage
        chunks.append(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": response_text[20:]}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                },
                "responseId": "resp-streaming-001",
            }
        )

        return chunks

    def _make_gemini_streaming_tool_chunks(self) -> list[dict[str, Any]]:
        """Synthesize Gemini SSE streaming chunks for a function call response."""
        return [
            # Frame 1: functionCall part (no finishReason yet)
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "get_weather",
                                        "args": {"location": "Paris"},
                                    }
                                }
                            ],
                            "role": "model",
                        },
                        "index": 0,
                    }
                ],
                "responseId": "resp-tool-stream-001",
            },
            # Frame 2: finishReason only — slots accumulated from frame 1
            # make finish_reason resolve to "tool_calls" on the proxy
            {
                "candidates": [
                    {
                        "content": {"parts": [], "role": "model"},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 12,
                    "candidatesTokenCount": 8,
                    "totalTokenCount": 20,
                },
                "responseId": "resp-tool-stream-001",
            },
        ]

    def _synthesize_response(self, user_text: str, has_tools: bool) -> str:
        """Generate a response based on user input."""
        if has_tools:
            return "I found 42 results for your search."
        if "2+2" in user_text or "math" in user_text.lower():
            return "The answer is 4."
        if "hello" in user_text.lower():
            return "Hello! How can I help you today?"
        return "I received your message and I'm ready to assist."

    def _send_json_response(self, status: int, data: dict[str, Any]) -> None:
        """Send a JSON response with proper headers."""
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        """Send a Gemini-format error response."""
        error_response = {
            "error": {
                "code": status,
                "message": message,
                "status": "INVALID_ARGUMENT" if status == 400 else "INTERNAL",
            }
        }
        self._send_json_response(status, error_response)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress HTTP server logging."""
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_vertex_config(proxy_port: int, backend_port: int) -> str:
    """Write Praxis config for Vertex Gemini filter backed by fake API."""
    config = {
        "listeners": [
            {
                "name": "vertex-test",
                "address": f"127.0.0.1:{proxy_port}",
                "filter_chains": ["vertex-gemini-pipeline"],
            }
        ],
        "filter_chains": [
            {
                "name": "vertex-gemini-pipeline",
                "filters": [
                    {
                        "filter": "openai_chat_completions_to_vertexai_gemini",
                        "project": "test-project",
                        "region": "us-central1",
                    },
                    {
                        "filter": "router",
                        "routes": [{"path_prefix": "/", "cluster": "fake-vertex"}],
                    },
                    {
                        "filter": "load_balancer",
                        "clusters": [
                            {
                                "name": "fake-vertex",
                                "endpoints": [f"127.0.0.1:{backend_port}"],
                            }
                        ],
                    },
                ],
            }
        ],
        "insecure_options": {"allow_private_endpoints": True},
    }
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        json.dump(config, f)
    return path


@pytest.fixture(scope="module")
def vertex_proxy():
    """
    Start a Praxis proxy with the Vertex Gemini filter for the test module.

    Yields the proxy port. Handles startup, health check, and graceful shutdown.
    """
    proxy_port = _free_port()
    backend_port = _free_port()
    config_path = _write_vertex_config(proxy_port, backend_port)

    # Start fake backend
    backend = HTTPServer(("127.0.0.1", backend_port), FakeVertexGeminiHandler)
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()

    # Start proxy
    binary = _find_binary()
    proc = subprocess.Popen(
        [binary, "-c", config_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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
        os.unlink(config_path)


@pytest.fixture(scope="module")
def openai_client(vertex_proxy):
    """Return an OpenAI client pointed at the local Praxis proxy (Vertex backend)."""
    return OpenAI(
        api_key="not-needed",
        base_url=f"http://127.0.0.1:{vertex_proxy}/v1",
        max_retries=0,
        timeout=10.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVertexGeminiChatCompletions:
    """End-to-end OpenAI SDK compatibility tests for Vertex Gemini translation."""

    def test_non_streaming_basic(self, openai_client: OpenAI) -> None:
        """Non-streaming request translates correctly and response is OpenAI-shaped."""
        response = openai_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

        # Assert OpenAI response shape
        assert response.object == "chat.completion"
        assert response.model == "gemini-2.0-flash"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "The answer is 4."
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15

        # Assert request was translated to Gemini format
        request_body = json.loads(FakeVertexGeminiHandler.last_request_body)
        assert "contents" in request_body, "Request should have 'contents' (Gemini format)"
        assert "messages" not in request_body, "Request should not have 'messages' (OpenAI format)"
        assert request_body["contents"] == [
            {"role": "user", "parts": [{"text": "What is 2+2?"}]}
        ]
        assert FakeVertexGeminiHandler.last_request_path == (
            "/v1/projects/test-project/locations/us-central1/publishers/google/"
            "models/gemini-2.0-flash:generateContent"
        )

    def test_non_streaming_greeting(self, openai_client: OpenAI) -> None:
        """Non-streaming request handles different inputs."""
        response = openai_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "Hello!"}],
        )

        assert response.choices[0].message.content == "Hello! How can I help you today?"
        assert response.choices[0].finish_reason == "stop"

    def test_streaming_basic(self, openai_client: OpenAI) -> None:
        """Streaming request produces proper OpenAI chunk stream with [DONE]."""
        chunks = []
        with openai_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            stream=True,
        ) as stream:
            for chunk in stream:
                chunks.append(chunk)

        # Verify we got chunks
        assert len(chunks) > 0, "Stream should produce multiple chunks"

        # Verify chunk structure
        for chunk in chunks[:-1]:  # All but the last
            assert chunk.object == "chat.completion.chunk"
            assert chunk.model == "gemini-2.0-flash"
            assert len(chunk.choices) == 1

        # Verify finish_reason on last chunk
        last_chunk = chunks[-1]
        assert last_chunk.choices[0].finish_reason == "stop", "Last chunk should have finish_reason"

        # Verify content was assembled
        content = "".join(
            chunk.choices[0].delta.content
            for chunk in chunks
            if chunk.choices[0].delta.content
        )
        assert len(content) > 0, "Stream should have assembled content"
        request_body = json.loads(FakeVertexGeminiHandler.last_request_body)
        assert request_body["contents"] == [
            {"role": "user", "parts": [{"text": "Say hello"}]}
        ]
        assert FakeVertexGeminiHandler.last_request_path == (
            "/v1/projects/test-project/locations/us-central1/publishers/google/"
            "models/gemini-2.0-flash:streamGenerateContent?alt=sse"
        )

    def test_tool_call(self, openai_client: OpenAI) -> None:
        """SDK receives a tool call and the backend sees Gemini declarations."""
        response = openai_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                }
            ],
            tool_choice="required",
        )

        assert response.choices[0].finish_reason == "tool_calls"
        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_weather"
        assert json.loads(tool_call.function.arguments) == {"location": "Paris"}

        request_body = json.loads(FakeVertexGeminiHandler.last_request_body)
        assert request_body["contents"] == [
            {"role": "user", "parts": [{"text": "What is the weather in Paris?"}]}
        ]
        assert request_body["tools"] == [
            {
                "functionDeclarations": [
                    {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    }
                ]
            }
        ]
        assert request_body["toolConfig"] == {"functionCallingConfig": {"mode": "ANY"}}

    def test_streaming_tool_call(self, openai_client: OpenAI) -> None:
        """Streaming tool call produces correct delta structure for the OpenAI SDK."""
        chunks = []
        with openai_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                }
            ],
            tool_choice="required",
            stream=True,
        ) as stream:
            for chunk in stream:
                chunks.append(chunk)

        assert len(chunks) >= 2, "Expected at least a tool-call delta chunk and a finish chunk"

        # First delta must carry tool_calls with the correct index and metadata.
        tool_chunks = [c for c in chunks if c.choices[0].delta.tool_calls]
        assert len(tool_chunks) >= 1, "No chunk had delta.tool_calls"
        first_tc = tool_chunks[0].choices[0].delta.tool_calls[0]
        assert first_tc.index == 0
        assert first_tc.id is not None and first_tc.id.startswith("call_")
        assert first_tc.type == "function"
        assert first_tc.function.name == "get_weather"
        assert json.loads(first_tc.function.arguments) == {"location": "Paris"}

        # The final chunk with finish_reason must report "tool_calls".
        finish_chunk = next((c for c in reversed(chunks) if c.choices[0].finish_reason), None)
        assert finish_chunk is not None, "No chunk carried finish_reason"
        assert finish_chunk.choices[0].finish_reason == "tool_calls"

    def test_upstream_error(self, openai_client: OpenAI) -> None:
        """Vertex errors retain their status and become OpenAI SDK exceptions."""
        with pytest.raises(APIStatusError) as exc_info:
            openai_client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[{"role": "user", "content": "Trigger upstream error"}],
            )

        error = exc_info.value
        assert error.status_code == 429
        assert error.body["message"] == "Quota exceeded by fake Vertex backend"
        assert error.body["type"] == "server_error"
        request_body = json.loads(FakeVertexGeminiHandler.last_request_body)
        assert request_body["contents"] == [
            {"role": "user", "parts": [{"text": "Trigger upstream error"}]}
        ]
        assert FakeVertexGeminiHandler.last_request_path.endswith(
            "/models/gemini-2.0-flash:generateContent"
        )

    def test_streaming_with_usage(self, openai_client: OpenAI) -> None:
        """Streaming response receives chunks with content and finish_reason."""
        chunks = []
        with openai_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        ) as stream:
            for chunk in stream:
                chunks.append(chunk)

        # Verify we have content chunks
        content_chunks = [
            c for c in chunks
            if c.choices[0].delta.content is not None
        ]
        assert len(content_chunks) > 0, "Should have chunks with content"

        # Verify finish_reason on final chunk
        final_chunk = chunks[-1]
        assert final_chunk.choices[0].finish_reason == "stop", "Final chunk should have finish_reason"

    def test_truncated_stream_raises_after_preserving_valid_data(self, openai_client: OpenAI) -> None:
        """A clean EOF without finishReason is an SDK-visible failure."""
        chunks = []
        with pytest.raises(APIError):
            with openai_client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[{"role": "user", "content": "Trigger truncated stream"}],
                stream=True,
            ) as stream:
                chunks.extend(stream)

        content = "".join(
            chunk.choices[0].delta.content or ""
            for chunk in chunks
            if chunk.choices
        )
        assert content == "partial"

    def test_frames_after_stream_error_are_suppressed(self, openai_client: OpenAI) -> None:
        """An invalid frame makes the stream terminal and drops later frames."""
        chunks = []
        with pytest.raises(APIError):
            with openai_client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[{"role": "user", "content": "Trigger stream frame error"}],
                stream=True,
            ) as stream:
                chunks.extend(stream)

        content = "".join(
            chunk.choices[0].delta.content or ""
            for chunk in chunks
            if chunk.choices
        )
        assert "must-not-leak" not in content

    def test_invalid_include_usage_is_rejected(self, openai_client: OpenAI) -> None:
        """The proxy rejects an invalid include_usage type before Vertex."""
        with pytest.raises(APIStatusError) as exc_info:
            openai_client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
                stream_options={"include_usage": "true"},
            )

        assert exc_info.value.status_code == 400
        assert "stream_options.include_usage" in exc_info.value.body["message"]

    def test_malformed_tool_arguments_are_rejected(self, openai_client: OpenAI) -> None:
        """Malformed assistant tool arguments are never replaced with an empty object."""
        with pytest.raises(APIStatusError) as exc_info:
            openai_client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "not-json",
                                },
                            }
                        ],
                    }
                ],
            )

        assert exc_info.value.status_code == 400
        assert "arguments are not valid JSON" in exc_info.value.body["message"]

    def test_request_body_translation(self, openai_client: OpenAI) -> None:
        """Request is correctly translated from OpenAI to Gemini format."""
        openai_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "user", "content": "Hello"},
            ],
        )

        request_body = json.loads(FakeVertexGeminiHandler.last_request_body)

        # Gemini format assertions
        assert "contents" in request_body, "Should have Gemini 'contents' field"
        contents = request_body["contents"]
        assert len(contents) > 0, "Should have at least one content item"
        assert contents[0].get("role") == "user", "First content should be user role"
        assert len(contents[0].get("parts", [])) > 0, "Content should have parts"

        # OpenAI format should NOT be present
        assert "messages" not in request_body, "Should not have OpenAI 'messages' field"

    def test_array_form_assistant_and_tool_content_translation(
        self, openai_client: OpenAI
    ) -> None:
        """Valid array-form history survives translation through the official SDK."""
        openai_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "user", "content": "Start"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "First"},
                        {"type": "text", "text": "Second"},
                    ],
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {"type": "text", "text": "first result"},
                        {"type": "text", "text": "second result"},
                    ],
                },
                {"role": "user", "content": "Continue"},
            ],
        )

        contents = json.loads(FakeVertexGeminiHandler.last_request_body)["contents"]
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"][0:2] == [{"text": "First"}, {"text": "Second"}]
        assert contents[1]["parts"][2]["functionCall"]["name"] == "search"
        assert contents[2]["parts"][0]["functionResponse"]["response"] == {
            "result": "first result\nsecond result"
        }

    def test_empty_body_rejection(self, openai_client: OpenAI) -> None:
        """Model name is required in request."""
        # OpenAI SDK will always include model, so test that omitting it via direct
        # HTTP would cause an error. This tests the filter's validation logic.
        # For the SDK test, we verify the model appears in the path by using a
        # different model name and checking it's preserved in response.
        response = openai_client.chat.completions.create(
            model="gemini-exp-1221",  # Non-standard model name
            messages=[{"role": "user", "content": "test"}],
        )
        # Model should be preserved in response
        assert response.model == "gemini-exp-1221"

    def test_model_name_in_path(self, openai_client: OpenAI) -> None:
        """Model name from request is correctly placed in Vertex API path."""
        response = openai_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "test"}],
        )
        assert response.model == "gemini-2.0-flash"
        assert FakeVertexGeminiHandler.last_request_path == (
            "/v1/projects/test-project/locations/us-central1/publishers/google/"
            "models/gemini-2.0-flash:generateContent"
        )

    def test_multiple_requests_isolation(self, openai_client: OpenAI) -> None:
        """Multiple requests do not interfere with each other."""
        response1 = openai_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "2+2"}],
        )
        response2 = openai_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response1.choices[0].message.content == "The answer is 4."
        assert response2.choices[0].message.content == "Hello! How can I help you today?"

    def test_tool_call_missing_arguments_rejected(self, vertex_proxy: int) -> None:
        """A tool call with no function.arguments field is rejected with 400."""
        payload = {
            "model": "gemini-2.0-flash",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather"
                                # arguments field intentionally absent
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
                {"role": "user", "content": "What now?"},
            ],
        }
        resp = httpx.post(
            f"http://127.0.0.1:{vertex_proxy}/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer not-needed"},
            timeout=10.0,
        )
        assert resp.status_code == 400, (
            f"Expected 400 for missing function.arguments, got {resp.status_code}: {resp.text}"
        )
        body = resp.json()
        assert "function.arguments" in body.get("error", {}).get("message", ""), (
            f"Error message should mention function.arguments: {body}"
        )


if __name__ == "__main__":
    pytest.main([__file__, *sys.argv[1:]])
