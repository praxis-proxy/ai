#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["anthropic>=0.40", "pytest>=8.0"]
# ///
"""Anthropic SDK compatibility for Vertex error and streaming translation."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from anthropic import Anthropic, APIError, RateLimitError


@pytest.fixture(scope="module")
def vertex_client():
    from conftest import find_binary, free_port, wait_for_proxy

    class Backend(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/models/count-tokens:rawPredict"):
                assert request["model"] == "claude-sonnet-4-5@20250929"
                assert "anthropic_version" not in request
                self.send_json(200, {"input_tokens": 5})
                return
            assert self.path.endswith(("/models/claude-sonnet-4-5@20250929:rawPredict", "/models/claude-sonnet-4-5@20250929:streamRawPredict"))
            assert "model" not in request
            assert request["anthropic_version"] == "vertex-2023-10-16"
            assert self.headers.get("x-api-key") is None
            assert self.headers.get("Accept-Encoding") is None
            if request.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                if request["messages"][0]["content"] == "successful stream":
                    events = [
                        {"type": "message_start", "message": {
                            "id": "msg_vertex_sdk", "type": "message", "role": "assistant",
                            "model": "claude-sonnet-4-5-20250929", "content": [],
                            "stop_reason": None, "stop_sequence": None,
                            "usage": {"input_tokens": 1, "output_tokens": 0},
                        }},
                        {"type": "content_block_start", "index": 0,
                         "content_block": {"type": "text", "text": ""}},
                        {"type": "content_block_delta", "index": 0,
                         "delta": {"type": "text_delta", "text": "hello"}},
                        {"type": "content_block_stop", "index": 0},
                        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                         "usage": {"output_tokens": 1}},
                        {"type": "message_stop"},
                    ]
                    for event in events:
                        frame = f"id: sdk-event\nretry: 1000\nevent: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
                        if event["type"] == "message_start":
                            split = frame.index(b"claude-sonnet") + len(b"claude-son")
                            self.wfile.write(frame[:split])
                            self.wfile.flush()
                            time.sleep(0.03)
                            self.wfile.write(frame[split:])
                        else:
                            self.wfile.write(frame)
                else:
                    self.wfile.write(b"event: ping\ndata: {\"type\":\"ping\"}\n\ndata: " + b"x" * 1024 + b"\n\n")
            elif request["messages"][0]["content"] == "successful response":
                self.send_json(200, {
                    "id": "msg_vertex_json", "type": "message", "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [{"type": "text", "text": "hello"}],
                    "stop_reason": "end_turn", "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                })
            else:
                self.send_json(429, {"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}})

        def send_json(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = free_port()
    config = open("examples/configs/vertex/messages-to-anthropic.yaml").read()
    config = config.replace("0.0.0.0:8080", f"127.0.0.1:{port}")
    config = config.replace("127.0.0.1:3000", f"127.0.0.1:{server.server_port}")
    config = config.replace("project: my-gcp-project", "project: my-gcp-project\n        max_body_bytes: 512")
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "config.yaml")
        with open(path, "w") as output:
            output.write(config)
        process = subprocess.Popen([find_binary(), "-c", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait_for_proxy(port)
            with Anthropic(api_key="client-key", base_url=f"http://127.0.0.1:{port}", max_retries=0) as client:
                yield client
        finally:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            server.shutdown()
            server.server_close()


def test_google_quota_error_is_anthropic_rate_limit_error(vertex_client):
    with pytest.raises(RateLimitError) as raised:
        vertex_client.messages.create(model="public-claude", max_tokens=8, messages=[{"role": "user", "content": "hi"}])
    assert raised.value.status_code == 429
    assert raised.value.body["error"]["type"] == "rate_limit_error"
    assert "RESOURCE_EXHAUSTED" in raised.value.body["error"]["message"]


def test_oversized_sse_frame_raises_sdk_error(vertex_client):
    with pytest.raises(APIError, match="SSE frame size limit"):
        with vertex_client.messages.create(model="public-claude-snapshot", max_tokens=8, messages=[{"role": "user", "content": "hi"}], stream=True) as stream:
            list(stream)


def test_streaming_model_is_restored_for_the_sdk(vertex_client):
    with vertex_client.messages.create(
        model="public-claude-snapshot", max_tokens=8,
        messages=[{"role": "user", "content": "successful stream"}], stream=True,
    ) as stream:
        events = list(stream)
    assert events[0].type == "message_start"
    assert events[0].message.model == "public-claude-snapshot"
    assert events[2].delta.text == "hello"
    assert events[-1].type == "message_stop"


def test_json_model_is_restored_for_the_sdk(vertex_client):
    message = vertex_client.messages.create(
        model="public-claude", max_tokens=8,
        messages=[{"role": "user", "content": "successful response"}],
    )
    assert message.model == "public-claude"
    assert message.content[0].text == "hello"


def test_count_tokens_uses_the_exact_mapped_snapshot(vertex_client):
    result = vertex_client.messages.count_tokens(
        model="public-claude-snapshot", messages=[{"role": "user", "content": "hi"}],
    )
    assert result.input_tokens == 5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
