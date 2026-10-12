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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from anthropic import Anthropic, APIError, RateLimitError


@pytest.fixture(scope="module")
def vertex_client():
    from conftest import find_binary, free_port, wait_for_proxy

    class Backend(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if request.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"event: ping\ndata: {\"type\":\"ping\"}\n\ndata: " + b"x" * 512 + b"\n\n")
            else:
                body = json.dumps({"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}}).encode()
                self.send_response(429)
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
    config = config.replace("project: my-gcp-project", "project: my-gcp-project\n        max_body_bytes: 256")
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
        vertex_client.messages.create(model="vertex/claude-sonnet-4-5", max_tokens=8, messages=[{"role": "user", "content": "hi"}])
    assert raised.value.status_code == 429
    assert raised.value.body["error"]["type"] == "rate_limit_error"
    assert "RESOURCE_EXHAUSTED" in raised.value.body["error"]["message"]


def test_oversized_sse_frame_raises_sdk_error(vertex_client):
    with pytest.raises(APIError, match="SSE frame size limit"):
        with vertex_client.messages.create(model="vertex/claude-sonnet-4-5", max_tokens=8, messages=[{"role": "user", "content": "hi"}], stream=True) as stream:
            list(stream)


if __name__ == "__main__":
    pytest.main([__file__, *sys.argv[1:]])
