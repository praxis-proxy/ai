#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "anthropic>=0.40",
#     "pytest>=8.0",
# ]
# ///
"""
Anthropic SDK compatibility tests for fatal proxy error response formatting.

Starts a Praxis proxy pointing to an unavailable upstream, then sends
requests via the official Anthropic Python SDK to verify that fatal proxy
errors (connection refusal and gateway timeout) are returned in the
native Anthropic {"type": "error", "error": {...}} format and correctly
parsed by the SDK into standard APIStatusError exceptions.
"""

import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from anthropic import APIStatusError, Anthropic


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _find_binary() -> str:
    if "PRAXIS_AI_BIN" in os.environ:
        if os.path.isfile(os.environ["PRAXIS_AI_BIN"]):
            return os.environ["PRAXIS_AI_BIN"]
        raise FileNotFoundError(
            f"PRAXIS_AI_BIN={os.environ['PRAXIS_AI_BIN']!r} not found"
        )
    for candidate in [
        "target/debug/praxis-ai",
        "target/release/praxis-ai",
        "target/debug/praxis",
        "target/release/praxis",
    ]:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "praxis binary not found — run `cargo build -p praxis-ai-proxy` first"
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


class _SlowHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        time.sleep(1.0)
        try:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def proxy_ports():
    dead_port = _free_port()
    slow_port = _free_port()
    refused_port = _free_port()
    timeout_port = _free_port()

    slow_server = HTTPServer(("127.0.0.1", slow_port), _SlowHandler)
    server_thread = threading.Thread(target=slow_server.serve_forever, daemon=True)
    server_thread.start()

    config = f"""
listeners:
  - name: refused_listener
    address: "127.0.0.1:{refused_port}"
    filter_chains: [classify_refused]
  - name: timeout_listener
    address: "127.0.0.1:{timeout_port}"
    filter_chains: [classify_timeout]

filter_chains:
  - name: classify_refused
    filters:
      - filter: anthropic_messages_format
        on_invalid: continue
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: dead_upstream
      - filter: load_balancer
        clusters:
          - name: dead_upstream
            endpoints:
              - "127.0.0.1:{dead_port}"

  - name: classify_timeout
    filters:
      - filter: anthropic_messages_format
        on_invalid: continue
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: slow_upstream
      - filter: load_balancer
        clusters:
          - name: slow_upstream
            read_timeout_ms: 200
            endpoints:
              - "127.0.0.1:{slow_port}"

insecure_options:
  allow_private_endpoints: true
"""
    cfg_fd, cfg_path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(cfg_fd, "w") as f:
        f.write(config)

    binary = _find_binary()
    proc = subprocess.Popen(
        [binary, "-c", cfg_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        _wait_for_proxy(refused_port)
        _wait_for_proxy(timeout_port)
        yield {"refused_port": refused_port, "timeout_port": timeout_port}
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        slow_server.shutdown()
        os.unlink(cfg_path)


@pytest.fixture(scope="module")
def anthropic_client(proxy_ports):
    return Anthropic(
        api_key="not-needed",
        base_url=f"http://127.0.0.1:{proxy_ports['refused_port']}",
        max_retries=0,
        timeout=10.0,
    )


@pytest.fixture(scope="module")
def timeout_anthropic_client(proxy_ports):
    return Anthropic(
        api_key="not-needed",
        base_url=f"http://127.0.0.1:{proxy_ports['timeout_port']}",
        max_retries=0,
        timeout=10.0,
    )


class TestAnthropicErrorFormatter:
    def test_messages_connection_refused(self, anthropic_client):
        with pytest.raises(APIStatusError) as exc_info:
            anthropic_client.messages.create(
                model="claude-opus-4-8",
                max_tokens=100,
                messages=[{"role": "user", "content": "hello"}],
            )
        err = exc_info.value
        assert err.status_code == 502
        assert err.body.get("type") == "error"
        error_obj = err.body.get("error", {})
        assert error_obj.get("type") == "api_error"
        assert "Upstream connection refused" in error_obj.get("message", "")
        assert err.body.get("request_id") is not None

    def test_messages_connection_refused_with_request_id(self, anthropic_client):
        custom_id = "req_sdk_custom_9999"
        with pytest.raises(APIStatusError) as exc_info:
            anthropic_client.messages.create(
                model="claude-opus-4-8",
                max_tokens=100,
                messages=[{"role": "user", "content": "hello"}],
                extra_headers={"x-request-id": custom_id},
            )
        err = exc_info.value
        assert err.status_code == 502
        assert err.body.get("type") == "error"
        error_obj = err.body.get("error", {})
        assert error_obj.get("type") == "api_error"
        assert err.body.get("request_id") == custom_id

    def test_messages_gateway_timeout(self, timeout_anthropic_client):
        with pytest.raises(APIStatusError) as exc_info:
            timeout_anthropic_client.messages.create(
                model="claude-opus-4-8",
                max_tokens=100,
                messages=[{"role": "user", "content": "hello"}],
            )
        err = exc_info.value
        assert err.status_code == 504
        assert err.body.get("type") == "error"
        error_obj = err.body.get("error", {})
        assert error_obj.get("type") == "timeout_error"
        assert "Upstream read timed out" in error_obj.get("message", "")
        assert err.body.get("request_id") is not None


if __name__ == "__main__":
    pytest.main([__file__, *sys.argv[1:]])
