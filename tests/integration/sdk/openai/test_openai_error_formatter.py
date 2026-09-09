#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "openai>=1.0",
#     "pytest>=8.0",
# ]
# ///
"""
OpenAI SDK compatibility tests for fatal proxy error response formatting.

Starts a Praxis proxy pointing to an unavailable upstream, then sends
requests via the official OpenAI Python SDK to verify that fatal proxy
errors (connection refusal and gateway timeout) are returned in the
native OpenAI {"error": {...}} format and correctly parsed by the SDK into
standard APIStatusError exceptions.
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
from openai import APIStatusError, OpenAI


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
      - filter: openai_responses_format
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
      - filter: openai_responses_format
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
def openai_client(proxy_ports):
    return OpenAI(
        api_key="not-needed",
        base_url=f"http://127.0.0.1:{proxy_ports['refused_port']}/v1",
        max_retries=0,
        timeout=10.0,
    )


@pytest.fixture(scope="module")
def timeout_openai_client(proxy_ports):
    return OpenAI(
        api_key="not-needed",
        base_url=f"http://127.0.0.1:{proxy_ports['timeout_port']}/v1",
        max_retries=0,
        timeout=10.0,
    )


class TestOpenAIErrorFormatter:
    def test_chat_completions_connection_refused(self, openai_client):
        with pytest.raises(APIStatusError) as exc_info:
            openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "hello"}],
            )
        err = exc_info.value
        assert err.status_code == 502
        assert err.code == "upstream_connect_refused"
        assert err.type == "server_error"
        assert err.body.get("param") is None
        assert "Upstream connection refused" in err.body.get("message", "")

    def test_responses_connection_refused(self, openai_client):
        with pytest.raises(APIStatusError) as exc_info:
            openai_client.responses.create(
                model="gpt-4.1",
                input="hello",
            )
        err = exc_info.value
        assert err.status_code == 502
        assert err.code == "upstream_connect_refused"
        assert err.type == "server_error"
        assert err.body.get("param") is None
        assert "Upstream connection refused" in err.body.get("message", "")

    def test_chat_completions_gateway_timeout(self, timeout_openai_client):
        with pytest.raises(APIStatusError) as exc_info:
            timeout_openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "hello"}],
            )
        err = exc_info.value
        assert err.status_code == 504
        assert err.code == "upstream_read_timeout"
        assert err.type == "server_error"
        assert err.body.get("param") is None
        assert "Upstream read timed out" in err.body.get("message", "")

    def test_responses_gateway_timeout(self, timeout_openai_client):
        with pytest.raises(APIStatusError) as exc_info:
            timeout_openai_client.responses.create(
                model="gpt-4.1",
                input="hello",
            )
        err = exc_info.value
        assert err.status_code == 504
        assert err.code == "upstream_read_timeout"
        assert err.type == "server_error"
        assert err.body.get("param") is None
        assert "Upstream read timed out" in err.body.get("message", "")


if __name__ == "__main__":
    pytest.main([__file__, *sys.argv[1:]])
