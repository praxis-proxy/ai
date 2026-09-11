"""Shared fixtures and utilities for Python SDK integration tests."""

import os
import signal
import socket
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def find_binary() -> str:
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


def wait_for_proxy(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"proxy did not start within {timeout}s on port {port}")


class SlowHandler(BaseHTTPRequestHandler):
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


@pytest.fixture(scope="session")
def error_proxy_ports():
    dead_port = free_port()
    slow_port = free_port()
    refused_port = free_port()
    timeout_port = free_port()

    slow_server = HTTPServer(("127.0.0.1", slow_port), SlowHandler)
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
      - filter: openai_format
        on_invalid: continue
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
      - filter: openai_format
        on_invalid: continue
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

    binary = find_binary()
    proc = subprocess.Popen(
        [binary, "-c", cfg_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        wait_for_proxy(refused_port)
        wait_for_proxy(timeout_port)
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
