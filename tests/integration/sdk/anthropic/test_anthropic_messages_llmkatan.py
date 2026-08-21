#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "anthropic>=0.40",
#     "httpx>=0.27",
#     "pytest>=8.0",
# ]
# ///
"""
Anthropic Messages API integration tests against llm-katan.

Starts a Praxis proxy with the Anthropic passthrough pipeline backed
by llm-katan's echo backend, then exercises non-streaming, streaming,
multi-turn, and usage extraction using the official Anthropic Python SDK.

Usage:
    cargo build -p praxis-ai-proxy
    uv run tests/integration/sdk/anthropic/test_anthropic_messages_llmkatan.py -s -v

Environment variables:
    LLM_KATAN_BASE_URL  llm-katan base URL (required; skips when unset)
    LLM_KATAN_API_KEY   llm-katan Anthropic API key (default: llm-katan-anthropic-key)
    LLM_KATAN_MODEL     model name (default: llm-katan-echo)
    PRAXIS_AI_BIN       path to praxis-ai binary (auto-detected if unset)
"""

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlparse

import pytest
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_KATAN_BASE_URL = os.environ.get("LLM_KATAN_BASE_URL")
LLM_KATAN_API_KEY = os.environ.get(
    "LLM_KATAN_API_KEY", "llm-katan-anthropic-key"
)
LLM_KATAN_MODEL = os.environ.get("LLM_KATAN_MODEL", "llm-katan-echo")
PRAXIS_AI_BIN = os.environ.get("PRAXIS_AI_BIN")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _find_binary() -> str:
    if PRAXIS_AI_BIN:
        if os.path.isfile(PRAXIS_AI_BIN):
            return PRAXIS_AI_BIN
        raise FileNotFoundError(
            f"PRAXIS_AI_BIN={PRAXIS_AI_BIN!r} not found"
        )
    for candidate in ["target/debug/praxis-ai", "target/release/praxis-ai"]:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "praxis-ai binary not found — run `cargo build -p praxis-ai-proxy` first"
    )


def _parse_llm_katan_url() -> tuple[str, int, bool]:
    """Parse LLM_KATAN_BASE_URL into (host, port, uses_tls)."""
    parsed = urlparse(LLM_KATAN_BASE_URL)
    tls = parsed.scheme == "https"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if tls else 80)
    return host, port, tls


def _llm_katan_reachable() -> bool:
    if not LLM_KATAN_BASE_URL:
        return False
    try:
        host, port, _ = _parse_llm_katan_url()
        with socket.create_connection((host, port), timeout=5):
            return True
    except OSError:
        return False


def _write_config(proxy_port: int) -> str:
    host, port, tls = _parse_llm_katan_url()
    tls_block = f"""
            tls:
              sni: "{host}" """ if tls else ""
    # Inline config: no matching example config exists for the
    # format+validate+protocol+token_count pipeline used here.
    config = f"""\
listeners:
  - name: test
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [anthropic]

filter_chains:
  - name: anthropic
    filters:
      - filter: anthropic_messages_format
        on_invalid: continue
      - filter: anthropic_validate
      - filter: anthropic_messages_protocol
        default_version: "2023-06-01"
      - filter: token_count
        provider: anthropic
      - filter: token_usage_headers
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: llm-katan
      - filter: load_balancer
        clusters:
          - name: llm-katan
            endpoints:
              - "{host}:{port}"{tls_block}
"""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(config)
    return path


def _wait_for_proxy(port: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"Praxis did not start within {timeout}s")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def praxis_proxy(tmp_path_factory, request):
    """Start a Praxis proxy backed by llm-katan for the test session."""
    if not LLM_KATAN_BASE_URL:
        pytest.skip("LLM_KATAN_BASE_URL not set — skipping")
    if not _llm_katan_reachable():
        pytest.skip(
            f"llm-katan not reachable at {LLM_KATAN_BASE_URL} — skipping"
        )

    port = _free_port()
    config_path = _write_config(port)
    binary = _find_binary()

    log_dir = tmp_path_factory.mktemp("anthropic-logs")
    log_path = str(log_dir / "praxis.log")
    log_file = open(log_path, "w")
    started = False

    proc = subprocess.Popen(
        [binary, "-c", config_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_proxy(port)
        started = True
        yield port
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        if not started or request.session.testsfailed > 0:
            with open(log_path) as f:
                print(
                    f"\n=== Praxis logs ===\n{f.read()}",
                    file=sys.stderr,
                )
        os.unlink(config_path)


@pytest.fixture(scope="session")
def anthropic_client(praxis_proxy):
    """Return an Anthropic client pointed at the local Praxis proxy."""
    return Anthropic(
        base_url=f"http://127.0.0.1:{praxis_proxy}",
        api_key=LLM_KATAN_API_KEY,
        max_retries=0,
        timeout=180,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAnthropicMessagesLLMKatan:
    """Integration tests for Anthropic Messages API through Praxis."""

    def test_non_streaming_basic(self, anthropic_client):
        response = anthropic_client.messages.create(
            model=LLM_KATAN_MODEL,
            max_tokens=128,
            messages=[{"role": "user", "content": "Hello from praxis-ai"}],
        )

        assert response.type == "message"
        assert response.role == "assistant"
        assert len(response.content) > 0
        assert response.content[0].type == "text"
        assert len(response.content[0].text) > 0
        assert response.stop_reason == "end_turn"

    def test_non_streaming_with_system(self, anthropic_client):
        response = anthropic_client.messages.create(
            model=LLM_KATAN_MODEL,
            max_tokens=128,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "What are you?"}],
        )

        assert response.type == "message"
        assert response.role == "assistant"
        assert len(response.content) > 0
        assert response.content[0].type == "text"
        assert len(response.content[0].text) > 0

    def test_streaming_basic(self, anthropic_client):
        event_types = set()

        with anthropic_client.messages.stream(
            model=LLM_KATAN_MODEL,
            max_tokens=128,
            messages=[{"role": "user", "content": "Stream test"}],
        ) as stream:
            for event in stream:
                event_types.add(event.type)

        assert "message_start" in event_types, (
            f"should see message_start; saw: {event_types}"
        )
        assert "content_block_delta" in event_types, (
            f"should see content_block_delta; saw: {event_types}"
        )
        assert "message_stop" in event_types, (
            f"should see message_stop; saw: {event_types}"
        )

    def test_streaming_collects_full_text(self, anthropic_client):
        collected = ""

        with anthropic_client.messages.stream(
            model=LLM_KATAN_MODEL,
            max_tokens=128,
            messages=[{"role": "user", "content": "Streaming text collection"}],
        ) as stream:
            for text in stream.text_stream:
                collected += text

        assert len(collected) > 0, "streamed text should not be empty"

    def test_multi_turn(self, anthropic_client):
        response = anthropic_client.messages.create(
            model=LLM_KATAN_MODEL,
            max_tokens=128,
            messages=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice!"},
                {"role": "user", "content": "What is my name?"},
            ],
        )

        assert response.type == "message"
        assert response.role == "assistant"
        assert len(response.content) > 0
        assert response.content[0].type == "text"
        assert len(response.content[0].text) > 0

    def test_usage_present(self, anthropic_client):
        response = anthropic_client.messages.create(
            model=LLM_KATAN_MODEL,
            max_tokens=128,
            messages=[{"role": "user", "content": "Usage test"}],
        )

        assert response.usage is not None, "usage should be present"
        assert response.usage.input_tokens > 0, (
            f"input_tokens should be > 0; got {response.usage.input_tokens}"
        )
        assert response.usage.output_tokens > 0, (
            f"output_tokens should be > 0; got {response.usage.output_tokens}"
        )


    def test_malformed_json_rejected(self, anthropic_client, praxis_proxy):
        """Verify anthropic_validate rejects malformed JSON end-to-end."""
        import httpx

        resp = httpx.post(
            f"http://127.0.0.1:{praxis_proxy}/v1/messages",
            content=b"not json {{{",
            headers={
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": LLM_KATAN_API_KEY,
            },
            timeout=30,
        )

        assert resp.status_code == 400, (
            f"malformed JSON should be rejected with 400; got {resp.status_code}"
        )
        body = resp.json()
        assert body["error"]["type"] == "invalid_request_error", (
            f"error type should be invalid_request_error; got {body}"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
