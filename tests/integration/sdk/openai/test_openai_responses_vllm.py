#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "openai>=2.0",
#     "pytest>=8.0",
# ]
# ///
"""
OpenAI Responses API integration tests against a real vLLM CPU backend.

Starts a Praxis proxy with the full responses pipeline backed by vLLM,
then exercises stateless requests, persistence, rehydration, and streaming
using the official OpenAI Python SDK.

Usage:
    cargo build -p praxis-ai-proxy
    uv run tests/integration/sdk/openai/test_openai_responses_vllm.py -s
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
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-0.6B")
OGX_BASE_URL = os.environ.get("OGX_BASE_URL", "http://127.0.0.1:8321")
PRAXIS_AI_BIN = os.environ.get("PRAXIS_AI_BIN")
CONFIG_PATH = "examples/configs/openai/responses/full-flow.yaml"

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


def _vllm_endpoint() -> str:
    parsed = urlparse(VLLM_BASE_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    return f"{host}:{port}"


def _ogx_endpoint() -> str:
    parsed = urlparse(OGX_BASE_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8321
    return f"{host}:{port}"


def _write_config(praxis_port: int, db_path: str) -> str:
    with open(CONFIG_PATH) as f:
        config = f.read()

    config = config.replace("127.0.0.1:8080", f"127.0.0.1:{praxis_port}")
    config = config.replace("127.0.0.1:3001", _vllm_endpoint())
    config = config.replace("127.0.0.1:9999", _ogx_endpoint())
    config = config.replace(
        "sqlite://responses.db?mode=rwc",
        f"sqlite://{db_path}?mode=rwc",
    )

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
    """Start a Praxis proxy backed by vLLM for the test session."""
    port = _free_port()
    db_dir = tmp_path_factory.mktemp("responses")
    db_path = str(db_dir / "responses.db")
    config_path = _write_config(port, db_path)
    binary = _find_binary()

    log_path = str(db_dir / "praxis.log")
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
def openai_client(praxis_proxy):
    """Return an OpenAI client pointed at the local Praxis proxy."""
    return OpenAI(
        base_url=f"http://127.0.0.1:{praxis_proxy}/v1",
        api_key="test",
        max_retries=0,
        timeout=180,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOpenAIResponsesVLLM:
    """Integration tests for the Responses API against a vLLM backend."""

    def test_stateless_request(self, openai_client):
        response = openai_client.responses.create(
            model=VLLM_MODEL,
            input="Say exactly: HELLO-PRAXIS /no_think",
            store=False,
            max_output_tokens=128,
        )

        assert response.status == "completed"
        assert "HELLO-PRAXIS" in response.output_text

    def test_store_and_retrieve(self, openai_client):
        response = openai_client.responses.create(
            model=VLLM_MODEL,
            input="Say exactly: STORED-OK /no_think",
            store=True,
            max_output_tokens=128,
        )

        assert response.status == "completed"
        assert response.id

        retrieved = openai_client.responses.retrieve(response.id)

        assert retrieved.id == response.id
        assert retrieved.status == "completed"

    def test_rehydrated_second_turn(self, openai_client):
        first = openai_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "Remember this nonce: VIOLET-7319. "
                "Acknowledge it. /no_think"
            ),
            store=True,
            max_output_tokens=128,
        )

        assert first.status == "completed"

        second = openai_client.responses.create(
            model=VLLM_MODEL,
            input=(
                "What nonce did I just tell you? "
                "Repeat it exactly. /no_think"
            ),
            previous_response_id=first.id,
            store=True,
            max_output_tokens=128,
        )

        assert second.status == "completed"
        assert "VIOLET-7319" in second.output_text

    def test_doc_extract_inline_file_to_input_text(self, openai_client):
        """Issue #397: inline file_data is extracted to input_text and
        consumed by vLLM inference.

        Sends an input_file with base64-encoded text through the full
        pipeline (file_resolve → doc_extract → responses_proxy → vLLM).
        The doc_extract filter converts the input_file to input_text
        before forwarding. Asserts a unique marker from the document
        appears in the model output, proving vLLM consumed the
        extracted text.
        """
        import base64

        marker = "PRAXIS-DOC-9271"
        file_content = f"The secret marker is: {marker}"
        file_data = (
            "data:text/plain;base64,"
            + base64.b64encode(file_content.encode()).decode()
        )

        response = openai_client.responses.create(
            model=VLLM_MODEL,
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": "document.txt",
                            "file_data": file_data,
                        },
                        {
                            "type": "input_text",
                            "text": (
                                "What marker appears in the document? "
                                "Repeat it exactly. /no_think"
                            ),
                        },
                    ],
                }
            ],
            store=False,
            max_output_tokens=128,
        )

        assert response.status == "completed"
        assert marker in response.output_text, (
            f"vLLM should produce output containing the document "
            f"marker '{marker}'; got: {response.output_text}"
        )

    def test_file_id_resolution_through_ogx(self, openai_client):
        """End-to-end: upload to OGX via Praxis, reference by file_id,
        verify vLLM output contains the file content.

        Pipeline: file_resolve (OGX) -> doc_extract -> responses_proxy -> vLLM
        """
        import io

        marker = "PRAXIS-OGX-FILE-4829"
        file_content = f"The secret marker is: {marker}"

        uploaded = openai_client.files.create(
            file=("marker-document.txt", io.BytesIO(file_content.encode())),
            purpose="user_data",
        )
        file_id = uploaded.id

        try:
            response = openai_client.responses.create(
                model=VLLM_MODEL,
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "file_id": file_id,
                            },
                            {
                                "type": "input_text",
                                "text": (
                                    "What marker appears in the document? "
                                    "Repeat it exactly. /no_think"
                                ),
                            },
                        ],
                    }
                ],
                store=False,
                max_output_tokens=128,
            )

            assert response.status == "completed"
            assert marker in response.output_text, (
                f"vLLM should produce output containing the file marker "
                f"'{marker}'; got: {response.output_text}"
            )
        finally:
            try:
                openai_client.files.delete(file_id)
            except Exception:
                pass

    def test_streaming(self, openai_client):
        stream = openai_client.responses.create(
            model=VLLM_MODEL,
            input="Say exactly: STREAM-OK /no_think",
            store=False,
            stream=True,
            max_output_tokens=128,
        )

        event_types = []
        text_parts = []
        final_status = None

        for event in stream:
            event_types.append(event.type)
            if event.type == "response.output_text.delta":
                text_parts.append(event.delta)
            if event.type == "response.completed":
                final_status = event.response.status

        assert event_types[0] == "response.created"
        assert event_types[-1] == "response.completed"
        assert final_status == "completed"
        assert "STREAM-OK" in "".join(text_parts)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
