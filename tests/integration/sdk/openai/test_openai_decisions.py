#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=3.26,<4", "pytest>=8.0"]
# ///
"""Official OpenAI-client coverage for the Decisions API pass-through."""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[4]


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _binary() -> str:
    configured = os.environ.get("PRAXIS_AI_BIN")
    if configured:
        return configured
    for candidate in ("target/debug/praxis-ai", "target/release/praxis-ai"):
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("build praxis-ai-proxy or set PRAXIS_AI_BIN")


class _BackendState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.scripted: list[tuple[int, bytes]] = []
        self.requests: list[tuple[str, dict]] = []

    def script(self, status: int, body: dict) -> None:
        with self.lock:
            self.scripted.append((status, json.dumps(body).encode()))

    def take(self, path: str, body: bytes) -> tuple[int, bytes]:
        with self.lock:
            self.requests.append((path, json.loads(body)))
            return self.scripted.pop(0)


STATE = _BackendState()


@pytest.fixture(autouse=True)
def reset_backend_state():
    with STATE.lock:
        STATE.scripted.clear()
        STATE.requests.clear()


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers["content-length"])
        status, body = STATE.take(self.path, self.rfile.read(length))
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


def _decision_response() -> dict:
    return {
        "model": "gpt-6-luna",
        "answers": [
            {"type": "predicate", "name": "refund_requested", "probability": 0.95},
            {
                "type": "choice",
                "name": "department",
                "choice": "billing",
                "probabilities": [
                    {"value": "billing", "probability": 0.93},
                    {"value": "other", "probability": 0.07},
                ],
                "confidence": 0.93,
            },
        ],
        "usage": {
            "input_tokens": 162,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 162,
        },
    }


def _config(proxy_port: int, backend_port: int) -> dict:
    return {
        "listeners": [{"name": "proxy", "address": f"127.0.0.1:{proxy_port}", "filter_chains": ["main"]}],
        "filter_chains": [{
            "name": "main",
            "filters": [
                {"filter": "ai_operation"},
                {"filter": "router", "routes": [{"path_prefix": "/", "cluster": "decisions-backend"}]},
                {"filter": "load_balancer", "clusters": [{
                    "name": "decisions-backend", "endpoints": [f"127.0.0.1:{backend_port}"]}]},
            ],
        }],
        "admin": {},
        "shutdown_timeout_secs": 5,
        "insecure_options": {"allow_private_endpoints": True},
    }


@pytest.fixture(scope="module")
def client():
    proxy_port, backend_port = _port(), _port()
    backend = ThreadingHTTPServer(("127.0.0.1", backend_port), _Handler)
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    with tempfile.TemporaryDirectory() as directory:
        config = Path(directory) / "praxis.yaml"
        config.write_text(json.dumps(_config(proxy_port, backend_port)))
        proc = subprocess.Popen([_binary(), "-c", str(config)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection(("127.0.0.1", proxy_port), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                raise TimeoutError("Praxis did not start")
            with OpenAI(api_key="caller-only", base_url=f"http://127.0.0.1:{proxy_port}/v1",
                        max_retries=0, timeout=10) as sdk:
                yield sdk
        finally:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    backend.shutdown()
    thread.join(timeout=5)


class TestOpenAIDecisionsPassthrough:
    def test_predicate_and_choice_answers_round_trip(self, client):
        STATE.script(200, _decision_response())
        decision = client.decisions.create(
            model="gpt-6-luna",
            input="I was charged twice for my order. Please return the extra charge.",
            questions=[
                {"type": "predicate", "name": "refund_requested",
                 "instructions": "Does the customer ask for a refund?"},
                {"type": "choice", "name": "department",
                 "instructions": "Which department should handle this complaint?",
                 "choices": [
                     {"value": "billing", "description": "Payments and refunds."},
                     {"value": "other", "description": "All other requests."},
                 ]},
            ],
        )
        answers = {a.name: a for a in decision.answers}
        assert answers["refund_requested"].type == "predicate"
        assert answers["refund_requested"].probability == 0.95
        assert answers["department"].type == "choice"
        assert answers["department"].choice == "billing"
        assert decision.usage.input_tokens == 162

    def test_the_request_body_reaches_the_backend_unmodified(self, client):
        STATE.script(200, _decision_response())
        client.decisions.create(
            model="gpt-6-luna",
            input="Order arrived damaged.",
            questions=[{"type": "predicate", "name": "damaged",
                        "instructions": "Does the text report damage?"}],
        )
        with STATE.lock:
            path, body = STATE.requests[-1]
        assert path == "/v1/decisions", "the SDK create call must POST the official path"
        assert body == {
            "model": "gpt-6-luna",
            "input": "Order arrived damaged.",
            "questions": [{"type": "predicate", "name": "damaged",
                           "instructions": "Does the text report damage?"}],
        }, "pass-through must not rewrite the request body"

    def test_refusal_answers_still_parse(self, client):
        STATE.script(200, {
            "model": "gpt-6-luna",
            "answers": [{"type": "refusal", "name": "damaged"}],
            "usage": {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10},
        })
        decision = client.decisions.create(
            model="gpt-6-luna",
            input="Order arrived damaged.",
            questions=[{"type": "predicate", "name": "damaged",
                        "instructions": "Does the text report damage?"}],
        )
        assert decision.answers[0].type == "refusal"
        assert decision.answers[0].name == "damaged"


if __name__ == "__main__":
    pytest.main([__file__, *sys.argv[1:]])
