#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=2.0,<4", "pytest>=8.0"]
# ///
"""Live OpenAI-client coverage for a selected Anthropic Messages provider served
by real vLLM.

An official OpenAI Chat Completions client reaches Praxis, which selects a
trusted Anthropic candidate from a signed routing overlay
(``intelligent_route``), translates the request through
``openai_chat_completions_to_anthropic[_stream]``, and forwards it to vLLM's
native Anthropic Messages surface (``/v1/messages``). The Anthropic JSON/SSE
response is translated back into OpenAI Chat Completions shape for the client.

Scope of this lane: it proves selected-provider ROUTING and the two TRANSLATION
filters end to end against the real model the product serves. The credential
boundary is deliberately vLLM-specific: the keyed vLLM `/v1/messages` surface
authenticates with ``Authorization: Bearer`` (an OpenAI-style ``--api-key``),
whereas a `messages` provider's overlay credential can only be ``apikey`` (real
Anthropic's ``x-api-key``). So, exactly like the shipped
``examples/configs/anthropic/messages-native-vllm.yaml``, this lane injects the
backend bearer with the core ``credential_injection`` filter keyed on the
``intelligent_route``-selected cluster. The shipped selected-provider CREDENTIAL
behavior (overlay ``apikey`` -> ``x-api-key``) remains covered by the
deterministic per-PR lane ``test_selected_anthropic_chat_completions.py``.

Required for the live cases: PRAXIS_TEST_VLLM_BASE_URL (local http URL),
PRAXIS_TEST_VLLM_MODEL, VLLM_API_KEY, and a built praxis-ai binary (or
PRAXIS_AI_BIN). Set PRAXIS_TEST_REQUIRE_LIVE=1 in CI so missing infrastructure
fails instead of reporting a misleading green run. The digest-reproduction test
needs none of these and always runs; the scripted-backend shim test needs only
a built praxis-ai binary (no GPU).
"""

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pytest
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/integration/fixtures/inference/routing/openai-to-anthropic.json"

# The client-facing model id the OpenAI SDK sends. `intelligent_route` matches it
# against the overlay candidate's `name` (picker.rs select_legacy); Praxis then
# rewrites the upstream model to the candidate's `target_model` (the served vLLM
# model). Decoupling the two is the point of selected-provider routing.
CLIENT_MODEL = "client-model"

# The overlay candidate must carry an `apikey` credential reference or the
# overlay validator rejects a `messages` provider at load
# (filters/src/routing/overlay.rs validate_provider_credential). This lane does
# not inject that reference (no `credential_inject` filter); the backend bearer
# is injected by the core `credential_injection` filter instead. The secretRef
# is therefore structurally valid but unconsumed.
NAMESPACE = "grid-live"
SECRET_NAME = "vllm-backend-secret"
SECRET_KEY = "api-key"

# Fields the Praxis overlay validator folds into the content-addressed digest,
# present-only, canonicalized with RFC 8785 and hashed with SHA-256. See
# `compute_semantic_digest` in filters/src/routing/overlay.rs. For the ASCII /
# integer / boolean values an overlay carries, RFC 8785 canonical form is
# identical to compact, key-sorted json.dumps; test_overlay_generator_* pins
# that equivalence against the checked-in fixture's real digest.
_SEMANTIC_FIELDS = ("candidates", "local_site", "network", "selection_policy")


def _semantic_digest(overlay_body: dict) -> str:
    semantic = {k: overlay_body[k] for k in _SEMANTIC_FIELDS if k in overlay_body}
    canonical = json.dumps(semantic, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _signed_overlay(target_model: str) -> dict:
    """Build a content-addressed overlay envelope selecting one Anthropic
    candidate that targets ``target_model``. The credential is `apikey` only to
    satisfy the `messages` provider validator; it is not injected (see module
    docstring)."""
    body = {
        "local_site": "site-a",
        "network": "grid-live",
        "candidates": [
            {
                "kind": "inference_model",
                "name": CLIENT_MODEL,
                "site": "site-a",
                "cluster": "anthropic-backend",
                "fresh": True,
                "stable_id": "model/provider-anthropic-vllm",
                "api_format": "messages",
                "path": "/v1/messages",
                "target_model": target_model,
                "credential": {
                    "strategy": "apikey",
                    "secretRef": {"name": SECRET_NAME, "namespace": NAMESPACE, "key": SECRET_KEY},
                },
            }
        ],
    }
    digest = _semantic_digest(body)
    return {
        "schema_version": "1.0.0",
        "revision": {"kind": "content_addressed", "algorithm": "sha256", "value": digest},
        "content_digest": {"algorithm": "sha256", "value": digest},
        "scope": {"network": "grid-live", "gateway": "live-gateway", "namespace": NAMESPACE, "local_site": "site-a"},
        "provenance": {
            "producer": "integration-test",
            "producer_version": "1",
            "source_name": "grid-live",
            "source_uid": "live-1",
            "source_generation": 1,
            "rendered_at": "2026-10-21T00:00:00Z",
        },
        "overlay": body,
    }


def _build_config(proxy_port: int, overlay_path: Path, authority: str) -> dict:
    """Selected-provider routing + both translation filters, with the backend
    bearer injected the native-vLLM way (core `credential_injection` keyed on the
    `intelligent_route`-selected cluster). `credential_injection` reads the token
    from the VLLM_API_KEY environment variable and strips any client-supplied
    Authorization first."""
    insecure = {"allow_private_endpoints": True}
    if os.geteuid() == 0:
        # The ephemeral GPU runner runs as root; opt in only here.
        insecure["allow_root"] = True
    return {
        "listeners": [{"name": "proxy", "address": f"127.0.0.1:{proxy_port}", "filter_chains": ["main"]}],
        "filter_chains": [
            {
                "name": "main",
                "filters": [
                    {"filter": "model_to_header", "header": "X-Model"},
                    {
                        "filter": "intelligent_route",
                        "overlay_file": str(overlay_path),
                        "model_header": "X-Model",
                        "clusters": ["anthropic-backend"],
                        "reload": {"enabled": False},
                    },
                    {
                        "filter": "openai_chat_completions_to_anthropic_stream",
                        "max_partial_event_bytes": 1048576,
                        "max_tool_blocks": 256,
                    },
                    {
                        "filter": "openai_chat_completions_to_anthropic",
                        "api_translation_enabled": True,
                        "max_body_bytes": 1048576,
                    },
                    {
                        "filter": "credential_injection",
                        "clusters": [
                            {
                                "name": "anthropic-backend",
                                "header": "Authorization",
                                "env_var": "VLLM_API_KEY",
                                "header_prefix": "Bearer ",
                                "strip_client_credential": True,
                            }
                        ],
                    },
                    {"filter": "load_balancer", "clusters": [{"name": "anthropic-backend", "endpoints": [authority]}]},
                ],
            }
        ],
        "admin": {},
        "shutdown_timeout_secs": 5,
        "insecure_options": insecure,
    }


def _required_live() -> bool:
    return os.environ.get("PRAXIS_TEST_REQUIRE_LIVE", "").lower() in {"1", "true"}


def _live_config() -> tuple[str, str, str]:
    required = ("PRAXIS_TEST_VLLM_BASE_URL", "PRAXIS_TEST_VLLM_MODEL", "VLLM_API_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        message = f"live vLLM OpenAI-SDK translation test requires {', '.join(missing)}"
        if _required_live():
            pytest.fail(message)
        pytest.skip(message)

    parsed = urlparse(os.environ["PRAXIS_TEST_VLLM_BASE_URL"])
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port or parsed.path not in {"", "/"}:
        pytest.fail("PRAXIS_TEST_VLLM_BASE_URL must be a local http://host:port URL")
    return f"{parsed.hostname}:{parsed.port}", os.environ["PRAXIS_TEST_VLLM_MODEL"], os.environ["VLLM_API_KEY"]


def _binary() -> str:
    configured = os.environ.get("PRAXIS_AI_BIN")
    candidates = [Path(configured)] if configured else [
        ROOT / "target/debug/praxis-ai",
        ROOT / "target/release/praxis-ai",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    pytest.fail("praxis-ai binary is missing; build praxis-ai-proxy or set PRAXIS_AI_BIN")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_proxy(port: int, process: subprocess.Popen, log_path: Path) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"Praxis exited during startup:\n{log_path.read_text()}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    pytest.fail(f"Praxis did not listen on {port}:\n{log_path.read_text()}")


def _start_proxy(directory: Path, authority: str, target_model: str, backend_key: str) -> tuple[subprocess.Popen, int]:
    """Write the overlay + shim config, start praxis-ai pointed at ``authority``
    with VLLM_API_KEY=``backend_key``, and wait for it to listen."""
    overlay_path = directory / "routing-overlay.json"
    overlay_path.write_text(json.dumps(_signed_overlay(target_model)))
    proxy_port = _free_port()
    config_path = directory / "praxis.yaml"
    config_path.write_text(json.dumps(_build_config(proxy_port, overlay_path, authority)))
    log_path = directory / "praxis.log"
    environment = os.environ.copy()
    environment["VLLM_API_KEY"] = backend_key
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [_binary(), "-c", str(config_path)], stdout=log, stderr=subprocess.STDOUT, env=environment
        )
    _wait_for_proxy(proxy_port, process, log_path)
    return process, proxy_port


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _dump_praxis_log(directory: Path) -> None:
    """Print a bounded tail of the gateway log to stderr so a failed live request
    leaves Praxis-side diagnostics in the CI step log before the temporary
    directory is deleted (this lane is not in the workflow's log-tail step)."""
    log_path = directory / "praxis.log"
    if log_path.exists():
        sys.stderr.write(f"--- praxis.log (tail) ---\n{log_path.read_text()[-20000:]}\n")


# -----------------------------------------------------------------------------
# Overlay digest self-check (deterministic; no binary, no backend)
# -----------------------------------------------------------------------------


def test_overlay_generator_reproduces_fixture_digest():
    """RFC 8785 (JSON Canonicalization Scheme), https://datatracker.ietf.org/doc/html/rfc8785:
    for the overlay's value types (ASCII strings, booleans, small integers) the
    compact, key-sorted ``json.dumps`` form equals JCS output, so the Python
    generator's SHA-256 must reproduce the digest the Rust overlay validator
    (compute_semantic_digest) computes over the same semantic fields."""
    fixture = json.loads(FIXTURE.read_text())
    assert _semantic_digest(fixture["overlay"]) == fixture["content_digest"]["value"], (
        "generator digest must reproduce the checked-in fixture's content_digest"
    )
    assert fixture["content_digest"]["value"] == fixture["revision"]["value"], (
        "content-addressed revision must equal the content_digest"
    )


# -----------------------------------------------------------------------------
# Credential-shim proof against a scripted Anthropic backend (deterministic; needs
# only the praxis-ai binary, no GPU). Proves routing + translation reach
# /v1/messages AND that the backend receives Authorization: Bearer <VLLM_API_KEY>
# and never the client credential or an x-api-key.
# -----------------------------------------------------------------------------

_SCRIPTED_BACKEND_KEY = "scripted-backend-bearer-key"


class _BackendCapture:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: deque[dict] = deque()


_CAPTURE = _BackendCapture()


class _AnthropicHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        length = int(self.headers.get("content-length", 0))
        self.rfile.read(length)
        authorization = self.headers.get("authorization")
        with _CAPTURE.lock:
            _CAPTURE.requests.append(
                {"path": self.path, "authorization": authorization, "x_api_key": self.headers.get("x-api-key")}
            )
        if authorization != f"Bearer {_SCRIPTED_BACKEND_KEY}":
            body = json.dumps({"type": "error", "error": {"type": "authentication_error", "message": "no bearer"}})
            encoded = body.encode()
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        body = json.dumps(
            {
                "type": "message",
                "id": "msg_scripted",
                "role": "assistant",
                "model": "claude-scripted",
                "content": [{"type": "text", "text": "hello from backend"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 3},
            }
        )
        encoded = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args: object) -> None:
        pass


def test_credential_shim_routes_and_translates_with_bearer():
    with _CAPTURE.lock:
        _CAPTURE.requests.clear()
    backend = ThreadingHTTPServer(("127.0.0.1", 0), _AnthropicHandler)
    authority = f"127.0.0.1:{backend.server_address[1]}"
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    with tempfile.TemporaryDirectory(prefix="praxis-selected-anthropic-shim-") as temporary:
        process, proxy_port = _start_proxy(Path(temporary), authority, "scripted-model", _SCRIPTED_BACKEND_KEY)
        try:
            with OpenAI(
                base_url=f"http://127.0.0.1:{proxy_port}/v1", api_key="caller-only", max_retries=0, timeout=10
            ) as client:
                response = client.chat.completions.create(
                    model=CLIENT_MODEL, max_tokens=16, messages=[{"role": "user", "content": "hi"}]
                )
            assert response.choices[0].message.content == "hello from backend", "Anthropic text must reach OpenAI message"
            assert response.usage.prompt_tokens == 5, "Anthropic input_tokens must map to prompt_tokens"
            assert response.usage.completion_tokens == 3, "Anthropic output_tokens must map to completion_tokens"
        except BaseException:
            _dump_praxis_log(Path(temporary))
            raise
        finally:
            _terminate(process)
    backend.shutdown()
    thread.join(timeout=5)
    captured = [r for r in _CAPTURE.requests if r["path"] == "/v1/messages"]
    assert captured, f"backend never received /v1/messages; saw {[r['path'] for r in _CAPTURE.requests]}"
    last = captured[-1]
    assert last["authorization"] == f"Bearer {_SCRIPTED_BACKEND_KEY}", "backend must receive the injected vLLM bearer"
    assert last["x_api_key"] is None, "no x-api-key may reach the backend"


# -----------------------------------------------------------------------------
# Live tests against real vLLM /v1/messages (GPU; gated on PRAXIS_TEST_REQUIRE_LIVE)
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_client():
    authority, served_model, backend_key = _live_config()
    with tempfile.TemporaryDirectory(prefix="praxis-selected-anthropic-vllm-") as temporary:
        process, proxy_port = _start_proxy(Path(temporary), authority, served_model, backend_key)
        client = OpenAI(
            base_url=f"http://127.0.0.1:{proxy_port}/v1", api_key="caller-only", max_retries=0, timeout=180
        )
        try:
            yield client, CLIENT_MODEL
        except BaseException:
            _dump_praxis_log(Path(temporary))
            raise
        finally:
            client.close()
            _terminate(process)


def test_live_json_and_usage(live_client):
    client, model = live_client
    response = client.chat.completions.create(
        model=model,
        max_tokens=64,
        messages=[{"role": "user", "content": "Say hello. /no_think"}],
    )
    content = response.choices[0].message.content
    assert content and content.strip(), "translated Anthropic text must reach the OpenAI message content"
    assert response.usage.prompt_tokens > 0, "Anthropic input usage must map to a positive prompt_tokens"
    assert response.usage.completion_tokens > 0, "Anthropic output usage must map to a positive completion_tokens"
    assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens, (
        "total_tokens must equal prompt_tokens + completion_tokens"
    )


def test_live_streaming(live_client):
    client, model = live_client
    chunks = list(
        client.chat.completions.create(
            model=model,
            max_tokens=64,
            stream=True,
            stream_options={"include_usage": True},
            messages=[{"role": "user", "content": "Say hello. /no_think"}],
        )
    )
    assert any(chunk.choices and chunk.choices[0].delta.content for chunk in chunks), (
        "translated SSE must deliver at least one Chat Completions content delta"
    )
    assert chunks[-1].usage, "final chunk must carry usage when include_usage is set"
    assert chunks[-1].usage.total_tokens > 0, "terminal usage must report positive total_tokens"


def test_live_tool_round_trip(live_client):
    client, model = live_client
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    prompt = "Use get_weather for Paris before answering. Do not guess. /no_think"
    first = client.chat.completions.create(
        model=model, max_tokens=128, tools=tools, tool_choice="auto",
        messages=[{"role": "user", "content": prompt}],
    )
    calls = first.choices[0].message.tool_calls or []
    assert calls, f"model must emit a tool call; got finish_reason={first.choices[0].finish_reason}"
    call = calls[0]
    assert call.function.name == "get_weather", "the translated tool call must name the requested function"
    assert "paris" in str(json.loads(call.function.arguments).get("city", "")).lower(), (
        "tool-call arguments must preserve the requested city"
    )
    second = client.chat.completions.create(
        model=model, max_tokens=128, tools=tools, tool_choice="none",
        messages=[
            {"role": "user", "content": prompt},
            first.choices[0].message.model_dump(exclude_none=True),
            {"role": "tool", "tool_call_id": call.id, "content": "17 degrees Celsius"},
        ],
    )
    final = second.choices[0].message.content
    assert final and final.strip(), "tool-result round trip must return translated assistant text"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
