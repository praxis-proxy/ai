#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=2.0,<4", "pytest>=8.0"]
# ///
"""Live OpenAI-client coverage for a selected Anthropic Messages provider served
by real vLLM.

An official OpenAI Chat Completions client reaches Praxis, which selects a
trusted Anthropic candidate from a signed routing overlay, translates the
request through ``openai_chat_completions_to_anthropic[_stream]``, and forwards
it to vLLM's native Anthropic Messages surface (``/v1/messages``). The
Anthropic JSON/SSE response is translated back into OpenAI Chat Completions
shape for the client.

This complements the deterministic fake-backend lane
(``test_selected_anthropic_chat_completions.py``): that one pins exact token
math and error/policy cases against a scripted server, while this one proves
the shipped translation against the real model the product serves.

Required for the live cases: PRAXIS_TEST_VLLM_BASE_URL (local http URL),
PRAXIS_TEST_VLLM_MODEL, VLLM_API_KEY, and a built praxis-ai binary (or
PRAXIS_AI_BIN). Set PRAXIS_TEST_REQUIRE_LIVE=1 in CI so missing infrastructure
fails instead of reporting a misleading green run. The digest-reproduction test
needs none of these and always runs.
"""

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import pytest
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/integration/fixtures/inference/routing/openai-to-anthropic.json"
EXAMPLE = ROOT / "examples/configs/openai-selected-provider-legacy-translation.yaml"

# The client-facing model id the OpenAI SDK sends. `intelligent_route` matches it
# against the overlay candidate's `name`; Praxis then rewrites the upstream model
# to the candidate's `target_model` (the served vLLM model). Decoupling the two
# is the whole point of selected-provider routing, so they are deliberately
# different here.
CLIENT_MODEL = "client-model"

# Projected-credential coordinates. These must agree between the overlay
# candidate's `secretRef` and the file written under the example's
# `projected_credential_mount_base`, or `credential_inject` fails closed (503).
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
    candidate that targets ``target_model`` behind a bearer credential."""
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
                    "strategy": "bearer_token",
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


def _render_config(proxy_port: int, overlay_path: Path, mount_base: Path, authority: str) -> str:
    """Render the shipped selected-provider example with test-local addresses and
    the real vLLM authority, so the lane exercises exactly what ships."""
    text = EXAMPLE.read_text()
    sentinels = (
        'address: "0.0.0.0:8080"',
        "127.0.0.1:9901",
        "/etc/praxis/routing/routing-overlay.json",
        "/run/secrets/provider-credentials",
        "127.0.0.1:8001",
        "insecure_options:\n",
    )
    for sentinel in sentinels:
        assert text.count(sentinel) == 1, f"{sentinel!r} is not unique in {EXAMPLE}"
    text = text.replace('address: "0.0.0.0:8080"', f'address: "127.0.0.1:{proxy_port}"')
    text = text.replace("127.0.0.1:9901", f"127.0.0.1:{_free_port()}")
    text = text.replace("/etc/praxis/routing/routing-overlay.json", str(overlay_path))
    text = text.replace("/run/secrets/provider-credentials", str(mount_base))
    text = text.replace("127.0.0.1:8001", authority)
    # The ephemeral GPU runner runs as root; opt in only in this generated config,
    # mirroring the Anthropic SDK harness.
    if os.geteuid() == 0:
        text = text.replace("insecure_options:\n", "insecure_options:\n  allow_root: true\n", 1)
    return text


def test_overlay_generator_reproduces_fixture_digest():
    fixture = json.loads(FIXTURE.read_text())
    assert _semantic_digest(fixture["overlay"]) == fixture["content_digest"]["value"]
    assert fixture["content_digest"]["value"] == fixture["revision"]["value"]


@pytest.fixture(scope="module")
def live_client():
    authority, served_model, backend_key = _live_config()
    binary = _binary()
    with tempfile.TemporaryDirectory(prefix="praxis-selected-anthropic-vllm-") as temporary:
        directory = Path(temporary)
        overlay_path = directory / "routing-overlay.json"
        overlay_path.write_text(json.dumps(_signed_overlay(served_model)))
        mount_base = directory / "secrets"
        secret_dir = mount_base / NAMESPACE / SECRET_NAME
        secret_dir.mkdir(parents=True)
        (secret_dir / SECRET_KEY).write_text(backend_key)

        proxy_port = _free_port()
        config_path = directory / "praxis.yaml"
        config_path.write_text(_render_config(proxy_port, overlay_path, mount_base, authority))
        log_path = directory / "praxis.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [binary, "-c", str(config_path)],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
        try:
            _wait_for_proxy(proxy_port, process, log_path)
            client = OpenAI(
                base_url=f"http://127.0.0.1:{proxy_port}/v1",
                api_key="caller-only",
                max_retries=0,
                timeout=180,
            )
            try:
                yield client, CLIENT_MODEL
            finally:
                client.close()
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def test_live_json_and_usage(live_client):
    client, model = live_client
    response = client.chat.completions.create(
        model=model,
        max_tokens=64,
        messages=[{"role": "user", "content": "Say hello. /no_think"}],
    )
    assert response.choices[0].message.content and response.choices[0].message.content.strip()
    assert response.usage.prompt_tokens > 0 and response.usage.completion_tokens > 0
    assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens


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
    assert any(chunk.choices and chunk.choices[0].delta.content for chunk in chunks)
    assert chunks[-1].usage and chunks[-1].usage.total_tokens > 0


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
    if not calls:
        pytest.xfail("pinned vLLM v0.29.0 did not emit a tool call for this turn")
    call = calls[0]
    assert call.function.name == "get_weather"
    assert "paris" in str(json.loads(call.function.arguments).get("city", "")).lower()
    second = client.chat.completions.create(
        model=model, max_tokens=128, tools=tools, tool_choice="none",
        messages=[
            {"role": "user", "content": prompt},
            first.choices[0].message.model_dump(exclude_none=True),
            {"role": "tool", "tool_call_id": call.id, "content": "17 degrees Celsius"},
        ],
    )
    assert second.choices[0].message.content and second.choices[0].message.content.strip()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
