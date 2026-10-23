#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.27",
#     "pytest>=8.0",
# ]
# ///
"""Kuadrant MCP Gateway authorization acceptance — through the Responses agentic loop.

Proves, end-to-end, that a REAL Kuadrant MCP Gateway enforces per-user tool
authorization (discovery filtering + per-call authorization) when driven through
Praxis's OpenAI Responses agentic loop. No Praxis product-code changes.

Prereqs (the Makefile target wires these up):
  1. The gateway stack is up: `bash tests/integration/kuadrant/stack_up.sh`
     (writes tests/integration/kuadrant/.stack-env).
  2. A praxis-ai binary exists (PRAXIS_AI_BIN or target/{debug,release}/praxis-ai).
     Items 1-3 run with store:false; the approval items (5/6b) need store-sqlite.

Run: make test-kuadrant-acceptance   (or: uv run <this file> -s)
"""
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pytest

REPO = Path(__file__).resolve().parents[4]
KDIR = REPO / "tests" / "integration" / "kuadrant"
STACK_ENV = KDIR / ".stack-env"
MINT = KDIR / "scripts" / "mint_user_jwt.py"
SCRIPTED_MODEL = KDIR / "scripts" / "scripted_model.py"
SWITCH = KDIR / "switch_authconfig.sh"
EXAMPLE_CONFIG = REPO / "examples" / "configs" / "openai" / "responses" / "mcp-kuadrant-gateway.yaml"

# The MCP tool entry the client sends: connector-only (connector_id, NOT server_url)
# so Praxis forwards the per-user context; auto-execute (no approval) for items 1-4.
MCP_TOOL = {
    "type": "mcp",
    "server_label": "gw",
    "connector_id": "trusted-mcp",
    "allowed_tools": ["weather_get_forecast"],
    "require_approval": "never",
}

# resource_access role sets for the two identities.
ROLES_AUTHORIZED = {"weather": {"roles": ["tool:get_forecast"]}}
ROLES_NONE = {"weather": {"roles": []}}


# --- low-level helpers -------------------------------------------------------

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _find_binary() -> str:
    if "PRAXIS_AI_BIN" in os.environ:
        p = os.environ["PRAXIS_AI_BIN"]
        if os.path.isfile(p):
            return p
        raise FileNotFoundError(f"PRAXIS_AI_BIN={p!r} not found")
    for c in ["target/debug/praxis-ai", "target/release/praxis-ai"]:
        if (REPO / c).is_file():
            return str(REPO / c)
    raise FileNotFoundError("praxis-ai binary not found — run `cargo build -p praxis-ai-proxy`")


def _wait_tcp(port: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"nothing listening on 127.0.0.1:{port} within {timeout}s")


def _wait_http(url: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            httpx.get(url, timeout=1.0)
            return
        except Exception:
            time.sleep(0.2)
    raise TimeoutError(f"no HTTP response from {url} within {timeout}s")


# --- per-user minting + request helpers (consume the `stack` fixture) --------

def _mint(stack, sub, resource_access, exp=600, tamper=False):
    cmd = [
        "uv", "run", "--quiet", str(MINT),
        "--key", stack["KUADRANT_ISSUER_PRIV"], "--kid", stack["KUADRANT_ISSUER_KID"],
        "--iss", stack["KUADRANT_ISSUER"], "--aud", stack["KUADRANT_AUDIENCE"],
        "--sub", sub, "--exp", str(exp), "--resource-access", json.dumps(resource_access),
    ]
    if tamper:
        cmd.append("--tamper")
    return subprocess.check_output(cmd).decode().strip()


def _reset_backend(stack):
    httpx.post(stack["KUADRANT_BACKEND_STATS"] + "/reset", timeout=5)


def _backend_count(stack, tool):
    return httpx.get(stack["KUADRANT_BACKEND_STATS"] + "/stats", timeout=5).json().get(tool, 0)


def _switch_authconfig(variant):
    """Apply an AuthConfig variant and wait for Authorino to reconcile it."""
    subprocess.run(["bash", str(SWITCH), variant], check=True)


def _responses(praxis_url, user, token, tools, stream=False, input_text="What is the weather in NYC?"):
    body = {
        "model": "scripted-model",
        "input": input_text,
        "store": False,
        "stream": stream,
        "tools": tools,
    }
    headers = {
        "Content-Type": "application/json",
        "x-auth-tenant": "acme",
        "x-auth-user": user,
        "x-user-mcp-key": token,
    }
    return httpx.post(praxis_url + "/v1/responses", headers=headers, json=body, timeout=90)


def _discovered_tools(resp_json):
    return [
        t.get("name")
        for o in (resp_json.get("output") or [])
        if o.get("type") == "mcp_list_tools"
        for t in (o.get("tools") or [])
    ]


def _mcp_call_ok(resp_json):
    return any(
        o.get("type") == "mcp_call" and not o.get("error") and o.get("output")
        for o in (resp_json.get("output") or [])
    )


def _mcp_call_error(resp_json):
    return any(
        o.get("type") == "mcp_call" and o.get("error")
        for o in (resp_json.get("output") or [])
    )


# --- fixtures ----------------------------------------------------------------

@pytest.fixture(scope="session")
def stack():
    """Read the running gateway stack's handles from .stack-env; skip if not up."""
    if not STACK_ENV.exists():
        pytest.skip(f"Kuadrant stack not up — run `bash {KDIR}/stack_up.sh` first")
    env = {}
    for line in STACK_ENV.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    _wait_http(env["KUADRANT_BACKEND_STATS"] + "/stats", 10)  # fail fast if gateway gone
    return env


@pytest.fixture(scope="session")
def scripted_model():
    """Deterministic Responses model upstream: emits the MCP tool call then a final
    message (no GPU). SUBSTR=forecast matches weather_get_forecast."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPTED_MODEL)],
        env={**os.environ, "PORT": str(port), "CALL_TOOL_SUBSTR": "forecast"},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _wait_http(f"http://127.0.0.1:{port}/", 15)
        yield port
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@pytest.fixture(scope="session")
def praxis(stack, scripted_model):
    """Real praxis-ai driving the gateway connector + the scripted model, from the
    committed example config patched to the live ports (Topology A, store:false)."""
    proxy_port = _free_port()
    cfg = EXAMPLE_CONFIG.read_text()
    cfg = cfg.replace("0.0.0.0:8080", f"127.0.0.1:{proxy_port}")
    cfg = cfg.replace("127.0.0.1:3001", f"127.0.0.1:{scripted_model}")
    cfg = cfg.replace("http://127.0.0.1:8000/mcp", stack["KUADRANT_GATEWAY_URL"])
    fd, cfg_path = tempfile.mkstemp(suffix=".yaml", prefix="praxis-kuadrant-")
    with os.fdopen(fd, "w") as f:
        f.write(cfg)
    proc = subprocess.Popen(
        [_find_binary(), "-c", cfg_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _wait_tcp(proxy_port, 30)
        yield f"http://127.0.0.1:{proxy_port}"
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        os.unlink(cfg_path)


# --- tests -------------------------------------------------------------------

def test_item1_authorized_user_discovers_and_invokes_buffered(stack, praxis, scripted_model):
    """userA (tool:get_forecast) discovers weather_get_forecast, the agentic loop
    dispatches it to the gateway, and the real backend executes exactly once."""
    token = _mint(stack, "userA", ROLES_AUTHORIZED)
    _reset_backend(stack)
    r = _responses(praxis, "userA", token, [MCP_TOOL], stream=False)
    assert r.status_code == 200, r.text
    j = r.json()
    assert "weather_get_forecast" in _discovered_tools(j), j
    assert _mcp_call_ok(j), j
    assert _backend_count(stack, "get_forecast") == 1


def test_item2_unauthorized_user_sees_empty_discovery(stack, praxis, scripted_model):
    """userB (no weather roles): the gateway filters weather_get_forecast out of
    discovery, so the model is never offered it and nothing dispatches."""
    token = _mint(stack, "userB", ROLES_NONE)
    _reset_backend(stack)
    r = _responses(praxis, "userB", token, [MCP_TOOL], stream=False)
    assert r.status_code == 200, r.text
    j = r.json()
    assert "weather_get_forecast" not in _discovered_tools(j), j
    assert _backend_count(stack, "get_forecast") == 0


def test_item3a_praxis_does_not_dispatch_undiscovered_tool(stack, praxis, scripted_model):
    """Even when the model force-emits a call to a tool userB never discovered,
    Praxis does not dispatch it to the gateway (defense in depth): no successful MCP
    call, no backend execution."""
    token = _mint(stack, "userB", ROLES_NONE)
    _reset_backend(stack)
    r = _responses(
        praxis, "userB", token, [MCP_TOOL], stream=False,
        input_text="force-call:weather_get_forecast",
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert not _mcp_call_ok(j), j
    assert _backend_count(stack, "get_forecast") == 0


def test_item3b_gateway_denies_discovered_but_forbidden_call(stack, praxis, scripted_model):
    """Open-discovery variant: userB CAN discover weather_get_forecast, but the
    gateway's per-call authz denies the invocation (403 -> mcp_call.error). Proves
    discovery is not enforcement and that the call reaches the gateway through
    Praxis dispatch."""
    _switch_authconfig("opendisco")
    try:
        token = _mint(stack, "userB", ROLES_NONE)
        _reset_backend(stack)
        r = _responses(praxis, "userB", token, [MCP_TOOL], stream=False)
        assert r.status_code == 200, r.text
        j = r.json()
        assert "weather_get_forecast" in _discovered_tools(j), j
        assert _mcp_call_error(j), j
        assert _backend_count(stack, "get_forecast") == 0
    finally:
        _switch_authconfig("filtered")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
