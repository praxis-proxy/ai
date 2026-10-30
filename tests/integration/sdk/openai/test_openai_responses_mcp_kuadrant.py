#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.27",
#     "pytest>=8.0",
# ]
# ///
"""Kuadrant MCP Gateway authorization acceptance — through the full-flow Responses loop.

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
import re
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
PASSING_NEMO = KDIR / "scripts" / "passing_nemo.py"
SWITCH = KDIR / "switch_authconfig.sh"
FULL_FLOW_CONFIG = REPO / "examples" / "configs" / "openai" / "responses" / "full-flow-agentic.yaml"

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

# A representative opaque x-mcp-authorized wristband forwarded by Praxis (the courier
# the issue names). In Topology A the gateway's Authorino mints its own, so the value
# is opaque to enforcement; it is forwarded unchanged and observed for item 7.
WRISTBAND = "praxis-forwarded-wristband-" + "a" * 32

# Same tool, but requiring explicit approval (two-turn flow) for items 5/6b.
APPROVAL_TOOL = {**MCP_TOOL, "require_approval": "always"}


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


def _ingress_lines(stack):
    """Parse the Envoy ingress observer's one-way digests (item 7) from envoy logs."""
    engine = stack["KUADRANT_CONTAINER_ENGINE"]
    out = subprocess.run([engine, "logs", "envoy"], capture_output=True, text=True)
    text = (out.stdout or "") + (out.stderr or "")
    lines = []
    for ln in text.splitlines():
        m = re.search(r"MCPINGRESS method=(\S+) bearer=(\S+) wristband=(\S+)", ln)
        if m:
            lines.append({"method": m.group(1), "bearer": m.group(2), "wristband": m.group(3)})
    return lines


def _ingress_since(stack, before, predicate, tries=25):
    """Poll the new ingress lines (those after index `before`) until `predicate`
    holds — container log capture can lag the HTTP response by a few ms."""
    new = _ingress_lines(stack)[before:]
    for _ in range(tries):
        if predicate(new):
            return new
        time.sleep(0.1)
        new = _ingress_lines(stack)[before:]
    return new


# --- streaming helpers -------------------------------------------------------

def _responses_stream(praxis_url, user, token, tools, input_text="What is the weather in NYC?"):
    """Drive a streaming Responses request; return (status, raw_sse_text)."""
    body = {"model": "scripted-model", "input": input_text, "store": False, "stream": True, "tools": tools}
    raw = ""
    with httpx.stream("POST", praxis_url + "/v1/responses", headers=_headers(user, token), json=body, timeout=90) as r:
        status = r.status_code
        for chunk in r.iter_text():
            raw += chunk
    return status, raw


def _sse_final_output(raw):
    """Extract the final response.output from the terminal response.completed event."""
    cur = None
    for line in raw.splitlines():
        if line.startswith("event:"):
            cur = line[len("event:"):].strip()
        elif line.startswith("data:") and cur == "response.completed":
            try:
                payload = json.loads(line[len("data:"):].strip())
            except Exception:
                continue
            return (payload.get("response") or {}).get("output") or []
    return []


def _sse_terminated_cleanly(raw):
    """A valid terminal SSE lifecycle: exactly one response.created and exactly one
    terminal event (response.completed for success, or response.failed/error)."""
    created = raw.count("event: response.created")
    terminal = (
        raw.count("event: response.completed")
        + raw.count("event: response.failed")
        + raw.count("event: error")
    )
    return created == 1 and terminal == 1


def _headers(user, token, wristband=WRISTBAND):
    h = {
        "Content-Type": "application/json",
        "x-auth-tenant": "acme",
        "x-auth-user": user,
        "x-user-mcp-key": token,
    }
    if wristband is not None:
        h["x-mcp-authorized"] = wristband
    return h


def _responses(praxis_url, user, token, tools, stream=False,
               input_text="What is the weather in NYC?", wristband=WRISTBAND):
    body = {
        "model": "scripted-model",
        "input": input_text,
        "store": False,
        "stream": stream,
        "tools": tools,
    }
    return httpx.post(praxis_url + "/v1/responses", headers=_headers(user, token, wristband),
                      json=body, timeout=90)


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


def _approval_request_id(resp_json):
    for o in resp_json.get("output") or []:
        if o.get("type") == "mcp_approval_request":
            return o.get("id")
    return None


def _has_final_answer(resp_json):
    return any(
        o.get("type") == "message"
        and o.get("role") == "assistant"
        and any(
            c.get("type") == "output_text" and "sunny, 21C" in c.get("text", "")
            for c in (o.get("content") or [])
        )
        for o in (resp_json.get("output") or [])
    )


def _nemo_count(port):
    return httpx.get(f"http://127.0.0.1:{port}/stats", timeout=5).json()["requests"]


def _full_flow_config(proxy_port, model_port, gateway_url, nemo_port, db_path):
    """Retarget the checked-in full-flow example to this test's local fixtures."""
    cfg = FULL_FLOW_CONFIG.read_text()

    def replace_exact(old, new, expected=1):
        nonlocal cfg
        actual = cfg.count(old)
        assert actual == expected, (
            f"expected {expected} occurrences of {old!r}, got {actual}"
        )
        cfg = cfg.replace(old, new)

    replace_exact('127.0.0.1:8080', f'127.0.0.1:{proxy_port}')
    replace_exact('127.0.0.1:3001', f'127.0.0.1:{model_port}', expected=6)
    replace_exact('http://127.0.0.1:8001/mcp', gateway_url)
    replace_exact(
        'sqlite://responses.db?mode=rwc',
        f'sqlite://{db_path}?mode=rwc',
        expected=2,
    )
    replace_exact(
        'http://127.0.0.1:3003/v1/checks',
        f'http://127.0.0.1:{nemo_port}/v1/checks',
    )
    return cfg


def _stop_process(proc):
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _post(praxis_url, user, token, body):
    return httpx.post(praxis_url + "/v1/responses", headers=_headers(user, token), json=body, timeout=90)


def _call_until(stack, praxis_url, user, token, predicate, tries=8):
    """Drive plain auto-exec calls until predicate(resp_json) holds — confirms an
    Authorino policy change actually took effect, instead of racing a fixed sleep."""
    j = None
    for _ in range(tries):
        _reset_backend(stack)
        r = _responses(praxis_url, user, token, [MCP_TOOL], stream=False)
        if r.status_code == 200:
            j = r.json()
            if predicate(j):
                return j
        time.sleep(1)
    return j


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
def passing_nemo():
    """The full-flow config screens tool results through NeMo; keep that real
    filter in the pipeline while using a deterministic passing endpoint."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(PASSING_NEMO)],
        env={**os.environ, "PORT": str(port)},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _wait_http(f"http://127.0.0.1:{port}/stats", 15)
        yield port
    finally:
        _stop_process(proc)


@pytest.fixture(scope="session")
def praxis(stack, scripted_model, passing_nemo):
    """Real praxis-ai running the canonical full-flow config against the gateway,
    scripted model, and passing NeMo fixture (Topology A, store:false)."""
    proxy_port = _free_port()
    tmpdir = tempfile.TemporaryDirectory(prefix="praxis-kuadrant-")
    cfg_path = Path(tmpdir.name) / "config.yaml"
    db_path = Path(tmpdir.name) / "responses.db"
    cfg_path.write_text(
        _full_flow_config(
            proxy_port, scripted_model, stack["KUADRANT_GATEWAY_URL"], passing_nemo, db_path
        )
    )
    proc = subprocess.Popen(
        [_find_binary(), "-c", str(cfg_path)],
        env={**os.environ, "WEB_SEARCH_API_KEY": "unused-kuadrant-test-key"},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _wait_tcp(proxy_port, 30)
        yield f"http://127.0.0.1:{proxy_port}"
    finally:
        _stop_process(proc)
        tmpdir.cleanup()


@pytest.fixture(scope="session")
def praxis_store(stack, scripted_model, passing_nemo):
    """Full-flow Praxis process with its configured SQLite stores, so approval
    create/resume survives two client turns (items 5/6b)."""
    proxy_port = _free_port()
    tmpdir = tempfile.TemporaryDirectory(prefix="praxis-kuadrant-store-")
    cfg_path = Path(tmpdir.name) / "config.yaml"
    db_path = Path(tmpdir.name) / "responses.db"
    cfg_path.write_text(
        _full_flow_config(
            proxy_port, scripted_model, stack["KUADRANT_GATEWAY_URL"], passing_nemo, db_path
        )
    )
    proc = subprocess.Popen(
        [_find_binary(), "-c", str(cfg_path)],
        env={**os.environ, "WEB_SEARCH_API_KEY": "unused-kuadrant-test-key"},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _wait_tcp(proxy_port, 30)
        yield f"http://127.0.0.1:{proxy_port}"
    finally:
        _stop_process(proc)
        tmpdir.cleanup()


# --- tests -------------------------------------------------------------------

def test_item1_authorized_user_discovers_and_invokes_buffered(
    stack, praxis, scripted_model, passing_nemo
):
    """userA (tool:get_forecast) discovers weather_get_forecast, the agentic loop
    dispatches it to the gateway, and the real backend executes exactly once."""
    token = _mint(stack, "userA", ROLES_AUTHORIZED)
    _reset_backend(stack)
    httpx.post(f"http://127.0.0.1:{passing_nemo}/reset", timeout=5)
    r = _responses(praxis, "userA", token, [MCP_TOOL], stream=False)
    assert r.status_code == 200, r.text
    j = r.json()
    assert "weather_get_forecast" in _discovered_tools(j), j
    assert _mcp_call_ok(j), j
    assert _has_final_answer(j), "the full agentic loop must return to the model after MCP dispatch"
    assert _nemo_count(passing_nemo) >= 1, "full-flow ai_guardrails must screen the MCP result"
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
    # Praxis must not dispatch the undiscovered tool at all: no mcp_call item is
    # produced (distinguishes client-side non-dispatch from dispatched-then-denied).
    assert not any(o.get("type") == "mcp_call" for o in (j.get("output") or [])), j
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


def test_item1_authorized_streaming(stack, praxis, scripted_model, passing_nemo):
    """Streaming equivalent of item 1: userA discovers + invokes over SSE, the
    logical stream completes cleanly, and the backend executes once."""
    token = _mint(stack, "userA", ROLES_AUTHORIZED)
    _reset_backend(stack)
    httpx.post(f"http://127.0.0.1:{passing_nemo}/reset", timeout=5)
    status, raw = _responses_stream(praxis, "userA", token, [MCP_TOOL])
    assert status == 200, raw
    assert raw.count("event: response.created") == 1, raw
    assert raw.count("event: response.completed") == 1, raw
    out = _sse_final_output(raw)
    assert any(
        o.get("type") == "mcp_list_tools"
        and any(t.get("name") == "weather_get_forecast" for t in (o.get("tools") or []))
        for o in out
    ), raw
    assert any(o.get("type") == "mcp_call" and not o.get("error") and o.get("output") for o in out), raw
    assert _has_final_answer({"output": out}), "stream must complete the model-tool-model loop"
    assert _nemo_count(passing_nemo) >= 1, "full-flow ai_guardrails must screen the MCP result"
    assert _backend_count(stack, "get_forecast") == 1


def test_item3a_undiscovered_streaming(stack, praxis, scripted_model):
    """Streaming item 3a: a force-emitted call to an undiscovered tool is not
    dispatched; the logical stream still completes cleanly with zero execution."""
    token = _mint(stack, "userB", ROLES_NONE)
    _reset_backend(stack)
    status, raw = _responses_stream(
        praxis, "userB", token, [MCP_TOOL], input_text="force-call:weather_get_forecast"
    )
    assert status == 200, raw
    assert _sse_terminated_cleanly(raw), raw
    assert not _mcp_call_ok({"output": _sse_final_output(raw)}), raw
    assert _backend_count(stack, "get_forecast") == 0


def test_item3b_gateway_denial_streaming(stack, praxis, scripted_model):
    """Streaming item 3b: under open-discovery userB invokes the tool, the gateway
    denies it, the tool error is surfaced, the stream completes cleanly, and the
    backend never executes."""
    _switch_authconfig("opendisco")
    try:
        token = _mint(stack, "userB", ROLES_NONE)
        _reset_backend(stack)
        status, raw = _responses_stream(praxis, "userB", token, [MCP_TOOL])
        assert status == 200, raw
        assert _sse_terminated_cleanly(raw), raw
        assert _mcp_call_error({"output": _sse_final_output(raw)}), raw
        assert _backend_count(stack, "get_forecast") == 0
    finally:
        _switch_authconfig("filtered")


def test_item7_same_forwarded_identity_reaches_discovery_and_dispatch(stack, praxis, scripted_model):
    """At gateway ingress (before Authorino replaces the header), the same per-user
    Bearer and forwarded x-mcp-authorized reach BOTH discovery (tools/list) and
    dispatch (tools/call), compared as one-way digests — never logging raw creds."""
    token = _mint(stack, "userA", ROLES_AUTHORIZED)
    _reset_backend(stack)
    before = len(_ingress_lines(stack))
    r = _responses(praxis, "userA", token, [MCP_TOOL], stream=False)
    assert r.status_code == 200, r.text
    assert _mcp_call_ok(r.json()), r.text  # both discovery and dispatch occurred
    new = _ingress_since(
        stack, before,
        lambda ls: any(x["method"] == "tools/list" for x in ls) and any(x["method"] == "tools/call" for x in ls),
    )
    lists = [ln for ln in new if ln["method"] == "tools/list"]
    calls = [ln for ln in new if ln["method"] == "tools/call"]
    assert lists and calls, new
    assert lists[-1]["bearer"] == calls[-1]["bearer"] != "0", new
    assert lists[-1]["wristband"] == calls[-1]["wristband"] != "0", new


def test_item4a_invalid_bearer_rejected_at_gateway(stack, praxis, scripted_model):
    """A tampered per-user Bearer is forwarded opaquely by Praxis (no short-circuit)
    and rejected at the gateway (Authorino), so no tool is invoked and the backend
    never executes."""
    bad = _mint(stack, "userA", ROLES_AUTHORIZED, tamper=True)
    _reset_backend(stack)
    before = len(_ingress_lines(stack))
    r = _responses(praxis, "userA", bad, [MCP_TOOL], stream=False)
    # Praxis forwards the tampered bearer opaquely (no short-circuit): it reaches the
    # gateway, Authorino rejects it, and discovery fails — surfaced as a connector
    # resolve failure (502), never a successful invocation.
    assert r.status_code == 502, r.text
    assert "trusted-mcp" in r.text and "resolve" in r.text, r.text
    assert _ingress_since(stack, before, lambda ls: len(ls) > 0), \
        "the bad bearer must reach the gateway (no short-circuit)"
    assert _backend_count(stack, "get_forecast") == 0


def test_item4_missing_bearer_slot_fails_closed_at_praxis(stack, praxis, scripted_model):
    """A configured credential slot that resolves to nothing (ingress omits
    x-user-mcp-key) fails closed AT PRAXIS before any gateway dial — distinct from a
    present-but-invalid Bearer, which is forwarded (item 4a)."""
    _reset_backend(stack)
    before = len(_ingress_lines(stack))
    headers = {
        "Content-Type": "application/json",
        "x-auth-tenant": "acme",
        "x-auth-user": "userA",
        "x-mcp-authorized": WRISTBAND,  # assertion present; the bearer slot is the one missing
    }
    body = {"model": "scripted-model", "input": "weather?", "store": False, "stream": False, "tools": [MCP_TOOL]}
    r = httpx.post(praxis + "/v1/responses", headers=headers, json=body, timeout=30)
    # Praxis fails closed locally: no gateway ingress for this request, no execution.
    assert len(_ingress_lines(stack)) == before, "gateway must see no request (Praxis short-circuits)"
    assert _backend_count(stack, "get_forecast") == 0
    # and the request did not succeed in invoking the tool
    try:
        assert not _mcp_call_ok(r.json()), r.text
    except ValueError:
        pass  # non-JSON error body is also acceptable


def test_item4b_invalid_forwarded_wristband_is_not_an_enforcement_point(stack, praxis, scripted_model):
    """EXPLICITLY UNCOVERED in Topology A: a garbage forwarded x-mcp-authorized has no
    enforcement effect — the front-door Authorino mints its own wristband. With a
    valid Bearer the call still succeeds. (A broker-trust topology would instead yield
    an empty discovery list, not a 401/403; we do not claim 'rejected' here.)"""
    token = _mint(stack, "userA", ROLES_AUTHORIZED)
    _reset_backend(stack)
    r = _responses(praxis, "userA", token, [MCP_TOOL], stream=False, wristband="garbage-not-a-jwt")
    assert r.status_code == 200, r.text
    assert _mcp_call_ok(r.json()), r.text
    assert _backend_count(stack, "get_forecast") == 1


def test_item5_require_approval_never_does_not_bypass(stack, praxis, scripted_model):
    """With require_approval:never there is no approval step to exploit; an
    unauthorized call still reaches the gateway and is denied (exec==0)."""
    _switch_authconfig("opendisco")
    try:
        token = _mint(stack, "userB", ROLES_NONE)
        _reset_backend(stack)
        r = _responses(praxis, "userB", token, [MCP_TOOL], stream=False)  # MCP_TOOL => require_approval:never
        assert r.status_code == 200, r.text
        j = r.json()
        assert _approval_request_id(j) is None, j
        assert _mcp_call_error(j), j
        assert _backend_count(stack, "get_forecast") == 0
    finally:
        _switch_authconfig("filtered")


def test_item5_approved_resume_still_denied_when_unauthorized(stack, praxis_store, scripted_model):
    """An approved resume must NOT bypass authorization: userB creates an approval,
    approves it, the resume re-dials the gateway, and it is still denied (exec==0)."""
    _switch_authconfig("opendisco")
    try:
        token = _mint(stack, "userB", ROLES_NONE)
        _reset_backend(stack)
        t1 = _post(praxis_store, "userB", token, {
            "model": "scripted-model", "input": "weather?", "store": True, "stream": False,
            "tools": [APPROVAL_TOOL]})
        assert t1.status_code == 200, t1.text
        j1 = t1.json()
        aid = _approval_request_id(j1)
        assert aid, j1
        assert _backend_count(stack, "get_forecast") == 0  # no dispatch before approval
        t2 = _post(praxis_store, "userB", token, {
            "model": "scripted-model", "previous_response_id": j1.get("id"), "store": True, "stream": False,
            "tools": [APPROVAL_TOOL],
            "input": [{"type": "mcp_approval_response", "approval_request_id": aid, "approve": True}]})
        assert t2.status_code == 200, t2.text
        assert _mcp_call_error(t2.json()), t2.text
        assert _backend_count(stack, "get_forecast") == 0
    finally:
        _switch_authconfig("filtered")


def test_item5_approved_resume_still_denied_streaming(stack, praxis_store, scripted_model):
    """Streaming resume: the approved resume denial completes a clean SSE lifecycle,
    surfaces the tool error, and never executes the backend."""
    _switch_authconfig("opendisco")
    try:
        token = _mint(stack, "userB", ROLES_NONE)
        _reset_backend(stack)
        t1 = _post(praxis_store, "userB", token, {
            "model": "scripted-model", "input": "weather?", "store": True, "stream": False,
            "tools": [APPROVAL_TOOL]})
        assert t1.status_code == 200, t1.text
        j1 = t1.json()
        aid = _approval_request_id(j1)
        assert aid, j1
        body = {
            "model": "scripted-model", "previous_response_id": j1.get("id"), "store": True, "stream": True,
            "tools": [APPROVAL_TOOL],
            "input": [{"type": "mcp_approval_response", "approval_request_id": aid, "approve": True}]}
        raw = ""
        with httpx.stream("POST", praxis_store + "/v1/responses", headers=_headers("userB", token),
                          json=body, timeout=90) as r:
            assert r.status_code == 200, r.read()
            for chunk in r.iter_text():
                raw += chunk
        assert _sse_terminated_cleanly(raw), raw
        assert _mcp_call_error({"output": _sse_final_output(raw)}), raw
        assert _backend_count(stack, "get_forecast") == 0
    finally:
        _switch_authconfig("filtered")


def test_item6a_revoke_between_discovery_and_dispatch(stack, praxis, scripted_model):
    """Live-policy revocation: userA is authorized, then the operator revokes userA's
    get_forecast at the gateway. userA still DISCOVERS the tool (revocation is
    per-call, not discovery) but the dispatch is now denied — exec==0."""
    token = _mint(stack, "userA", ROLES_AUTHORIZED)
    _switch_authconfig("filtered")
    try:
        _reset_backend(stack)
        baseline = _responses(praxis, "userA", token, [MCP_TOOL], stream=False)
        assert _mcp_call_ok(baseline.json()), baseline.text
        assert _backend_count(stack, "get_forecast") == 1
        _switch_authconfig("revoked")
        j = _call_until(stack, praxis, "userA", token, _mcp_call_error)
        assert j is not None and _mcp_call_error(j), "revocation did not take effect"
        assert "weather_get_forecast" in _discovered_tools(j), j  # still discovered -> per-call revocation
        assert _backend_count(stack, "get_forecast") == 0
    finally:
        _switch_authconfig("filtered")


def test_item6a_revoke_streaming(stack, praxis, scripted_model):
    """Streaming equivalent of 6a: after revocation the streaming dispatch is denied
    with a clean terminal SSE lifecycle and zero execution."""
    token = _mint(stack, "userA", ROLES_AUTHORIZED)
    try:
        _switch_authconfig("revoked")
        assert _mcp_call_error(_call_until(stack, praxis, "userA", token, _mcp_call_error) or {}), \
            "revocation did not take effect"
        _reset_backend(stack)
        status, raw = _responses_stream(praxis, "userA", token, [MCP_TOOL])
        assert status == 200, raw
        assert _sse_terminated_cleanly(raw), raw
        assert _mcp_call_error({"output": _sse_final_output(raw)}), raw
        assert _backend_count(stack, "get_forecast") == 0
    finally:
        _switch_authconfig("filtered")


def test_item6b_revoke_between_approval_and_resume(stack, praxis, praxis_store, scripted_model):
    """Revoke between approval creation and resume: userA creates an approval (allowed),
    the operator revokes get_forecast, and the approved resume re-dials the gateway and
    is now denied — the approval was validly consumed but authorization still denies."""
    token = _mint(stack, "userA", ROLES_AUTHORIZED)
    _switch_authconfig("filtered")
    try:
        _reset_backend(stack)
        t1 = _post(praxis_store, "userA", token, {
            "model": "scripted-model", "input": "weather?", "store": True, "stream": False,
            "tools": [APPROVAL_TOOL]})
        assert t1.status_code == 200, t1.text
        j1 = t1.json()
        aid = _approval_request_id(j1)
        assert aid, j1
        assert _backend_count(stack, "get_forecast") == 0
        _switch_authconfig("revoked")
        assert _mcp_call_error(_call_until(stack, praxis, "userA", token, _mcp_call_error) or {}), \
            "revocation did not take effect before resume"
        _reset_backend(stack)
        t2 = _post(praxis_store, "userA", token, {
            "model": "scripted-model", "previous_response_id": j1.get("id"), "store": True, "stream": False,
            "tools": [APPROVAL_TOOL],
            "input": [{"type": "mcp_approval_response", "approval_request_id": aid, "approve": True}]})
        assert t2.status_code == 200, t2.text
        assert _mcp_call_error(t2.json()), t2.text
        assert _backend_count(stack, "get_forecast") == 0
    finally:
        _switch_authconfig("filtered")


def test_item6b_revoke_resume_streaming(stack, praxis, praxis_store, scripted_model):
    """Streaming equivalent of 6b: the revoked approved-resume denial streams a clean
    terminal SSE lifecycle, surfaces the tool error, and never executes."""
    token = _mint(stack, "userA", ROLES_AUTHORIZED)
    _switch_authconfig("filtered")
    try:
        _reset_backend(stack)
        t1 = _post(praxis_store, "userA", token, {
            "model": "scripted-model", "input": "weather?", "store": True, "stream": False,
            "tools": [APPROVAL_TOOL]})
        assert t1.status_code == 200, t1.text
        j1 = t1.json()
        aid = _approval_request_id(j1)
        assert aid, j1
        _switch_authconfig("revoked")
        assert _mcp_call_error(_call_until(stack, praxis, "userA", token, _mcp_call_error) or {}), \
            "revocation did not take effect before resume"
        _reset_backend(stack)
        body = {
            "model": "scripted-model", "previous_response_id": j1.get("id"), "store": True, "stream": True,
            "tools": [APPROVAL_TOOL],
            "input": [{"type": "mcp_approval_response", "approval_request_id": aid, "approve": True}]}
        raw = ""
        with httpx.stream("POST", praxis_store + "/v1/responses", headers=_headers("userA", token),
                          json=body, timeout=90) as r:
            assert r.status_code == 200, r.read()
            for chunk in r.iter_text():
                raw += chunk
        assert _sse_terminated_cleanly(raw), raw
        assert _mcp_call_error({"output": _sse_final_output(raw)}), raw
        assert _backend_count(stack, "get_forecast") == 0
    finally:
        _switch_authconfig("filtered")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
