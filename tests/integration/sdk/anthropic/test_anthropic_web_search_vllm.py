#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "anthropic==1.9.0",
#     "pytest>=8.0",
# ]
# ///
"""Official Anthropic SDK -> Praxis -> real vLLM hosted web-search acceptance.

Drives the shipped ``anthropic/web-search-to-openai-vllm.yaml`` example — the
server-owned Anthropic ``WebSearch`` loop translated onto a Chat-Completions-only
vLLM backend — against a live model in two provider modes:

    * Mode A (``test_web_search_loop_with_stubbed_provider``): a live vLLM model
      with a LOCAL body-authenticated search stub that fabricates a fixed result.
      Deterministic and secret-free, so it proves the request-side wiring — forced
      ``tool_choice`` -> translation -> managed classification -> provider dispatch
      — with the credential travelling in the Tavily request body.
    * Mode B (``test_live_tavily_web_search_returns_real_sources``): the same loop
      against the REAL Tavily provider, exercising the live provider request
      format and response parsing end to end.

Observability for Mode B. The managed filter appends the search result only into
the re-entry request and SUPPRESSES the ``WebSearch`` tool-use block, so — unlike
the OpenAI Responses web-search test — the client response carries no
``web_search_call`` sources to assert on. Worse, when the provider fails (401,
429, timeout, schema drift) the filter appends an ``is_error`` tool result and the
model still emits a plausible text answer, so asserting only "a non-empty answer
came back" would pass on a broken Tavily integration. Mode B therefore points the
provider ``base_url`` at a loopback relay that forwards each search to the real
Tavily API and CAPTURES the upstream response, then asserts an observed successful
provider result (HTTP 200 with parsed absolute-URL sources) plus the resolved real
key in the request body. Only the ``base_url`` constant is bypassed; the real key,
network, request format, and response shape are all exercised.

Determinism comes from forcing, not prompting: the client declares a ``WebSearch``
tool and forces it with ``tool_choice={"type":"tool","name":"WebSearch"}``. In the
translated pipeline this lowers to a forced Chat Completions function call, which
vLLM enforces through constrained decoding even on a small model. On the first
re-entry the filter relaxes ``tool_choice`` to ``auto`` so the model can emit a
final text answer and the loop terminates.

Required: PRAXIS_TEST_VLLM_BASE_URL (local http URL), PRAXIS_TEST_VLLM_MODEL,
VLLM_API_KEY, and a built praxis-ai binary (or PRAXIS_AI_BIN). Set
PRAXIS_TEST_REQUIRE_LIVE=1 in CI so missing infrastructure fails instead of
reporting a misleading green run. Mode B additionally requires TAVILY_API_KEY;
PRAXIS_TEST_REQUIRE_LIVE_WEB_SEARCH=1 turns a missing key into a failure rather
than a skip.
"""

import json
import os
from pathlib import Path
import signal
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import pytest
from anthropic import Anthropic


ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = ROOT / "examples/configs/anthropic/web-search-to-openai-vllm.yaml"
STUB_SEARCH_KEY = "test-key"
TAVILY_UPSTREAM = "https://api.tavily.com/search"


def _required_live() -> bool:
    return os.environ.get("PRAXIS_TEST_REQUIRE_LIVE", "").lower() in {"1", "true"}


def _require_live_web_search() -> bool:
    return os.environ.get("PRAXIS_TEST_REQUIRE_LIVE_WEB_SEARCH") == "1"


def _live_config() -> tuple[str, str, str]:
    required = ("PRAXIS_TEST_VLLM_BASE_URL", "PRAXIS_TEST_VLLM_MODEL", "VLLM_API_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        message = f"live vLLM web-search SDK test requires {', '.join(missing)}"
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


# ---------------------------------------------------------------------------
# Local Tavily-shaped search stub (Mode A: fabricates a fixed result)
# ---------------------------------------------------------------------------


class _TavilyStub:
    """A threaded body-authenticated Tavily stub exposing its captured bodies."""

    def __init__(self):
        self.port = _free_port()
        self.requests: list[dict] = []
        captured = self.requests

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):  # noqa: N802 (http.server API)
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b""
                try:
                    captured.append(json.loads(raw))
                except json.JSONDecodeError:
                    captured.append({})
                body = json.dumps(
                    {
                        "results": [
                            {
                                "title": "Potato - Wikipedia",
                                "url": "https://en.wikipedia.org/wiki/Potato",
                                "content": "The potato is a starchy tuber native to the Americas.",
                            }
                        ]
                    }
                ).encode()
                # Close each connection so the proxy never pools a stale keep-alive
                # socket to this stub across the loop's search callouts.
                self.close_connection = True
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> "_TavilyStub":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


# ---------------------------------------------------------------------------
# Local Tavily relay (Mode B: forwards to real Tavily and captures the result)
# ---------------------------------------------------------------------------


class _TavilyRelay:
    """Forward each search to the real Tavily API and capture the upstream result.

    Each captured entry is ``{"request": <sent body>, "status": <upstream code>,
    "response": <parsed upstream body>}``. A failed forward (auth, rate limit,
    timeout, TLS, schema drift) is captured with a non-200 status and surfaced to
    Praxis, so Mode B can assert an observed *successful* provider result rather
    than trusting the model's fallback answer.
    """

    def __init__(self):
        self.port = _free_port()
        self.captured: list[dict] = []
        captured = self.captured
        context = ssl.create_default_context()

        def relay(raw: bytes) -> tuple[int, bytes]:
            request = urllib.request.Request(
                TAVILY_UPSTREAM, data=raw, headers={"Content-Type": "application/json"}, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=20, context=context) as response:
                    return response.status, response.read()
            except urllib.error.HTTPError as exc:
                # Surface the real upstream status (e.g. 401/429) so a broken
                # credential or quota fails the run instead of passing silently.
                return exc.code, exc.read()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                return 599, json.dumps({"error": f"tavily relay failed: {exc}"}).encode()

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):  # noqa: N802 (http.server API)
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b""
                try:
                    request_body = json.loads(raw)
                except json.JSONDecodeError:
                    request_body = {}
                status, payload = relay(raw)
                try:
                    response_body = json.loads(payload)
                except json.JSONDecodeError:
                    response_body = payload.decode("utf-8", "replace")
                captured.append({"request": request_body, "status": status, "response": response_body})
                self.close_connection = True
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> "_TavilyRelay":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _write_config(proxy_port: int, authority: str, search_port: int, *, stub_key: bool) -> str:
    """Retarget the shipped example to live vLLM and a loopback search endpoint.

    ``stub_key`` swaps the ``${WEB_SEARCH_API_KEY}`` placeholder for a known literal
    (Mode A); otherwise the placeholder is left intact so the real Tavily key is
    resolved from the environment (Mode B). In both modes the provider ``base_url``
    is pointed at ``search_port`` on loopback, so the executor's SSRF opt-in is
    required.
    """
    config = CONFIG_PATH.read_text()

    assert config.count('address: "127.0.0.1:8080"') == 1, CONFIG_PATH
    config = config.replace('address: "127.0.0.1:8080"', f'address: "127.0.0.1:{proxy_port}"')

    assert config.count('"127.0.0.1:8000"') == 1, CONFIG_PATH
    config = config.replace('"127.0.0.1:8000"', f'"{authority}"')

    key_line = f"api_key: {STUB_SEARCH_KEY}" if stub_key else "api_key: ${WEB_SEARCH_API_KEY}"
    replaced = config.replace(
        "api_key: ${WEB_SEARCH_API_KEY}",
        f"{key_line}\n                base_url: http://127.0.0.1:{search_port}",
    )
    assert replaced != config, "example drift: provider api_key not found"
    config = replaced

    # The loopback search endpoint needs the executor's SSRF opt-in.
    replaced = config.replace(
        "allow_private_endpoints: true",
        "allow_private_endpoints: true\n  allow_private_upstreams: true",
    )
    assert replaced != config, "example drift: insecure_options not found"
    config = replaced

    # The ephemeral GPU runner runs as root; opt in only in this generated config.
    if os.geteuid() == 0 and "allow_root:" not in config:
        assert config.count("insecure_options:\n") == 1, CONFIG_PATH
        config = config.replace("insecure_options:\n", "insecure_options:\n  allow_root: true\n", 1)

    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(config)
    return path


def _start_proxy(config_path: str, extra_env: dict[str, str]):
    binary = _binary()
    fd, log_path = tempfile.mkstemp(suffix=".log")
    log_file = os.fdopen(fd, "w")
    environment = os.environ.copy()
    environment.update(extra_env)
    proc = subprocess.Popen(
        [binary, "-c", config_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    return proc, log_file, Path(log_path)


def _messages_kwargs(model: str) -> dict:
    return {
        "model": model,
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Use web search to look up potato, then summarize in one sentence."}
        ],
        "tools": [
            {
                "name": "WebSearch",
                "description": "Search the web",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
        # Force the first round so a small model reliably opens the managed loop;
        # the filter relaxes tool_choice to auto on re-entry so it can then answer.
        "tool_choice": {"type": "tool", "name": "WebSearch"},
    }


def _assert_terminal_message(response) -> str:
    """Assert one clean terminal Anthropic message with no leaked managed block."""
    assert response.type == "message", response
    assert response.role == "assistant", response
    text_blocks = [block for block in response.content if block.type == "text" and block.text.strip()]
    assert text_blocks, f"expected a non-empty text answer: {response.model_dump_json()}"
    # The managed WebSearch tool-use block is resolved server-side; none may leak.
    assert all(block.type != "tool_use" for block in response.content), response.model_dump_json()
    return " ".join(block.text for block in text_blocks)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stubbed_search_stack(request):
    """Live vLLM model + local body-authenticated Tavily stub (Mode A)."""
    authority, model, backend_key = _live_config()
    search = _TavilyStub().start()
    proxy_port = _free_port()
    config_path = _write_config(proxy_port, authority, search.port, stub_key=True)
    proc, log_file, log_path = _start_proxy(config_path, {"VLLM_API_KEY": backend_key})
    started = False
    try:
        _wait_for_proxy(proxy_port, proc, log_path)
        started = True
        yield {"proxy_port": proxy_port, "model": model, "search": search}
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        search.stop()
        if not started or request.session.testsfailed > 0:
            print(f"\n=== Praxis logs ===\n{log_path.read_text()}", file=sys.stderr)
        os.unlink(config_path)
        os.unlink(log_path)


@pytest.fixture(scope="module")
def live_tavily_stack(request):
    """Live vLLM model + real Tavily provider observed through a capture relay (Mode B)."""
    authority, model, backend_key = _live_config()
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_key:
        message = "TAVILY_API_KEY is required for credentialed live web search"
        if _require_live_web_search():
            pytest.fail(message)
        pytest.skip(message)

    relay = _TavilyRelay().start()
    proxy_port = _free_port()
    config_path = _write_config(proxy_port, authority, relay.port, stub_key=False)
    proc, log_file, log_path = _start_proxy(
        config_path, {"VLLM_API_KEY": backend_key, "WEB_SEARCH_API_KEY": tavily_key}
    )
    started = False
    try:
        _wait_for_proxy(proxy_port, proc, log_path)
        started = True
        yield {"proxy_port": proxy_port, "model": model, "relay": relay, "tavily_key": tavily_key}
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        relay.stop()
        if not started or request.session.testsfailed > 0:
            print(f"\n=== Praxis logs ===\n{log_path.read_text()}", file=sys.stderr)
        os.unlink(config_path)
        os.unlink(log_path)


def _client(proxy_port: int) -> Anthropic:
    return Anthropic(
        base_url=f"http://127.0.0.1:{proxy_port}",
        api_key="sdk-client-key-must-not-reach-vllm",
        max_retries=0,
        timeout=180,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_web_search_loop_with_stubbed_provider(stubbed_search_stack):
    """Live model forces a WebSearch call; the local stub answers it (Mode A)."""
    search = stubbed_search_stack["search"]
    search.requests.clear()
    client = _client(stubbed_search_stack["proxy_port"])
    try:
        response = client.messages.create(**_messages_kwargs(stubbed_search_stack["model"]))
    finally:
        client.close()

    text = _assert_terminal_message(response)
    assert text, response.model_dump_json()

    # The managed loop dispatched at least one provider callout, body-authenticated
    # with the configured key and a non-empty reconstructed query.
    assert search.requests, "the forced tool call must dispatch a managed search"
    first = search.requests[0]
    assert first.get("api_key") == STUB_SEARCH_KEY, f"Tavily key must travel in the body: {first}"
    assert isinstance(first.get("query"), str) and first["query"].strip(), f"query must be populated: {first}"


def test_live_tavily_web_search_returns_real_sources(live_tavily_stack):
    """The managed loop runs one REAL Tavily search, observed end to end (Mode B).

    The managed filter suppresses the ``WebSearch`` tool-use block and falls back
    to an ``is_error`` tool result on provider failure, so a terminal text answer
    alone does not prove the integration works. The capture relay lets the test
    assert the observed provider result: the resolved real key travelled in the
    body, Tavily returned HTTP 200, and the parsed payload carried real
    absolute-URL sources.
    """
    relay = live_tavily_stack["relay"]
    relay.captured.clear()
    client = _client(live_tavily_stack["proxy_port"])
    try:
        response = client.messages.create(**_messages_kwargs(live_tavily_stack["model"]))
    finally:
        client.close()

    # The loop still completed with a clean terminal answer.
    text = _assert_terminal_message(response)
    assert text, response.model_dump_json()

    # An observed, SUCCESSFUL real Tavily result — the crux of Mode B.
    assert relay.captured, "the forced tool call must dispatch a real Tavily search"
    result = relay.captured[0]
    assert result["status"] == 200, f"real Tavily must return 200, not {result['status']}: {result['response']}"

    sent = result["request"]
    assert sent.get("api_key") == live_tavily_stack["tavily_key"], "the resolved real key must travel in the body"
    assert isinstance(sent.get("query"), str) and sent["query"].strip(), f"query must be populated: {sent}"

    upstream = result["response"]
    assert isinstance(upstream, dict), f"Tavily response must be JSON: {upstream}"
    sources = upstream.get("results")
    assert isinstance(sources, list) and sources, f"Tavily returned no sources: {upstream}"
    assert all(
        isinstance(source.get("url"), str) and source["url"].startswith(("http://", "https://"))
        for source in sources
    ), f"every real source needs an absolute URL: {sources}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))
