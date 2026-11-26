#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["anthropic>=1.17,<3", "openai>=3.26,<4", "pytest>=8.0", "pyyaml>=6.0"]
# ///
"""Exercise model classification through both official clients and the example."""

import json
import signal
import subprocess
import threading
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml
from anthropic import Anthropic
from conftest import find_binary, free_port, wait_for_proxy
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[3]
OPENAI_MODEL = {"id": "gpt-test", "object": "model", "created": 0, "owned_by": "stub"}
ANTHROPIC_MODEL = {
    "id": "claude-test",
    "type": "model",
    "created_at": "2026-01-01T00:00:00Z",
    "display_name": "Claude Test",
}


def backend(stack, model, query, listing):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == "/v1/models" and parse_qs(url.query) == query:
                body = listing
            elif url.path == f"/v1/models/{model['id']}":
                body = model
            else:
                self.send_error(404)
                return
            encoded = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = stack.enter_context(ThreadingHTTPServer(("127.0.0.1", 0), Handler))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    stack.callback(thread.join)
    stack.callback(server.shutdown)
    return f"127.0.0.1:{server.server_port}"


@pytest.fixture(scope="module")
def gateway(tmp_path_factory):
    example = ROOT / "examples/configs/openai/operation-classifier.yaml"
    config = yaml.safe_load(example.read_text())
    config["shutdown_timeout_secs"] = 5
    with ExitStack() as stack:
        openai = backend(stack, OPENAI_MODEL, {}, {"object": "list", "data": [OPENAI_MODEL]})
        anthropic = backend(
            stack,
            ANTHROPIC_MODEL,
            {"limit": ["1"], "after_id": ["claude-prev"]},
            {
                "data": [ANTHROPIC_MODEL],
                "first_id": "claude-test",
                "last_id": "claude-test",
                "has_more": False,
            },
        )
        endpoints = {"openai-models-backend": openai, "anthropic-backend": anthropic}
        port = free_port()
        config["listeners"][0]["address"] = f"127.0.0.1:{port}"
        for cluster in config["filter_chains"][0]["filters"][-1]["clusters"]:
            endpoint = endpoints.get(cluster["name"])
            if endpoint is None:
                # The default route must fail if classification regresses.
                endpoint = f"127.0.0.1:{free_port()}"
            cluster["endpoints"] = [endpoint]
        path = tmp_path_factory.mktemp("model-operations") / "proxy.yaml"
        path.write_text(yaml.safe_dump(config))
        with subprocess.Popen([find_binary(), "-c", str(path)]) as proxy:
            try:
                wait_for_proxy(port)
                yield f"http://127.0.0.1:{port}"
            finally:
                proxy.send_signal(signal.SIGINT)
                try:
                    proxy.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proxy.kill()


def test_openai_model_list_and_retrieval(gateway):
    with OpenAI(api_key="test", base_url=f"{gateway}/v1", max_retries=0, timeout=10) as client:
        assert [model.id for model in client.models.list()] == ["gpt-test"]
        assert client.models.retrieve("gpt-test").owned_by == "stub"


def test_anthropic_model_list_and_retrieval(gateway):
    with Anthropic(api_key="test", base_url=gateway, max_retries=0, timeout=10) as client:
        page = client.models.list(limit=1, after_id="claude-prev")
        assert [model.id for model in page.data] == ["claude-test"]
        assert client.models.retrieve("claude-test").display_name == "Claude Test"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
