# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27", "openai>=2,<4", "pytest>=8", "pyyaml>=6"]
# ///
"""Controlled gateway evidence for image translation and explicit compaction.

The backend deliberately does not infer anything. These tests prove the real
Praxis filters satisfy the upstream wire checks, not live model compliance.
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openai import OpenAI, BadRequestError
import pytest

from run_responses_acceptance import gateway, run_template
from test_responses_conformance import (
    REPO_ROOT, _ensure_suite, _find_binary, _free_port, _run_compliance, _wait_for_proxy,
)


@pytest.fixture(scope="module")
def acceptance_gateway(tmp_path_factory):
    captured = []

    class Backend(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            captured.append((self.path, body))
            if body["model"] == "slow-acceptance-model":
                time.sleep(0.25)
            payload = json.dumps({
                "id": "chatcmpl_controlled", "object": "chat.completion",
                "model": body["model"],
                "choices": [{"index": 0, "message": {
                    "role": "assistant", "content": "Controlled backend response."
                }, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    directory = tmp_path_factory.mktemp("openresponses-acceptance")
    config = (REPO_ROOT / "tests/conformance/openresponses/managed.yaml").read_text()
    port = _free_port()
    config = config.replace("127.0.0.1:8080", f"127.0.0.1:{port}")
    config = config.replace("127.0.0.1:3001", f"127.0.0.1:{backend.server_port}")
    config = config.replace("sqlite://acceptance.db?mode=rwc", f"sqlite://{directory}/responses.db?mode=rwc")
    config_path = directory / "praxis.yaml"
    config_path.write_text(config)
    log_path = directory / "praxis.log"
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen([_find_binary(), "-c", str(config_path)], stdout=log, stderr=subprocess.STDOUT)
            try:
                _wait_for_proxy(port, process, log_path)
                yield f"http://127.0.0.1:{port}/v1", captured
            finally:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    finally:
        backend.shutdown()
        backend.server_close()
        thread.join()


@pytest.mark.parametrize("template_id", ["image-input", "compact-response", "compact-missing-model"])
def test_upstream_http_acceptance_with_controlled_backend(acceptance_gateway, monkeypatch, template_id):
    base_url, captured = acceptance_gateway
    monkeypatch.setenv("OPENRESPONSES_API_KEY", "acceptance-dummy")
    before = len(captured)
    result = run_template(_ensure_suite(), base_url, "acceptance-model", {
        "id": template_id, "kind": "endpoint", "transport": "http",
    })
    assert result["status"] == "passed", result.get("errors")
    forwarded = captured[before:]
    if template_id == "compact-missing-model":
        assert forwarded == [], "missing model must fail before a backend call"
    else:
        assert len(forwarded) == 1
        assert forwarded[0][0] == "/v1/chat/completions"
    if template_id == "image-input":
        parts = forwarded[0][1]["messages"][0]["content"]
        image = next(part for part in parts if part["type"] == "image_url")
        assert image["image_url"]["url"].startswith("data:image/png;base64,")


def test_openai_sdk_image_and_compact(acceptance_gateway):
    base_url, captured = acceptance_gateway
    with OpenAI(base_url=base_url, api_key="acceptance-dummy", max_retries=0) as client:
        response = client.responses.create(model="acceptance-model", input=[{
            "role": "user", "content": [
                {"type": "input_text", "text": "Describe the image."},
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "low"},
            ],
        }])
        assert response.output_text == "Controlled backend response."
        assert captured[-1][1]["messages"][0]["content"][1]["image_url"]["detail"] == "low"
        compacted = client.responses.compact(model="acceptance-model", input=[{
            "role": "user", "content": "Remember Tuesday.",
        }])
        assert compacted.object == "response.compaction"
        assert any(item.type == "compaction" for item in compacted.output)
        before = len(captured)
        with pytest.raises(BadRequestError):
            client.post("/responses/compact", cast_to=dict, body={"input": "Remember Tuesday."})
        assert len(captured) == before


def test_cpu_request_timeout_does_not_change_validation(acceptance_gateway, monkeypatch):
    """A short deadline fails; the CPU budget admits the identical valid response."""
    import test_responses_conformance
    from urllib.parse import urlsplit

    base_url, _captured = acceptance_gateway
    port = urlsplit(base_url).port
    monkeypatch.setattr(test_responses_conformance, "VLLM_MODEL", "slow-acceptance-model")
    suite = _ensure_suite()
    short = _run_compliance(suite, port, {"basic-response"}, timeout_ms=25)
    assert short.returncode != 0
    failed = json.loads(short.stdout)["results"][0]
    assert "timed out" in " ".join(failed["errors"])
    normal = _run_compliance(suite, port, {"basic-response"})
    assert normal.returncode == 0, normal.stdout + normal.stderr
    assert json.loads(normal.stdout)["summary"] == {
        "passed": 1, "failed": 0, "skipped": 0, "total": 1,
    }


def test_live_image_acceptance(monkeypatch):
    """Run the upstream image template on the existing real vision GPU lane."""
    names = ("PRAXIS_TEST_VLLM_BASE_URL", "PRAXIS_TEST_VLLM_MODEL", "VLLM_API_KEY")
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        message = f"live image acceptance requires {', '.join(missing)}"
        if os.environ.get("PRAXIS_TEST_REQUIRE_LIVE") == "1":
            pytest.fail(message)
        pytest.skip(message)
    monkeypatch.setenv("OPENRESPONSES_API_KEY", os.environ["VLLM_API_KEY"])
    model = os.environ["PRAXIS_TEST_VLLM_MODEL"]
    with gateway("managed", os.environ["PRAXIS_TEST_VLLM_BASE_URL"], model) as (base_url, _info):
        result = run_template(_ensure_suite(), base_url, model, {
            "id": "image-input", "kind": "endpoint", "transport": "http",
        })
        assert result["status"] == "passed", result.get("errors")
        with OpenAI(base_url=base_url, api_key=os.environ["VLLM_API_KEY"], max_retries=0) as client:
            response = client.responses.create(model=model, input=result["request"]["input"])
            assert response.status == "completed"
            assert response.output_text, "the SDK must receive the vision model's answer"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
