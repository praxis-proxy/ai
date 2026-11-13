# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27", "pytest>=8.0", "pyyaml>=6.0"]
# ///
"""Credential-free checks that acceptance evidence cannot silently omit tests."""

import json
import subprocess

import pytest

from run_responses_acceptance import configure_gateway, redact, run_template, summarize


def test_https_backend_configuration_preserves_pipeline(tmp_path):
    config = configure_gateway("managed", "https://backend.example/v1", "vision-model", 8081, tmp_path)
    filters = config["filter_chains"][0]["filters"]
    compact = next(item for item in filters if item["filter"] == "openai_responses_compact")
    assert compact["inference_url"] == "https://backend.example/v1/chat/completions"
    assert compact["default_model"] == "vision-model"
    websocket = filters[1]["branch_chains"][0]["chains"][0]["filters"][-1]["clusters"][0]
    assert websocket["endpoints"] == ["backend.example:443"]
    assert websocket["tls"] == {"sni": "backend.example"}
    assert websocket["http"]["authority"] == "backend.example"


@pytest.mark.parametrize("url", ["https://user:key@host", "http://host/prefix", "http://host?key=secret"])
def test_backend_credentials_and_path_prefixes_are_rejected(tmp_path, url):
    with pytest.raises(ValueError):
        configure_gateway("managed", url, "model", 8081, tmp_path)


@pytest.mark.parametrize("status", ["failed", "skipped"])
def test_nonpassing_template_prevents_all_passed(status):
    assert not summarize([{"status": "passed"}, {"status": status}])["all_passed"]
    assert not summarize([])["all_passed"]


@pytest.mark.parametrize("payload,exit_code", [
    ({"results": []}, 0),
    ({"results": [{"id": "wrong", "status": "passed"}]}, 0),
    ({"results": [{"id": "basic-response", "status": "passed"}]}, 1),
    ({"results": [{"id": "basic-response", "status": "pending"}]}, 0),
    ({"results": [{"id": "basic-response", "status": "passed"}] * 2}, 0),
])
def test_invalid_cli_evidence_fails(monkeypatch, tmp_path, payload, exit_code):
    def invoke(command, **kwargs):
        assert "--api-key" not in command
        return subprocess.CompletedProcess(command, exit_code, json.dumps(payload), "")

    monkeypatch.setattr(subprocess, "run", invoke)
    result = run_template(tmp_path, "http://localhost/v1", "model", {
        "id": "basic-response", "kind": "endpoint", "transport": "http",
    })
    assert result["status"] == "failed"
    assert result["errors"]


def test_credential_echo_is_redacted_recursively():
    value = {"response": [{"error": 'Bearer secret-123 with "quotes"'}]}
    assert "secret-123" not in json.dumps(redact(value, "secret-123"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
