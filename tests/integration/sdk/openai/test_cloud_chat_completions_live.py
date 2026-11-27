# /// script
# requires-python = ">=3.11"
# dependencies = ["openai==3.49.0", "pytest==8.4.2"]
# ///
"""Opt-in Azure and Vertex Chat Completions smoke through Praxis."""

import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

import pytest
from openai import OpenAI


def identifier(name):
    value = os.environ[name]
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        pytest.fail(f"Invalid {name}")
    return value


def gateway_config(provider, model, port):
    if provider == "azure":
        host = identifier("AZURE_OPENAI_RESOURCE") + ".openai.azure.com"
        filters = [
            {"filter": "path_rewrite", "replace": {
                "pattern": "^/v1/chat/completions$",
                "replacement": f"/openai/deployments/{model}/chat/completions",
            }},
            {"filter": "openai_chat_completions_to_azureai_chat_completions",
             "api_version": "2024-10-21"},
        ]
        header, prefix = "api-key", ""
    else:
        host = "aiplatform.googleapis.com"
        filters = [{
            "filter": "openai_chat_completions_to_vertexai_gemini",
            "project": identifier("VERTEX_AI_PROJECT"),
            "region": "global",
        }]
        header, prefix = "Authorization", "Bearer "

    filters.extend([
        {"filter": "router", "routes": [{"path_prefix": "/", "cluster": provider}]},
        {"filter": "credential_injection", "clusters": [{
            "name": provider, "header": header, "header_prefix": prefix,
            "env_var": "PRAXIS_CLOUD_SMOKE_CREDENTIAL", "strip_client_credential": True,
        }]},
        {"filter": "load_balancer", "clusters": [{
            "name": provider, "endpoints": [f"{host}:443"],
            "http": {"authority": host}, "tls": {"sni": host},
        }]},
    ])
    return {
        "listeners": [{"name": "smoke", "address": f"127.0.0.1:{port}",
                       "filter_chains": [provider]}],
        "filter_chains": [{"name": provider, "filters": filters}],
        "shutdown_timeout_secs": 1,
    }


def wait_for_gateway(process, port):
    deadline = time.monotonic() + 15
    while process.poll() is None and time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    pytest.fail("Praxis gateway did not start")


def test_live_cloud_chat_completion(tmp_path):
    provider = os.getenv("PRAXIS_CLOUD_SMOKE_PROVIDER")
    if not provider:
        pytest.skip("Enable a live cloud provider explicitly")
    assert provider in {"azure", "vertex"}
    model = identifier("PRAXIS_CLOUD_SMOKE_MODEL")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    config = tmp_path / "praxis.json"
    config.write_text(json.dumps(gateway_config(provider, model, port)))
    binary = Path("target/debug/praxis-ai").resolve()
    with subprocess.Popen(
        [str(binary), "-c", str(config)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) as gateway:
        try:
            wait_for_gateway(gateway, port)
            with OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="gateway-test",
                        max_retries=0, timeout=60) as client:
                response = client.chat.completions.create(
                    model=model, max_completion_tokens=64,
                    messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                )
            choice, = response.choices
            assert choice.message.role == "assistant"
            assert choice.message.content
            assert response.usage.prompt_tokens > 0
            assert response.usage.completion_tokens > 0
        finally:
            gateway.terminate()
            try:
                gateway.wait(timeout=10)
            except subprocess.TimeoutExpired:
                gateway.kill()
                gateway.wait()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
