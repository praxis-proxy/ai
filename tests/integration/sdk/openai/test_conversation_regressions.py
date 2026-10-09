#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27", "openai>=2.0", "pytest>=8.0"]
# ///
"""PR #1362 conversation regressions against loopback-only SDK backends.

Run after building praxis-ai with full,store-sqlite features:
    uv run tests/integration/sdk/openai/test_conversation_regressions.py -v
"""

import http.server
import json
import signal
import subprocess
import threading

import pytest
from openai import OpenAI
from test_openai_conversations import (
    OWNER_HEADER,
    _chunked_response_store_filters,
    _find_binary,
    _free_port,
    _owner_assertion,
    _proxy_env,
    _wait_for_proxy,
    _wait_for_store_ready,
)


@pytest.fixture(params=[False, True], ids=["native", "translated"])
def conversation_backend(request, tmp_path):
    """Capture inference requests without contacting a model or using credentials."""
    captured = []
    output = []
    translated = request.param

    class Backend(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            if self.headers.get("Transfer-Encoding") == "chunked":
                chunks = []
                while size := int(self.rfile.readline().strip(), 16):
                    chunks.append(self.rfile.read(size))
                    self.rfile.read(2)
                self.rfile.readline()
                data = b"".join(chunks)
            else:
                data = self.rfile.read(int(self.headers["Content-Length"]))
            body = json.loads(data)
            captured.append((self.path, body))
            if translated:
                response = {
                    "id": "chatcmpl_regression",
                    "object": "chat.completion",
                    "created": 1000,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "done",
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            else:
                response = {
                    "id": f"resp_regression_{len(captured)}",
                    "object": "response",
                    "created_at": 1000,
                    "model": "test-model",
                    "status": "completed",
                    "output": output
                    or [
                        {
                            "id": f"msg_regression_{len(captured)}",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "done",
                                    "annotations": [],
                                }
                            ],
                        }
                    ],
                }
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    backend = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    port = _free_port()
    conversations, store = _chunked_response_store_filters(
        str(tmp_path / "store.db"), port
    )
    filters = [
        {"filter": "state_owner", "mode": "trusted_owner", "header": OWNER_HEADER},
        {"filter": "ai_operation"},
        conversations,
        {"filter": "openai_responses_request"},
        store,
        {"filter": "openai_responses_rehydrate"},
    ]
    if translated:
        filters.extend(
            [
                {"filter": "responses_to_chat_completions"},
                {
                    "filter": "path_rewrite",
                    "replace": {
                        "pattern": "^/v1/responses/?$",
                        "replacement": "/v1/chat/completions",
                    },
                },
            ]
        )
    else:
        filters.append({"filter": "openai_responses_proxy"})
    filters.extend(
        [
            {
                "filter": "router",
                "routes": [{"path_prefix": "/", "cluster": "backend"}],
            },
            {
                "filter": "load_balancer",
                "clusters": [
                    {
                        "name": "backend",
                        "endpoints": [f"127.0.0.1:{backend.server_port}"],
                    }
                ],
            },
        ]
    )
    config = tmp_path / "proxy.json"
    config.write_text(
        json.dumps(
            {
                "listeners": [
                    {
                        "name": "test",
                        "address": f"127.0.0.1:{port}",
                        "filter_chains": ["test"],
                    }
                ],
                "filter_chains": [{"name": "test", "filters": filters}],
                "insecure_options": {"allow_private_endpoints": True},
            }
        )
    )
    readiness_port = _free_port()
    process = subprocess.Popen(
        [_find_binary(), "-c", str(config)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=_proxy_env(readiness_port),
    )
    try:
        _wait_for_proxy(port, process)
        _wait_for_store_ready(readiness_port, process)
        with OpenAI(
            api_key="not-needed",
            base_url=f"http://127.0.0.1:{port}/v1",
            default_headers={OWNER_HEADER: _owner_assertion("alice")},
            max_retries=0,
            timeout=10,
        ) as client:
            yield client, captured, output, translated
    finally:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        backend.shutdown()
        backend.server_close()
        thread.join(timeout=5)


def test_configuration_update_controls_subsequent_inference(conversation_backend):
    client, captured, _, translated = conversation_backend
    conversation = client.conversations.create(
        items=[
            {
                "type": "configuration_update",
                "reasoning": {"effort": "low"},
            }
        ]
    )
    client.conversations.items.create(
        conversation.id,
        items=[
            {
                "type": "configuration_update",
                "reasoning": {"effort": "high"},
            }
        ],
    )
    for explicit, expected in [(None, "high"), ("medium", "medium"), (None, "high")]:
        kwargs = {} if explicit is None else {"reasoning": {"effort": explicit}}
        response = client.responses.create(
            model="test-model",
            conversation=conversation.id,
            input="hello",
            **kwargs,
        )
        assert response.status == "completed"
        path, body = captured[-1]
        assert path == ("/v1/chat/completions" if translated else "/v1/responses")
        effort = (
            body.get("reasoning_effort")
            if translated
            else body.get("reasoning", {}).get("effort")
        )
        assert effort == expected
        assert "configuration_update" not in json.dumps(
            body.get("messages" if translated else "input")
        )
    assert len(captured) == 3


@pytest.mark.parametrize(
    "fields",
    [
        {"name": None},
        {"namespace": None},
        {"name": None, "namespace": None},
        {"call_id": None},
        {"call_id": None, "name": None, "namespace": None},
    ],
)
def test_nullable_function_output_fields(conversation_backend, fields):
    client, _, _, _ = conversation_backend
    conversation = client.conversations.create()
    created = client.conversations.items.create(
        conversation.id,
        items=[
            {
                "type": "function_call_output",
                "call_id": "call_nullable",
                "output": "done",
                **fields,
            }
        ],
    )
    item = created.data[0]
    assert item.type == "function_call_output"
    assert item.call_id == fields.get("call_id", "call_nullable")
    assert item.output == "done"
    retrieved = client.conversations.items.retrieve(
        item.id, conversation_id=conversation.id
    )
    listed = client.conversations.items.list(conversation.id).data[0]
    for value in [item, retrieved, listed]:
        wire = value.model_dump(exclude_unset=True)
        for field in fields:
            assert field not in value.model_fields_set
            assert field not in wire


def test_structured_mcp_failure_append_back(conversation_backend):
    client, _, output, translated = conversation_backend
    if translated:
        pytest.skip("Chat Completions cannot emit native MCP failure items")
    # Controlled native-provider evidence isolates append-back schema validation.
    output.append(
        {
            "id": "mcp_failure_regression",
            "type": "mcp_call",
            "name": "weather",
            "server_label": "weather",
            "arguments": "{}",
            "status": "failed",
            "error": {
                "type": "mcp_tool_execution_error",
                "content": "controlled tool failure",
            },
        }
    )
    conversation = client.conversations.create()
    response = client.responses.create(
        model="test-model", conversation=conversation.id, input="weather?"
    )
    assert response.status == "completed"
    failure = next(item for item in response.output if item.type == "mcp_call")
    assert failure.error.type == "mcp_tool_execution_error"
    page = client.conversations.items.list(conversation.id, order="asc")
    stored = next(item for item in page.data if item.type == "mcp_call")
    assert stored.id == failure.id
    assert stored.error.model_dump() == failure.error.model_dump()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
