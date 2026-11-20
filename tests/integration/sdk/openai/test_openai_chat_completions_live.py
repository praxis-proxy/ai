#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai==3.49.0", "pytest==8.4.2"]
# ///
"""Opt-in live Chat Completions checks through a local Praxis gateway.

Start Praxis with tests/integration/fixtures/openai/chat-completions-live.yaml.
Set PRAXIS_LIVE_OPENAI_BASE_URL to its loopback /v1 URL and OPENAI_API_KEY
to an upstream key the gateway forwards. Run this file with uv run --script.
"""

import json
import os
import secrets
from urllib.parse import urlsplit

import pytest
from openai import NotFoundError, OpenAI

MODEL = "gpt-4.1-mini-2025-04-14"
EMBEDDING_MODEL = "text-embedding-3-small"


@pytest.fixture(scope="module")
def client():
    base_url = os.getenv("PRAXIS_LIVE_OPENAI_BASE_URL")
    if not base_url:
        pytest.skip("Set PRAXIS_LIVE_OPENAI_BASE_URL to opt in to live tests")
    url = urlsplit(base_url)
    if (
        url.scheme != "http"
        or url.hostname not in {"127.0.0.1", "::1"}
        or url.path.rstrip("/") != "/v1"
        or url.username is not None
        or url.query
        or url.fragment
    ):
        pytest.fail("Expected a loopback Praxis gateway URL ending in /v1")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.fail("OPENAI_API_KEY is required for live tests")
    with OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=30) as sdk:
        yield sdk


def test_completion_reports_model_and_usage(client):
    response = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "Reply with OK."}],
        max_completion_tokens=16,
    )
    assert response.model == MODEL
    assert response.choices[0].message.content
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0


def test_stream_delivers_deltas_terminal_reason_and_usage(client):
    with client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Reply with OK."}],
        max_completion_tokens=64,
        stream=True,
        stream_options={"include_usage": True},
    ) as stream:
        chunks = list(stream)
    assert any(choice.delta.content for chunk in chunks for choice in chunk.choices)
    assert [choice.finish_reason for chunk in chunks for choice in chunk.choices
            if choice.finish_reason is not None] == ["stop"]
    assert chunks[-1].usage.completion_tokens > 0


def test_function_call_result_can_continue_conversation(client):
    messages = [{"role": "user", "content": "Call the status tool, then copy its result verbatim."}]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=[{"type": "function", "function": {"name": "status", "parameters": {
            "type": "object", "properties": {}, "additionalProperties": False,
        }}}],
        tool_choice={"type": "function", "function": {"name": "status"}},
    )
    choice = response.choices[0]
    # OpenAI may report stop even when a forced function call is present.
    call, = choice.message.tool_calls
    assert call.function.name == "status"
    assert json.loads(call.function.arguments) == {}
    tool_result = secrets.token_hex(4)
    messages.extend([
        choice.message.model_dump(exclude_none=True, include={"role", "content", "tool_calls"}),
        {"role": "tool", "tool_call_id": call.id, "content": tool_result},
    ])
    followup = client.chat.completions.create(
        model=MODEL, messages=messages, max_completion_tokens=32,
    )
    assert tool_result in followup.choices[0].message.content


def test_unknown_model_preserves_openai_error(client):
    with pytest.raises(NotFoundError) as exc:
        client.chat.completions.create(
            model="praxis-qualification-nonexistent-model",
            messages=[{"role": "user", "content": "Reply with OK."}],
        )
    assert exc.value.code == "model_not_found"


def test_embeddings_batch_preserves_order_dimensions_and_usage(client):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL, input=["A blue sky.", "A deep ocean."],
        dimensions=64, encoding_format="float",
    )
    assert response.model == EMBEDDING_MODEL
    assert [item.index for item in response.data] == [0, 1]
    assert all(len(item.embedding) == 64 for item in response.data)
    assert response.data[0].embedding != response.data[1].embedding
    assert response.usage.prompt_tokens > 0
    assert response.usage.total_tokens == response.usage.prompt_tokens


def test_responses_passthrough_reports_text_and_usage(client):
    response = client.responses.create(
        model=MODEL, input="Reply with OK.", max_output_tokens=32, store=False,
    )
    assert response.status == "completed"
    assert response.model == MODEL
    assert response.output_text
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0


def test_responses_stream_reports_text_and_completion(client):
    with client.responses.create(
        model=MODEL, input="Reply with OK.", max_output_tokens=32,
        store=False, stream=True,
    ) as stream:
        event_types = [event.type for event in stream]
    assert "response.output_text.delta" in event_types
    assert event_types[-1] == "response.completed"


if __name__ == "__main__":
    import sys

    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
