# /// script
# dependencies = ["anthropic==1.17.0", "pytest==8.4.2"]
# ///
"""Opt-in Anthropic SDK checks against official OpenAI through Praxis."""

import os
import secrets
from urllib.parse import urlsplit

import pytest
from anthropic import Anthropic, NotFoundError

MODEL = "gpt-4.1-mini-2025-04-14"


@pytest.fixture(scope="module")
def client():
    base_url = os.getenv("PRAXIS_LIVE_MESSAGES_BASE_URL")
    if not base_url:
        pytest.skip("Set PRAXIS_LIVE_MESSAGES_BASE_URL to opt in to live tests")
    url = urlsplit(base_url)
    if (
        url.scheme != "http"
        or url.hostname not in {"127.0.0.1", "::1"}
        or url.path not in {"", "/"}
        or url.username is not None
        or url.query
        or url.fragment
    ):
        pytest.fail("Expected a loopback Praxis gateway URL")
    with Anthropic(base_url=base_url, api_key="local-test-only", max_retries=0, timeout=30) as sdk:
        yield sdk


def test_message_translates_text_and_usage(client):
    message = client.messages.create(
        model=MODEL, messages=[{"role": "user", "content": "Reply with OK."}],
        max_tokens=16,
    )
    assert message.model == MODEL
    assert message.content[0].type == "text"
    assert message.content[0].text
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


def test_stream_translates_deltas_and_terminates(client):
    with client.messages.stream(
        model=MODEL, messages=[{"role": "user", "content": "Reply with OK."}],
        max_tokens=16,
    ) as stream:
        events = list(stream)
        message = stream.get_final_message()
    assert events[0].type == "message_start"
    assert any(event.type == "content_block_delta" for event in events)
    assert events[-1].type == "message_stop"
    assert message.stop_reason == "end_turn"
    assert message.usage.output_tokens > 0


def test_tool_result_translates_on_followup(client):
    messages = [{"role": "user", "content": "Call the status tool, then copy its result verbatim."}]
    result = client.messages.create(
        model=MODEL, max_tokens=64, messages=messages,
        tools=[{"name": "status", "input_schema": {"type": "object", "properties": {}}}],
        tool_choice={"type": "tool", "name": "status"},
    )
    tool, = [block for block in result.content if block.type == "tool_use"]
    assert tool.name == "status"
    assert tool.input == {}
    value = secrets.token_hex(4)
    messages.extend([
        {"role": "assistant", "content": [tool.model_dump()]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tool.id, "content": value},
        ]},
    ])
    followup = client.messages.create(model=MODEL, max_tokens=32, messages=messages)
    assert any(block.type == "text" and value in block.text for block in followup.content)


def test_unknown_model_translates_upstream_error(client):
    with pytest.raises(NotFoundError) as exc:
        client.messages.create(
            model="praxis-qualification-nonexistent-model",
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_tokens=16,
        )
    # The translator preserves error types shared by both APIs.
    assert exc.value.status_code == 404
    assert exc.value.body["error"]["type"] == "invalid_request_error"
