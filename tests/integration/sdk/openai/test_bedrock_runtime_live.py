#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai==3.49.0", "pytest==8.4.2"]
# ///
"""Opt-in SDK checks against Bedrock Runtime through the loopback gateway."""

import json
import os
import sys

import pytest
from openai import APIStatusError, OpenAI


@pytest.fixture(scope="module")
def client():
    if os.getenv("PRAXIS_LIVE_BEDROCK") != "1":
        pytest.skip("Set PRAXIS_LIVE_BEDROCK=1 to opt in")
    with OpenAI(
        base_url="http://127.0.0.1:8080/v1",
        api_key="gateway-test",
        max_retries=0,
        timeout=60,
    ) as sdk:
        yield sdk


def test_stream_reports_text_finish_and_usage(client):
    with client.chat.completions.create(
        model=os.environ["BEDROCK_MODEL"],
        messages=[{"role": "user", "content": "Count from one to ten in words."}],
        max_completion_tokens=512,
        stream=True,
        stream_options={"include_usage": True},
    ) as stream:
        chunks = list(stream)
    assert any(choice.delta.content for chunk in chunks for choice in chunk.choices)
    assert [choice.finish_reason for chunk in chunks for choice in chunk.choices
            if choice.finish_reason is not None] == ["stop"]
    assert chunks[-1].usage.prompt_tokens > 0
    assert chunks[-1].usage.completion_tokens > 0


def test_tool_result_continues_conversation(client):
    model = os.environ["BEDROCK_MODEL"]
    messages = [{"role": "user", "content": "Call the status tool, then repeat its result verbatim."}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[{"type": "function", "function": {
            "name": "status", "description": "Get the current status.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        }}],
        max_completion_tokens=512,
    )
    message = response.choices[0].message
    call, = message.tool_calls
    assert call.function.name == "status"
    assert json.loads(call.function.arguments) == {}
    result = "STATUS-2907"
    messages.extend([
        message.model_dump(exclude_none=True, include={"role", "content", "tool_calls"}),
        {"role": "tool", "tool_call_id": call.id, "content": result},
    ])
    final = client.chat.completions.create(
        model=model, messages=messages, max_completion_tokens=512,
    )
    assert result in (final.choices[0].message.content or "")


def test_invalid_request_returns_provider_error(client):
    with pytest.raises(APIStatusError) as error:
        client.chat.completions.create(
            model=os.environ["BEDROCK_MODEL"],
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_completion_tokens=-1,
        )
    assert error.value.status_code in {400, 422}
    assert isinstance(error.value.body, dict)
    body = error.value.body.get("error", error.value.body)
    assert isinstance(body.get("message"), str) and body["message"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
