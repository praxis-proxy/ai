# /// script
# requires-python = ">=3.11"
# dependencies = ["anthropic==1.17.0", "openai==3.49.0", "pytest==8.4.2"]
# ///
"""Live native and translated Anthropic provider checks through Praxis."""

import json
import os
import secrets
import sys

import pytest
from anthropic import Anthropic, NotFoundError
from openai import BadRequestError, OpenAI

MODEL = "claude-haiku-4-5-20251001"
PROMPT = "Call status for service gateway, then repeat the tool result verbatim."
SCHEMA = {
    "type": "object",
    "properties": {"service": {"type": "string", "enum": ["gateway"]}},
    "required": ["service"],
    "additionalProperties": False,
}


def test_messages():
    if os.getenv("PRAXIS_LIVE_ANTHROPIC") != "messages":
        pytest.skip("Enable the native Messages route")
    tool = {"name": "status", "description": "Get service status.", "input_schema": SCHEMA}
    messages = [{"role": "user", "content": PROMPT}]
    with Anthropic(base_url="http://127.0.0.1:8080", api_key="local-test",
                   max_retries=0, timeout=45) as client:
        first = client.messages.create(
            model=MODEL, max_tokens=64, messages=messages, tools=[tool],
            tool_choice={"type": "tool", "name": "status"},
        )
        call, = [block for block in first.content if block.type == "tool_use"]
        assert call.name == "status" and call.input == {"service": "gateway"}
        assert first.usage.output_tokens > 0
        result = secrets.token_hex(4)
        messages.extend([
            {"role": "assistant", "content": [block.model_dump(exclude_none=True) for block in first.content]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call.id, "content": result}]},
        ])
        with client.messages.stream(model=MODEL, max_tokens=64, messages=messages, tools=[tool]) as stream:
            assert result in "".join(stream.text_stream)
            final = stream.get_final_message()
        assert final.stop_reason == "end_turn"
        assert final.usage.output_tokens > 0


def test_messages_unknown_model_preserves_error():
    if os.getenv("PRAXIS_LIVE_ANTHROPIC") != "messages":
        pytest.skip("Enable the native Messages route")
    with Anthropic(base_url="http://127.0.0.1:8080", api_key="local-test",
                   max_retries=0, timeout=45) as client:
        with pytest.raises(NotFoundError) as error:
            client.messages.create(
                model="praxis-missing-anthropic-model", max_tokens=1,
                messages=[{"role": "user", "content": "Hello."}],
            )
    assert error.value.status_code == 404
    assert error.value.type == "not_found_error"


def test_chat_translation():
    if os.getenv("PRAXIS_LIVE_ANTHROPIC") != "chat":
        pytest.skip("Enable the translated Chat Completions route")
    tool = {"type": "function", "function": {
        "name": "status", "description": "Get service status.", "parameters": SCHEMA,
    }}
    messages = [{"role": "user", "content": PROMPT}]
    with OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-test",
                max_retries=0, timeout=45) as client:
        first = client.chat.completions.create(
            model=MODEL, max_completion_tokens=64, messages=messages, tools=[tool],
            tool_choice={"type": "function", "function": {"name": "status"}},
        )
        call, = first.choices[0].message.tool_calls
        assert call.function.name == "status"
        assert json.loads(call.function.arguments) == {"service": "gateway"}
        assert first.usage.completion_tokens > 0
        result = secrets.token_hex(4)
        messages.extend([
            first.choices[0].message.model_dump(exclude_none=True, include={"role", "content", "tool_calls"}),
            {"role": "tool", "tool_call_id": call.id, "content": result},
        ])
        with client.chat.completions.create(
            model=MODEL, max_completion_tokens=64, messages=messages, tools=[tool],
            stream=True, stream_options={"include_usage": True},
        ) as stream:
            chunks = list(stream)
        assert result in "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices)
        assert [choice.finish_reason for chunk in chunks for choice in chunk.choices
                if choice.finish_reason is not None] == ["stop"]
        assert all(chunk.usage is None for chunk in chunks[:-1])
        assert chunks[-1].choices == []
        assert chunks[-1].usage.prompt_tokens > 0
        assert chunks[-1].usage.completion_tokens > 0


def test_chat_translation_streamed_tool_call():
    if os.getenv("PRAXIS_LIVE_ANTHROPIC") != "chat":
        pytest.skip("Enable the translated Chat Completions route")
    tool = {"type": "function", "function": {
        "name": "status", "description": "Get service status.", "parameters": SCHEMA,
    }}
    messages = [{"role": "user", "content": PROMPT}]
    with OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-test",
                max_retries=0, timeout=45) as client:
        with client.chat.completions.create(
            model=MODEL, max_completion_tokens=64, messages=messages, tools=[tool],
            tool_choice={"type": "function", "function": {"name": "status"}},
            stream=True, stream_options={"include_usage": True},
        ) as stream:
            chunks = list(stream)
        choices = [choice for chunk in chunks for choice in chunk.choices]
        deltas = [call for choice in choices for call in (choice.delta.tool_calls or [])]
        assert deltas and {delta.index for delta in deltas} == {0}
        call_id, = {delta.id for delta in deltas if delta.id}
        name = "".join(delta.function.name or "" for delta in deltas if delta.function)
        arguments = "".join(delta.function.arguments or "" for delta in deltas if delta.function)
        assert name == "status"
        assert json.loads(arguments) == {"service": "gateway"}
        assert [choice.finish_reason for choice in choices if choice.finish_reason] == ["tool_calls"]
        assert all(chunk.usage is None for chunk in chunks[:-1])
        assert chunks[-1].choices == []
        assert chunks[-1].usage.prompt_tokens > 0
        assert chunks[-1].usage.completion_tokens > 0

        result = secrets.token_hex(4)
        messages.extend([
            {"role": "assistant", "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": name, "arguments": arguments},
            }]},
            {"role": "tool", "tool_call_id": call_id, "content": result},
        ])
        final = client.chat.completions.create(
            model=MODEL, max_completion_tokens=64, messages=messages, tools=[tool],
        )
        choice, = final.choices
        assert choice.finish_reason == "stop"
        assert result in (choice.message.content or "")
        assert final.usage.completion_tokens > 0


def test_chat_translation_maps_provider_validation_error():
    if os.getenv("PRAXIS_LIVE_ANTHROPIC") != "chat":
        pytest.skip("Enable the translated Chat Completions route")
    with OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-test",
                max_retries=0, timeout=45) as client:
        # Valid for Chat Completions, outside the upstream Anthropic range.
        with pytest.raises(BadRequestError) as error:
            client.chat.completions.create(
                model=MODEL, max_completion_tokens=1, temperature=1.5,
                messages=[{"role": "user", "content": "Hello."}],
            )
    assert error.value.status_code == 400
    assert isinstance(error.value.body, dict)
    assert "temperature" in error.value.body["message"].lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
