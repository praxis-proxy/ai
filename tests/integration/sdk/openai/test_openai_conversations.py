#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.27",
#     "openai>=2.0",
#     "pytest>=8.0",
# ]
# ///
"""
OpenAI SDK compatibility tests for the openai_conversations filter.

Starts a Praxis proxy with an in-memory SQLite conversations store,
then exercises the Conversations API using the official OpenAI Python
SDK to verify wire-format compatibility.

Usage:
    cargo build -p praxis-ai-proxy
    uv run tests/integration/sdk/openai/test_openai_conversations.py -v
"""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time

import httpx
import pytest
from openai import BadRequestError, NotFoundError, OpenAI

# When set to a postgres:// URL (the vllm-responses-postgres CI job), the
# conversations store runs against PostgreSQL instead of the default in-memory
# SQLite, so this suite exercises the same store backend as the responses tests.
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _find_binary() -> str:
    configured = os.environ.get("PRAXIS_AI_BIN")
    if configured:
        if os.path.isfile(configured):
            return configured
        raise FileNotFoundError(f"PRAXIS_AI_BIN={configured!r} not found")
    for candidate in ["target/debug/praxis-ai", "target/release/praxis-ai"]:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "praxis-ai binary not found — run `cargo build -p praxis-ai-proxy` first"
    )


def _conversations_filter() -> dict:
    """Build the openai_conversations filter config for the configured store.

    Defaults to in-memory SQLite; switches to PostgreSQL when DATABASE_URL is a
    postgres:// URL, matching the responses tests' backend selection so both
    suites cover the same store backend in CI.
    """
    cfg = {
        "filter": "openai_conversations",
        "conversations_table": "conversations",
        "items_table": "conversation_items",
    }
    if DATABASE_URL.startswith("postgres"):
        cfg.update(
            {
                "backend": "postgres",
                "database_url": DATABASE_URL,
                # Local CI postgres service is loopback + non-TLS.
                "allow_private_database_url": True,
                "ssl_mode": "disable",
            }
        )
    else:
        cfg.update(
            {
                "backend": "sqlite",
                "database_url": "sqlite::memory:",
            }
        )
    return cfg


def _write_config(port: int) -> str:
    config = {
        "listeners": [
            {
                "name": "test",
                "address": f"127.0.0.1:{port}",
                "filter_chains": ["conversations-pipeline"],
            }
        ],
        "filter_chains": [
            {
                "name": "conversations-pipeline",
                "filters": [_conversations_filter()],
            }
        ],
    }
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        json.dump(config, f)
    return path


def _wait_for_proxy(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"proxy did not start within {timeout}s")


@pytest.fixture(scope="session")
def praxis_proxy():
    """Start a Praxis proxy for the test session and tear it down after."""
    port = _free_port()
    config_path = _write_config(port)
    binary = _find_binary()

    proc = subprocess.Popen(
        [binary, "-c", config_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_proxy(port)
        yield port
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        os.unlink(config_path)


@pytest.fixture(scope="session")
def openai_client(praxis_proxy):
    """Return an OpenAI client pointed at the local Praxis proxy."""
    return OpenAI(
        api_key="not-needed",
        base_url=f"http://127.0.0.1:{praxis_proxy}/v1",
        max_retries=0,
        timeout=10.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOpenAIConversations:
    """Wire-format compatibility tests for conversation CRUD."""

    def test_conversation_create(self, openai_client):
        conversation = openai_client.conversations.create(
            metadata={"topic": "demo"},
        )

        assert conversation.id.startswith("conv_")
        assert conversation.object == "conversation"
        assert conversation.metadata["topic"] == "demo"
        assert isinstance(conversation.created_at, int)
        assert conversation.created_at > 0

    def test_conversation_create_no_metadata(self, openai_client):
        conversation = openai_client.conversations.create()

        assert conversation.object == "conversation"
        assert conversation.metadata == {}

        openai_client.conversations.delete(conversation.id)

    def test_conversation_retrieve(self, openai_client):
        conversation = openai_client.conversations.create(
            metadata={"topic": "demo"},
        )

        retrieved = openai_client.conversations.retrieve(conversation.id)

        assert retrieved.id == conversation.id
        assert retrieved.object == "conversation"
        assert retrieved.metadata["topic"] == "demo"
        assert retrieved.created_at == conversation.created_at

    def test_conversation_retrieve_nonexistent(self, openai_client):
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.conversations.retrieve("conv_nonexistent")
        assert exc_info.value.status_code == 404

    def test_conversation_update(self, openai_client):
        conversation = openai_client.conversations.create(
            metadata={"topic": "demo"},
        )

        updated = openai_client.conversations.update(
            conversation.id,
            metadata={"topic": "project-x"},
        )

        assert updated.id == conversation.id
        assert updated.metadata["topic"] == "project-x"
        assert updated.created_at == conversation.created_at

    def test_conversation_update_nonexistent(self, openai_client):
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.conversations.update(
                "conv_nonexistent",
                metadata={"topic": "nope"},
            )
        assert exc_info.value.status_code == 404

    def test_conversation_delete(self, openai_client):
        conversation = openai_client.conversations.create(
            metadata={"topic": "demo"},
        )

        deleted = openai_client.conversations.delete(conversation.id)

        assert deleted.id == conversation.id
        assert deleted.object == "conversation.deleted"
        assert deleted.deleted is True

    def test_deleted_conversation_not_retrievable(self, openai_client):
        conversation = openai_client.conversations.create()
        openai_client.conversations.delete(conversation.id)

        with pytest.raises(NotFoundError) as exc_info:
            openai_client.conversations.retrieve(conversation.id)
        assert exc_info.value.status_code == 404

    def test_conversation_delete_preserves_items(self, openai_client):
        conversation = openai_client.conversations.create(
            items=[
                {
                    "id": "item_keep",
                    "type": "message",
                    "role": "user",
                    "content": "keep me",
                },
            ],
        )
        openai_client.conversations.delete(conversation.id)

        item = openai_client.conversations.items.retrieve(
            "item_keep",
            conversation_id=conversation.id,
        )

        assert item.id == "item_keep"
        assert item.type == "message"
        assert item.content[0].text == "keep me"

    def test_empty_item_list_is_sdk_compatible(self, openai_client):
        conversation = openai_client.conversations.create()

        page = openai_client.conversations.items.list(conversation.id)

        assert page.object == "list"
        assert page.data == []
        assert page.first_id == ""
        assert page.last_id == ""
        assert page.has_more is False

    def test_conversation_delete_nonexistent(self, openai_client):
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.conversations.delete("conv_nonexistent")
        assert exc_info.value.status_code == 404

    def test_initial_items_are_sdk_compatible(self, openai_client):
        conversation = openai_client.conversations.create(
            metadata={"topic": "items"},
            items=[
                {"type": "message", "role": "user", "content": "hello"},
            ],
        )

        page = openai_client.conversations.items.list(
            conversation.id,
            order="asc",
        )

        item = page.data[0]
        assert item.id.startswith("item_")
        assert item.type == "message"
        assert item.role == "user"
        assert item.status == "completed"
        assert item.content[0].type == "input_text"
        assert item.content[0].text == "hello"

    def test_item_create_returns_all_items(self, openai_client):
        conversation = openai_client.conversations.create()

        created = openai_client.conversations.items.create(
            conversation.id,
            items=[
                {
                    "id": "item_batch_user",
                    "type": "message",
                    "role": "user",
                    "content": "question",
                },
                {
                    "id": "item_batch_assistant",
                    "type": "message",
                    "role": "assistant",
                    "content": "answer",
                },
            ],
        )

        assert created.object == "list"
        assert [item.id for item in created.data] == [
            "item_batch_user",
            "item_batch_assistant",
        ]
        assert created.first_id == "item_batch_user"
        assert created.last_id == "item_batch_assistant"
        assert created.has_more is False

    def test_item_list_cursor_pagination_and_order(self, openai_client):
        conversation = openai_client.conversations.create(
            items=[
                {
                    "id": f"item_page_{index}",
                    "type": "message",
                    "role": "user",
                    "content": f"message {index}",
                }
                for index in range(3)
            ],
        )

        first = openai_client.conversations.items.list(
            conversation.id,
            limit=2,
            order="asc",
        )
        assert [item.id for item in first.data] == [
            "item_page_0",
            "item_page_1",
        ]
        assert first.first_id == "item_page_0"
        assert first.last_id == "item_page_1"
        assert first.has_more is True

        second = openai_client.conversations.items.list(
            conversation.id,
            after=first.last_id,
            limit=2,
            order="asc",
        )
        assert [item.id for item in second.data] == ["item_page_2"]
        assert second.has_more is False

        descending = openai_client.conversations.items.list(
            conversation.id,
            order="desc",
        )
        assert [item.id for item in descending.data] == [
            "item_page_2",
            "item_page_1",
            "item_page_0",
        ]

    def test_item_crud_is_sdk_compatible(self, openai_client):
        conversation = openai_client.conversations.create()

        created = openai_client.conversations.items.create(
            conversation.id,
            items=[
                {"type": "message", "role": "assistant", "content": "hi"},
            ],
        )
        item = created.data[0]
        assert item.id.startswith("item_")
        assert item.type == "message"
        assert item.role == "assistant"
        assert item.status == "completed"
        assert item.content[0].type == "output_text"
        assert item.content[0].text == "hi"
        assert item.content[0].annotations == []

        retrieved = openai_client.conversations.items.retrieve(
            item.id,
            conversation_id=conversation.id,
        )
        assert retrieved.id == item.id
        assert retrieved.status == "completed"
        assert retrieved.content[0].text == "hi"

        deleted = openai_client.conversations.items.delete(
            item.id,
            conversation_id=conversation.id,
        )
        assert deleted.id == conversation.id
        with pytest.raises(NotFoundError):
            openai_client.conversations.items.retrieve(
                item.id,
                conversation_id=conversation.id,
            )

    def test_encoded_item_id_is_sdk_compatible(self, openai_client):
        conversation = openai_client.conversations.create()

        openai_client.conversations.items.create(
            conversation.id,
            items=[
                {
                    "id": "item with space",
                    "type": "message",
                    "role": "user",
                    "content": "encoded",
                },
            ],
        )

        retrieved = openai_client.conversations.items.retrieve(
            "item with space",
            conversation_id=conversation.id,
        )
        assert retrieved.id == "item with space"
        assert retrieved.content[0].text == "encoded"

        openai_client.conversations.items.create(
            conversation.id,
            items=[
                {
                    "id": "item_after_space",
                    "type": "message",
                    "role": "assistant",
                    "content": "after",
                },
            ],
        )

        page = openai_client.conversations.items.list(
            conversation.id,
            after="item with space",
            order="asc",
        )
        assert [item.id for item in page.data] == ["item_after_space"]

        openai_client.conversations.items.delete(
            "item with space",
            conversation_id=conversation.id,
        )
        with pytest.raises(NotFoundError):
            openai_client.conversations.items.retrieve(
                "item with space",
                conversation_id=conversation.id,
            )

    def test_item_operations_for_missing_resources(self, openai_client):
        missing_conversation = "conv_missing_sdk_integration"
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.conversations.items.list(missing_conversation)
        assert exc_info.value.status_code == 404

        with pytest.raises(NotFoundError) as exc_info:
            openai_client.conversations.items.create(
                missing_conversation,
                items=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": "unreachable",
                    }
                ],
            )
        assert exc_info.value.status_code == 404

        conversation = openai_client.conversations.create()
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.conversations.items.retrieve(
                "item_missing_sdk_integration",
                conversation_id=conversation.id,
            )
        assert exc_info.value.status_code == 404

        with pytest.raises(NotFoundError) as exc_info:
            openai_client.conversations.items.delete(
                "item_missing_sdk_integration",
                conversation_id=conversation.id,
            )
        assert exc_info.value.status_code == 404

    def test_duplicate_item_id_is_rejected(self, openai_client):
        conversation = openai_client.conversations.create(
            items=[
                {
                    "id": "item_duplicate",
                    "type": "message",
                    "role": "user",
                    "content": "first",
                }
            ],
        )

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.conversations.items.create(
                conversation.id,
                items=[
                    {
                        "id": "item_duplicate",
                        "type": "message",
                        "role": "user",
                        "content": "second",
                    }
                ],
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.parametrize(
        "query",
        ["limit=101", "limit=-1", "order=sideways", "after="],
    )
    def test_invalid_item_list_query_is_rejected(
        self,
        openai_client,
        query,
    ):
        conversation = openai_client.conversations.create()
        response = httpx.get(
            f"{str(openai_client.base_url).rstrip('/')}"
            f"/conversations/{conversation.id}/items?{query}",
            headers={"Authorization": "Bearer not-needed"},
            timeout=10,
        )
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["message"]

    def test_conversation_invalid_metadata_type(self, openai_client):
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.conversations.create(metadata="not-an-object")
        assert exc_info.value.status_code == 400

    def test_conversation_metadata_too_many_keys(self, openai_client):
        metadata = {f"key{i}": f"val{i}" for i in range(17)}
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.conversations.create(metadata=metadata)
        assert exc_info.value.status_code == 400

    def test_full_workflow(self, openai_client):
        conversation = openai_client.conversations.create(
            metadata={"topic": "workflow-test"},
        )
        assert conversation.id.startswith("conv_")

        updated = openai_client.conversations.update(
            conversation.id,
            metadata={"topic": "workflow-complete"},
        )
        assert updated.metadata["topic"] == "workflow-complete"
        assert updated.created_at == conversation.created_at

        deleted = openai_client.conversations.delete(conversation.id)
        assert deleted.deleted is True

        with pytest.raises(NotFoundError):
            openai_client.conversations.retrieve(conversation.id)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
