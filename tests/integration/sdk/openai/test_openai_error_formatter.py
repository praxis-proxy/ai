#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "openai>=1.0",
#     "pytest>=8.0",
# ]
# ///
"""
OpenAI SDK compatibility tests for fatal proxy error response formatting.

Sends requests via the official OpenAI Python SDK to verify that fatal proxy
errors (connection refusal and gateway timeout) are returned in the
native OpenAI {"error": {...}} format and correctly parsed by the SDK into
standard APIStatusError exceptions.
"""

import sys

import pytest
from openai import APIStatusError, OpenAI


@pytest.fixture(scope="module")
def openai_client(error_proxy_ports):
    return OpenAI(
        api_key="not-needed",
        base_url=f"http://127.0.0.1:{error_proxy_ports['refused_port']}/v1",
        max_retries=0,
        timeout=10.0,
    )


@pytest.fixture(scope="module")
def timeout_openai_client(error_proxy_ports):
    return OpenAI(
        api_key="not-needed",
        base_url=f"http://127.0.0.1:{error_proxy_ports['timeout_port']}/v1",
        max_retries=0,
        timeout=10.0,
    )


class TestOpenAIErrorFormatter:
    def test_chat_completions_connection_refused(self, openai_client):
        with pytest.raises(APIStatusError) as exc_info:
            openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "hello"}],
            )
        err = exc_info.value
        assert err.status_code == 502
        assert err.code == "upstream_connect_refused"
        assert err.type == "server_error"
        assert err.body.get("param") is None
        assert "Upstream connection refused" in err.body.get("message", "")

    def test_responses_connection_refused(self, openai_client):
        with pytest.raises(APIStatusError) as exc_info:
            openai_client.responses.create(
                model="gpt-4.1",
                input="hello",
            )
        err = exc_info.value
        assert err.status_code == 502
        assert err.code == "upstream_connect_refused"
        assert err.type == "server_error"
        assert err.body.get("param") is None
        assert "Upstream connection refused" in err.body.get("message", "")

    def test_chat_completions_gateway_timeout(self, timeout_openai_client):
        with pytest.raises(APIStatusError) as exc_info:
            timeout_openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "hello"}],
            )
        err = exc_info.value
        assert err.status_code == 504
        assert err.code == "upstream_read_timeout"
        assert err.type == "server_error"
        assert err.body.get("param") is None
        assert "Upstream read timed out" in err.body.get("message", "")

    def test_responses_gateway_timeout(self, timeout_openai_client):
        with pytest.raises(APIStatusError) as exc_info:
            timeout_openai_client.responses.create(
                model="gpt-4.1",
                input="hello",
            )
        err = exc_info.value
        assert err.status_code == 504
        assert err.code == "upstream_read_timeout"
        assert err.type == "server_error"
        assert err.body.get("param") is None
        assert "Upstream read timed out" in err.body.get("message", "")


if __name__ == "__main__":
    pytest.main([__file__, *sys.argv[1:]])
