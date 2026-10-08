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
def classifier_only_client(error_proxy_ports):
    """Client for a chain that runs only the head-based operation classifier.

    No body-format filter is present, so OpenAI error shaping can only come
    from the classifier.
    """
    return OpenAI(
        api_key="not-needed",
        base_url=f"http://127.0.0.1:{error_proxy_ports['classifier_only_port']}/v1",
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
        assert err.status_code == 502, "upstream connect refusal must surface as 502"
        assert err.code == "upstream_connect_refused", "error code must identify the connect refusal"
        assert err.type == "server_error", "error type must be server_error"
        assert err.body.get("param") is None, "connect refusal error must not carry a param"
        assert "Upstream connection refused" in err.body.get("message", ""), "error message must describe the connection refusal"

    def test_responses_connection_refused(self, openai_client):
        with pytest.raises(APIStatusError) as exc_info:
            openai_client.responses.create(
                model="gpt-4.1",
                input="hello",
            )
        err = exc_info.value
        assert err.status_code == 502, "upstream connect refusal must surface as 502"
        assert err.code == "upstream_connect_refused", "error code must identify the connect refusal"
        assert err.type == "server_error", "error type must be server_error"
        assert err.body.get("param") is None, "connect refusal error must not carry a param"
        assert "Upstream connection refused" in err.body.get("message", ""), "error message must describe the connection refusal"

    def test_chat_completions_error_shape_without_a_responses_filter(self, classifier_only_client):
        """A Chat Completions client keeps the OpenAI error schema on a chain
        with no body-format filter, so the SDK still raises APIStatusError
        rather than failing to parse RFC 9457 problem details."""
        with pytest.raises(APIStatusError) as exc_info:
            classifier_only_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "hello"}],
            )
        err = exc_info.value
        assert err.status_code == 502, "upstream connect refusal must surface as 502"
        assert err.code == "upstream_connect_refused", "error code must identify the connect refusal"
        assert err.type == "server_error", "error type must be server_error"
        assert "Upstream connection refused" in err.body.get("message", ""), "error message must describe the connection refusal"

    def test_responses_error_shape_without_a_responses_filter(self, classifier_only_client):
        """The same chain shapes Responses errors from head classification too."""
        with pytest.raises(APIStatusError) as exc_info:
            classifier_only_client.responses.create(model="gpt-4.1", input="hello")
        err = exc_info.value
        assert err.status_code == 502, "upstream connect refusal must surface as 502"
        assert err.code == "upstream_connect_refused", "error code must identify the connect refusal"
        assert err.type == "server_error", "error type must be server_error"

    def test_chat_completions_gateway_timeout(self, timeout_openai_client):
        with pytest.raises(APIStatusError) as exc_info:
            timeout_openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "hello"}],
            )
        err = exc_info.value
        assert err.status_code == 504, "upstream read timeout must surface as 504"
        assert err.code == "upstream_read_timeout", "error code must identify the read timeout"
        assert err.type == "server_error", "error type must be server_error"
        assert err.body.get("param") is None, "read timeout error must not carry a param"
        assert "Upstream read timed out" in err.body.get("message", ""), "error message must describe the read timeout"

    def test_responses_gateway_timeout(self, timeout_openai_client):
        with pytest.raises(APIStatusError) as exc_info:
            timeout_openai_client.responses.create(
                model="gpt-4.1",
                input="hello",
            )
        err = exc_info.value
        assert err.status_code == 504, "upstream read timeout must surface as 504"
        assert err.code == "upstream_read_timeout", "error code must identify the read timeout"
        assert err.type == "server_error", "error type must be server_error"
        assert err.body.get("param") is None, "read timeout error must not carry a param"
        assert "Upstream read timed out" in err.body.get("message", ""), "error message must describe the read timeout"


if __name__ == "__main__":
    pytest.main([__file__, *sys.argv[1:]])
