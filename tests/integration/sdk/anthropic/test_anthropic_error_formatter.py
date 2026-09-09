#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "anthropic>=0.40",
#     "pytest>=8.0",
# ]
# ///
"""
Anthropic SDK compatibility tests for fatal proxy error response formatting.

Sends requests via the official Anthropic Python SDK to verify that fatal proxy
errors (connection refusal and gateway timeout) are returned in the
native Anthropic {"type": "error", "error": {...}} format and correctly
parsed by the SDK into standard APIStatusError exceptions.
"""

import sys

import pytest
from anthropic import APIStatusError, Anthropic


@pytest.fixture(scope="module")
def anthropic_client(error_proxy_ports):
    return Anthropic(
        api_key="not-needed",
        base_url=f"http://127.0.0.1:{error_proxy_ports['refused_port']}",
        max_retries=0,
        timeout=10.0,
    )


@pytest.fixture(scope="module")
def timeout_anthropic_client(error_proxy_ports):
    return Anthropic(
        api_key="not-needed",
        base_url=f"http://127.0.0.1:{error_proxy_ports['timeout_port']}",
        max_retries=0,
        timeout=10.0,
    )


class TestAnthropicErrorFormatter:
    def test_messages_connection_refused(self, anthropic_client):
        with pytest.raises(APIStatusError) as exc_info:
            anthropic_client.messages.create(
                model="claude-opus-4-8",
                max_tokens=100,
                messages=[{"role": "user", "content": "hello"}],
            )
        err = exc_info.value
        assert err.status_code == 502
        assert err.body.get("type") == "error"
        error_obj = err.body.get("error", {})
        assert error_obj.get("type") == "api_error"
        assert "Upstream connection refused" in error_obj.get("message", "")
        assert err.body.get("request_id") is not None

    def test_messages_connection_refused_with_request_id(self, anthropic_client):
        custom_id = "req_sdk_custom_9999"
        with pytest.raises(APIStatusError) as exc_info:
            anthropic_client.messages.create(
                model="claude-opus-4-8",
                max_tokens=100,
                messages=[{"role": "user", "content": "hello"}],
                extra_headers={"x-request-id": custom_id},
            )
        err = exc_info.value
        assert err.status_code == 502
        assert err.body.get("type") == "error"
        error_obj = err.body.get("error", {})
        assert error_obj.get("type") == "api_error"
        assert err.body.get("request_id") == custom_id

    def test_messages_gateway_timeout(self, timeout_anthropic_client):
        with pytest.raises(APIStatusError) as exc_info:
            timeout_anthropic_client.messages.create(
                model="claude-opus-4-8",
                max_tokens=100,
                messages=[{"role": "user", "content": "hello"}],
            )
        err = exc_info.value
        assert err.status_code == 504
        assert err.body.get("type") == "error"
        error_obj = err.body.get("error", {})
        assert error_obj.get("type") == "timeout_error"
        assert "Upstream read timed out" in error_obj.get("message", "")
        assert err.body.get("request_id") is not None


if __name__ == "__main__":
    pytest.main([__file__, *sys.argv[1:]])
