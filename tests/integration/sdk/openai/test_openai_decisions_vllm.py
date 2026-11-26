#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai==3.26.0", "httpx>=0.28,<1", "pytest>=8.0"]
# ///
"""Exercise both decision APIs through Praxis against live vLLM CPU."""

import os
import sys

import httpx
import pytest
from openai import OpenAI


@pytest.fixture
def endpoint():
    return os.environ["PRAXIS_DECISIONS_URL"].rstrip("/")


@pytest.fixture
def model():
    return os.environ["VLLM_DECISIONS_MODEL"]


def test_decisions_api(endpoint, model):
    with OpenAI(base_url=f"{endpoint}/v1", api_key="local", timeout=60, max_retries=0) as client:
        decision = client.decisions.create(
            model=model,
            input="My card was charged twice for the same order.",
            questions=[
                {
                    "type": "predicate",
                    "name": "duplicate_charge",
                    "instructions": "Does the customer report a duplicate charge?",
                },
                {
                    "type": "choice",
                    "name": "team",
                    "instructions": "Which team should handle this ticket?",
                    "choices": [{"value": "billing"}, {"value": "shipping"}],
                },
            ],
        )

    assert decision.model == model
    predicate, choice = decision.answers
    assert (predicate.name, predicate.type) == ("duplicate_charge", "predicate")
    assert 0 <= predicate.probability <= 1
    assert (choice.name, choice.type) == ("team", "choice")
    assert {option.value for option in choice.probabilities} == {"billing", "shipping"}
    assert choice.choice in {"billing", "shipping"}
    assert sum(option.probability for option in choice.probabilities) == pytest.approx(1, abs=1e-4)
    assert decision.usage.input_tokens > 0
    assert decision.usage.output_tokens > 0


def test_system_one(endpoint, model):
    with httpx.Client(timeout=60) as client:
        response = client.post(
            f"{endpoint}/v1/systemone",
            json={
                "model": model,
                "state": {"ticket": "My card was charged twice for the same order."},
                "questions": {
                    "team": {
                        "type": "choice",
                        "instructions": "Which team should handle this ticket?",
                        "criteria": {"billing": "payments and refunds", "shipping": "deliveries"},
                    },
                },
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

    assert response.status_code == 200, response.text
    decision = response.json()
    assert decision["model"] == model
    assert decision["object"] == "structured_decision"
    assert set(decision["answers"]) == {"team"}
    answer = decision["answers"]["team"]
    assert answer["type"] == "choice"
    assert set(answer["probabilities"]) == {"billing", "shipping"}
    assert answer["choice"] in answer["probabilities"]
    assert sum(answer["probabilities"].values()) == pytest.approx(1, abs=1e-4)
    assert decision["usage"]["input_tokens"] > 0
    assert decision["usage"]["output_tokens"] > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
