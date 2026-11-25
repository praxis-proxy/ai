# /// script
# requires-python = ">=3.11"
# dependencies = ["openai==3.49.0", "pytest==8.4.2"]
# ///
"""Live vLLM Embeddings checks through a loopback Praxis gateway."""

import math
import os
import sys

import pytest
from openai import NotFoundError, OpenAI


@pytest.fixture(scope="module")
def client():
    if os.getenv("PRAXIS_LIVE_VLLM_EMBEDDINGS") != "1":
        pytest.skip("Set PRAXIS_LIVE_VLLM_EMBEDDINGS=1 to opt in")
    with OpenAI(
        base_url="http://127.0.0.1:8080/v1",
        api_key="local-test",
        max_retries=0,
        timeout=60,
    ) as sdk:
        yield sdk


def test_embedding_batch(client):
    model = os.environ["EMBEDDING_MODEL"]
    inputs = ["A blue sky.", "A deep ocean."]
    batch = client.embeddings.create(model=model, input=inputs, encoding_format="float")
    assert batch.model == model
    assert [item.index for item in batch.data] == [0, 1]
    for item, text in zip(batch.data, inputs, strict=True):
        single = client.embeddings.create(model=model, input=text, encoding_format="float")
        assert len(item.embedding) == 768
        assert all(math.isfinite(value) for value in item.embedding)
        assert item.embedding == pytest.approx(single.data[0].embedding, abs=1e-4)
    assert batch.data[0].embedding != batch.data[1].embedding
    assert batch.usage.prompt_tokens > 0
    assert batch.usage.total_tokens == batch.usage.prompt_tokens


def test_unknown_embedding_model(client):
    with pytest.raises(NotFoundError):
        client.embeddings.create(model="praxis-missing-embedding-model", input="hello")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
