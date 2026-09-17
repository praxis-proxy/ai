"""Pytest configuration shared by the OpenAI SDK integration suites."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "critical_vllm: small live-vLLM smoke coverage required on pull requests",
    )
    config.addinivalue_line(
        "markers",
        "real_inference: requires a real model to consume transformed context",
    )
    config.addinivalue_line(
        "markers",
        "vllm_compat: requires behavior specific to the real vLLM frontend/backend",
    )
