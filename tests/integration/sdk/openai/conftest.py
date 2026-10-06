"""Pytest configuration shared by the OpenAI SDK integration suites."""

import os
import sys
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--sdk-version",
        action="store",
        default=os.getenv("SDK_VERSION", "auto"),
        choices=["auto", "2.x", "3.x"],
        help="Specify expected OpenAI SDK version lane (2.x or 3.x)",
    )


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
    config.addinivalue_line(
        "markers",
        "sdk_version(version): test targeted at a specific SDK version lane (e.g. 2.x, 3.x)",
    )

    # This directory is itself named `openai`, so when the real SDK is absent
    # `import openai` resolves to it as an implicit namespace package rather
    # than failing. Such a module has no `__version__`, so read it defensively:
    # suites that declare no `openai` dependency, such as the OpenResponses
    # conformance runner, must land in the `unknown` lane instead of raising
    # AttributeError out of `pytest_configure`.
    try:
        import openai

        installed_version = getattr(openai, "__version__", "not-installed")
    except ImportError:
        installed_version = "not-installed"

    detected_lane = "3.x" if installed_version.startswith("3.") else "2.x" if installed_version.startswith("2.") else "unknown"

    config._openai_sdk_version = installed_version
    config._openai_sdk_lane = detected_lane

    requested_lane = config.getoption("--sdk-version")
    if requested_lane != "auto" and requested_lane != detected_lane:
        raise pytest.UsageError(
            f"Requested SDK version lane '{requested_lane}', but installed openai "
            f"package version is {installed_version} ({detected_lane})."
        )


def pytest_report_header(config):
    version = getattr(config, "_openai_sdk_version", "unknown")
    lane = getattr(config, "_openai_sdk_lane", "unknown")
    backend = os.getenv("VLLM_TEST_BACKEND", "live")
    db_url = os.getenv("DATABASE_URL", "")
    db_backend = "postgres" if db_url.startswith("postgres") else "sqlite"
    return [
        f"OpenAI SDK Version: {version} (Lane: {lane})",
        f"Test Backend: {backend} | DB Backend: {db_backend}",
    ]


@pytest.fixture(scope="session")
def openai_sdk_info(request):
    """Return tuple of (installed_version, version_lane)."""
    config = request.config
    return (
        getattr(config, "_openai_sdk_version", "unknown"),
        getattr(config, "_openai_sdk_lane", "unknown"),
    )

