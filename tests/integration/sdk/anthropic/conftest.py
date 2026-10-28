"""Pytest configuration shared by the Anthropic SDK integration suites."""

import importlib.util
import os
from pathlib import Path

import pytest


def _load_shared_sdk_conftest():
    """Load ``tests/integration/sdk/conftest.py`` by explicit path.

    Suites in this directory import its subprocess helpers
    (``find_binary``, ``free_port``, ``wait_for_proxy``) via
    ``from conftest import ...``. That used to resolve to the sdk-level
    conftest because this directory had none of its own; now that it does,
    the bare module name resolves here instead. Re-exporting the helpers
    keeps those imports pointed at the shared implementations.
    """
    path = Path(__file__).resolve().parents[1] / "conftest.py"
    spec = importlib.util.spec_from_file_location("_praxis_sdk_shared_conftest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_shared = _load_shared_sdk_conftest()
find_binary = _shared.find_binary
free_port = _shared.free_port
wait_for_proxy = _shared.wait_for_proxy


def pytest_addoption(parser):
    parser.addoption(
        "--anthropic-sdk-version",
        action="store",
        default=os.getenv("ANTHROPIC_SDK_VERSION", "auto"),
        choices=["auto", "1.x", "2.x"],
        help="Specify expected Anthropic SDK version lane (1.x or 2.x)",
    )


def pytest_configure(config):
    # This directory is itself named `anthropic`, so when the real SDK is
    # absent `import anthropic` resolves to it as an implicit namespace
    # package rather than failing. Such a module has no `__version__`, so
    # read it defensively (mirrors the openai conftest).
    try:
        import anthropic

        installed_version = getattr(anthropic, "__version__", "not-installed")
    except ImportError:
        installed_version = "not-installed"

    if installed_version.startswith("1."):
        detected_lane = "1.x"
    elif installed_version.startswith("2."):
        detected_lane = "2.x"
    else:
        detected_lane = "unknown"

    config._anthropic_sdk_version = installed_version
    config._anthropic_sdk_lane = detected_lane

    requested_lane = config.getoption("--anthropic-sdk-version")
    if requested_lane != "auto" and requested_lane != detected_lane:
        raise pytest.UsageError(
            f"Requested SDK version lane '{requested_lane}', but installed anthropic "
            f"package version is {installed_version} ({detected_lane})."
        )


def pytest_report_header(config):
    version = getattr(config, "_anthropic_sdk_version", "unknown")
    lane = getattr(config, "_anthropic_sdk_lane", "unknown")
    return f"Anthropic SDK Version: {version} (Lane: {lane})"


@pytest.fixture(scope="session")
def anthropic_sdk_info(request):
    """Return tuple of (installed_version, version_lane)."""
    config = request.config
    return (
        getattr(config, "_anthropic_sdk_version", "unknown"),
        getattr(config, "_anthropic_sdk_lane", "unknown"),
    )


# Environment variables the Anthropic SDK reads at client construction. A
# developer shell commonly exports ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL,
# ANTHROPIC_AUTH_TOKEN, ANTHROPIC_CUSTOM_HEADERS (which can inject a stray
# Authorization header), and ANTHROPIC_*_MODEL picks. The SDK picks these up
# and would otherwise leak extra credentials or headers onto the wire, making
# deterministic wire-header assertions impossible. Tests construct clients
# with explicit arguments, so scrub the ambient environment.
#
# The scrub is session-scoped and runs before any module-scoped client
# fixture constructs its SDK client, because the SDK reads these variables
# once at construction time. MonkeyPatch restores the original values when
# the session ends, so later code in the same pytest process (other
# directories' suites, plugins) sees the caller's environment again.
_ANTHROPIC_ENV_PREFIXES = ("ANTHROPIC_",)


@pytest.fixture(autouse=True, scope="session")
def _sanitize_anthropic_env():
    with pytest.MonkeyPatch.context() as mp:
        for name in list(os.environ):
            if name.startswith(_ANTHROPIC_ENV_PREFIXES):
                mp.delenv(name)
        yield
