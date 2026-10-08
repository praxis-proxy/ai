"""Pytest hooks for the live GPU qualification report.

Loaded only by the GPU workflow with ``-p qualification_plugin``. The file is
written at session end, including collection and fixture failures. A killed
pytest process leaves no file, which the finalizer treats as incomplete.
"""

import json
import os
from importlib import metadata
from datetime import datetime, timezone
from pathlib import Path

_selected = []
_deselected = []
_cases = {}
_profiles = {}
_started_at = None

# These client fixtures select the actual gateway pipeline. Keep the mapping
# here so a test added with a new client fixture cannot silently count as
# native proof. "supporting" covers local stub or non-Responses paths.
PROFILE_FIXTURES = {
    "native": {
        "openai_client", "other_owner_openai_client", "compression_openai_client",
        "irr_streaming_client", "compact_client", "client_tool_compat_client",
        "agentic_client", "agentic_proxy", "file_search_client",
        "file_search_streaming_client",
    },
    "translation": {
        "chat_streaming_client", "reasoning_client", "web_search_chat_streaming_client",
        "client_tool_compat_chat_client", "translated_agentic_client",
        "file_search_chat_client", "live_tavily_client",
    },
    "supporting": {
        "witness_backend_client", "witness_tool_client",
        "witness_replay_limited_tool_client", "provider_compaction_client",
        "reasoning_capture_client", "file_resolve_stub_env", "model_rewrite_client",
        "model_rewrite_proxy",
    },
}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def installed_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def redact(value):
    text = str(value)
    for name in ("TAVILY_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HUGGING_FACE_HUB_TOKEN", "DATABASE_URL"):
        secret = os.environ.get(name)
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


def profile_for_item(item):
    override = getattr(item.obj, "qualification_profile", None)
    if override is not None:
        if override in PROFILE_FIXTURES:
            return override
        return "unclassified"
    fixtures = set(item.fixturenames)
    matches = [name for name, names in PROFILE_FIXTURES.items() if fixtures & names]
    return matches[0] if len(matches) == 1 else "unclassified"


def pytest_sessionstart(session):
    global _started_at
    _started_at = timestamp()


def pytest_deselected(items):
    _deselected.extend({"id": item.nodeid, "profile": profile_for_item(item),
                        "reason": "excluded by test selection"} for item in items)


def pytest_collection_finish(session):
    _selected.extend(item.nodeid for item in session.items)
    _profiles.update({item.nodeid: profile_for_item(item) for item in session.items})


def pytest_runtest_logreport(report):
    case = _cases.setdefault(report.nodeid, {"id": report.nodeid, "outcome": "unexecuted", "reason": ""})
    if report.outcome == "failed" and str(report.longrepr).startswith("[XPASS(strict)]"):
        case.update(outcome="xpassed", reason=redact(report.longrepr))
    elif report.outcome == "failed":
        case.update(outcome="failed", reason=redact(report.longrepr))
    elif hasattr(report, "wasxfail"):
        case.update(
            outcome="xfailed" if report.skipped else "xpassed",
            reason=redact(report.wasxfail),
        )
    elif report.skipped:
        case.update(outcome="skipped", reason=redact(report.longrepr))
    elif report.when == "call" and report.passed:
        case.update(outcome="passed", reason="")


def pytest_sessionfinish(session, exitstatus):
    destination = os.environ.get("PRAXIS_QUALIFICATION_RESULTS")
    if not destination:
        return
    sdk_version = getattr(session.config, "_openai_sdk_version", installed_version("openai"))
    sdk_lane = getattr(session.config, "_openai_sdk_lane", "unknown")
    cases = [{**_cases.get(nodeid, {"id": nodeid, "outcome": "unexecuted", "reason": "test did not run"}),
              "profile": _profiles[nodeid],
              "sdk_version": sdk_version,
              "sdk_lane": sdk_lane} for nodeid in _selected]
    deselected = [{**item, "sdk_version": sdk_version, "sdk_lane": sdk_lane} for item in _deselected]

    dependencies = {
        name: installed_version(name)
        for name in ("openai", "anthropic", "pytest", "httpx")
    }
    if sdk_version:
        dependencies["openai"] = sdk_version
    if sdk_lane and sdk_version:
        dependencies[f"openai_{sdk_lane}"] = sdk_version

    path = Path(destination)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            existing_selected = existing.get("selected", [])
            existing_deselected = existing.get("deselected", [])
            existing_exit_code = existing.get("exit_code", 0)
            existing_deps = existing.get("dependencies", {})
            cases = existing_selected + cases
            deselected = existing_deselected + deselected
            exitstatus = max(int(exitstatus), int(existing_exit_code))
            merged_deps = {**existing_deps, **dependencies}
            existing_openai = existing_deps.get("openai")
            current_openai = dependencies.get("openai")
            if existing_openai and current_openai and existing_openai != current_openai:
                tokens = [token.strip() for token in existing_openai.split(",") if token.strip()]
                if current_openai.strip() not in tokens:
                    tokens.append(current_openai.strip())
                    merged_deps["openai"] = ", ".join(tokens)
            dependencies = merged_deps
        except Exception as err:
            raise RuntimeError(f"Failed to read or merge existing qualification results from {path}: {err}") from err

    data = {
        "exit_code": int(exitstatus),
        "started_at": _started_at,
        "finished_at": timestamp(),
        "selected": cases,
        "deselected": deselected,
        "dependencies": dependencies,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
