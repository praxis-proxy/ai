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
_started_at = None


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


def pytest_sessionstart(session):
    global _started_at
    _started_at = timestamp()


def pytest_deselected(items):
    _deselected.extend(item.nodeid for item in items)


def pytest_collection_finish(session):
    _selected.extend(item.nodeid for item in session.items)


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
    cases = [_cases.get(nodeid, {"id": nodeid, "outcome": "unexecuted", "reason": "test did not run"}) for nodeid in _selected]
    data = {
        "exit_code": int(exitstatus),
        "started_at": _started_at,
        "finished_at": timestamp(),
        "selected": cases,
        "deselected": [{"id": nodeid, "reason": "excluded by test selection"} for nodeid in _deselected],
        "dependencies": {
            name: installed_version(name)
            for name in ("openai", "anthropic", "pytest", "httpx")
        },
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
