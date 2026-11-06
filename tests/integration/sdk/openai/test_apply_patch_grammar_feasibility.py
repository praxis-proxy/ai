# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Praxis Contributors
"""Backend compiler-behavior gate for the Codex ``apply_patch`` grammar (#1834).

This establishes XGrammar's *compiler behavior* for the escaping-aware backend
pattern that ``openai_client_tool_compat`` emits on a lowered ``apply_patch``
grammar tool's ``input``: that it constrains a JSON function argument to the
pinned patch grammar, and which pattern *encoding* XGrammar actually enforces.

It is NOT end-to-end enforcement evidence — a real model under automatic/forced
tool choice on both inference paths is the GPU qualification (see the live
acceptance test). Validated under ``xgrammar==0.2.8`` / vLLM 0.32.0.

The patterns are read from the committed fixtures that a Rust drift-guard test
(``backend_pattern_fixtures_match_consts``) pins to the runtime constants, so the
gate exercises the exact string the backend receives.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

xgr = pytest.importorskip("xgrammar", reason="xgrammar is only present in the vLLM test environment")

REPO_ROOT = Path(__file__).resolve().parents[4]
PATTERN_DIR = REPO_ROOT / "apis" / "src" / "openai" / "responses" / "openai_client_tool_compat"
BACKEND_PATTERN_BASE = (PATTERN_DIR / "apply_patch_backend_pattern_base.txt").read_text().rstrip("\n")
BACKEND_PATTERN_ENVID = (PATTERN_DIR / "apply_patch_backend_pattern_envid.txt").read_text().rstrip("\n")

try:
    XGRAMMAR_VERSION = version("xgrammar")
except PackageNotFoundError:  # pragma: no cover - importorskip already guards import
    XGRAMMAR_VERSION = "unknown"

# The escaping behaviour this gate establishes is version-specific; it was validated
# on xgrammar 0.2.8 (the version the pinned vLLM 0.32.0 acceptance image bundles).
# On any other version the gate skips loudly rather than assert behaviour it has not
# verified — a visible signal to re-establish the evidence, never a silent pass.
pytestmark = pytest.mark.skipif(
    XGRAMMAR_VERSION != "0.2.8",
    reason=f"apply_patch feasibility gate is pinned to xgrammar 0.2.8, found {XGRAMMAR_VERSION}",
)

# Raw (decoded-value) host pattern — the form a spec-compliant JSON-Schema backend
# would match against the decoded string. Used to demonstrate that XGrammar matches
# the *encoded* JSON-string body instead, so the escaped `JsonString` encoding is
# required (the scoping ruling). This mirrors `APPLY_PATCH_HOST_PATTERN_BASE`.
_CH = r"[^\n]"
HOST_PATTERN_BASE = (
    r"\*\*\* Begin Patch\n(?:(?:\*\*\* Add File: " + _CH + r"+\n(?:\+" + _CH + r"*\n)+"
    r"|\*\*\* Delete File: " + _CH + r"+\n"
    r"|\*\*\* Update File: " + _CH + r"+\n(?:\*\*\* Move to: " + _CH + r"+\n)?"
    r"(?:(?:@@(?: " + _CH + r"+)?\n|[+\- ]" + _CH + r"*\n)+(?:\*\*\* End of File\n)?)?))+"
    r"\*\*\* End Patch\n?"
)


def _schema(pattern: str | None) -> str:
    prop: dict = {"type": "string"}
    if pattern is not None:
        prop["pattern"] = pattern
    return json.dumps(
        {
            "type": "object",
            "properties": {"input": prop},
            "required": ["input"],
            "additionalProperties": False,
        }
    )


def _grammar(pattern: str | None):
    # any_whitespace=False mirrors vLLM strict function-calling.
    return xgr.Grammar.from_json_schema(_schema(pattern), any_whitespace=False)


def _accepts(grammar, patch: str, *, ensure_ascii: bool = True) -> bool:
    doc = json.dumps({"input": patch}, ensure_ascii=ensure_ascii)
    return xgr.testing._is_grammar_accept_string(grammar, doc)  # noqa: SLF001


VALID = [
    "*** Begin Patch\n*** Add File: a.txt\n+hello\n+world\n*** End Patch\n",
    "*** Begin Patch\n*** Delete File: old.py\n*** End Patch\n",
    "*** Begin Patch\n*** Update File: x.rs\n@@ fn main\n-old\n+new\n context\n*** End Patch\n",
    "*** Begin Patch\n*** Update File: a\n*** Move to: b\n context\n*** End of File\n*** End Patch\n",
    "*** Begin Patch\n*** Add File: a\n+1\n*** End Patch",  # optional final newline
    '*** Begin Patch\n*** Add File: a\n+say "hi" \\ bye\n*** End Patch\n',  # quotes + backslash
    "*** Begin Patch\n*** Add File: a\n+1\n*** Delete File: b\n*** End Patch\n",  # multiple hunks
]
INVALID = [
    "*** Begin Patch\n*** End Patch\n",  # no hunks
    "*** Begin Patch\n*** Add File: a\n+1\n*** End Patch\nGARBAGE\n",  # trailing garbage
    "*** Begin Patch\n*** Update File: a\nXnope\n*** End Patch\n",  # bad change prefix
    "*** Begin Patch\n*** Frobnicate File: a\n+1\n*** End Patch\n",  # bogus header
    "just some assistant prose, not a patch",  # plain text
    "",  # empty
]


def test_control_schema_without_pattern_accepts_anything():
    g = _grammar(None)
    assert _accepts(g, "literally anything"), "an unconstrained string must accept arbitrary input"


@pytest.mark.parametrize("patch", VALID)
def test_base_pattern_accepts_valid_patches(patch):
    assert _accepts(_grammar(BACKEND_PATTERN_BASE), patch)


@pytest.mark.parametrize("patch", INVALID)
def test_base_pattern_rejects_malformed_patches(patch):
    assert not _accepts(_grammar(BACKEND_PATTERN_BASE), patch)


def test_base_pattern_handles_unicode_both_encodings():
    patch = "*** Begin Patch\n*** Add File: café/日本.txt\n+naïve ☃\n*** End Patch\n"
    g = _grammar(BACKEND_PATTERN_BASE)
    assert _accepts(g, patch, ensure_ascii=True), "\\uXXXX-escaped unicode accepted"
    assert _accepts(g, patch, ensure_ascii=False), "raw-UTF-8 unicode accepted"


def test_environment_id_variant_gating():
    env_patch = "*** Begin Patch\n*** Environment ID: env-1\n*** Add File: a\n+1\n*** End Patch\n"
    plain = "*** Begin Patch\n*** Add File: a\n+1\n*** End Patch\n"
    env_grammar = _grammar(BACKEND_PATTERN_ENVID)
    assert _accepts(env_grammar, env_patch), "env-id grammar accepts the Environment ID line"
    assert _accepts(env_grammar, plain), "env-id is optional under its grammar"
    assert not _accepts(_grammar(BACKEND_PATTERN_BASE), env_patch), "base grammar rejects an Environment ID line"


def test_raw_newline_document_is_rejected():
    # A literal newline inside a JSON string is invalid JSON; the escaping-aware
    # pattern must reject it (it only matches the escaped \n form).
    raw = '{"input": "*** Begin Patch\n*** Add File: a\n+1\n*** End Patch\n"}'
    assert not xgr.testing._is_grammar_accept_string(_grammar(BACKEND_PATTERN_BASE), raw)  # noqa: SLF001


def test_decoded_encoding_does_not_enforce_under_xgrammar():
    # Evidence for the encoding-scoping ruling: XGrammar matches the RAW JSON-string
    # body, so the decoded-value (raw-newline) pattern does NOT accept a correctly
    # JSON-escaped patch. This is why the default `json_string` encoding is required
    # for XGrammar/vLLM and the raw form is reserved for a decoded-value backend.
    valid = "*** Begin Patch\n*** Add File: a\n+1\n*** End Patch\n"
    assert _accepts(_grammar(BACKEND_PATTERN_BASE), valid), "escaping-aware pattern accepts the escaped patch"
    assert not _accepts(_grammar(HOST_PATTERN_BASE), valid), "the decoded-value pattern does not enforce under XGrammar"
