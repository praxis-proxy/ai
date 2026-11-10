"""Release support report assembly, honest missing-evidence handling, and diffing."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/support_report.py"
spec = importlib.util.spec_from_file_location("support_report", SCRIPT)
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)

TAG = "v0.10.0"
SHA = "a" * 40


def matrix(cell_status="translated"):
    return {
        "schema_version": 1,
        "source": "docs/conformance/translation-matrix.yaml",
        "conformance_claim_caveat": "Qualified number, not the bare percentage.",
        "preconditions": [{"id": "pre", "summary": "Applies everywhere", "evidence": "f.rs:1", "detail": []}],
        "features": [{"id": "text-chat", "label": "Plain text"}],
        "native_surfaces": [{"id": "responses", "client_api": "OpenAI Responses",
                             "operations": 1, "status": "native", "note": "n", "evidence": "f.rs:2"}],
        "surfaces": [{
            "id": "anthropic-to-chat", "short": "Anthropic -> CC",
            "client_api": "Anthropic Messages", "backend_api": "OpenAI Chat Completions",
            "filters": ["f"], "clean_rejection_requires": [], "unknown_field_policy": "reject",
            "cells": [
                {"feature": "text-chat", "label": "Plain text", "status": cell_status,
                 "note": None, "evidence": "f.rs:3", "blocker": None},
                {"feature": "reasoning", "label": "Reasoning", "status": "dropped",
                 "note": "Silently dropped", "evidence": "f.rs:4", "blocker": "TODO-ISSUE fix"},
            ],
        }],
        "cross_surface_gaps": [],
        "status_totals": {cell_status: 1, "dropped": 1},
    }


def openai():
    return {
        "scope": {"areas": ["Conversations", "Responses"],
                  "include_deprecated": False, "include_beta": False},
        "reference": {"upstream": {"revision": "b" * 40}},
        "capability_coverage": {"total_operations": 7, "covered_operations_count": 7,
                                "coverage_percent": 100.0,
                                "ignored": {"deprecated": 0, "beta": 7},
                                "by_area": [{"area": "Responses", "total": 7,
                                             "covered": 7, "coverage_percent": 100.0}]},
        "runtime_verification": {"all_passed": True},
    }


def openresponses():
    return {
        "suite": {"commit": "c" * 40, "bun": "1.3.12", "zod": "3.25.76"},
        "coverage": {"supported": 7, "unsupported": 0, "inapplicable": 10,
                     "in_scope": 7, "total": 17, "percent": 100.0},
        "supported": ["basic"], "unsupported": [], "inapplicable": ["websocket"],
    }


def qualification(sha=SHA):
    return {
        "schema_version": 3, "profile": "responses-gateway-live-vllm-gpu",
        "requested": True, "status": "passed", "reason": "selected cases completed",
        "workflow": {"run_id": 17, "attempt": 1,
                     "url": "https://github.com/praxis-proxy/ai/actions/runs/17"},
        "gateway": {"checkout_sha": sha, "binary_sha256": "d" * 64, "praxis_version": "0.8.0"},
        "backend": {"version": "0.32.0", "local_image_id": "sha256:" + "e" * 64},
        "model": {"identifier": "Qwen/Qwen3-8B", "resolved_revision": "f" * 40},
        "configuration": {"reference_path": "examples/configs/x.yaml", "sha256": "1" * 64},
        "suites": {
            "native_responses": {"status": "passed", "totals": {"passed": 3},
                                 "dependencies": {"openai": "2.9.0"}},
            "codex_acceptance": {"status": "success",
                                 "cases": [{"id": "codex_native", "status": "success"},
                                           {"id": "codex_translated", "status": "failure"}]},
            "claude_acceptance": {"status": "success",
                                  "cases": [{"id": "claude_native_edits", "status": "success"}]},
        },
        "limitations": [{"text": "Selected text behavior only", "url": None}],
    }


def release_qualification():
    return {
        "schema_version": 1, "profile": "release-ci-evidence", "release_sha": SHA,
        "selection": "exact commit only",
        "limitations": ["No older green fallback"],
        "workflows": [{"name": "Tests", "workflow": "tests.yaml", "scope": "tests",
                       "status": "success", "reason": "job outcomes",
                       "job_totals": {"success": 2},
                       "run": {"html_url": "https://example.invalid/run"}}],
    }


def write_fixtures(directory, *, matrix_value=None, qualification_value=None,
                   release_qualification_value=None):
    directory = Path(directory)
    files = {
        "matrix.json": matrix_value if matrix_value is not None else matrix(),
        "openai.json": openai(),
        "openresponses.json": openresponses(),
    }
    for name, value in files.items():
        (directory / name).write_text(json.dumps(value), encoding="utf-8")
    if qualification_value is not None:
        (directory / "qualification.json").write_text(
            json.dumps(qualification_value), encoding="utf-8")
    if release_qualification_value is not None:
        (directory / "release-qualification.json").write_text(
            json.dumps(release_qualification_value), encoding="utf-8")
    return directory


def arguments(directory, **overrides):
    directory = Path(directory)
    values = {
        "root": str(directory),
        "tag": TAG, "sha": SHA, "version": None,
        "generated_at": "2026-11-09T00:00:00+00:00",
        "translation_matrix": str(directory / "matrix.json"),
        "openai_conformance": str(directory / "openai.json"),
        "openresponses": str(directory / "openresponses.json"),
        "qualification": None, "release_qualification": None,
        "previous": None, "existing_notes": None,
        "output_dir": str(directory / "out"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def generate(args):
    with contextlib.redirect_stdout(io.StringIO()):
        support.generate(args)
    return (json.loads(Path(args.output_dir, "support-report.json").read_text(encoding="utf-8")),
            Path(args.output_dir, "support-report.md").read_text(encoding="utf-8"))


class SupportReportTest(unittest.TestCase):
    def test_missing_live_evidence_is_rendered_not_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            write_fixtures(directory)
            report, markdown = generate(arguments(directory))

            self.assertEqual(report["live_validation"]["status"], "not validated")
            self.assertEqual(report["ci_evidence"]["status"], "unavailable")
            self.assertEqual(report["evidence"], "partial")
            self.assertIn("### Live validation", markdown)
            self.assertIn("**not validated**", markdown)
            self.assertIn("No live claims are made", markdown)
            self.assertNotIn("None", markdown)
            self.assertEqual(support.verify(report, markdown, TAG, SHA), [])

    def test_live_evidence_populates_provenance_and_scenarios(self):
        with tempfile.TemporaryDirectory() as directory:
            write_fixtures(directory, qualification_value=qualification(),
                           release_qualification_value=release_qualification())
            report, markdown = generate(arguments(
                directory,
                qualification=str(Path(directory) / "qualification.json"),
                release_qualification=str(Path(directory) / "release-qualification.json")))

            self.assertEqual(report["live_validation"]["status"], "passed")
            self.assertEqual(report["evidence"], "complete")
            self.assertEqual(report["provenance"]["gateway"]["praxis_version"], "0.8.0")
            self.assertEqual(report["ci_evidence"]["workflows"][0]["name"], "Tests")
            self.assertIn("codex_native", markdown)
            self.assertIn("codex_translated", markdown)
            self.assertIn("Qwen3-8B", markdown)
            self.assertIn("| Tests | success |", markdown)

    def test_provenance_mismatch_refuses_live_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            write_fixtures(directory, qualification_value=qualification(sha="9" * 40))
            report, markdown = generate(arguments(
                directory, qualification=str(Path(directory) / "qualification.json")))

            self.assertEqual(report["live_validation"]["status"], "provenance mismatch")
            self.assertIn("No live claims are made", markdown)
            self.assertNotIn("codex_native", markdown)
            # The other checkout's provenance must not be attributed to this release.
            self.assertEqual(report["provenance"]["gateway"], {})
            self.assertEqual(report["provenance"]["backend"], {})
            self.assertEqual(report["provenance"]["model"], {})
            self.assertEqual(report["provenance"]["configuration"], {})
            self.assertNotIn("0.8.0", json.dumps(report["provenance"]))
            self.assertNotIn("d" * 64, json.dumps(report["provenance"]))
            provenance = markdown.split("### Provenance", 1)[1].split("###", 1)[0]
            self.assertNotIn("0.8.0", provenance)
            self.assertNotIn("d" * 64, provenance)
            self.assertIn("| Praxis core | unavailable |", provenance)
            self.assertEqual(report["evidence"], "partial")

    def test_unrun_ci_workflows_do_not_complete_the_evidence(self):
        for status in ("not_run", "unavailable"):
            with tempfile.TemporaryDirectory() as directory:
                evidence = release_qualification()
                evidence["workflows"][0]["status"] = status
                write_fixtures(directory,
                               qualification_value=qualification(),
                               release_qualification_value=evidence)
                report, markdown = generate(arguments(
                    directory,
                    qualification=str(Path(directory) / "qualification.json"),
                    release_qualification=str(Path(directory) / "release-qualification.json")))

                self.assertEqual(report["live_validation"]["status"], "passed")
                self.assertEqual(len(report["ci_evidence"]["workflows"]), 1)
                self.assertEqual(report["evidence"], "partial")
                self.assertIn(status, markdown)
                self.assertEqual(support.verify(report, markdown, TAG, SHA), [])

    def test_ci_evidence_for_another_commit_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = release_qualification()
            evidence["release_sha"] = "b" * 40
            write_fixtures(directory, qualification_value=qualification(),
                           release_qualification_value=evidence)
            report, markdown = generate(arguments(
                directory,
                qualification=str(Path(directory) / "qualification.json"),
                release_qualification=str(Path(directory) / "release-qualification.json")))

            self.assertEqual(report["ci_evidence"]["status"], "provenance mismatch")
            self.assertEqual(report["ci_evidence"]["workflows"], [])
            self.assertEqual(report["evidence"], "partial")
            self.assertIn("not release commit", markdown)
            self.assertNotIn("| Tests |", markdown)
            self.assertEqual(support.verify(report, markdown, TAG, SHA), [])

    def test_ci_evidence_without_a_release_sha_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = release_qualification()
            del evidence["release_sha"]
            write_fixtures(directory, qualification_value=qualification(),
                           release_qualification_value=evidence)
            report, markdown = generate(arguments(
                directory,
                qualification=str(Path(directory) / "qualification.json"),
                release_qualification=str(Path(directory) / "release-qualification.json")))

            self.assertEqual(report["ci_evidence"]["status"], "unavailable")
            self.assertEqual(report["ci_evidence"]["workflows"], [])
            self.assertEqual(report["evidence"], "partial")
            self.assertIn("no recorded release SHA", markdown)
            self.assertEqual(support.verify(report, markdown, TAG, SHA), [])

    def test_failed_client_suite_is_rendered_without_scenarios(self):
        with tempfile.TemporaryDirectory() as directory:
            value = qualification()
            value["suites"]["codex_acceptance"] = {
                "status": "failure",
                "reason": "Codex CLI job failed before writing its scenario artifact",
                "cases": []}
            write_fixtures(directory, qualification_value=value)
            report, markdown = generate(arguments(
                directory, qualification=str(Path(directory) / "qualification.json")))

            self.assertEqual(report["live_validation"]["status"], "passed")
            self.assertIn("Client acceptance suites:", markdown)
            self.assertIn("| Codex | failure | 0 | Codex CLI job failed before writing "
                          "its scenario artifact |", markdown)
            self.assertNotIn("codex_native", markdown)
            self.assertIn("claude_native_edits", markdown)

    def test_qualification_without_a_checkout_sha_is_a_provenance_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            value = qualification()
            value["status"] = "incomplete"
            value["reason"] = "actual checkout SHA unavailable"
            value["gateway"]["checkout_sha"] = None
            write_fixtures(directory, qualification_value=value)
            report, markdown = generate(arguments(
                directory, qualification=str(Path(directory) / "qualification.json")))

            self.assertEqual(report["live_validation"]["status"], "provenance mismatch")
            self.assertIn("Qualification checkout SHA unavailable for release commit", markdown)
            self.assertIn("No live claims are made", markdown)
            self.assertNotIn("codex_native", markdown)
            self.assertEqual(report["provenance"]["gateway"], {})
            self.assertEqual(report["evidence"], "partial")

    def test_unavailable_and_not_requested_statuses_are_preserved(self):
        for status in ("unavailable", "not_requested"):
            with tempfile.TemporaryDirectory() as directory:
                value = {"schema_version": 3, "profile": "responses-gateway-live-vllm-gpu",
                         "requested": False, "status": status, "reason": "no run",
                         "gateway": {"checkout_sha": None}, "workflow": {"url": None},
                         "backend": {}, "model": {}, "configuration": {},
                         "suites": {}, "limitations": []}
                write_fixtures(directory, qualification_value=value)
                report, markdown = generate(arguments(
                    directory, qualification=str(Path(directory) / "qualification.json")))

                self.assertEqual(report["live_validation"]["status"], status)
                self.assertEqual(report["evidence"], "partial")
                self.assertIn("No live claims are made", markdown)
                self.assertEqual(support.verify(report, markdown, TAG, SHA), [])

    def test_conformance_percentage_carries_its_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            write_fixtures(directory)
            _, markdown = generate(arguments(directory))

            self.assertIn("The count covers non-beta, non-deprecated Conversations and "
                          "Responses operations and excludes 7 beta operations.", markdown)
            self.assertIn("Qualified number, not the bare percentage.", markdown)

    def test_zero_counts_render_as_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            value = matrix()
            value["surfaces"][0]["cells"][0]["status"] = "rejected"
            write_fixtures(directory, matrix_value=value)
            _, markdown = generate(arguments(directory))

            self.assertIn("| Anthropic -> CC |", markdown)
            row = [line for line in markdown.splitlines()
                   if line.startswith("| Anthropic")][0]
            self.assertIn("| 0 |", row)

    def test_previous_release_diff_reports_status_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            write_fixtures(directory)
            first, _ = generate(arguments(directory))
            previous = Path(directory) / "previous.json"
            previous.write_text(json.dumps(first), encoding="utf-8")

            changed = matrix(cell_status="dropped")
            changed["surfaces"][0]["cells"][0]["blocker"] = "TODO-ISSUE now a gap"
            write_fixtures(directory, matrix_value=changed)
            report, markdown = generate(arguments(
                directory, previous=str(previous)))

            self.assertEqual(report["previous"]["release"], TAG)
            self.assertEqual(report["previous"]["cells"], [
                {"surface": "anthropic-to-chat", "feature": "text-chat",
                 "from": "translated", "to": "dropped"}])
            self.assertIn("### Changes since v0.10.0", markdown)

    def test_verify_rejects_wrong_commit_and_missing_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            write_fixtures(directory)
            report, markdown = generate(arguments(directory))

            self.assertTrue(support.verify(report, markdown, TAG, "b" * 40))
            self.assertTrue(support.verify(report, "no markers"))

    def test_release_notes_section_is_replaced_not_duplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            write_fixtures(directory)
            notes = Path(directory) / "notes.md"
            notes.write_text("# Release notes\n\nExisting body.\n", encoding="utf-8")
            generate(arguments(directory, existing_notes=str(notes)))
            out_notes = Path(directory) / "out/release-notes.md"
            generate(arguments(directory, existing_notes=str(out_notes)))

            composed = (Path(directory) / "out/release-notes.md").read_text(encoding="utf-8")
            self.assertEqual(composed.count(support.START), 1)
            self.assertEqual(composed.count(support.END), 1)
            self.assertIn("Existing body.", composed)

    def test_harness_pins_are_read_from_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "tests/integration/fixtures/codex-cli/pin.toml"
            manifest.parent.mkdir(parents=True)
            manifest.write_text('[codex]\nversion = "0.144.1"\n', encoding="utf-8")
            pins = support.harness_pins(Path(directory))

            self.assertEqual(pins[0]["client"], "Codex")
            self.assertEqual(pins[0]["version"], "0.144.1")


if __name__ == "__main__":
    unittest.main()
