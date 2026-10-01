"""Focused checks for native GPU report and release-note behavior."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/vllm_qualification.py"
spec = importlib.util.spec_from_file_location("vllm_qualification", SCRIPT)
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)
plugin_spec = importlib.util.spec_from_file_location(
    "qualification_plugin", Path(__file__).resolve().parents[1] / "integration/sdk/openai/qualification_plugin.py"
)
plugin = importlib.util.module_from_spec(plugin_spec)
plugin_spec.loader.exec_module(plugin)


def result(cases, exit_code=0):
    return {"selected": [{"id": name, "outcome": outcome, "reason": reason}
                         for name, outcome, reason in cases],
            "deselected": [{"id": "excluded", "reason": "excluded by test selection"}],
            "exit_code": exit_code, "dependencies": {"openai": "2.9.0"}}


def report():
    return {
        "schema_version": 1, "profile": qualification.PROFILE,
        "requested": True, "status": "passed", "reason": "selected cases completed",
        "workflow": {"run_id": 17, "attempt": 2, "url": "https://github.com/praxis-proxy/ai/actions/runs/17"},
        "gateway": {"checkout_sha": "a" * 40, "binary_sha256": "b" * 64,
                    "praxis_version": "0.7.2"},
        "backend": {"version": "0.30.0", "local_image_id": "sha256:" + "c" * 64},
        "model": {"identifier": "Qwen/Qwen3-8B", "resolved_revision": None},
        "configuration": {"reference_path": "examples/configs/openai/responses/full-flow-agentic.yaml",
                          "sha256": "d" * 64, "storage_backend": "postgresql"},
        "suites": {"native_responses": {"status": "passed",
                                        "selected": [{"id": "native text", "outcome": "passed", "reason": ""}],
                                        "totals": {"passed": 2, "xfailed": 1},
                                        "dependencies": {"openai": "2.9.0"}},
                   "credentialed_tools": {"status": "skipped"},
                   "claude_acceptance": {"status": "success"},
                   "codex_acceptance": {"status": "skipped"}},
        "limitations": qualification.LIMITATIONS,
    }


class QualificationTest(unittest.TestCase):
    def test_strict_xpass_keeps_unexpected_pass_classification(self):
        plugin.pytest_runtest_logreport(SimpleNamespace(
            nodeid="strict_xpass", outcome="failed", when="call",
            longrepr="[XPASS(strict)] contract changed", skipped=False, passed=False,
        ))
        self.assertEqual(plugin._cases["strict_xpass"]["outcome"], "xpassed")

    def test_selected_deselected_and_xfail_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(result([("passed", "passed", ""),
                                               ("expected", "xfailed", "known boundary"),
                                               ("skipped", "skipped", "no credentials")])) )
            suite = qualification.suite("native_responses", path)
        self.assertEqual(suite["status"], "passed")
        self.assertEqual(suite["totals"]["xfailed"], 1)
        self.assertEqual(len(suite["deselected"]), 1)
        self.assertEqual(suite["selected"][2]["reason"], "no credentials")

    def test_missing_zero_and_failed_cases_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            self.assertEqual(qualification.suite("native", path)["status"], "unexecuted")
            path.write_text(json.dumps(result([])))
            self.assertEqual(qualification.suite("native", path)["status"], "incomplete")
            path.write_text(json.dumps(result([("a", "failed", "assertion")])) )
            self.assertEqual(qualification.suite("native", path)["status"], "failed")
            path.write_text(json.dumps(result([("a", "xpassed", "contract changed")])) )
            self.assertEqual(qualification.suite("native", path)["status"], "needs_review")
            path.write_text(json.dumps(result([("a", "skipped", "no secret")])) )
            self.assertEqual(qualification.suite("native", path)["status"], "skipped")

    def test_hosted_summary_handles_runner_and_checkout_failures(self):
        missing = qualification.finish("/no/report.json", True, "failure", "skipped", "skipped", "skipped",
                                       17, 1, "https://example.test/run", "a" * 40)
        self.assertEqual(missing["status"], "incomplete")
        self.assertEqual(missing["phases"]["provisioning"], "failure")
        mismatch = report()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.json"
            path.write_text(json.dumps(mismatch))
            finished = qualification.finish(path, True, "success", "success", "success", "success",
                                            17, 2, "https://example.test/run", "z" * 40)
        self.assertEqual(finished["status"], "incomplete")
        old_attempt = report()
        old_attempt["workflow"]["attempt"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.json"
            path.write_text(json.dumps(old_attempt))
            finished = qualification.finish(path, True, "success", "success", "skipped", "skipped",
                                            17, 2, "https://example.test/run", "a" * 40)
        self.assertEqual(finished["status"], "incomplete")
        self.assertIn("current-attempt", finished["reason"])
        missing_binary = report()
        missing_binary["gateway"]["binary_sha256"] = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.json"
            path.write_text(json.dumps(missing_binary))
            finished = qualification.finish(path, True, "success", "success", "skipped", "skipped",
                                            17, 2, "https://example.test/run", "a" * 40)
        self.assertEqual(finished["status"], "incomplete")
        self.assertIn("gateway binary hash", finished["reason"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.json"
            path.write_text(json.dumps(report()))
            post_test_failure = qualification.finish(path, True, "success", "failure", "skipped", "skipped",
                                                     17, 2, "https://example.test/run", "a" * 40)
            not_requested = qualification.finish(path, False, "success", "success", "skipped", "skipped",
                                                 17, 2, "https://example.test/run", "a" * 40)
        self.assertEqual(post_test_failure["status"], "incomplete")
        self.assertEqual(not_requested["status"], "not_requested")

    def test_optional_acceptance_scenarios_keep_provenance_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "acceptance.json"
            path.write_text(json.dumps({"run_id": 17, "attempt": 2,
                                        "checkout_sha": "a" * 40,
                                        "cases": [{"id": "codex_native", "status": "success"},
                                                  {"id": "codex_translated", "status": "failure"}]}))
            result = qualification.acceptance(path, "failure", 17, 2, "a" * 40, "Codex")
            self.assertEqual([case["status"] for case in result["cases"]], ["success", "failure"])
            wrong_attempt = qualification.acceptance(path, "success", 17, 3, "a" * 40, "Codex")
            self.assertEqual(wrong_attempt["status"], "unreported")

    def test_latest_gpu_attempt_does_not_fall_back_to_old_green(self):
        runs = [{"id": 10, "status": "completed"}, {"id": 12, "status": "completed"},
                {"id": 13, "status": "completed"}]
        jobs = {10: [{"name": "Run complete live vLLM Responses suite (GPU)", "conclusion": "success"}],
                12: [{"name": "Start ephemeral EC2 GPU runner", "conclusion": "failure"}],
                13: [{"name": "Run complete live vLLM Responses suite (GPU)", "conclusion": "skipped"}]}
        self.assertEqual(qualification.choose_run(runs, jobs)["id"], 12)

    def test_exact_run_attempt_checkout_and_profile_required(self):
        source = report()
        run = {"id": 17, "run_attempt": 2, "head_sha": "a" * 40}
        self.assertIsNone(qualification.verify(source, run, "a" * 40))
        for mutation in (
            {"profile": "simulator"},
            {"gateway": {"checkout_sha": "z" * 40}},
            {"workflow": {"run_id": 17, "attempt": 1}},
            {"suites": {"native_responses": {"status": "passed", "selected": []}}},
        ):
            altered = {**source, **mutation}
            self.assertIsNotNone(qualification.verify(altered, run, "a" * 40))

    def test_missing_evidence_and_notes_idempotency(self):
        missing = qualification.unavailable("a" * 40, "artifact expired")
        section = qualification.render(missing)
        self.assertIn("unavailable for this release commit", section)
        self.assertNotIn("Native selected cases:", section)
        first = qualification.compose_notes("## Changelog\n\n- change\n", section)
        second = qualification.compose_notes(first, section)
        self.assertEqual(first, second)
        self.assertEqual(second.count(qualification.START), 1)
        self.assertIn("- change", second)

    def test_reviewable_release_note_fixture(self):
        fixture = Path(__file__).with_name("release-notes.fixture.md").read_text()
        notes = qualification.compose_notes("## Changelog\n\n- Example change\n", qualification.render(report()))
        self.assertEqual(notes, fixture)
        self.assertIn("OpenAI Python `2.9.0`", notes)
        self.assertIn("Credentialed Tools: skipped", notes)


if __name__ == "__main__":
    unittest.main()
