"""Focused checks for native GPU report and release-note behavior."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


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
        "schema_version": qualification.VERSION, "profile": qualification.PROFILE,
        "requested": True, "status": "passed", "reason": "native and translation cases completed",
        "workflow": {"run_id": 17, "attempt": 2, "url": "https://github.com/praxis-proxy/ai/actions/runs/17"},
        "gateway": {"checkout_sha": "a" * 40, "binary_sha256": "b" * 64,
                    "praxis_version": "0.7.2"},
        "backend": {"version": "0.30.0", "local_image_id": "sha256:" + "c" * 64},
        "model": {"identifier": "Qwen/Qwen3-8B", "resolved_revision": None},
        "configuration": {"reference_path": "examples/configs/openai/responses/full-flow-agentic.yaml",
                          "sha256": "d" * 64, "storage_backend": "postgresql"},
        "suites": {"native_responses": {"status": "passed",
                                        "selected": [{"id": "native text", "profile": "native", "outcome": "passed", "reason": "", "sdk_version": "2.9.0", "sdk_lane": "2.x"}],
                                        "totals": {"passed": 1, "xfailed": 0},
                                        "dependencies": {"openai": "2.9.0"}},
                   "translation": {"status": "passed",
                                   "selected": [{"id": "translated stream", "profile": "translation", "outcome": "passed", "reason": "", "sdk_version": "2.9.0", "sdk_lane": "2.x"}],
                                   "totals": {"passed": 1, "xfailed": 0}},
                   "supporting": {"status": "passed",
                                  "selected": [{"id": "stubbed replay", "profile": "supporting", "outcome": "passed", "reason": "", "sdk_version": "2.9.0", "sdk_lane": "2.x"}],
                                  "totals": {"passed": 1, "xfailed": 0}},
                   "unclassified": {"status": "empty", "selected": []},
                   "credentialed_tools": {"status": "skipped"},
                   "claude_acceptance": {"status": "success"},
                   "codex_acceptance": {"status": "skipped"}},
        "limitations": qualification.LIMITATIONS,
    }


class QualificationTest(unittest.TestCase):
    def test_client_fixture_profiles_and_unknown_paths(self):
        def item(*fixtures, override=None):
            return SimpleNamespace(
                fixturenames=fixtures,
                obj=SimpleNamespace(qualification_profile=override),
            )

        self.assertEqual(plugin.profile_for_item(item("openai_client")), "native")
        self.assertEqual(plugin.profile_for_item(item("chat_streaming_client")), "translation")
        self.assertEqual(plugin.profile_for_item(item("reasoning_capture_client")), "supporting")
        self.assertEqual(plugin.profile_for_item(item("new_gateway_client")), "unclassified")
        self.assertEqual(plugin.profile_for_item(item("openai_client", "chat_streaming_client")), "unclassified")
        self.assertEqual(plugin.profile_for_item(item(override="supporting")), "supporting")

    def test_mixed_profiles_do_not_turn_skipped_native_into_passed_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            cases = directory / "cases.json"
            cases.write_text(json.dumps({
                "selected": [
                    {"id": "native_case", "profile": "native", "outcome": "skipped", "reason": "no native proof"},
                    {"id": "translation_case", "profile": "translation", "outcome": "passed", "reason": ""},
                ],
                "deselected": [], "exit_code": 0,
                "dependencies": {"openai": "2.9.0"},
            }))
            credentialed = directory / "credentialed.json"
            credentialed.write_text(json.dumps(result([("credentialed", "skipped", "no key")])))
            output = directory / "raw.json"
            with mock.patch.object(qualification, "command", return_value="available"), \
                 mock.patch.object(qualification, "model_revision", return_value=None):
                qualification.capture(SimpleNamespace(
                    responses=cases, credentialed=credentialed, setup="passed", output=output,
                ))
            captured = json.loads(output.read_text())

        self.assertEqual(captured["suites"]["native_responses"]["status"], "skipped")
        self.assertEqual(captured["suites"]["native_responses"]["totals"]["passed"], 0)
        self.assertEqual(captured["suites"]["translation"]["status"], "passed")
        self.assertEqual(captured["suites"]["translation"]["totals"]["passed"], 1)
        self.assertNotEqual(captured["status"], "passed")

    def test_translation_failure_does_not_reclassify_native_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps({
                "selected": [
                    {"id": "native_case", "profile": "native", "outcome": "passed", "reason": ""},
                    {"id": "translation_case", "profile": "translation", "outcome": "failed", "reason": "adapter error"},
                ],
                "exit_code": 1, "dependencies": {"openai": "2.9.0"},
            }))
            native = qualification.suite("native_responses", path, "native")
            translation = qualification.suite("translation", path, "translation")
        self.assertEqual(native["status"], "passed")
        self.assertEqual(translation["status"], "failed")

    def test_combined_pass_requires_attributed_native_and_translation_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "cases.json"
            selected = [
                {"id": profile, "profile": profile, "outcome": "passed", "reason": ""}
                for profile in ("native", "translation", "supporting")
            ]
            credentialed = directory / "credentialed.json"
            credentialed.write_text(json.dumps(result([("credentialed", "skipped", "no key")])))
            output = directory / "raw.json"
            args = SimpleNamespace(responses=path, credentialed=credentialed,
                                   setup="passed", output=output)
            with mock.patch.object(qualification, "command", return_value="available"), \
                 mock.patch.object(qualification, "model_revision", return_value=None):
                path.write_text(json.dumps({"selected": selected, "exit_code": 0,
                                            "dependencies": {"openai": "2.9.0"}}))
                qualification.capture(args)
                self.assertEqual(json.loads(output.read_text())["status"], "passed")
                selected.append({"id": "new fixture", "profile": "unclassified",
                                 "outcome": "passed", "reason": ""})
                path.write_text(json.dumps({"selected": selected, "exit_code": 0,
                                            "dependencies": {"openai": "2.9.0"}}))
                qualification.capture(args)
            self.assertEqual(json.loads(output.read_text())["status"], "incomplete")

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

        mislabeled = report()
        mislabeled["suites"]["native_responses"]["selected"][0]["profile"] = "translation"
        self.assertIn("native", qualification.verify(mislabeled, run, "a" * 40))
        native_skipped = report()
        native_skipped["suites"]["native_responses"]["status"] = "skipped"
        native_skipped["suites"]["native_responses"]["selected"][0]["outcome"] = "skipped"
        self.assertIn("native", qualification.verify(native_skipped, run, "a" * 40))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.json"
            path.write_text(json.dumps(native_skipped))
            finished = qualification.finish(path, True, "success", "success", "skipped", "skipped",
                                            17, 2, "https://example.test/run", "a" * 40)
        self.assertEqual(finished["status"], "incomplete")

    def test_missing_evidence_and_notes_idempotency(self):
        missing = qualification.unavailable("a" * 40, "artifact expired")
        section = qualification.render(missing)
        self.assertIn("unavailable for this release commit", section)
        self.assertNotIn("Native Responses:", section)
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

    def test_qualification_plugin_merges_lanes_and_fails_on_corrupt_results(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "qualification.json"
            session = SimpleNamespace(
                config=SimpleNamespace(_openai_sdk_version="2.9.0", _openai_sdk_lane="2.x")
            )
            with mock.patch.dict("os.environ", {"PRAXIS_QUALIFICATION_RESULTS": str(destination)}):
                plugin.pytest_sessionfinish(session, 0)
                data_lane1 = json.loads(destination.read_text(encoding="utf-8"))
                self.assertEqual(data_lane1["dependencies"]["openai"], "2.9.0")
                self.assertEqual(data_lane1["dependencies"]["openai_2.x"], "2.9.0")

                session_lane2 = SimpleNamespace(
                    config=SimpleNamespace(_openai_sdk_version="3.0.0", _openai_sdk_lane="3.x")
                )
                plugin.pytest_sessionfinish(session_lane2, 0)
                data_lane2 = json.loads(destination.read_text(encoding="utf-8"))
                self.assertEqual(data_lane2["dependencies"]["openai"], "2.9.0, 3.0.0")
                self.assertEqual(data_lane2["dependencies"]["openai_2.x"], "2.9.0")
                self.assertEqual(data_lane2["dependencies"]["openai_3.x"], "3.0.0")

                destination.write_text("invalid json {")
                with self.assertRaises(RuntimeError):
                    plugin.pytest_sessionfinish(session, 0)

    def test_qualification_plugin_merges_prerelease_and_release_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "qualification.json"
            rc_session = SimpleNamespace(
                config=SimpleNamespace(_openai_sdk_version="3.0.0rc1", _openai_sdk_lane="3.x")
            )
            rel_session = SimpleNamespace(
                config=SimpleNamespace(_openai_sdk_version="3.0.0", _openai_sdk_lane="3.x")
            )
            with mock.patch.dict("os.environ", {"PRAXIS_QUALIFICATION_RESULTS": str(destination)}):
                plugin.pytest_sessionfinish(rc_session, 0)
                plugin.pytest_sessionfinish(rel_session, 0)
                data = json.loads(destination.read_text(encoding="utf-8"))
                self.assertEqual(data["dependencies"]["openai"], "3.0.0rc1, 3.0.0")


if __name__ == "__main__":
    unittest.main()
