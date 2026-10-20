"""Exact-commit CI selection, truthful outcomes, and release CLI integration."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".github/scripts"))
import release_qualification as release
import vllm_qualification as gpu


REPO = "praxis-proxy/ai"
SHA = "a" * 40


def run(run_id=17, workflow="tests.yaml", **overrides):
    return {"id": run_id, "run_attempt": 2, "head_sha": SHA,
            "head_repository": {"full_name": REPO}, "path": f".github/workflows/{workflow}",
            "event": "push", "status": "completed", "conclusion": "success",
            "html_url": f"https://github.com/{REPO}/actions/runs/{run_id}", **overrides}


def job(name="unit tests", run_id=17, **overrides):
    return {"id": 123, "run_id": run_id, "run_attempt": 2, "head_sha": SHA,
            "name": name, "status": "completed", "conclusion": "success",
            "html_url": f"https://github.com/{REPO}/actions/runs/{run_id}/job/123",
            "steps": [{"name": "Run tests", "status": "completed", "conclusion": "success"},
                      {"name": "Optional provider test", "status": "completed", "conclusion": "skipped"}],
            **overrides}


def fixture_report():
    with mock.patch.object(release, "pages", side_effect=[
            [run()], [job(), job("optional", conclusion="skipped")], []]):
        tested = release.workflow_evidence(REPO, SHA, "tests.yaml", "Tests", "Rust unit tests and lint")
        missing = release.workflow_evidence(REPO, SHA, "anthropic-vllm-vision.yaml", "Vision GPU", "Anthropic SDK image requests")
    return {"release_sha": SHA, "created_at": "2026-10-20T10:00:00+00:00",
            "selection": "Newest exact-commit run and latest attempt; no older green fallback.",
            "limitations": ["Job counts are not test counts; skipped steps remain visible."],
            "workflows": [tested, missing], "gpu": gpu.unavailable(SHA, "no completed Responses GPU attempt for this commit")}


class ReleaseQualificationTest(unittest.TestCase):
    def evidence(self, runs=None, jobs=None, **kwargs):
        with mock.patch.object(release, "pages", side_effect=[
                [run()] if runs is None else runs, [job()] if jobs is None else jobs]) as pages:
            entry = release.workflow_evidence(REPO, SHA, "tests.yaml", "Tests", "Unit tests", **kwargs)
        return entry, pages

    def test_latest_run_and_attempt_win_even_when_failed_or_pending(self):
        for current in (run(18, conclusion="failure"), run(18, status="in_progress", conclusion=None)):
            with self.subTest(status=current["status"]):
                entry, pages = self.evidence([current, run()], [job(run_id=18)])
                self.assertEqual(entry["run"]["id"], 18)
                self.assertEqual(entry["status"], release.status(current))
                self.assertIn("/runs/18/attempts/2/jobs", pages.call_args.args[0])

    def test_pr_fork_other_workflow_and_other_commit_are_not_exact_evidence(self):
        for invalid in (run(event="pull_request"), run(event="pull_request_target"),
                        run(head_repository={"full_name": "fork/ai"}), run(head_repository=None),
                        run(head_sha="b" * 40), run(workflow="coverage.yaml")):
            with self.subTest(run=invalid):
                entry, pages = self.evidence([invalid])
                self.assertEqual(entry["status"], "not_run")
                self.assertEqual(pages.call_count, 1)

    def test_all_skipped_jobs_do_not_become_passing_execution(self):
        entry, _ = self.evidence(jobs=[job(conclusion="skipped")])
        self.assertEqual(entry["status"], "skipped")
        entry, _ = self.evidence(jobs=[job(), job("optional", conclusion="skipped")])
        self.assertEqual(entry["status"], "success")
        self.assertEqual(entry["job_totals"], {"success": 1, "skipped": 1})
        self.assertEqual(entry["jobs"][0]["steps"][1]["conclusion"], "skipped")

    def test_successful_workflow_cannot_hide_failed_or_pending_jobs(self):
        for changed, expected in ((job(conclusion="failure"), "failure"),
                                  (job(status="in_progress", conclusion=None), "incomplete"),
                                  (job(conclusion=None), "incomplete")):
            with self.subTest(expected=expected):
                entry, _ = self.evidence(jobs=[changed])
                self.assertEqual(entry["status"], expected)

    def test_missing_jobs_api_failure_and_job_provenance_remain_unavailable(self):
        for jobs in ([], [job(head_sha="b" * 40)], [job(run_id=18)], [job(run_attempt=1)]):
            with self.subTest(jobs=jobs):
                entry, _ = self.evidence(jobs=jobs)
                self.assertEqual(entry["status"], "unavailable")
        with mock.patch.object(release, "pages", side_effect=ValueError("GitHub workflow_runs lookup failed")):
            entry = release.workflow_evidence(REPO, SHA, "tests.yaml", "Tests", "Unit tests")
        self.assertEqual(entry["status"], "unavailable")
        self.assertIn("lookup failed", entry["reason"])

    def test_current_release_uses_its_test_job_while_publication_runs(self):
        current = run(21, "release.yaml", status="in_progress", conclusion=None)
        with mock.patch.object(release, "api", return_value=current) as api, \
                mock.patch.object(release, "pages", return_value=[
                    job("test", run_id=21), job("github-release", run_id=21, status="in_progress", conclusion=None)]):
            entry = release.workflow_evidence(REPO, SHA, "release.yaml", "Release gate", "Test steps", 21, 2)
        self.assertEqual(entry["status"], "success")
        self.assertEqual([item["name"] for item in entry["jobs"]], ["test"])
        api.assert_called_once_with(f"repos/{REPO}/actions/runs/21/attempts/2")
        with mock.patch.object(release, "api", return_value=current):
            wrong_attempt = release.workflow_evidence(REPO, SHA, "release.yaml", "Release gate", "Test steps", 21, 3)
        self.assertEqual(wrong_attempt["status"], "unavailable")

    def test_pagination_collects_later_results_and_never_returns_partial_success(self):
        first = {"total_count": 101, "workflow_runs": [run(i) for i in range(100)]}
        with mock.patch.object(release, "api", side_effect=[first, {"workflow_runs": [run(101)]}]) as api:
            runs = release.pages("runs?head_sha=" + SHA, "workflow_runs")
        self.assertEqual(len(runs), 101)
        self.assertIn("&per_page=100&page=2", api.call_args.args[0])
        with mock.patch.object(release, "api", side_effect=[first, None]):
            with self.assertRaisesRegex(ValueError, "lookup failed"):
                release.pages("runs", "workflow_runs")
        with mock.patch.object(release, "api", return_value={"total_count": 1001, "jobs": [job()] * 100}):
            with self.assertRaisesRegex(ValueError, "pagination limit"):
                release.pages("jobs", "jobs")

    def test_missing_gpu_does_not_erase_ci_and_report_notes_are_idempotent(self):
        report = fixture_report()
        section = release.render(report)
        old = gpu.compose_notes("## Changelog\n\n- existing change", gpu.render(report["gpu"]))
        once = gpu.compose_notes(old, section, release.START, release.END)
        twice = gpu.compose_notes(once, section, release.START, release.END)
        self.assertEqual(once, twice)
        self.assertEqual(twice.count(release.START), 1)
        self.assertEqual(twice.count(gpu.START), 1)
        self.assertIn("existing change", twice)
        self.assertIn("| Tests | Rust unit tests and lint | success | 1 skipped, 1 success |", section)
        self.assertIn("no completed Responses GPU attempt", section)
        self.assertIn("run_live_vllm=true", section)
        self.assertIn("Optional provider test: skipped", release.render(report, detailed=True))
        self.assertEqual(section, Path(__file__).with_name("release-ci-notes.fixture.md").read_text())

    def test_cli_writes_all_reports_and_preserves_release_notes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            existing = "## Changelog\n\n- existing change\n"
            (directory / "notes.md").write_text(existing)
            (directory / "gpu.json").write_text(json.dumps(gpu.unavailable(SHA, "no completed GPU attempt")))
            endpoints = {
                f"repos/{REPO}/actions/runs/21/attempts/2": run(21, "release.yaml", status="in_progress", conclusion=None),
                f"repos/{REPO}/actions/runs/21/attempts/2/jobs?per_page=100&page=1": {"jobs": [job("test", run_id=21)]},
                f"repos/{REPO}/actions/workflows/tests.yaml/runs?head_sha={SHA}&per_page=100&page=1": {"workflow_runs": [run()]},
                f"repos/{REPO}/actions/runs/17/attempts/2/jobs?per_page=100&page=1": {"jobs": [job()]},
            }
            (directory / "endpoints.json").write_text(json.dumps(endpoints))
            fake_gh = directory / "gh"
            fake_gh.write_text(
                "#!/usr/bin/env python3\nimport json, os, sys\nfrom pathlib import Path\n"
                "responses = json.loads(Path(os.environ['QUALIFICATION_TEST_ENDPOINTS']).read_text())\n"
                "print(json.dumps(responses.get(sys.argv[2], {'workflow_runs': []})))\n")
            fake_gh.chmod(0o755)
            argv = [sys.executable, str(ROOT / ".github/scripts/release_qualification.py"),
                    "--repo", REPO, "--sha", SHA, "--run-id", "21", "--attempt", "2",
                    "--gpu-report", str(directory / "gpu.json"), "--existing-notes", str(directory / "notes.md"),
                    "--output-dir", str(directory)]
            environment = {**os.environ, "PATH": str(directory) + os.pathsep + os.environ["PATH"],
                           "QUALIFICATION_TEST_ENDPOINTS": str(directory / "endpoints.json")}
            result = subprocess.run(argv, env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((directory / "release-qualification.json").read_text())
            self.assertEqual(report["release_sha"], SHA)
            self.assertEqual([entry["status"] for entry in report["workflows"][:3]], ["success", "success", "not_run"])
            self.assertEqual(report["gpu"]["status"], "unavailable")
            self.assertIn("Optional provider test: skipped", (directory / "release-qualification.md").read_text())
            notes = (directory / "release-notes.md").read_text()
            self.assertIn("existing change", notes)
            self.assertIn(release.START, notes)
            self.assertIn(gpu.START, notes)
            # A repeat publication must replace both marked sections.
            (directory / "notes.md").write_text(notes)
            repeated = subprocess.run(argv, env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            refreshed = (directory / "release-notes.md").read_text()
            self.assertEqual(refreshed.count(release.START), 1)
            self.assertEqual(refreshed.count(gpu.START), 1)
            # Wrong-commit GPU evidence must not produce a release report.
            (directory / "gpu.json").write_text(json.dumps(gpu.unavailable("b" * 40, "wrong commit")))
            rejected = subprocess.run(argv, env=environment, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("release commit", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
