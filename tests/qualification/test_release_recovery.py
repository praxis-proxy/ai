"""Publication recovery must retain the original tag and successful gate evidence."""

import copy
import json
import os
import subprocess
import tempfile
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".github/scripts"))
import release_recovery as recovery

REPO = "praxis-proxy/ai"
SHA = "a" * 40
TAG = "v0.11.0"


def evidence():
    run = {"id": 17, "run_attempt": 2, "head_sha": SHA,
           "head_repository": {"full_name": REPO}, "event": "push",
           "path": ".github/workflows/release.yaml", "head_branch": TAG,
           "status": "completed", "conclusion": "failure"}
    jobs = [{"run_id": 17, "run_attempt": 2, "head_sha": SHA, "name": name,
             "status": "completed", "conclusion": "success"} for name in recovery.GATES]
    jobs.append({**jobs[0], "name": "github-release", "conclusion": "failure"})
    return run, jobs


class ReleaseRecoveryTest(unittest.TestCase):
    def test_publication_failure_can_reuse_successful_gates(self):
        run, jobs = evidence()
        recovery.verify_run(REPO, TAG, SHA, run, jobs)

    def test_wrong_source_workflow_tag_and_pending_runs_are_rejected(self):
        for changed in ({"head_sha": "b" * 40}, {"head_repository": {"full_name": "fork/ai"}},
                        {"path": ".github/workflows/tests.yaml"}, {"head_branch": "main"},
                        {"status": "in_progress"}, {"event": "pull_request"}, {"event": "schedule"}):
            with self.subTest(changed=changed):
                run, jobs = evidence()
                run.update(changed)
                with self.assertRaises(ValueError):
                    recovery.verify_run(REPO, TAG, SHA, run, jobs)

    def test_missing_failed_skipped_pending_and_duplicate_gates_are_rejected(self):
        for name in recovery.GATES:
            for outcome in ("missing", "failure", "skipped", "in_progress", "duplicate"):
                with self.subTest(name=name, outcome=outcome):
                    run, jobs = evidence()
                    target = next(job for job in jobs if job["name"] == name)
                    if outcome == "missing":
                        jobs.remove(target)
                    elif outcome == "duplicate":
                        jobs.append(copy.deepcopy(target))
                    elif outcome == "in_progress":
                        target.update(status=outcome, conclusion=None)
                    else:
                        target["conclusion"] = outcome
                    with self.assertRaises(ValueError):
                        recovery.verify_run(REPO, TAG, SHA, run, jobs)

    def test_wrong_job_provenance_and_failed_extra_jobs_are_rejected(self):
        for changed in ({"head_sha": "b" * 40}, {"run_attempt": 1}, {"run_id": 18}):
            with self.subTest(changed=changed):
                run, jobs = evidence()
                jobs[0].update(changed)
                with self.assertRaises(ValueError):
                    recovery.verify_run(REPO, TAG, SHA, run, jobs)
        run, jobs = evidence()
        jobs.append({**jobs[0], "name": "fips-host", "conclusion": "failure"})
        with self.assertRaises(ValueError):
            recovery.verify_run(REPO, TAG, SHA, run, jobs)

    def test_cli_checks_tagged_version_and_emits_original_attempt(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "Cargo.toml").write_text('[workspace.package]\nversion = "0.11.0"\n')
            subprocess.run(["git", "init", "-q"], cwd=directory, check=True)
            subprocess.run(["git", "add", "Cargo.toml"], cwd=directory, check=True)
            subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.test",
                            "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"], cwd=directory, check=True)
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()
            run, jobs = evidence()
            run["head_sha"] = sha
            for job in jobs:
                job["head_sha"] = sha
            fixture = directory / "responses.json"
            fixture.write_text(json.dumps({"run": run, "jobs": jobs}))
            fake = directory / "gh"
            fake.write_text("#!/usr/bin/env python3\nimport json, os, sys\n"
                            "data = json.load(open(os.environ['RECOVERY_FIXTURE']))\n"
                            "print(json.dumps({'jobs': data['jobs']} if '/jobs?' in sys.argv[2] else data['run']))\n")
            fake.chmod(0o755)
            environment = {**os.environ, "PATH": str(directory) + os.pathsep + os.environ["PATH"],
                           "RECOVERY_FIXTURE": str(fixture)}
            command = [sys.executable, str(root / ".github/scripts/release_recovery.py"),
                       "--repo", REPO, "--tag", TAG, "--run-id", "17"]
            accepted = subprocess.run(command, cwd=directory, env=environment, text=True, capture_output=True)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertEqual(accepted.stdout, "attempt=2\n")
            (directory / "Cargo.toml").write_text('[workspace.package]\nversion = "0.12.0"\n')
            rejected = subprocess.run(command, cwd=directory, env=environment, text=True, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("workspace version", rejected.stderr)

    def test_invalid_release_versions_are_rejected(self):
        for tag in ("main", "v0.11.0-rc.1", "v0.011.0", "v0.11.0\n"):
            with self.subTest(tag=tag):
                run, jobs = evidence()
                with self.assertRaises(ValueError):
                    recovery.verify_run(REPO, tag, SHA, run, jobs)


if __name__ == "__main__":
    unittest.main()
