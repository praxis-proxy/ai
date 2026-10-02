"""Local checks for safe EC2 runner startup diagnostics."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/diagnose_ec2_runner.sh"


class RunnerDiagnosticsTests(unittest.TestCase):
    def run_diagnostics(self, **overrides: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary:
            bin_dir = Path(temporary) / "bin"
            bin_dir.mkdir()
            aws = bin_dir / "aws"
            aws.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    if [[ "$2" == describe-instance-status ]]; then
                      echo '[{"State":"running","SystemCheck":"ok","InstanceCheck":"ok"}]'
                    elif [[ "$2" == get-console-output ]]; then
                      if [[ "${MOCK_CONSOLE_DENIED:-}" == 1 ]]; then
                        echo 'UnauthorizedOperation' >&2
                        exit 1
                      fi
                      printf '%s\\n' "$MOCK_CONSOLE"
                    fi
                    """
                )
            )
            aws.chmod(0o755)
            gh = bin_dir / "gh"
            gh.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    if [[ "${MOCK_GH_DENIED:-}" == 1 ]]; then
                      echo 'gh: forbidden (HTTP 403)' >&2
                      exit 1
                    fi
                    printf '%s\\n' "$MOCK_RUNNERS"
                    """
                )
            )
            gh.chmod(0o755)
            environment = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "RUNNER_TEMP": temporary,
                "INSTANCE_ID": "i-test",
                "RUNNER_LABEL": "test-runner",
                "GITHUB_REPOSITORY": "praxis-proxy/ai",
                "GH_TOKEN": "pat-secret",
                "MOCK_CONSOLE": "None",
                "MOCK_RUNNERS": json.dumps({"total_count": 0, "runners": []}),
                **overrides,
            }
            return subprocess.run(
                ["bash", str(SCRIPT)],
                capture_output=True,
                check=False,
                env=environment,
                text=True,
            )

    def test_boot_markers_do_not_publish_raw_console_or_token(self) -> None:
        result = self.run_diagnostics(
            MOCK_CONSOLE=(
                "Cloud-init v. 24 finished\n"
                "curl: (6) Could not resolve host: github.com\n"
                "ERROR: Your runner version is out of date and can no longer register with GitHub.\n"
                "config.sh --token registration-token-secret"
            ),
            MOCK_RUNNERS=json.dumps(
                {
                    "total_count": 1,
                    "runners": [
                        {
                            "status": "offline",
                            "labels": [{"name": "test-runner"}],
                        }
                    ],
                }
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cloud-init completed: yes", result.stdout)
        self.assertIn("network or download error: yes", result.stdout)
        self.assertIn("runner version rejected: yes", result.stdout)
        self.assertIn("Runner label is present in GitHub with status: offline", result.stdout)
        self.assertNotIn("registration-token-secret", result.stdout + result.stderr)

    def test_denied_console_and_runner_api_are_reported_separately(self) -> None:
        result = self.run_diagnostics(MOCK_CONSOLE_DENIED="1", MOCK_GH_DENIED="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("grant ec2:GetConsoleOutput", result.stdout)
        self.assertIn("Runner PAT could not list GitHub runners (HTTP 403)", result.stdout)
        self.assertNotIn("pat-secret", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
