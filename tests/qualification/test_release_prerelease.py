"""Pre-1.0 releases must be marked as GitHub pre-releases."""

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/release-prerelease.sh"


def prerelease(version):
    return subprocess.run(
        ["bash", str(SCRIPT), version],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


class ReleasePrereleaseTest(unittest.TestCase):
    def test_pre_1_0_versions_are_prereleases(self):
        for version in ("0.0.1", "0.1.0", "0.10.0", "v0.10.0"):
            with self.subTest(version=version):
                self.assertEqual(prerelease(version), "true")

    def test_1_0_and_later_are_stable(self):
        for version in ("1.0.0", "1.2.3", "2.0.0", "10.0.0", "v1.0.0"):
            with self.subTest(version=version):
                self.assertEqual(prerelease(version), "false")


if __name__ == "__main__":
    unittest.main()
