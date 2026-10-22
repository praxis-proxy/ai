"""Local lint uses CI's toolchains instead of an arbitrary cargo on PATH."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MAKE = shutil.which("make")


class LintToolchainsTest(unittest.TestCase):
    def run_target(self, target, *, fail_nightly=False, channel="9.8.7"):
        with tempfile.TemporaryDirectory(prefix="lint-toolchains-") as directory:
            root = Path(directory)
            shutil.copyfile(ROOT / "Makefile", root / "Makefile")
            shutil.copyfile(ROOT / "Cargo.toml", root / "Cargo.toml")
            (root / "rust-toolchain.toml").write_text(f'[toolchain]\nchannel = "{channel}"\n')
            binary = root / "bin"
            binary.mkdir()
            log = root / "rustup.jsonl"
            rustup = binary / "rustup"
            rustup.write_text(f"#!{sys.executable}\n" + r'''
import json, os, sys
args = sys.argv[1:]
with open(os.environ["LINT_TEST_LOG"], "a") as output:
    output.write(json.dumps(args) + "\n")
if args[:3] == ["toolchain", "install", "nightly"] and os.environ.get("FAIL_NIGHTLY"):
    sys.exit(73)
if args[:2] == ["which", "--toolchain"]:
    print(os.path.join(os.environ["LINT_TEST_ROOT"], "toolchains", args[2], "bin", "cargo"))
    sys.exit(0)
if args[:1] == ["run"]:
    expected = os.path.join(os.environ["LINT_TEST_ROOT"], "toolchains", args[1], "bin")
    if os.environ["PATH"].split(os.pathsep)[0] != expected:
        sys.exit(96)  # cargo's rustc/clippy/rustfmt children must use the same toolchain.
if args[:1] == ["run"] and args[2:4] == ["cargo", "tree"]:
    print("allowed-a v1.0.0\nallowed-b v1.0.0")
''')
            rustup.chmod(0o700)
            cargo = binary / "cargo"
            cargo.write_text("#!/bin/sh\necho 'unexpected cargo from PATH' >&2\nexit 95\n")
            cargo.chmod(0o700)
            environment = {key: value for key, value in os.environ.items()
                           if key not in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL")}
            environment.update(PATH=f"{binary}{os.pathsep}{environment['PATH']}",
                               LINT_TEST_LOG=str(log), LINT_TEST_ROOT=str(root),
                               FAIL_NIGHTLY="1" if fail_nightly else "")
            result = subprocess.run([MAKE, target], cwd=root, env=environment,
                                    text=True, capture_output=True, timeout=30)
            calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            return result, calls

    def test_lint_refreshes_nightly_and_uses_the_repository_stable_pin(self):
        result, calls = self.run_target("lint-clippy")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls[0][:3], ["toolchain", "install", "9.8.7"])
        self.assertIn("clippy", calls[0])
        self.assertEqual(calls[1][:3], ["toolchain", "install", "nightly"])
        self.assertIn("rustfmt", calls[1])
        self.assertIn("--allow-downgrade", calls[1])
        commands = [call for call in calls if call[0] == "run"]
        for call in commands:
            self.assertEqual(call[:3], ["run", "nightly" if call[3] == "fmt" else "9.8.7", "cargo"])
        clippy = [call for call in commands if call[3] == "clippy"]
        self.assertEqual(len(clippy), 3, "default, opt-in and lean Clippy passes must remain")
        for call in clippy:
            self.assertEqual(call[-3:], ["--", "-D", "warnings"])
        self.assertIn(["run", "nightly", "cargo", "fmt", "--all", "--", "--check"], calls)

    def test_failed_nightly_refresh_stops_before_any_lint(self):
        result, calls = self.run_target("lint-clippy", fail_nightly=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([call[:3] for call in calls], [
            ["toolchain", "install", "9.8.7"], ["toolchain", "install", "nightly"],
        ])

    def test_fmt_refreshes_the_same_nightly_before_formatting(self):
        result, calls = self.run_target("fmt")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls[-1], ["run", "nightly", "cargo", "fmt", "--all"])
        self.assertIn(["toolchain", "install", "nightly", "--profile", "minimal",
                       "--component", "rustfmt", "--allow-downgrade", "--no-self-update"], calls)

    def test_missing_repository_pin_stops_before_installing_any_toolchain(self):
        result, calls = self.run_target("lint-clippy", channel="")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing channel in rust-toolchain.toml", result.stderr)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
