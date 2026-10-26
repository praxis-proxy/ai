#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Praxis Contributors

"""Controlled lifecycle tests: no containers or registry credentials required."""

import os
from pathlib import Path
import selectors
import signal
import subprocess
import tempfile
import unittest


FIXTURE = Path(__file__).with_name("start-sentinel-topology.sh")
# Test-only process budget; production discovery/operation deadlines do not
# change when an overloaded developer host needs more shell-launch headroom.
TEST_TIMEOUT = float(os.environ.get("TOKEN_RATE_LIMIT_FIXTURE_TEST_TIMEOUT", "30"))
FAKE_ENGINE = """#!/usr/bin/env bash
set -eu
case "$1" in
  rm) printf '%s\\n' "$*" >> "$FIXTURE_ENGINE_LOG" ;;
  exec)
    [[ "$2" != -i ]] || IFS= read -r password
    case "$*" in
      *get-master-addr-by-name*) printf '127.0.0.1\\n26379\\n' ;;
      *'INFO replication'*)
        offset=42
        [[ "$*" != *'-p 26380 '* ]] || offset="$FIXTURE_REPLICA_OFFSET"
        printf 'master_repl_offset:%s\\r\\nmaster_link_status:up\\r\\n' "$offset"
        ;;
      *'INFO server'*) printf 'redis_version:8.10.2\\r\\n' ;;
    esac
    ;;
esac
"""


class SentinelFixtureLifecycle(unittest.TestCase):
    def run_fixture(self, stop_signal=None, replica_offset=42):
        with tempfile.TemporaryDirectory(prefix="sentinel-lifecycle-") as directory:
            root = Path(directory)
            engine = root / "fake-engine"
            engine.write_text(FAKE_ENGINE)
            engine.chmod(0o700)
            log = root / "engine.log"
            env_file = root / "fixture.env"
            env = os.environ.copy()
            env.update(
                CONTAINER_ENGINE=str(engine), FIXTURE_ENGINE_LOG=str(log),
                FIXTURE_REPLICA_OFFSET=str(replica_offset),
            )
            args = [
                "bash", str(FIXTURE), "--product", "redis", "--image",
                "controlled-fixture@sha256:0000", "--env-file", str(env_file),
            ]
            if stop_signal is None:
                args += ["--", "sh", "-c", "exit 7"]
            process = subprocess.Popen(
                args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
            )
            try:
                if stop_signal is not None:
                    with selectors.DefaultSelector() as selector:
                        selector.register(process.stdout, selectors.EVENT_READ)
                        ready = b""
                        while b"Sentinel topology ready:" not in ready:
                            self.assertTrue(
                                selector.select(timeout=TEST_TIMEOUT),
                                f"fixture must become ready within {TEST_TIMEOUT}s",
                            )
                            # Read the descriptor directly: TextIO.readline can
                            # buffer the readiness line while select sees no
                            # remaining descriptor bytes and waits forever.
                            chunk = os.read(process.stdout.fileno(), 4096)
                            self.assertTrue(chunk, "fixture must stay alive until signalled")
                            ready += chunk
                    self.assertTrue(env_file.exists(), "ready fixture must retain its environment file")
                    process.send_signal(stop_signal)
                _, stderr = process.communicate(timeout=TEST_TIMEOUT)
            finally:
                # Kill only this test's process group if a failing fixture left
                # a child holding its output pipes open.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
            expected = 7 if stop_signal is None else 128 + stop_signal
            self.assertEqual(process.returncode, expected, f"fixture exit status must propagate: {stderr}")
            removed = log.read_text().splitlines()
            self.assertEqual(len(removed), 5, "cleanup must remove each container exactly once")
            self.assertEqual(len(set(removed)), 5, "cleanup must not repeat container removal")
            self.assertFalse(env_file.exists(), "cleanup must remove the environment file")

    def test_term_exits_and_cleans_once(self):
        self.run_fixture(signal.SIGTERM)

    def test_int_exits_and_cleans_once(self):
        self.run_fixture(signal.SIGINT)

    def test_child_failure_exits_and_cleans_once(self):
        self.run_fixture()

    def test_replica_ahead_of_primary_snapshot_is_ready(self):
        self.run_fixture(replica_offset=43)

    def test_cleanup_is_owned_by_parent_and_is_idempotent(self):
        source = FIXTURE.read_text()
        start = source.index("cleanup() {")
        cleanup = source[start:source.index("\n}\n", start) + 2]
        with tempfile.TemporaryDirectory(prefix="sentinel-cleanup-owner-") as directory:
            root = Path(directory)
            engine = root / "engine"
            engine.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$FIXTURE_ENGINE_LOG"\n')
            engine.chmod(0o700)
            data = root / "data"
            data.mkdir()
            env_file = root / "fixture.env"
            env_file.touch()
            log = root / "engine.log"
            environment = os.environ.copy()
            environment.update(
                FIXTURE_ENGINE=str(engine), FIXTURE_ENGINE_LOG=str(log),
                FIXTURE_DATA=str(data), FIXTURE_ENV_FILE=str(env_file),
            )
            program = (
                'set -Eeuo pipefail\nfixture_owner_pid=${BASHPID:-$$}\ncleanup_started=0\n'
                'engine="$FIXTURE_ENGINE"\nprefix=owned-fixture\n'
                'tmp_dir="$FIXTURE_DATA"\nenv_file="$FIXTURE_ENV_FILE"\nenv_file_written=1\n'
                + cleanup + '\n(cleanup)\n'
                '[[ -d "$tmp_dir" && -e "$env_file" ]] || exit 12\ncleanup\ncleanup\n'
            )
            subprocess.run(["bash", "-c", program], env=environment, check=True, capture_output=True)
            self.assertEqual(len(log.read_text().splitlines()), 5, "only the parent may clean, exactly once")
            self.assertFalse(data.exists(), "parent cleanup must remove its data directory")
            self.assertFalse(env_file.exists(), "parent cleanup must remove its environment file")

    def test_standalone_contracts_require_fixture_credentials(self):
        environment = {"PATH": os.environ["PATH"]}
        result = subprocess.run(
            ["bash", str(FIXTURE.with_name("run-standalone-contracts.sh"))],
            env=environment, capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0, "missing fixture inputs must fail, not skip")
        self.assertIn("run inside the pinned topology fixture", result.stderr)

    def test_standalone_contracts_encode_credentials_and_select_existing_targets(self):
        with tempfile.TemporaryDirectory(prefix="standalone-contracts-") as directory:
            root = Path(directory)
            make = root / "make"
            make.write_text(
                '#!/usr/bin/env bash\nset -eu\n'
                '[[ -z "${TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS+x}" ]]\n'
                'printf "%s\\t%s\\t%s\\n" "$*" "$RUST_TEST_THREADS" '
                '"$TOKEN_RATE_LIMIT_VALKEY_URL" >> "$FIXTURE_CONTRACT_LOG"\n'
            )
            make.chmod(0o700)
            engine = root / "engine"
            engine.write_text('#!/usr/bin/env bash\nset -eu\nprintf "%s\\n" "$*" >> "$FIXTURE_STOP_LOG"\n')
            engine.chmod(0o700)
            log = root / "contracts.log"
            stops = root / "stops.log"
            environment = {
                "PATH": f"{root}{os.pathsep}{os.environ['PATH']}",
                "TOKEN_RATE_LIMIT_SENTINEL_PRIMARY_PORT": "26379",
                "TOKEN_RATE_LIMIT_SENTINEL_DATA_PASSWORD": "not:real@secret",
                "TOKEN_RATE_LIMIT_SENTINEL_ENGINE": str(engine),
                "TOKEN_RATE_LIMIT_SENTINEL_CONTAINER_PREFIX": "owned-fixture",
                "TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS": "redis://127.0.0.1:26381",
                "FIXTURE_CONTRACT_LOG": str(log),
                "FIXTURE_STOP_LOG": str(stops),
            }
            subprocess.run(
                ["bash", str(FIXTURE.with_name("run-standalone-contracts.sh"))],
                env=environment, check=True, capture_output=True, text=True,
            )
            calls = [line.split("\t") for line in log.read_text().splitlines()]
            self.assertEqual(
                stops.read_text().splitlines(),
                [f"stop owned-fixture-sentinel-{index}" for index in range(1, 4)],
                "standalone isolation must stop only this fixture's Sentinels",
            )
            self.assertEqual(
                [call[0] for call in calls],
                ["test-token-rate-limit-valkey-unit", "test-token-rate-limit-valkey-integration"],
                "the pinned lane must run both existing standalone contract tiers",
            )
            self.assertTrue(all(call[1] == "1" for call in calls), "live contract tests must run serially")
            self.assertTrue(
                all(call[2] == "redis://:not%3Areal%40secret@127.0.0.1:26379" for call in calls),
                "standalone URLs must percent-encode fixture credentials",
            )


if __name__ == "__main__":
    unittest.main()
