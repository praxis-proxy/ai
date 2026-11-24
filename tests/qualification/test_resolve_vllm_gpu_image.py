"""Decision paths of the resolve-vllm-gpu-image action.

Runs the action's own shell script against a fake container runtime, so the
paths an acceptance-only dispatch depends on (reuse what an earlier job on the
runner prepared, pull a matching published image, or fall back to a build) are
exercised without a registry or GPU.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACTION = ROOT / ".github/actions/resolve-vllm-gpu-image/action.yml"
IMAGE = "ghcr.io/praxis-proxy/vllm-gpu:Qwen3-8B"
MODEL = "Qwen/Qwen3-8B"

# Stands in for docker/podman. State lives in a JSON file: the image under the
# tag locally and in the registry (each a label map, or null when absent), plus
# a log of the subcommands it was asked to run.
FAKE_RUNTIME = """\
#!/usr/bin/env python3
import json, os, re, sys
path = os.environ["FAKE_RUNTIME_STATE"]
state = json.load(open(path))
args = sys.argv[1:]
state["calls"].append(args[0])
def save():
    json.dump(state, open(path, "w"))
if args[:2] == ["image", "inspect"]:
    save()
    local = state["local"]
    if local is None:
        sys.exit(1)
    fmt = next((a.split("=", 1)[1] for a in args if a.startswith("--format=")), None)
    if fmt is None and "--format" in args:
        fmt = args[args.index("--format") + 1]
    match = re.search(r'Labels "([^"]+)"', fmt or "")
    if match:
        print(local.get(match.group(1), "<no value>"))
    elif "RepoDigests" in (fmt or ""):
        print("ghcr.io/praxis-proxy/vllm-gpu@sha256:" + "d" * 64)
    sys.exit(0)
if args[0] == "login":
    save()
    assert args[1:] == ["ghcr.io", "--username", "tester", "--password-stdin"]
    assert sys.stdin.read().strip() == "test-token"
    sys.exit(0 if state["login_ok"] else 1)
if args[0] == "pull":
    state["local"] = state["registry"]
    save()
    sys.exit(0 if state["registry"] is not None else 1)
if args[0] == "rmi":
    state["local"] = None
    save()
    sys.exit(0)
save()
sys.exit(2)
"""


def action_script():
    """The `run:` block of the action's single step, dedented."""
    lines = ACTION.read_text(encoding="utf-8").splitlines()
    start = lines.index("      run: |") + 1
    body = []
    for line in lines[start:]:
        if line and not line.startswith("        "):
            break
        body.append(line)
    return textwrap.dedent("\n".join(body)) + "\n"


def labels(**overrides):
    containerfile = hashlib.sha256((ROOT / "vllm/Containerfile").read_bytes()).hexdigest()
    value = {
        "praxis.vllm.containerfile-sha256": containerfile,
        "praxis.vllm.inference-model": MODEL,
        "praxis.vllm.inference-revision": "",
    }
    value.update(overrides)
    return value


class ResolveVllmGpuImageTest(unittest.TestCase):
    def resolve(self, local, registry, reuse_local, *, runtime_name="docker", auth=False, login_ok=True):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            runtime = directory / runtime_name
            runtime.write_text(FAKE_RUNTIME, encoding="utf-8")
            runtime.chmod(0o755)
            state = directory / "state.json"
            state.write_text(json.dumps({"local": local, "registry": registry, "calls": [], "login_ok": login_ok}))
            output = directory / "output"
            output.touch()
            env = dict(os.environ, IMAGE=IMAGE, EXPECTED_MODEL=MODEL, EXPECTED_REVISION="",
                       GHCR_USERNAME="tester" if auth else "", GHCR_TOKEN="test-token" if auth else "",
                       CONTAINER_RUNTIME=str(runtime), REUSE_LOCAL=reuse_local,
                       GITHUB_OUTPUT=str(output), FAKE_RUNTIME_STATE=str(state),
                       PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}")
            subprocess.run(["bash", "-c", action_script()], cwd=ROOT, env=env, check=True,
                           capture_output=True, text=True)
            outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
            final = json.loads(state.read_text())
        return outputs, final

    def test_reuses_a_matching_image_an_earlier_job_prepared(self):
        # An earlier job built the image here; the registry still holds a stale
        # tag that a pull would put back in its place.
        outputs, state = self.resolve(labels(), labels(**{"praxis.vllm.inference-model": "old"}), "true")
        self.assertEqual(outputs, {"build-required": "false", "digest": "", "source": "runner"})
        self.assertNotIn("pull", state["calls"])
        self.assertEqual(state["local"], labels())

    def test_registry_login_only_after_local_miss(self):
        for runtime in ("docker", "podman"):
            with self.subTest(runtime=runtime):
                outputs, state = self.resolve(labels(), None, "true", runtime_name=runtime, auth=True, login_ok=False)
                self.assertEqual(outputs["source"], "runner")
                self.assertNotIn("login", state["calls"])
                outputs, state = self.resolve(None, labels(), "true", runtime_name=runtime, auth=True, login_ok=False)
                self.assertEqual(outputs, {"build-required": "true", "digest": "", "source": "build"})
                self.assertIn("login", state["calls"])
                self.assertNotIn("pull", state["calls"])
                outputs, state = self.resolve(None, labels(), "true", runtime_name=runtime, auth=True)
                self.assertEqual(outputs["source"], "registry")
                self.assertLess(state["calls"].index("login"), state["calls"].index("pull"))

    def test_pulls_when_the_local_image_is_stale(self):
        outputs, state = self.resolve(labels(**{"praxis.vllm.containerfile-sha256": "0" * 64}), labels(), "true")
        self.assertEqual(outputs["source"], "registry")
        self.assertEqual(outputs["build-required"], "false")
        self.assertTrue(outputs["digest"].endswith("d" * 64))
        self.assertIn("pull", state["calls"])

    def test_pulls_without_reuse_local_even_when_present(self):
        outputs, state = self.resolve(labels(), labels(), "false")
        self.assertEqual(outputs["source"], "registry")
        self.assertIn("pull", state["calls"])

    def test_requires_a_build_when_the_published_image_differs(self):
        outputs, state = self.resolve(None, labels(**{"praxis.vllm.inference-revision": "abc"}), "true")
        self.assertEqual(outputs, {"build-required": "true", "digest": "", "source": "build"})
        # The stale pull is removed so it cannot shadow the build.
        self.assertIsNone(state["local"])

    def test_requires_a_build_when_nothing_is_published(self):
        outputs, _ = self.resolve(None, None, "true")
        self.assertEqual(outputs, {"build-required": "true", "digest": "", "source": "build"})


if __name__ == "__main__":
    unittest.main()
