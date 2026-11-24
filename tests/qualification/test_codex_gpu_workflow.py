# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github/workflows/vllm-integration.yaml").read_text()
CODEX_JOB = "vllm-gpu-codex-acceptance"
CODEX_PIN = tomllib.loads((ROOT / "tests/integration/fixtures/codex-cli/pin.toml").read_text())["vllm"]


def step_script(job, step):
    job_body = re.split(r"\n  [a-z][a-z0-9-]*:\n", WORKFLOW.split(f"\n  {job}:\n", 1)[1], maxsplit=1)[0]
    body = job_body.split(f"      - name: {step}\n", 1)[1].split("\n      - ", 1)[0]
    return "\n".join(line.removeprefix("          ") for line in body.split("        run: |\n", 1)[1].splitlines())


class CodexGpuDispatchTest(unittest.TestCase):
    def validate(self, **overrides):
        env = {
            **os.environ,
            "CODEX_MODEL": "openai/gpt-oss-120b",
            "CODEX_VLLM_MODEL": "gpt-oss-120b",
            "CODEX_TENSOR_PARALLEL_SIZE": "1",
            "CODEX_MAX_MODEL_LEN": "16384",
            "RUN_CODEX": "true",
            "RUN_LIVE": "false",
            "RUN_CLAUDE": "false",
            "RUN_OPENCODE": "false",
            "RUN_PI": "false",
            "RUN_DEGRADATION": "false",
            "RUN_COMPACTION_ONLY": "false",
            **overrides,
        }
        return subprocess.run(
            ["bash", "-c", step_script("gpu-start-runner", "Validate Codex dispatch")],
            env=env, capture_output=True, text=True, check=False,
        )

    def test_candidate_checkpoints_can_dispatch_without_other_gpu_suites(self):
        for model, alias, tp in (("openai/gpt-oss-120b", "gpt-oss-120b", "1"),
                                 ("google/gemma-4-31B-it", "gemma-4", "4"),
                                 ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "nemotron-3", "4")):
            with self.subTest(model=model):
                result = self.validate(CODEX_MODEL=model, CODEX_VLLM_MODEL=alias, CODEX_TENSOR_PARALLEL_SIZE=tp)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_legacy_qwen_dispatch_still_accepts_full_suite(self):
        result = self.validate(CODEX_MODEL="", CODEX_VLLM_MODEL="", RUN_LIVE="true")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_candidate_rejects_other_gpu_suites_and_missing_codex_flag(self):
        for flag in ("RUN_LIVE", "RUN_CLAUDE", "RUN_OPENCODE", "RUN_PI", "RUN_DEGRADATION", "RUN_COMPACTION_ONLY"):
            with self.subTest(flag=flag):
                self.assertNotEqual(self.validate(**{flag: "true"}).returncode, 0)
        self.assertNotEqual(self.validate(RUN_CODEX="false").returncode, 0)

    def test_candidate_rejects_invalid_alias_and_context_length(self):
        for overrides in ({"CODEX_VLLM_MODEL": ""}, {"CODEX_VLLM_MODEL": "org/checkpoint"},
                          {"CODEX_MAX_MODEL_LEN": "16000"}, {"CODEX_TENSOR_PARALLEL_SIZE": "0"}):
            with self.subTest(overrides=overrides):
                self.assertNotEqual(self.validate(**overrides).returncode, 0)

    def test_candidate_vllm_argv_uses_dispatch_settings(self):
        for runtime, device in (("docker", ["--gpus", "all"]), ("podman", ["--device", "nvidia.com/gpu=all"])):
            with self.subTest(runtime=runtime), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                executable = directory / runtime
                executable.write_text('#!/bin/bash\nprintf "%s\\0" "$@" >> "$ARGV_CAPTURE"\n')
                executable.chmod(0o755)
                capture = directory / "argv"
                env = {
                    **os.environ,
                    "PATH": f"{directory}:{os.environ['PATH']}",
                    "ARGV_CAPTURE": str(capture),
                    "GITHUB_ENV": str(directory / "github-env"),
                    "VLLM_SERVED_IMAGE": "",
                    "CONTAINER_RUNTIME": runtime,
                    "GPU_HF_CACHE_DIR": str(directory / "hf"),
                    "CODEX_VLLM_BASE_IMAGE": "vllm-test-image",
                    "CODEX_VLLM_FLAGS": " ".join(CODEX_PIN["launch"]["flags"]),
                    "CODEX_CANDIDATE": "true",
                    "CODEX_MODEL": "openai/gpt-oss-120b",
                    "CODEX_MODEL_REVISION": "a" * 40,
                    "CODEX_VLLM_MODEL": "gpt-oss-120b",
                    "CODEX_BACKEND_TOKEN": "synthetic-credential",
                    "CODEX_MAX_MODEL_LEN": "24576",
                    "CODEX_TENSOR_PARALLEL_SIZE": "4",
                    "CODEX_TOOL_CALL_PARSER": "openai",
                    "CODEX_REASONING_PARSER": "openai_gptoss",
                }
                script = step_script(CODEX_JOB, "Start keyed vLLM for Codex")
                subprocess.run(["bash", "-c", script], env=env, check=True)
                argv = capture.read_text().split("\0")[:-1]
                self.assertEqual(argv[:6], ["pull", "vllm-test-image", "run", "-d", "--name", "vllm"])
                self.assertEqual(argv[6:9], [*device, "--ipc=host"])
                self.assertIn("--pull=never", argv)
                mount = f"{directory}/hf:/root/.cache/huggingface" + (":Z" if runtime == "podman" else "")
                self.assertEqual(argv[argv.index("-v") + 1], mount)
                self.assertEqual(argv[argv.index("-e") + 1], "HUGGING_FACE_HUB_TOKEN")
                self.assertEqual(argv[argv.index("--model") - 1], "vllm-test-image")
                self.assertEqual(argv[argv.index("--model"):], [
                    "--model", "openai/gpt-oss-120b", "--max-model-len", "24576",
                    "--enable-auto-tool-choice", "--tool-call-parser", "openai",
                    "--reasoning-parser", "openai_gptoss", "--gpu-memory-utilization", "0.9", "--seed", "0",
                    "--revision", "a" * 40, "--tokenizer-revision", "a" * 40, "--tensor-parallel-size", "4",
                    "--served-model-name", "gpt-oss-120b", "--api-key", "synthetic-credential",
                ])

                # check-pins accepts the equals form for these options as well.
                for option in ("--model", "--max-model-len", "--tool-call-parser", "--reasoning-parser"):
                    env["CODEX_VLLM_FLAGS"] = env["CODEX_VLLM_FLAGS"].replace(f"{option} ", f"{option}=")
                capture.unlink()
                subprocess.run(["bash", "-c", script], env=env, check=True)
                self.assertEqual(capture.read_text().split("\0")[:-1], argv)

                capture.unlink()
                env.update(CODEX_TENSOR_PARALLEL_SIZE="1", CODEX_TOOL_CALL_PARSER="none",
                           CODEX_REASONING_PARSER="none")
                subprocess.run(["bash", "-c", script], env=env, check=True)
                single_gpu_argv = capture.read_text().split("\0")
                for absent in ("--ipc=host", "--tensor-parallel-size", "--enable-auto-tool-choice",
                               "--tool-call-parser", "--reasoning-parser"):
                    self.assertNotIn(absent, single_gpu_argv)

    def test_baked_codex_launch_preserves_pins_and_dispatch_overrides(self):
        flags = CODEX_PIN["launch"]["flags"]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            runtime = directory / "docker"
            runtime.write_text('#!/bin/bash\nprintf "%s\\0" "$@" >> "$ARGV_CAPTURE"\n')
            runtime.chmod(0o755)
            capture = directory / "argv"
            env = {
                **os.environ,
                "PATH": f"{directory}:{os.environ['PATH']}",
                "ARGV_CAPTURE": str(capture),
                "GITHUB_ENV": str(directory / "github-env"),
                "CONTAINER_RUNTIME": "docker",
                "CODEX_CANDIDATE": "false",
                "CODEX_VLLM_FLAGS": " ".join(flags),
                "CODEX_VLLM_MODEL": CODEX_PIN["served_model_name"],
                "CODEX_BACKEND_TOKEN": "synthetic-credential",
                "VLLM_SERVED_IMAGE": "test-baked-image",
                "CODEX_MAX_MODEL_LEN": "",
                "CODEX_TOOL_CALL_PARSER": "",
                "CODEX_REASONING_PARSER": "",
                "CODEX_TENSOR_PARALLEL_SIZE": "1",
            }
            script = step_script(CODEX_JOB, "Start keyed vLLM for Codex")
            for overrides in (False, True):
                with self.subTest(overrides=overrides):
                    if overrides:
                        env.update(CODEX_MAX_MODEL_LEN="24576", CODEX_TOOL_CALL_PARSER="none",
                                   CODEX_REASONING_PARSER="openai_gptoss", CODEX_TENSOR_PARALLEL_SIZE="4",
                                   CODEX_VLLM_MODEL="custom-qwen")
                    subprocess.run(["bash", "-c", script], env=env, check=True)
                    argv = capture.read_text().split("\0")[:-1]
                    capture.unlink()
                    self.assertEqual(argv[0], "run")
                    self.assertIn("--pull=never", argv)
                    self.assertNotIn("-v", argv)
                    self.assertNotIn("-e", argv)
                    args = argv[argv.index("test-baked-image") + 1:]
                    effective = env.copy()
                    github_env = Path(env["GITHUB_ENV"])
                    if github_env.exists():
                        effective.update(line.split("=", 1) for line in github_env.read_text().splitlines())
                        github_env.unlink()
                    for variable, option in (("CODEX_MAX_MODEL_LEN", "--max-model-len"),
                                             ("CODEX_TOOL_CALL_PARSER", "--tool-call-parser"),
                                             ("CODEX_REASONING_PARSER", "--reasoning-parser")):
                        expected = args[args.index(option) + 1] if option in args else "none"
                        self.assertEqual(effective[variable], expected)
                    self.assertEqual(args[args.index("--model") + 1], flags[flags.index("--model") + 1])
                    self.assertEqual(args[args.index("--served-model-name") + 1], env["CODEX_VLLM_MODEL"])
                    self.assertEqual(args[args.index("--api-key") + 1], env["CODEX_BACKEND_TOKEN"])
                    if overrides:
                        self.assertIn("--ipc=host", argv)
                        for option, expected in (("--tensor-parallel-size", "4"), ("--max-model-len", "24576"),
                                                 ("--reasoning-parser", "openai_gptoss")):
                            self.assertEqual(args.count(option), 1)
                            self.assertEqual(args[args.index(option) + 1], expected)
                        self.assertNotIn("--enable-auto-tool-choice", args)
                        self.assertNotIn("--tool-call-parser", args)
                    else:
                        for index, flag in enumerate(flags):
                            if flag.startswith("--"):
                                self.assertEqual(args.count(flag), 1)
                                if index + 1 < len(flags) and not flags[index + 1].startswith("--"):
                                    self.assertEqual(args[args.index(flag) + 1], flags[index + 1])

    def test_codex_preflight_preserves_qwen_budget_and_can_skip_chat(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            curl = directory / "curl"
            curl.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$ARGV_CAPTURE"\n')
            curl.chmod(0o755)
            capture = directory / "argv"
            script = step_script(CODEX_JOB, "Preflight vLLM Responses and Chat Completions")
            for candidate, skip_chat, chat_budget in (("false", "false", 1),
                                                       ("true", "false", 16),
                                                       ("true", "true", None)):
                with self.subTest(candidate=candidate, skip_chat=skip_chat):
                    capture.unlink(missing_ok=True)
                    env = {
                        **os.environ,
                        "PATH": f"{directory}:{os.environ['PATH']}",
                        "ARGV_CAPTURE": str(capture),
                        "CODEX_BACKEND_TOKEN": "synthetic-credential",
                        "CODEX_VLLM_MODEL": "gpt-oss-120b" if candidate == "true" else "qwen3-8b",
                        "CODEX_CANDIDATE": candidate,
                        "CODEX_SKIP_TRANSLATED": skip_chat,
                    }
                    subprocess.run(["bash", "-c", script], env=env, check=True)
                    args = capture.read_text().splitlines()
                    urls = [arg for arg in args if arg.startswith("http://127.0.0.1:8000/")]
                    paths = ["health", "v1/models", "v1/responses"]
                    if chat_budget is not None:
                        paths.append("v1/chat/completions")
                        self.assertEqual(json.loads(args[-2])["max_tokens"], chat_budget)
                    self.assertEqual(urls, [f"http://127.0.0.1:8000/{path}" for path in paths])

    def test_provenance_records_chains_only_from_executed_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            native_chain = ["ai_operation", "openai_client_tool_compat", "credential_injection", "load_balancer"]
            (directory / "codex-acceptance-native.log").write_text(
                f"Codex acceptance filter chain: {json.dumps(native_chain)}\n"
            )
            env = {
                **os.environ,
                "RUNNER_TEMP": str(directory),
                "GITHUB_RUN_ID": "123",
                "GITHUB_RUN_ATTEMPT": "2",
                "CODEX_CANDIDATE": "true",
                "CODEX_VERSION": "0.170.0",
                "CODEX_MODEL_REVISION": "a" * 40,
                "CODEX_MODEL": "openai/gpt-oss-120b",
                "CODEX_VLLM_MODEL": "gpt-oss-120b",
                "VLLM_SERVED_IMAGE": "vllm-test-image",
                "CODEX_MAX_MODEL_LEN": "24576",
                "CODEX_TENSOR_PARALLEL_SIZE": "4",
                "CODEX_TOOL_CALL_PARSER": "openai",
                "CODEX_REASONING_PARSER": "openai_gptoss",
                "CODEX_GPU_INSTANCE_TYPE": "g6e.12xlarge",
                "CODEX_NATIVE": "success",
                "CODEX_TRANSLATED": "skipped",
                "CODEX_COMPACTION": "failure",
            }
            (directory / "codex-vllm-image.json").write_text(json.dumps({"image_tag": "stale"}))
            script = step_script(CODEX_JOB, "Capture Codex acceptance scenarios")
            subprocess.run(["bash", "-c", script], cwd=ROOT, env=env, check=True)
            result = json.loads((directory / "codex-acceptance.json").read_text())
            self.assertEqual((result["run_id"], result["attempt"]), (123, 2))
            self.assertEqual(result["provenance"]["filter_chains"], {"codex_native": native_chain})
            self.assertTrue(result["provenance"]["candidate"])
            self.assertNotIn("backend", result)
            self.assertEqual(result["provenance"]["backend"]["vllm_image"], "vllm-test-image")
            self.assertEqual(result["provenance"]["backend"]["model"], "openai/gpt-oss-120b")
            self.assertEqual(result["provenance"]["backend"]["max_model_len"], 24576)
            self.assertEqual(result["provenance"]["backend"]["resolved_revision"], "a" * 40)
            self.assertEqual(result["provenance"]["runner"]["instance_type"], "g6e.12xlarge")
            self.assertEqual([case["status"] for case in result["cases"]], ["success", "skipped", "failure"])

            # The prepare step supplies the baked checkpoint, revision, and image identity.
            baked = {"model": "Qwen/Qwen3-8B", "model_revision": "b" * 40,
                     "image_tag": "test-baked-image", "image_source": "registry",
                     "local_image_id": "sha256:test-baked", "registry_digest": "sha256:test-registry"}
            (directory / "codex-vllm-image.json").write_text(json.dumps(baked))
            env.update(CODEX_CANDIDATE="false", CODEX_MODEL=baked["model"], CODEX_MODEL_REVISION="",
                       BAKED_REVISION=baked["model_revision"],
                       CODEX_VLLM_MODEL="custom-qwen", VLLM_SERVED_IMAGE=baked["image_tag"])
            subprocess.run(["bash", "-c", script], cwd=ROOT, env=env, check=True)
            result = json.loads((directory / "codex-acceptance.json").read_text())
            self.assertFalse(result["provenance"]["candidate"])
            self.assertEqual(result["backend"], baked)
            backend = result["provenance"]["backend"]
            for key, expected in (("model", baked["model"]), ("resolved_revision", baked["model_revision"]),
                                  ("vllm_image", baked["image_tag"]), ("served_model_name", "custom-qwen"),
                                  ("max_model_len", 24576), ("tensor_parallel_size", 4),
                                  ("tool_call_parser", "openai"), ("reasoning_parser", "openai_gptoss")):
                self.assertEqual(backend[key], expected, key)

            env["CODEX_COMPACTION"] = "success"
            missing = subprocess.run(["bash", "-c", script], cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("Missing filter-chain evidence for: codex_compaction", missing.stderr)

            # If preparation failed before launch, the effective context is unknown.
            (directory / "codex-vllm-image.json").unlink()
            env.update(CODEX_NATIVE="skipped", CODEX_COMPACTION="skipped", CODEX_MAX_MODEL_LEN="",
                       VLLM_SERVED_IMAGE="", CODEX_MODEL="", BAKED_REVISION="")
            subprocess.run(["bash", "-c", script], cwd=ROOT, env=env, check=True)
            result = json.loads((directory / "codex-acceptance.json").read_text())
            self.assertIsNone(result["provenance"]["backend"]["max_model_len"])
            self.assertNotIn("backend", result)
