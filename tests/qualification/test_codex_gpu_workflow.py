# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github/workflows/vllm-integration.yaml").read_text()
CODEX_JOB = "vllm-gpu-codex-acceptance"
CANDIDATE_CRON = "42 7 * * 1"
CANDIDATE_WORKFLOW = ROOT / ".github/workflows/codex-candidate-qualification.yaml"
RUNNER_ACTION = (ROOT / ".github/actions/start-ec2-runner/action.yml").read_text()


def job_body(job):
    return re.split(r"\n  [a-z][a-z0-9-]*:\n", WORKFLOW.split(f"\n  {job}:\n", 1)[1], maxsplit=1)[0]


def step_script(job, step):
    job_body_text = job_body(job)
    body = job_body_text.split(f"      - name: {step}\n", 1)[1].split("\n      - ", 1)[0]
    return "\n".join(line.removeprefix("          ") for line in body.split("        run: |\n", 1)[1].splitlines())


class CodexGpuScheduleTest(unittest.TestCase):
    def test_opt_in_schedule_only_runs_candidate_lane(self):
        self.assertNotIn(f'- cron: "{CANDIDATE_CRON}"', WORKFLOW)
        self.assertIn('- cron: "17 4 * * *"', WORKFLOW)
        candidate = " ".join(CANDIDATE_WORKFLOW.read_text().split())
        self.assertIn(f'- cron: "{CANDIDATE_CRON}"', candidate)
        self.assertIn(
            "github.repository == 'praxis-proxy/ai' && "
            "github.ref == 'refs/heads/main' && "
            "vars.CODEX_GPT_OSS_120B_SCHEDULE == 'true'",
            candidate,
        )
        self.assertIn("uses: ./.github/workflows/vllm-integration.yaml", candidate)
        self.assertIn("scheduled_codex_candidate: true", candidate)
        start = " ".join(job_body("gpu-start-runner").split())
        self.assertIn("if: | github.event_name == 'schedule' ||", start)
        self.assertIn("inputs.scheduled_codex_candidate && 'p5.4xlarge'", start)
        self.assertIn("&& vars.AWS_GPU_FALLBACK_SUBNET_IDS || ''", start)
        for job in ("vllm-responses", "vllm-live-cpu", "vllm-responses-postgres"):
            with self.subTest(job=job):
                self.assertIn("github.event_name != 'schedule'", job_body(job).split("    steps:", 1)[0])
        for job in ("pins", "vllm-gpu-claude-acceptance", "vllm-gpu-opencode-acceptance",
                    "vllm-gpu-pi-acceptance", "vllm-gpu-full-suite"):
            with self.subTest(job=job):
                condition = " ".join(job_body(job).split("    steps:", 1)[0].split())
                self.assertIn("github.event_name == 'schedule' && !inputs.scheduled_codex_candidate", condition)
        codex = " ".join(job_body(CODEX_JOB).split())
        self.assertIn("needs.gpu-start-runner.result == 'success'", codex)
        for value in ("openai/gpt-oss-120b", "gpt-oss-120b", "openai", "openai_gptoss"):
            self.assertIn(f"inputs.scheduled_codex_candidate && '{value}'", codex)
        self.assertIn("CODEX_CANDIDATE: >- ${{ inputs.scheduled_codex_candidate ||", codex)
        self.assertIn("inputs.scheduled_codex_candidate || inputs.codex_skip_translated", codex)
        self.assertIn("env.CODEX_SKIP_TRANSLATED != 'true'", codex)
        for case in ("codex_native", "codex_compaction"):
            with self.subTest(case=case):
                step = job_body(CODEX_JOB).split(f"        id: {case}\n", 1)[1].split("\n      - ", 1)[0]
                self.assertIn("PRAXIS_TEST_CODEX_CANDIDATE: ${{ env.CODEX_CANDIDATE }}", step)
                self.assertNotIn("CODEX_SKIP_TRANSLATED", step)
        stop = job_body("gpu-stop-runner")
        self.assertIn(CODEX_JOB, stop)
        self.assertIn("always() &&", stop)
        self.assertIn("needs.gpu-start-runner.outputs.ec2-instance-id != ''", stop)
        summary = " ".join(job_body("gpu-qualification-summary").split())
        self.assertIn(CODEX_JOB, summary)
        self.assertIn("if: always()", summary)
        self.assertIn("GPU_REQUESTED: >- ${{ (github.event_name == 'schedule' && !inputs.scheduled_codex_candidate)", summary)

    def test_fallback_passes_ordered_same_region_configs_to_pinned_action(self):
        self.assertIn("availability-zones-config: ${{ steps.subnets.outputs.config }}", RUNNER_ACTION)
        step = RUNNER_ACTION.split("    - name: Build EC2 subnet fallback\n", 1)[1].split("\n    - ", 1)[0]
        self.assertIn("if: inputs.fallback-subnet-ids != ''", step)
        script = "\n".join(line.removeprefix("        ") for line in step.split("      run: |\n", 1)[1].splitlines())
        primary, first, second = ("subnet-11111111", "subnet-22222222", "subnet-33333333")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            env = {
                **os.environ,
                "GITHUB_OUTPUT": str(output),
                "AWS_GPU_AMI_ID": "ami-test",
                "AWS_SUBNET_ID": primary,
                "AWS_SECURITY_GROUP_ID": "sg-test",
                "FALLBACK_SUBNET_IDS": f"{first}\n{second} {primary}",
            }
            result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text().removeprefix("config=")), [
                {"imageId": "ami-test", "subnetId": subnet, "securityGroupId": "sg-test"}
                for subnet in (primary, first, second)
            ])
            output.unlink()
            for bad in ("not-a-subnet", "subnet-abc", f"{first} {second} subnet-44444444"):
                with self.subTest(bad=bad):
                    result = subprocess.run(
                        ["bash", "-c", script], env={**env, "FALLBACK_SUBNET_IDS": bad},
                        capture_output=True, text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(output.exists())


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
                    "CONTAINER_RUNTIME": runtime,
                    "GPU_HF_CACHE_DIR": str(directory / "hf"),
                    "VLLM_GPU_IMAGE": "vllm-test-image",
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
                self.assertEqual(argv[argv.index("--model"):], [
                    "--model", "openai/gpt-oss-120b", "--served-model-name", "gpt-oss-120b",
                    "--max-model-len", "24576", "--gpu-memory-utilization", "0.9", "--seed", "0",
                    "--api-key", "synthetic-credential", "--revision", "a" * 40,
                    "--tokenizer-revision", "a" * 40, "--tensor-parallel-size", "4",
                    "--enable-auto-tool-choice", "--tool-call-parser", "openai",
                    "--reasoning-parser", "openai_gptoss",
                ])

                capture.unlink()
                env.update(CODEX_TENSOR_PARALLEL_SIZE="1", CODEX_TOOL_CALL_PARSER="none",
                           CODEX_REASONING_PARSER="none")
                subprocess.run(["bash", "-c", script], env=env, check=True)
                single_gpu_argv = capture.read_text().split("\0")
                for absent in ("--ipc=host", "--tensor-parallel-size", "--enable-auto-tool-choice",
                               "--tool-call-parser", "--reasoning-parser"):
                    self.assertNotIn(absent, single_gpu_argv)

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
                "VLLM_GPU_IMAGE": "vllm-test-image",
                "CODEX_MAX_MODEL_LEN": "24576",
                "CODEX_TENSOR_PARALLEL_SIZE": "4",
                "CODEX_TOOL_CALL_PARSER": "openai",
                "CODEX_REASONING_PARSER": "openai_gptoss",
                "CODEX_GPU_INSTANCE_TYPE": "g6e.12xlarge",
                "CODEX_NATIVE": "success",
                "CODEX_TRANSLATED": "skipped",
                "CODEX_COMPACTION": "failure",
            }
            script = step_script(CODEX_JOB, "Capture Codex acceptance scenarios")
            subprocess.run(["bash", "-c", script], cwd=ROOT, env=env, check=True)
            result = json.loads((directory / "codex-acceptance.json").read_text())
            self.assertEqual((result["run_id"], result["attempt"]), (123, 2))
            self.assertEqual(result["provenance"]["filter_chains"], {"codex_native": native_chain})
            self.assertEqual(result["provenance"]["backend"]["max_model_len"], 24576)
            self.assertEqual(result["provenance"]["backend"]["resolved_revision"], "a" * 40)
            self.assertEqual(result["provenance"]["runner"]["instance_type"], "g6e.12xlarge")
            self.assertEqual([case["status"] for case in result["cases"]], ["success", "skipped", "failure"])

            env["CODEX_COMPACTION"] = "success"
            missing = subprocess.run(["bash", "-c", script], cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("Missing filter-chain evidence for: codex_compaction", missing.stderr)
