# Nightly GPU test suite

This page is the single entry point for understanding what our GPU-backed
nightly CI runs, what hardware it uses, and where the deeper documentation
lives. The GPU-backed jobs run against **real vLLM on a GPU**; ordinary PRs
also get faster CPU-side coverage against the inference simulator (see the
`vllm-responses` / `vllm-live-cpu` jobs below).

> Note: `.github/workflows/nightly.yaml` is **not** the GPU suite. Despite the
> name it is a CPU-only nightly (lint, docs, unit tests, audit, coverage). The
> GPU nightly is the `schedule` trigger inside `vllm-integration.yaml`.

## What runs, and when

| Workflow | Schedule (UTC) | What it exercises |
| --- | --- | --- |
| [`vllm-integration.yaml`](../../.github/workflows/vllm-integration.yaml) | `17 4 * * *` | Main GPU suite: live Responses SDK suite, OpenAI Agents SDK loop, and the Claude Code / Codex CLI acceptance jobs against real Qwen3-8B. |
| [`anthropic-vllm-vision.yaml`](../../.github/workflows/anthropic-vllm-vision.yaml) | `31 6 * * *` | Vision companion: Anthropic SDK image requests through Praxis to `Qwen/Qwen3-VL-4B-Instruct` (the main nightly model is text-only). |
| [`vllm-gpu-container.yaml`](../../.github/workflows/vllm-gpu-container.yaml) | push / PR to `main` on path filters (`vllm/Containerfile`, the GPU actions, and the workflow itself), `merge_group`, and manual | Builds, live-tests, and publishes the model-baked GPU image to GHCR. |
| [`vllm-dev-endpoint.yaml`](../../.github/workflows/vllm-dev-endpoint.yaml) | manual only | On-demand vLLM endpoint over a Cloudflare tunnel for interactive testing. Not part of CI. |

All also accept `workflow_dispatch`, and the main suite can be forced on a PR by
applying the `vllm-full-suite` label.

## Hardware and model

- **Instance:** `g5.xlarge` — NVIDIA A10G, 24 GiB, compute capability 8.6
  (FlashAttention-2 available). Provisioned as an ephemeral EC2 runner and torn
  down unconditionally after the run (`gpu-start-runner` / `gpu-stop-runner`).
- **Model:** `Qwen/Qwen3-8B` (~15.3 GiB bf16) — chosen because it supports
  tool-calling (file_search, MCP); the 0.6B CPU simulator model used on PRs does
  not.
- **Built image:** `ghcr.io/praxis-proxy/vllm-gpu:Qwen3-8B`.

## Use cases: the client-path matrix

The suite's whole point is proving real coding clients and SDKs complete real
work through Praxis against real vLLM, across **both** production bridging paths
— native passthrough and request/response translation — so a Chat-Completions-only
backend can serve the same client.

| Client / SDK | Path | Praxis bridge | vLLM endpoint |
| --- | --- | --- | --- |
| **Codex CLI** | native | Responses passed through | `/v1/responses` |
| **Codex CLI** | translated | Responses → Chat Completions | `/v1/chat/completions` |
| **Claude Code** | native | Anthropic Messages passed through | `/v1/messages` |
| **Claude Code** | translated | `anthropic_messages_to_chat_completions[_stream]` | `/v1/chat/completions` |
| **OpenAI Agents SDK** | native | SDK-owned tool loop over Responses | `/v1/responses` |
| **OpenAI SDK (Responses)** | native | Responses SDK suite, SDK **2.x and 3.x** | `/v1/responses` |

Details per use case:

- **Codex CLI — native vs translated.** A pinned Codex binary completes the same
  coding workflow twice: once over vLLM's native `/v1/responses` (with a
  PostgreSQL response store), once with Praxis translating Responses → Chat
  Completions. Tests:
  `codex_http::pinned_codex_completes_native_vllm_coding_workflow_over_http`
  and `..._completes_chat_backend_coding_workflow_over_http`
  (config `openai/responses/codex-http-chat-translation.yaml`).
- **Claude Code — native vs translated.** A pinned Claude Code binary completes
  the same multi-step task under four scenarios: {native (`messages-native-vllm.yaml`)
  / transformed (`messages-to-openai-vllm.yaml`)} × {deterministic `acceptEdits`
  / client-initiated auto-mode}. A fifth scenario covers the #1418 read-only
  planning turn over the native path (asserts convergence and no reasoning-channel
  leak into the user-visible answer).
- **OpenAI Agents SDK.** The SDK's `Runner` drives a client-owned function-tool
  loop (model → tool → model) end to end through `POST /v1/responses`, with the
  tool call forced via `tool_choice="required"` so a real model can't flake the
  loop. Only `/v1/responses` ever crosses Praxis.
- **OpenAI SDK (Responses), 2.x and 3.x.** The full Responses SDK suite runs
  against both major SDK generations to catch protocol drift. Tests marked
  `real_inference` / `vllm_compat` run only in live (GPU) mode; the same tests
  run on PRs against the CPU simulator (`VLLM_TEST_BACKEND=simulator`).

See also the vision companion workflow (`anthropic-vllm-vision.yaml`), which
covers the Anthropic SDK image/vision path against `Qwen3-VL-4B-Instruct`.

## Jobs in the main suite (`vllm-integration.yaml`)

- `changes` — path-filter gate.
- `vllm-responses`, `vllm-live-cpu`, `vllm-responses-postgres` — fast CPU-side
  coverage on PRs (simulator + `llm-d-inference-sim`).
- `gpu-start-runner` / `gpu-stop-runner` — provision / tear down the GPU EC2 box.
- `vllm-gpu-full-suite` — builds the model-baked image and runs the complete
  live Responses SDK suite (SDK 2.x + 3.x) plus the OpenAI Agents SDK loop
  against real Qwen3-8B with a PostgreSQL store.
- `vllm-gpu-claude-acceptance` — pinned Claude Code over native + translated
  Anthropic paths (acceptEdits + auto mode), including the #1418 planning-turn
  regression guard, in a locked-down network namespace.
- `vllm-gpu-codex-acceptance` — pinned Codex over native Responses + translated
  Chat paths.
- `pins` — parses `tests/integration/fixtures/claude-code-cli/pin.toml`.
- `gpu-qualification-summary` — writes `qualification.json` and the Actions
  summary even if provisioning fails.

## Where the test code lives

There is no README inside the SDK test directories; per-file module docstrings
carry the detail.

- `tests/integration/sdk/openai/test_openai_responses_vllm.py` — core live
  Responses suite (`VLLM_TEST_BACKEND=live` vs `simulator`).
- `tests/integration/sdk/openai/test_openai_agents_sdk.py` — OpenAI Agents SDK
  loop (deterministic vs live GPU mode).
- `tests/integration/sdk/anthropic/test_anthropic_messages_vllm.py` — Anthropic
  SDK → Praxis → real vLLM Messages acceptance.
- `tests/integration/sdk/anthropic/test_anthropic_web_search_vllm.py` — Anthropic
  web search.
- `tests/integration/tests/suite/claude_code_vllm.rs` — Claude Code acceptance
  (native vs transformed path diagram, scenarios incl. #1418).
- `tests/integration/tests/suite/codex_http.rs` — Codex acceptance + pin-update
  instructions.

## Running it locally

Reproduce the core suite against a local GPU (see
[cli-vllm-through-praxis.md](cli-vllm-through-praxis.md) for serving vLLM):

```console
cargo build -p praxis-ai-proxy --features full,store-sqlite
VLLM_MODEL=Qwen/Qwen3-8B VLLM_TEST_BACKEND=live \
  uv run tests/integration/sdk/openai/test_openai_responses_vllm.py
```

## Deeper references

- [vllm-qualification.md](vllm-qualification.md) — what the suite produces
  (`qualification.json` schema, report fields, pass/fail semantics, release
  selection).
- [cli-vllm-through-praxis.md](cli-vllm-through-praxis.md) — running Codex /
  Claude Code / OpenCode through Praxis and vLLM; reasoning-flag regression.
- [`vllm/README.md`](../../vllm/README.md) — GPU container: build, run, health
  check, hardware/model rationale.
- [llmd-integration-testing.md](llmd-integration-testing.md) — adjacent CPU-side
  `llm-d-inference-sim` tests.
