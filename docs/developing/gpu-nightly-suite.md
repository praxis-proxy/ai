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
| [`vllm-gpu-container.yaml`](../../.github/workflows/vllm-gpu-container.yaml) | push / PR to `main` on path filters (`vllm/Containerfile`, `vllm/images.json`, the GPU actions, and the workflow itself), `merge_group`, and manual | Builds, live-tests, and publishes **every** model-baked GPU image in [`vllm/images.json`](../../vllm/images.json) to GHCR. Sole publisher; the two nightlies consume what it publishes. |
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
- **Images:** `ghcr.io/praxis-proxy/vllm-gpu:Qwen3-8B` (main suite) and
  `:Qwen3-VL-4B-Instruct` (vision companion), both published by
  `vllm-gpu-container.yaml` from [`vllm/images.json`](../../vllm/images.json).
  Each also gets an immutable `:<tag>-<commit sha>` tag. The nightlies pull
  these rather than rebuilding them; see
  [Reusing published images](#reusing-published-images).

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
| **OpenCode** | native | Chat Completions passed through | `/v1/chat/completions` |
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
- **CLI self-compaction (Codex + Claude Code).** Two acceptance tests prove a
  real pinned CLI grows its context past its own window, performs its **own**
  self-compaction, and keeps working through Praxis afterward — without ever
  calling an SDK compact endpoint, injecting a canned summary, or enabling a
  Praxis compaction filter. The compaction signal is captured directly: for Codex
  from the client's inline-summarization POST on the client→upstream wire
  (`codex_http::pinned_codex_compaction_crosses_context_window_over_http`), and
  for Claude Code from its stream-json `compact_boundary {trigger: auto}` event
  (`claude_code_vllm::pinned_claude_code_compaction_crosses_context_window_on_native_vllm`).
  See the limitation note below for how task-completion is graded on Qwen3-8B.

See also the vision companion workflow (`anthropic-vllm-vision.yaml`), which
covers the Anthropic SDK image/vision path against `Qwen3-VL-4B-Instruct`.

### Known model limitation: self-compaction task completion on Qwen3-8B

The self-compaction tests prove two things independently of model capability,
and these are **always hard assertions**:

1. The pinned CLI performed its **own** self-compaction. Codex's summarization
   request is observed **traversing Praxis** on the wire. Claude Code's
   `compact_boundary {trigger: auto}` is a client-side stream-json event that is
   local to the CLI (it does not itself cross Praxis); the test observes that event
   and separately verifies post-boundary requests continue **through Praxis**.
2. The session **continued after the boundary** — at least one more successful
   tool call flowed through Praxis post-compaction.

What Qwen3-8B cannot do reliably is **finish the downstream chore** after it
self-compacts. It either falls into a loop re-reading the same ballast file until
the deadline, or drops the high-entropy marker across its own lossy summary and
writes the wrong value. This is a model-capability limit, not a proxy or flake,
and it was observed on **both** lanes against live Qwen3-8B.

Both lanes grade identically by design — **XFAIL on Qwen3-8B**:

- **Codex.** After the hard wire-compaction proof, the downstream
  task-completion assertions (correct `result.txt`, wire ordering of the write
  after compaction, the ordered JSONL trace, `./verify.sh`) are **skipped** when
  the live model matches `COMPACTION_XFAIL_MODEL` (`qwen3-8b`), with an
  `XFAIL (...)` line logged.
- **Claude Code.** Same shape: after the hard proofs (self-compaction boundary,
  the pre-compaction Read carrying the marker through Praxis, and a
  post-compaction tool call through Praxis), the downstream completion assertions
  (successful post-compaction Edit of the uppercase marker into
  `result/value.txt`, then a passing `./verify.sh`) are **skipped** on
  `COMPACTION_XFAIL_MODEL`, with an `XFAIL (...)` line logged. The prompt steers a
  post-compaction re-read of `source/value.txt` to recover the marker, but the
  oracle never mandated that specific tool call.

Each lane uses a shorter deadline (`COMPACTION_XFAIL_CHILD_TIMEOUT`, 180s) on the
XFAIL model so a looping run ends quickly once compaction is already captured.
Point either test at a more capable model and its full completion oracle runs as
hard assertions automatically.

To iterate on just these two tests without running the ~90-minute full suite,
dispatch `vllm-integration.yaml` with `run_compaction_only=true` **alone** (do
NOT also pass `run_live_vllm=true`, which would trigger the full suite): it
provisions the GPU runner and runs only the Codex and Claude Code compaction
steps, skipping the full Responses suite and every other acceptance step.

### Known model limitation: server-side web search on Qwen3-8B

The Claude Code server-side web-search acceptance test
(`claude_code_vllm::pinned_claude_code_resolves_server_side_web_search_through_tavily`)
drives the **real** `claude` binary, which puts `WebSearch` on the wire as a
custom client tool with `tool_choice: null` — it never *forces* the tool. Qwen3-8B
does not reliably drive that loop to a finished answer. Two failure shapes were
observed on live Qwen3-8B, both model-capability limits rather than proxy bugs or
flakes:

- it never decides to call `WebSearch`, so the managed `anthropic_web_search` loop
  dispatches no search and the turn returns an empty answer; or
- it *does* call `WebSearch`, Praxis resolves a real Tavily search server-side (a
  valid exchange, tool_use suppressed) and the turn exits `success`, but the 8B
  model then synthesizes **no** final answer from the results (`result:"\n\n"`).

> Root cause of the Oct 10–12 2026 red run — a test-infra bug, not either model
> limit above. Issue #1389 moved the Tavily credential from a body `api_key`
> field to an `Authorization: Bearer` header, but the loopback capture-relay
> (`tests/utils/src/tavily_relay.rs`) still forwarded only the body, so real
> Tavily answered **401** and the managed loop fell back to an `is_error` tool
> result — surfacing as the empty-answer shape above. The relay now forwards the
> `Authorization` header verbatim and records it, and the oracle validates the
> resolved key travels there (a **proxy regression**, below, if it is missing or
> wrong), so a dropped credential fails hard instead of masquerading as a model
> limit.

Praxis's server-side Tavily resolution is proven on every run by the
**forced-tool-choice** SDK web-search tests, which set
`tool_choice={type:tool,name:WebSearch}` to open the loop reliably on a small
model:

- `test_live_tavily_web_search_returns_real_sources` (OpenAI Responses and
  Anthropic SDK), and
- the `ANTHROPIC_STUBBED_WEB_SEARCH` / `ANTHROPIC_CREDENTIALED_WEB_SEARCH`
  acceptance scenarios.

So the Claude Code lane grades like the compaction lane: **hard-assert everything
the proxy owns, XFAIL only the model-completion outcome.** The
`check_server_side_web_search` oracle classifies each failure (`WebSearchFailure`):

- **Proxy regression — hard on every model.** A leaked `WebSearch` tool_use (the
  managed loop must suppress it), an `is_error` flag on an otherwise-successful
  turn, or any Tavily exchange that *did* run being malformed or failed (non-200,
  the resolved key missing from the forwarded `Authorization: Bearer` header per
  issue #1389, empty query, empty/relative-URL results). These are checked
  **first** and always, so a broken managed loop or relay can never hide behind the
  limitation.
- **Model incomplete — XFAIL on `WEB_SEARCH_XFAIL_MODEL` (`qwen3-8b`), hard on a
  capable model.** The turn did not converge, the model never called `WebSearch`
  (no Tavily exchange), or it got real results and produced no usable final
  answer. On `qwen3-8b` these are **skipped** with an `XFAIL (...)` line logged,
  and the child must still have run to completion cleanly through Praxis.

Making the proxy inject/force `tool_choice` for the managed web-search tool (so
small backends open the loop through the real client) would let this lane drop the
XFAIL — tracked separately as a possible filter feature, not done here.

## Reusing published images

Neither nightly rebuilds the model-baked image by default. Each calls
[`resolve-vllm-gpu-image`](../../.github/actions/resolve-vllm-gpu-image/action.yml),
which pulls the published tag and keeps it only when the image's provenance
labels match that checkout — the same `vllm/Containerfile` hash, model ID, and
pinned model revision. On a mismatch, an unreachable registry, or a tag that
does not exist yet, it falls back to building from source.

That keeps two properties at once: a nightly where nothing changed skips a
multi-gigabyte rebuild entirely, and a PR that edits the `Containerfile` or
repins a model is still validated against the image it actually describes. A
lagging or failed publish degrades to a local build rather than breaking the
run or, worse, qualifying a stale backend.

The qualification record reports which happened: `backend.image_source` is
`registry` or `local-build`, and `backend.registry_digest` is populated only
for a pulled image.

## Jobs in the main suite (`vllm-integration.yaml`)

- `changes` — path-filter gate.
- `vllm-responses`, `vllm-live-cpu`, `vllm-responses-postgres` — fast CPU-side
  coverage on PRs (simulator + `llm-d-inference-sim`).
- `gpu-start-runner` / `gpu-stop-runner` — provision / tear down the GPU EC2 box.
- `vllm-gpu-full-suite` — resolves the model-baked image (pulling the
  published build, or building from source when it does not match the
  checkout) and runs the complete live Responses SDK suite (SDK 2.x + 3.x)
  plus the OpenAI Agents SDK loop against real Qwen3-8B with a PostgreSQL
  store.
- `vllm-gpu-claude-acceptance` — pinned Claude Code over native + translated
  Anthropic paths (acceptEdits + auto mode), including the #1418 planning-turn
  regression guard, in a locked-down network namespace.
- `vllm-gpu-codex-acceptance` — pinned Codex over native Responses + translated
  Chat paths.
- `vllm-gpu-opencode-acceptance` — pinned OpenCode over the native Chat
  Completions path: one text turn and one tool-call round trip, in its own
  locked-down network namespace.
- `pins` — parses `tests/integration/fixtures/claude-code-cli/pin.toml` and
  `tests/integration/fixtures/opencode-cli/pin.toml`, cross-validates their
  invariants, and exports every value the acceptance jobs consume so those
  jobs carry no hardcoded version strings.
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
- `tests/integration/tests/suite/opencode_vllm.rs` — OpenCode acceptance
  (`opencode_vllm_plugin_auth_completes_text_turn`,
  `opencode_vllm_tool_call_round_trip_writes_hash_verified_marker`). The
  deterministic CPU lane is `suite/opencode.rs`, run by the `opencode-cpu`
  job in `integration.yaml`.
- `tests/integration/tests/suite/live_vllm.rs` — namespace, egress-probe and
  example-patching scaffolding shared by the live CLI suites.

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
  Claude Code / OpenCode through Praxis and vLLM; reasoning-flag regression,
  the headless `opencode run` form, and the OpenCode plugin's fail-open
  behavior.
- [`vllm/README.md`](../../vllm/README.md) — GPU container: build, run, health
  check, hardware/model rationale.
- [llmd-integration-testing.md](llmd-integration-testing.md) — adjacent CPU-side
  `llm-d-inference-sim` tests.
