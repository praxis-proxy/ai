# Nightly GPU test suite

This page covers what the GPU nightly CI runs, on which hardware, and where
the detailed docs are. The GPU jobs run against real vLLM on a GPU. PRs get
faster CPU coverage: `vllm-responses` runs against the inference simulator,
and `vllm-live-cpu` runs against real vLLM on CPU with `Qwen/Qwen3-0.6B`.

> Note: `.github/workflows/nightly.yaml` is **not** the GPU suite. It is a
> CPU-only nightly (lint, docs, unit tests, audit, coverage). The GPU nightly
> is the `schedule` trigger inside `vllm-integration.yaml`.

## What runs, and when

| Workflow | Schedule (UTC) | What it exercises |
| --- | --- | --- |
| [`vllm-integration.yaml`](../../.github/workflows/vllm-integration.yaml) | `17 4 * * *` | Main GPU suite: live Responses SDK suite, OpenAI Agents SDK loop, and the Claude Code / Codex CLI acceptance jobs against real Qwen3-8B. |
| [`anthropic-vllm-vision.yaml`](../../.github/workflows/anthropic-vllm-vision.yaml) | `31 6 * * *` | Vision companion: Anthropic SDK image requests, OpenResponses image acceptance, and an OpenAI SDK image request through Responses-to-Chat translation, against `Qwen/Qwen3-VL-4B-Instruct` (the main nightly model is text-only). |
| [`vllm-gpu-container.yaml`](../../.github/workflows/vllm-gpu-container.yaml) | push / PR to `main` on path filters (`vllm/Containerfile`, `vllm/images.json`, the GPU actions, and the workflow itself), `merge_group`, and manual | Builds and live-tests every model-baked GPU image in [`vllm/images.json`](../../vllm/images.json); push and manual runs also publish them to GHCR, and a manual run can select one image. It is the only publisher; the two nightlies consume its images. |
| [`vllm-dev-endpoint.yaml`](../../.github/workflows/vllm-dev-endpoint.yaml) | manual only | On-demand vLLM endpoint over a Cloudflare tunnel for interactive testing. Not part of CI. |

All of them also accept `workflow_dispatch`. Apply the `vllm-full-suite` label
to force the main suite on a PR.

## Hardware and model

- **Instance:** `g5.xlarge`: NVIDIA A10G, 24 GiB, compute capability 8.6
  (FlashAttention-2 available). An ephemeral EC2 runner, always torn down
  after the run (`gpu-start-runner` / `gpu-stop-runner`).
- **Model:** `Qwen/Qwen3-8B` (~15.3 GiB bf16). It supports tool calling
  (file_search, MCP); the 0.6B CPU model used on PRs does not.
- **Images:** `ghcr.io/praxis-proxy/vllm-gpu:Qwen3-8B` (main suite and every
  client acceptance job) and `:Qwen3-VL-4B-Instruct` (vision companion), both
  published by `vllm-gpu-container.yaml` from
  [`vllm/images.json`](../../vllm/images.json). Each also gets an immutable
  `:<tag>-<commit sha>` tag. The nightlies pull these instead of rebuilding
  them; see [Reusing published images](#reusing-published-images).

For verified CLI edits, permission modes and compaction outcomes by model and API route, see [Coding harness compatibility](../conformance/coding-harness-compatibility.md).

## Use cases: the client-path matrix

The suite proves that real coding clients and SDKs complete real work through
Praxis against real vLLM, over both native passthrough and request/response
translation, so a Chat-Completions-only backend can serve the same client.

| Client / SDK | Path | Praxis bridge | vLLM endpoint |
| --- | --- | --- | --- |
| **Codex CLI** | native | Responses passed through | `/v1/responses` |
| **Codex CLI** | translated | Responses → Chat Completions | `/v1/chat/completions` |
| **Claude Code** | native | Anthropic Messages passed through | `/v1/messages` |
| **Claude Code** | translated | `anthropic_messages_to_chat_completions[_stream]` | `/v1/chat/completions` |
| **OpenCode** | native | Chat Completions passed through, CLI **1.x and 2.x** | `/v1/chat/completions` |
| **pi** | native | Chat Completions passed through (`openai-completions` provider) | `/v1/chat/completions` |
| **OpenAI Agents SDK** | native | SDK-owned tool loop over Responses | `/v1/responses` |
| **OpenAI SDK (Responses)** | native | Responses SDK suite, SDK **2.x and 3.x** | `/v1/responses` |
| **OpenAI SDK (Chat Completions)** | translated (selected provider) | `openai_chat_completions_to_anthropic[_stream]` | `/v1/messages` |
| **Anthropic SDK (vision companion)** | native + translated | Messages passthrough / Messages → Chat Completions | `/v1/messages` / `/v1/chat/completions` |
| **OpenResponses + OpenAI SDK (vision companion)** | translated | Responses → Chat Completions | `/v1/chat/completions` |

Details per use case:

- **Codex CLI.** A pinned Codex binary uses its grammar-constrained
  `apply_patch` tool to create `result.txt`, returns the tool output through
  Praxis, and runs the verifier. Native `/v1/responses` uses a PostgreSQL
  response store:
  `codex_http::pinned_codex_completes_native_vllm_apply_patch_workflow_over_http`
  and
  `codex_http::pinned_codex_completes_translated_vllm_apply_patch_workflow_over_http`
  (configs `openai/responses/client-tool-compat.yaml` and
  `openai/responses/client-tool-compat-chat-completions.yaml`). The test
  selects Codex's `gpt-5.5` model metadata so the CLI exposes grammar-based
  `apply_patch`, then rewrites the request's model field to the live vLLM
  model; inference still runs on Qwen. An offline pinned HTTP test checks that
  this catalog entry declares the grammar tool, so a client upgrade cannot
  silently remove it (0.167.0 did that for `gpt-5.4`).
- **Codex CLI live web search.** Two credentialed GPU steps enable live search on
  the pinned client, one per Responses route. Each adds the Tavily dispatcher and
  agentic loop transition to its client-tool compatibility example. A host-side
  relay records the real Tavily exchange while the isolated CLI reaches Praxis.
  A passing case requires a completed CLI answer citing a returned source URL.
  Read the independent `codex_native_search` and `codex_translated_search` results
  in `suites.codex_acceptance.cases`. Fork PRs and compaction-only runs skip both.
- **Claude Code.** A pinned Claude Code binary completes the same multi-step
  task in four scenarios: {native (`messages-native-vllm.yaml`) / transformed
  (`messages-to-openai-vllm.yaml`)} × {deterministic `acceptEdits` /
  client-initiated auto-mode}. A fifth scenario covers the #1418 read-only
  planning turn on the native path: it asserts convergence and no
  reasoning-channel leak into the user-visible answer.
- **OpenAI Agents SDK.** The SDK's `Runner` drives a client-owned function-tool
  loop (model → tool → model) through `POST /v1/responses`.
  `tool_choice="required"` forces the tool call so a real model cannot flake
  the loop. Only `/v1/responses` crosses Praxis.
- **OpenAI SDK (Responses), 2.x and 3.x.** The full Responses SDK suite runs on
  both major SDK versions to catch protocol drift. Tests marked
  `real_inference` / `vllm_compat` run only in live (GPU) mode; the rest also
  run on PRs against the CPU simulator (`VLLM_TEST_BACKEND=simulator`).
- **OpenAI SDK (Chat Completions), selected-provider translation.** An OpenAI
  Chat Completions client is routed by `intelligent_route` to a selected
  Anthropic candidate and translated by
  `openai_chat_completions_to_anthropic[_stream]` to vLLM's native
  `/v1/messages`, then back to OpenAI shape. This is real-model coverage for
  PR #1736 (JSON, streaming, tool round-trip). A `messages` provider's overlay
  credential is `apikey`-only, while keyed vLLM wants `Authorization: Bearer`,
  so the core `credential_injection` filter injects the backend bearer (as in
  `messages-native-vllm.yaml`). The selected-provider `x-api-key` path is
  covered by the deterministic per-PR lane
  `integration.yaml::selected-anthropic-sdk`. Test:
  `test_selected_anthropic_chat_completions_vllm.py`.
- **CLI self-compaction (Codex and Claude Code).** Two tests prove a pinned CLI
  grows its context past its own window, compacts it itself, and keeps working
  through Praxis. They never call an SDK compact endpoint, inject a canned
  summary, or enable a Praxis compaction filter. For Codex the signal is the
  client's inline-summarization POST on the client→upstream wire
  (`codex_http::pinned_codex_compaction_crosses_context_window_over_http`); for
  Claude Code it is the stream-json `compact_boundary {trigger: auto}` event
  (`claude_code_vllm::pinned_claude_code_compaction_crosses_context_window_on_native_vllm`).
  See below for how task completion is graded on Qwen3-8B.

### Vision companion (`anthropic-vllm-vision.yaml`)

The vision workflow reuses the revision-pinned `vision` image from
[`vllm/images.json`](../../vllm/images.json), serving `Qwen3-VL-4B-Instruct`
with a generated backend bearer token. It runs two test commands:

| Command | What it proves |
| --- | --- |
| `uv run tests/integration/sdk/anthropic/test_anthropic_messages_vllm.py -s -k image` | Official Anthropic SDK image inputs reach the real model through both the native Messages and translated Chat Completions examples; generated red/blue PNGs are identified by color. |
| `uv run tests/integration/sdk/openai/test_responses_acceptance_gateway.py -s -k live_image` | The pinned upstream OpenResponses `image-input` template passes through the managed Responses-to-Chat profile, then the official OpenAI SDK sends the same image input and receives a completed response with nonempty output text. |

Both commands require live infrastructure via `PRAXIS_TEST_REQUIRE_LIVE=1`.
The OpenResponses step uses pinned Bun and a Praxis binary built with
`--no-default-features --features standard,openai-all,store-sqlite`. Only
`live_image` is selected from the gateway test file; its controlled backend and
timeout regression tests run in CPU translation CI.

The vision job does not run the upstream OpenResponses text, tool, compaction,
or WebSocket templates. The full-suite runner, its managed/passthrough
profiles, and the seven outstanding WebSocket cases are in
[Full OpenResponses acceptance](../conformance/openresponses-acceptance.md).

### Known model limitation: self-compaction task completion on Qwen3-8B

Hard assertions on every model: the pinned CLI performed its own
self-compaction. Codex's summarization request is observed crossing Praxis on
the wire. Claude Code's `compact_boundary {trigger: auto}` is local to the CLI
and does not cross Praxis, so the test observes that event and separately
verifies that the pre-compaction Read carrying the per-run marker reached
Praxis and returned successfully.

Qwen3-8B cannot reliably finish the task after it compacts. It may stop after
its summary, re-read the same ballast file until the deadline, or lose the
high-entropy marker in its summary and write the wrong value. A deterministic
scripted-provider integration test covers compaction-history rehydration and
forwarding separately. On Qwen3-8B the following are XFAIL, logged as an
`XFAIL (...)` line:

- **Codex.** After the hard wire-compaction and post-compaction continuation
  checks, the task-completion assertions (correct `result.txt`, the write
  ordered after compaction on the wire, the ordered JSONL trace, `./verify.sh`)
  are skipped when the non-candidate lane serves the exact alias
  `qwen3-8b`. A candidate dispatch never uses this fallback.
- **Claude Code.** The test checks for a post-compaction tool call. On
  Qwen3-8B a missing call is an expected XFAIL; an emitted call must follow
  compaction. The lane does not claim a post-compaction request crossed Praxis
  when none was emitted. The completion assertions (a post-compaction Edit of
  the uppercase marker into `result/value.txt`, then a passing `./verify.sh`)
  are skipped on `COMPACTION_XFAIL_MODEL`. The prompt steers a re-read of
  `source/value.txt` to recover the marker, but the oracle does not require
  that call.

On other models, including other Claude Code models and the Codex lane,
post-compaction continuation and task completion are hard requirements. Point
either test at a more capable model and the full oracle runs as hard
assertions. On the XFAIL model each lane uses a shorter deadline
(`COMPACTION_XFAIL_CHILD_TIMEOUT`, 180s) so a looping run ends soon after
compaction is captured.

To run only these two tests instead of the ~90-minute full suite, dispatch
`vllm-integration.yaml` with `run_compaction_only=true` **alone**. Do not also
pass `run_live_vllm=true`, which triggers the full suite. The dispatch
provisions the GPU runner and runs only the Codex and Claude Code compaction
steps.

### Qualifying a Codex candidate checkpoint

Manually dispatch `vllm-integration.yaml` with `run_codex_acceptance=true`
and every other `run_*` input false, including `run_compaction_only`. Set
`codex_model` to the Hugging Face checkpoint ID and
`codex_served_model_name` to a slash-free alias. This runs the Codex GPU job
without holding the candidate GPU for the Qwen3-8B full suite. The daily
Qwen3-8B nightly and labeled jobs keep their Qwen3-8B settings. Candidate
dispatches use the manifest's digest-pinned upstream vLLM image and download
the checkpoint at its resolved Hugging Face revision.

Set `gpu_instance_type`, `codex_tensor_parallel_size`, and
`codex_max_model_len` for the checkpoint and runner. The context length must
exceed Codex's 16,000-token test window. Select `codex_tool_call_parser` and
`codex_reasoning_parser` from the pinned vLLM version; `none` omits the
server flag. For `openai/gpt-oss-120b`, use the alias `gpt-oss-120b`, tool
parser `openai`, and reasoning parser `openai_gptoss`. For Gemma 4 and
Nemotron 3, select the exact checkpoint and matching parsers; these are
candidates, not claims of passing GPU runs.

The native test requires a completed `apply_patch` edit through
`openai_client_tool_compat` and positive reasoning-token usage in a completed
Codex turn. The compaction test requires Codex's own compaction request,
continued traffic through Praxis, and a correct result that passes
verification. `codex_skip_translated=true` skips only the translated Chat case
and its preflight; both native gates still run.

Download `qualification.json` from the `vllm-qualification` artifact. Under
`suites.codex_acceptance`, `cases` holds the scenario results and
`provenance` holds the checkpoint and alias, the Hugging Face commit SHA
pinned for candidate runs, the pinned vLLM image, parser and GPU settings,
Codex version, and ordered Praxis filter names by case. The report merges only
Codex evidence matching its run ID, attempt, and checkout SHA. On a Codex-only
run, the top-level `not_requested` status refers to the full Responses suite,
not the Codex cases.

#### Opt in to weekly gpt-oss-120b qualification

A repository maintainer can set the Actions variable
`AWS_GPU_FALLBACK_SUBNET_IDS` to up to two whitespace-separated subnet IDs in
launch order. Use approved subnets in distinct availability zones, in the same
region and VPC as the primary subnet, reachable with the configured runner
role and security group. Codex candidate runs try the primary subnet first,
then the fallbacks if EC2 rejects the launch. Unset, the runner uses only the
primary subnet. The instance type does not change during fallback.

Set `CODEX_GPT_OSS_120B_SCHEDULE=true` to enable the **Codex Candidate GPU
Qualification** workflow on `main` in `praxis-proxy/ai`, Mondays at 07:42
UTC. It runs `openai/gpt-oss-120b` on `p5.4xlarge` with tensor parallel size
1 and a 16,384-token context, native Responses and compaction only. It shares
the runner cleanup and qualification artifact of a manual run, but does not
count as a Qwen nightly or release Responses qualification. In
`qualification.json`, check that `suites.codex_acceptance` records success for
both `codex_native` and `codex_compaction` for the intended checkpoint. If the
launch fails in every configured subnet, the candidate stays unqualified and
the next scheduled run tries again. Unset the variable when weekly runs are no
longer needed.

### Known model limitation: server-side web search on Qwen3-8B

The Claude Code server-side web-search test
(`claude_code_vllm::pinned_claude_code_resolves_server_side_web_search_through_tavily`)
drives the real `claude` binary, which sends `WebSearch` as a custom client
tool with `tool_choice: null` and never forces it. Qwen3-8B does not reliably
finish that loop. Two failures were seen on live Qwen3-8B, both model limits
rather than proxy bugs:

- it never calls `WebSearch`, so the managed `anthropic_web_search` loop runs
  no search and the turn returns an empty answer; or
- it calls `WebSearch`, Praxis resolves a real Tavily search server-side (valid
  exchange, tool_use suppressed), the turn exits `success`, and the model
  writes no final answer from the results (`result:"\n\n"`).

The forced-tool-choice SDK web-search tests prove Praxis's server-side Tavily
resolution on every run. They set `tool_choice={type:tool,name:WebSearch}` so
a small model opens the loop:

- `test_live_tavily_web_search_returns_real_sources` (OpenAI Responses and
  Anthropic SDK), and
- the `ANTHROPIC_STUBBED_WEB_SEARCH` / `ANTHROPIC_CREDENTIALED_WEB_SEARCH`
  acceptance scenarios.

As in the compaction lane, everything the proxy owns is a hard assertion and
only the model's completion is XFAIL. The `check_server_side_web_search`
oracle classifies each failure (`WebSearchFailure`):

- **Proxy regression: hard on every model, checked first.** A leaked
  `WebSearch` tool_use (the managed loop must suppress it), an `is_error` flag
  on an otherwise successful turn, or a Tavily exchange that ran but was
  malformed or failed: non-200, the resolved key missing from the forwarded
  `Authorization: Bearer` header (issue #1389), an empty query, or empty or
  relative-URL results. The loopback relay
  (`tests/utils/src/tavily_relay.rs`) forwards and records the `Authorization`
  header, so a dropped credential fails here instead of looking like a model
  limit.
- **Model incomplete: XFAIL on `WEB_SEARCH_XFAIL_MODEL` (`qwen3-8b`), hard on
  a capable model.** The turn did not converge, the model never called
  `WebSearch` (no Tavily exchange), or it got real results and gave no usable
  answer. On `qwen3-8b` these are skipped with an `XFAIL (...)` line, and
  the child must still run to completion cleanly through Praxis.

Having the proxy force `tool_choice` for the managed web-search tool would let
this lane drop the XFAIL. That is tracked separately as a possible filter
feature.

## Reusing published images

Neither nightly rebuilds the model-baked image by default. Each job serving a
baked model calls
[`resolve-vllm-gpu-image`](../../.github/actions/resolve-vllm-gpu-image/action.yml),
which pulls the published tag and keeps it only when the image's provenance
labels match the checkout: the same `vllm/Containerfile` hash, model ID, and
pinned model revision. On a mismatch, an unreachable registry, or a missing
tag, it builds from source. A nightly with no changes skips a multi-gigabyte
rebuild, a PR that edits the `Containerfile` or repins a model is tested
against the image it describes, and a lagging or failed publish falls back to
a local build instead of breaking the run or qualifying a stale backend.

In the main suite, each Qwen3-8B job goes through
[`prepare-vllm-gpu-image`](../../.github/actions/prepare-vllm-gpu-image/action.yml),
which builds the image if resolution fails, including when GHCR login fails.
The jobs run one after another on the same instance, so client acceptance jobs
first look for an image an earlier job prepared there and serve it without
pulling. After a local build this matters, because the registry still holds
the stale tag. An acceptance-only dispatch (`run_claude_acceptance`,
`run_compaction_only`, and the other client inputs) skips the full suite, so
its first acceptance job pulls the published image or, when none matches the
checkout (for example right after a vLLM bump), builds it. Only that build
gets the Hugging Face token; no serving container receives it, and no job
downloads weights at run time. The exception is a
[Codex candidate dispatch](#qualifying-a-codex-candidate-checkpoint): its
serving container gets the token and downloads the checkpoint.

The qualification report records which image each lane served; see
[image provenance fields](vllm-qualification.md#report-version-and-interpretation).

## Jobs in the main suite (`vllm-integration.yaml`)

- `changes`: path-filter gate.
- `vllm-responses`, `vllm-responses-postgres`: fast CPU coverage on PRs
  against `llm-d-inference-sim`.
- `vllm-live-cpu`: CPU coverage on PRs against real vLLM serving
  `Qwen/Qwen3-0.6B`.
- `gpu-start-runner` / `gpu-stop-runner`: provision and tear down the GPU EC2
  instance.
- `vllm-gpu-full-suite`: resolves the model-baked image (pulling the published
  build, or building from source when it does not match the checkout) and runs
  the complete live Responses SDK suite (SDK 2.x + 3.x) plus the OpenAI Agents
  SDK loop against real Qwen3-8B with a PostgreSQL store.
- `vllm-gpu-claude-acceptance`: pinned Claude Code over native and translated
  Anthropic paths (acceptEdits + auto mode), including the #1418 planning-turn
  regression guard, in a locked-down network namespace. It also runs the
  selected-provider OpenAI-chat → `/v1/messages` lane host-side against the
  same keyed vLLM (step `selected_anthropic_openai_chat`, recorded in the
  `claude_acceptance` qualification suite).
- `vllm-gpu-codex-acceptance`: pinned Codex over native Responses and
  translated Chat paths, with the vLLM launch flags pinned in
  `tests/integration/fixtures/codex-cli/pin.toml`.
- `vllm-gpu-opencode-acceptance`: pinned OpenCode over native Chat
  Completions: one text turn and one tool-call round trip for each of the
  pinned 1.x and 2.x clients, against one shared vLLM container in its own
  locked-down network namespace. Each major reports separate cases
  (`opencode_text`, `opencode_tool_call`, `opencode_v2_text`,
  `opencode_v2_tool_call`).
- `vllm-gpu-pi-acceptance`: pinned pi over the same native Chat Completions
  path, with the documented `models.json`: one text turn and one tool-call
  round trip (`pi_text`, `pi_tool_call`). It runs its own vLLM container from
  the OpenCode backend pins and launches the client in its own locked-down
  network namespace.
- `pins`: parses `tests/integration/fixtures/claude-code-cli/pin.toml`,
  `tests/integration/fixtures/codex-cli/pin.toml`,
  `tests/integration/fixtures/opencode-cli/pin.toml`,
  `tests/integration/fixtures/opencode-cli-v2/pin.toml`, and
  `tests/integration/fixtures/pi-cli/pin.toml`, and cross-validates them: each
  OpenCode manifest against its declared major, pi against the OpenCode backend
  pins it shares, and every `[vllm]` launch against the baked weights of its
  `vllm/images.json` entry. It exports every value the acceptance jobs use, so
  those jobs hardcode no version strings.
- `gpu-qualification-summary`: writes `qualification.json` and the Actions
  summary even if provisioning fails.

## Where the test code lives

The SDK test directories have no README; each file's module docstring has the
details.

- `tests/integration/sdk/openai/test_openai_responses_vllm.py`: core live
  Responses suite (`VLLM_TEST_BACKEND=live` vs `simulator`).
- `tests/integration/sdk/openai/test_openai_agents_sdk.py`: OpenAI Agents SDK
  loop (deterministic vs live GPU mode).
- `tests/integration/sdk/anthropic/test_anthropic_messages_vllm.py`: Anthropic
  SDK → Praxis → real vLLM Messages acceptance.
- `tests/integration/sdk/openai/test_responses_acceptance_gateway.py`:
  OpenResponses and OpenAI SDK image acceptance in the vision companion
  (`-k live_image`); controlled image/compaction and timeout regression
  coverage in CPU translation CI (`-k 'not live_image'`).
- `tests/integration/sdk/anthropic/test_anthropic_web_search_vllm.py`:
  Anthropic web search.
- `tests/integration/sdk/openai/test_selected_anthropic_chat_completions_vllm.py`:
  the selected-provider Chat Completions lane above. Includes a deterministic
  overlay-digest check and a scripted-backend credential-shim test that run
  without a GPU.
- `tests/integration/tests/suite/claude_code_vllm.rs`: Claude Code acceptance
  (native vs transformed path diagram, scenarios incl. #1418).
- `tests/integration/tests/suite/codex_http.rs`: Codex acceptance and
  pin-update instructions.
- `tests/integration/tests/suite/opencode_vllm.rs`: OpenCode acceptance
  (`opencode_vllm_options_headers_auth_completes_text_turn`,
  `opencode_vllm_tool_call_round_trip_writes_hash_verified_marker`). The
  deterministic CPU lane is `suite/opencode.rs`, run by the `opencode-cpu`
  job in `integration.yaml`.
- `tests/integration/tests/suite/opencode_v2_vllm.rs`: OpenCode 2.x
  acceptance (`opencode_v2_vllm_plugin_auth_completes_text_turn`,
  `opencode_v2_vllm_tool_call_round_trip_writes_hash_verified_marker`). The
  deterministic CPU lane is `suite/opencode_v2.rs`, run by the
  `opencode-v2-cpu` job in `integration.yaml`.
- `tests/integration/tests/suite/pi_vllm.rs`: pi acceptance
  (`pi_vllm_documented_config_completes_text_turn`,
  `pi_vllm_tool_call_round_trip_writes_hash_verified_marker`). The
  deterministic CPU lane is `suite/pi.rs`, run by the `pi-cpu` job in
  `integration.yaml`.
- `tests/integration/tests/suite/live_vllm.rs`: namespace, egress-probe, and
  example-patching scaffolding shared by the live CLI suites.

## Running it locally

Reproduce the core suite on a local GPU (see
[cli-vllm-through-praxis.md](cli-vllm-through-praxis.md) for serving vLLM):

```console
cargo build -p praxis-ai-proxy --features full,openai-all,store-sqlite
VLLM_MODEL=Qwen/Qwen3-8B VLLM_TEST_BACKEND=live \
  uv run tests/integration/sdk/openai/test_openai_responses_vllm.py
```

## Deeper references

- [Full OpenResponses acceptance](../conformance/openresponses-acceptance.md):
  the separately invoked upstream suite, test profiles, and scope of the
  recorded live results.
- [vllm-qualification.md](vllm-qualification.md): the `qualification.json`
  schema, report fields, pass/fail rules, and release selection.
- [cli-vllm-through-praxis.md](cli-vllm-through-praxis.md): running Codex,
  Claude Code, OpenCode, and pi through Praxis and vLLM; the reasoning-flag
  regression, the headless `opencode run` and `pi --print` forms, each
  OpenCode major's credential recipes, and how the 2.x plugin and pi's
  credential recipe behave when they fail.
- [`vllm/README.md`](../../vllm/README.md): GPU container build, run, health
  check, and hardware/model rationale.
- [llmd-integration-testing.md](llmd-integration-testing.md): adjacent CPU-side
  `llm-d-inference-sim` tests.
