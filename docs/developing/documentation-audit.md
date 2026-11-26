# Operator documentation support audit

Reviewed **2026-11-13** against main
`56dc39a25d7f44da0467c28f0e85bfa4ca220130` (the checkout matched fetched
`origin/main` before edits). This page records the evidence behind the operator
documentation and the work that remains. Readers start at the
[client/provider selector](../getting-started/choose-client-and-provider.md).
The audit extended the onboarding and client-provider guides from #1976 rather
than adding competing guides.

## Evidence and support boundaries

The audit checked [server feature wiring](../../server/Cargo.toml),
[filter registration](../../filters/src/register.rs), API implementation modules,
complete example YAML, functional example tests, SDK tests, and the
[generated translation matrix](../conformance/translation-matrix.md).
Registration shows a filter is available, not that it conforms to the
protocol. Fixture replay shows behavior against the recorded or synthetic
backend, not a live provider/model. A test gated on infrastructure can skip and
prove nothing live.

| Capability | Source and existing test evidence | Operator boundary |
| --- | --- | --- |
| Native Chat/Responses forwarding | [Build from source](../getting-started/build-from-source.md), [HTTP passthrough config](../../examples/configs/openai/responses/http-passthrough.yaml), [server feature graph](../../server/Cargo.toml) | Standard routing can forward the upstream's APIs without local managed filters; no gateway tool execution follows from that |
| Native Anthropic Messages | [Native config](../../examples/configs/anthropic/messages-native-vllm.yaml), [functional tests](../../tests/integration/tests/suite/examples/anthropic_messages_native_vllm.rs), [SDK lanes](../../tests/integration/sdk/anthropic/test_anthropic_sdk_compat.py) | Provider/model owns semantics; SDK v1/v2 credential/version differences matter |
| Messages to Chat | [Request/finite adapter](../../apis/src/anthropic/messages_to_chat_completions), [SSE adapter](../../apis/src/anthropic/messages_to_chat_completions_stream), [functional tests](../../tests/integration/tests/suite/examples/anthropic_messages_to_openai_vllm.rs) | Standard adapters; thinking/cache opt-ins lose semantics; document/server-tool/error-result rejection; count_tokens is not translated |
| Chat to selected Anthropic | [Adapter](../../apis/src/anthropic), [selected-provider example test](../../tests/integration/tests/suite/examples/intelligent_route_selected_anthropic_legacy.rs), [OpenAI SDK test](../../tests/integration/sdk/openai/test_selected_anthropic_chat_completions.py) | Requires trusted selected profile and explicit permission; ignored fields and per-version contract documented separately |
| Azure Chat adaptation | [Example test](../../tests/integration/tests/suite/examples/azure_translation.rs), [adapter reference](../filters/providers/openai_chat_completions_to_azureai_chat_completions.md) | Standard adapter; Entra acquisition is optional experimental; safety annotation loss remains in matrix |
| Vertex Gemini translation | [Example test](../../tests/integration/tests/suite/examples/vertex_gemini.rs), [adapter reference](../filters/providers/openai_chat_completions_to_vertexai_gemini.md) | Standard adapter; GCP acquisition optional experimental; strict schema and accounting limitations |
| Vertex Anthropic | [Example test](../../tests/integration/tests/suite/examples/vertex_anthropic.rs), [reference](../filters/providers/anthropic_messages_to_vertexai_anthropic.md) | Experimental adapter; selected cluster metadata, endpoint/model adaptation, authorized GCP identity required |
| Bedrock Converse | [Implementation](../../apis/src/bedrock/converse), [example test](../../tests/integration/tests/suite/examples/bedrock.rs), [complete config](../../examples/configs/bedrock/chat-completions-to-converse.yaml) | Standard translator and default signer; signing is not itself API support; history/reasoning/cache gaps remain |
| Managed Responses, continuation and Conversations | [Responses modules](../../apis/src/openai/responses), [Conversations](../../apis/src/openai/conversations), [SDK tests](../../tests/integration/sdk/openai/test_openai_conversations.py), [recipe](../guides/managed-responses.md) | Experimental opt-ins plus selected store backend for persistence; managed restrictions differ from passthrough |
| MCP execution and server-owned tools | [Loop config](../../examples/configs/openai/responses/agentic-loop.yaml), [MCP outbound tests](../../tests/integration/tests/suite/examples/openai_mcp_outbound_chain.rs), [retained-payload contract](../architecture/responses-retained-payload.md) | Discovery, dispatch, loop continuation, approval, and credential boundaries must all be composed; client function calls remain client-owned |
| MCP broker and A2A routing | [Protocol architecture](../architecture/agentic-protocols.md), [broker config](../../examples/configs/mcp-stateless-broker.yaml), [A2A affinity config](../../examples/configs/a2a-task-routing.yaml) | Protocol routing/brokering does not imply Responses tool execution or a hosted agent implementation |
| Aliases and provider routing | [Model mapping tests](../../tests/integration/tests/suite/examples/model_to_provider.rs), [dynamic routing contract](../filters/routing/intelligent_route.md) | Alias, dynamic selection, endpoint retries, and affinity are separate mechanisms; affinity is process-local |
| Guardrails | [Implementation](../../filters/src/guardrails), [functional tests](../../tests/integration/tests/suite/examples/guardrails.rs), [response tests](../../tests/integration/tests/suite/examples/guardrails_response.rs) | Operation identity required; fail-closed screening; Chat-only redaction; response streaming restrictions |
| Tokens, limits and metering | [Usage implementation](../../filters/src/token_usage), [count tests](../../tests/integration/tests/suite/examples/token_counting.rs), [tenant metrics tests](../../tests/integration/tests/suite/examples/token_usage_metrics.rs), [metering reference](../filters/control/external_metering.md) | Observed usage is not enforcement; token ceilings, budgets and inflight tracking are optional experimental; body counts arrive after headers |
| Credentials, identity and enrichment | [Registration](../../filters/src/register.rs), [credential boundaries](../guides/README.md#credentials), [prompt enrichment](../filters/core/prompt_enrich.md) | Header capture is not authentication; distinguish cluster credentials, selected-provider credentials, and destination-bound callout secrets |

The matrix statuses are `translated`, `passthrough`, `lossy`, `rejected`,
`n/a`, and `gap`. Its known gaps include accepted-but-incomplete behavior and
must stay visible when a provider recipe is promoted. Examples with local
stubs, placeholder providers, or a broad path route do not qualify a provider.
The pinned GPU/model lanes in [Anthropic evidence](../anthropic-messages.md)
were not rerun for this audit.

## Documentation changes

- Reorganized the landing page around onboarding, APIs, providers, routing,
  protection, tools/state, operations, and reference/development.
- Extended the selector with selected Anthropic, Bedrock, Conversations,
  explicit mode/build/auth distinctions, and compatibility status.
- Added Chat Completions to the hosted quickstart, and a cloud adapter runbook
  that completes the existing examples with TLS, credentials, and startup.
- Reused the native/translated Anthropic and model-mapping examples in recipes.
- Added complete local recipes for guardrails with usage and for an
  experimental managed Responses/MCP/SQLite lab.
- Moved Rust hosting/registration material to [development](filter-hosts.md)
  and added an [operations path](../operations/README.md).

## Verification for this change

These checks passed against the reviewed source plus the documentation edits:

- `cargo build -p praxis-ai-proxy` (standard) and
  `cargo build -p praxis-ai-proxy --features openai-all,store-sqlite,basic-auth-filter`,
  plus a third build that added `vertex-anthropic-filter` to that optional
  set. Separate copies of the binaries were used to check feature availability.
- `make doc`, with warnings denied by the repository target.
- `cargo xtask lint-filter-docs`, `lint-example-tests`, `lint-markdown-links`,
  `sync-example-readme`, `sync-inference-readme`, `sync-responses-readme`,
  `sync-flow-visualizers`, `check-inference`, and `translation-matrix`.
  These check reference and manifest consistency; they do not run every
  example or replay every inference fixture.
- `git diff --check`, plus a local-target/fragment check over new and changed
  Markdown, because the repository link linter enumerates tracked files and
  misses new, unstaged ones.

Configuration validation ran the binaries above with placeholder secrets and,
where applicable, YAML extracted from the documentation:

| Binary | Configurations accepted by `--validate` |
| --- | --- |
| Standard | Hosted quickstart, combined guardrails/usage recipe, model-to-provider example, Azure Chat example, Vertex Gemini example, Bedrock Converse example |
| Optional build above | Native and translated credential-isolated Anthropic examples, managed Responses/MCP/SQLite recipe |
| Optional build plus `vertex-anthropic-filter` | Original Vertex Anthropic example and its cloud-runbook adaptation |

The standard binary rejected the managed recipe with an unknown-filter error,
confirming that the optional build is necessary. The cloud-runbook adaptations
for Azure, Vertex Gemini, Bedrock, and Vertex Anthropic, including the added
TLS/SNI/Host and credential settings, passed validation with placeholder
secrets and the appropriate binary. They still need real credentials,
permissions, model access, and live provider tests.

A temporary local HTTP recording harness ran the documented configurations
against controlled stubs and asserted on responses and captured requests:

- Static alias selected port 9001 and rewrote the model; an unmatched model
  selected port 9002 unchanged.
- Guardrail pass reached inference; a blocked verdict returned 403 with no new
  inference request. The inference bearer was absent from NeMo calls, and
  provider usage produced prompt/output token metrics.
- Native and translated Anthropic finite responses reached the proper backend
  path and returned Messages JSON. Missing gateway auth returned 401; caller
  `x-api-key` and gateway Basic credentials did not reach the backend, which
  received only its injected bearer.
- The managed recipe served local Conversation creation, stored a finite
  Responses result (confirmed in SQLite), and rejected Code Interpreter before
  backend contact.
- MCP discovery produced a tool approval request without executing a call.
  Resuming with approval executed exactly one tool call, then returned the
  final model answer. MCP requests did not receive the inference bearer.

These checks used the exact documented ports and filter chains. Stub verdicts
say nothing about real NeMo rail behavior, and synthetic model output says
nothing about a model's ability to select tools. SSE runtime paths, approval
denial/replay, and broader protocol contracts rest on the source and test
evidence linked above; the harness did not rerun them. No credentialed
hosted-provider request, live MCP service, GPU run, or production sizing
exercise was performed. Temporary proxies and stubs were stopped afterwards.

## Remaining gaps and separate blockers

- **Bedrock generated-reference omission:** `openai_chat_completions_to_bedrock_converse`
  is registered and exercised by tests, but the generated filter inventory has
  no page for it, and `cargo xtask lint-filter-docs` passes anyway. Fix
  generator discovery/source metadata in a separate change; do not hand-edit the
  generated reference. Until then the docs link the implementation,
  configuration, and compatibility evidence instead of an exact-settings page.
- **Cloud account provisioning and qualification:** the [provider runbook](../guides/providers/README.md)
  completes gateway configuration and requests around existing examples. It
  assumes cloud accounts, regional permissions, model/deployment access, and a
  supported credential source; those prerequisites and live cloud validation
  were not checked here.
- **Runtime product gaps:** generated translation-matrix gaps, response-phase
  streaming guardrails, and body-to-response-header token timing remain visible.
  No Rust behavior was changed to conceal or fix them.
- **Operational qualification:** this audit provides no new sustained-load
  sizing, autoscaling, cross-replica affinity, or cloud/model certification.

## Development versus v0.10.0

The release tag `v0.10.0` resolves to
`f96d0114a995765793b1fb33f86f94ecc298eb08`. This audit is based on later main,
with later onboarding, provider, translation, and Conversations changes, and
edited no released snapshot. New recipes, navigation, and current-main
compatibility statements belong in **Development** until a release containing
them is published and imported by the website. Do not backport the selector's
support claims to v0.10.0 without auditing that tag's code and binary
separately.
