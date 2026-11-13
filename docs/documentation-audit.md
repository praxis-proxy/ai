# Operator documentation support audit

Reviewed **2026-11-13**, against main
`56dc39a25d7f44da0467c28f0e85bfa4ca220130` (the checkout matched fetched
`origin/main` before edits). This document records evidence and remaining work;
the [client/provider selector](client-provider-paths.md) is the reader entry
point. The onboarding/client-provider guides from #1976 were already present
and are extended rather than replaced with competing guides.

## Evidence and support boundaries

The audit checked [server feature wiring](../server/Cargo.toml),
[filter registration](../filters/src/register.rs), API implementation modules,
complete example YAML, functional example tests, SDK tests, and the
[generated translation matrix](conformance/translation-matrix.md).
Registration establishes availability, not protocol conformance. Fixture replay
establishes behavior against the recorded or synthetic backend, not a live
provider/model. A test gated on infrastructure can skip and prove nothing live.

| Capability | Source and existing test evidence | Operator boundary |
| --- | --- | --- |
| Native Chat/Responses forwarding | [Quickstart](quickstart.md), [HTTP passthrough config](../examples/configs/openai/responses/http-passthrough.yaml), [server feature graph](../server/Cargo.toml) | Standard routing can forward the upstream's APIs without local managed filters; no gateway tool execution follows from that |
| Native Anthropic Messages | [Native config](../examples/configs/anthropic/messages-native-vllm.yaml), [functional tests](../tests/integration/tests/suite/examples/anthropic_messages_native_vllm.rs), [SDK lanes](../tests/integration/sdk/anthropic/test_anthropic_sdk_compat.py) | Provider/model owns semantics; SDK v1/v2 credential/version differences matter |
| Messages to Chat | [Request/finite adapter](../apis/src/anthropic/messages_to_chat_completions), [SSE adapter](../apis/src/anthropic/messages_to_chat_completions_stream), [functional tests](../tests/integration/tests/suite/examples/anthropic_messages_to_openai_vllm.rs) | Standard adapters; thinking/cache opt-ins lose semantics; document/server-tool/error-result rejection; count_tokens is not translated |
| Chat to selected Anthropic | [Adapter](../apis/src/anthropic), [selected-provider example test](../tests/integration/tests/suite/examples/intelligent_route_selected_anthropic_legacy.rs), [OpenAI SDK test](../tests/integration/sdk/openai/test_selected_anthropic_chat_completions.py) | Requires trusted selected profile and explicit permission; ignored fields and per-version contract documented separately |
| Azure Chat adaptation | [Example test](../tests/integration/tests/suite/examples/azure_translation.rs), [adapter reference](filters/openai_chat_completions_to_azureai_chat_completions.md) | Standard adapter; Entra acquisition is optional experimental; safety annotation loss remains in matrix |
| Vertex Gemini translation | [Example test](../tests/integration/tests/suite/examples/vertex_gemini.rs), [adapter reference](filters/openai_chat_completions_to_vertexai_gemini.md) | Standard adapter; GCP acquisition optional experimental; strict schema and accounting limitations |
| Vertex Anthropic | [Example test](../tests/integration/tests/suite/examples/vertex_anthropic.rs), [reference](filters/anthropic_messages_to_vertexai_anthropic.md) | Experimental adapter; selected cluster metadata, endpoint/model adaptation, authorized GCP identity required |
| Bedrock Converse | [Implementation](../apis/src/bedrock/converse), [example test](../tests/integration/tests/suite/examples/bedrock.rs), [complete config](../examples/configs/bedrock/chat-completions-to-converse.yaml) | Standard translator and default signer; signing is not itself API support; history/reasoning/cache gaps remain |
| Managed Responses, continuation and Conversations | [Responses modules](../apis/src/openai/responses), [Conversations](../apis/src/openai/conversations), [SDK tests](../tests/integration/sdk/openai/test_openai_conversations.py), [recipe](recipes/managed-responses.md) | Experimental opt-ins plus selected store backend for persistence; managed restrictions differ from passthrough |
| MCP execution and server-owned tools | [Loop config](../examples/configs/openai/responses/agentic-loop.yaml), [MCP outbound tests](../tests/integration/tests/suite/examples/openai_mcp_outbound_chain.rs), [retained-payload contract](architecture/responses-retained-payload.md) | Discovery, dispatch, loop continuation, approval, and credential boundaries must all be composed; client function calls remain client-owned |
| MCP broker and A2A routing | [Protocol architecture](architecture/agentic-protocols.md), [broker config](../examples/configs/mcp-stateless-broker.yaml), [A2A affinity config](../examples/configs/a2a-task-routing.yaml) | Protocol routing/brokering does not imply Responses tool execution or a hosted agent implementation |
| Aliases and provider routing | [Model mapping tests](../tests/integration/tests/suite/examples/model_to_provider.rs), [dynamic routing contract](filters/intelligent_route.md) | Alias, dynamic selection, endpoint retries, and affinity are separate mechanisms; affinity is process-local |
| Guardrails | [Implementation](../filters/src/guardrails), [functional tests](../tests/integration/tests/suite/examples/guardrails.rs), [response tests](../tests/integration/tests/suite/examples/guardrails_response.rs) | Operation identity required; fail-closed screening; Chat-only redaction; response streaming restrictions |
| Tokens, limits and metering | [Usage implementation](../filters/src/token_usage), [count tests](../tests/integration/tests/suite/examples/token_counting.rs), [tenant metrics tests](../tests/integration/tests/suite/examples/token_usage_metrics.rs), [metering reference](filters/external_metering.md) | Observed usage is not enforcement; token ceilings, budgets and inflight tracking are optional experimental; body counts arrive after headers |
| Credentials, identity and enrichment | [Registration](../filters/src/register.rs), [credential boundaries](recipes/README.md#credentials), [prompt enrichment](filters/prompt_enrich.md) | Header capture is not authentication; distinguish cluster credentials, selected-provider credentials, and destination-bound callout secrets |

The matrix explicitly distinguishes `translated`, `passthrough`, `lossy`,
`rejected`, `n/a`, and `gap`. Its known gaps include accepted-but-incomplete
behavior and must remain visible when promoting a provider recipe. Examples
with local stubs, placeholder providers, or a broad path route are not complete
provider qualifications. See [Anthropic evidence](anthropic-messages.md) for
existing pinned GPU/model lanes; they were not rerun as part of this audit.

## Documentation changes

- Reorganized the landing page around onboarding, APIs, providers, routing,
  protection, tools/state, operations, and reference/development.
- Extended the existing selector to include selected Anthropic, Bedrock,
  Conversations, explicit mode/build/auth distinctions, and compatibility status.
- Added Chat Completions to the existing hosted quickstart, and a cloud adapter
  runbook completing the existing examples with TLS, credentials, and startup.
- Reused the native/translated Anthropic and model-mapping examples in recipes.
- Added full local composition recipes for guardrails with usage and for an
  experimental managed Responses/MCP/SQLite lab.
- Moved Rust hosting/registration material to [development](developing/filter-hosts.md).
- Added an [operations path](operations.md).

## Verification for this change

The following checks passed against the reviewed source plus documentation edits:

- `cargo build -p praxis-ai-proxy` (standard) and
  `cargo build -p praxis-ai-proxy --features openai-all,store-sqlite,basic-auth-filter`.
  A third build added `vertex-anthropic-filter` to that optional feature set.
  Separate copies of the resulting binaries were used to check feature availability.
- `make doc`, with warnings denied by the repository target.
- `cargo xtask lint-filter-docs`, `lint-example-tests`, `lint-markdown-links`,
  `sync-example-readme`, `sync-inference-readme`, `sync-responses-readme`,
  `sync-flow-visualizers`, `check-inference`, and `translation-matrix`.
  These are reference/manifest consistency checks, not execution of all examples
  or replay of every inference fixture.
- `git diff --check`, plus a supplemental local-target/fragment check over new
  and changed Markdown. The repository link linter enumerates tracked files,
  so it does not by itself check new, unstaged files.

Configuration validation used the actual binaries above, with placeholder
secrets and YAML extracted from the documentation where applicable:

| Binary | Configurations accepted by `--validate` |
| --- | --- |
| Standard | Hosted quickstart, combined guardrails/usage recipe, model-to-provider example, Azure Chat example, Vertex Gemini example, Bedrock Converse example |
| Optional build above | Native and translated credential-isolated Anthropic examples, managed Responses/MCP/SQLite recipe |
| Optional build plus `vertex-anthropic-filter` | Original Vertex Anthropic example and its cloud-runbook adaptation |

The standard binary also rejected the managed recipe with an unknown-filter
error, confirming that the optional build is necessary. The cloud-runbook
adaptations for Azure, Vertex Gemini, Bedrock, and Vertex Anthropic also passed validation with placeholder secrets and the appropriate
binary, including the added TLS/SNI/Host and credential settings. These checks
still need real credentials, permissions, model access, and live provider tests.

A temporary local HTTP recording harness exercised the documented configurations
against controlled stubs, with assertions on responses and captured requests:

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

These local checks used the exact documented ports and filter chains. Stub
verdicts do not establish real NeMo rail behavior, and synthetic model output
does not establish a model's ability to select tools. SSE runtime paths,
approval denial/replay, and broader protocol contracts rely on the linked
existing source/test evidence here; they were not rerun in this harness.
No credentialed hosted-provider request, live MCP service, GPU run, or
production sizing exercise was performed. Temporary proxies and stubs were
stopped after verification.

## Remaining gaps and separate blockers

- **Bedrock generated-reference omission:** `openai_chat_completions_to_bedrock_converse`
  is registered and exercised by tests, but the generated filter inventory has
  no corresponding page. `cargo xtask lint-filter-docs` passes with this omission.
  Fix generator discovery/source metadata in a separate change; do not hand-edit
  the generated reference. This documentation links the implementation,
  configuration, and compatibility evidence without inventing an exact-settings page.
- **Cloud account provisioning and qualification:** the [provider runbook](recipes/providers.md)
  completes gateway configuration and requests around existing examples. It
  assumes cloud accounts, regional permissions, model/deployment access, and
  a supported credential source. Those external prerequisites and live cloud
  validation remain outside the checks run here.
- **Runtime product gaps:** generated translation-matrix gaps, response-phase
  streaming guardrails, and body-to-response-header token timing remain visible.
  No Rust behavior was changed to conceal or fix them.
- **Operational qualification:** this audit provides no new sustained-load
  sizing, autoscaling, cross-replica affinity, or cloud/model certification.

## Development versus v0.10.0

The release tag `v0.10.0` resolves to
`f96d0114a995765793b1fb33f86f94ecc298eb08`. This change is based on later main,
including later onboarding, provider, translation, and Conversations changes.
No released snapshot was edited. New recipes/navigation and current-main
compatibility statements belong in **Development** until a release containing
them is published and imported by the website. Do not backport this selector's
support claims to v0.10.0 without auditing that tag's code and binary separately.
