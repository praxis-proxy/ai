# Praxis AI Documentation

These pages describe **Development (current main)**. Build features and
compatibility can differ in a released version, so use the documentation
snapshot that matches your binary or image.

## Getting started

- [Quickstart](getting-started/quickstart.md): run the published container
  image and forward your first OpenAI Responses request.
- [Choose a client and provider](getting-started/choose-client-and-provider.md):
  protocol, credentials, feature selection, and compatibility limits in one table.
- [Build from source](getting-started/build-from-source.md): build, validate,
  and forward Responses and Chat Completions requests from a checkout.

## Guides

Each of the [guides](guides/README.md) has prerequisites, a complete
configuration, verification, and cleanup. [Credential boundaries](guides/README.md#credentials)
apply to all of them.

- Client APIs: [Anthropic Messages](guides/anthropic.md), native and
  translated; [experimental managed Responses and tools](guides/managed-responses.md),
  including [Conversations and state](guides/managed-responses.md#conversations-and-state).
- Route requests: [model aliases and provider routing](guides/routing.md).
- Connect providers: [Azure OpenAI](guides/providers/azure-openai.md),
  [Vertex AI Gemini](guides/providers/vertex-gemini.md),
  [Vertex AI Anthropic](guides/providers/vertex-anthropic.md), and
  [Amazon Bedrock](guides/providers/bedrock.md), with
  [shared provider prerequisites](guides/providers/README.md).
- Protect and control: [screen requests with guardrails](guides/guardrails.md),
  [observe and meter token usage](guides/token-usage.md), and
  [limit token usage](guides/token-limits.md).

## Concepts and architecture

- [Anthropic Messages API](anthropic-messages.md): native forwarding,
  translation, SDK versions, and backend qualification.
- [AI inference architecture](architecture/ai-inference.md) and
  [routing facts](architecture/routing-facts.md): which component owns each fact.
- [MCP and A2A](architecture/agentic-protocols.md): protocol routing and
  brokering are distinct from model-driven execution.
- [Outbound callout security](architecture/outbound-callouts.md).
- [Response store](architecture/response-store.md) and
  [retained-payload budget](architecture/responses-retained-payload.md).
- [OpenTelemetry](architecture/opentelemetry.md).
- [Cryptographic inventory](architecture/cryptographic-inventory.md) and
  [PostgreSQL cryptographic boundary](architecture/postgres-cryptographic-boundary.md).

## Operations

- [Operations checklist and troubleshooting](operations/README.md): deployment,
  configuration validation, observability, reload, and upgrades.
- [Capacity planning](operations/capacity-planning.md): file descriptors and admission.
- [FIPS 140-3 deployment](operations/fips.md).
- Response store maintenance: [schema migration](operations/store/schema-migration.md)
  and [legacy item position repair](operations/store/legacy-item-position-repair.md).
- [Migrating to 0.2](operations/upgrades/migrating-to-0.2.md).
- [Security policy](../.github/SECURITY.md).

## Reference

- Filters: [capability overview](filters/README.md), the generated
  [filter reference](filters/reference.md) grouped by category, and
  [writing extensions](filters/extensions.md).
- [Feature and build reference](features.md#cargo-features).
- [Example catalog](../examples/README.md), including
  [routing examples](../examples/README.md#general).
- Conformance: [OpenAI conformance scope, report, and verification](conformance/README.md),
  [OpenAI Responses state conformance](conformance/openai-responses-state-conformance.md),
  [OpenResponses translation coverage](conformance/openresponses-translation.md),
  [OpenResponses acceptance](conformance/openresponses-acceptance.md), and the
  [provider translation compatibility matrix](conformance/translation-matrix.md).

## Developing

- [Development setup](developing/getting-started.md),
  [adding filters](developing/adding-filters.md), and
  [embedding filters](developing/filter-hosts.md).
- [Flow visualizers](developing/flow-visualizers.md) and
  [type design](developing/type-design.md).
- [CLI clients with vLLM](developing/cli-vllm-through-praxis.md),
  [OpenShell](developing/openshell-cli-vllm-through-praxis.md),
  [vLLM qualification](developing/vllm-qualification.md),
  [GPU testing](developing/gpu-nightly-suite.md), and
  [llm-d integration testing](developing/llmd-integration-testing.md).
- [Hosted passthrough nightly](developing/hosted-passthrough-nightly.md) and
  [hosted translation nightly](developing/hosted-translation-nightly.md).
- [OpenAI and Anthropic compatibility](developing/openai-anthropic-compatibility.md)
  and the [Anthropic Messages replay test plan](developing/anthropic-messages-replay-test-plan.md).
- [FIPS tooling](developing/fips.md).
- [Release process](developing/release.md),
  [project management](developing/project-management.md), and
  [contributing](../CONTRIBUTING.md).
- [Documentation support audit](developing/documentation-audit.md): reviewed
  revision and validation evidence for these pages.
