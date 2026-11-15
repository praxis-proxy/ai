# Praxis AI Documentation

Choose a client API and an upstream first, then add routing and policy.
These pages describe **Development (current main)**. Build features and
compatibility can differ from released v0.10.0; use the documentation snapshot
matching your binary. See the [support audit](documentation-audit.md) for the
reviewed revision and validation evidence.

## Get started

- [Quickstart](quickstart.md): build, validate, and forward your first hosted request.
- [Choose a client and provider](client-provider-paths.md): protocol, credentials,
  feature selection, and compatibility limits in one table.
- [Operator recipes](recipes/README.md): prerequisites, full configurations,
  startup, requests, expected results, and cleanup.

## Client APIs

- [Chat Completions](quickstart.md#chat-completions): native OpenAI-compatible forwarding.
- [Anthropic Messages](anthropic-messages.md): native forwarding, translation,
  SDK versions, and backend qualification.
- [Responses](recipes/managed-responses.md): distinguish native passthrough
  from experimental gateway-managed state and tool execution.
- [Conversations](recipes/managed-responses.md#conversations-and-state): local
  experimental CRUD, storage, and continuation boundaries.
- [Translation compatibility](conformance/translation-matrix.md): supported,
  lossy, rejected, and known-gap behavior by direction.

## Connect providers and credentials

- [Provider selection](client-provider-paths.md): OpenAI-compatible, Anthropic,
  Azure, Vertex, and Bedrock paths.
- [Credential boundaries](recipes/README.md#credentials): caller keys,
  gateway authentication, backend secrets, and callout credentials.
- [Hosted OpenAI](quickstart.md), [native or translated Anthropic](recipes/anthropic.md),
  and [Azure, Vertex, and Bedrock adapters](recipes/providers.md).

## Route requests

- [Model aliases and provider routing](recipes/routing.md): start with a static
  mapping, then choose dynamic selection, failover, or session affinity.
- [Routing facts](architecture/routing-facts.md): which component owns each fact.
- [Routing examples](../examples/README.md#general): complete YAML chains.

## Protect and control

- [Guardrails and token usage](recipes/guardrails-token-usage.md): external
  screening, usage extraction, streaming limits, and metrics.
- [Operator capability overview](filters/README.md): identity, token limits,
  metering, credentials, and required filter order.
- [Outbound callout security](architecture/outbound-callouts.md).

## Tools and state

- [Experimental managed Responses and tools](recipes/managed-responses.md).
- [MCP and A2A](architecture/agentic-protocols.md): protocol routing and brokering
  are distinct from model-driven execution.
- [Files and stores](architecture/response-store.md),
  [retained-payload budget](architecture/responses-retained-payload.md), and
  [store migrations](store/schema-migration.md).

## Operate

- [Operations checklist and troubleshooting](operations.md): deployment,
  configuration validation, observability, and upgrades.
- [Capacity planning](capacity-planning.md): file descriptors and admission.
- [FIPS deployment](fips.md) and [OpenTelemetry](architecture/opentelemetry.md).
- [Security policy](../.github/SECURITY.md).

## Reference and development

- [Feature and build reference](features.md#cargo-features).
- [Generated filter reference](filters/reference.md) and [example catalog](../examples/README.md).
- [Development setup](developing/getting-started.md),
  [adding filters](developing/adding-filters.md), and [embedding filters](developing/filter-hosts.md).
- [CLI clients with vLLM](developing/cli-vllm-through-praxis.md),
  [OpenShell](developing/openshell-cli-vllm-through-praxis.md),
  [qualification](developing/vllm-qualification.md), and [GPU testing](developing/gpu-nightly-suite.md).

### Conformance

- [OpenAI conformance scope, report, and verification](conformance/README.md).
- [OpenAI Responses state conformance](conformance/openai-responses-state-conformance.md).
- [OpenResponses translation coverage](conformance/openresponses-translation.md).
- [Provider translation compatibility matrix](conformance/translation-matrix.md).

### Architecture

- [AI inference architecture](architecture/ai-inference.md),
  [flow visualizers](developing/flow-visualizers.md), and [type design](developing/type-design.md).
- [Cryptographic inventory](architecture/cryptographic-inventory.md),
  [PostgreSQL cryptographic boundary](architecture/postgres-cryptographic-boundary.md),
  and [FIPS tooling](developing/fips.md).
- [Release process](release.md), [legacy 0.2 migration](migrating-to-0.2.md),
  [project management](developing/project-management.md), and [contributing](../CONTRIBUTING.md).
