# Praxis AI Documentation

Praxis AI provides AI inference filters, agentic protocol support, and
provider API integrations on top of [Praxis](https://github.com/praxis-proxy/praxis).

## Start here

- [Quickstart](quickstart.md): build and forward a first hosted API request
- [Client and provider paths](client-provider-paths.md): choose a protocol,
  upstream, credentials, and build features
- [Feature overview](features.md)
- [Example configurations](../examples/README.md)
- [Generated filter reference](filters/reference.md)

## Architecture

- [AI inference](architecture/ai-inference.md)
- [OpenAI Responses retained-payload budget](architecture/responses-retained-payload.md)
- [Routing facts](architecture/routing-facts.md)
- [Agentic protocols](architecture/agentic-protocols.md)
- [Response store](architecture/response-store.md)
- [PostgreSQL cryptographic boundary](architecture/postgres-cryptographic-boundary.md)
- [Outbound callout security](architecture/outbound-callouts.md)
- [Cryptographic operations inventory](architecture/cryptographic-inventory.md)

## Provider guides

- [Anthropic Messages](anthropic-messages.md)
- [Anthropic Messages replay testing](anthropic-messages-replay-test-plan.md)
- [AI filter overview](filters/README.md)

## Conformance

- [OpenAI conformance scope, report, and verification](conformance/README.md)
- [OpenAI Responses state conformance](conformance/openai-responses-state-conformance.md)
- [OpenResponses translation coverage](conformance/openresponses-translation.md)
- [Provider translation compatibility matrix](conformance/translation-matrix.md)

## Development

For repository contributors building, testing, or extending Praxis AI:

- [Development setup](developing/getting-started.md)
- [Run your Code Harness through Praxis and vLLM](developing/cli-vllm-through-praxis.md)
- [Run your Code Harness in OpenShell through Praxis and vLLM](developing/openshell-cli-vllm-through-praxis.md)
- [Nightly GPU test suite](developing/gpu-nightly-suite.md)
- [Adding filters](developing/adding-filters.md)
- [Flow visualizers](developing/flow-visualizers.md)
- [Type design](developing/type-design.md)
- [FIPS tooling](developing/fips.md)
- [Project management](developing/project-management.md)
- [Proposal process](https://github.com/praxis-proxy/enhancements)

## Operations and releases

- [Capacity planning: file descriptors](capacity-planning.md)
- [Release process](release.md)
- [FIPS 140-3](fips.md)
- [Migrating to 0.2.0](migrating-to-0.2.md)
- [Security policy](../.github/SECURITY.md)
- [Contributing](../CONTRIBUTING.md)
