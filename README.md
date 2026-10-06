<img width="1200" height="400" alt="praxis-ai-banner" src="https://github.com/user-attachments/assets/2696dc84-22ad-4a34-81d8-962f7ace86c0" />

[![Tests](https://github.com/praxis-proxy/ai/actions/workflows/tests.yaml/badge.svg)](https://github.com/praxis-proxy/ai/actions/workflows/tests.yaml)
[![Coverage: ≥95%](https://img.shields.io/badge/Coverage-≥95%25-brightgreen.svg)](https://github.com/praxis-proxy/ai/actions/workflows/coverage.yaml)
[![MSRV: 1.96](https://img.shields.io/badge/MSRV-1.96-brightgreen.svg)](https://blog.rust-lang.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Praxis AI is an AI gateway built on
[Praxis](https://github.com/praxis-proxy/praxis).** It brings provider-aware
routing, protocol translation, stateful OpenAI APIs, and agent traffic into a
configurable proxy. Clients keep their API shape while the gateway selects
backends and applies policy.

## What can it do?

- **Route by request content** — classify OpenAI Responses, Chat Completions,
  and Anthropic Messages traffic; select backends by API format, model, or MCP
  tool name
  ([unified gateway](examples/configs/anthropic/unified-gateway.yaml),
  [intelligent routing](examples/configs/intelligent-route-all-capabilities.yaml)).
- **Proxy or translate provider APIs** — forward native traffic or serve
  Anthropic Messages and OpenAI Responses clients from Chat Completions
  backends, streaming included
  ([Anthropic](examples/configs/anthropic/messages-to-openai.yaml),
  [Responses](examples/configs/openai/responses/codex-http-chat-translation.yaml)).
- **Support code harnesses** — run Codex through Responses and Claude Code
  through Messages, reaching native vLLM or Chat Completions via translation
  ([coding client guide](docs/developing/cli-vllm-through-praxis.md)).
- **Manage OpenAI response state** — persist and rehydrate Responses history
  and serve Conversations locally, backed by PostgreSQL or SQLite
  ([response store guide](docs/architecture/response-store.md),
  [Conversations example](examples/configs/openai/conversations/conversations.yaml)).
- **Connect tools and agents** — run Responses tool loops with MCP, web and
  file search; route stateless MCP calls and A2A follow-ups
  ([agentic Responses](examples/configs/openai/responses/full-flow-agentic.yaml),
  [MCP broker](examples/configs/mcp-stateless-broker.yaml),
  [A2A routing](examples/configs/a2a-task-routing.yaml)).
- **Apply policy and measure usage** — inject credentials, enrich prompts, call
  guardrails, and report token usage ([feature overview](docs/features.md)).
- **Extend the pipeline** with custom Rust filters on Praxis's `HttpFilter`
  interface.

See the [feature overview](docs/features.md) and
[filter reference](docs/filters/README.md) for the full list.

## Architecture

Clients keep their provider-native protocols while Praxis AI classifies,
transforms, and routes traffic through one policy-driven gateway.

![Codex and other OpenAI clients, Claude Code and other Anthropic clients, and MCP and A2A traffic flow through Praxis AI to inference and agent backends, with side services for tools, storage, guardrails, and metering](assets/praxis-ai-architecture.svg)

### Praxis AI and Praxis

[Praxis][praxis-proxy/praxis] supplies the proxy runtime, listeners, TLS, load
balancing, and filter framework; Praxis AI packages the AI-specific filters and
server on top of it. Separate repositories let the AI integrations evolve
independently while Praxis stays useful for general proxy workloads. See our
[conventions] for project structure and development practices.

[praxis-proxy/praxis]:https://github.com/praxis-proxy/praxis
[conventions]:https://github.com/praxis-proxy/conventions

## Quick start

Build and start the gateway with its built-in configuration:

```console
make release
./target/release/praxis-ai
```

`make release` builds the `full` feature set; `cargo build -p praxis-ai-proxy`
builds the smaller `standard` set without the stateful OpenAI filter groups (see
[Cargo features](docs/features.md#cargo-features)).

Then check that it is running:

```console
curl http://127.0.0.1:8080/
```

```json
{"status": "ok", "server": "praxis-ai"}
```

Ready to connect a backend? Follow the [quickstart](docs/quickstart.md), or
choose from the [example configurations](examples/README.md) for OpenAI,
Anthropic, MCP, A2A, routing, guardrails, token usage, and more.

## Learn your way around

| If you want to… | Start here |
| --- | --- |
| Run Praxis AI locally | [Quickstart](docs/quickstart.md) |
| Browse supported capabilities | [Feature overview](docs/features.md) |
| Configure a filter | [Filter reference](docs/filters/README.md) |
| Understand the design | [Architecture docs](docs/README.md#architecture) |
| Build or test the workspace | [Development guide](docs/developing/getting-started.md) |
| Add a new filter | [Adding filters](docs/developing/adding-filters.md) |

Praxis AI handles the AI-specific layer. For listeners, TLS, load balancing,
rate limiting, health checks, and other core proxy features, visit the
[Praxis repository](https://github.com/praxis-proxy/praxis).

> [!IMPORTANT]
> Praxis AI is alpha software. APIs, configuration, and operational
> behavior may change before `v1.0.0`. See the [security policy]
> for the supported release line.

Released container images are published to
[`ghcr.io/praxis-proxy/ai`][container images] and can be pulled with Docker or
Podman:

```console
docker pull ghcr.io/praxis-proxy/ai:latest
```

See the [release documentation] for image contents and tagging, and the
[development guide] for source builds. A FIPS 140-3 build for Red Hat
Enterprise Linux is published under the same tags with a `-fips` suffix (for
example `latest-fips`); see [FIPS 140-3](docs/fips.md).

## Contributing

Contributions are welcome, from bug reports and documentation fixes to new
filters and protocol support. Before opening a pull request, please read the
[contributing guide](.github/CONTRIBUTING.md) and
[development setup](docs/developing/getting-started.md).

For larger changes, open a [feature request] and follow the
[proposal process](https://github.com/praxis-proxy/enhancements) so we can shape
the idea together.

[Open an issue][issues] · [Request a feature][feature request] ·
[Open a pull request][pull requests]

[issues]: https://github.com/praxis-proxy/ai/issues/new
[pull requests]: https://github.com/praxis-proxy/ai/compare
[container images]: https://github.com/praxis-proxy/ai/pkgs/container/ai
[development guide]: docs/developing/getting-started.md
[feature request]: https://github.com/praxis-proxy/ai/issues/new?template=feature-request.yml
[release documentation]: docs/release.md
[security policy]: .github/SECURITY.md

## License

Apache 2.0
