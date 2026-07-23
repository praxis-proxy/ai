# Praxis AI Documentation

Praxis AI is an **AI-native proxy server** and **AI Gateway**
(AI API Gateway) for routing, managing, enriching, and
parsing inference and agentic traffic. It builds on the
[Praxis](https://github.com/praxis-proxy/praxis) proxy
framework and composable filter pipeline.

Run the `praxis-ai` binary for all features below. See
[core docs](https://github.com/praxis-proxy/praxis/tree/main/docs)
for shared proxy configuration.

## Getting started

- [AI Gateway overview](overview.md)
- [Quickstart](quickstart.md)
- [Features](features.md)
- [Example configs](../examples/README.md)

## Guides

- [OpenAI Responses API](openai-responses.md)
- [Anthropic Messages API](anthropic-messages.md)
- [Token counting](token-counting.md)

## Architecture

- [Crate layout](architecture/crate-layout.md)
- [AI inference pipeline](architecture/ai-inference.md)
- [Agentic protocols](architecture/agentic-protocols.md)
- [Response store](architecture/response-store.md)

## Filters

- [Filter overview](filters/README.md)
- [Filter reference](filters/reference.md)
- [Extensions](filters/extensions.md)

## Developing

- [Getting started](developing/getting-started.md)
- [Conventions](developing/conventions.md)
- [Adding filters](developing/adding-filters.md)

## Core operating docs

TLS, security hardening, and YAML schema details live in the
[Praxis core documentation](https://github.com/praxis-proxy/praxis/tree/main/docs/operating).

## Reference

- [Proposals](proposals.md)
- [Anthropic Messages replay test plan](anthropic-messages-replay-test-plan.md)
