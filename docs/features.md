# Features

Praxis AI extends the [Praxis proxy framework][praxis] with AI-specific
filters and provider integrations. This page describes the capability
areas; the generated [filter reference](filters/reference.md) is the
authoritative inventory of filter names and configuration documentation.

For base proxy capabilities such as TLS, HTTP/2, TCP, WebSocket, load
balancing, rate limiting, compression, CORS, health checks, and credential
injection, see the [Praxis core documentation][praxis].

## Inference routing and transformation

- Classify requests by provider format, model, streaming mode, and tool
  composition, then promote those facts for policy-driven routing.
- Rewrite models and enrich prompts without changing application code.
- Translate Anthropic Messages requests and responses for
  Chat Completions-compatible inference backends.
- Normalize provider protocol headers and validate the JSON envelope fields
  required by the proxy.

## OpenAI Responses and Conversations

- Proxy native Responses API traffic and rebuild enriched request state.
- Store and rehydrate response history through SQLite or PostgreSQL.
- Serve the Conversations API locally and maintain conversation items.
- Resolve file references, extract supported document content, and accumulate
  streaming response state.
- Parse tool configuration and dispatch MCP or web-search tool calls in an
  agentic loop.

See the generated [OpenAI filter inventory](filters/reference.md#openai)
for the complete list.

## Anthropic Messages

- Classify and validate Anthropic Messages requests.
- Normalize native Anthropic protocol headers.
- Translate request, response, and streaming event formats for compatible
  OpenAI-style backends.

See the generated [Anthropic filter inventory](filters/reference.md#anthropic).

## Agentic protocols

- Classify and route Model Context Protocol (MCP) traffic.
- Broker configured MCP catalog operations and upstream tool discovery.
- Classify Agent-to-Agent (A2A) requests and route task or context follow-ups.
- Build on the JSON-RPC primitives provided by Praxis core.

## Safety and observability

- Evaluate request content through an external guardrail provider.
- Extract token usage across supported provider response formats.
- Expose normalized token counts through downstream response headers.

## Extensibility

Custom Rust filters implement the `HttpFilter` trait from `praxis-filter` and
register through the `register_filters!` macro. External filter crates can
self-register at build time with `[package.metadata.praxis-filters]`.

Start with the [example configurations](../examples/README.md), then use the
[filter reference](filters/reference.md) for exact configuration fields.

[praxis]: https://github.com/praxis-proxy/praxis
