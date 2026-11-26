# Features

Praxis AI adds AI-specific filters and provider integrations to the
[Praxis proxy framework][praxis]. This page summarizes the capability areas
and the Cargo features that build them. The generated
[filter reference](filters/reference.md) is the authoritative list of filter
names and configuration fields.

Base proxy capabilities (TLS, HTTP/2, TCP, WebSocket, load balancing, rate
limiting, compression, CORS, health checks, credential injection) are covered
in the [Praxis core documentation][praxis]. For client setup, provider
authentication, and build choices, see
[Choose a client and provider](getting-started/choose-client-and-provider.md).
For operator tasks, required ordering, and examples, see the
[capability overview](filters/README.md).

## Inference routing and transformation

- Classify requests by provider format, model, streaming mode, and tool
  composition, and promote those facts for policy-driven routing.
- Rewrite models and enrich prompts without changing application code.
- Translate Anthropic Messages requests and responses for
  Chat Completions-compatible inference backends.
- Adapt Chat Completions to selected Anthropic Messages, Azure OpenAI,
  Vertex Gemini, or Bedrock Converse paths, with their documented limitations.
- Normalize provider protocol headers and validate the JSON envelope fields
  the proxy needs.

## OpenAI Responses and Conversations

Native forwarding works with ordinary standard routing. The state and tool
capabilities below need the experimental managed filters and the matching
store and tool feature groups; see the [managed recipe](guides/managed-responses.md).

- Proxy native Responses API traffic and rebuild enriched request state.
- Store and rehydrate response history through SQLite or PostgreSQL.
- Serve the Conversations API locally and maintain conversation items.
- Resolve file references, extract supported document content, and accumulate
  streaming response state.
- Parse tool configuration and dispatch MCP or web-search tool calls in an
  agentic loop.

In the filter reference, OpenAI request and translation filters are under
[Client APIs and translation](filters/reference.md#client-apis-and-translation),
and Responses state and tool filters under
[Agents, tools, and state](filters/reference.md#agents-tools-and-state). The
[Responses retained-payload budget] explains how the agentic loop bounds the
payload it holds across inference rounds, tools, streaming, and Store
operations.

## Anthropic Messages

Choose [native forwarding or a supported translation path](getting-started/choose-client-and-provider.md).
The [Messages guide](anthropic-messages.md) records SDK version and backend
qualification limits.

- Classify and validate Anthropic Messages requests.
- Normalize native Anthropic protocol headers.
- Translate request, response, and streaming event formats for compatible
  OpenAI-style backends.

The filters are listed under [Anthropic filters](filters/reference.md#client-apis-and-translation)
in the filter reference.

## Agentic protocols

- Classify and route Model Context Protocol (MCP) traffic.
- Broker configured MCP catalog operations and upstream tool discovery.
- Classify Agent-to-Agent (A2A) requests and route task or context follow-ups.
- Build on the JSON-RPC primitives provided by Praxis core.

## Safety and observability

- Evaluate request content through an external guardrail provider.
- Extract token usage across supported provider response formats.
- Expose normalized token counts through downstream response headers.

## Cargo features

The default production-proxy build (`cargo build -p praxis-ai-proxy`),
`make release`, and the published container image select `full`, which
currently contains `standard`. Responses, Conversations, their stores, and the
agentic loop are experimental and off by default.

Opt in to `openai-responses` for Responses support, and separately to
`openai-agentic-loop` for ephemeral multi-request agents. Each activates the
`experimental` marker; selecting `experimental` alone enables neither.
`openai-all` enables the complete experimental OpenAI stack, including the
loop, without selecting a database backend:

```console
cargo build -p praxis-ai-proxy --features openai-all,store-sqlite
```

The FIPS build defaults to `aws-sigv4-filter`; Responses and certificate-only
PostgreSQL are explicit opt-ins there too. See [FIPS 140-3](operations/fips.md).
Graduating from experimental requires the workload and autoscaling coverage
tracked in [#1717](https://github.com/praxis-proxy/ai/issues/1717).

| Feature | Filters it adds | Notable dependencies |
|---------|-----------------|----------------------|
| `aws-sigv4-filter` (part of `standard`) | `aws_sigv4_sign` | `aws-credential-types`; the signature is computed by the system OpenSSL |
| `openai-responses` | `openai_responses_request`, `openai_responses_proxy`, `openai_stream_events`, `openai_responses_to_chat_completions`, `openai_doc_extract`, `openai_client_tool_compat`, `openai_file_search_dispatch`, `openai_web_search_dispatch` | none beyond the default build |
| `openai-agentic-loop` | `openai_agentic_loop` | `openai-responses`; activates `experimental` |
| `openai-file-resolve-filter` | `openai_file_resolve` | none beyond `openai-responses`; uses the shared subrequest client |
| `store-postgres`, `store-postgres-cert-auth`, `store-sqlite`, `store-all` | `openai_responses_store`, `openai_responses_rehydrate`, and the SQL backends | `sqlx` (PostgreSQL adds native TLS through the system OpenSSL; the certificate-only profile omits password authentication) |
| `openai-conversations` | `openai_conversations` | `jsonschema`, `utoipa`, a store backend |
| `openai-files` | none yet (Files wire contracts, generated OpenAPI, and conformance checks) | `utoipa`; activates `experimental` |
| `openai-compact` | `openai_responses_compact` | `tiktoken-rs`, a store backend |
| `openai-mcp-tools` | `openai_mcp_tool_resolve`, `openai_mcp_dispatch`, `openai_mcp_streaming_selector` | `rmcp`, a store backend |
| `openai-all` | every OpenAI group above, without choosing a store backend | |
| `full` | every non-experimental filter (currently `standard`) | |

All the OpenAI groups above depend on experimental Responses. The
store-backed groups need a backend at runtime, so pair them with
`store-postgres`, `store-postgres-cert-auth`, or `store-sqlite` (for example
`--features openai-all,store-sqlite`).

These features are also experimental and opt-in: `http-callout-filter`,
`azure-ad-filter`, `gcp-adc-filter`, `token-rate-limit-filter`,
`token-ceiling-filter`, `inflight-tracker-filter`, `vertex-anthropic-filter`,
and `basic-auth-filter`. `llmd-ext-proc` and `opentelemetry` are opt-in as
well. A configuration that names a filter the binary was built without fails
at startup with an unknown filter type error.

## Extensibility

Custom Rust filters implement the `HttpFilter` trait from `praxis-filter` and
register through the `register_filters!` macro. External filter crates can
self-register at build time with `[package.metadata.praxis-filters]`.

Start with the [example configurations](../examples/README.md), then use the
[filter reference](filters/reference.md) for exact configuration fields.

[praxis]: https://github.com/praxis-proxy/praxis
[Responses retained-payload budget]: architecture/responses-retained-payload.md
