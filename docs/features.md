# Features

Praxis AI extends the [Praxis proxy framework][praxis]
with AI-specific filters for inference routing, provider
APIs, token counting, guardrails, agentic protocols,
response storage, and prompt enrichment.

For base proxy features (TLS, HTTP/2, TCP, WebSocket,
load balancing, rate limiting, compression, CORS,
health checks, etc.), see the
[Praxis core documentation][praxis].

[praxis]: https://github.com/praxis-proxy/praxis

## AI Inference

- **Model-based routing** (`model_to_header`): extracts
  the `model` field from JSON request bodies and
  promotes it to an `X-Model` header, enabling
  header-based routing to provider-specific clusters.
  Uses StreamBuffer to inspect the body before upstream
  selection.
- **Credential injection** (`credential_injection`):
  per-cluster API key injection with client credential
  stripping. Supports inline values and environment
  variable sources. Pair with a source discriminator
  (IP ACL, client auth) to control which clients get
  credential upgrades.
- **Prompt enrichment** (`prompt_enrich`): inject system
  or user messages into OpenAI-compatible chat
  completion request bodies at the proxy layer. Static
  configured messages are prepended or appended to the
  `messages` array before forwarding upstream.

## OpenAI Responses API

- **Responses API classification**
  (`openai_responses_format`): classifies OpenAI
  Responses API and Chat Completions API requests by
  inspecting the request body. Promotes format, model,
  stream, and routing mode (stateless/stateful) to
  configurable headers, metadata, and filter results
  for downstream routing via branch chains.
- **Responses API validation**
  (`openai_responses_validate`): validates Responses
  API parameter combinations (stream/background,
  background/store conflicts), extracts conversation
  IDs, and generates cryptographically random response
  and conversation IDs with `resp_` and `conv_`
  prefixes.
- **Model rewrite**
  (`openai_responses_model_rewrite`): rewrites the
  `model` field in Responses API request bodies.
- **Response rehydration**
  (`openai_responses_rehydrate`): validates
  `previous_response_id` by fetching the stored
  response, confirming its status is `"completed"`,
  and populating `ResponsesState` with the full
  conversation history.
- **Response store** (`openai_response_store`):
  persists non-streaming Responses API responses to a
  configured storage backend (SQLite, PostgreSQL).
- **Conversation management**
  (`openai_conversations`): handles all
  `/v1/conversations` endpoints locally.
- **Responses proxy** (`openai_responses_proxy`): rebuilds
  the request body from `ResponsesState` when present.

## Anthropic Messages API

- **Messages classification**
  (`anthropic_messages_format`): classifies Anthropic
  Messages API requests and promotes routing facts to
  headers, metadata, and filter results.
- **Messages protocol**
  (`anthropic_messages_protocol`): normalizes Anthropic
  Messages protocol headers for native backends.
- **Stream event translation**
  (`anthropic_stream_events`): transforms streaming SSE
  responses between OpenAI and Anthropic formats,
  processing each chunk as it arrives.
- **API translation** (`anthropic_to_openai`):
  transforms Anthropic Messages API requests to Chat
  Completions-compatible request bodies and transforms
  compatible responses back.
- **Request validation** (`anthropic_validate`):
  validates Anthropic Messages request bodies for
  proxy-owned JSON envelope requirements.

## AI Agentic

- **JSON-RPC 2.0 foundation** (`json_rpc`): request
  envelope parsing and method/id extraction for HTTP
  POST bodies, enabling method-based routing for
  MCP/A2A-style traffic.
- **MCP proxying** (`mcp`): Model Context Protocol
  broker with tool discovery and routing via the
  filter pipeline.
- **A2A proxying** (`a2a`): Agent-to-Agent protocol
  support with task routing via the filter pipeline.

## Security and Observability

- **AI guardrails** (`ai_guardrails`): calls an
  external AI guardrail provider to evaluate request
  bodies. The provider determines whether content
  should be passed, blocked, or redacted.
- **Token usage headers** (`token_usage_headers`):
  injects `Praxis-Token-Input`, `Praxis-Token-Output`,
  and `Praxis-Token-Total` headers into downstream
  responses when token usage data is present in filter
  metadata.

## Extensions

- **Rust extensions**: compile-time custom filters with
  zero overhead via the `HttpFilter` trait from
  `praxis-filter` and `register_filters!` macro.
- **Auto-discovery**: external filter crates
  self-register at build time via
  `[package.metadata.praxis-filters]`.
