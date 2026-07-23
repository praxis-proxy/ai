# Features

Praxis AI adds provider, agentic, and observability filters
on the [Praxis proxy framework][praxis]. See the
[AI Gateway overview](overview.md) for how they fit together.

For TLS, routing, load balancing, rate limits, and other core
filters, see [Praxis core documentation][praxis].

[praxis]: https://github.com/praxis-proxy/praxis

## AI Inference

- **Model-based routing** (`model_to_header`): extracts
  the `model` field from JSON request bodies and
  promotes it to an `X-Model` header, enabling
  header-based routing to provider-specific clusters.
  Uses StreamBuffer to inspect the body before upstream
  selection.
- **Credential injection** (`credential_injection`, core):
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
- **Stream events** (`openai_stream_events`): accumulates
  native Responses API SSE events for downstream filters.
- **Tool routing** (`openai_tool_parse`): parses `tools` and
  `tool_choice` for branch-chain routing without mutating
  the body.

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

- **JSON-RPC 2.0 foundation** (`json_rpc`, core): request
  envelope parsing for MCP/A2A-style traffic.
- **MCP proxying** (`mcp`): MCP broker with catalog and
  session metadata; `tools/call` is not forwarded by the
  stateless broker profile.
- **A2A proxying** (`a2a`): task routing and SSE detection
  for Agent-to-Agent traffic.

## Security and Observability

- **AI guardrails** (`ai_guardrails`): calls an external AI
  guardrail provider to evaluate request (and eventually response)
  bodies. The provider determines whether content should be passed,
  blocked, or redacted. Request-side evaluation is not wired yet;
  response-side work is tracked in
  [#50](https://github.com/praxis-proxy/ai/issues/50).
- **Token counting** (`token_count`): extracts usage from
  provider responses (JSON and SSE) into filter metadata.
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
