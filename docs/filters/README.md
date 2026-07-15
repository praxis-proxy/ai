# Filters

Praxis AI registers AI-specific filters into the
[Praxis filter pipeline][praxis-filters]. For the base
filter system architecture (pipeline execution, filter
traits, body access, conditional execution, filter
chains), see the Praxis core filter documentation.

[praxis-filters]: https://github.com/praxis-proxy/praxis/tree/main/docs/filters

## AI Filter Categories

AI filters are organized across two crates:

```text
apis/src/                 Provider API filters
  anthropic/              Anthropic Messages API
  openai/                 OpenAI Responses/Chat API
  classifier/             Request classification
  store/                  Response persistence
  token_usage/            Token counting

filters/src/              Cross-provider filters
  agentic/                MCP, A2A, JSON-RPC
  guardrails/             AI content guardrails
  inference/              Model routing, credentials
  prompt_enrich/          Prompt injection
```

### Provider APIs (`praxis-ai-apis`)

| Filter | Description |
|--------|-------------|
| `anthropic_messages_format` | Classifies Anthropic Messages API requests |
| `anthropic_messages_protocol` | Normalizes Anthropic protocol headers |
| `anthropic_stream_events` | SSE format translation (OpenAI / Anthropic) |
| `anthropic_to_openai` | Anthropic-to-Chat Completions body translation |
| `anthropic_validate` | Anthropic request envelope validation |
| `openai_responses_format` | Classifies Responses/Chat Completions requests |
| `openai_responses_model_rewrite` | Rewrites `model` field in request bodies |
| `openai_responses_validate` | Validates and enriches Responses API requests |
| `openai_responses_rehydrate` | Fetches stored responses for conversation context |
| `openai_response_store` | Persists responses to storage backend |
| `openai_conversations` | Handles `/v1/conversations` endpoints |
| `openai_responses_proxy` | Rebuilds request body from `ResponsesState` |

### Cross-Provider Filters (`praxis-ai-filters`)

| Filter | Description |
|--------|-------------|
| `a2a` | A2A protocol metadata extraction |
| `mcp` | MCP protocol broker and routing |
| `json_rpc` | JSON-RPC 2.0 envelope parsing |
| `ai_guardrails` | External guardrail provider integration |
| `model_to_header` | Promotes `model` body field to header |
| `prompt_enrich` | Injects messages into chat completions |
| `token_usage_headers` | Token count response headers |

## Registration

AI filters are registered at startup in
`server/src/lib.rs` via the `register_ai_filters`
function. This adds them to the base `FilterRegistry`
alongside Praxis core builtins:

```rust
let mut registry = FilterRegistry::with_builtins();
register_ai_filters(&mut registry);
```

## Base Proxy Filters

Praxis AI inherits all base proxy filters from Praxis
core (router, load balancer, rate limiter, headers,
CORS, IP ACL, guardrails, compression, etc.). These are
included via `FilterRegistry::with_builtins()`. See the
[Praxis core filter reference][praxis-filters] for their
configuration.

## Related

- [Filter Reference](reference.md):
  configuration for all AI filters
- [Extensions](extensions.md): writing custom filters
