# Embedding AI filters

This page covers Rust registration and host integration. For YAML pipelines,
see the [capability overview](../filters/README.md).

## Organization

AI filters live in two crates:

```text
apis/src/                 Provider API integrations
  anthropic/              Anthropic Messages API
  openai/                 OpenAI Responses and Conversations APIs
  classifier/             Shared request classification
  store/                  Response and conversation persistence

filters/src/              Cross-provider behavior
  agentic/                MCP and A2A
  guardrails/             AI content guardrails
  inference/              Model routing
  prompt_enrich/          Prompt injection
  token_usage/            Token counting, headers, and per-tenant metrics
```

Praxis AI also inherits the base proxy filters from Praxis core through
`FilterRegistry::with_builtins()`: routing, load balancing, headers,
credential injection, JSON-RPC parsing, CORS, compression, IP ACLs, and other
general proxy behavior. The [Praxis core filter reference][praxis-filters]
documents them.

## Registration

`praxis_ai_filters::register_ai_filters` registers the in-tree AI filters.
Consumers that need only AI filters, such as an Envoy ExtProc, can depend on
`praxis-ai-filters` without the proxy crate:

```rust
use praxis_filter::FilterRegistry;

let mut registry = FilterRegistry::with_builtins();
praxis_ai_filters::register_ai_filters(&mut registry);
// Or: let registry = praxis_ai_filters::build_ai_registry();
```

`praxis-ai-proxy` builds the full registry in three ownership layers:

```rust
let mut registry = FilterRegistry::with_builtins();
praxis_ai_filters::register_ai_filters(&mut registry);
register_external_filters(&mut registry); // proxy-only
```

Core filters, in-tree AI filters, and auto-discovered extensions stay in
separate layers, and external filter auto-discovery is proxy-only.

Pipelines that use OpenAI store or rehydrate filters must also install the
response-store extension:

```rust
pipeline.add_pipeline_extension(
    Box::new(praxis_ai_apis::store::ResponseStoreRegistry::new()),
);
```

The AI proxy does this in `server/src/pipelines.rs`; other hosts, such as
ExtProc, must do the same.

[praxis-filters]: https://github.com/praxis-proxy/praxis/tree/main/docs/filters
