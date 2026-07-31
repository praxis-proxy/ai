# Filters

Praxis AI registers AI-specific filters into the
[Praxis filter pipeline][praxis-filters]. The generated
[filter reference](reference.md) is the authoritative inventory of filter
names, descriptions, and configuration documentation.

For pipeline execution, filter traits, body access, conditional execution,
filter chains, and core filters, see the Praxis core filter documentation.

## Organization

AI filters are organized across two crates:

```text
apis/src/                 Provider API integrations
  anthropic/              Anthropic Messages API
  openai/                 OpenAI Responses and Conversations APIs
  classifier/             Shared request classification
  store/                  Response and conversation persistence
  token_usage/            Provider token-usage parsing

filters/src/              Cross-provider behavior
  agentic/                MCP and A2A
  guardrails/             AI content guardrails
  inference/              Model routing
  prompt_enrich/          Prompt injection
  token_usage/            Token counting and headers
```

Praxis AI also inherits all base proxy filters from Praxis core through
`FilterRegistry::with_builtins()`. This includes routing, load balancing,
headers, credential injection, JSON-RPC parsing, CORS, compression, IP ACLs,
and other general proxy behavior. Those filters are intentionally documented
in the [Praxis core filter reference][praxis-filters], not duplicated here.

## Registration

In-tree AI filters are registered by
`praxis_ai_filters::register_ai_filters`. Downstream
consumers that only need AI filters (for example an
Envoy ExtProc) can depend on `praxis-ai-filters` without
the proxy crate:

```rust
use praxis_filter::FilterRegistry;

let mut registry = FilterRegistry::with_builtins();
praxis_ai_filters::register_ai_filters(&mut registry);
// Or: let registry = praxis_ai_filters::build_ai_registry();
```

`praxis-ai-proxy` builds the full registry in three
ownership layers:

```rust
let mut registry = FilterRegistry::with_builtins();
praxis_ai_filters::register_ai_filters(&mut registry);
register_external_filters(&mut registry); // proxy-only
```

This keeps core filters, in-tree AI filters, and
auto-discovered extensions at clear ownership
boundaries. External filter auto-discovery stays
proxy-only.

Pipelines that use OpenAI store or rehydrate filters
must also install the response-store extension:

```rust
pipeline.add_pipeline_extension(
    Box::new(praxis_ai_apis::store::ResponseStoreRegistry::new()),
);
```

The AI proxy does this in `server/src/pipelines.rs`.
Other hosts (such as ExtProc) must do the same.

## Related documentation

- [Generated filter reference](reference.md)
- [Feature overview](../features.md)
- [Example configurations](../../examples/README.md)
- [Writing extensions](extensions.md)

[praxis-filters]: https://github.com/praxis-proxy/praxis/tree/main/docs/filters
