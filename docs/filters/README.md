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

`server/src/lib.rs` builds the registry in three steps:

```rust
let mut registry = FilterRegistry::with_builtins();
register_ai_filters(&mut registry);
register_external_filters(&mut registry);
```

This keeps core filters, in-tree AI filters, and auto-discovered extensions at
clear ownership boundaries.

## Related documentation

- [Generated filter reference](reference.md)
- [Feature overview](../features.md)
- [Example configurations](../../examples/README.md)
- [Writing extensions](extensions.md)

[praxis-filters]: https://github.com/praxis-proxy/praxis/tree/main/docs/filters
