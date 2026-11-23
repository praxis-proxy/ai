# Migrating to 0.2.0

Version 0.2.0 moves token usage parsing from
`praxis-ai-apis` into the private token usage subsystem in
`praxis-ai-filters`.

## Proxy configuration

No configuration changes are required. The `token_count` and
`token_usage_headers` filter names, provider values, and metadata
keys remain unchanged.

## Rust API

The following 0.1.x API is removed:

```rust
praxis_ai_apis::token_usage::{
    TokenUsage,
    TokenUsageProvider,
    set_token_usage,
}
```

Provider response parsing now belongs to `TokenCountFilter` and is
not exposed as a standalone library API. Add the filter to a pipeline
and select the provider through its YAML configuration. Extracted
counts remain available through these `HttpFilterContext` metadata
keys:

- `token.input`
- `token.output`
- `token.total`

Custom filters that produce token counts can write the same keys with
`HttpFilterContext::set_metadata`. Code that requires standalone
provider parsing must remain on 0.1.x or own that parsing until a new
public abstraction is introduced.
