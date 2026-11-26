# Migrating to 0.2.0

Version 0.2.0 moves token usage parsing from `praxis-ai-apis` into the private
token usage subsystem in `praxis-ai-filters`.

## Proxy configuration

No changes are required. The `token_count` and `token_usage_headers` filter
names, provider values, and metadata keys are unchanged.

## Rust API

The following 0.1.x API is removed:

```rust
praxis_ai_apis::token_usage::{
    TokenUsage,
    TokenUsageProvider,
    set_token_usage,
}
```

Provider response parsing now lives in `TokenCountFilter` and has no
standalone library API. Add the filter to a pipeline and select the provider
in its YAML configuration. Extracted counts are still available through these
`HttpFilterContext` metadata keys:

- `token.input`
- `token.output`
- `token.total`

Custom filters that produce token counts can write the same keys with
`HttpFilterContext::set_metadata`. Code that needs standalone provider parsing
must stay on 0.1.x or implement the parsing itself until a new public
abstraction exists.
