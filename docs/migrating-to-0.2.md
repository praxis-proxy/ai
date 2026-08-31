# Migrating to 0.2.0

Version 0.2.0 moves token usage parsing from
`praxis-ai-apis` into the private token usage subsystem in
`praxis-ai-filters`, and normalizes the failure-policy keys
used by outbound callout filters.

## Proxy configuration

The `token_count` and `token_usage_headers` filter names,
provider values, and metadata keys remain unchanged.

### Outbound callout failure policy

Callout filters previously spelled the same fail-open/fail-closed
choice three different ways. They now share one key, `on_failure`,
with unchanged `open` / `closed` values and an unchanged `closed`
default. Rename the key in place:

| Filter | 0.1.x key | 0.2.0 key |
| --- | --- | --- |
| `anthropic_web_search` | `provider_failure_mode` | `on_failure` |
| `openai_web_search` | `provider_failure_mode` | `on_failure` |
| `openai_responses_compact` | `callout_failure_mode` | `on_failure` |
| `openai_file_search_callout` | `callout_failure_mode` | `on_failure` |
| `http_callout` | `on_failure` | `on_failure` (unchanged) |

The old keys are not accepted as aliases. Because these filters use
`deny_unknown_fields`, a stale key fails validation at startup with a
message naming the offending field.

`openai_file_resolve`'s `on_missing` key is **not** part of this
rename and keeps its `continue` / `reject` values, and `continue`
remains its default.

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
