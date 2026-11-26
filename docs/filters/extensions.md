# Extensions

Praxis AI inherits the extension system from [Praxis core][praxis]. Custom
filters implement `HttpFilter` or `TcpFilter` from `praxis-filter` and
register into the shared `FilterRegistry`.

[praxis]: https://github.com/praxis-proxy/praxis

## Auto-Discovery (Recommended)

External filter crates can register themselves into Praxis AI at build time.
The operator adds a `Cargo.toml` dependency and YAML config, with no Rust code
changes.

### How It Works

1. The external crate declares its filters with `export_filters!`.
2. Its `Cargo.toml` carries a `[package.metadata.praxis-filters]` marker.
3. The Praxis AI server's `build.rs` runs `cargo metadata`, finds marked
   crates, and generates registration code.
4. At startup, discovered filters are registered alongside built-ins and AI
   filters.

### External Crate Setup

In the external crate's `Cargo.toml`:

```toml
[package]
name = "my-token-quota"
version = "0.1.0"

# Marker: tells the build script this crate exports
# filters.
[package.metadata.praxis-filters]

[dependencies]
async-trait = "0.1"
praxis-proxy-filter = "0.3"
serde = { version = "1", features = ["derive"] }
serde_yaml = { package = "yaml_serde", version = "0.10" }
```

In the external crate's `src/lib.rs`:

```rust
use async_trait::async_trait;
use praxis_filter::{
    FilterAction, FilterError, HttpFilter,
    HttpFilterContext, export_filters,
};

pub struct TokenQuotaFilter { /* ... */ }

#[async_trait]
impl HttpFilter for TokenQuotaFilter {
    fn name(&self) -> &'static str { "token_quota" }

    async fn on_request(
        &self, _ctx: &mut HttpFilterContext<'_>,
    ) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }
}

impl TokenQuotaFilter {
    pub fn from_config(
        config: &serde_yaml::Value,
    ) -> Result<Box<dyn HttpFilter>, FilterError> {
        // Parse config and construct the filter
        Ok(Box::new(Self { /* ... */ }))
    }
}

export_filters! {
    http "token_quota" => TokenQuotaFilter::from_config,
}
```

### Operator Usage

Add the crate to the Praxis AI server's `Cargo.toml`:

```toml
[dependencies]
my-token-quota = "0.1"
```

Then reference the filter by name in YAML:

```yaml
filter_chains:
  - name: main
    filters:
      - filter: token_quota
        max_tokens_per_minute: 100000
```

Rebuild and run.

### Duplicate Detection

Filter names must be unique across all sources. If an external filter name
collides with a built-in, AI, or other external filter, the server panics at
startup with an error naming the collision.

## Rust Extensions (Manual Registration)

Use manual registration for a custom Praxis AI binary with inline filters
that you do not share as a separate crate. These are compile-time extensions
with no runtime overhead:

1. Implement `HttpFilter` (`on_request`, `on_response`, body hooks).
2. Register it with `register_filters!`.
3. Reference it by name in YAML filter chains.

### HTTP Filter

```rust
use async_trait::async_trait;
use serde::Deserialize;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError,
    HttpFilter, HttpFilterContext, Rejection,
    register_filters,
};

struct ModelBlocklist {
    blocked: Vec<String>,
}

impl ModelBlocklist {
    pub fn from_config(
        config: &serde_yaml::Value,
    ) -> Result<Box<dyn HttpFilter>, FilterError> {
        #[derive(Deserialize)]
        struct Cfg {
            blocked_models: Vec<String>,
        }

        let cfg: Cfg =
            serde_yaml::from_value(config.clone())?;
        Ok(Box::new(Self {
            blocked: cfg.blocked_models,
        }))
    }
}

#[async_trait]
impl HttpFilter for ModelBlocklist {
    fn name(&self) -> &'static str {
        "model_blocklist"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer { max_bytes: Some(8_192) }
    }

    async fn on_request(
        &self, ctx: &mut HttpFilterContext<'_>,
    ) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }
}

// In your binary:
register_filters! {
    http "model_blocklist" => ModelBlocklist::from_config,
}
```

### Registration

`register_filters!` takes protocol-prefixed entries, as in the example above
(`http "model_blocklist" => ModelBlocklist::from_config`). It generates a
`custom_registry()` function that returns a `FilterRegistry` with built-in,
AI, and custom filters. Use it with the test utilities (`start_proxy_with_registry`) or in
your own server bootstrap built from the workspace crates (`praxis-core`,
`praxis-filter`, `praxis-ai-apis`, `praxis-ai-filters`).

### YAML Config

Keys next to `filter:` in a filter chain entry reach `from_config` as a
`serde_yaml::Value`:

```yaml
filter_chains:
  - name: ai
    filters:
      - filter: model_blocklist
        blocked_models:
          - "gpt-3.5-turbo"
          - "claude-2"
        conditions:
          - when:
              methods: ["POST"]
```

Custom filters behave like built-ins: same ordering, context access, and
short-circuiting. The [filter overview](README.md) covers the AI filters.

## Best Practices

### Keep filters stateless when possible

Read all configuration in `from_config` and keep the filter struct
immutable. Filters are shared across requests and must be `Send + Sync`; when
you need shared mutable state such as counters or connection tracking, use
atomics or interior mutability with a minimal lock scope.

### Return early with `Reject`, not panics

Use `FilterAction::Reject(Rejection::status(code))` to stop request
processing. Never panic inside a filter, because a panic takes down the worker
thread. Return `Err(...)` for unexpected failures and let the pipeline send
the 500 response.

### Declare body access accurately

Declare `request_body_access()` or `response_body_access()` only if the
filter inspects or modifies the body, since each declaration changes how the
pipeline buffers data. `BodyAccess::None` (the default) has no overhead. Use
`ReadOnly` to inspect without modifying, and `ReadWrite` only to mutate chunks
in place.

### Choose the right body mode

- `Stream`: lowest latency; chunks pass through as they arrive. Use it for
  filters that inspect only headers or process chunks independently.
- `StreamBuffer`: chunks pass through filters incrementally, but forwarding
  upstream waits until `Release` or end-of-stream. Use it when body content
  affects routing (for example model field extraction) or when you need the
  complete body before upstream selection (for example guardrail scanning).
  Set `max_bytes` to bound memory.

### Use `extra_request_headers` for metadata

Promote values a filter extracts from the body or derives to a request header
through `ctx.extra_request_headers`. Downstream filters such as the router can
then read them without depending on your filter.

### Provide `from_config` validation

Validate every configuration value in `from_config` so bad config fails at
startup with a descriptive error, not at request time. Parse and type-check
every field, and use `#[serde(default)]` for optional fields.

### Test with the integration harness

Write end-to-end tests with the integration test utilities (`free_port`,
`start_backend`, `start_proxy_with_registry`). Register the filter with
`FilterFactory::Http(Arc::new(factory))`, build a minimal YAML config, and
assert on status codes and response bodies. `tests/integration/` has
examples.
