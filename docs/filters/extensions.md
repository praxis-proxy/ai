# Extensions

Praxis AI inherits the full extension system from
[Praxis core][praxis]. Custom filters implement
`HttpFilter` or `TcpFilter` from `praxis-filter` and
register into the shared `FilterRegistry`.

[praxis]: https://github.com/praxis-proxy/praxis

## filter_metadata

Praxis attaches a per-request `filter_metadata` map to
[`HttpFilterContext`][http-filter-context] from Praxis
core. Filters store flat string key-value pairs that
persist across all HTTP lifecycle phases on the same
request (request, request-body, response,
response-body).

Keys use dot-prefix namespacing by convention (for
example `token.input`, `token.output`, `a2a.method`).
Upstream filters write values; downstream filters read
them in later phases without coupling to each other's
internals. Note the phase-ordering constraint: `token_count`
writes body-derived counts in `on_response_body`, while
`token_usage_headers` reads metadata in `on_response`, so
those two filters cannot chain into response headers in one
pass (see [token-counting](../token-counting.md) and
[proposal 00214](../proposals/00214_token-usage-response-headers.md)).

Custom filters can read and write `filter_metadata` via
`ctx.filter_metadata` in `on_request`, `on_response`,
and body hooks.

[http-filter-context]: https://github.com/praxis-proxy/praxis/blob/main/filter/src/context.rs

## Auto-Discovery (Recommended)

External filter crates can self-register into Praxis AI
at build time. The operator adds a `Cargo.toml`
dependency and writes YAML config with zero Rust code
changes.

### How It Works

1. The external crate uses `export_filters!` to declare
   its filters
2. The crate's `Cargo.toml` carries a
   `[package.metadata.praxis-filters]` marker
3. The Praxis AI server's `build.rs` runs
   `cargo metadata`, discovers marked crates, and
   generates registration code
4. At startup, discovered filters are registered
   alongside built-ins and AI filters

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
praxis-proxy-filter = "0.4"
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

Rebuild and run — no other changes needed.

### Duplicate Detection

If an external filter name collides with a built-in, AI,
or another external filter, the server panics at startup
with a clear error message. Filter names must be unique
across all sources.

## Rust Extensions (Manual Registration)

Compile-time extensions with zero overhead. Implement
`HttpFilter` from `praxis-filter`, register it, and
reference it in YAML config. Use this approach when
building a custom Praxis AI binary with inline filters
that don't need to be shared as a separate crate.

1. Implement `HttpFilter` (`on_request`, `on_response`,
   body hooks)
2. Register with `register_filters!`
3. Reference by name in YAML filter chains

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

The `register_filters!` macro uses protocol-prefixed
syntax:

```rust
register_filters! {
    http "model_blocklist" => ModelBlocklist::from_config,
}
```

The macro generates a `custom_registry()` function that
returns a `FilterRegistry` with built-in, AI, and custom
filters. Use it with the test utilities
(`start_proxy_with_registry`) or build your own server
bootstrap from the workspace crates (`praxis-core`,
`praxis-filter`, `praxis-ai-apis`, `praxis-ai-filters`).

### YAML Config

Any keys placed alongside `filter:` in the filter chain
entry are passed to `from_config` as a
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

Custom filters participate identically to built-ins:
same ordering, context access, and short-circuit
capability.

See the [filter system documentation](README.md) for
the AI filter overview.

## Best Practices

### Keep filters stateless when possible

Prefer reading all configuration at construction time
(in `from_config`) and keeping the filter struct
immutable. When shared mutable state is required (e.g.
counters, connection tracking), use atomics or interior
mutability with minimal lock scope. Filters are shared
across requests and must be `Send + Sync`.

### Return early with `Reject`, not panics

Use `FilterAction::Reject(Rejection::status(code))` to
abort request processing. Never panic inside a filter;
a panic takes down the worker thread. Return
`Err(...)` for unexpected failures and let the pipeline
handle the 500 response.

### Declare body access accurately

Only declare `request_body_access()` or
`response_body_access()` if your filter actually
inspects or modifies the body. Each declaration changes
how the pipeline buffers data. `BodyAccess::None` (the
default) avoids overhead. Use `ReadOnly` if you inspect
but do not modify, and `ReadWrite` only if you mutate
chunks in place.

### Choose the right body mode

- `Stream`: lowest latency; chunks flow through as they
  arrive. Best for filters that inspect headers only or
  process chunks independently.
- `StreamBuffer`: chunks flow through filters
  incrementally but forwarding to upstream is deferred
  until `Release` or end-of-stream. Use when body
  content influences routing (e.g. model field
  extraction), when you need the complete body (e.g.
  guardrail scanning), or when you need to inspect the
  full body before upstream selection. Set `max_bytes`
  to avoid unbounded memory growth.

### Use `extra_request_headers` for metadata

When your filter extracts values from the body or
computes derived data, promote it to a request header
via `ctx.extra_request_headers`. This makes the value
visible to downstream filters (e.g. the router) without
coupling filters to each other.

### Provide `from_config` validation

Validate all configuration values in `from_config`
rather than deferring checks to request time. Fail fast
at startup with a descriptive error. Parse and
type-check every field; use `#[serde(default)]` for
optional fields with sensible defaults.

### Test with the integration harness

Use the integration test utilities (`free_port`,
`start_backend`, `start_proxy_with_registry`) to write
end-to-end tests for custom filters. Register your
filter with `FilterFactory::Http(Arc::new(factory))`,
build a minimal YAML config, and assert on status codes
and response bodies. See `tests/integration/` for
examples.

Built-in filter reference pages are generated from source.
After changing a filter config struct, run:

```console
cargo xtask generate-filter-docs
```

CI runs `cargo xtask lint-filter-docs` as part of `make lint`.
Every example under `examples/configs/` must have an
integration test (or an entry in the SKIP allowlist):

```console
cargo xtask lint-example-tests
```
