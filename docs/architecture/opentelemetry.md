# OpenTelemetry Routing Semantics

Praxis AI can add AI routing decisions to the request trace created and
exported by Praxis core. Build the proxy with:

```sh
cargo build --release -p praxis-ai-proxy --features opentelemetry
```

The feature is disabled by default. It does not install an exporter or parse
OpenTelemetry environment variables. Configure exporting, propagation,
sampling, and request lifecycle tracing through Praxis core.

After `intelligent_route` successfully selects a provider, the feature emits a
short `routing.select` child span. The span contains bounded routing identity,
admission, locality, rank, tier, and overlay revision attributes. It never
records request or response bodies, prompts, credentials, authorization
headers, cookies, or session keys.

The division of responsibility is intentional:

```text
Praxis core request span
  |
  `-- routing.select        (Praxis AI)
        |
        `-- upstream hop    (Praxis core)
```

Praxis core owns the complete HTTP span lifetime and transport boundaries.
Praxis AI records only the semantic decision it makes. This prevents duplicate
request roots, conflicting trace propagation, and multiple exporter runtimes.
