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

After `provider_route` validates the edge-selected candidate and resolves it
to a provider-local backend cluster, the feature emits a short `provider.route`
child span. The span records the resolved backend cluster the request was
routed to; it is not proof that a downstream endpoint, pod, or model server
successfully served the request. It contains:

- `provider.id`: the configured provider-boundary identifier for this
  listener. This is a configuration value, not necessarily the mTLS peer
  identity.
- `provider.backend.cluster`: the configured backend cluster the candidate
  resolved to.
- `provider.route.model`: the configured model accepted for the resolved
  route.
- `provider.route.candidate_id`: the edge-selected candidate ID that was
  validated and resolved.
- `overlay.revision`: present only when the edge supplied a serving overlay
  revision that passed syntax and trust-boundary validation. It is
  correlation evidence only, not a provider-local config revision and not an
  authorization decision.

Like `routing.select`, this span never records request or response bodies,
prompts, credentials, authorization headers, cookies, session keys, or raw
request identifiers.

The division of responsibility is intentional:

```text
Edge Praxis core request span
  |
  +-- routing.select          (Praxis AI, intelligent_route)
  `-- upstream hop            (Praxis core)

Provider Praxis core request span
  |
  +-- provider.route          (Praxis AI, provider_route)
  `-- upstream hop            (Praxis core)
```

Each semantic decision span is short-lived and is a sibling of the later
transport span within the same request. When trace context is propagated
between gateways, Praxis core connects the edge provider-hop client span to
the provider request span.

Praxis core owns the complete HTTP span lifetime and transport boundaries at
both the edge and the provider-local listener. Praxis AI records only the
semantic decisions it makes at each hop. This prevents duplicate request
roots, conflicting trace propagation, and multiple exporter runtimes.
