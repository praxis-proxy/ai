# OpenTelemetry routing semantics

Praxis AI can add AI routing decisions to the request trace that Praxis core
creates and exports. Build the proxy with:

```sh
cargo build --release -p praxis-ai-proxy --features opentelemetry
```

The feature is off by default. It installs no exporter and parses no
OpenTelemetry environment variables. Configure exporting, propagation,
sampling, and request lifecycle tracing through Praxis core.

No span from this feature records request or response bodies, prompts,
credentials, authorization headers, cookies, or session keys. `provider.route`
also omits raw request identifiers.

## Routing spans

After `intelligent_route` selects a provider, the feature emits a short
`routing.select` child span with bounded routing identity, admission, locality,
rank, tier, and overlay revision attributes.

After `provider_route` validates the edge-selected candidate and resolves it to
a provider-local backend cluster, the feature emits a short `provider.route`
child span. It records the cluster the request was routed to, which does not
prove that a downstream endpoint, pod, or model server served the request. Its
attributes:

- `provider.id`: the configured provider-boundary identifier for this listener.
  It is a configuration value and may differ from the mTLS peer identity.
- `provider.backend.cluster`: the configured backend cluster the candidate
  resolved to.
- `provider.route.model`: the configured model accepted for the resolved route.
- `provider.route.candidate_id`: the edge-selected candidate ID that was
  validated and resolved.
- `overlay.revision`: present only when the edge supplied a serving overlay
  revision that passed syntax and trust-boundary validation. It is correlation
  evidence only: it is neither a provider-local config revision nor an
  authorization decision.

Span ownership:

```text
Edge Praxis core request span
  |
  +-- routing.select          (Praxis AI, intelligent_route)
  `-- upstream hop            (Praxis core)

Provider Praxis core request span
  |
  +-- provider.route          (Praxis AI, provider_route)
  `-- upstream hop            (Praxis core)

Either request span, when its pipeline runs token_rate_limit
  |
  +-- token_rate_limit        (Praxis AI; admission until the response body ends)
  `-- ...
```

The routing spans are short-lived siblings of the later transport span in the
same request. When trace context is propagated between gateways, Praxis core
connects the edge provider-hop client span to the provider request span.

Praxis core owns the HTTP span lifetime and transport boundaries at both the
edge and the provider-local listener. Praxis AI records only the decisions it
makes at each hop, which avoids duplicate request roots, conflicting trace
propagation, and multiple exporter runtimes.

Agentic callouts follow the same rule. When a listener enables the core
`trace_context` filter, Praxis AI projects only its typed `TraceContext` into
isolated web-search, OGX file-search, OGX file-resolution, and MCP child
requests. Praxis core then emits the same `x-request-id` and W3C trace ID with a
fresh span ID for every outbound hop. No ambient request extensions, provider
credentials, or MCP authorization assertions are copied into trace fields.

## Token rate limiting

With the experimental `token_rate_limit` filter and the `opentelemetry` feature
both enabled, every request that reaches a matching rule gets a request-scoped
`token_rate_limit` span. A hierarchy denial, authentication miss, or backend
error before the rule reservation uses the same span. Fields:

- `token_rate_limit.rule`: configured rule name, or `hierarchy:org`,
  `hierarchy:team`, or `hierarchy:user` for a hierarchy-level outcome.
- `token_rate_limit.algorithm`: `sliding_window` or `token_bucket`.
- `token_rate_limit.estimated_cost`: tokens reserved or considered at admission.
- `token_rate_limit.decision`: `admitted` (reservation made), `denied` (429,
  budget exhausted), `soft_over_quota` (soft mode forwarded over budget with
  annotation), `unauthenticated` (401, no trusted subject to key the budget
  on), `error` (503, the backend failed and the filter failed closed),
  `bypassed` (explicit fail-open before a mutation was dispatched), or
  `unconfirmed` (explicit fail-open after a mutation may have been dispatched).
  Soft decisions align with `praxis_trl_requests_total{result=…}`.
- `token_rate_limit.actual_cost`: provider-reported weighted usage, recorded
  when the response body ends. Absent when the decision was not `admitted` or
  no usage metadata was produced.

The span opens at admission and closes when the response body ends, because
`actual_cost` is known only after provider usage has been read. Its lifetime
therefore covers the streamed response. It does not contain the authenticated
subject, internal budget key, prompt, body, model, or other request-specific
identity. Valkey reconciliation runs asynchronously after response processing
and reports through metrics and accounting logs, not the span.

Quota-source, fail-open, and Sentinel discovery and failover outcomes are
metrics only. See the
[`token_rate_limit` metrics](../filters/control/token_rate_limit.md) for
`praxis_trl_quota_source_missing_total{rule}`,
`praxis_trl_fail_open_admissions_total{rule,backend,outcome}`,
`praxis_trl_sentinel_discovery_total{phase,result}`, and
`praxis_trl_sentinel_failover_total{outcome}`. Fail-open admissions are not
counted in `praxis_trl_backend_errors_total`. The `backend` label is `memory`
or `shared_kv`; Redis and Valkey both report `shared_kv`.
