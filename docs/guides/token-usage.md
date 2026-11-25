# Observe and meter token usage

Extract provider-reported token usage into metrics, choose trusted identity
labels for per-tenant counters, and report usage to an external metering
service. Usage extraction observes; it does not enforce limits.

## Prerequisites and configuration

Follow the [guide prerequisites](README.md). Start from one of these complete
configurations:

- [Screen requests with guardrails](guardrails.md): request screening plus
  `stream_usage_inject` and `token_count` with OpenAI-format usage metrics.
- The [token-counting example](../../examples/configs/token-counting.yaml):
  `model_to_header`, `token_usage_headers`, and `token_count` without
  guardrails.

Both expose the admin metrics endpoint on `127.0.0.1:9901`. Keep it private in
a deployed service.

## Verify

After a completed request, fetch the metrics:

```console
curl --fail-with-body http://127.0.0.1:9901/metrics
```

`/metrics` should contain `praxis_ai_prompt_tokens_total` and
`praxis_ai_output_tokens_total` when usage is supplied and parsed. For
streaming Chat Completions, `stream_usage_inject` requests final SSE usage from
supporting backends; counts become available when that usage arrives, and
missing usage or a truncated stream cannot provide complete billing evidence.

## Usage extraction

`token_count` reads JSON/SSE usage from the configured provider dialect. Choose
the dialect it actually observes when composing translators. Provider-reported
cache and reasoning fields may be missing, and translation can lose details;
review the [compatibility matrix](../conformance/translation-matrix.md).
Body-derived counts arrive after response headers: `token_usage_headers`
does not turn these metrics into `Praxis-Token-*` headers merely by reordering
filters. The Bedrock InvokeModel header-only path is a separate case.

## Per-tenant counters and metering

For bounded per-tenant counters, declare [token_usage_metrics](../filters/observability/token_usage_metrics.md)
before `token_count`, provide explicit tenant/model allowlists, and derive
identity from authenticated metadata. Client headers are insufficient.
Prometheus labels do not provide authorization or billing-grade settlement.
[External metering](../../examples/configs/external-metering-file-key.yaml)
adds balance checks and usage reporting to a separate service.

## Filter reference

- [`token_count`](../filters/observability/token_count.md)
- [`stream_usage_inject`](../filters/observability/stream_usage_inject.md)
- [`token_usage_headers`](../filters/observability/token_usage_headers.md)
- [`token_usage_metrics`](../filters/observability/token_usage_metrics.md)
- [`external_metering`](../filters/control/external_metering.md)
- [`model_to_header`](../filters/routing/model_to_header.md)
