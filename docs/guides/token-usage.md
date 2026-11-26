# Observe and meter token usage

Turn provider-reported token usage into metrics, label per-tenant counters with
trusted identity, and report usage to an external metering service. Usage
extraction only observes; `external_metering` also rejects requests with `429`
when the service reports no remaining balance. To enforce local limits, see
[Limit token usage](token-limits.md).

## Prerequisites and configuration

Follow the [guide prerequisites](README.md) and start from one of these
complete configurations:

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

`praxis_ai_prompt_tokens_total` and `praxis_ai_output_tokens_total` appear once
the backend supplies usage and Praxis parses it. For streaming Chat
Completions, `stream_usage_inject` asks supporting backends for final SSE
usage, and the counts update when it arrives. Missing usage or a truncated
stream leaves the counts incomplete as billing evidence.

## Usage extraction

`token_count` reads JSON or SSE usage in the configured provider dialect. When
you compose translators, set the dialect the filter actually sees. Cache and
reasoning fields may be missing, and translation can drop details; see the
[compatibility matrix](../conformance/translation-matrix.md). Body-derived
counts arrive after the response headers are sent, so reordering filters does
not make `token_usage_headers` emit them as `Praxis-Token-*` headers. The
Bedrock InvokeModel header-only path is a separate case.

## Per-tenant counters and metering

For bounded per-tenant counters, declare [token_usage_metrics](../filters/observability/token_usage_metrics.md)
before `token_count`, give it explicit tenant and model allowlists, and derive
identity from authenticated metadata rather than client headers. Prometheus
labels are neither authorization nor billing-grade settlement.
[External metering](../../examples/configs/external-metering-file-key.yaml)
adds balance checks and usage reporting to a separate service.

## Filter reference

- [`token_count`](../filters/observability/token_count.md)
- [`stream_usage_inject`](../filters/observability/stream_usage_inject.md)
- [`token_usage_headers`](../filters/observability/token_usage_headers.md)
- [`token_usage_metrics`](../filters/observability/token_usage_metrics.md)
- [`external_metering`](../filters/control/external_metering.md)
- [`model_to_header`](../filters/routing/model_to_header.md)
