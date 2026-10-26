# Example dashboards

## Per-tenant token usage

[`token-usage-metrics.json`](token-usage-metrics.json) visualizes the opt-in
`praxis_ai_tokens_total{tenant, model, kind}` counter emitted by the
`token_usage_metrics` filter.

1. Run Praxis AI with
   [`../configs/token-usage-metrics.yaml`](../configs/token-usage-metrics.yaml).
2. Configure Prometheus to scrape `http://127.0.0.1:9901/metrics`.
3. Import `token-usage-metrics.json` into Grafana and select that Prometheus
   datasource.

The filter requires authenticated tenant metadata plus explicit tenant/model
allowlists, so client headers cannot create or expand metric series. Protect
the admin endpoint and Grafana/Prometheus with operator authentication and
network policy: labels do not enforce tenant access control. Durable
usage/cost reporting remains a separate concern tracked by
[`ai#301`](https://github.com/praxis-proxy/ai/issues/301) and
[`ai#211`](https://github.com/praxis-proxy/ai/issues/211).
