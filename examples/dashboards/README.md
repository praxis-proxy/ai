# Example dashboards

## Per-tenant token usage (POC)

[`token-usage-metrics.json`](token-usage-metrics.json) visualizes the opt-in
`praxis_ai_tokens_total{tenant, model, kind}` counter emitted by the
`token_usage_metrics` filter.

1. Run Praxis AI with
   [`../configs/token-usage-metrics.yaml`](../configs/token-usage-metrics.yaml).
2. Configure Prometheus to scrape `http://127.0.0.1:9901/metrics`.
3. Import `token-usage-metrics.json` into Grafana and select that Prometheus
   datasource.

This dashboard is intended for POCs or deployments with a small, known tenant
set. It does not enforce tenant access control. Production tenant isolation and
durable usage/cost reporting are separate concerns tracked by
[`ai#301`](https://github.com/praxis-proxy/ai/issues/301) and
[`ai#211`](https://github.com/praxis-proxy/ai/issues/211).
