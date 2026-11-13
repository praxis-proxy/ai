# Operate Praxis AI

Start with a working [recipe](recipes/README.md), then carry its exact binary,
configuration, credentials, and backend assumptions into deployment.

## Build and deploy

Record the source revision or image digest and enabled features with each
configuration. The default build selects standard features; managed Responses
and several identity/token/cloud helpers require explicit opt-ins. An unknown
filter at startup usually means the selected binary lacks the configured
feature. Use the [feature table](features.md#cargo-features) and
[FIPS deployment guide](fips.md) for the appropriate build.

Run `praxis-ai --validate -c CONFIG` using the deployment binary and environment
before startup. Validation is not a provider connectivity or permission test.
Keep Host, TLS SNI, trust roots, endpoint, credentials, and model/deployment
names consistent. Native loopback recipes deliberately allow private addresses;
carry that exception into production only for destinations that need it.

Bind public listeners behind your chosen authentication and TLS boundary.
Keep admin/metrics endpoints private. Use the [Praxis core deployment docs](https://github.com/praxis-proxy/praxis/tree/main/docs)
for base listener, transport, container, Kubernetes, and configuration reload
mechanics; this repository's AI-specific recipes do not replace them.

## Capacity and observability

Start with [file-descriptor planning](capacity-planning.md) and the
[descriptor-budget example](../examples/configs/file-descriptor-limits.yaml).
Bound connections, concurrency, request bodies, upstream timeouts, buffered
responses, SSE events, and tool callout fan-out. Managed loops also need the
[retained-payload budget](architecture/responses-retained-payload.md).
Do not infer gateway memory sizing or replica scaling rules from model token
throughput alone. Measure your selected filters, payloads, concurrent streams,
backend stalls, and tool rounds under sustained load and recovery.

Use private admin health/readiness/metrics endpoints, access logs, request IDs,
[token usage](recipes/guardrails-token-usage.md), and
[OpenTelemetry](architecture/opentelemetry.md) where enabled. Token labels are
not tenant authorization; missing provider usage is not zero billing.
Avoid collecting prompts, secrets, or tool results in unrestricted debug logs.

## Troubleshoot by boundary

| Symptom | Check |
| --- | --- |
| Listener unavailable | Startup log, bind address/port, selected config, process health |
| Unknown filter | Exact binary features, not just the source tree's filter list |
| Configuration validates but inference fails | DNS, endpoint/Host/SNI, trust roots, backend readiness, model availability |
| 401/403 | Distinguish gateway auth, inference credentials, tool credentials, and provider permissions |
| Wrong provider/model | Inspect selection metadata and backend recordings; check alias path match, fallback, catalog, and affinity |
| Translation 400 | Rejected fields, version contract, supported content blocks, explicit lossy opt-ins |
| Truncated or failed stream | Backend terminal events, idle/total deadlines, SSE limits, adapter and loop finalization; an HTTP 200 alone is insufficient |
| No token headers/counters | Body/header timing, configured response dialect, backend usage field, and SSE usage opt-in |
| State unavailable or wrong owner | Store feature/backend, database permissions/migrations, authenticated owner context, local versus provider-owned route |
| Tools never execute | Model call selection, dispatcher, loop owner/IRR transition, approval, destination access and credentials |

For a reproducible report, include revision/digest, enabled features, redacted
configuration, request and response shape, streaming terminal outcome, logs
with request IDs, and the backend actually exercised. Separate fixture replay
from a live provider or GPU run.

## Reload and upgrade

Configuration reload rebuilds filter pipelines; in-memory budgets and local
routing affinity can reset. Check the relevant filter's lifetime contract and
do not assume it coordinates replicas. Validate the replacement configuration
with the intended binary before deployment and exercise affected paths afterward.

Read release notes and compare your enabled features and configuration against
the destination version's docs. Review [store migration requirements](store/schema-migration.md)
and [legacy item repair](store/legacy-item-position-repair.md) before changing
persisted deployments; retain a tested backup and rollback procedure.
The [release process](release.md) is a maintainer workflow, not an operator
upgrade compatibility promise. Development documentation added here is not
part of the published v0.10.0 snapshot until separately released and imported.
