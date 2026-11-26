# Operate Praxis AI

Get a [guide](../guides/README.md) working first, then carry its exact binary,
configuration, credentials, and backend assumptions into deployment. This page
is the deployment checklist; the section also includes:

- [Capacity planning](capacity-planning.md): file descriptors and admission.
- [FIPS 140-3 deployment](fips.md).
- Response store maintenance: [schema migration](store/schema-migration.md) and
  [legacy item position repair](store/legacy-item-position-repair.md).
- Upgrades: [migrating to 0.2](upgrades/migrating-to-0.2.md).

## Build and deploy

Record the source revision or image digest and the enabled features with each
configuration. The default build has the standard features; managed Responses
and several identity, token, and cloud helpers need explicit opt-ins. An
unknown filter at startup usually means the binary was built without that
feature. Pick the build from the [feature table](../features.md#cargo-features)
and the [FIPS deployment guide](fips.md).

Before startup, run `praxis-ai --validate -c CONFIG` with the deployment binary
and environment. Validation does not test provider connectivity or permissions.
Keep Host, TLS SNI, trust roots, endpoint, credentials, and model or deployment
names consistent. The loopback recipes allow private addresses on purpose; in
production, allow them only for destinations that need it.

Put public listeners behind your authentication and TLS boundary, and keep
admin and metrics endpoints private. The [Praxis core deployment docs](https://github.com/praxis-proxy/praxis/tree/main/docs)
cover listeners, transport, containers, Kubernetes, and configuration reload.

## Capacity and observability

Start with [file-descriptor planning](capacity-planning.md) and the
[descriptor-budget example](../../examples/configs/file-descriptor-limits.yaml).
Bound connections, concurrency, request bodies, upstream timeouts, buffered
responses, SSE events, and tool callout fan-out. Managed loops also need a
[retained-payload budget](../architecture/responses-retained-payload.md).
Model token throughput alone does not tell you gateway memory or replica
counts; measure your filters, payloads, concurrent streams, backend stalls, and
tool rounds under sustained load and during recovery.

For observability, use the private admin health, readiness, and metrics
endpoints, access logs, request IDs, [token usage](../guides/token-usage.md),
and [OpenTelemetry](../architecture/opentelemetry.md) where enabled. Token
labels are not tenant authorization, and missing provider usage does not mean
zero billing. Keep prompts, secrets, and tool results out of unrestricted debug
logs.

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

A reproducible report includes the revision or digest, enabled features,
redacted configuration, request and response shape, how the stream terminated,
logs with request IDs, and the backend actually exercised. Say whether the
result came from fixture replay or a live provider or GPU run.

## Reload and upgrade

Configuration reload rebuilds filter pipelines, which can reset in-memory
budgets and local routing affinity. Check each filter's lifetime contract;
none should be assumed to coordinate across replicas. Validate the new
configuration with the intended binary before deploying it, and exercise the
affected paths afterward.

Before upgrading, read the release notes and compare your features and
configuration with the target version's docs. For persisted deployments, read
the [store migration requirements](store/schema-migration.md) and
[legacy item repair](store/legacy-item-position-repair.md), and keep a tested
backup and rollback procedure. The [release process](../developing/release.md)
is a maintainer workflow and makes no upgrade compatibility promise. These
Development docs are not part of the published v0.10.0 snapshot until released.
