# Capacity Planning: File Descriptors

Praxis AI runs on the Praxis proxy core and inherits its
file descriptor handling. At startup it raises its soft
`RLIMIT_NOFILE` to the hard limit (or pins it with
`runtime.max_open_files`) and warns when the limit is
small. Near the limit it answers new HTTP requests with
`503` and `Retry-After: 1` and closes new TCP connections
instead of failing them (`runtime.shed_on_fd_pressure`).
It keeps serving cached DNS answers when a lookup fails
for lack of descriptors, and exports
`praxis_process_open_fds` and `praxis_process_max_fds`.

The [Praxis capacity planning guide] explains those
mechanisms, the core budget (two descriptors per proxied
request, idle keep-alive clients, connection pools), and
how to raise the hard limit in containers. This page
covers what Praxis AI adds to that budget.

[Praxis capacity planning guide]: https://github.com/praxis-proxy/praxis/blob/main/docs/operating/capacity-planning.md

## Metering

`external_metering` makes two callouts for every metered
request, and each holds a descriptor while it runs:

- The **balance check** runs before the request is
  proxied, so an in-flight metered request holds a
  client, a metering, and then an upstream descriptor.
- The **usage report** runs in the background after the
  response, and can overlap the same client's next
  request.

Budget three descriptors per concurrent metered request,
and four when clients send requests back to back. In
tests, 128 metered streaming keep-alive clients peaked at
about 420 descriptors over an idle baseline of about 40.

The metering host is resolved through the proxy's DNS
cache, the one upstream clusters use, so lookups do not
cost descriptors per request.

## Other Callouts

Every other [outbound callout](architecture/outbound-callouts.md)
(guardrails, web search, file resolution, token fetches,
MCP) also holds a descriptor while it runs.

## Bounding Callouts

| Setting | Effect |
| --------------------------------------- | ------------------------------------------------ |
| `runtime.subrequest_max_connections` | Cap concurrent callouts on the shared sub-request client (unbounded by default) |
| `runtime.subrequest_pool_size` | Idle callout connections kept for reuse |

Metering and most other callout filters share this
client. Past the cap, a callout waits for a slot within
its own timeout and then fails: a balance check follows
`fail_open`, and a usage report is dropped and counted in
`praxis_ai_metering_report_failures_total`. Size the cap
for peak callout concurrency rather than as a throttle.

## Stores

Each response or conversation store backend keeps a
connection pool, and every pooled connection holds at
least one descriptor. Budget its `pool.max_connections`
(default 10).

## The Startup Recommendation

The startup warning (`open file limit is low for this
configuration`) budgets two descriptors per connection
allowed by `max_connections`, the idle pools, and a fixed
baseline. It does not count callouts, so add
`subrequest_max_connections` (or one to two descriptors
per concurrent metered request when it is unset) and the
store pools.

## Monitoring

The admin endpoint's `/metrics` carries
`praxis_process_open_fds`, `praxis_process_max_fds`, and
`praxis_overload_rejects_total{reason="file_descriptors"}`
(shed requests), and `praxis_ai_metering_report_failures_total`
counts usage reports that could not be delivered. The
Praxis guide also mentions `GET /api/stats`; the Praxis AI
admin endpoint does not serve it.

## Example

[examples/configs/file-descriptor-limits.yaml] sizes a
metered gateway with every setting on this page.

[examples/configs/file-descriptor-limits.yaml]: ../examples/configs/file-descriptor-limits.yaml
