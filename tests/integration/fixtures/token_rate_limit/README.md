# Sentinel topology fixture

`start-sentinel-topology.sh` starts one writable primary, one replica, and
three Sentinel processes using a **digest-pinned** Redis or Valkey image. It
exports `TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS` and runs a command while the
fixture is alive; all containers and temporary configuration are removed when
the command exits.

Example Redis lane:

```console
make test-token-rate-limit-sentinel-integration \
  SENTINEL_PRODUCT=redis \
  SENTINEL_IMAGE=docker.io/library/redis:8.10.2-alpine@sha256:3811787313eba226a2ef38658c6ccb91cd5e110edc89c37767de373120a0e5a0
```

Example Red Hat Valkey lane (the registry pull secret must already be loaded
into the container engine):

```console
make test-token-rate-limit-sentinel-integration \
  SENTINEL_PRODUCT=valkey \
  SENTINEL_IMAGE=registry.redhat.io/rhel10/valkey-8@sha256:5929be16ac020c4851d8dc1f5ead6d48c6da066a9a74349711186925ebd553b9
```

The fixture intentionally does not use mutable tags. Failover, replication
offset checks, authentication/TLS profiles, and ambiguous-reply fault
injection remain separate qualification phases. The automated smoke lane also
checks dead-Sentinel endpoint convergence and both sliding-window and
token-bucket quota denial; this fixture owns deterministic topology
startup/readiness and lifecycle cleanup rather than the complete release
matrix.

Readiness snapshots the primary replication offset and waits for an attached
replica to acknowledge at least that offset. The two observed offsets need not
be equal: Sentinel traffic may advance the log between INFO calls.

PR CI runs credential-free smoke against official Redis and an immutable
upstream Valkey image. Upstream Valkey results are **not** Red Hat image
qualification. The exact Red Hat Valkey release target above must also pass the
complete suite in an authorized environment, with its reviewed source SHA,
image digest, server version, and results recorded separately. This preserves
equal backend support without exposing registry credentials to fork PR code;
credentialed Red Hat CI automation remains a follow-up.

The smoke lane also runs the existing standalone state, reconciliation, and
cross-gateway identity/quota contracts. The fixture invokes
`run-standalone-contracts.sh`, which requires Python 3 to encode the ephemeral
data credentials and selects the existing Valkey-named contract targets for
either wire-compatible product. Its proxy binary includes `basic-auth-filter`
as well as `token-rate-limit-filter` for the authenticated-subject contract.
These backend-backed results are compatibility evidence, not hermetic unit
coverage.
The standalone phase stops only its own Sentinels and removes Sentinel endpoint
variables so the fixed-endpoint contracts cannot depend on failover. Each HA
phase starts a fresh topology with all three Sentinels active.

Set `SENTINEL_DEAD_ENDPOINT=redis://127.0.0.1:1` to prepend an unavailable
Sentinel endpoint. The real client must skip it and discover through the
remaining healthy endpoints.

Set `SENTINEL_FAILOVER_BEFORE_COMMAND=1` to enable the planned-failover test.
The integration test first warms existing proxy connections and admits quota
for both algorithms. It snapshots the primary replication offset after these
admissions and waits for the replica to acknowledge that offset before
triggering Sentinel promotion. After the replica becomes the reported primary,
the test checks warm-connection recovery and continued sliding-window and
token-bucket quota denial without extra upstream calls.

Set `SENTINEL_REPLICATION_HEALTH_TEST=1` with
`SENTINEL_ON_FAILURE=closed` or `open` to stop the replica after convergence,
enable `min-replicas-to-write`, and exercise the corresponding NOREPLICAS
admission behavior.

Set `TOKEN_RATE_LIMIT_SENTINEL_DEBUG=1` to print container logs when topology
readiness or planned failover fails.

The fixture generates independent random data-node and Sentinel passwords for
each run. For a diagnostic run, set either or both of
`TOKEN_RATE_LIMIT_SENTINEL_DATA_PASSWORD` and
`TOKEN_RATE_LIMIT_SENTINEL_PASSWORD` in the fixture environment to override
them. The script configures separate credentials, passes CLI credentials over
stdin, and exports them only to the child test process; do not place secrets in
the repository or command-line arguments.
