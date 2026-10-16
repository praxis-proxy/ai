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
  SENTINEL_IMAGE=registry.redhat.io/rhel10/valkey-8@sha256:77a0c5aad7e39c35e2d575073e8d8b42417e2cbdba0f5626db4f59ade84e1fbe
```

The fixture intentionally does not use mutable tags. Failover, replication
offset checks, authentication/TLS profiles, and ambiguous-reply fault
injection remain separate qualification phases. The automated smoke lane also
checks dead-Sentinel endpoint convergence and both sliding-window and
token-bucket quota denial; this fixture owns deterministic topology
startup/readiness and lifecycle cleanup rather than the complete release
matrix.

Set `SENTINEL_DEAD_ENDPOINT=redis://127.0.0.1:1` to prepend an unavailable
Sentinel endpoint. The real client must skip it and discover through the
remaining healthy endpoints.

Set `SENTINEL_FAILOVER_BEFORE_COMMAND=1` to request a planned Sentinel
failover after readiness and before the child test command. The fixture waits
for replication offsets to converge before promotion, then waits for the
replica to become the reported primary before starting the test.

Set `SENTINEL_REPLICATION_HEALTH_TEST=1` with
`SENTINEL_ON_FAILURE=closed` or `open` to stop the replica after convergence,
enable `min-replicas-to-write`, and exercise the corresponding NOREPLICAS
admission behavior.

Set `TOKEN_RATE_LIMIT_SENTINEL_DEBUG=1` to print container logs when topology
readiness or planned failover fails.

For an authenticated run, set either or both of
`TOKEN_RATE_LIMIT_SENTINEL_DATA_PASSWORD` and
`TOKEN_RATE_LIMIT_SENTINEL_PASSWORD` in the fixture environment. The script
configures separate data-node and Sentinel credentials, passes CLI credentials
over stdin, and exports them only to the child test process; do not place them
in the repository or command-line arguments.

Controlled self-signed TLS qualification must additionally set
`PRAXIS_TOKEN_RATE_LIMIT_ALLOW_INSECURE_TLS=1`; certificate verification is
otherwise required.
