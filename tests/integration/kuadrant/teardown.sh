#!/usr/bin/env bash
# Tear down the Kuadrant acceptance stack (containers + network). Keeps generated
# .run/ artifacts and .stack-env unless --purge is given.
set -u
cd "$(dirname "$0")"
BASE="$(pwd)"
recorded_engine=$(sed -n 's/^KUADRANT_CONTAINER_ENGINE=//p' .stack-env 2>/dev/null || true)
ENGINE="${CONTAINER_ENGINE:-$recorded_engine}"
[ -z "$ENGINE" ] && ENGINE="$(command -v podman || command -v docker || true)"
[ -z "$ENGINE" ] && { echo "no container engine" >&2; exit 1; }
for name in envoy authorino kube-apiserver etcd mcp-broker weather-backend jwks; do
  container="kuadrant-accept-$name"
  owner=$("$ENGINE" inspect --format '{{index .Config.Labels "praxis.kuadrant-accept"}}' "$container" 2>/dev/null || true)
  [ "$owner" != "$BASE" ] || "$ENGINE" rm -f "$container" >/dev/null
done
"$ENGINE" network rm kuadrant-accept >/dev/null 2>&1
if [ "${1:-}" = "--purge" ]; then rm -rf .run .stack-env; echo "purged .run and .stack-env"; fi
echo "teardown complete"
