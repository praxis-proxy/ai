#!/usr/bin/env bash
# Tear down the Kuadrant acceptance stack (containers + network). Keeps generated
# .run/ artifacts and .stack-env unless --purge is given.
set -u
cd "$(dirname "$0")"
ENGINE="${CONTAINER_ENGINE:-}"
[ -z "$ENGINE" ] && ENGINE="$(command -v podman || command -v docker || true)"
[ -z "$ENGINE" ] && { echo "no container engine" >&2; exit 1; }
"$ENGINE" rm -f envoy authorino kube-apiserver etcd mcp-broker weather-backend jwks >/dev/null 2>&1
"$ENGINE" network rm kuadrant-accept >/dev/null 2>&1
if [ "${1:-}" = "--purge" ]; then rm -rf .run .stack-env; echo "purged .run and .stack-env"; fi
echo "teardown complete"
