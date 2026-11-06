#!/usr/bin/env bash
# Apply a named AuthConfig variant to the running gateway and wait for Authorino to
# reconcile it (Ready=True, and the generation advanced when the spec changed).
# Usage: switch_authconfig.sh <filtered|opendisco|revoked>
#   filtered  — per-user discovery filtering (default policy)
#   opendisco — tools exposed to all; per-call authz still enforced (3b proof)
#   revoked   — userA's get_forecast removed (live-revocation proof, item 6)
set -euo pipefail
cd "$(dirname "$0")"
[ -f .stack-env ] || { echo "stack not up (.stack-env missing)" >&2; exit 1; }
set -a; . ./.stack-env; set +a
export KUBECONFIG="$KUADRANT_KUBECONFIG"

variant="${1:?usage: switch_authconfig.sh <filtered|opendisco|revoked>}"
cfg="configs/authconfig-${variant}.yaml"
[ -f "$cfg" ] || { echo "unknown variant '$variant' ($cfg not found)" >&2; exit 1; }

gen_before=$(kubectl --request-timeout=10s -n default get authconfig mcp-edge -o jsonpath='{.metadata.generation}' 2>/dev/null || echo 0)
kubectl --request-timeout=20s apply -f "$cfg" >/dev/null

for _ in $(seq 1 30); do
  gen=$(kubectl --request-timeout=5s -n default get authconfig mcp-edge -o jsonpath='{.metadata.generation}' 2>/dev/null || echo 0)
  ready=$(kubectl --request-timeout=5s -n default get authconfig mcp-edge -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "")
  if [ "$ready" = "True" ] && [ "$gen" != "$gen_before" ]; then
    echo "switched to '$variant' (generation $gen_before -> $gen, Ready)"; exit 0
  fi
  sleep 1
done
# Re-applying an identical spec does not bump generation; accept Ready in that case.
ready=$(kubectl --request-timeout=5s -n default get authconfig mcp-edge -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "")
[ "$ready" = "True" ] && { echo "switched to '$variant' (Ready, generation unchanged)"; exit 0; }
echo "timeout waiting for AuthConfig '$variant' to reconcile" >&2; exit 1
