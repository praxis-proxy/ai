#!/usr/bin/env bash
# Apply a named AuthConfig variant to the running gateway and wait for Authorino to
# report readiness. The acceptance driver additionally probes the expected
# discovery and dispatch decisions: Ready alone can describe an older generation.
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

kubectl --request-timeout=20s apply -f "$cfg" >/dev/null

# Authorino v0.28 has no observedGeneration. This is only a readiness check;
# _switch_authconfig in the acceptance driver probes the new decisions before use.
for _ in $(seq 1 30); do
  ready=$(kubectl --request-timeout=5s -n default get authconfig mcp-edge -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "")
  if [ "$ready" = "True" ]; then
    echo "applied '$variant' (Ready; decision probe required)"; exit 0
  fi
  sleep 1
done
echo "timeout waiting for AuthConfig '$variant' readiness" >&2; exit 1
