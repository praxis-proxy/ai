#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
cd "$repo_root"

keycloak_ca="$script_dir/certs/keycloak.crt"

require_http() {
    local url="$1"
    local ca_file="${2:-}"
    local -a curl_args=(--fail-with-body --silent --show-error --max-time 3 --output /dev/null)
    if [[ -n "$ca_file" ]]; then
        curl_args+=(--cacert "$ca_file")
    fi
    if ! curl "${curl_args[@]}" "$url"; then
        printf 'service unavailable: %s\nStart the Compose services and source-built Praxis as documented in %s/README.md\n' \
            "$url" "$script_dir" >&2
        exit 1
    fi
}

[[ -s "$script_dir/catalog.json" ]] || {
    printf 'missing %s/catalog.json; run the manual discovery command in README.md first\n' "$script_dir" >&2
    exit 1
}
[[ -s "$keycloak_ca" ]] || {
    printf 'missing %s; start the Compose Keycloak service first\n' "$keycloak_ca" >&2
    exit 1
}

require_http 'https://localhost:18443/realms/mcp-poc/.well-known/openid-configuration' "$keycloak_ca"
require_http 'http://127.0.0.1:18101/__poc/metrics'
require_http 'http://127.0.0.1:13200/ready'
require_http 'http://127.0.0.1:13000/api/health'
require_http 'http://127.0.0.1:19090/-/ready'
TOKEN="$(curl --fail-with-body --silent --show-error \
    --cacert "$keycloak_ca" \
    --request POST \
    --header 'Content-Type: application/x-www-form-urlencoded' \
    --data-urlencode 'grant_type=password' \
    --data-urlencode 'client_id=praxis-gateway' \
    --data-urlencode 'username=demo' \
    --data-urlencode 'password=demo' \
    'https://localhost:18443/realms/mcp-poc/protocol/openid-connect/token' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')"
MCP_META='{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientInfo":{"name":"mcp-poc-smoke","version":"1.0"},"io.modelcontextprotocol/clientCapabilities":{}}'

# Sampling is configured in the Praxis process. These parent contexts give
# the list and call requests trace IDs that can be looked up after the smoke run.
new_traceparent() {
    python3 -c 'import secrets; print(f"00-{secrets.token_hex(16)}-{secrets.token_hex(8)}-01")'
}
trace_id_from_parent() {
    local remainder="${1#00-}"
    printf '%s\n' "${remainder%%-*}"
}
TOOLS_LIST_TRACEPARENT="$(new_traceparent)"
FASTMCP_CALL_TRACEPARENT="$(new_traceparent)"
EVERYTHING_CALL_TRACEPARENT="$(new_traceparent)"
RESTRICTED_TOOLS_LIST_TRACEPARENT="$(new_traceparent)"

mcp_post() {
    local url="$1" method="$2" id="$3" params="$4"
    shift 4
    curl --fail-with-body --silent --show-error "$url" \
        -H 'Content-Type: application/json' \
        -H 'Accept: application/json, text/event-stream' \
        -H 'Mcp-Protocol-Version: 2026-07-28' \
        -H "Mcp-Method: $method" "$@" \
        --data-binary "{\"jsonrpc\":\"2.0\",\"id\":$id,\"method\":\"$method\",\"params\":$params}"
}

mcp_gateway_post() {
    local url="$1" method="$2" id="$3" params="$4"
    shift 4
    mcp_post "$url" "$method" "$id" "$params" \
        -H "Authorization: Bearer $TOKEN" "$@"
}

show_tools() {
    python3 -c 'import json,sys; tools=json.load(sys.stdin)["result"]["tools"]; print(f"{len(tools)} tools: " + ", ".join(tool["name"] for tool in tools))'
}

show_text_result() {
    python3 -c 'import json,sys; result=json.load(sys.stdin)["result"]; print("\n".join(item["text"] for item in result.get("content",[]) if "text" in item))'
}

printf '%s\n' 'Direct upstream tools/list:'
mcp_post 'http://127.0.0.1:18101/mcp' tools/list 1 "{\"_meta\":$MCP_META}" | show_tools
mcp_post 'http://127.0.0.1:18102/mcp' tools/list 2 "{\"_meta\":$MCP_META}" | show_tools

printf '%s\n' 'Direct upstream tools/call:'
mcp_post 'http://127.0.0.1:18101/mcp' tools/call 3 \
    "{\"name\":\"echo\",\"arguments\":{\"message\":\"direct-fastmcp\"},\"_meta\":$MCP_META}" \
    -H 'Mcp-Name: echo' | show_text_result
mcp_post 'http://127.0.0.1:18102/mcp' tools/call 4 \
    "{\"name\":\"get-sum\",\"arguments\":{\"a\":19,\"b\":23},\"_meta\":$MCP_META}" \
    -H 'Mcp-Name: get-sum' | show_text_result

printf '%s\n' 'Through Praxis tools/discover and tools/list:'
mcp_gateway_post 'http://127.0.0.1:18080/mcp' server/discover 5 "{\"_meta\":$MCP_META}" \
    | python3 -c 'import json,sys; print("supported versions:", json.load(sys.stdin)["result"]["supportedVersions"])'
mcp_gateway_post 'http://127.0.0.1:18080/mcp' tools/list 6 "{\"_meta\":$MCP_META}" \
    -H "traceparent: $TOOLS_LIST_TRACEPARENT" | show_tools

printf '%s\n' 'Through Praxis prefixed tools/call:'
mcp_gateway_post 'http://127.0.0.1:18080/mcp' tools/call 7 \
    "{\"name\":\"fastmcp_echo\",\"arguments\":{\"message\":\"through-fastmcp\"},\"_meta\":$MCP_META}" \
    -H 'Mcp-Name: fastmcp_echo' \
    -H "traceparent: $FASTMCP_CALL_TRACEPARENT" | show_text_result
mcp_gateway_post 'http://127.0.0.1:18080/mcp' tools/call 8 \
    "{\"name\":\"everything_get-sum\",\"arguments\":{\"a\":19,\"b\":23},\"_meta\":$MCP_META}" \
    -H 'Mcp-Name: everything_get-sum' \
    -H "traceparent: $EVERYTHING_CALL_TRACEPARENT" | show_text_result

MCP_POC_RESTRICTED_LIST_TRACEPARENT="$RESTRICTED_TOOLS_LIST_TRACEPARENT" \
    "$script_dir/verify-auth-denials.sh"

wait_for_tempo_trace() {
    local trace_id="$1"
    local trace_url="http://127.0.0.1:13200/api/traces/$trace_id"
    for _ in {1..20}; do
        if curl --fail --silent --output /dev/null --max-time 2 "$trace_url"; then
            return 0
        fi
        sleep 1
    done
    printf 'trace not found in Tempo after 20 seconds: %s\n' "$trace_id" >&2
    return 1
}

printf '%s\n' 'Checking traces in Tempo:'
check_trace() {
    local label="$1"
    local traceparent="$2"
    local trace_id
    trace_id="$(trace_id_from_parent "$traceparent")"
    wait_for_tempo_trace "$trace_id"
    printf '  %s: %s\n' "$label" "$trace_id"
}
check_trace 'demo tools/list' "$TOOLS_LIST_TRACEPARENT"
check_trace 'FastMCP tools/call' "$FASTMCP_CALL_TRACEPARENT"
check_trace 'Everything tools/call' "$EVERYTHING_CALL_TRACEPARENT"
check_trace 'restricted-user tools/list authorization filter' "$RESTRICTED_TOOLS_LIST_TRACEPARENT"

wait_for_praxis_metric() {
    local query='sum(praxis_http_requests_total{job="praxis"})'
    local response
    local value
    for _ in {1..20}; do
        if response="$(curl --fail --silent --show-error --max-time 2 --get \
            --data-urlencode "query=$query" \
            'http://127.0.0.1:19090/api/v1/query')" \
            && value="$(python3 -c 'import json,sys; result=json.load(sys.stdin); assert result["status"] == "success"; samples=result["data"]["result"]; assert samples; print(samples[0]["value"][1])' <<<"$response" 2>/dev/null)" \
            && python3 -c 'import sys; assert float(sys.stdin.read()) > 0' <<<"$value" 2>/dev/null; then
            printf 'Praxis HTTP request counter scraped by Prometheus: %s\n' "$value"
            return 0
        fi
        sleep 1
    done
    printf 'Praxis HTTP request metrics were not scraped by Prometheus after 20 seconds\n' >&2
    return 1
}

wait_for_praxis_metric
