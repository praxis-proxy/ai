#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
compose_file="$script_dir/compose.yaml"
mcp_meta='{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientInfo":{"name":"poc-auth-check","version":"1.0"},"io.modelcontextprotocol/clientCapabilities":{}}'

fastmcp_posts() {
    curl --fail-with-body --silent --show-error http://127.0.0.1:18101/__poc/metrics \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["mcp_posts"])'
}

everything_posts() {
    docker compose -f "$compose_file" logs --no-color everything \
        | python3 -c 'import sys; print(sum("Received MCP POST request" in line for line in sys.stdin))'
}

request_status() {
    local method="$1"
    local id="$2"
    local name="$3"
    local params="$4"
    local bearer="$5"
    local -a name_header=()
    local -a auth_header=()

    if [[ -n "$name" ]]; then
        name_header=(-H "Mcp-Name: $name")
    fi
    if [[ -n "$bearer" ]]; then
        auth_header=(-H "Authorization: Bearer $bearer")
    fi

    curl --silent --show-error --output /dev/null --write-out '%{http_code}' \
        http://127.0.0.1:18080/mcp \
        -H 'Content-Type: application/json' \
        -H 'Mcp-Protocol-Version: 2026-07-28' \
        -H "Mcp-Method: $method" \
        "${name_header[@]}" "${auth_header[@]}" \
        --data-binary "{\"jsonrpc\":\"2.0\",\"id\":$id,\"method\":\"$method\",\"params\":$params}"
}

mcp_token() {
    curl --fail-with-body --silent --show-error \
        --cacert "$script_dir/certs/keycloak.crt" \
        --request POST \
        --header 'Content-Type: application/x-www-form-urlencoded' \
        --data-urlencode 'grant_type=password' \
        --data-urlencode 'client_id=praxis-gateway' \
        --data-urlencode "username=$1" \
        --data-urlencode "password=$2" \
        'https://localhost:18443/realms/mcp-poc/protocol/openid-connect/token' \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])'
}

new_traceparent() {
    python3 -c 'import secrets; print(f"00-{secrets.token_hex(16)}-{secrets.token_hex(8)}-01")'
}

trace_id_from_parent() {
    local remainder="${1#00-}"
    printf '%s\n' "${remainder%%-*}"
}

request_body() {
    local method="$1"
    local id="$2"
    local name="$3"
    local params="$4"
    local bearer="$5"
    local traceparent="${6:-}"
    local -a name_header=()
    local -a auth_header=()
    local -a trace_header=()

    if [[ -n "$name" ]]; then
        name_header=(-H "Mcp-Name: $name")
    fi
    if [[ -n "$bearer" ]]; then
        auth_header=(-H "Authorization: Bearer $bearer")
    fi
    if [[ -n "$traceparent" ]]; then
        trace_header=(-H "traceparent: $traceparent")
    fi

    curl --silent --show-error --max-time 10 \
        http://127.0.0.1:18080/mcp \
        -H 'Content-Type: application/json' \
        -H 'Accept: application/json, text/event-stream' \
        -H 'Mcp-Protocol-Version: 2026-07-28' \
        -H "Mcp-Method: $method" \
        "${name_header[@]}" "${auth_header[@]}" \
        "${trace_header[@]}" \
        --data-binary "{\"jsonrpc\":\"2.0\",\"id\":$id,\"method\":\"$method\",\"params\":$params}"
}

fastmcp_before="$(fastmcp_posts)"
everything_before="$(everything_posts)"
list_params="{\"_meta\":$mcp_meta}"
fastmcp_call='{"name":"fastmcp_echo","arguments":{"message":"denied"},"_meta":'"$mcp_meta"'}'
everything_call='{"name":"everything_get-sum","arguments":{"a":1,"b":2},"_meta":'"$mcp_meta"'}'

missing_discover="$(request_status server/discover 1 '' "$list_params" '')"
invalid_discover="$(request_status server/discover 2 '' "$list_params" 'not-a-jwt')"
missing_list="$(request_status tools/list 3 '' "$list_params" '')"
invalid_list="$(request_status tools/list 4 '' "$list_params" 'not-a-jwt')"
missing_fastmcp="$(request_status tools/call 5 fastmcp_echo "$fastmcp_call" '')"
invalid_fastmcp="$(request_status tools/call 6 fastmcp_echo "$fastmcp_call" 'not-a-jwt')"
missing_everything="$(request_status tools/call 7 everything_get-sum "$everything_call" '')"
invalid_everything="$(request_status tools/call 8 everything_get-sum "$everything_call" 'not-a-jwt')"

fastmcp_after="$(fastmcp_posts)"
everything_after="$(everything_posts)"
printf 'server/discover missing/invalid: %s/%s; tools/list: %s/%s; FastMCP tools/call: %s/%s; Everything tools/call: %s/%s\n' \
    "$missing_discover" "$invalid_discover" "$missing_list" "$invalid_list" \
    "$missing_fastmcp" "$invalid_fastmcp" "$missing_everything" "$invalid_everything"
printf 'MCP backend POST counts: FastMCP %s -> %s; Everything %s -> %s\n' \
    "$fastmcp_before" "$fastmcp_after" "$everything_before" "$everything_after"

test "$missing_discover" = 401
test "$invalid_discover" = 401
test "$missing_list" = 401
test "$invalid_list" = 401
test "$missing_fastmcp" = 401
test "$invalid_fastmcp" = 401
test "$missing_everything" = 401
test "$invalid_everything" = 401
test "$fastmcp_before" = "$fastmcp_after"
test "$everything_before" = "$everything_after"

limited_token="$(mcp_token limited limited)"
restricted_list_traceparent="${MCP_POC_RESTRICTED_LIST_TRACEPARENT:-$(new_traceparent)}"
restricted_list_trace_id="$(trace_id_from_parent "$restricted_list_traceparent")"
restricted_list="$(request_body tools/list 9 '' "$list_params" "$limited_token" "$restricted_list_traceparent")"
printf '%s' "$restricted_list" \
    | python3 -c 'import json,sys; names=[tool["name"] for tool in json.load(sys.stdin)["result"]["tools"]]; assert names == ["fastmcp_echo"], names'
printf 'Restricted user tools/list trace: %s\n' "$restricted_list_trace_id"

fastmcp_before_restricted="$(fastmcp_posts)"
everything_before_restricted="$(everything_posts)"
restricted_echo_call='{"name":"fastmcp_echo","arguments":{"message":"restricted-fastmcp"},"_meta":'"$mcp_meta"'}'
restricted_echo="$(request_body tools/call 10 fastmcp_echo "$restricted_echo_call" "$limited_token")"
printf '%s' "$restricted_echo" \
    | python3 -c 'import json,sys; texts=[item.get("text","") for item in json.load(sys.stdin)["result"].get("content",[])]; assert any("fastmcp-echo: restricted-fastmcp" in text for text in texts), texts'

restricted_denial="$(request_body tools/call 11 everything_get-sum "$everything_call" "$limited_token")"
printf '%s' "$restricted_denial" \
    | python3 -c 'import json,sys; response=json.load(sys.stdin); assert response["error"]["code"] == -32003, response'
fastmcp_after_restricted="$(fastmcp_posts)"
everything_after_restricted="$(everything_posts)"
printf 'restricted user tools/list: fastmcp_echo only; FastMCP call allowed; Everything call denied (-32003)\n'
printf 'Restricted user MCP POST counts: FastMCP %s -> %s; Everything %s -> %s\n' \
    "$fastmcp_before_restricted" "$fastmcp_after_restricted" \
    "$everything_before_restricted" "$everything_after_restricted"
test "$fastmcp_after_restricted" -eq "$((fastmcp_before_restricted + 1))"
test "$everything_before_restricted" = "$everything_after_restricted"
