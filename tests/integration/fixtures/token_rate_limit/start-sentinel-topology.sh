#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Praxis Contributors

# Start a digest-pinned primary/replica and three Sentinels, export connection
# variables, run an optional command, and clean every container on exit.
set -Eeuo pipefail

product=""
image=""
base_port=26379
env_file=""
data_password=${TOKEN_RATE_LIMIT_SENTINEL_DATA_PASSWORD:-}
sentinel_password=${TOKEN_RATE_LIMIT_SENTINEL_PASSWORD:-}
env_file_written=0
debug=${TOKEN_RATE_LIMIT_SENTINEL_DEBUG:-}
replication_health_test=${TOKEN_RATE_LIMIT_SENTINEL_REPLICATION_HEALTH_TEST:-}
on_failure=${TOKEN_RATE_LIMIT_SENTINEL_ON_FAILURE:-}
keepalive_pid=""
fixture_owner_pid=${BASHPID:-$$}
cleanup_started=0

random_password() {
    openssl rand -hex 24
}

# Authenticated fixture coverage is the default, but callers may override the
# values for a diagnostic run. Keep the two generated credentials independent
# and export them only to the child process below.
[[ -n "$data_password" ]] || data_password=$(random_password)
[[ -n "$sentinel_password" ]] || sentinel_password=$(random_password)
[[ "$data_password" != "$sentinel_password" ]] || {
    echo "Sentinel and data credentials must be distinct" >&2
    exit 2
}
while (($#)); do
    case "$1" in
        --product) product=$2; shift 2 ;;
        --image) image=$2; shift 2 ;;
        --base-port) base_port=$2; shift 2 ;;
        --env-file) env_file=$2; shift 2 ;;
        --) shift; break ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ "$product" != redis && "$product" != valkey ]]; then
    echo "--product must be redis or valkey" >&2
    exit 2
fi
if [[ "$image" != *@sha256:* ]]; then
    echo "--image must include an immutable @sha256 digest" >&2
    exit 2
fi
if [[ -n "$on_failure" && "$on_failure" != closed && "$on_failure" != open ]]; then
    echo "TOKEN_RATE_LIMIT_SENTINEL_ON_FAILURE must be closed or open" >&2
    exit 2
fi
engine=${CONTAINER_ENGINE:-}
if [[ -z "$engine" ]]; then
    command -v podman >/dev/null 2>&1 && engine=podman || engine=docker
fi
command -v "$engine" >/dev/null 2>&1 || { echo "container engine not found: $engine" >&2; exit 1; }

primary_port=$base_port
replica_port=$((base_port + 1))
sentinel_1_port=$((base_port + 2))
sentinel_2_port=$((base_port + 3))
sentinel_3_port=$((base_port + 4))
prefix="praxis-trl-sentinel-${product}-$$"
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/praxis-trl-sentinel.XXXXXX")
mount_suffix=""
[[ "$engine" == podman* ]] && mount_suffix=":Z"

if [[ "$product" == redis ]]; then
    server_binary=redis-server
    cli_binary=redis-cli
    config_path=/etc/redis/redis.conf
else
    server_binary=valkey-server
    cli_binary=valkey-cli
    config_path=/etc/valkey/valkey.conf
fi

cleanup() {
    # Subshells can inherit EXIT handling while a signal races with a fork.
    # Only the original shell owns the topology, and teardown runs once.
    if [[ ${BASHPID:-$$} != "$fixture_owner_pid" || ${BASH_SUBSHELL:-0} != 0 ]] || ((cleanup_started)); then
        return 0
    fi
    cleanup_started=1
    trap - EXIT
    set +e
    local pid
    # Include a sleeper started just before a signal, even if the handler ran
    # before its PID could be assigned to keepalive_pid.
    for pid in $(jobs -pr); do
        kill "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
    done
    for role in primary replica sentinel-1 sentinel-2 sentinel-3; do
        "$engine" rm -f "${prefix}-${role}" >/dev/null 2>&1
    done
    rm -rf "$tmp_dir"
    if ((env_file_written)); then
        rm -f -- "$env_file"
    fi
}

debug_logs() {
    if [[ "$debug" == 1 ]]; then
        for role in primary replica sentinel-1 sentinel-2 sentinel-3; do
            echo "--- ${prefix}-${role} logs ---" >&2
            "$engine" logs "${prefix}-${role}" >&2 || true
        done
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cat >"$tmp_dir/primary.conf" <<EOF
port $primary_port
bind 127.0.0.1
protected-mode no
save ""
appendonly no
loglevel notice
enable-debug-command local
EOF
if [[ -n "$data_password" ]]; then
    printf 'requirepass %s\n' "$data_password" >>"$tmp_dir/primary.conf"
fi
cat >"$tmp_dir/replica.conf" <<EOF
port $replica_port
bind 127.0.0.1
protected-mode no
save ""
appendonly no
replicaof 127.0.0.1 $primary_port
loglevel notice
enable-debug-command local
EOF
if [[ -n "$data_password" ]]; then
    printf 'requirepass %s\nmasterauth %s\n' "$data_password" "$data_password" >>"$tmp_dir/replica.conf"
fi
for index in 1 2 3; do
    port=$((base_port + index + 1))
    cat >"$tmp_dir/sentinel-${index}.conf" <<EOF
port $port
bind 127.0.0.1
protected-mode no
sentinel monitor mymaster 127.0.0.1 $primary_port 2
sentinel down-after-milliseconds mymaster 1000
sentinel failover-timeout mymaster 5000
sentinel parallel-syncs mymaster 1
loglevel notice
EOF
    if [[ -n "$sentinel_password" ]]; then
        printf 'requirepass %s\n' "$sentinel_password" >>"$tmp_dir/sentinel-${index}.conf"
    fi
    if [[ -n "$data_password" ]]; then
        printf 'sentinel auth-pass mymaster %s\n' "$data_password" >>"$tmp_dir/sentinel-${index}.conf"
    fi
done
chmod 644 "$tmp_dir/primary.conf" "$tmp_dir/replica.conf"
# Sentinel rewrites its monitored topology into this file. The parent
# directory is 0700, the files are removed by cleanup, and no configuration
# path is exposed outside this fixture's private temporary directory.
chmod 666 "$tmp_dir"/sentinel-*.conf

run_server() {
    local role=$1 config=$2
    shift 2
    "$engine" run -d --name "${prefix}-${role}" --network host --pull=never \
        -v "$config:$config_path$mount_suffix" "$image" "$server_binary" "$config_path" "$@" >/dev/null
}
run_server primary "$tmp_dir/primary.conf"
run_server replica "$tmp_dir/replica.conf"
run_server sentinel-1 "$tmp_dir/sentinel-1.conf" --sentinel
run_server sentinel-2 "$tmp_dir/sentinel-2.conf" --sentinel
run_server sentinel-3 "$tmp_dir/sentinel-3.conf" --sentinel

data_cli() {
    local container=$1
    shift
    if [[ -n "$data_password" ]]; then
        # Pass the secret over stdin instead of putting it in the container
        # engine command line or the host process list.
        printf '%s\n' "$data_password" |
            "$engine" exec -i "$container" sh -c \
                'IFS= read -r REDISCLI_AUTH; export REDISCLI_AUTH; exec "$@"' \
                sh "$cli_binary" "$@"
    else
        "$engine" exec "$container" "$cli_binary" "$@"
    fi
}
sentinel_cli() {
    local container=$1
    shift
    if [[ -n "$sentinel_password" ]]; then
        # Pass the secret over stdin instead of putting it in the container
        # engine command line or the host process list.
        printf '%s\n' "$sentinel_password" |
            "$engine" exec -i "$container" sh -c \
                'IFS= read -r REDISCLI_AUTH; export REDISCLI_AUTH; exec "$@"' \
                sh "$cli_binary" "$@"
    else
        "$engine" exec "$container" "$cli_binary" "$@"
    fi
}

info_field() {
    local container=$1 port=$2 field=$3
    data_cli "$container" -p "$port" INFO replication |
        awk -F: -v field="$field" '$1 == field {gsub("\r", "", $2); print $2; exit}'
}

wait_for_replication_catchup() {
    local deadline=$((SECONDS + 30))
    local primary_offset replica_offset replica_link
    while ((SECONDS < deadline)); do
        primary_offset=$(info_field "${prefix}-primary" "$primary_port" master_repl_offset)
        replica_offset=$(info_field "${prefix}-replica" "$replica_port" master_repl_offset)
        replica_link=$(info_field "${prefix}-replica" "$replica_port" master_link_status)
        # Sentinel traffic can advance replication between these INFO calls.
        # The replica must acknowledge the observed primary snapshot, not an
        # impossible simultaneous equality with a continuously advancing log.
        if [[ "$primary_offset" =~ ^[0-9]+$ && "$replica_offset" =~ ^[0-9]+$ && "$replica_link" == up ]] &&
            ((replica_offset >= primary_offset)); then
            export TOKEN_RATE_LIMIT_SENTINEL_REPLICATION_OFFSET_CONFIRMED=1
            export TOKEN_RATE_LIMIT_SENTINEL_PRIMARY_REPLICATION_OFFSET="$primary_offset"
            export TOKEN_RATE_LIMIT_SENTINEL_REPLICA_REPLICATION_OFFSET="$replica_offset"
            return 0
        fi
        sleep 1
    done
    echo "replication offsets did not converge: primary=$primary_offset replica=$replica_offset link=$replica_link" >&2
    return 1
}

deadline=$((SECONDS + 30))
while ((SECONDS < deadline)); do
    if sentinel_cli "${prefix}-sentinel-1" -p "$sentinel_1_port" \
        SENTINEL get-master-addr-by-name mymaster >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
if ! ((SECONDS < deadline)); then
    echo "Sentinel topology did not become ready" >&2
    debug_logs
    exit 1
fi

master=$(sentinel_cli "${prefix}-sentinel-1" -p "$sentinel_1_port" \
    SENTINEL get-master-addr-by-name mymaster | tr '\n' ':')
if [[ "$master" != "127.0.0.1:${primary_port}:" ]]; then
    echo "unexpected primary: $master" >&2
    debug_logs
    exit 1
fi

if ! wait_for_replication_catchup; then
    debug_logs
    exit 1
fi

if [[ "$product" == valkey ]]; then
    server_version_field=valkey_version
    expected_server_version=8.0.11
else
    server_version_field=redis_version
    expected_server_version=8.10.2
fi
server_version=$(data_cli "${prefix}-primary" -p "$primary_port" INFO server |
    awk -F: -v field="$server_version_field" '$1 == field {gsub("\r", "", $2); print $2; exit}')
if [[ "$server_version" != "$expected_server_version" ]]; then
    echo "unexpected $product server version: field=$server_version_field version=$server_version expected=$expected_server_version" >&2
    debug_logs
    exit 1
fi
echo "Sentinel fixture product=$product image=$image server_version=$server_version replication_offset=$TOKEN_RATE_LIMIT_SENTINEL_PRIMARY_REPLICATION_OFFSET"

# Planned failover is triggered by the integration test after it has warmed an
# existing proxy connection. Failing over here would only exercise discovery
# before the proxy ever owns a stale connection.

if [[ "$replication_health_test" == 1 ]]; then
    data_cli "${prefix}-primary" -p "$primary_port" CONFIG SET min-replicas-to-write 1 >/dev/null
    data_cli "${prefix}-primary" -p "$primary_port" CONFIG SET min-replicas-max-lag 1 >/dev/null
    "$engine" stop "${prefix}-replica" >/dev/null
    deadline=$((SECONDS + 30))
    connected_replicas=""
    while ((SECONDS < deadline)); do
        connected_replicas=$(info_field "${prefix}-primary" "$primary_port" connected_slaves)
        [[ "$connected_replicas" == 0 ]] && break
        sleep 1
    done
    if [[ "$connected_replicas" != 0 ]]; then
        echo "replication health test did not reach zero connected replicas: $connected_replicas" >&2
        debug_logs
        exit 1
    fi
fi

export TOKEN_RATE_LIMIT_SENTINEL_PRODUCT="$product"
export TOKEN_RATE_LIMIT_SENTINEL_IMAGE="$image"
export TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS="redis://127.0.0.1:${sentinel_1_port},redis://127.0.0.1:${sentinel_2_port},redis://127.0.0.1:${sentinel_3_port}"
if [[ -n "${TOKEN_RATE_LIMIT_SENTINEL_DEAD_ENDPOINT:-}" ]]; then
    export TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS="${TOKEN_RATE_LIMIT_SENTINEL_DEAD_ENDPOINT},${TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS}"
fi
export TOKEN_RATE_LIMIT_SENTINEL_SERVICE=mymaster
export TOKEN_RATE_LIMIT_SENTINEL_DATA_PASSWORD="$data_password"
export TOKEN_RATE_LIMIT_SENTINEL_PASSWORD="$sentinel_password"
export TOKEN_RATE_LIMIT_SENTINEL_ON_FAILURE="$on_failure"
export TOKEN_RATE_LIMIT_SENTINEL_ENGINE="$engine"
export TOKEN_RATE_LIMIT_SENTINEL_CONTAINER_PREFIX="$prefix"
export TOKEN_RATE_LIMIT_SENTINEL_PRIMARY_CONTAINER="${prefix}-primary"
export TOKEN_RATE_LIMIT_SENTINEL_REPLICA_CONTAINER="${prefix}-replica"
export TOKEN_RATE_LIMIT_SENTINEL_SENTINEL_CONTAINER="${prefix}-sentinel-1"
export TOKEN_RATE_LIMIT_SENTINEL_SENTINEL_PORT="$sentinel_1_port"
export TOKEN_RATE_LIMIT_SENTINEL_PRIMARY_PORT="$primary_port"
export TOKEN_RATE_LIMIT_SENTINEL_REPLICA_PORT="$replica_port"

if [[ -n "$env_file" ]]; then
    if [[ -e "$env_file" ]]; then
        echo "--env-file already exists; refusing to overwrite it: $env_file" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$env_file")"
    (umask 077; : >"$env_file")
    chmod 600 "$env_file"
    env_file_written=1
    {
        for name in $(compgen -e | grep '^TOKEN_RATE_LIMIT_SENTINEL_'); do
            printf '%s=%q\n' "$name" "${!name}"
        done
    } >"$env_file"
fi

if (($#)); then
    "$@"
    rc=$?
    exit "$rc"
fi
echo "Sentinel topology ready: $product $image"
while :; do
    sleep 3600 &
    keepalive_pid=$!
    wait "$keepalive_pid"
    keepalive_pid=""
done
