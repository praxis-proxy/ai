#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Praxis Contributors

# Run the existing standalone contracts against the fixture's exact pinned
# product. Keep this live compatibility evidence separate from unit coverage.
set -Eeuo pipefail

: "${TOKEN_RATE_LIMIT_SENTINEL_PRIMARY_PORT:?run inside the pinned topology fixture}"
: "${TOKEN_RATE_LIMIT_SENTINEL_DATA_PASSWORD:?fixture data credentials are required}"
: "${TOKEN_RATE_LIMIT_SENTINEL_ENGINE:?fixture engine is required}"
: "${TOKEN_RATE_LIMIT_SENTINEL_CONTAINER_PREFIX:?fixture container ownership is required}"

# Keep fixed-endpoint contracts independent from failover. The HA contracts
# run separately against fresh topologies with all three Sentinels active.
for index in 1 2 3; do
    "$TOKEN_RATE_LIMIT_SENTINEL_ENGINE" stop \
        "${TOKEN_RATE_LIMIT_SENTINEL_CONTAINER_PREFIX}-sentinel-${index}" >/dev/null
done
unset TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS

# Encode credentials without putting them in an argument list or printing them.
TOKEN_RATE_LIMIT_VALKEY_URL=$(python3 - <<'PY'
import os
from urllib.parse import quote

password = quote(os.environ["TOKEN_RATE_LIMIT_SENTINEL_DATA_PASSWORD"], safe="")
port = os.environ["TOKEN_RATE_LIMIT_SENTINEL_PRIMARY_PORT"]
print(f"redis://:{password}@127.0.0.1:{port}")
PY
)
export TOKEN_RATE_LIMIT_VALKEY_URL
export RUST_TEST_THREADS=1

make test-token-rate-limit-valkey-unit
make test-token-rate-limit-valkey-integration
