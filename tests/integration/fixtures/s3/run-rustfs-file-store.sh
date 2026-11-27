#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Praxis Contributors

# Disposable single-bucket RustFS lane for the public S3FileStore API.
set -euo pipefail

: "${CONTAINER_ENGINE:?set CONTAINER_ENGINE to docker or podman}"
# Official multi-arch RustFS 1.0.5 image; keep the S3 service immutable.
RUSTFS_IMAGE=${RUSTFS_IMAGE:-ghcr.io/rustfs/rustfs:1.0.5@sha256:b81d16a90ee25bd9511280127ca1471788c6e874cf0f0da759645f46b2dc1d74}
[[ "$RUSTFS_IMAGE" =~ ^ghcr\.io/rustfs/rustfs:[^@]+@sha256:[0-9a-f]{64}$ ]] || {
  echo "RUSTFS_IMAGE must be an official immutable @sha256 reference" >&2
  exit 2
}
command -v "$CONTAINER_ENGINE" >/dev/null || {
  echo "Container engine unavailable: $CONTAINER_ENGINE" >&2
  exit 2
}
command -v aws >/dev/null || {
  echo "AWS CLI is required for independent S3 assertions" >&2
  exit 2
}

export RUSTFS_S3_ENDPOINT=http://127.0.0.1:9000
export RUSTFS_S3_REGION=us-east-1
export RUSTFS_S3_BUCKET=praxis-files-s3-e2e
export AWS_ACCESS_KEY_ID=praxisrustfsci
export AWS_SECRET_ACCESS_KEY=praxisrustfsci-secret-key-2026
export AWS_DEFAULT_REGION="$RUSTFS_S3_REGION"
export AWS_EC2_METADATA_DISABLED=true
export AWS_MAX_ATTEMPTS=1
export AWS_PAGER=
unset AWS_SESSION_TOKEN AWS_PROFILE AWS_DEFAULT_PROFILE

config_dir=$(mktemp -d "${TMPDIR:-/tmp}/praxis-rustfs.XXXXXX")
container_id=
cleanup() {
  status=$?
  if [[ -n "$container_id" ]]; then
    if (( status != 0 )); then
      "$CONTAINER_ENGINE" logs "$container_id" >&2 || true
    fi
    "$CONTAINER_ENGINE" rm -f "$container_id" >/dev/null || true
  fi
  rm -rf "$config_dir"
}
trap cleanup EXIT

# Keep the independent AWS CLI on path-style addressing and away from the
# developer's configured AWS profiles or real cloud credentials.
export AWS_CONFIG_FILE="$config_dir/config"
export AWS_SHARED_CREDENTIALS_FILE="$config_dir/credentials"
cat > "$AWS_CONFIG_FILE" <<'EOF'
[default]
s3 =
    addressing_style = path
EOF
: > "$AWS_SHARED_CREDENTIALS_FILE"

"$CONTAINER_ENGINE" pull "$RUSTFS_IMAGE"
container_id=$("$CONTAINER_ENGINE" run -d --rm \
  --publish 127.0.0.1:9000:9000 \
  --env RUSTFS_ACCESS_KEY="$AWS_ACCESS_KEY_ID" \
  --env RUSTFS_SECRET_KEY="$AWS_SECRET_ACCESS_KEY" \
  --env RUSTFS_REGION="$RUSTFS_S3_REGION" \
  --env RUSTFS_ADDRESS=:9000 \
  --env RUSTFS_CONSOLE_ENABLE=false \
  "$RUSTFS_IMAGE" /data)

ready=false
for _attempt in {1..60}; do
  if curl --fail --silent --show-error --max-time 2 "$RUSTFS_S3_ENDPOINT/health/ready" >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 1
done
if [[ "$ready" != true ]]; then
  echo "RustFS did not become ready at $RUSTFS_S3_ENDPOINT/health/ready" >&2
  exit 1
fi

aws --endpoint-url "$RUSTFS_S3_ENDPOINT" --region "$RUSTFS_S3_REGION" \
  s3api create-bucket --bucket "$RUSTFS_S3_BUCKET" --no-cli-pager >/dev/null

test_args=(--include-ignored --test-threads=1)
if [[ -n "${V:-}" ]]; then
  test_args+=(--nocapture)
fi
cargo test -p praxis-ai-apis --no-default-features \
  --features openai-files-s3,object_store/aws --test s3_file_store -- "${test_args[@]}"
