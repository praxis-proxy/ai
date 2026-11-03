#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Praxis Contributors
#
# Configure Codex, Claude Code, OpenCode, and pi inside an OpenShell sandbox to
# reach a host-side Praxis gateway, then exec the given command:
#
#   openshell sandbox create --provider praxis --env VLLM_MODEL=qwen3-8b \
#     --from praxis-harness:local -- praxis-harness-env claude
#
# GATEWAY_AUTH_PASSWORD comes from the attached `praxis` provider and is an
# OpenShell placeholder, not the password. Every header below is built from
# that placeholder; the OpenShell proxy substitutes the real password inside
# the Basic credential on its way to the gateway, and nowhere else. If the
# variable does not look like a placeholder, the real password was passed with
# --env and is already inside the sandbox, so this script exits.
#
# Optional overrides: PRAXIS_GATEWAY_HOST (default host.openshell.internal),
# PRAXIS_MESSAGES_PORT (8080), PRAXIS_RESPONSES_PORT (8081),
# PRAXIS_MODEL_CONTEXT (32768), PRAXIS_MODEL_MAX_OUTPUT (8192).

set -eu

: "${GATEWAY_AUTH_PASSWORD:?attach the Praxis gateway provider with --provider}"
: "${VLLM_MODEL:?set --env VLLM_MODEL=<served model name>}"

# Fail closed: a raw password came from --env, so the real secret is already
# inside the sandbox. Refuse to configure a harness with it.
case "$GATEWAY_AUTH_PASSWORD" in
  *openshell:resolve:env:* | *OPENSHELL-RESOLVE-ENV-*) ;;
  *)
    echo "praxis-harness-env: GATEWAY_AUTH_PASSWORD is not an OpenShell placeholder;" \
      "pass it through a provider, not --env" >&2
    exit 1
    ;;
esac

host="${PRAXIS_GATEWAY_HOST:-host.openshell.internal}"
messages_url="http://${host}:${PRAXIS_MESSAGES_PORT:-8080}"
responses_url="http://${host}:${PRAXIS_RESPONSES_PORT:-8081}"
context="${PRAXIS_MODEL_CONTEXT:-32768}"
max_output="${PRAXIS_MODEL_MAX_OUTPUT:-8192}"

token="$(printf 'gateway:%s' "$GATEWAY_AUTH_PASSWORD" | base64 | tr -d '\n')"
export PRAXIS_GATEWAY_BASIC_TOKEN="$token"
export PRAXIS_GATEWAY_AUTH="Basic $token"

mkdir -p "$HOME"

# Claude Code: native /v1/messages. The API key is a placeholder Praxis strips.
export ANTHROPIC_BASE_URL="$messages_url"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-local-claude-client-key}"
export ANTHROPIC_CUSTOM_HEADERS="Authorization: Basic $token"
export ANTHROPIC_MODEL="$VLLM_MODEL"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$VLLM_MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$VLLM_MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$VLLM_MODEL"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="$max_output"
export CLAUDE_CODE_MAX_CONTEXT_TOKENS="$context"
export CLAUDE_CODE_AUTO_MODE_SERVER=0
mkdir -p "${CLAUDE_CONFIG_DIR:-$HOME/.claude}"

# Codex: native /v1/responses on its own listener. No env_key, so the Basic
# credential is the only Authorization header Codex sends. Codex's own
# bubblewrap sandbox cannot create user namespaces inside an OpenShell
# sandbox, so every tool command would fail; OpenShell already confines it.
codex_home="${CODEX_HOME:-$HOME/.codex}"
mkdir -p "$codex_home"
cat > "$codex_home/config.toml" <<EOF
model = "$VLLM_MODEL"
model_provider = "praxis"
web_search = "disabled"
sandbox_mode = "danger-full-access"

[model_providers.praxis]
name = "Praxis"
base_url = "$responses_url/v1"
wire_api = "responses"
env_http_headers = { Authorization = "PRAXIS_GATEWAY_AUTH" }
EOF

# OpenCode 1.x: native /v1/chat/completions.
opencode_dir="${XDG_CONFIG_HOME:-$HOME/.config}/opencode"
mkdir -p "$opencode_dir"
cat > "$opencode_dir/opencode.json" <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "small_model": "praxis/$VLLM_MODEL",
  "provider": {
    "praxis": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Praxis",
      "options": {
        "baseURL": "$messages_url/v1",
        "headers": { "Authorization": "{env:PRAXIS_GATEWAY_AUTH}" }
      },
      "models": {
        "$VLLM_MODEL": { "limit": { "context": $context, "output": $max_output } }
      }
    }
  }
}
EOF

# pi: native /v1/chat/completions. The apiKey is a required placeholder that
# the Authorization header replaces.
pi_dir="${PI_CODING_AGENT_DIR:-$HOME/.pi/agent}"
mkdir -p "$pi_dir"
cat > "$pi_dir/models.json" <<EOF
{
  "providers": {
    "praxis": {
      "name": "Praxis",
      "baseUrl": "$messages_url/v1",
      "api": "openai-completions",
      "apiKey": "local-pi-client-key",
      "headers": { "Authorization": "Basic \${PRAXIS_GATEWAY_BASIC_TOKEN}" },
      "models": [
        { "id": "$VLLM_MODEL", "contextWindow": $context, "maxTokens": $max_output }
      ]
    }
  }
}
EOF

exec "$@"
