# Run Codex or Claude Code through Praxis and vLLM

This guide runs a real coding client through a local Praxis gateway while vLLM
provides inference. The supported paths are:

```text
Codex       -> Praxis /v1/responses -> vLLM /v1/responses (native)
Codex       -> Praxis /v1/responses -> vLLM /v1/chat/completions (translated)
Claude Code -> Praxis /v1/messages  -> vLLM /v1/messages (native)
Claude Code -> Praxis /v1/messages  -> vLLM /v1/chat/completions (translated)
```

Use the native paths when vLLM exposes the corresponding Responses or
Anthropic-compatible API. For Codex, Praxis keeps `/v1/responses` end to end
and lowers only client-owned tool types that vLLM does not understand. Use a
translated path for a backend that only exposes OpenAI Chat Completions.

## 1. Start vLLM

### On-demand GPU endpoint

In GitHub Actions, open **vLLM Dev Endpoint**, choose **Run workflow**, and keep
the defaults for Qwen3-8B. The workflow provisions a temporary GPU, publishes a
Cloudflare URL in the run summary, and tears everything down at the requested
deadline or when the run is cancelled.

Record the two summary values:

```console
export VLLM_URL=https://example.trycloudflare.com
export VLLM_MODEL=qwen3-8b
```

The Quick Tunnel is an unauthenticated, best-effort development endpoint. Treat
its random URL as temporary sensitive data, do not use it for production, and
cancel the workflow as soon as testing is complete.

### Existing or local vLLM

If vLLM is already reachable, set the same variables to its URL and exact served
model name. A keyed local server can be started with:

```console
export VLLM_API_KEY="$(openssl rand -hex 32)"
vllm serve Qwen/Qwen3-8B \
  --served-model-name qwen3-8b \
  --max-model-len 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser deepseek_r1 \
  --api-key "$VLLM_API_KEY"
```

Use `VLLM_URL=http://127.0.0.1:8000` for that server.

## 2. Point a Praxis example at vLLM

Choose one example and copy it outside `examples/`:

```console
# Codex, native Responses API (preferred for vLLM)
cp examples/configs/openai/responses/client-tool-compat.yaml praxis-vllm.yaml

# Codex, translated to Chat Completions
cp examples/configs/openai/responses/codex-http-chat-translation.yaml praxis-vllm.yaml

# Claude Code, native Anthropic API (preferred for vLLM)
cp examples/configs/anthropic/messages-native-vllm.yaml praxis-vllm.yaml

# Claude Code, translated to Chat Completions
cp examples/configs/anthropic/messages-to-openai-vllm.yaml praxis-vllm.yaml
```

The native Codex example uses `127.0.0.1:3001` as its fixture backend; change
that endpoint to `127.0.0.1:8000` for the local vLLM server. The other examples
already target port 8000. For the on-demand HTTPS endpoint, remove `https://`
from `VLLM_URL` and replace the selected example's vLLM backend endpoint with:

```yaml
endpoints:
  - "example.trycloudflare.com:443"
tls:
  sni: "example.trycloudflare.com"
```

For the translated Codex example, replace the fixed provider credential with
the same environment-backed form used by the Claude examples:

```yaml
- filter: credential_injection
  clusters:
    - name: codex-chat-provider
      header: Authorization
      env_var: VLLM_API_KEY
      header_prefix: "Bearer "
      strip_client_credential: true
```

Set a backend key. It must match `--api-key` for a keyed vLLM server; any random
value is sufficient for the unauthenticated dev endpoint:

```console
export VLLM_API_KEY="${VLLM_API_KEY:-$(openssl rand -hex 32)}"
export GATEWAY_AUTH_PASSWORD="$(openssl rand -hex 24)"
```

Build and start Praxis. Running it in the background keeps the two generated
values in the shell that will launch the client. The SQLite feature enables the
local response store used by the native Codex example and is harmless for the
other paths:

```console
cargo build -p praxis-ai-proxy --features store-sqlite
./target/debug/praxis-ai -c praxis-vllm.yaml > /tmp/praxis-vllm.log 2>&1 &
export PRAXIS_PID=$!
```

Praxis listens at `http://127.0.0.1:8080`.

## 3. Connect Codex

Use an isolated Codex home so this test does not replace normal settings:

```console
export CODEX_HOME="$(mktemp -d)"
# Native /v1/responses passes this credential through to keyed vLLM.
export PRAXIS_API_KEY="$VLLM_API_KEY"
cat > "$CODEX_HOME/config.toml" <<EOF
model = "$VLLM_MODEL"
model_provider = "praxis"
web_search = "disabled"

[model_providers.praxis]
name = "Local Praxis"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
env_key = "PRAXIS_API_KEY"
EOF

codex exec --skip-git-repo-check \
  "Inspect this directory, create praxis-vllm-check.txt, then summarize the change."
```

With the preferred native example, Codex sends Responses API traffic to Praxis,
which forwards it to vLLM's `/v1/responses` endpoint. Praxis lowers and restores
Codex-specific client tool types but does not translate the inference request to
Chat Completions.

If you selected the translated Codex example instead, set
`PRAXIS_API_KEY=local-codex-client-key`. Praxis then translates Responses to
Chat Completions, injects `VLLM_API_KEY`, and streams the translated response
back.

## 4. Connect Claude Code

The Claude examples authenticate the client with Basic auth and independently
replace its Anthropic key with `VLLM_API_KEY` toward vLLM:

```console
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_API_KEY=local-claude-client-key
export ANTHROPIC_CUSTOM_HEADERS="Authorization: Basic $(printf 'gateway:%s' "$GATEWAY_AUTH_PASSWORD" | base64)"
export ANTHROPIC_MODEL="$VLLM_MODEL"
export ANTHROPIC_DEFAULT_MODEL="$VLLM_MODEL"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$VLLM_MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$VLLM_MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$VLLM_MODEL"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

claude --model "$VLLM_MODEL"
```

`GATEWAY_AUTH_PASSWORD` and `VLLM_API_KEY` must be present in the environment of
the Praxis process. The other variables configure Claude Code. When finished,
stop local Praxis and cancel the on-demand endpoint workflow:

```console
kill "$PRAXIS_PID"
```

## Troubleshooting

- `401` from Praxis on the Claude path: verify the Basic authorization header.
- `401` from vLLM: `VLLM_API_KEY` does not match the key passed to vLLM.
- Model not found: use the exact slash-free served name, normally `qwen3-8b`.
- TLS or connection failure: use only the tunnel hostname in the endpoint and
  `tls.sni`; do not include `https://` in Praxis's `endpoints` entry.
- Claude startup probes may call `/v1/messages/count_tokens`. Native vLLM
  supports it; the translated Chat path can return 404 and Claude degrades
  gracefully.
