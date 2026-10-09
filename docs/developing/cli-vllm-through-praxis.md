# Run Codex, Claude Code, or OpenCode through Praxis and vLLM

This guide runs a real coding client through a local Praxis gateway while vLLM
provides inference. The supported paths are:

```text
Codex       -> Praxis /v1/responses        -> vLLM /v1/responses (native)
Codex       -> Praxis /v1/responses        -> vLLM /v1/chat/completions (translated)
Claude Code -> Praxis /v1/messages         -> vLLM /v1/messages (native)
Claude Code -> Praxis /v1/messages         -> vLLM /v1/chat/completions (translated)
OpenCode    -> Praxis /v1/chat/completions -> vLLM /v1/chat/completions (native)
```

One gateway can serve all three clients over their native paths at once; see
[One gateway for all three clients](#one-gateway-for-all-three-clients).

Use the native paths when vLLM exposes the corresponding Responses or
Anthropic-compatible API. For Codex, Praxis keeps `/v1/responses` end to end
and lowers only client-owned tool types that vLLM does not understand. Use a
translated path for a backend that only exposes OpenAI Chat Completions.

## 1. Start vLLM

### On-demand GPU endpoint

In GitHub Actions, open **vLLM Dev Endpoint**, choose **Run workflow**, and keep
the defaults for Qwen3-8B. The workflow provisions a temporary GPU, publishes a
Cloudflare URL in the run summary, and tears everything down at the requested
deadline or when the run is cancelled. Its defaults already serve Qwen3-8B with
`--reasoning-parser qwen3` and thinking off, which is what the coding clients
below expect; change the `reasoning_parser` and `enable_thinking` inputs only
for a different model or when you specifically want visible reasoning.

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
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --gpu-memory-utilization 0.97 \
  --enforce-eager \
  --api-key "$VLLM_API_KEY"
```

The same server can be run from the official
[`vllm/vllm-openai`](https://hub.docker.com/r/vllm/vllm-openai) image. Its
entrypoint already starts the server, so the arguments are the `vllm serve`
flags used above, and vLLM reads the backend key from `VLLM_API_KEY` in the
container environment instead of the command line:

```console
export VLLM_API_KEY="$(openssl rand -hex 32)"
docker run --rm --name vllm \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e VLLM_API_KEY \
  docker.io/vllm/vllm-openai:latest \
  --model Qwen/Qwen3-8B \
  --served-model-name qwen3-8b \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --gpu-memory-utilization 0.97 \
  --enforce-eager
```

With Podman, replace `--gpus all` with `--device nvidia.com/gpu=all` and leave
the rest unchanged. Pin a released tag rather than `latest` for reproducible
behavior; the on-demand GPU endpoint workflow uses `v0.29.0-cu129`. The first
run downloads a multi-gigabyte image plus the model weights, and the Hugging
Face cache mount keeps the weights for later runs.

Use `VLLM_URL=http://127.0.0.1:8000` for either local server.

### Reasoning flags matter for Claude Code

The two reasoning-related flags above are deliberate for this setup and are the
main difference from the flags CI serves:

- `--reasoning-parser qwen3` matches the served model family. The parser name
  selects how vLLM splits a `<think>` block out of the completion into
  `reasoning_content`; a parser written for a different family can leave that
  text in the assistant message instead.
- `--default-chat-template-kwargs '{"enable_thinking":false}'` turns Qwen3's
  thinking off at the chat-template level, so the model does not emit a
  `<think>` block for the agentic turns Claude Code drives in the first place.

Serving Qwen3-8B with `--reasoning-parser deepseek_r1` and thinking left on
made Claude Code's plan mode loop endlessly, emitting reasoning text into the
rendered plan instead of a finished one. Switching to the two flags above
resolved it. They were verified together, so change them as a pair; if you need
visible reasoning for other work, expect agentic clients to degrade.

A nightly GPU regression guards this pairing. The
`vllm-gpu-claude-acceptance` job drives a read-only planning turn through the
native Anthropic path and fails if the run exhausts its turn budget instead of
answering, or if a `<think>` delimiter reaches the user-visible text. The
scenario is pinned in `[claude_code.launch.planning]` of
`tests/integration/fixtures/claude-code-cli/pin.toml`; reverting either flag
above is expected to turn it red. It is a headless approximation of the turn
that broke, not interactive plan mode, which the CLI does not expose to
`claude -p`.

The 32,768-token window is intentional for Claude Code auto mode. Its
client-initiated safety classifier reserves 2,112 output tokens independently
of the main Claude Code output-token setting and includes a large client-owned
prompt; 16K and 18K servers reject later classifier turns before inference. On
an A10G, eager execution reclaims CUDA-graph memory and the 0.97 utilization is
reserved for this single-user development workload. If you lower the window or
share the GPU, do not use auto mode unless the classifier request still fits.

The window is small enough that Claude Code's default output budget does not fit
beside a working prompt, so section 4 also sets `CLAUDE_CODE_MAX_OUTPUT_TOKENS`.

## 2. Point a Praxis example at vLLM

Choose one example for your client and copy it outside `examples/`:

```console
# Codex, native Responses API (preferred for vLLM)
cp examples/configs/openai/responses/client-tool-compat.yaml praxis-vllm.yaml

# Codex, translated to Chat Completions
cp examples/configs/openai/responses/codex-http-chat-translation.yaml praxis-vllm.yaml

# Claude Code, native Anthropic API (preferred for vLLM), or OpenCode
cp examples/configs/anthropic/messages-native-vllm.yaml praxis-vllm.yaml

# Claude Code, translated to Chat Completions
cp examples/configs/anthropic/messages-to-openai-vllm.yaml praxis-vllm.yaml

# All three clients at once, native paths only
cp examples/configs/coding-harness-gateway-vllm.yaml praxis-vllm.yaml
```

OpenCode was tested with the
[native Claude configuration](../../examples/configs/anthropic/messages-native-vllm.yaml).
Its catch-all route forwards `/v1/chat/completions` to vLLM without body
translation, and Anthropic validation is scoped to `/v1/messages`.
The same gateway can serve both Claude Code and OpenCode, provided vLLM
supports their respective endpoints.

### One gateway for all three clients

[`coding-harness-gateway-vllm.yaml`](../../examples/configs/coding-harness-gateway-vllm.yaml)
composes the two native examples above into one configuration, one proxy
process, and one vLLM backend, sharing a single `GATEWAY_AUTH_PASSWORD` across
all three CLIs. Every client keeps its own wire format end to end; nothing is
translated. Use it when you want one gateway running instead of switching
configs per client. For a single client, the focused examples above are less to
read.

It binds **two** listeners:

| Listener | Serves | Client base URL |
| --- | --- | --- |
| `127.0.0.1:8080` | Claude Code (`/v1/messages`), OpenCode (`/v1/chat/completions`) | `http://127.0.0.1:8080` |
| `127.0.0.1:8081` | Codex (`/v1/responses`) | `http://127.0.0.1:8081/v1` |

The split is not cosmetic. Codex's rich client tools need
`openai_client_tool_compat`, which needs `openai_stream_events`, which fails
closed outside an `iterative_request_router` step. The IRR is a terminal filter
that cannot share a chain with a top-level `load_balancer`, so everything in
that chain would have to route through it — and the IRR buffers a sub-request
response unless a filter selects streaming, which the native `/v1/messages` and
`/v1/chat/completions` paths do not. Routing them through it would hold each
turn's SSE until the turn finished. Two listeners keep all three clients
streaming incrementally. Because every CLI configures its own base URL, the
second port costs one line of client config.

Everything else in this guide applies unchanged, except that Codex authenticates
with Basic rather than Bearer — see [section 3](#3-connect-codex). The store
needs the same absolute `database_url` as the native Codex example when running
from the container image; see
[the writable database path note](#the-store-backed-examples-need-a-writable-database-path).

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

### Run Praxis from source

Build and start Praxis. Running it in the background keeps the two generated
values in the shell that will launch the client. The SQLite feature enables the
local response store used by the native Codex and unified examples, and is
harmless for the other paths:

```console
cargo build -p praxis-ai-proxy --no-default-features \
  --features standard,openai-all,store-sqlite
./target/debug/praxis-ai -c praxis-vllm.yaml > /tmp/praxis-vllm.log 2>&1 &
export PRAXIS_PID=$!
```

Praxis listens at `http://127.0.0.1:8080`.

### Run Praxis from the official container image

Released images are published to
[`ghcr.io/praxis-proxy/ai`](https://ghcr.io/praxis-proxy/ai). The `latest` tag
tracks the most recent release; substitute a version tag such as `0.4` to pin
one:

```console
docker pull ghcr.io/praxis-proxy/ai:latest
```

The image entrypoint is `praxis-ai`, its working directory is `/etc/praxis`,
and it runs as the unprivileged `praxis` user (UID 100). Mounting the config at
`/etc/praxis/praxis.yaml` is enough — the binary falls back to `praxis.yaml` in
the working directory — but passing `-c` is clearer and works for any mount
path. `GATEWAY_AUTH_PASSWORD` and `VLLM_API_KEY` are read from the *proxy
process* environment at pipeline build time, so they must be forwarded into the
container, not merely exported on the host. The `z` mount suffix relabels the
config for SELinux hosts such as Fedora and RHEL; drop it elsewhere.

Host networking is the closest match to the source build and needs no config
edits at all, because `127.0.0.1` inside the container is the host loopback for
both the listener and the vLLM backend:

```console
docker run -d --rm --name praxis-vllm \
  --network host \
  -e VLLM_API_KEY -e GATEWAY_AUTH_PASSWORD \
  -v "$PWD/praxis-vllm.yaml:/etc/praxis/praxis.yaml:ro,z" \
  ghcr.io/praxis-proxy/ai:latest -c /etc/praxis/praxis.yaml
```

Praxis listens at `http://127.0.0.1:8080`, and sections 3 through 5 apply
unchanged. Docker's `--network host` is Linux-only.

To publish a port instead of sharing the host network stack, make three edits
to the copied config:

1. Change the listener to `address: "0.0.0.0:8080"`. A listener bound to
   `127.0.0.1` inside the container's own namespace is unreachable through a
   published port.
2. Change the local vLLM endpoint from `127.0.0.1:8000` to
   `host.docker.internal:8000`, and pass
   `--add-host=host.docker.internal:host-gateway` on the command line.
3. Add `allow_private_upstreams: true` beside the existing
   `allow_private_endpoints: true`. The existing option covers literal private
   endpoints written in the config; the new one is needed because that
   hostname *resolves* to a private address, and without it the request fails
   with `upstream hostname resolved to private/reserved IP address`.

```console
docker run -d --rm --name praxis-vllm \
  -p 8080:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e VLLM_API_KEY -e GATEWAY_AUTH_PASSWORD \
  -v "$PWD/praxis-vllm.yaml:/etc/praxis/praxis.yaml:ro,z" \
  ghcr.io/praxis-proxy/ai:latest -c /etc/praxis/praxis.yaml
```

With Podman, both commands work unchanged. Podman also resolves
`host.containers.internal` on its own, so the published-port config can use
that name and drop `--add-host`.

The on-demand GPU endpoint needs no backend-specific container handling: the
tunnel hostname is public, so the `endpoints` and `tls.sni` edits above are the
only vLLM-side change with either networking mode.

Follow startup with `docker logs -f praxis-vllm`. For an explicit readiness
probe, add an admin listener to the copied config:

```yaml
admin:
  address: "127.0.0.1:9901"
```

The admin endpoint must bind loopback unless
`insecure_options.allow_public_admin: true` is set, so query it from inside the
container with
`docker exec praxis-vllm wget -qO- http://127.0.0.1:9901/healthy` (or directly
from the host under `--network host`). When finished, replace the `kill` in
the cleanup section with `docker rm -f praxis-vllm`.

#### The store-backed examples need a writable database path

The image is built with `full,store-sqlite`, so `client-tool-compat.yaml` — the
preferred native Codex example — and `coding-harness-gateway-vllm.yaml` both run
on the stock image. The other three configurations in this section have no store
filter and need no change at all.

The one edit the store examples do need is an absolute `database_url`. The
image's `/etc/praxis` working directory is root-owned while the process runs as
`praxis` (UID 100), so the example's relative `sqlite://responses.db?mode=rwc`
cannot be created: the proxy starts, and the first `/v1/responses` request
fails with HTTP 500 and `unable to open database file`. Point `database_url` at
`/var/lib/praxis`, the writable state directory owned by the container user:

```yaml
- filter: openai_response_store
  backend: sqlite
  database_url: "sqlite:///var/lib/praxis/responses.db?mode=rwc"
```

```console
docker run -d --rm --name praxis-vllm \
  --network host \
  -e VLLM_API_KEY -e GATEWAY_AUTH_PASSWORD \
  -v "$PWD/praxis-vllm.yaml:/etc/praxis/praxis.yaml:ro,z" \
  ghcr.io/praxis-proxy/ai:latest -c /etc/praxis/praxis.yaml
```

The database then lives in the container's writable layer and disappears with
`--rm`, which is usually what a CLI test loop wants. To keep responses across
restarts, mount a volume at `/var/lib/praxis`: `-v praxis-state:/var/lib/praxis`
with Podman adds `:U` to chown it to the container user, and a host directory
needs `chown 100:100` (or `-m 0777`) before the first run.

SQLite in the image landed after `v0.5.0`, so a pinned older tag still rejects
the config at startup with
`backend 'sqlite' is unavailable; rebuild with the 'store-sqlite' feature`. Use
`latest` or a tag newer than `v0.5.0`.

The image can also serve this example with `backend: postgres`,
`allow_private_database_url: true`, and a reachable PostgreSQL instance — see
`examples/configs/openai/responses/response-store-postgres-mtls.yaml` — but for
a single-developer CLI loop SQLite is less setup.

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

### Codex on the unified gateway

The unified example gates its Responses listener with the same `basic_auth`
filter the Claude examples use, so Codex presents the gateway's Basic
credential instead of a Bearer token. Codex builds its provider headers from
`http_headers` and `env_http_headers` independently of `env_key`, and applies
the `env_key` bearer by *appending* `Authorization` rather than replacing it —
setting both sends two `Authorization` headers. So omit `env_key` entirely and
carry the credential in `env_http_headers`:

```console
export CODEX_HOME="$(mktemp -d)"
export PRAXIS_GATEWAY_AUTH="Basic $(printf 'gateway:%s' "$GATEWAY_AUTH_PASSWORD" | base64)"
cat > "$CODEX_HOME/config.toml" <<EOF
model = "$VLLM_MODEL"
model_provider = "praxis"
web_search = "disabled"

[model_providers.praxis]
name = "Local Praxis"
base_url = "http://127.0.0.1:8081/v1"
wire_api = "responses"
env_http_headers = { Authorization = "PRAXIS_GATEWAY_AUTH" }
EOF

codex exec --skip-git-repo-check \
  "Inspect this directory, create praxis-vllm-check.txt, then summarize the change."
```

With no `env_key`, no `experimental_bearer_token`, and `requires_openai_auth`
left at its default, Codex resolves an unauthenticated provider and contributes
no auth headers of its own. The isolated `CODEX_HOME` is required rather than
merely tidy here: a logged-in Codex home would supply ambient credentials that
this provider would otherwise inherit.

Unlike the single-client native path, the client credential is no longer the
backend credential. `basic_auth` verifies it and strips it, and
`credential_injection` supplies `VLLM_API_KEY` toward vLLM — the same
three-credential separation the Claude examples use.

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
# Keep the requested output budget inside the 32,768-token vLLM window.
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192
# Required if you select Claude Code's auto permission mode with this backend.
export CLAUDE_CODE_AUTO_MODE_SERVER=0

claude --model "$VLLM_MODEL"
# To exercise client-classified auto mode explicitly:
# claude --permission-mode auto --model "$VLLM_MODEL"
```

`GATEWAY_AUTH_PASSWORD` and `VLLM_API_KEY` must be present in the environment of
the Praxis process. The other variables configure Claude Code.

The translated `messages-to-openai-vllm.yaml` route enables
`allow_lossy_features: [prompt_caching, extended_thinking]`, so you do **not**
need to disable Anthropic thinking or prompt caching in Claude Code. Chat
Completions cannot represent either feature, so Praxis strips the wire markers an
unmodified client sends and reports the loss (a `WARN` log, the
`praxis_anthropic_messages_to_chat_completions_degraded_total` counter, and an
`x-degraded-features` response header) instead of rejecting the request.

To see the strict behavior instead — a 400 when either feature appears — use the
`messages-to-openai.yaml` route (empty allowlist) and disable both features in
Claude Code:

```console
export CLAUDE_CODE_DISABLE_THINKING=1
export DISABLE_PROMPT_CACHING=1
```

`CLAUDE_CODE_MAX_OUTPUT_TOKENS` is required for a 32,768-token server. Claude
Code does not know the served model's real window, so it sends the default
`max_tokens` for the Anthropic model name it believes it is calling — around
21,000 tokens. vLLM reserves `max_tokens` against `--max-model-len` before
inference, so that budget plus a modest prompt exceeds the window and the
request fails with HTTP 400 even though the prompt itself is small. Capping the
budget at 8192 leaves roughly 24K for input. Claude Code still sizes its own
auto-compaction against the window it assumes, not the real one, so run
`/compact` by hand if a long session creeps back into the limit. A larger window
is the alternative, but Qwen3-8B is natively 32,768 tokens and anything beyond
it needs `--rope-scaling` on `vllm serve`.

Praxis and vLLM do not implement Anthropic's server-side auto-mode classifier
protocol. `CLAUDE_CODE_AUTO_MODE_SERVER=0` makes Claude Code initiate the
classifier model requests through Praxis instead. This setting only affects
Claude Code's `auto` permission mode; the classifier still consumes model
inference and is not an on-device check. Do not set it to `1` for this setup.
See [Anthropic's auto-mode classifier documentation](https://code.claude.com/docs/en/auto-mode-classifier-billing).

## 5. Connect OpenCode

This example targets OpenCode 1.x. Check your version with `opencode --version`.
OpenCode 2.x requires a different plugin implementation; see the
[OpenCode migration guide](https://opencode.ai/v2/docs/migrate-v1/).

Use the native Claude configuration from section 2. It forwards OpenCode's
`/v1/chat/completions` requests to vLLM and authenticates clients with Basic
auth using username `gateway` and password from `GATEWAY_AUTH_PASSWORD`.
Add this configuration to
`~/.config/opencode/opencode.jsonc`, merging the `plugin` and `provider`
entries with any existing settings:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["./praxis-auth.ts"],
  "provider": {
    "praxis": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Praxis",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": {
        "qwen3-8b": {
          "name": "Qwen3 8B",
          "limit": { "context": 32768, "output": 8192 }
        }
      }
    }
  }
}
```

Replace `qwen3-8b` with the exact served name in `VLLM_MODEL` and adjust the
limits to match your vLLM server. For a remote Praxis gateway, replace the
`baseURL` with its HTTPS address, keeping the `/v1` suffix. Use HTTP only for
a loopback gateway. Export the same URL from your shell so the plugin can
verify the destination independently of project configuration:

```console
export PRAXIS_BASE_URL="https://gateway.example.com/v1"
```

For the local `http://127.0.0.1:8080/v1` example, no URL export is needed.

Create `~/.config/opencode/praxis-auth.ts`:

```typescript
export default async () => ({
  config: async (config) => {
    const options = config.provider.praxis.options
    const trustedURL = process.env.PRAXIS_BASE_URL ?? "http://127.0.0.1:8080/v1"
    if (options.baseURL !== trustedURL) {
      throw new Error("Praxis baseURL does not match the trusted gateway")
    }
    const password = process.env.GATEWAY_AUTH_PASSWORD
    if (!password) throw new Error("GATEWAY_AUTH_PASSWORD is required for Praxis")
    const token = Buffer.from(`gateway:${password}`).toString("base64")
    options.headers = { ...options.headers, Authorization: `Basic ${token}` }
  },
})
```

The plugin checks the merged provider URL before attaching Basic auth with
username `gateway`. It rejects project overrides that change the destination.
Set `PRAXIS_BASE_URL` only to a gateway you trust; it must exactly match
`baseURL`, including any trailing slash.

Launch OpenCode from the shell where `GATEWAY_AUTH_PASSWORD` was exported in
section 2. It must match the password in the Praxis process environment;
OpenCode does not need `VLLM_API_KEY`:

```console
opencode --model "praxis/$VLLM_MODEL"
```

Quit and restart OpenCode after changing its configuration or plugin. You can
also select the configured model with `/models`.

## Cleanup

When finished, stop local Praxis and cancel the on-demand endpoint workflow:

```console
kill "$PRAXIS_PID"
```

## Troubleshooting

- `401` from Praxis on the Claude or OpenCode path: verify the Basic
  authorization header and that the client's `GATEWAY_AUTH_PASSWORD` matches
  the gateway's password.
- `401` from vLLM: `VLLM_API_KEY` does not match the key passed to vLLM.
- `401` from Praxis on the Codex path of the unified gateway: `env_key` is
  probably still set alongside `env_http_headers`. Codex appends the `env_key`
  bearer rather than replacing the configured header, so the request carries
  two `Authorization` headers and `basic_auth` does not see the Basic one it
  expects. Remove `env_key` from the provider block. A stale logged-in
  `CODEX_HOME` can do the same thing; use the `mktemp -d` home from section 3.
- Model not found: use the exact slash-free served name, normally `qwen3-8b`.
- `400` with `request body is not JSON` on a request that has no body, such as
  `GET /v1/models`, logged as
  `request body rejected by filter filter="openai_responses_format"`: a
  Responses classifier is running unconditioned on every request the listener
  accepts, and an empty body classifies as non-JSON, which `on_invalid: reject`
  turns into a 400 before routing. Both Codex examples in section 2 lead with
  the head-driven `openai_responses_request` instead, which releases any
  operation it does not recognize without reading a body. A config derived from
  an older copy needs the same swap: drop the leading `openai_responses_format`
  and let `openai_responses_request` carry the `on_invalid` and `headers` that
  classifier used to carry.
  Do not reach for `on_invalid: continue` instead: it clears the probe but also
  forwards genuinely malformed Responses bodies to vLLM rather than rejecting
  them at the gateway.
- RFC 9457 `application/problem+json` where an OpenAI client expects
  `{"error": {...}}`: the chain is missing `ai_operation`. The body
  classifiers each resolve one protocol — `openai_responses_request` covers
  Responses, and `anthropic_messages_request` installs its formatter only on
  the Anthropic Messages surface — so neither owns the protocol decision for
  the Chat Completions traffic a coding client also sends, and the
  request reaches core's error path with no formatter installed. `ai_operation`
  classifies from the request head and installs the matching formatter for
  every OpenAI and Anthropic protocol. Every example in section 2 leads with
  it, directly after `basic_auth`.
  This bites OpenCode hardest: it is the one client whose traffic
  (`/v1/chat/completions`) shares a listener with another protocol, so a config
  derived from an older copy sends it problem details its SDK cannot parse.
- `400` with `maximum context length is 32768 tokens` and a requested output
  count near 21,000: Claude Code's default output budget does not fit the
  window. Set `CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192` as shown in section 4.
- `400` with `maximum context length is 32768 tokens` on the first turn, with a
  reported input count near 28,000 and a modest output count: the output cap
  from the previous entry is already in effect and the startup prompt itself is
  the problem. Claude Code loads MCP tool definitions, plugin skills, and
  `CLAUDE.md` before the session begins, so an empty conversation can consume
  most of the window. Run `/context` for the breakdown; it is a local command
  and still works while every request is failing. Start Claude Code with
  `--strict-mcp-config` to drop globally configured MCP servers, disable
  unneeded plugins with `/plugin`, and prefer a working directory whose
  `CLAUDE.md` is small or absent.
- Claude Code never finishes a turn — plan mode keeps looping, or reasoning
  text appears in the answer or the rendered plan: the server is emitting
  thinking that the client is not meant to see. Serve Qwen3 with
  `--reasoning-parser qwen3` and
  `--default-chat-template-kwargs '{"enable_thinking":false}'` as shown in
  section 1, and restart vLLM; the reasoning parser is a server flag, so
  nothing on the Praxis or client side changes it. On the on-demand endpoint,
  check the `reasoning_parser` and `enable_thinking` inputs of the run.
- TLS or connection failure: use only the tunnel hostname in the endpoint and
  `tls.sni`; do not include `https://` in Praxis's `endpoints` entry.
- Connection refused on 8080 from another machine, while vLLM on 8000 answers:
  every example in section 2 binds the listener to `127.0.0.1:8080`, so Praxis
  is reachable only from its own host. This is independent of how Praxis runs —
  `--network host` needs the same `address: "0.0.0.0:8080"` edit as the
  published-port path, not just the port mapping. After that edit, open 8080 on
  the host firewall; a loopback-only listener refuses the connection
  immediately, whereas a blocked port usually hangs or reports no route. Binding
  to `0.0.0.0` exposes the gateway to the network, so keep the authentication
  filter in place. The unified example has a second listener on `127.0.0.1:8081`
  that needs the same treatment, plus `-p 8081:8081` under the published-port
  container path.
- Claude startup probes may call `/v1/messages/count_tokens`. Native vLLM
  supports it; the translated Chat path can return 404 and Claude degrades
  gracefully.
