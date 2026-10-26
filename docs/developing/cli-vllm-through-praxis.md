# Run your Code Harness through Praxis and vLLM

This guide runs a real coding client through a local Praxis gateway while vLLM
provides inference. The supported paths are:

```text
Codex       -> Praxis /v1/responses        -> vLLM /v1/responses (native)
Codex       -> Praxis /v1/responses        -> vLLM /v1/chat/completions (translated)
Claude Code -> Praxis /v1/messages         -> vLLM /v1/messages (native)
Claude Code -> Praxis /v1/messages         -> vLLM /v1/chat/completions (translated)
OpenCode    -> Praxis /v1/chat/completions -> vLLM /v1/chat/completions (native)
pi          -> Praxis /v1/chat/completions -> vLLM /v1/chat/completions (native)
```

One gateway can serve all four clients over their native paths at once; see
[Serving all four clients from one gateway](#serving-all-four-clients-from-one-gateway).

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
behavior; the on-demand GPU endpoint workflow uses the CUDA 13 default
`v0.32.0` image. The `-cu129` variant currently resolves a CUDA 13 torchcodec
wheel that cannot load without `libnvrtc.so.13`. The first run downloads a
multi-gigabyte image plus the model weights, and the Hugging Face cache mount
keeps the weights for later runs.

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
client-initiated safety classifier asks for roughly 2,112 output tokens
independently of the main Claude Code output-token setting and includes a large
client-owned prompt, which on 16K and 18K servers can alone exceed the window.
On an A10G, eager execution reclaims CUDA-graph memory and the 0.97 utilization
is reserved for this single-user development workload. If you lower the window
or share the GPU, do not use auto mode unless the classifier request still
fits.

Claude Code's default output budget also covers most of this window; section 4
explains why that is harmless on the vLLM version this guide serves.

## 2. Point a Praxis example at vLLM

Choose one example for your client and copy it outside `examples/`:

```console
# Codex, native Responses API (preferred for vLLM)
cp examples/configs/openai/responses/client-tool-compat.yaml praxis-vllm.yaml

# Codex, translated to Chat Completions
cp examples/configs/openai/responses/codex-http-chat-translation.yaml praxis-vllm.yaml

# Claude Code, native Anthropic API (preferred for vLLM), OpenCode, or pi
cp examples/configs/anthropic/messages-native-vllm.yaml praxis-vllm.yaml

# Claude Code, translated to Chat Completions
cp examples/configs/anthropic/messages-to-openai-vllm.yaml praxis-vllm.yaml
```

OpenCode was tested with the
[native Claude configuration](../../examples/configs/anthropic/messages-native-vllm.yaml).
It declares an explicit `/v1/chat/completions` route that forwards to vLLM
without body translation, and Anthropic validation is scoped to `/v1/messages`.
pi uses the same configuration and the same `/v1/chat/completions` route. The
same gateway can serve Claude Code, OpenCode, and pi together, provided vLLM
supports their respective endpoints.

### Serving all four clients from one gateway

One proxy process can serve all four CLIs over their native paths at once.
There is no example config for this, because it is the two native examples
above placed in one file: the Claude Code, OpenCode, and pi chain from
`anthropic/messages-native-vllm.yaml` and the Codex chain from
`openai/responses/client-tool-compat.yaml`, behind two listeners, pointed at
the same vLLM server and sharing one `GATEWAY_AUTH_PASSWORD`.

```yaml
listeners:
  # Claude Code (/v1/messages), OpenCode and pi (/v1/chat/completions).
  - name: messages-chat-gateway
    address: "127.0.0.1:8080"
    filter_chains: [claude-and-opencode]

  # Codex (/v1/responses).
  - name: responses-gateway
    address: "127.0.0.1:8081"
    filter_chains: [codex-responses]

filter_chains:
  - name: claude-and-opencode
    filters: # ... from anthropic/messages-native-vllm.yaml
  - name: codex-responses
    # Prefix with the same basic_auth filter the chain above uses, so one
    # shared credential gates both listeners. Give this chain's cluster its
    # own name; the endpoint is the same vLLM server. Also add a
    # credential_injection inside the IRR inference step — see below.
    filters: # ... from openai/responses/client-tool-compat.yaml
```

Two edits to the Codex chain are load-bearing, and the second is easy to miss.
The example has no auth filter, so it forwards the client's Bearer credential
to keyed vLLM untouched. Prefixing `basic_auth` (with its default
`strip_authorization`) removes that credential — and the chain has no
`credential_injection` of its own to replace it, so without a second edit the
sub-request reaches vLLM unauthenticated and fails with 401. Add the injection
inside the IRR `inference` step, after `router`, mirroring the Claude chain's
injection:

```yaml
              - filter: router
                routes:
                  - path_prefix: "/"
                    cluster: "vllm-responses-backend"
              # `basic_auth` at the head of the chain stripped the caller's
              # `Authorization`, so the sub-request reaches keyed vLLM
              # unauthenticated without this.
              - filter: credential_injection
                clusters:
                  - name: vllm-responses-backend
                    header: Authorization
                    env_var: VLLM_API_KEY
                    header_prefix: "Bearer "
                    strip_client_credential: true
              - filter: load_balancer
                # ... unchanged from the example
```

The cluster name inside the step must match the one the step's `router` and
`load_balancer` select.

Keep Codex on a listener of its own. Its rich client tools need
`openai_client_tool_compat`, which needs `openai_stream_events`, which fails
closed outside an `iterative_request_router` step. The IRR is a terminal filter
that cannot share a chain with a top-level `load_balancer`, so everything on
that listener would have to route through it — and the IRR buffers a
sub-request response unless a filter selects streaming, which the native
`/v1/messages` and `/v1/chat/completions` paths do not. Routing them through it
would hold each turn's SSE until the turn finished, so Claude Code, OpenCode,
and pi would render nothing until completion. Because every CLI configures its own
base URL, the second port costs one line of client config.

Point Claude Code, OpenCode, and pi at `http://127.0.0.1:8080` and Codex at
`http://127.0.0.1:8081/v1`, then follow sections 3 to 6 unchanged with two
exceptions: Codex now sits behind `basic_auth` and must present the shared
Basic credential rather than a Bearer key — see
[Codex behind the gateway's Basic auth](#codex-behind-the-gateways-basic-auth) —
and from the container image the store needs the same absolute `database_url`
as the native Codex example, see
[the writable database path note](#the-native-codex-example-needs-a-writable-database-path).

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
local response store used by the native Codex example and is harmless for the
other paths:

```console
cargo build -p praxis-ai-proxy --no-default-features \
  --features standard,openai-all,store-sqlite
./target/debug/praxis-ai -c praxis-vllm.yaml > /tmp/praxis-vllm.log 2>&1 &
export PRAXIS_PID=$!
```

Praxis listens at `http://127.0.0.1:8080`.

### Run Praxis from a container image

The published image leaves experimental Responses and the agentic loop out.
Build an image with an explicit opt-in for the examples in this guide:

```console
docker build -f Containerfile -t praxis-ai-responses \
  --build-arg PRAXIS_AI_FEATURES=full,openai-all,store-sqlite .
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
  praxis-ai-responses -c /etc/praxis/praxis.yaml
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
  praxis-ai-responses -c /etc/praxis/praxis.yaml
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

#### The native Codex example needs a writable database path

The custom image above includes `openai-all,store-sqlite`, so
`client-tool-compat.yaml` — the preferred native Codex example — is available. The other three
configurations in this section have no store filter and need no change at all.

The one edit the store example does need is an absolute `database_url`. The
image's `/etc/praxis` working directory is root-owned while the process runs as
`praxis` (UID 100), so the example's relative `sqlite://responses.db?mode=rwc`
cannot be created: the proxy starts, and the first `/v1/responses` request
fails with HTTP 500 and `unable to open database file`. Point `database_url` at
`/var/lib/praxis`, the writable state directory owned by the container user:

```yaml
- filter: openai_responses_store
  backend: sqlite
  database_url: "sqlite:///var/lib/praxis/responses.db?mode=rwc"
```

```console
docker run -d --rm --name praxis-vllm \
  --network host \
  -e VLLM_API_KEY -e GATEWAY_AUTH_PASSWORD \
  -v "$PWD/praxis-vllm.yaml:/etc/praxis/praxis.yaml:ro,z" \
  praxis-ai-responses -c /etc/praxis/praxis.yaml
```

The database then lives in the container's writable layer and disappears with
`--rm`, which is usually what a CLI test loop wants. To keep responses across
restarts, mount a volume at `/var/lib/praxis`: `-v praxis-state:/var/lib/praxis`
with Podman adds `:U` to chown it to the container user, and a host directory
needs `chown 100:100` (or `-m 0777`) before the first run.

The stock image rejects these experimental filters at startup. A custom image
can also serve this example with PostgreSQL by selecting `store-postgres`
instead of `store-sqlite`; see
`examples/configs/openai/responses/response-store-postgres-mtls.yaml`.

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

### Codex behind the gateway's Basic auth

The native Codex example has no `basic_auth` filter, so the client credential
above *is* the backend credential. Put Codex behind a gateway that
authenticates clients — as the Claude examples do, and as the
[all-in-one gateway](#serving-all-four-clients-from-one-gateway) does — and it
must present the Basic credential instead. Codex builds its provider headers
from `http_headers` and `env_http_headers` independently of `env_key`, and
applies the `env_key` bearer by *appending* `Authorization` rather than
replacing it, so setting both sends two `Authorization` headers. Omit `env_key`
entirely and carry the credential in `env_http_headers`:

```console
export PRAXIS_GATEWAY_AUTH="Basic $(printf 'gateway:%s' "$GATEWAY_AUTH_PASSWORD" | base64)"
```

```toml
[model_providers.praxis]
name = "Local Praxis"
base_url = "http://127.0.0.1:8081/v1"
wire_api = "responses"
env_http_headers = { Authorization = "PRAXIS_GATEWAY_AUTH" }
```

With no `env_key`, no `experimental_bearer_token`, and `requires_openai_auth`
left at its default, Codex resolves an unauthenticated provider and contributes
no auth headers of its own. The isolated `CODEX_HOME` is load-bearing rather
than merely tidy here: a logged-in Codex home would supply ambient credentials
that this provider would otherwise inherit.

`basic_auth` then verifies the client credential and strips it, and
`credential_injection` supplies `VLLM_API_KEY` toward vLLM — the same
three-credential separation the Claude examples use.

### The Codex app-server daemon outlives your shell

Codex 0.162 and newer do not issue provider requests from the TUI process: it
starts a long-lived `app-server` daemon under
`$CODEX_HOME/app-server-daemon/` and attaches to it. The daemon resolves
`env_http_headers` itself and keeps the environment of the shell that first
spawned it, so re-exporting credentials in a new shell does nothing for a
daemon that is already running — it keeps sending whatever
`PRAXIS_GATEWAY_AUTH` was, or was not, set to back then. The symptom is Codex
401ing against the gateway while a plain `curl` from the same shell succeeds,
because the curl carries the fresh credential and the daemon does not.

The tell is in the Praxis access log: the user agent carries two versions, the
TUI's and the daemon's, and a mismatch means the daemon is stale:

```text
user_agent.original="codex-tui/0.162.1 (…) (codex-tui; 0.160.1)"
                                 TUI ↑              ↑ daemon actually serving
```

Kill the daemon, then relaunch from a shell with the credential exported:

```console
kill "$(jq -r .pid "$CODEX_HOME/app-server-daemon/daemon.pid")"
kill "$(jq -r .pid "$CODEX_HOME/app-server-daemon/daemon-updater.pid")"
pgrep -af app-server   # confirm nothing survived
codex
```

A Codex upgrade does not restart the daemon either, so expect to repeat this
whenever the two versions in the user agent diverge. The isolated
`CODEX_HOME="$(mktemp -d)"` from the start of this section sidesteps the
problem for a fresh daemon but not for one spawned earlier in the same home.

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
# Optional headroom: cap the requested output budget (see the note below).
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192
# Tell Claude Code the served model's real window so it auto-compacts before
# vLLM rejects the prompt (see the note below).
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=32768
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

Claude Code does not know the served model's real window, so it sends the
default `max_tokens` for the Anthropic model name it believes it is calling —
around 21,000 tokens. vLLM treats that budget as an upper bound, not a
reservation: `get_max_tokens` caps it at `--max-model-len` minus the prompt
length, and a request is rejected up front only when the prompt itself exceeds
the window or the budget alone exceeds `--max-model-len` (verified against
vLLM 0.32.0 — the version the on-demand endpoint and the GPU nightly serve —
on both the Chat Completions and the Anthropic-compatible `/v1/messages`
paths). Claude Code's default stays below the 32,768-token window, so an
oversized budget costs no prompt space, and the 8192 cap above is optional: it
is kept to make each turn's output budget predictable on the 32,768-token
server, not to prevent a failure.

Claude Code also sizes its own auto-compaction against the window it assumes
for a model name it does not recognize, not the server's real one, so on its
own it will not compact in time: once the prompt alone passes
`--max-model-len`, every later request fails with HTTP 400 and an over-long
conversation cannot be compacted, only `/clear`ed.
`CLAUDE_CODE_MAX_CONTEXT_TOKENS` names the real window for a model Claude Code
does not recognize — any name that is not a canonical `claude-*` id, which
includes the served alias — so set it to the server's `--max-model-len`. It is
the variable the GPU nightly's Claude Code acceptance test sets against the
pinned Claude Code 2.1.278, whose compaction scenario asserts that the client
then auto-compacts below the server window on its own. Do not reach for
`CLAUDE_CODE_AUTO_COMPACT_WINDOW` here: it accepts 100,000 to 1,000,000 tokens,
and a smaller value draws a startup warning and is treated as the 100k minimum,
still above the 32,768-token server. Run `/compact` by hand if a long session
still creeps toward the limit. A larger window is the alternative, but Qwen3-8B
is natively 32,768 tokens and anything beyond it needs `--rope-scaling` on
`vllm serve`.

Praxis and vLLM do not implement Anthropic's server-side auto-mode classifier
protocol. `CLAUDE_CODE_AUTO_MODE_SERVER=0` makes Claude Code initiate the
classifier model requests through Praxis instead. This setting only affects
Claude Code's `auto` permission mode; the classifier still consumes model
inference and is not an on-device check. Do not set it to `1` for this setup.
See [Anthropic's auto-mode classifier documentation](https://code.claude.com/docs/en/auto-mode-classifier-billing).

## 5. Connect OpenCode

OpenCode 1.x and 2.x need different setups. Check your version with
`opencode --version`: 1.x prints a bare version such as `1.19.3`, and 2.x
prints `opencode v2.1.2`. The two majors ship as different npm packages
(`opencode-ai` for 1.x, `@opencode/cli` for 2.x) and use different
configuration schemas, so a 1.x configuration does not carry over to 2.x.

Both use the native Claude configuration from section 2. It forwards OpenCode's
`/v1/chat/completions` requests to vLLM and authenticates clients with Basic
auth using username `gateway` and password from `GATEWAY_AUTH_PASSWORD`.
Nothing changes on the Praxis side between the two majors.

### OpenCode 1.x

These steps are verified against OpenCode 1.19.3.

Add this configuration to
`~/.config/opencode/opencode.jsonc`, merging the `provider` entries with any
existing settings:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "praxis": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Praxis",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "headers": {
          "Authorization": "{env:PRAXIS_GATEWAY_AUTH}"
        }
      },
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
a loopback gateway.

OpenCode substitutes `{env:VAR}` in config values at load time, so the
credential never has to be written into the file. Export it before launching —
from the shell where `GATEWAY_AUTH_PASSWORD` was exported in section 2, which
must match the password in the Praxis process environment; OpenCode does not
need `VLLM_API_KEY`:

```console
export PRAXIS_GATEWAY_AUTH="Basic $(printf 'gateway:%s' "$GATEWAY_AUTH_PASSWORD" | base64)"
opencode --model "praxis/$VLLM_MODEL"
```

An unset `PRAXIS_GATEWAY_AUTH` substitutes to an empty string, which sends a
well-formed but empty `Authorization` header the gateway rejects with a bare
`401` — see the troubleshooting section.

If background traffic such as session titles should stay on the backend, pin
it the way Claude Code's model aliases are pinned: set
`"small_model": "praxis/qwen3-8b"` beside the provider, or it may resolve to
a model the backend does not serve and fail with a 404 on the model name.

Quit and restart OpenCode after changing its configuration. You can also
select the configured model with `/models`.

#### Why `options.headers`, and not `apiKey`

The credential must ride `options.headers.Authorization`, never
`options.apiKey`: `apiKey` is sent as `Authorization: Bearer`, which the
gateway's `basic_auth` — an RFC 7617 `Basic` parser — rejects with `401`.
`headers` is the one option that puts a complete, arbitrary `Authorization`
header on every request.

One caveat to know about: the published `https://opencode.ai/config.json`
schema does not declare a `headers` key under `provider.<id>.options` — it
names only `apiKey`, `baseURL`, `enterpriseUrl`, `setCacheKey`, `timeout`,
`headerTimeout`, and `chunkTimeout`. The key works anyway because OpenCode
passes the whole `options` record through to the `@ai-sdk/openai-compatible`
factory, which accepts `headers` — verified in source and live against
OpenCode 1.19.3 (the version this guide's integration tests pin), and earlier
against 1.18.35. Treat it as verified runtime behavior rather than a documented
contract: an OpenCode release that started validating provider options
against the schema would drop the key, and the per-model alternative below is
the schema-declared form to fall back to.

A second caveat, and a security one: the header follows the resolved
`baseURL` wherever configuration takes it. OpenCode deep-merges the global
configuration ahead of the project's, so a project that overrides only
`provider.praxis.options.baseURL` inherits the global
`options.headers.Authorization` and presents the gateway credential to
whatever host it chose. The plugin recipe this guide previously published
guarded exactly this — it refused to attach the credential unless `baseURL`
matched the expected gateway — and plain `options.headers` cannot express
that guard. If you open projects you do not fully trust, keep the `praxis`
provider entry in each trusted project's own `opencode.jsonc` rather than
the global file, and do not export `PRAXIS_GATEWAY_AUTH` in the shell that
launches OpenCode there: `{env:...}` substitution runs for project
configuration too, so the exported variable alone is enough for a project
to send the credential anywhere. The per-model alternative below shares the
same property.

#### Headless invocation

`opencode run` drives a single turn non-interactively, which is what CI and
scripts want:

```console
opencode run --model "praxis/$VLLM_MODEL" --format json -- "Summarize README.md"
```

The prompt is a **positional argument**. There is no `-p` prompt flag: on `run`,
`-p`/`--password` is the OpenCode *server* Basic-auth password, so copying the
`claude -p "..."` habit silently passes your prompt as a password. Use `--` to
separate the prompt from the flags.

`--format json` emits raw JSON events, one per line, instead of formatted
output. A successful turn emits `step_start`, one or more `text` events
(assistant text is at `part.text`), then `step_finish`; a failed turn emits an
`error` event carrying `statusCode` and the resolved request URL. `--print-logs`
and `--log-level DEBUG` are global flags (not `run` flags) and send diagnostics
to stderr — use them to confirm which `baseURL` the provider actually resolved.

**Redirect stdin from `/dev/null`.** If stdin stays open, `opencode run` blocks
indefinitely after logging `init` — before it issues a single request — even
though the prompt was supplied as an argument. In a script or CI job that
presents as a hang with no output and no network activity.

#### Alternative: per-model headers

A **per-model** `headers` map is part of the published schema — unlike the
provider-level `headers` pass-through above — so the gateway credential can be
attached declaratively:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "praxis": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Praxis",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": {
        "qwen3-8b": {
          "name": "Qwen3 8B",
          "limit": { "context": 32768, "output": 8192 },
          "headers": { "Authorization": "Basic <base64 of gateway:password>" }
        }
      }
    }
  }
}
```

The model object rejects unknown keys, so a misspelling is a hard startup error
rather than a silently ignored setting. The trade-off is the opposite of the
`{env:}` form: the base64 credential is baked into a file on disk, with no
environment lookup at runtime. Both recipes send the credential wherever
`baseURL` points — see the security caveat above. Use this form for CI and
ephemeral credentials, not for a developer's `~/.config`.

### OpenCode 2.x

These steps are verified against OpenCode 2.1.2. For the upstream changes, see
the [OpenCode V1 migration guide](https://opencode.ai/v2/docs/migrate-v1/).

Add this configuration to `~/.config/opencode/opencode.jsonc`, merging the
`providers` entry with any existing settings. 2.x renames `provider` to
`providers`, `npm` to `package` (with an `aisdk:` prefix), and moves
`options.baseURL` to `settings.baseURL`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "providers": {
    "praxis": {
      "package": "aisdk:@ai-sdk/openai-compatible",
      "name": "Praxis",
      "settings": { "baseURL": "http://127.0.0.1:8080/v1" },
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
`baseURL` with its HTTPS address, keeping the `/v1` suffix, and export the same
URL so the plugin can verify the destination:

```console
export PRAXIS_BASE_URL="https://gateway.example.com/v1"
```

For the local `http://127.0.0.1:8080/v1` example, no URL export is needed.

> **2.x ignores unknown configuration keys.** OpenCode 1.x rejected a
> misspelled key at startup. 2.1.2 ignores it without any message, so a typo
> such as `setting` instead of `settings` drops the `baseURL`. Copy the keys
> exactly.

Create `~/.config/opencode/plugins/praxis-auth.ts`. OpenCode 2.x loads every
plugin in that directory automatically, so the configuration needs no
`plugins` entry:

```typescript
import { Plugin } from "@opencode/plugin"

export default Plugin.define({
  id: "praxis-auth",
  async setup(ctx) {
    await ctx.session.hook(
      "model.request",
      (event) => {
        const trustedURL = process.env.PRAXIS_BASE_URL ?? "http://127.0.0.1:8080/v1"
        if (event.baseURL !== trustedURL) {
          throw new Error("Praxis baseURL does not match the trusted gateway")
        }
        const password = process.env.GATEWAY_AUTH_PASSWORD
        if (!password) throw new Error("GATEWAY_AUTH_PASSWORD is required for Praxis")
        const token = Buffer.from(`gateway:${password}`).toString("base64")
        event.headers.Authorization = `Basic ${token}`
      },
      { providerID: "praxis" },
    )
  },
})
```

The plugin runs before every model request to the `praxis` provider. It
checks the resolved `baseURL` against `PRAXIS_BASE_URL`, then attaches Basic
auth with username `gateway`. Set `PRAXIS_BASE_URL` only to a gateway you
trust. It must exactly match `baseURL`, including any trailing slash.

> **The guard now blocks the request, but a plugin that fails to load still
> fails open.** Verified against OpenCode 2.1.2:
>
> - If the hook throws (a `baseURL` mismatch or an unset
>   `GATEWAY_AUTH_PASSWORD`), OpenCode does **not** send the request. It emits
>   an `error` event carrying the plugin's message and exits `1`. This is
>   stricter than 1.x, where the turn went out without credentials.
> - If the plugin fails to **load** (a syntax error, a 1.x-style plugin, or a
>   `plugins` entry that points at a file rather than a directory), OpenCode
>   logs `failed to load plugin` at WARN and **continues**, sending every turn
>   with no `Authorization` header and exiting `0`. The message only appears
>   with `--print-logs`.
> - If you change the `providerID` in the plugin so it no longer matches the
>   provider, the hook never runs and nothing is logged.
>
> Against a real Praxis gateway an unauthenticated turn is rejected with
> `401`. Read a sudden `401` as a possible plugin load failure.

Launch OpenCode from the shell where `GATEWAY_AUTH_PASSWORD` was exported in
section 2. OpenCode does not need `VLLM_API_KEY`:

```console
opencode --model "praxis/$VLLM_MODEL"
```

#### Headless invocation (2.x)

```console
opencode run --standalone --model "praxis/$VLLM_MODEL" --format json "Summarize README.md" < /dev/null
```

The 2.x command line differs from 1.x in ways that fail quietly:

- **Pass `--standalone` in scripts.** Without it, `opencode run` connects to a
  shared background server (`opencode serve --service`) and starts one if none
  is running. That server keeps running after `run` exits. `--standalone` runs
  a private server that exits with the command.
- **Do not put `--` before the prompt.** In 2.1.2, `opencode run -- "hello"`
  sends `hello hello`, so the prompt is duplicated. Pass the prompt as the
  final argument with no separator.
- `-p` is not a flag on `run` in 2.x. `opencode run -p "..."` prints the help
  text and exits `1`.
- There is no `--pure` flag. To run without the plugin, remove it from
  `~/.config/opencode/plugins/`.
- **Redirect stdin from `/dev/null`.** As in 1.x, `opencode run` hangs with no
  output and no network activity while stdin stays open.

`--format json` emits one JSON event per line. A plain text turn emits only
`text` events (assistant text is at `part.text`). A turn that calls a tool
emits `step_start`, `tool_use`, `step_finish`, then `text`. A failed turn emits
an `error` event with `error.type` and `error.message`. A gateway `401`
appears as `error.type` `provider.auth` with `error.status` `401`. Use
`--print-logs --log-level warn` to see plugin load failures. `ERROR` hides
them, because they are logged at WARN.

2.x renames some built-in tools: `bash` is now `shell` and `task` is now
`subagent`. Use the new names when a prompt asks for a tool by name.

#### Alternative: provider headers without a plugin (2.x)

2.x accepts a `headers` map at the provider level, and `{env:NAME}`
substitution works inside it. You can attach the gateway credential with no
plugin and keep it out of the file:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "providers": {
    "praxis": {
      "package": "aisdk:@ai-sdk/openai-compatible",
      "name": "Praxis",
      "settings": { "baseURL": "http://127.0.0.1:8080/v1" },
      "headers": { "Authorization": "Basic {env:PRAXIS_GATEWAY_BASIC_TOKEN}" },
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

Export the Basic token, which is the base64 encoding of `gateway:<password>`:

```console
export PRAXIS_GATEWAY_BASIC_TOKEN="$(printf 'gateway:%s' "$GATEWAY_AUTH_PASSWORD" | base64)"
```

If the variable is unset, OpenCode substitutes an empty string and the gateway
answers `401`. Do not install the plugin alongside this configuration.

**The plugin is still the better choice for interactive use.** It refuses to
send anything when a project-level config has pointed `baseURL` at another
host. The `headers` map has no such check and sends the token wherever
`baseURL` points.

## 6. Connect pi

These steps are verified against pi 1.4.0. Install that version from the
[upstream release](https://github.com/earendil-works/pi/releases/tag/v1.4.0),
or with `npm install -g @earendil-works/pi-coding-agent@1.4.0` (Node 22.19 or
later), and check it with `pi --version`.

pi reaches custom endpoints through providers declared in
`~/.pi/agent/models.json`. It can speak several wire APIs; use
`openai-completions`, which sends native Chat Completions to the same
`/v1/chat/completions` route OpenCode uses. The native Claude configuration
from section 2 forwards those requests to vLLM without translation,
authenticates the client with Basic auth, and injects `VLLM_API_KEY` toward
vLLM. Nothing changes on the Praxis side.

Create `~/.pi/agent/models.json`, or merge the `praxis` provider into the one
you have:

```json
{
  "$schema": "https://pi.dev/schemas/models.schema.json",
  "providers": {
    "praxis": {
      "name": "Praxis",
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "local-pi-client-key",
      "headers": { "Authorization": "Basic ${PRAXIS_GATEWAY_BASIC_TOKEN}" },
      "models": [
        { "id": "qwen3-8b", "name": "Qwen3 8B", "contextWindow": 32768, "maxTokens": 8192 }
      ]
    }
  }
}
```

Replace `qwen3-8b` with the exact served name in `VLLM_MODEL` and adjust
`contextWindow` and `maxTokens` to match your vLLM server.

The `Authorization` header carries the gateway credential. Export the Basic
token it reads, which is the base64 encoding of `gateway:<password>`:

```console
export PRAXIS_GATEWAY_BASIC_TOKEN="$(printf 'gateway:%s' "$GATEWAY_AUTH_PASSWORD" | base64)"
```

Then launch pi from that shell. pi does not need `VLLM_API_KEY`:

```console
pi --model "praxis/$VLLM_MODEL"
```

Notes on this configuration, verified against pi 1.4.0:

- **`apiKey` is a placeholder, but it must be set.** With no `apiKey`, pi
  reports `No API key found for praxis.` and sends nothing. The configured
  `Authorization` header replaces the `Bearer` key pi would otherwise send, so
  the placeholder never leaves the client.
- **A missing token fails closed.** `${PRAXIS_GATEWAY_BASIC_TOKEN}` resolves at
  request time. If the variable is unset, pi sends no request and reports
  `Failed to resolve provider "praxis" header "Authorization" from environment
  variable: PRAXIS_GATEWAY_BASIC_TOKEN`. Header values also accept a leading
  `!command`, if you would rather read the token from a secret manager.
- **No `compat` or `reasoning` settings are needed for vLLM.** With
  `reasoning` left at its default of `false`, pi sends no reasoning or
  thinking fields, and Qwen3 thinking stays off through the server flags in
  section 1. Do not set `thinkingTokenBudgetField` for this setup.
- **The token goes wherever `baseUrl` points.** Unlike the OpenCode plugins,
  there is no destination check, so keep `models.json` pointed at your
  gateway.

#### Headless invocation

`pi --print` drives a single turn non-interactively, which is what CI and
scripts use:

```console
pi --print --mode json --offline --no-session --model "praxis/$VLLM_MODEL" -- "Summarize README.md" < /dev/null
```

- **Redirect stdin from `/dev/null`.** With stdin left open, `pi --print`
  waits for end of input and sends nothing. Anything piped in is prepended to
  the prompt.
- **In `--mode json`, a failed turn still exits `0`.** A gateway `401` or an
  unresolved header shows up only as `stopReason: "error"` with an
  `errorMessage` on the final assistant `message_end` event. `--mode text`
  exits `1` for the same failures. Scripts that use JSON mode must check the
  stop reason, not the exit status.
- pi needs no egress beyond Praxis; it completes normally in a loopback-only
  network namespace. `--offline` turns off its startup network operations
  anyway.
- pi registers four tools (`read`, `bash`, `edit`, `write`) and runs them
  without a permission prompt, including in `--print` mode. Run it in a
  workspace you are willing to let the model change.

## Cleanup

When finished, stop local Praxis and cancel the on-demand endpoint workflow:

```console
kill "$PRAXIS_PID"
```

## Troubleshooting

- `401` from Praxis on the Claude, OpenCode, or pi path: verify the Basic
  authorization header and that the client's `GATEWAY_AUTH_PASSWORD` matches
  the gateway's password.
- `opencode run` hangs after logging `init` and nothing reaches Praxis: stdin
  is still open. Redirect it from `/dev/null`. This is not a network problem —
  OpenCode needs no egress beyond Praxis itself, and completes normally in a
  loopback-only network namespace.
- OpenCode reaches the gateway but every turn fails with a bare `401` and no
  other error: `PRAXIS_GATEWAY_AUTH` was not exported in the shell that
  launched OpenCode, so `{env:PRAXIS_GATEWAY_AUTH}` substituted to an empty
  string and the request carries an empty `Authorization` header, which the
  gateway rejects like any bad credential. Export the variable and restart
  OpenCode; `--print-logs --log-level DEBUG` confirms which `baseURL` and
  headers the provider actually resolved.
- OpenCode reaches api.openai.com instead of Praxis, or asks for
  `OPENAI_API_KEY`: the provider `options` were not applied, which upstream has
  reported for both the bundled adapter and sub-sessions. Confirm the resolved
  `baseURL` with `--print-logs --log-level DEBUG` and pin a known-good version.
- OpenCode 2.x sends requests with no `Authorization` header and exits `0`: the
  plugin did not load. Re-run with `--print-logs --log-level warn` and look for
  `failed to load plugin`. Common causes: a 1.x-style plugin, a syntax error,
  or a `plugins` entry that points at a file rather than a directory. Leaving
  the plugin in `~/.config/opencode/plugins/` with no `plugins` entry avoids
  the last one.
- OpenCode 2.x exits `1` with `Praxis baseURL does not match the trusted
  gateway` or `GATEWAY_AUTH_PASSWORD is required for Praxis`, and nothing
  reaches Praxis: the plugin's guard refused the request. Fix
  `PRAXIS_BASE_URL` or export the password.
- OpenCode 2.x processes keep running after `opencode run` exits: the run
  started the shared background server. Pass `--standalone`, or stop the
  server with `opencode service stop`.
- `pi --print` hangs and nothing reaches Praxis: stdin is still open. Redirect
  it from `/dev/null`.
- pi reports `No API key found for praxis.`: the `praxis` provider in
  `~/.pi/agent/models.json` has no `apiKey`. Set the placeholder shown in
  section 6; the `Authorization` header still carries the real credential.
- pi reports `Failed to resolve provider "praxis" header "Authorization"` and
  nothing reaches Praxis: `PRAXIS_GATEWAY_BASIC_TOKEN` is not exported in the
  shell that launched pi.
- `401` from vLLM: `VLLM_API_KEY` does not match the key passed to vLLM.
- `401` from vLLM on the Codex listener only, in the
  [all-in-one gateway](#serving-all-four-clients-from-one-gateway): the
  `basic_auth` prefixed onto the Codex chain strips the client credential the
  example forwards, and no `credential_injection` inside the IRR `inference`
  step replaced it. Add the injection shown in section 2, after the step's
  `router` and before its `load_balancer`.
- `401` from Praxis on a Codex path gated by `basic_auth`: `env_key` is
  probably still set alongside `env_http_headers`. Codex appends the `env_key`
  bearer rather than replacing the configured header, so the request carries
  two `Authorization` headers and `basic_auth` does not see the Basic one it
  expects. Remove `env_key` from the provider block. A stale logged-in
  `CODEX_HOME` can do the same thing; use the `mktemp -d` home from section 3.
- Codex 401s while a plain `curl` to the same listener from the same shell
  succeeds: the `app-server` daemon is stale and holds the environment of the
  shell that first spawned it. Check the Praxis access log for a user agent
  carrying two Codex versions, then kill the daemon and relaunch — see
  [The Codex app-server daemon outlives your shell](#the-codex-app-server-daemon-outlives-your-shell).
  A Codex upgrade leaves the daemon running too.
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
  every OpenAI and Anthropic protocol; put it directly after `basic_auth`, as
  the Codex examples and the native Claude Code and OpenCode example do.
  This bites OpenCode hardest: it is the one client whose traffic
  (`/v1/chat/completions`) shares a listener with another protocol, so a config
  derived from an older copy sends it problem details its SDK cannot parse.
- `400` with `maximum context length is 32768 tokens`: the prompt itself
  exceeds the server's window. vLLM clamps the requested output budget to the
  space left after the prompt rather than holding it against the window, so an
  oversized `max_tokens` is never the cause on its own. Reduce what Claude
  Code loads before the first turn — see the `/context` entry below.
- `400` with `maximum context length is 32768 tokens` on the first turn, with a
  reported input count near 28,000 and a modest output count: the startup
  prompt itself is the problem. Claude Code loads MCP tool definitions, plugin
  skills, and `CLAUDE.md` before the session begins, so an empty conversation
  can consume most of the window. Run `/context` for the breakdown; it is a
  local command and still works while every request is failing. Start Claude
  Code with `--strict-mcp-config` to drop globally configured MCP servers,
  disable unneeded plugins with `/plugin`, and prefer a working directory whose
  `CLAUDE.md` is small or absent.
- Claude Code never finishes a turn — plan mode keeps looping, or reasoning
  text appears in the answer or the rendered plan: the server may be emitting
  thinking that the client is not meant to see, which can desynchronize the
  agentic loop. Serve Qwen3 with
  `--reasoning-parser qwen3` and
  `--default-chat-template-kwargs '{"enable_thinking":false}'` as shown in
  section 1, and restart vLLM; the reasoning parser is a server flag, so
  nothing on the Praxis or client side changes it. On the on-demand endpoint,
  check the `reasoning_parser` and `enable_thinking` inputs of the run.
- Auto mode fails with `Error: You are not in plan mode. To enter plan mode,
  call the EnterPlanMode tool first.`: treat this as a symptom, not a root
  cause. It can be a plain Claude Code plan-state failure, or reasoning text
  bleeding into the response and desynchronizing the agentic loop. Check whether
  reasoning text is also present before applying the Qwen3 settings above.
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
  filter in place and terminate TLS in front of it — the shared Basic password
  is otherwise readable on the wire. A second listener, such as the Codex one
  in the
  [all-in-one gateway](#serving-all-four-clients-from-one-gateway), needs the
  same treatment plus its own `-p 8081:8081` under the published-port container
  path.
- Claude startup probes may call `/v1/messages/count_tokens`. Native vLLM
  supports it; the translated Chat path can return 404 and Claude degrades
  gracefully.
