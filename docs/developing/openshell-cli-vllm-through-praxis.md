# Run your Code Harness in OpenShell through Praxis and vLLM

This guide runs Codex, Claude Code, OpenCode, and pi inside
[NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell) sandboxes while Praxis
and vLLM keep serving inference from the host. It builds on
[Run your Code Harness through Praxis and vLLM](cli-vllm-through-praxis.md):
the vLLM setup, Praxis build, client flags, and troubleshooting there all still
apply. This guide covers only what the sandbox changes. It was verified against
OpenShell 0.1.6 with the Podman compute driver, with each harness at the version
pinned under `tests/integration/fixtures/` completing a file-writing tool turn
against Qwen3-30B on vLLM.

```text
┌──────────── OpenShell sandbox ────────────┐
│ Claude Code / OpenCode / pi / Codex        │
│   GATEWAY_AUTH_PASSWORD = placeholder      │
│            │                               │
│   OpenShell proxy: policy check, then      │
│   swaps in the real gateway password       │
└────────────┼───────────────────────────────┘
             │  host.openshell.internal:8080 (messages, chat)
             │  host.openshell.internal:8081 (responses)
             ▼
   Praxis on the host: basic_auth -> native route -> inject VLLM_API_KEY
             │
             ▼
   vLLM (local or on-demand endpoint)
```

The setup keeps each credential at its own boundary:

| Credential | Held by | Reaches |
| --- | --- | --- |
| Gateway password (`GATEWAY_AUTH_PASSWORD`) | OpenShell's credential store and the Praxis process | Praxis only; the sandbox sees a placeholder |
| vLLM key (`VLLM_API_KEY`) | The Praxis process | vLLM only; never enters a sandbox |
| Client API key placeholders | The harness | Stripped by Praxis |

Sandboxes can reach the two Praxis listeners and nothing else unless you grant
more access.

## How OpenShell reaches a host gateway

OpenShell 0.1 removed its managed inference route (`inference.local` and the
`openshell inference` commands). A sandboxed client now calls its provider's
native API directly, and access comes from a **provider profile** attached to
the sandbox. Four properties of that model shape this setup:

- **Stock agent profiles don't fit.** The upstream `claude-code` and `codex`
  profiles allow only `api.anthropic.com` and the OpenAI hosts. Pointing a
  client's base URL elsewhere does not carry the credential with it, so this
  guide ships its own profile for the Praxis gateway.
- **Loopback is never allowed.** OpenShell never authorizes sandbox egress to
  loopback. Sandboxes reach host services through `host.openshell.internal`,
  so the Praxis listeners must bind an address the container bridge can reach.
  Every other Praxis example binds `127.0.0.1`.
- **Credentials are placeholders.** The sandbox environment carries
  `openshell:resolve:env:...` values. The OpenShell proxy substitutes the real
  value only for requests to the profile's endpoints, and only where it inspects
  HTTP (`protocol: rest`). That includes Basic credentials: the proxy decodes
  `Authorization: Basic`, substitutes the placeholder, and re-encodes it. So a
  harness can build `Basic base64("gateway:$GATEWAY_AUTH_PASSWORD")` from the
  placeholder, and Praxis receives the real password.
- **`binaries` gates access.** Only the listed executables may use the
  profile's endpoints. Matching uses real paths, globs are allowed, and child
  processes inherit the rule.

## 1. Start vLLM and Praxis on the host

Install OpenShell and start its gateway:

```console
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh
```

Start vLLM as in [section 1 of the base guide](cli-vllm-through-praxis.md#1-start-vllm),
and record `VLLM_URL` and `VLLM_MODEL`.

Copy the sandbox gateway example. It serves Claude Code, OpenCode, and pi on
port 8080 and Codex on port 8081, all over their native APIs, behind one
`basic_auth` credential:

```console
cp examples/configs/openshell/sandboxed-coding-clients.yaml praxis-openshell.yaml
```

The example targets a local vLLM at `127.0.0.1:8000`. For the on-demand HTTPS
endpoint, change both `load_balancer` endpoints and add `tls.sni` as described
in [section 2 of the base guide](cli-vllm-through-praxis.md#2-point-a-praxis-example-at-vllm).

For a vLLM server elsewhere on your network, replace both `127.0.0.1:8000`
endpoints with its `host:port`. If that hostname resolves to a private address,
also add `allow_private_upstreams: true` beside the existing
`allow_private_endpoints: true`. Without it, requests fail with
`upstream hostname resolved to private/reserved IP address`.

Both listeners bind `0.0.0.0`, because traffic from sandboxes arrives on the
host's container-bridge interface. Firewall ports 8080 and 8081 from external
networks, or replace `0.0.0.0` with the bridge address your OpenShell driver
uses (for example `docker0`'s address).

Set the two secrets in the shell that starts Praxis, then build and start it:

```console
export VLLM_API_KEY="${VLLM_API_KEY:-$(openssl rand -hex 32)}"
export GATEWAY_AUTH_PASSWORD="$(openssl rand -hex 24)"

cargo build -p praxis-ai-proxy --no-default-features \
  --features standard,openai-all,store-sqlite
./target/debug/praxis-ai -c praxis-openshell.yaml > /tmp/praxis-openshell.log 2>&1 &
export PRAXIS_PID=$!
```

The Codex chain uses experimental Responses filters, so the stock container
image rejects this config. To run Praxis in a container instead, build the
custom image from
[the base guide](cli-vllm-through-praxis.md#run-praxis-from-a-container-image),
run it with `--network host`, and change the example's `database_url` to
`sqlite:///var/lib/praxis/responses.db?mode=rwc`.

## 2. Create the Praxis gateway provider

[`examples/openshell/praxis-gateway-profile.yaml`](../../examples/openshell/praxis-gateway-profile.yaml)
is an OpenShell provider profile, not a Praxis config. It allows the gateway
routes on `host.openshell.internal:8080` and `:8081` with L7 enforcement,
declares `GATEWAY_AUTH_PASSWORD` as a Basic credential bound to those
endpoints, and lists the harness binaries that may use them. Trim `binaries` to
what your image installs.

Lint and import it, then create a provider from the password in your shell:

```console
openshell profile lint -f examples/openshell/praxis-gateway-profile.yaml
openshell profile import -f examples/openshell/praxis-gateway-profile.yaml
openshell provider create --name praxis --type praxis-gateway \
  --credential GATEWAY_AUTH_PASSWORD
```

`--credential KEY` reads the value from the CLI's environment, so the password
never appears in the command line. OpenShell stores it and never exports it.

## 3. Build a sandbox image

[`examples/openshell/Containerfile.harnesses`](../../examples/openshell/Containerfile.harnesses)
installs the npm packages you name on the official Node image, along with the
search tools harnesses would otherwise try to download, and the
[`praxis-harness-env`](../../examples/openshell/praxis-harness-env.sh) entrypoint
used below. Take each version from its pin under
`tests/integration/fixtures/*/pin.toml`:

```console
docker build -f examples/openshell/Containerfile.harnesses -t praxis-harness:local \
  --build-arg HARNESS_PACKAGES="@anthropic-ai/claude-code@<pin> @openai/codex@<pin> opencode-ai@<pin> @earendil-works/pi-coding-agent@<pin>" \
  examples/openshell
```

The build context is `examples/openshell`, because the repository's
`.dockerignore` excludes `examples/`. Build with the container engine your
OpenShell gateway uses. With Podman, tag the image `localhost/praxis-harness:local`
and use that reference with `--from`.

The image keeps every harness's settings under `/tmp`, because the default
sandbox policy makes only `/tmp` and the working directory writable.

`praxis-harness-env` turns the provider's placeholder into each harness's
gateway configuration and then runs the command you give it:

- It exports `PRAXIS_GATEWAY_AUTH` (`Basic <token>`) and
  `PRAXIS_GATEWAY_BASIC_TOKEN`, plus the Claude Code variables from
  [section 4 of the base guide](cli-vllm-through-praxis.md#4-connect-claude-code),
  pointed at `http://host.openshell.internal:8080`.
- It writes the Codex `config.toml` (pointed at port 8081, with
  `env_http_headers`, no `env_key`, and Codex's inner sandbox off; see
  [Codex](#codex)), the OpenCode 1.x `opencode.json`, and the pi `models.json`.
- It warns if `GATEWAY_AUTH_PASSWORD` is not an OpenShell placeholder, which
  means the real password was passed with `--env`.

`VLLM_MODEL` is required. `PRAXIS_MODEL_CONTEXT` (default 32768) and
`PRAXIS_MODEL_MAX_OUTPUT` (default 8192) set the context and output limits the
clients assume.

Check the path end to end before starting a harness:

```console
openshell sandbox create --name praxis-smoke --from praxis-harness:local \
  --provider praxis --env VLLM_MODEL="$VLLM_MODEL" -- \
  praxis-harness-env sh -c 'curl -sS -H "Authorization: $PRAXIS_GATEWAY_AUTH" \
    http://host.openshell.internal:8080/v1/models'
```

A model list means policy, credential substitution, `basic_auth`, and the vLLM
key all worked. Delete the sandbox with `openshell sandbox delete praxis-smoke`.

## 4. Run a harness

Each command below starts a sandbox with the `praxis` provider attached. For a
single non-interactive turn, replace the command after `--` with the harness's
headless form. Add
`--upload .:/workspace` and drop the trailing command to work on a local
project; OpenShell does not allow both, so run `praxis-harness-env <harness>`
from the shell that opens instead. Copy changes back with
`openshell sandbox download`.

### Claude Code

```console
openshell sandbox create --name claude --from praxis-harness:local \
  --provider praxis --env VLLM_MODEL="$VLLM_MODEL" -- \
  praxis-harness-env claude --model "$VLLM_MODEL"
```

Headless: `praxis-harness-env claude -p 'reply with OK' < /dev/null`.
The reasoning-parser, context-window, and auto-mode notes in
[section 4 of the base guide](cli-vllm-through-praxis.md#4-connect-claude-code)
apply unchanged.

### Codex

```console
openshell sandbox create --name codex --from praxis-harness:local \
  --provider praxis --env VLLM_MODEL="$VLLM_MODEL" -- \
  praxis-harness-env codex
```

Headless: `praxis-harness-env codex exec --skip-git-repo-check 'reply with OK'`.
Each sandbox starts with a fresh `CODEX_HOME`, so the stale `app-server` daemon
described in the base guide cannot carry over between sandboxes.

The generated configuration sets `sandbox_mode = "danger-full-access"`. Codex
normally wraps each tool command in its own bubblewrap sandbox, but bubblewrap
cannot create user namespaces inside an OpenShell sandbox. Every command then
fails with `bwrap: No permissions to create a new namespace`, and the model
reports success without having changed anything. OpenShell already confines
the process, so Codex's inner sandbox adds nothing here. The startup warning
about not creating PATH aliases under `/tmp` is harmless.

### OpenCode

```console
openshell sandbox create --name opencode --from praxis-harness:local \
  --provider praxis --env VLLM_MODEL="$VLLM_MODEL" -- \
  praxis-harness-env opencode --model "praxis/$VLLM_MODEL"
```

Headless:
`praxis-harness-env opencode run --model "praxis/$VLLM_MODEL" --format json -- 'reply with OK' < /dev/null`.
The generated configuration is the OpenCode 1.x `options.headers` form. For 2.x,
follow [the base guide's 2.x section](cli-vllm-through-praxis.md#opencode-2x)
with `baseURL` `http://host.openshell.internal:8080/v1`.

### pi

```console
openshell sandbox create --name pi --from praxis-harness:local \
  --provider praxis --env VLLM_MODEL="$VLLM_MODEL" -- \
  praxis-harness-env pi --model "praxis/$VLLM_MODEL"
```

Headless:
`praxis-harness-env pi --print --mode json --offline --no-session --model "praxis/$VLLM_MODEL" -- 'reply with OK' < /dev/null`.
pi's four tools run without a permission prompt, but now only inside the sandbox.

## 5. Watch and extend the policy

Denied connections show up in the sandbox log:

```console
openshell logs claude --since 5m --source sandbox
```

Don't filter with `--level warn`, which hides policy events. OpenShell turns
denials into draft rules. Review and approve them for anything the harness
legitimately needs, such as a package registry:

```console
openshell rule get claude --status pending
openshell rule approve claude --chunk-id <chunk-id>
```

Approved network rules hot-reload into the running sandbox. Filesystem and
process policy is fixed when a sandbox is created.

Some denials are expected and harmless, because none of them is needed to
complete a turn:

- Claude Code sends `HEAD /api/hello` to the gateway as a connectivity probe.
  The profile allows only the inference routes, so the probe is denied.
- Codex tries to reach `chatgpt.com`, `github.com`, and `api.github.com` for
  account and update checks.
- OpenCode tries to reach `models.opencode.ai` for its model catalog and
  `registry.npmjs.org` for provider packages; the `@ai-sdk/openai-compatible`
  adapter it needs is bundled.
- pi makes no requests beyond Praxis.

Rules without a `query` matcher allow any query string, so Claude Code's
`POST /v1/messages?beta=true` matches the `/v1/messages` rule as written.

## Cleanup

```console
openshell sandbox delete claude codex opencode pi
kill "$PRAXIS_PID"
```

## Troubleshooting

- **Connection refused or timeout to `host.openshell.internal`:** the Praxis
  listener is still bound to `127.0.0.1`, or a host firewall blocks the
  container bridge. Check with `ss -ltnp | grep -E ':808[01]'` on the host.
- **`policy_denied` for a request to the gateway:** the requesting executable is
  not in the profile's `binaries`. The denial log names the path OpenShell
  resolved; add it, or run `readlink -f "$(command -v <harness>)"` inside the
  sandbox. A method or path outside the profile's `rules` is denied the same
  way, so a new client endpoint needs a new rule.
- **HTTP 403 `credential_endpoint_mismatch`:** the client sent the placeholder
  to a host or port outside the profile, usually a base URL that doesn't use
  `host.openshell.internal` with port 8080 or 8081.
- **401 from Praxis with a placeholder in the request:** substitution did not
  happen. The header must be built inside the sandbox from the provider's
  `GATEWAY_AUTH_PASSWORD`, and the endpoint must use `protocol: rest` so
  OpenShell inspects it. A password passed with `--env` is sent as typed, and
  `praxis-harness-env` warns about it.
- **401 from Praxis after rotating the password:** update both the Praxis
  process environment and the provider, then start a new process in the
  sandbox; running processes keep their environment.
- **401 from vLLM:** `VLLM_API_KEY` in the Praxis environment does not match
  the key vLLM was started with. The sandbox is not involved.
- **The image cannot be found:** build it with the engine your OpenShell gateway
  uses, or push it to a registry the gateway can pull from.

For client-specific failures, see the
[base guide's troubleshooting](cli-vllm-through-praxis.md#troubleshooting).
