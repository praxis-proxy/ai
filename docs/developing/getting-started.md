# Getting Started

## Requirements

- Rust stable 1.96+
- Rust nightly
- CMake 3.31+
- Docker 29.3.0+ or Podman (for container builds and RustFS S3 integration
  tests; the FIPS image, its compliance check and Red Hat's scanner need
  podman on Linux)
- AWS CLI v2 (for `make test-integration` and `make test-s3-file-store-e2e`)
- Go 1.26+ (`make fips-scanner`, optional)

Install the development-only Cargo tools used by the Make targets:

```console
cargo install cargo-audit cargo-deny
cargo install cargo-machete --version 0.9.2 --locked
rustup toolchain install nightly --component rustfmt
```

`cargo-machete` is pinned to match CI because 0.10.1 uses `Path::is_empty`,
which is unstable on Rust 1.96.

## Conventions

Read [CONTRIBUTING.md] before contributing. It covers code style, testing
requirements, file organization, and security practices, and submissions that
do not follow it are rejected.

[CONTRIBUTING.md]:../../CONTRIBUTING.md

## Build

```console
make build
make release
make check
```

### Test

```console
make test
```

```console
make test-integration
```

CI runs some suites from the local aggregate targets separately.

| Local target | Runs | CI targets |
| --- | --- | --- |
| `make test-integration` | store-all suite, its experimental filter permutation, and live RustFS S3 FileStore tests | `make test-integration-core`, `make test-integration-experimental`, `make test-s3-file-store` |
| `make test-inference-fixtures` | inference fixtures and the xtask fixture tests | `make test-inference-fixtures-core`, `make test-inference-xtask` |
| `make test-environment-suite` | `make test-environment`'s environment tests and the llmd-ext-proc integration suite | |
| `make test-store-features` | feature graph checks, per-group Clippy, and API tests | its `-check`, `-clippy`, and `-test` targets |

The RustFS container requires an available port at `127.0.0.1:9000`.

### OpenAI SDK Integration Tests (Dual 2.x / 3.x Lanes)

Praxis AI supports OpenAI Python SDK 2.x and 3.x, and local and CI runs test
each major version separately. The suites need a build with the experimental
OpenAI filters:

```console
cargo build -p praxis-ai-proxy --features openai-all,store-sqlite
```

To run `test_openai_responses_vllm.py` in simulator mode locally, start the `inference-sim` container first:

```console
podman run --detach --name inference-sim --network host \
  ghcr.io/llm-d/llm-d-inference-sim@sha256:32144df791330a0006b747edfdf2b114a0fe728e023a9d1b3463eeb48d32abb9 \
  --model=praxis-test-model --served-model-name=praxis-test-model --mode=echo --max-model-len=8192 --skip-tool-validation --port=8000
```

SDK 2.x lane (`-k "not file_search"` skips tests that need local OGX vector search):

```console
uv run --with "openai<3" tests/integration/sdk/openai/test_openai_conversations.py -v
VLLM_MODEL=praxis-test-model VLLM_TEST_BACKEND=simulator uv run --with "openai<3" tests/integration/sdk/openai/test_openai_responses_vllm.py -s -m "not real_inference and not vllm_compat" -k "not file_search"
```

SDK 3.x lane:

```console
uv run --with "openai>=3,<4" tests/integration/sdk/openai/test_openai_conversations.py -v
VLLM_MODEL=praxis-test-model VLLM_TEST_BACKEND=simulator uv run --with "openai>=3,<4" tests/integration/sdk/openai/test_openai_responses_vllm.py -s -m "not real_inference and not vllm_compat" -k "not file_search"
```

Pass `--sdk-version=2.x` or `--sdk-version=3.x` to verify the installed SDK
version at pytest startup.

### Anthropic SDK Integration Tests (Dual 1.x / 2.x Lanes)

Praxis AI also supports Anthropic Python SDK 1.x and 2.x. SDK v2 changes the
client's wire defaults (`anthropic-version: 2026-10-21` and
`authorization: Bearer` instead of `2023-06-01` and `x-api-key`), so each lane
is pinned and tested separately:

```console
cargo build -p praxis-ai-proxy
uv run --with "anthropic==1.17.0" tests/integration/sdk/anthropic/test_anthropic_sdk_compat.py -v --anthropic-sdk-version=1.x
uv run --with "anthropic==2.3.0" tests/integration/sdk/anthropic/test_anthropic_sdk_compat.py -v --anthropic-sdk-version=2.x
```

The pinned versions match the `make test-anthropic-sdk-v1` and
`test-anthropic-sdk-v2` targets and CI. `make test-anthropic-sdk` runs both
lanes locally; CI runs them as sequential steps in the `anthropic-sdk` job of
`integration.yaml` after one binary build. `--anthropic-sdk-version=1.x` or
`--anthropic-sdk-version=2.x` verifies the SDK version at pytest startup. The
suites clear ambient `ANTHROPIC_*` environment variables so wire-header
assertions stay deterministic.

### OpenAI Agents SDK Responses Compatibility

The OpenAI Agents SDK function-calling loop is tested through the Responses
API; the OpenAI Agents API is not implemented. `test_openai_agents_sdk.py`
runs the SDK `Runner` through a deterministic, client-owned function-tool loop
against a local Praxis listener. The scripted Responses backend emits a
`function_call`, the SDK runs a local Python tool exactly once, the
continuation request carries the matching `function_call_output`, and the
final message returns a unique marker. The test asserts that only
`POST /v1/responses` is called, never `/v1/agents/*` or Chat Completions.

The test pins `openai-agents` and its own compatible `openai` in its PEP 723
header, so it runs once rather than per storage backend or OpenAI SDK lane:

```console
cargo build -p praxis-ai-proxy --features openai-all,store-sqlite
uv run tests/integration/sdk/openai/test_openai_agents_sdk.py -s
```

It needs no credentials, external network, inference probability, retry, or
`xfail`. `-s` prints the pinned Agents SDK and OpenAI SDK versions.

### FIPS Build and Compliance Check

Praxis AI targets FIPS 140-3 on Red Hat Enterprise Linux by performing all
cryptography in the RHEL OpenSSL FIPS provider. The published build
(`make release`, `make container`) enables every non-experimental filter.
The FIPS build turns off what is known not to be compliant yet (the policy
engine, the response stores and the reqwest-based filters; the Makefile's
FIPS section says why for each), so nobody has to know which features to
pick. The feature set is defined once, as
`FIPS_FEATURES` in the `Makefile`:

```console
make release-fips    # FIPS build, release profile, into target/fips
make build-fips      # same, debug profile, without the crate manifest
make container-fips  # FIPS runtime image on UBI 9, tagged praxis-ai:<version>-fips
make lint-fips       # clippy and rustfmt for the FIPS feature set
make test-fips       # unit tests for the FIPS feature set
make test-integration-fips  # the integration suite as the FIPS build
make test-schema-fips       # the schema suite as the FIPS build
make test-fips-host  # the suites as the FIPS build, inside the UBI 9 toolchain image, on a FIPS host
```

`make container-fips`, `make fips-check`, `make fips-smoke` and
`make fips-scan` need a Linux podman (rootless or root). On Debian and
Ubuntu, run `make fips-signature-store` once first. Nothing is installed
into the runtime image beyond the binary and the legal files.

The compliance checks (`fips-deps`, `fips-report`, `fips-check`, the Red Hat
scanner targets) and the FIPS-host targets (`fips-host-check`,
`fips-runtime-probe`) are described in [FIPS Tooling](fips.md).
[FIPS 140-3](../operations/fips.md) is the operator's view.

### Supply Chain Safety

`make audit` runs `cargo audit` and `cargo deny check`. `deny.toml` bans
wildcard version requirements, unknown registries, and unknown git sources,
and warns on multiple versions of one crate. All crates enforce
`#![deny(unsafe_code)]`, and Clippy runs with `-D warnings`.

The workspace is split across `apis`, `filters`, `server`, `tests`, and
`xtask`, with shared dependencies managed in the root `Cargo.toml`.

See [SECURITY.md](../../.github/SECURITY.md) for supported versions and
vulnerability reporting.

## Security: Binding Low Ports

On Unix, Praxis refuses to start as root (UID 0). The check runs before any
port binding or protocol registration. To bind ports below 1024:

- Grant `CAP_NET_BIND_SERVICE` to the binary:
  `sudo setcap cap_net_bind_service=+ep ./target/release/praxis-ai`
- Run behind a reverse proxy or load balancer that handles port 80/443.
- Use systemd socket activation to pass pre-bound sockets.

## Insecure Options

> **Warning.** These flags are for development and testing only. Never enable
> them in production. Each one demotes a security check from an error to a
> warning.

All flags live under `insecure_options` in the YAML config and default to
`false`.

| Flag | Effect |
| ------ | -------- |
| `allow_open_security_filters` | Allow security-critical filters (`ip_acl`, `forwarded_headers`) to use `failure_mode: open`. Without it, open security filters are rejected because a runtime error would bypass enforcement. |
| `allow_private_health_checks` | Allow health check endpoints that resolve to loopback (`127.0.0.0/8`), link-local (`169.254.0.0/16`), or cloud metadata addresses. Blocked by default as SSRF protection. |
| `allow_public_admin` | Allow the admin health endpoint to bind to a public interface (`0.0.0.0` / `[::]`), which is otherwise a validation error. |
| `allow_root` | Allow starting as root (UID 0). Praxis refuses to run as root by default. |
| `allow_tls_without_sni` | Allow upstream TLS connections without an explicit SNI hostname. Most TLS servers require SNI, so missing SNI is otherwise a validation error. |
| `allow_unbounded_body` | Allow unbounded body processing. Covers two checks that are otherwise rejected at startup: `body_limits.max_request_bytes` or `max_response_bytes` set to `null`, and `StreamBuffer` body mode without a `max_bytes` limit. |
| `csrf_log_only` | Run the CSRF filter in log-only mode: evaluate all rules but log violations as warnings instead of rejecting requests, for example during an initial rollout. |
| `skip_pipeline_validation` | Demote pipeline ordering errors, such as filter placement, to warnings instead of failing startup. |

For listeners that buffer requests, a finite `body_limits.max_request_bytes`
must not exceed `body_limits.max_buffer_bytes`; startup and reload reject the
configuration otherwise.

Example overriding two flags for local development:

```yaml
admin:
  address: "0.0.0.0:9901"

insecure_options:
  allow_public_admin: true
  allow_private_health_checks: true
```

## Shared Build Cache with sccache

[sccache] caches compiled artifacts, so switching branches, cleaning
`target/`, or working across git worktrees does not rebuild every dependency.

### Setup

[Install sccache][sccache-install], then add this to your shell profile
(`~/.bashrc`, `~/.zshrc`):

```sh
export RUSTC_WRAPPER=sccache
```

### Warming the cache

Run a full clippy pass in any worktree to fill the cache:

```console
cargo clippy --workspace --all-targets
```

Later builds reuse the cache. Cargo still prints `Compiling` / `Checking` for
every crate, but cache hits finish in milliseconds. Check hit rates with
`sccache --show-stats`; see [sccache usage][sccache-usage] for configuration.

[sccache]: https://github.com/mozilla/sccache
[sccache-install]: https://github.com/mozilla/sccache#installation
[sccache-usage]: https://github.com/mozilla/sccache#usage

## Performance & Benchmarking

Include benchmark or load-test evidence in PRs that materially change
request processing, storage, streaming, or protocol parsing costs.

`make bench-model-to-provider` runs the ignored model-to-provider request-body
benchmark in the release profile; the Tests workflow runs it in its filter job.
It reports nanoseconds per request over 10,000 iterations for mapped models
with 64-byte, 1 KiB, and 16 KiB prompts, and an unknown model with 1 KiB and
16 KiB prompts. Timings cover request-context construction, shared body
ownership, the filter call, and cleanup, without network or backend latency.
Shared CI runner timings are informational, not a throughput guarantee or
latency budget.
