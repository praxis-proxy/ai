# Full OpenResponses acceptance

The [upstream acceptance suite](https://www.openresponses.org/compliance)
and our [translation coverage](openresponses-translation.md) answer different
questions. Translation coverage runs seven selected templates against a
text-only CPU backend. Full acceptance runs every template in the pinned
upstream inventory, including image input, compaction, and WebSockets.

## Verified result: November 12, 2026

All 17 templates were run with Bun 1.3.12 against a local macOS Praxis binary
built at `24653c7a4edd38d8fef43fe553773dea92b92d42` with the test harness
changes in this working tree. The suite revision was
`dce9ddb91921427c2ba06901c7dfaf0cc7056799`. The backend was real vLLM 0.32.0
on a `g5.xlarge` GPU, serving `Qwen/Qwen3-VL-4B-Instruct` as `qwen3-vl-4b`,
with the Hermes tool parser, qwen3 reasoning parser and thinking disabled.
The dev workflow did not pin the model weight revision.

| Profile | Passed | Failed | Skipped |
| --- | --- | --- | --- |
| Praxis-managed HTTP, backend-owned WebSockets | 10 | 7 | 0 |
| Native Responses passthrough | 1 | 16 | 0 |

The managed profile passed all nine HTTP endpoint cases and the local phase
schema fixture. This includes actual image inference (the model identified
the supplied red heart), function calling, SSE streaming, compaction, and
missing-model rejection. The separate real-image check also passed through
the official OpenAI Python SDK 3.49.0.

All seven managed-profile failures were WebSocket connection failures. An
upgrade request sent directly to the backend returned HTTP 405, and the
[vLLM 0.32.0 Responses router](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/entrypoints/openai/responses/api_router.py)
has no WebSocket handler. These results do not demonstrate a Praxis transport
failure, but they do mean this deployment cannot claim full suite compliance.
The native profile also failed Responses-resource schema checks (including
missing `store` and incompatible/null fields) and returned HTTP 405 for
compaction. Its only pass was the local schema fixture.

The [GPU dev workflow run](https://github.com/praxis-proxy/ai/actions/runs/43208165669)
provided the backend; the acceptance tests ran locally, not inside that workflow.
The workflow was cancelled after testing and its stop-runner job completed
successfully, releasing the ephemeral GPU instance.
Detailed JSON reports are local build artifacts at
`target/openresponses/managed-gpu.json` and
`target/openresponses/passthrough-gpu.json`. Both runs were complete. The tested
binary's SHA-256 was
`4d25349519ea4996da4847d5e0a9a5da28b269d149096c3845ee67547328e233`.

Full compliance remains unproven until the seven WebSocket templates pass
through Praxis against a capable backend, or Praxis gains its own managed
WebSocket Responses execution path. A future implementation must cover
connection-local `store:false` continuation, reconnect recovery, cache eviction
after failure, and compaction into a new chain. Merely accepting the upgrade
does not satisfy those cases.

## Run every template

Build Praxis with the experimental Responses filters and SQLite enabled:

```console
cargo build -p praxis-ai-proxy --no-default-features --features standard,openai-all,store-sqlite
```

With Bun 1.3.12 and uv installed, run against a live backend:

```console
export OPENRESPONSES_API_KEY=test
make test-responses-acceptance RESPONSES_ACCEPTANCE_ARGS='--upstream-url http://127.0.0.1:8000 --model your-vision-model --backend "vLLM version and model revision" --profile managed --evidence live --output target/openresponses/managed.json'
```

This launches an isolated local Praxis listener and SQLite database, uses the
versioned [managed profile](../../tests/conformance/openresponses/managed.yaml),
and stops the listener on completion. HTTPS backends use certificate verification,
TLS SNI and the matching upstream Host authority. The backend must serve the
standard `/v1` endpoints; arbitrary URL prefixes are not supported.

Use `--base-url http://127.0.0.1:8080/v1` instead of `--upstream-url` to test an
already deployed Praxis configuration. Credentials come only from the environment
and are redacted if echoed in results. Reports contain request and response
payloads, so review them before sharing.

The runner discovers upstream's exported inventory and executes every template
serially to avoid inference contention. There is no exclusion/filter option.
A failure, skip, missing result or malformed CLI result prevents success. JSON
is saved after each template; `complete: false` means the run was interrupted.
The report records the suite revision, Praxis HEAD/dirty state, backend
description, evidence type and, when launching locally, the binary SHA-256 and
effective configuration with the upstream authority redacted.

## Who owns each behavior

| Profile | HTTP Responses create | `/responses/compact` | WebSocket state |
| --- | --- | --- | --- |
| `managed` | Praxis translates to backend Chat Completions | Praxis summarizes through backend Chat Completions | Native backend, via passthrough |
| `passthrough` | Native backend | Native backend | Native backend |

`--profile passthrough` uses the [native profile](../../tests/conformance/openresponses/passthrough.yaml)
to distinguish backend failures from managed-filter behavior. It proves forwarding
compatibility, not that Praxis implements the backend's features. In particular,
the managed profile's local compaction output must also be accepted by the
native backend for the WebSocket compact/new-chain test to pass.

One upstream template, `response-output-phase-schema`, validates a local fixture
without making an endpoint request. Its result is marked `kind: schema_fixture`.
Passing it does not prove that Praxis generates assistant phase labels. Passing
the acceptance suite also does not establish compliance with every normative
requirement in the full specification.

## Controlled checks and real vision coverage

```console
uv run tests/integration/sdk/openai/test_responses_acceptance_runner.py
uv run tests/integration/sdk/openai/test_responses_acceptance_gateway.py -k 'not live_image'
```

The controlled backend returns fixed Chat Completions payloads. These checks
exercise image translation, compaction and missing-model rejection through the
real Praxis binary, upstream templates, and the OpenAI SDK. They establish
gateway behavior, not model inference compliance. Use `--evidence controlled`
for any full-suite run against a simulator or mock backend.

The existing Anthropic vLLM vision workflow additionally runs the upstream
OpenResponses image template through the managed profile. It reuses the pinned
vision image from `vllm/images.json`, requires live infrastructure, and does not
silently skip when the backend is unavailable. The complete suite remains a
separate explicit run, since the vision lane serves without a tool-call parser.

For interactive live investigation, `vllm-dev-endpoint.yaml` can provision an
ephemeral GPU backend. Select a vision-capable model, then use its private
session URL as `--upstream-url`. Do not commit the session URL or credentials.
Cancel the workflow after testing and verify its stop-runner job completes.
