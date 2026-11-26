# Full OpenResponses acceptance

Full acceptance runs every template in the pinned
[upstream acceptance suite](https://www.openresponses.org/compliance)
inventory, including image input, compaction, and WebSockets.
[Translation coverage](openresponses-translation.md) answers a narrower
question: it runs seven selected templates against a text-only CPU backend.
The CPU lane uses upstream's exported runner with an explicit 300-second
request budget, because the small model can exceed the suite's 30-second
default. Templates and validators are unchanged. The full acceptance CLI keeps
upstream's 30-second HTTP request / WebSocket turn deadline, under a separate
360-second subprocess cap described below.

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
schema fixture: actual image inference (the model identified the supplied red
heart), tool declaration acceptance with a translated `function_call` output,
SSE streaming, compaction, and missing-model rejection. The managed profile
allows one inference round (`max_iterations: 1`), so the tool case neither
executes the function nor tests a `function_call_output` continuation. A
separate real-image check also passed through the official OpenAI Python SDK
3.49.0.

All seven managed-profile failures were WebSocket connection failures. An
upgrade request sent directly to the backend returned HTTP 405, and the
[vLLM 0.32.0 Responses router](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/entrypoints/openai/responses/api_router.py)
has no WebSocket handler. That is not a Praxis transport failure, but this
deployment still cannot claim full suite compliance. The native profile also
failed Responses-resource schema checks (including missing `store` and
incompatible or null fields) and returned HTTP 405 for compaction. Its only
pass was the local schema fixture.

The [GPU dev workflow run](https://github.com/praxis-proxy/ai/actions/runs/43208165669)
provided the backend; the acceptance tests ran locally, not inside that
workflow. The workflow was cancelled after testing, and its stop-runner job
completed and released the ephemeral GPU instance. Both runs were complete;
the JSON reports are local build artifacts at
`target/openresponses/managed-gpu.json` and
`target/openresponses/passthrough-gpu.json`. The tested binary's SHA-256 was
`4d25349519ea4996da4847d5e0a9a5da28b269d149096c3845ee67547328e233`.

Full compliance stays unproven until the seven WebSocket templates pass
through Praxis against a capable backend, or Praxis gains its own managed
WebSocket Responses execution path. Such a path must cover connection-local
`store:false` continuation, reconnect recovery, cache eviction after failure,
and compaction into a new chain; accepting the upgrade alone does not satisfy
those cases.

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
and stops the listener when done. HTTPS backends get certificate verification,
TLS SNI and the matching upstream Host authority. The backend must serve the
standard `/v1` endpoints; arbitrary URL prefixes are not supported.

To test an already deployed Praxis configuration, pass
`--base-url http://127.0.0.1:8080/v1` instead of `--upstream-url`. Credentials
come only from the environment and are redacted if echoed in results. Reports
contain request and response payloads, so review them before sharing.

The runner discovers upstream's exported inventory and runs every template
serially to avoid inference contention; there is no exclusion or filter
option. A failure, skip, missing result or malformed CLI result prevents
success. JSON is saved after each template, and `complete: false` means the
run was interrupted. The report records the suite revision, Praxis HEAD and
dirty state, backend description, evidence type and, when launching locally,
the binary SHA-256 and effective configuration with the upstream authority
redacted.

The Python harness caps each template's CLI subprocess at 360 seconds,
covering startup and all HTTP requests or WebSocket turns in that template.
This wall-clock guard leaves upstream's per-request and per-turn deadline
unchanged. If it expires, the JSON ledger records a failed result whose
`errors` holds Python's subprocess timeout message: a harness process
timeout, not an upstream schema-validation verdict.

## Who owns each behavior

| Profile | HTTP Responses create | `/responses/compact` | WebSocket state |
| --- | --- | --- | --- |
| `managed` | Praxis translates to backend Chat Completions | Praxis summarizes through backend Chat Completions | Native backend, via passthrough |
| `passthrough` | Native backend | Native backend | Native backend |

`--profile passthrough` uses the [native profile](../../tests/conformance/openresponses/passthrough.yaml)
to separate backend failures from managed-filter behavior. It proves
forwarding compatibility, not that Praxis implements the backend's features.
For the WebSocket compact/new-chain test to pass, the native backend must also
accept the managed profile's local compaction output.

One upstream template, `response-output-phase-schema`, validates a local
fixture without making an endpoint request; its result is marked
`kind: schema_fixture`. Passing it does not prove that Praxis generates
assistant phase labels. Passing the acceptance suite also does not establish
compliance with every normative requirement in the full specification.

## Controlled checks and real vision coverage

```console
uv run tests/integration/sdk/openai/test_responses_acceptance_runner.py
uv run tests/integration/sdk/openai/test_responses_acceptance_gateway.py -k 'not live_image'
```

The controlled backend returns fixed Chat Completions payloads. These checks
exercise image translation, compaction and missing-model rejection through
the real Praxis binary, upstream templates, and the OpenAI SDK. They establish
gateway behavior, not model inference compliance. Use `--evidence controlled`
for any full-suite run against a simulator or mock backend.

The Anthropic vLLM vision workflow also runs the upstream OpenResponses image
template through the managed profile. It reuses the pinned vision image from
`vllm/images.json`, needs live infrastructure, and does not silently skip
when the backend is unavailable. The complete suite stays a separate explicit
run, because the vision lane serves without a tool-call parser.

For interactive live investigation, `vllm-dev-endpoint.yaml` can provision an
ephemeral GPU backend. Pick a vision-capable model and pass its private
session URL as `--upstream-url`. Do not commit the session URL or credentials.
Cancel the workflow after testing and check that its stop-runner job
completes.
