<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 Praxis Contributors
-->

# Stored-session replay fixtures

These fixtures are sanitized examples shaped like stored agent sessions.
Each fixture includes one or more ordered turns with the request sent by
an agent client and the response returned by the mocked upstream model
service.

The samples are intentionally small. They exercise the Praxis example
configuration paths for Anthropic Messages and OpenAI Responses while
leaving room for future import tooling that can normalize real Claude
or Codex session logs into the same fixture schema.

Replay fixtures are useful when a type is too broad for hand-written smoke
tests alone. A fixture should preserve the JSON shape that a real client sent,
then let the integration suite replay that traffic through a Praxis example
configuration.

## Current fixtures

- `claude/messages-basic.json` replays one Anthropic Messages turn through
  `examples/configs/anthropic/messages-protocol.yaml`.
- `claude/messages-image.json` replays an Anthropic Messages image block
  through native protocol forwarding and Chat Completions translation.
- `claude/messages-tool-cycle.json` replays a sanitized Claude Code client-tool
  cycle with preserved source records for the assistant `tool_use` and user
  `tool_result` records.
- `claude/messages-thinking.json` replays a sanitized Claude Code NPV prompt
  with a real `thinking` source record, preserving it in `source_records` while
  replaying only the visible Anthropic response text.
- `codex/responses-basic.json` replays one OpenAI Responses turn through
  `examples/configs/openai/responses/full-flow.yaml` and verifies the response
  can be read back from the response store.

## Fixture schema

Fixtures are loaded by `tests/utils/src/session_replay.rs`. The loader rejects
unknown fields, so every top-level and turn-level field must be documented here
before it is used.

```json
{
  "source": "Human-readable origin of the sanitized sample",
  "protocol": "anthropic_messages",
  "turns": [
    {
      "name": "stable-fixture-local-turn-name",
      "path": "/v1/messages",
      "request": {},
      "response": {},
      "source_records": []
    }
  ]
}
```

Top-level fields:

- `source`: describes the origin and sanitization level of the sample, such as
  `Sanitized Claude Code session log`. Keep scenario names in the turn `name`
  and fixture filename, not in `source`.
- `protocol`: currently `anthropic_messages` or `openai_responses`.
- `turns`: ordered request/response pairs from the session.

Turn fields:

- `name`: stable name for assertions and failure output.
- `path`: HTTP path used by the client, such as `/v1/messages` or
  `/v1/responses`.
- `request`: JSON request body sent by the agent client.
- `response`: JSON response body returned by the mocked upstream service.
- `source_records`: optional original session records used to derive the turn.
  Importers should preserve records here when the replayable `request` and
  `response` fields are projections of a richer session log.
  For OpenAI Responses/Codex sessions, `request` and `response` should be
  direct copies of the original `response_item.payload.request` and
  `response_item.payload.response` values, not normalized or filtered shapes.

## How to add a replay example

1. Pick one behavior to harden.

   Good fixtures are small and realistic. Prefer a focused example for a tool
   call, a multimodal message, a structured output, or a response-store edge
   case over one huge transcript that is hard to debug.

2. Save a sanitized JSON fixture under the client or provider directory.

   Use `claude/` for Claude-shaped Messages traffic and `codex/` for
   Codex-shaped Responses traffic. Add another directory when the client shape
   matters enough to name separately.

3. Preserve the real request and response shape.

   Remove secrets, hostnames, account identifiers, user text that should not be
   committed, local file paths, and unstable timestamps. Replace real IDs with
   deterministic fixture IDs like `msg_replay_tool_call` or
   `resp_replay_structured_output`. When redacting linked IDs such as `uuid`,
   `parentUuid`, `promptId`, or `sessionId`, preserve equality and parent-child
   relationships instead of inventing new links.

4. Add or extend an integration test in
   `tests/integration/tests/suite/examples/session_replay.rs`.

   Reuse `SessionReplay::load(...)`, start the mocked backend with the fixture
   response, send the fixture request through the matching example config, and
   assert the client-visible response. Add protocol-specific assertions when
   Praxis behavior should change, such as response-store retrieval.

5. Extend `tests/utils/src/session_replay.rs` only when the schema needs a new
   documented concept.

   For a new protocol, add a `ReplayProtocol` variant, loader tests, and at
   least one functional integration test. Keep raw provider payloads in
   `request` and `response` as `serde_json::Value` so the fixture can cover the
   wide API surface without changing Rust structs for every provider field.

6. Run the focused replay checks.

   ```console
   cargo test -p praxis-test-utils session_replay
   cargo test -p praxis-tests-integration session_replay -- --nocapture
   ```

## Importing local sessions

Use `cargo xtask make-replay-fixture` to convert a local Claude Code or Codex
session log into this fixture schema:

```console
cargo xtask make-replay-fixture ~/.codex/sessions/2026/07/07/session.jsonl \
  --provider auto \
  --out tests/integration/fixtures/replay/codex/my-example.json
```

When `--out` is omitted, the generated fixture is printed to stdout. Provider
selection defaults to `--provider auto`; use `--provider codex` or
`--provider claude` when a file is recognizable but auto-detection is not
specific enough.

Typical local session locations:

- Codex: `~/.codex/sessions/YYYY/MM/DD/*.jsonl`
- Claude Code project sessions:
  `~/.claude/projects/<project-slug>/<session-id>.jsonl`
- Claude Code subagent sessions:
  `~/.claude/projects/<project-slug>/<session-id>/subagents/*.jsonl`
- Claude Code prompt history: `~/.claude/history.jsonl`

The importer is intentionally conservative. Codex import currently expects
JSONL records with `response_item.payload.request` and
`response_item.payload.response`. Claude Code import pairs a user message with
the following assistant Messages response and uses a deterministic fallback
`max_tokens` value in the generated request. Always review and sanitize the
generated fixture before committing it.

## Coverage targets

Add fixtures when a real stored Claude or Codex session shows a shape the
gateway should keep handling. Useful next examples include:

- Anthropic Messages with `tools`, `tool_use`, and `tool_result` content.
- OpenAI Responses with function tools, tool calls, and tool outputs.
- Mixed content arrays with text plus non-text blocks.
- Additional Anthropic reasoning variants, such as `redacted_thinking` blocks
  or thinking signatures mixed with tool calls.
- Structured output requests with nested JSON schemas.
- Responses using `previous_response_id` or stored-response retrieval.
- Requests with `store: false`, null fields, booleans, arrays, and numeric
  usage fields.
- Refusal, incomplete, or error-shaped provider responses.
- Streaming fixtures, once the replay schema grows an SSE response form.

Each fixture should assert at least one behavior beyond "the request returned
200" when the example is meant to exercise a Praxis feature.

## Fixture checklist

Before committing a new replay example:

- The fixture is valid JSON.
- The payload contains no credentials, account IDs, local paths, or private
  user content.
- IDs and timestamps are deterministic.
- The `source` explains where the shape came from without exposing sensitive
  details.
- The `path` matches the protocol.
- The integration test uses an example config under `examples/configs/`.
- The assertions cover the Praxis behavior that made the fixture worth adding.
