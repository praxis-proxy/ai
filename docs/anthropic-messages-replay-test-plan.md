# Anthropic Messages Replay Test Plan

Status: draft

Owner: `franciscojavierarceo`

## What?

Define the Anthropic Messages API shapes Praxis should cover in replay,
passthrough, and translation tests. The goal is not to reimplement Anthropic's
full request validator. Praxis should validate only the envelope and fields it
owns locally, then preserve the request and response bodies as much as possible
when passing them through.

This test plan extends the broader Messages API filter proposal in
[00484_anthropic-messages-api-filters.md](proposals/00484_anthropic-messages-api-filters.md)
with concrete fixture and test coverage.

## Sources

The test plan is based on:

- Anthropic's
  [Messages API documentation](https://platform.claude.com/docs/en/build-with-claude/working-with-messages)
- Anthropic's tool-use documentation:
  [overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview),
  [defining tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools),
  [handling tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls),
  and the [tool reference](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference)
- Anthropic's
  [extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking)
  and
  [stop reason](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)
  documentation
- A local aggregate scan of Claude Code JSONL sessions under
  `~/.claude/projects`

The local scan was used only for shape discovery. It should not be copied into
fixtures verbatim unless the content is reviewed and sanitized first.

## Local Session Inventory

The local Claude Code sessions are dominated by client-tool traffic and are a
good source for realistic replay shapes:

- 114 JSONL session files and 10,300 records
- Message roles: 4,295 assistant records and 3,082 user records
- Content block types:
  - `tool_use`: 2,615
  - `tool_result`: 2,615
  - `text`: 1,491
  - string content: 434
  - `thinking`: 222
  - `image`: 3
- Image source types: `base64` only
- Stop reasons:
  - `tool_use`: 2,948
  - `end_turn`: 325
  - `stop_sequence`: 17
- Common local tool names include `Bash`, `Read`, `Edit`, `Agent`,
  `TaskUpdate`, `Write`, `TaskCreate`, `Skill`, `WebFetch`, and `WebSearch`.
- Tool result content appears as plain strings, `is_error` strings, lists of
  text blocks, and one list containing an image block.

The scan did not find server tool response shapes such as `server_tool_use`.
Those cases should be curated from public documentation or generated synthetic
fixtures.

## Validation Policy

Praxis should keep Anthropic Messages validation intentionally narrow:

- Validate that request bodies are valid JSON objects before downstream filters
  depend on them.
- Classify and promote only proxy-owned routing facts such as protocol format,
  model, streaming mode, and whether tools are present.
- Preserve unknown request and response fields.
- Leave Anthropic-owned semantics to the backend, including model capability
  checks, role ordering, parameter ranges, tool schema validity, and unsupported
  feature combinations.
- When transforming to OpenAI Chat Completions, test only the fields Praxis
  actually maps or intentionally drops with an observable warning.
- When importing replay fixtures, preserve the original `source_records`
  structure. Generated replay requests and responses may be added next to the
  source records, but source records must not be modified to make the test pass.

For response passthrough, the expected behavior is pure passthrough minus
Praxis-owned source, protocol, and transport metadata. Content blocks, generated
JSON, tool calls, thinking blocks, usage fields, stop reasons, and provider
metadata should not be rewritten on native Anthropic paths.

## Test Matrix

### Request Envelope and Classification

Cover these cases with unit or integration tests:

- Minimal Messages request with `model`, `max_tokens`, and one user message.
- String user content and array-based `text` content.
- Requests recognized by `/v1/messages` path or `anthropic-version` header even
  when body structure overlaps with OpenAI Chat Completions.
- `stream: true` classification and routing metadata.
- `tools` presence setting the internal tools metadata.
- Top-level `system` as a string and as text blocks.
- Mid-conversation `system` messages preserved when present.
- Last-assistant prefill content preserved when present.
- Malformed JSON or non-object JSON rejected by the validate filter.

### Content Blocks

Cover all content block shapes Praxis may see or transform:

- User and assistant `text` blocks.
- Image blocks with `source.type` values:
  - `base64`
  - `url`
  - `file`
- Image media types:
  - `image/jpeg`
  - `image/png`
  - `image/gif`
  - `image/webp`
- `thinking` and `redacted_thinking` blocks in source records and responses.
- Tool result nested content blocks:
  - `text`
  - `image`
  - `document`
  - `search_result`

The current local sessions only show base64 images, so URL, file, document, and
search-result fixtures should be curated.

### Tool Definitions

Cover tool definitions that Anthropic accepts and Praxis may transform:

- User-defined tool with `name`, `description`, and `input_schema`.
- Tool `name` values using the documented alphanumeric, underscore, and hyphen
  shape.
- Optional `input_examples` preserved on passthrough paths.
- Optional properties preserved or intentionally handled:
  - `cache_control`
  - `strict`
  - `defer_loading`
  - `allowed_callers`
- `tool_choice` variants:
  - `auto`
  - `any`
  - `tool`
  - `none`
- Parallel-tool controls such as `disable_parallel_tool_use` and their OpenAI
  translation equivalent when supported.
- Client-side tool names seen in local sessions, especially `Bash`, `Read`,
  `Edit`, and `WebFetch`.
- Anthropic-schema client tools from the public reference:
  - `memory`
  - `bash`
  - `text_editor`
  - `computer`
- Server tools from the public reference:
  - `web_search`
  - `web_fetch`
  - `code_execution`
  - `advisor`
  - `tool_search`
  - `mcp_toolset`

Server-tool fixtures should be synthetic unless we capture a real local session
that includes them.

### Tool Call Lifecycle

Cover complete tool loops, not only isolated content blocks:

- Assistant message with a single `tool_use` block and `stop_reason:
  "tool_use"`.
- Assistant text followed by `tool_use`.
- Assistant message with multiple parallel `tool_use` blocks.
- User follow-up containing `tool_result` blocks immediately after the
  matching assistant tool request.
- `tool_result.content` as:
  - string
  - list of text blocks
  - list containing an image block
  - empty or omitted content
- `tool_result.is_error: true`.
- Mixed server and client tool blocks, such as `server_tool_use` plus
  client-side `tool_use`.
- Pending server-tool continuations where the follow-up user message contains
  only `tool_result` blocks.
- Programmatic tool-calling metadata such as `caller` and `container` when a
  fixture is available.

Ordering requirements should be documented, but Praxis should enforce them only
where a transformation would otherwise produce an invalid backend request.

### Response and Stop Reasons

Cover each documented stop reason with either a passthrough fixture or a
translation unit test:

- `end_turn`
- `max_tokens`
- `stop_sequence`
- `tool_use`
- `pause_turn`
- `refusal`
- `model_context_window_exceeded`

The refusal case should include `stop_details`. Tool-use responses should
include usage fields and at least one content block with `type: "tool_use"`.

### Streaming

Streaming replay is deferred until the replay schema can represent SSE
responses. Unit and integration tests for `anthropic_stream_events` should still
cover:

- Text `content_block_delta` events.
- Tool call input deltas.
- Usage deltas.
- Stop events and final message events.
- Partial UTF-8 across chunks.
- Mixed text and tool-use content in one response.
- Finish-reason mapping for tool use and max tokens.

## Fixture Plan

Use sanitized local Claude Code sessions for:

- A basic text-only Messages exchange.
- A base64 image request.
- A client-tool cycle with `Bash`, `Read`, or `Edit`.
- A client-tool cycle with `is_error: true`.
- A `tool_result` with list-based text content.
- The existing list-based image tool result case.
- A response containing `thinking` followed by final text or tool use.

Use curated fixtures for:

- URL and file image sources.
- GIF and WebP media types if not captured locally.
- Document and search-result tool result blocks.
- Server tools and `server_tool_use`.
- `pause_turn`, `refusal`, and `model_context_window_exceeded`.
- SSE replay once the fixture schema supports it.

Every fixture should include enough assertions to prove the behavior that made
it worth adding. A fixture that only returns HTTP 200 is insufficient unless the
shape itself is the regression target.

## Priority

### P0

- Add a Claude Code tool-use replay fixture that preserves source records and
  verifies the generated request/response remains replayable.
- Add Anthropic-to-OpenAI translation coverage for `tool_use` and
  `tool_result`, including backend-visible OpenAI tool calls and tool-role
  messages.
- Add a thinking fixture that proves source records preserve thinking content
  even when translation drops unsupported thinking blocks for OpenAI backends.
- Add a tool-result error fixture.
- Add a list-content tool-result fixture.

### P1

- Add URL and file image source coverage.
- Add stop-reason fixtures for `max_tokens`, `stop_sequence`, and `refusal`.
- Add tool-choice translation tests for `auto`, `any`, `tool`, and `none`.
- Add optional tool-definition property passthrough tests.

### P2

- Add curated server-tool fixtures.
- Add `pause_turn` and `model_context_window_exceeded` fixtures.
- Extend replay fixtures to represent streaming SSE responses.
- Add programmatic tool-calling metadata coverage if we capture or synthesize a
  representative request.

## Open Questions

- Should replay fixtures grow a first-class SSE response format, or should SSE
  stay covered only by stream parser tests for now?
- Should `anthropic_validate` remain envelope-only, or should Praxis validate a
  small set of transformation-owned invariants before converting to OpenAI Chat
  Completions?
- How much should committed fixtures reflect Claude Code-specific tool names
  versus public Anthropic tool names?
- What sanitization workflow should we use for large local tool results so the
  original source record structure is preserved without leaking private local
  content?
