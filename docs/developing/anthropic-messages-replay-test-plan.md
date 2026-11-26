# Anthropic Messages Replay Test Plan

Status: draft

Owner: `franciscojavierarceo`

## What?

The Anthropic Messages API shapes Praxis should cover in replay, passthrough,
and translation tests. Praxis validates only the envelope and the fields it
owns, and otherwise preserves request and response bodies; it does not
reimplement Anthropic's request validator.

This plan adds fixture and test coverage to the Messages API filter proposal,
[00484_anthropic-messages-api-filters.md](https://github.com/praxis-proxy/enhancements/blob/main/proposals/00484_anthropic-messages-api-filters.md).

## Sources

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

The local scan was for shape discovery only. Do not copy it into fixtures
unless the content has been reviewed and sanitized.

## Local Session Inventory

The local Claude Code sessions are mostly client-tool traffic, which makes
them a good source of realistic replay shapes:

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

The scan found no server tool response shapes such as `server_tool_use`, so
those fixtures come from public documentation or are synthetic.

## Validation Policy

Praxis keeps Anthropic Messages validation narrow:

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

Native Anthropic response paths are pure passthrough apart from Praxis-owned
source, protocol, and transport metadata. Content blocks, generated JSON, tool
calls, thinking blocks, usage fields, stop reasons, and provider metadata are
not rewritten.

## Test Matrix

### Request Envelope and Classification

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

### Tool Definitions

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

### Tool Call Lifecycle

Cover complete tool loops, not just isolated content blocks:

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

Document ordering requirements, but enforce them only where a transformation
would otherwise produce an invalid backend request.

### Response and Stop Reasons

Cover each documented stop reason with a passthrough fixture or a
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

Streaming replay waits until the replay schema can represent SSE responses.
Unit and integration tests for `anthropic_messages_to_chat_completions_stream`
still cover:

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

Each fixture needs assertions for the behavior it was added to prove. A
fixture that only checks for HTTP 200 is not enough unless the shape itself
is the regression target.

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
- Should `anthropic_messages_request` remain envelope-only, or should Praxis
  validate a small set of transformation-owned invariants before converting to
  OpenAI Chat Completions?
- How much should committed fixtures reflect Claude Code-specific tool names
  versus public Anthropic tool names?
- What sanitization workflow should we use for large local tool results so the
  original source record structure is preserved without leaking private local
  content?
