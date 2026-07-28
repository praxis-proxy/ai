---
issue: https://github.com/praxis-proxy/praxis/issues/211
discussion: https://github.com/praxis-proxy/praxis/issues/20
status: superseded
authors:
  - szedan-rh
graduation_criteria:
  - Token counts available for streaming responses across all five providers
  - Token counts available for non-streaming responses
  - Downstream filters can consume counts without provider-specific logic
  - Response bodies pass through unmodified
stakeholders:
  - shaneutt
  - twghu
---

# Streaming Token Counting — Implementation Design

> **Superseded.** The design described here was implemented and then
> consolidated into the private `filters/src/token_usage/` subsystem.
> The public `TokenUsageProvider` enum and `extract_streaming_tokens`
> function no longer exist; streaming parsing is handled by private
> functions in `filters/src/token_usage/streaming.rs`. See proposal
> [#210] for the current design.

## Overview

A `token_count` filter that extracts token usage from AI
inference responses — both streaming (SSE) and non-streaming
(JSON) — and writes unified counts to `filter_metadata` for
downstream consumers. The filter is transparent: response
bodies and status codes pass through unchanged.

### Module Location

The filter lives at `filter/src/builtins/http/ai/token_count/`
with the following structure:

```text
token_count/
├── mod.rs       # TokenCountFilter, HttpFilter impl, from_config
├── streaming.rs # Per-provider SSE event extraction
└── tests.rs     # Unit tests
```

Registered as `"token_count"` in
`filter/src/registry.rs` via:

```rust
register_http(factories, "token_count", TokenCountFilter::from_config);
```

### Configuration

```rust
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenCountConfig {
    provider: TokenUsageProvider,
}
```

Single required field — the AI provider determines which
JSON structure to parse. `TokenUsageProvider` already
implements `Deserialize` via its existing enum definition
([#216]). No other knobs needed; body mode and access are
fixed at filter construction.

```yaml
- filter: token_count
  provider: openai    # openai | anthropic | google | bedrock | azure
```

### Body Access

The filter declares read-only streaming access to response
bodies. It never modifies response bytes or status codes.

```rust
fn response_body_access(&self) -> BodyAccess { BodyAccess::ReadOnly }
fn response_body_mode(&self)   -> BodyMode   { BodyMode::Stream }
```

`BodyMode::Stream` is used for both streaming and
non-streaming responses. The filter accumulates
non-streaming body chunks internally (via `filter_metadata`
hex encoding, following the A2A filter pattern) rather than
requesting `StreamBuffer`, because the body mode is
declared per-filter, not per-request. `StreamBuffer` would
force buffering of streaming responses too, breaking TTFT
latency for the dominant use case.

### Detection Logic (`on_response`)

The filter inspects the response `content-type` header to
decide which extraction path to use. This mirrors the
pattern in `A2aFilter::on_response`.

```text
on_response:
  1. Skip non-success responses (status < 200 or >= 300)
  2. Read content-type header
  3. If text/event-stream → set metadata flag "token_count.mode" = "sse"
  4. If application/json  → set metadata flag "token_count.mode" = "json"
  5. Otherwise            → no-op (filter skips body phase)
```

The mode flag in `filter_metadata` tells `on_response_body`
which path to run. Without a flag, the body hook returns
`Continue` immediately — zero overhead for non-AI traffic
that passes through the filter due to pipeline placement.

### Non-Streaming Path (JSON)

For `application/json` responses, body chunks are
accumulated in `filter_metadata` using hex encoding
(the same `accumulate_response_hex` / `decode_hex` pattern
used by the A2A filter). A configurable byte limit
(default 1 MiB) caps memory growth.

```text
on_response_body (mode = "json"):
  1. Append chunk bytes to hex buffer in filter_metadata
  2. If accumulated bytes exceed limit → clear state, return
  3. On end_of_stream:
     a. Decode hex buffer back to bytes
     b. Call extract_token_usage(provider, &bytes)
     c. If Some(usage) → ctx.set_token_usage(...)
     d. Clear hex buffer metadata
```

`extract_token_usage` ([#216]) handles all five providers.
The JSON structure is identical whether the response was
non-streaming or a buffered payload — no additional
parsing logic needed.

### Streaming Path (SSE)

For `text/event-stream` responses, body chunks are fed
through the existing SSE scanner
(`builtins::http::ai::agentic::a2a::sse`). The scanner is
already `pub(crate)` and protocol-agnostic — it extracts
`data:` payloads from SSE frames regardless of what the
payloads contain.

Scanner state (`SseScanState`) is persisted between
`on_response_body` calls via hex-encoded `filter_metadata`
keys, following the same save/load pattern as the A2A
filter.

For each completed SSE `data:` payload, the filter
attempts token extraction:

```text
on_response_body (mode = "sse"):
  1. Load SseScanState from filter_metadata
  2. Call scan_sse_chunk(&mut state, chunk, max_scratch)
  3. For each completed payload:
     a. Try extract_token_usage(provider, &payload)
     b. If Some(usage) → store as latest in filter_metadata
     c. If None → try extract_streaming_tokens(provider, &payload)
     d. If partial data found → merge into accumulated counts
  4. Save state back to filter_metadata
  5. On end_of_stream or overflow → write final counts via
     ctx.set_token_usage(...) and clear state
```

Step 3a handles providers that include complete usage data
in a single SSE event (OpenAI, Google, Azure). The JSON
structure inside the event matches the non-streaming format,
so `extract_token_usage` works directly.

Step 3c handles providers that spread token counts across
multiple events (Anthropic, Bedrock). The
`extract_streaming_tokens` function in `streaming.rs`
performs lightweight provider-specific extraction:

#### Provider-Specific Streaming Behavior

| Provider  | Input tokens | Output tokens | Strategy |
|-----------|-------------|---------------|----------|
| OpenAI    | Final `usage` event | Final `usage` event | `extract_token_usage` on each payload; last wins |
| Azure     | Same as OpenAI | Same as OpenAI | Same as OpenAI |
| Anthropic | `message_start` event (`message.usage.input_tokens`) | Final `message_delta` event (`usage.output_tokens`) | Accumulate across events |
| Google    | Final chunk (`usageMetadata`) | Final chunk (`usageMetadata`) | `extract_token_usage` on each payload; last wins |
| Bedrock   | Converse stream (`contentBlockDelta` metadata) | Converse stream | Accumulate across events |

#### `extract_streaming_tokens`

A small function in `streaming.rs` that extracts partial
token data from a single SSE event when
`extract_token_usage` returns `None`. It returns
`Option<(Option<u64>, Option<u64>)>` — partial input and
output counts that the caller merges into accumulated
state.

```rust
pub(super) fn extract_streaming_tokens(
    provider: TokenUsageProvider,
    event_data: &[u8],
) -> Option<(Option<u64>, Option<u64>)>
```

For Anthropic, this handles two event shapes:

- `message_start`: parses `message.usage.input_tokens`
  (nested under `message`, not at root — which is why
  `extract_token_usage` misses it)
- `message_delta`: parses `usage.output_tokens` at root

For Bedrock streaming (Converse ConverseStream), this
handles the metadata events that carry token counts
in `contentBlockDelta` structures.

OpenAI, Azure, and Google do not need this fallback —
their streaming events either contain complete usage
(parsed by `extract_token_usage`) or no usage at all.

### Accumulated State

Per-request token state is carried in `filter_metadata`
using the `token_count.` key prefix:

| Key | Purpose |
|-----|---------|
| `token_count.mode` | `"sse"` or `"json"` — set in `on_response` |
| `token_count.input` | Accumulated input token count (streaming) |
| `token_count.output` | Accumulated output token count (streaming) |
| `token_count.buf_hex` | Hex-encoded JSON body buffer (non-streaming) |
| `token_count.buf_bytes` | Byte count of buffered body (non-streaming) |
| `token_count.sse_*` | SSE scanner state (line_buf, data_buf, etc.) |

On `end_of_stream`, the filter calls
`ctx.set_token_usage(input, output, None)` to write the
final counts to the well-known `token.input`,
`token.output`, `token.total` keys ([#212]). It then
clears all `token_count.*` working keys.

### SSE Scanner Reuse

The SSE scanner in `agentic::a2a::sse` is already
`pub(crate)` and its implementation is protocol-agnostic.
It can be imported directly from within the `praxis-filter`
crate without changes.

The scanner's hex state persistence helpers
(`load_sse_scan_state`, `save_sse_scan_state`,
`set_hex_metadata`, `decode_hex`) are private to the A2A
module. Rather than duplicating them, promote the hex
encoding/decoding utilities to a shared location:

```text
filter/src/builtins/http/ai/hex.rs
  pub(crate) fn encode_hex(data: &[u8]) -> String
  pub(crate) fn decode_hex(hex: &str) -> Option<Vec<u8>>
  pub(crate) fn set_hex_metadata(ctx, key, data)
  pub(crate) fn accumulate_hex(ctx, key, chunk, max) -> bool
```

Both A2A and token_count import from `hex.rs`. The A2A
module's existing private functions become thin wrappers
or direct imports.

### Error Handling

The filter follows the transparent-proxy principle:
parse failures are silently ignored, never producing
errors or rejections.

- Malformed JSON → `extract_token_usage` returns `None` →
  no token counts set, response passes through unchanged
- SSE parse overflow → stop scanning, write whatever counts
  were accumulated, clear state
- Missing `usage` fields → `None` → no-op
- Non-AI traffic hitting the filter → no mode flag set →
  body hook returns `Continue` immediately

`tracing::debug!` logs are emitted for diagnostic
visibility without affecting the data path.

### Pipeline Placement

The filter runs in the response phase. It should be placed
after the router and before any filter that consumes token
counts (rate limiting, access logging, header injection):

```yaml
filter_chains:
  - name: ai-pipeline
    filters:
      - filter: router
        # ...
      - filter: load_balancer
        # ...
      - filter: token_count         # extracts counts
        provider: openai
      - filter: token_usage_headers  # reads counts (#214)
      - filter: access_log           # logs counts
```

### Example Config

```yaml
# Token Counting
#
# Extracts token usage from AI inference responses
# (streaming and non-streaming) and makes counts
# available to downstream filters via filter metadata.
#
# Usage:
#   cargo run -p praxis -- -c examples/configs/ai/token-counting.yaml

listeners:
  - name: ai-gateway
    address: "0.0.0.0:8080"
    protocol: http
    filter_chains:
      - name: ai-pipeline
        filters:
          - filter: token_count
            provider: openai

          - filter: router
            routes:
              - path_prefix: "/v1/"
                cluster: openai-backend

    clusters:
      - name: openai-backend
        endpoints:
          - address: "api.openai.com:443"
            tls: true
```

### Test Plan

**Unit tests** (`token_count/tests.rs`):

1. Config parsing — valid and invalid provider values
2. Non-streaming extraction — full JSON responses for all
   five providers
3. Streaming extraction — SSE event sequences for all
   five providers, including:
   - Single event with complete usage (OpenAI, Google)
   - Split events requiring accumulation (Anthropic)
   - Chunk boundaries splitting SSE frames
   - Events without usage data (skipped)
   - `data: [DONE]` sentinel handling
4. Content-type detection — `text/event-stream`,
   `application/json`, with and without charset params
5. Error cases — malformed JSON, missing usage fields,
   error responses, empty bodies
6. Byte limit enforcement — buffer overflow for
   non-streaming, scratch overflow for streaming

**Integration tests**
(`tests/integration/tests/suite/examples/`):

End-to-end test with a mock backend that returns both
streaming SSE and non-streaming JSON responses. Verify
that `token.input`, `token.output`, `token.total` metadata
keys are populated and readable by a downstream filter.

**Example config validation**
(`tests/schema/`):

Parse the example config and verify it passes schema
validation.

[#212]: https://github.com/praxis-proxy/praxis/issues/212
[#214]: https://github.com/praxis-proxy/praxis/issues/214
[#216]: https://github.com/praxis-proxy/praxis/issues/216
