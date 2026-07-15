---
issue: https://github.com/praxis-proxy/ai/issues/119
discussion: https://github.com/praxis-proxy/ai/issues/119
status: proposed
authors:
  - hschwart
graduation_criteria:
  - Token counts available for streaming SSE responses across OpenAI, Anthropic, and Vertex AI; Bedrock InvokeModel via response headers
  - Final aggregated counts written to filter metadata on stream completion
  - Response bodies pass through unmodified
  - Streaming SSE end-to-end integration test coverage
stakeholders:
  - shaneutt
  - twghu
---

# Streaming Token Accumulation

## What?

Incremental token count accumulation from streaming SSE responses.
As the proxy forwards provider event streams to clients, it parses
`data:` payloads, extracts per-chunk token usage fields, and
accumulates them with provider-correct semantics until the stream
completes. Final aggregated counts are written to filter metadata
for downstream consumers (access logging, cost tracking, rate
limiting, header injection).

The filter is transparent: response bodies and status codes pass
through unchanged. Only metadata is produced.

### Goals

- Parse SSE `data:` events for incremental token usage fields
- Accumulate across chunks with correct override semantics (some
  providers send cumulative totals, others send deltas or split
  input/output across separate events)
- Emit final aggregated counts on stream completion (`[DONE]`
  sentinel or stream close)
- Handle provider-specific streaming formats: OpenAI, Anthropic,
  Bedrock, and Vertex AI (Google)
- Reuse the unified `TokenUsage` representation from the provider
  mapping library ([#87]) where possible
- Write final counts to filter metadata ([#71]) via
  `set_token_usage()`

### Non-Goals

- Non-streaming token extraction ([#83]) — complementary work;
  this proposal focuses on the streaming path
- Client-side token estimation (tiktoken-style pre-request counting)
- Token-based rate limiting logic (reads from metadata; separate concern)
- Injecting token headers into downstream responses ([#86])
- Provider response translation or normalization beyond token fields
- Buffering the entire stream before forwarding (must preserve
  streaming latency / TTFT)

### Relationship to [#84]

This proposal tracks [#119] in the `praxis-proxy/ai` repository.
Issue [#84] covers streaming token counting via SSE event parsing
within the Token Counting epic ([#71]). Proposal
[`00211_streaming-token-counting.md`](./00211_streaming-token-counting.md)
was an earlier design draft from the `praxis-proxy/praxis` tracker;
the **How?** section below documents the implementation adopted in
this repository.

## Why?

### Motivation

Most AI inference in production uses streaming responses. Without
streaming accumulation, token counts are only available for
non-streaming requests. That leaves rate limiting, billing, access
logging, and observability incomplete for the majority of real
traffic.

Providers report token usage inside SSE events, but they do not
agree on *when* or *how* counts appear:

- **OpenAI / Azure**: usage typically arrives in a single event
  near the end of the stream, just before `[DONE]`
- **Anthropic**: `input_tokens` in `message_start` at the
  beginning; `output_tokens` in `message_delta` near the end
- **Google (Vertex / Gemini)**: `usageMetadata` on the final
  chunk; no `[DONE]` sentinel — stream close is the signal
- **Bedrock**: Converse stream metadata events carry counts in
  provider-specific structures

A single "parse the last chunk" strategy is insufficient.
Streaming accumulation must track partial counts across the
request lifecycle and resolve a final total only when the stream
ends.

This work is the streaming complement to response-based token
counting ([#83]) and the shared filter metadata contract
([#71]). Together they let any downstream filter consume token
counts without provider-specific SSE logic.

### User Stories

- As a **rate limiting filter**, I need accurate token counts
  after streaming responses complete so that I can enforce
  per-user token budgets for the dominant production traffic
  pattern.

- As a **cost tracking system**, I need final input and output
  token counts per streaming request so that spend attribution
  is complete without re-implementing provider SSE formats.

- As a **logging filter**, I need token usage in filter metadata
  at stream completion so that structured usage records cover
  streaming and non-streaming traffic equally.

- As a **platform operator**, I need one accumulation model per
  provider so that adding streaming support for a new backend
  does not require updating every downstream consumer.

- As a **filter author** implementing header injection ([#86]),
  I need streaming requests to produce the same `token.input`,
  `token.output`, and `token.total` metadata keys as
  non-streaming requests once the stream finishes.

## How?

### Implementation

- [#262](https://github.com/praxis-proxy/ai/pull/262) — `token_count` filter with streaming and non-streaming paths
- [#261](https://github.com/praxis-proxy/ai/pull/261) — `extract_streaming_tokens` for Anthropic partial events and Bedrock metadata JSON
- [#240](https://github.com/praxis-proxy/ai/pull/240) — Bedrock `InvokeModel` header-based extraction
- [#321](https://github.com/praxis-proxy/ai/pull/321) — provider parser unit tests

### Requirements

1. A `token_count` filter at `filters/src/token_count/` registered as
   `"token_count"` in the server filter registry.
2. Single required YAML key `provider` selecting the extraction strategy.
3. For SSE responses (`text/event-stream`), parse `data:` payloads
   incrementally without buffering the full stream (`BodyMode::Stream`).
4. Accumulate token counts across SSE events with provider-correct
   merge semantics and emit final counts on `end_of_stream`.
5. Delegate complete-usage parsing to `extract_token_usage` ([#87])
   and partial-event parsing to `extract_streaming_tokens`.
6. Write final counts via `set_token_usage()` to `token.input`,
   `token.output`, and `token.total` ([#71]).
7. Response bodies and status codes pass through unchanged.

### Design

#### Module layout

```text
filters/src/token_count/
├── mod.rs       # TokenCountFilter, SSE/JSON paths, metadata state
└── tests.rs     # Unit tests (streaming + non-streaming)

apis/src/token_usage/
├── mod.rs       # TokenUsage, extract_token_usage, set_token_usage
├── providers.rs # Non-streaming JSON parsers per provider
└── streaming.rs # extract_streaming_tokens (Anthropic partial events; Bedrock metadata JSON)
```

The filter reuses the A2A SSE scanner at
`filters/src/agentic/a2a/sse.rs` for frame reassembly.

#### Configuration

```yaml
- filter: token_count
  provider: openai   # openai | anthropic | google | bedrock | bedrock_invoke_model | azure
```

`TokenUsageProvider` implements `Deserialize` for YAML parsing ([#261]).

#### HttpFilter hooks (streaming path)

| Hook | Behaviour |
|------|-----------|
| `response_body_access` | `BodyAccess::ReadOnly` |
| `response_body_mode` | `BodyMode::Stream` (preserves TTFT) |
| `on_response` | On success + `text/event-stream`, set `token_count.mode = sse` |
| `on_response_body` | Feed chunks through SSE scanner; extract per payload; finalize on `end_of_stream` |

Non-success responses are skipped. No mode flag → body hook is a no-op.

#### Streaming extraction flow

```text
on_response_body (mode = sse):
  1. Load SseScanState from filter_metadata (hex-encoded)
  2. scan_sse_chunk(chunk) → completed data: payloads
  3. For each payload:
     a. Skip data: [DONE]
     b. Try extract_token_usage(provider, payload) → complete usage (last-wins)
     c. Else extract_streaming_tokens(provider, payload) → partial counts (max-merge)
  4. Save scanner state to filter_metadata
  5. On end_of_stream or scratch overflow → set_token_usage(...) and clear state
```

#### Provider-specific streaming behaviour

| Provider | Input tokens | Output tokens | Strategy |
|----------|-------------|---------------|----------|
| OpenAI / Azure | Final `usage` event | Final `usage` event | `extract_token_usage` per payload; last complete event wins |
| Anthropic | `message_start` (`message.usage.input_tokens`) | `message_delta` (`usage.output_tokens`) | `extract_streaming_tokens`; max-merge across events |
| Google (Gemini / Vertex) | Final `usageMetadata` chunk | Final `usageMetadata` chunk | `extract_token_usage` per payload; last wins |
| Bedrock InvokeModel | Response headers | Response headers | Header-only path in `on_response` ([#240]); no SSE |

> **Bedrock Converse streaming is not supported.** Production Bedrock
> Converse uses AWS binary event-stream (`application/vnd.amazon.eventstream`),
> not `text/event-stream` with `data:` frames. The filter only arms the SSE
> scanner for `text/event-stream`, so Converse streaming does not exercise
> this path. Non-streaming Converse JSON and InvokeModel headers are covered
> separately. See [`00220_token-counting-integration-tests.md`](./00220_token-counting-integration-tests.md)
> for the test boundary.

#### Accumulation semantics

Partial counts use **max-merge** (`existing.max(new)`). This is
correct for:

- **Last-wins providers** (OpenAI, Google): a later complete usage
  event overwrites both input and output atomically.
- **Split-event providers** (Anthropic): `input_tokens` and
  `output_tokens` arrive in separate events and are merged
  independently.
- **Cumulative providers**: the latest running total is always kept.

Intermediate SSE events with partial usage are not summed, avoiding
double-count when the final event already contains totals.

#### Stream completion signal

`end_of_stream` from the proxy is authoritative. Providers that emit
`data: [DONE]` (OpenAI) and providers that do not (Google Gemini) are
both handled correctly. The `[DONE]` sentinel is explicitly ignored
for token extraction.

#### Per-request metadata state

Working state uses the `token_count.*` prefix in `filter_metadata`:

| Key | Purpose |
|-----|---------|
| `token_count.mode` | `"sse"` or `"json"` |
| `token_count.input` | Accumulated input count (streaming) |
| `token_count.output` | Accumulated output count (streaming) |
| `token_count.sse_*` | SSE scanner state (line_buf, data_buf, etc.) |

On `end_of_stream`, `set_token_usage()` writes `token.input`,
`token.output`, `token.total` and all `token_count.*` keys are cleared.

#### Pipeline placement

```yaml
filters:
  - filter: token_usage_headers   # optional consumer ([#86])
  - filter: token_count
    provider: openai
  - filter: router
```

`token_count` must be declared **after** downstream consumers such as
`token_usage_headers` in the YAML list. Response hooks execute in
**reverse** declaration order, so `token_count` runs first and
populates `filter_metadata` before consumers read it. See
[`examples/configs/token-usage-headers.yaml`](../../examples/configs/token-usage-headers.yaml).

#### Example config

See [`examples/configs/token-counting.yaml`](../../examples/configs/token-counting.yaml).

#### Test plan

**Unit tests** (`filters/src/token_count/tests.rs`):

- SSE extraction for OpenAI, Anthropic, and Google
- Chunk boundaries splitting SSE frames
- `data: [DONE]` handling, overflow finalization, metadata cleanup

**Unit tests** (`apis/src/token_usage/tests.rs`, [#321]):

- `extract_streaming_tokens` for Anthropic `message_start` /
  `message_delta` and Bedrock metadata JSON (unit tests use synthetic
  `text/event-stream` fixtures; not production Bedrock Converse wire format)

**Integration tests** (`tests/integration/tests/suite/examples/token_count.rs`):

- Proxy starts and forwards traffic with `token-counting.yaml`
- Bedrock `InvokeModel` header path observable end-to-end via
  `token_usage_headers`

Streaming SSE end-to-end integration coverage remains a follow-up.

[#71]: https://github.com/praxis-proxy/ai/issues/71
[#83]: https://github.com/praxis-proxy/ai/issues/83
[#84]: https://github.com/praxis-proxy/ai/issues/84
[#86]: https://github.com/praxis-proxy/ai/issues/86
[#87]: https://github.com/praxis-proxy/ai/issues/87
[#119]: https://github.com/praxis-proxy/ai/issues/119
