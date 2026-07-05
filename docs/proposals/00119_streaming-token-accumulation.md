---
issue: https://github.com/praxis-proxy/ai/issues/119
discussion: https://github.com/praxis-proxy/ai/issues/119
status: proposed
authors:
  - hschwart
graduation_criteria:
  - Token counts available for streaming SSE responses across OpenAI, Anthropic, Bedrock, and Vertex AI
  - Final aggregated counts written to filter metadata on stream completion
  - Response bodies pass through unmodified
  - How? section with implementation design or PR list
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
contains an earlier implementation design (How?) imported from the
`praxis-proxy/praxis` tracker.

This document is the **What?/Why?** entry point for [#119].
[#84] and [`00211`](./00211_streaming-token-counting.md) remain
reference material until maintainers decide whether to adopt, adapt,
or supersede them during the How? phase.

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

[#71]: https://github.com/praxis-proxy/ai/issues/71
[#83]: https://github.com/praxis-proxy/ai/issues/83
[#84]: https://github.com/praxis-proxy/ai/issues/84
[#86]: https://github.com/praxis-proxy/ai/issues/86
[#87]: https://github.com/praxis-proxy/ai/issues/87
[#119]: https://github.com/praxis-proxy/ai/issues/119
