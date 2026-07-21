---
issue: https://github.com/praxis-proxy/ai/issues/353
status: proposed
authors:
  - mkoushni
graduation_criteria:
  - Task-route capture work is O(n) in total response body bytes, not O(n·k) / O(n²) under fine-grained chunking
  - Opportunistic pre-EOS capture behavior is unchanged (route still captured as soon as the JSON body is complete, without waiting for `end_of_stream`)
  - No full-buffer hex-decode or `serde_json::from_slice` attempt on a buffer known to hold incomplete JSON
  - Existing A2A response-body unit test coverage passes unmodified
stakeholders:
  - aslakknutsen
---

# A2A Task Route Capture Parse Complexity

## What?

Bound the per-chunk cost of non-streaming A2A task-route capture so
that total work across a response body is O(n) in body bytes rather than O(n·k) for body size n and chunk count k (O(n²) when chunks are O(1) bytes).

`try_capture_from_buffer` in `filters/src/agentic/a2a/mod.rs` runs on
every `on_response_body` invocation while capture is enabled. Each
call hex-decodes the *entire* accumulated buffer and attempts a full
`serde_json::from_slice` parse over it, then discards the result if
the JSON is incomplete. Because the buffer grows on every chunk and
this decode-and-parse attempt repeats on every chunk, total work is
proportional to the sum of accumulated buffer sizes across all
chunks — Θ(n·k): each of the k chunks re-processes the growing prefix; for fixed n that is linear in k, and Θ(n²) when k scales with n (e.g. 1-byte chunks).

### Goals

- Only attempt the (expensive) hex-decode + JSON parse once the
  accumulated buffer plausibly contains a complete, balanced JSON
  value, using an incremental, string-aware brace/bracket depth
  scan updated per chunk (O(chunk) work, not O(buffer) work).
- Preserve opportunistic capture: a route must still be captured as
  soon as the JSON body is complete, without waiting for
  `end_of_stream` — this is required because Pingora may not deliver
  a separate EOS callback after the final data chunk, and is covered
  by existing tests (e.g. `json_response_split_across_chunks_captures_opportunistically`).
- Preserve existing behavior for the bounded-body-size cap
  (`max_response_body_bytes`), multibyte UTF-8 chunk-boundary
  handling, and invalid/oversized-JSON fallback paths.
- Keep the change local to the non-streaming capture path
  (`accumulate_response_hex` / `try_capture_from_buffer`); the SSE
  capture path (`process_sse_response_chunk`) already scans
  incrementally and is not affected.

### Non-Goals

- Replacing the hex-encoding scheme used to survive chunk boundaries
  that split multibyte UTF-8 sequences. That mechanism is not the
  source of the quadratic behavior on its own and is out of scope
  here.
- A general streaming JSON parser. The response shapes handled by
  `extract_task_route` are fixed, small JSON-RPC objects; a full
  incremental parser is unnecessary complexity for this case.
- Changing `on_response` SSE-vs-non-SSE classification (see #352,
  handled separately).
- Any change to the SSE streaming capture path or its scanner
  (`filters/src/agentic/a2a/sse.rs`).

## Why?

### Motivation

`try_capture_from_buffer` exists to let the A2A filter capture a task
route from a non-streaming `SendMessage` (or non-SSE
`SendStreamingMessage` / `SubscribeToTask`) response body as soon as
it is fully received, without waiting for an end-of-stream signal
that Pingora does not reliably deliver as a separate callback. That
requirement is legitimate and already has test coverage.

The problem is *how* completeness is currently checked: by repeating
the full parse attempt, from scratch, over the entire buffer, on
every chunk. For a response delivered in `k` chunks over `n` total
bytes, this produces work proportional to the sum of prefix lengths ≈ n(k+1)/2 = Θ(n·k); with unit-sized chunks (k ≈ n) that is Θ(n²), because each of the `k` attempts re-decodes and re-parses
everything accumulated so far rather than only the newly arrived
bytes.

In the common case — a small JSON-RPC response arriving in one or
two chunks — this is negligible. But nothing bounds the number of
chunks a backend or intermediary may use to deliver a body up to the
configured `max_response_body_bytes` cap (65,536 bytes by default).
A body delivered in many small chunks (whether by a slow/segmented
backend, a malicious backend, or simply an unusual proxy hop)
repeats full hex-decode and JSON-parse work on an ever-growing buffer
on every chunk, which is avoidable extra CPU work on the request path
of every response subject to task-route capture.

### User Stories

- As a **platform operator** running A2A task routing in production,
  I want per-response CPU cost for route capture to scale linearly
  with response size, so that unusual chunking patterns from
  backends do not create disproportionate CPU load on the proxy.

- As a **filter maintainer**, I want the non-streaming capture path
  to only pay for a full JSON parse when the buffer is plausibly
  complete, so that the existing opportunistic, pre-EOS capture
  behavior is preserved without repeating unnecessary work per chunk.

- As a **security-conscious operator**, I want the cost of handling a
  many-small-chunks response body to be bounded and predictable, so
  that it cannot be used to amplify CPU cost on the proxy within the
  existing body-size cap.
