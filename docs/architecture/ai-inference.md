# AI Inference

Body-aware classification, routing, and enrichment
for AI inference traffic, built on the filter
pipeline and StreamBuffer body access pattern.

## Overview

AI inference filters identify the operation from the
request head, then — for operations that carry one —
extract routing signals from the body (model, stream
mode, store flag) and promote them to headers,
metadata, and filter results for downstream routing via
branch chains.

Protocol and operation identity come from the request
head, not the body shape. The `ai_operation` filter
reads the HTTP method, normalized path, and protocol
headers alone and publishes a typed match (OpenAI
Responses, Conversations, Chat Completions, or Anthropic
Messages, plus the specific operation) before any body
is read. A matched operation is authoritative over body
shape: a Chat-Completions-shaped body posted to
`POST /v1/responses` is still a Responses request.

`openai_responses_request` then runs for matched
Responses operations. It reads the body only to extract
facts (model, stream, store, background, mode) and,
when configured as the managed owner, to initialize
state. Bodyless operations (fetch, delete, cancel) and
operations whose body the specification marks optional
promote operation identity without reading a body.

```text
Request Head (method, path, protocol headers)
  |
  v
ai_operation (publishes typed operation match; no body read)
  |
  v
Request Body (only for operations that carry one)
  |
  v
openai_responses_request (body facts; optional state + ID generation)
  |
  v
Branch Chains / Router (routing decisions)
  |
  v
Upstream
```

## Classification Pipeline

### Operation Identity (request head)

The `ai_operation` filter
(`operation_classifier/mod.rs`) identifies the operation
from the HTTP method, normalized path, and protocol
headers alone — the body is never read. Every
protocol-owned registry (OpenAI Responses,
Conversations, Chat Completions, Anthropic Messages) is
consulted through one shared matcher, so a single filter
recognizes all providers. It publishes a typed
`AiOperationMatch` in request extensions, plus metadata
and filter results for branching, before any body-reading
filter runs. Sub-resource endpoints that lack a body
(`GET /v1/responses/{id}`,
`POST /v1/responses/{id}/cancel`, etc.) are matched here
from the path.

### Body Facts

The classifier (`classifier/mod.rs`) is a pure function
with no I/O. For a matched Responses operation,
`openai_responses_request` parses the request body JSON
once and returns a `ClassifiedRequest` with the extracted
facts. The matched operation is authoritative: the format
fact is forced to Responses regardless of body shape, so
a Chat-Completions-shaped body on `POST /v1/responses` is
classified as a Responses request.

When no operation forces the format — a request routed on
body shape alone — the pure classifier falls back to
shape-based detection precedence:

1. `input` field present: **Responses API**
2. `messages` + `max_tokens` + Anthropic signals
   (`system` or typed content blocks):
   **Anthropic Messages API**
3. `messages` alone: **Chat Completions API**
4. Valid JSON without recognized fields:
   **UnknownJson**
5. Invalid JSON: **InvalidJson**
6. Non-JSON content type: **NonJson**

### Metadata Propagation

The `openai_responses_request` filter promotes classified
facts using three channels:

- **Filter metadata**: durable key-value pairs
  (e.g. `openai_responses_request.model`) that
  persist across Pingora phases. Used for
  cross-filter communication.
- **Extra request headers**: added to the upstream
  request (e.g. `X-Praxis-AI-Format`). Used for
  header-based routing in the router filter.
- **Filter results**: written to `FilterResultSet`
  for branch chain condition evaluation.

All promoted values are validated against a 256-byte
length limit and checked for control characters
before propagation.

Inside `iterative_request_router` steps, metadata is
reset for credential isolation but extensions persist.
Filters that read classifier metadata must have the
classifier re-run inside their step. See
`examples/configs/inference/fallback-with-translation.yaml`
for the full pattern.

### Stateful vs Stateless

Responses API requests are classified as "stateful"
when any of these hold:

- `previous_response_id` is present
- `tools` is present
- `store` is not explicitly `false`
- `has_conversation` is true
- `has_prompt_id` is true

`has_prompt_id` detects OpenAI's deprecated reusable prompt
object (`prompt: { id, version, variables }`). OpenAI retires
reusable prompts and `v1/prompts` on 2026-11-30; on managed
paths, clients should move the saved prompt's content into the
structured `input` form covered by content-policy extraction.
Top-level `instructions` is not currently included in that
extraction, so content moved there bypasses managed-path content
screening. Praxis keeps recognizing the shape for routing while
OpenAI still supports it.

Requests with `background: true` are rejected before mode
classification because Praxis does not implement the
asynchronous Responses lifecycle.

Stateful mode influences routing decisions (e.g.
directing to clusters with response store access).

## StreamBuffer Body Access

StreamBuffer is the key enabler for AI inference
filters. It accumulates request body chunks and
defers upstream forwarding until the filter releases
or end-of-stream:

1. Buffer the JSON envelope (model name, parameters,
   prompt prefix).
2. Extract routing signals from the buffered bytes.
3. Select the upstream based on body content.
4. Release the buffered prefix and stream the
   remainder.

This peek-then-stream pattern avoids the latency of
external processor architectures while providing body
visibility where it matters.

Filters declare `BodyAccess::ReadOnly` +
`BodyMode::StreamBuffer { max_bytes }` to opt in.
Only `PromptEnrichFilter` uses `ReadWrite` (it
modifies the messages array).

## Filters

### `ai_operation`

Identifies the operation from the request head — HTTP
method, normalized path, and protocol headers — without
reading the body. Publishes a typed `AiOperationMatch`
in request extensions plus metadata and filter results
for branching, so downstream filters know the protocol
and operation before any body-reading filter runs. Runs
in both the header and body phases: the body-phase hook
inspects only the head and publishes once, guaranteeing
the match is available before a downstream filter's
buffered body pre-read. A matched operation is
authoritative over body shape.

### `model_to_header`

Extracts the `model` field from JSON request bodies
and promotes it to a configurable header (default
`X-Model`). Enables header-based routing to
provider-specific clusters.

Any client-supplied copy of the header is stripped so
routing cannot be spoofed. When a trusted hop in front
of Praxis already sets the header, an
`unless: {headers_present: [...]}` condition on the
filter skips it entirely, preserving the incoming value.

### `openai_responses_format`

Classifies AI API request bodies and promotes format,
model, stream, store, background, and mode to
headers, metadata, and filter results.

### `openai_responses_request`

Runs for the Responses operations `ai_operation`
matched. Reads the request body to extract and promote
format, model, stream, store, background, and mode to
headers, metadata, and filter results, parsing the JSON
once. As the managed owner it also generates
cryptographically random response and conversation IDs
with `resp_` and `conv_` prefixes and initializes state.
Provider-owned parameter combinations pass through
unchanged. Offers both the pre-read and bound-upstream
body phases, so a chain can defer it until a logical
provider is bound, and `initialize_state: false` lets a
pure routing chain classify without building state. A
pre-routing facts pass (`initialize_state: false`) always
caches its one parse for a later managed owner, which
re-validates the cached body before reuse so an
intervening body rewrite is never reused stale; on a chain
with no managed owner that request-scoped cache is simply
released, unused, when the request ends.

### `anthropic_messages_format`

Classifies Anthropic Messages API requests and
promotes format metadata.

### `prompt_enrich`

Injects system or user messages into
OpenAI-compatible chat completion request bodies.
Static configured messages are prepended or appended
to the `messages` array. Uses `BodyAccess::ReadWrite`.

### `credential_injection`

Per-cluster API key injection with client credential
stripping. Supports inline values and environment
variable sources.

### `openai_response_store`

Persists non-streaming Responses API responses. See
[Response Store](response-store.md) for details.

## Key Files

- `apis/src/operation_classifier/mod.rs`:
  `AiOperationFilter` (request-head operation identity)
- `apis/src/classifier/mod.rs`:
  pure format classifier
- `apis/src/openai/responses/request/mod.rs`:
  `OpenaiResponsesRequestFilter`
- `filters/src/inference/model_to_header.rs`:
  `ModelToHeaderFilter`
- `filters/src/prompt_enrich/`:
  prompt enrichment filter
- `apis/src/anthropic/`:
  Anthropic Messages format filter

## Related

- [Response Store](response-store.md)
- [Agentic Protocols](agentic-protocols.md)
- [Features](../features.md)
