# AI inference

How Praxis AI classifies, routes, and enriches inference traffic using the
filter pipeline and `StreamBuffer` body access.

## Overview

AI inference filters identify the operation from the request head, then, for
operations that carry a body, extract routing signals from it (model, stream
mode, store flag) and promote them to headers, metadata, and filter results that
branch chains and the router consume.

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

## Classification pipeline

### Operation identity (request head)

The `ai_operation` filter (`operation_classifier/mod.rs`) identifies the
operation from the HTTP method, normalized path, and protocol headers. It never
reads the body. One shared matcher consults every protocol-owned registry
(OpenAI Responses, Conversations, Chat Completions, Anthropic Messages), so a
single filter recognizes all providers. It publishes a typed `AiOperationMatch`
in request extensions, plus metadata and filter results for branching.
Bodyless sub-resource endpoints (`GET /v1/responses/{id}`,
`POST /v1/responses/{id}/cancel`, etc.) are matched here from the path.

The filter runs once in the request-head phase, which core executes before any
downstream filter's buffered body pre-read and before the main request phase.
A consumer that reads the body under `BodyMode::StreamBuffer` therefore already
has the match.

A matched operation is authoritative over body shape: a Chat-Completions-shaped
body posted to `POST /v1/responses` is still a Responses request.

### Body facts

For a matched Responses operation, `openai_responses_request` parses the body
JSON once through the classifier (`classifier/mod.rs`), a pure function with no
I/O, and gets a `ClassifiedRequest` with the extracted facts (model, stream,
store, background, mode). The matched operation forces the format fact to
Responses. Bodyless operations (fetch, delete, cancel) and operations whose
body the specification marks optional promote operation identity without
reading a body.

When no operation forces the format, as with a request routed on body shape
alone, the classifier falls back to shape detection in this order:

1. `input` field present: Responses API
2. `messages` + `max_tokens` + Anthropic signals
   (`system` or typed content blocks):
   Anthropic Messages API
3. `messages` alone: Chat Completions API
4. Valid JSON without recognized fields:
   `UnknownJson`
5. Invalid JSON: `InvalidJson`
6. Non-JSON content type: `NonJson`

### Metadata propagation

`openai_responses_request` promotes classified facts on three channels:

- Filter metadata (e.g. `openai_responses_request.model`): persists across
  Pingora phases for cross-filter communication.
- Extra request headers (e.g. `X-Praxis-AI-Format`): added to the upstream
  request for header-based routing in the router filter.
- Filter results: written to `FilterResultSet` for branch chain conditions.

Every promoted value is checked against a 256-byte limit and for control
characters first. See [Routing Facts](routing-facts.md) for which facts routing
filters may read.

Inside `iterative_request_router` steps, metadata is reset for credential
isolation but extensions persist, so a filter that reads classifier metadata
needs the classifier to re-run inside its step. See
`examples/configs/inference/fallback-with-translation.yaml` for the pattern.

### Stateful vs stateless

A Responses API request is "stateful" when any of these hold:

- `previous_response_id` is present
- `tools` is present
- `store` is not explicitly `false`
- `has_conversation` is true
- `has_prompt_id` is true

Stateful mode affects routing, for example directing the request to clusters
with response store access.

`has_prompt_id` detects OpenAI's deprecated reusable prompt object
(`prompt: { id, version, variables }`). OpenAI retires reusable prompts and
`v1/prompts` on 2026-11-30. On managed paths, clients should move the saved
prompt's content into the structured `input` form that content-policy
extraction covers. Top-level `instructions` is not part of that extraction, so
content moved there skips managed-path content screening. Praxis keeps
recognizing the shape for routing while OpenAI still supports it.

Requests with `background: true` are rejected before mode classification,
because Praxis does not implement the asynchronous Responses lifecycle.

For an agentic Responses pipeline, the
[retained-payload budget](responses-retained-payload.md) accounts for input,
history, tool results, and output across inference rounds.

## StreamBuffer body access

`StreamBuffer` accumulates request body chunks and holds back upstream
forwarding until the filter releases them or the stream ends:

1. Buffer the JSON envelope (model name, parameters, prompt prefix).
2. Extract routing signals from the buffered bytes.
3. Select the upstream based on body content.
4. Release the buffered prefix and stream the remainder.

This gives filters body visibility without the latency of an external
processor. Filters opt in by declaring body access and
`BodyMode::StreamBuffer { max_bytes }`. Body writers such as
`model_to_provider` and `prompt_enrich` use `BodyAccess::ReadWrite` to replace
the buffered request before forwarding it.

## Filters

Configuration for each filter lives on its generated page:
[`ai_operation`](../filters/core/ai_operation.md),
[`model_to_header`](../filters/routing/model_to_header.md),
[`model_to_provider`](../filters/routing/model_to_provider.md),
[`openai_responses_request`](../filters/apis/openai_responses_request.md),
[`anthropic_messages_request`](../filters/apis/anthropic_messages_request.md),
[`anthropic_messages_to_vertexai_anthropic`](../filters/providers/anthropic_messages_to_vertexai_anthropic.md),
[`prompt_enrich`](../filters/core/prompt_enrich.md), and
[`openai_responses_store`](../filters/agents/openai_responses_store.md)
(see also [Response Store](response-store.md)). How they fit together:

### `model_to_header`

Promotes the body `model` field to a configurable header (default `X-Model`)
for header-based routing to provider-specific clusters. Any client-supplied
copy of the header is stripped so routing cannot be spoofed. When a trusted hop
in front of Praxis already sets the header, an
`unless: {headers_present: [...]}` condition on the filter skips it and keeps
the incoming value.

### `model_to_provider`

Maps an exact client model and request path to an existing load-balancer
cluster and target model. It stages the cluster during body pre-read and
selects it during the request phase; the load balancer chooses endpoints and
owns health checks and retries. This is ordinary cluster selection: it does not
publish the frozen binding read by `bound_upstream` conditions.

Unknown models and paths keep their bodies and select `fallback_upstream` when
configured. The original client model stays in request metadata for metering
and optional provider response adapters. An adapter that uses cluster
application-provider metadata must run after selection and require the
appropriate tags.

### `openai_responses_request`

As the managed owner, it generates cryptographically random response and
conversation IDs with `resp_` and `conv_` prefixes and initializes state.
Provider-owned parameter combinations pass through unchanged. It offers both
the pre-read and bound-upstream body phases, so a chain can defer it until a
logical provider is bound. With `initialize_state: false` it classifies for a
pure routing chain without building state, and caches its one parse for a
later managed owner. That owner re-validates the cached body before reuse, so
it never reuses a body that an intervening filter rewrote. On a chain with no
managed owner, the request-scoped cache is released unused when the request
ends.

### `anthropic_messages_request`

Owns the Anthropic create-message body. The Messages registry decides the
operation from the request head, then the body is deserialized once for
envelope validation, routing facts, and canonical state.

### `anthropic_messages_to_vertexai_anthropic`

Translates the Anthropic wire format only after the load balancer selects a
cluster tagged with HTTP `application_provider: vertex`. Its selected-upstream
body hook receives the body already rewritten by `model_to_provider`, which
keeps provider selection and public-to-target model mapping. The exact target
model, including any snapshot suffix, moves into the Messages URL or stays in
the count-tokens body. Responses restore the public model recorded by the
mapper, including `message_start` frames processed incrementally. Other
providers and untagged clusters pass through without Vertex translation. The
[Vertex example](../../examples/configs/vertex/messages-to-anthropic.yaml)
includes both aliases and a separate fallback upstream.

### `credential_injection`

Per-cluster API key injection with client credential stripping, from inline
values or environment variables.

## Key files

- `apis/src/operation_classifier/mod.rs`:
  `AiOperationFilter` (request-head operation identity)
- `apis/src/classifier/mod.rs`:
  pure format classifier
- `apis/src/openai/responses/request/mod.rs`:
  `OpenaiResponsesRequestFilter`
- `filters/src/inference/model_to_header.rs`:
  `ModelToHeaderFilter`
- `filters/src/inference/model_to_provider.rs`:
  `ModelToProviderFilter`
- `filters/src/prompt_enrich/`:
  prompt enrichment filter
- `apis/src/anthropic/`:
  Anthropic Messages format filter

## Related

- [OpenAI Responses retained-payload budget](responses-retained-payload.md)
- [Response Store](response-store.md)
- [Agentic Protocols](agentic-protocols.md)
- [Features](../features.md)
