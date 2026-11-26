# OpenAI Responses: stateful agentic loop conformance

This page lists which state-related behavior Praxis's OpenAI Responses
pipeline (`apis/src/openai/responses/`) claims when it fronts a
`/v1/responses`-compatible backend such as vLLM or llm-d. It covers the
stateful agentic-loop pipeline: the `rehydrate` + `response_store` +
`agentic_loop` filter set, as wired in `examples/configs/`. A minimal
passthrough chain without those filters forwards every field to the backend
and claims none of the state behavior below.

## Responses support and qualification

Clients call `/v1/responses` in both GPU lanes. **Native** sends inference to
vLLM's `/v1/responses`. **Translated** sends it to vLLM's
`/v1/chat/completions`. The text nightly serves **Qwen/Qwen3-8B** on an
**NVIDIA A10G** and runs the Responses suite with **OpenAI Python 2.x and
3.x** against a PostgreSQL-backed gateway. The table identifies what each
check exercises. Use the linked run evidence to determine which cases passed
for a particular checkout.

| Client operation or parameter | What the application can rely on | How we exercise it |
| --- | --- | --- |
| Persist a response with `store` | Managed Responses stores completed responses by default. Set `store: false` to disable response persistence. | Native and translated SDK cases create with `store: true`, retrieve by ID, page through input items, and delete the stored response. |
| Continue with `previous_response_id` and `input` | Praxis loads a completed response and replays its history into inference. The native pipeline restores the previous ID on eligible JSON responses and SSE lifecycle events. | Native SDK cases check the echoed ID in buffered and streamed replies and reject an invalid ID. Translated cases exercise stored continuation, including returning custom tool output to the Chat backend. |
| Attach a turn with `conversation` | Praxis loads the conversation history and appends the completed turn. Conversation items can persist even when the response itself uses `store: false`. | Native SDK cases create a conversation, run multiple turns, and list its items. A streaming case checks append after completion with response storage disabled. Standalone Conversations CRUD exercises the gateway store without model inference. |
| Stream with `stream` and `stream_options` | Clients receive Responses SSE. The native lane consumes upstream Responses events. The translated lane constructs them from Chat chunks. | Native SDK cases cover usage options and continuation. Translated SDK cases exercise event decoding and the `include_obfuscation` option. |
| Call tools with `tools` and `tool_choice` | A client can receive a function call and submit its output on a later turn. The agentic pipeline also uses `max_tool_calls` for local dispatch and reads `parallel_tool_calls` for scheduling. | A separate native GPU case forces a function call and completes the loop with **OpenAI Agents SDK 0.23.1 / OpenAI Python 3.26.0**. The Responses suite also exercises custom-tool continuation through the Chat lane. |
| Request compaction with `context_management` | With the compact filter configured, Praxis can summarize rehydrated history before inference. | SDK cases exercise below- and over-threshold history using the compact configuration. Consult the per-case profile for the backend used. |
| Set generation options | Native forwards fields including `instructions`, `temperature`, `top_p`, `max_output_tokens`, `reasoning`, and `text` to the provider. Translation maps supported fields to Chat equivalents. | Native SDK cases exercise structured output, logprobs via `include`, and `top_p` on a continuation. The independent [translation oracle](openresponses-translation.md) exercises system prompts, multi-turn input, tool calling, and streaming on CPU vLLM with **Qwen/Qwen3-0.6B**. |
| Set `background: true` or a non-null `prompt` template | The managed vLLM path rejects these requests. | The [state flag reference](#state-related-request-flags) documents the validation boundary. |

**Finding results.** The [vLLM Integration runs](https://github.com/praxis-proxy/ai/actions/workflows/vllm-integration.yaml)
publish `qualification.json` with the checkout, resolved vLLM and SDK
versions, model, and per-case profiles and outcomes. Supporting fixtures
and skipped cases are identified separately. For release-specific results,
use the qualification section and attached report on the
[GitHub Release](https://github.com/praxis-proxy/ai/releases).
See the [report field guide](../developing/vllm-qualification.md) and
[GPU suite inventory](../developing/gpu-nightly-suite.md) for reproduction
details and the separate vision and coding-client lanes.

For verified CLI edits, permission modes and compaction outcomes by model and API route, see [Coding harness compatibility](coding-harness-compatibility.md).

## Conformance boundary (read this first)

On `POST /v1/responses` Praxis forwards the client bytes to the backend
verbatim (`SelectedUpstreamBodyOutcome::Continue`). It rebuilds the outbound
body only when rehydrated history must be replayed, the agentic loop appended
tool results, or a filter (including the agentic loop) changed a
provider-visible field
(`responses_proxy::request_needs_rebuild` in
`apis/src/openai/responses/responses_proxy/mod.rs`). So:

1. Praxis claims conformance only for state it implements:
   `previous_response_id` continuation, `conversation` linkage, response
   persistence (`store`), and the local lifecycle endpoints (`GET` / `DELETE` /
   `input_items`) served from its own response store.
2. The backend owns conformance for generation flags. `temperature`,
   `reasoning`, `text`/response_format, `max_output_tokens`, `instructions`,
   `truncation`, `metadata`, `service_tier`, and the rest are forwarded to the
   backend's `/v1/responses` implementation (e.g. vLLM). Praxis neither
   validates nor transforms them.

The one state mechanic Praxis adds is rehydration. When a continuation
selector is present, Praxis rebuilds prior turns from its store, replays them
through the `input` array, and strips the selector from the upstream body, so
the backend always runs statelessly. On the way out it restores only
`previous_response_id`, never `conversation`. The buffered restore applies
only to an eligible finite `200 OK` JSON response; streaming uses a separate
per-frame path (see the `stream` row).

> Out of scope: the Chat-translation profile. The
> `openai_responses_to_chat_completions` filter is a separate deployment for
> Chat-native backends. It maps `max_output_tokens`→`max_completion_tokens`,
> `reasoning.effort`→`reasoning_effort`, `instructions`→system message, and
> `text.format`→`response_format`; rejects `truncation != "disabled"` and
> `background != false` with `400`; and drops
> `metadata`/`store`/`include`/`max_tool_calls` from the upstream body. None of
> that applies to the native `/v1/responses` → vLLM passthrough described here.
> See the [translation matrix](translation-matrix.md#openai-responses-to-openai-chat-completions).

## State-related request flags

| Flag | Owner | Praxis behavior on `POST /v1/responses` | Conformance claim |
| --- | --- | --- | --- |
| `previous_response_id` | **Praxis (stateful)** | Looked up in the Praxis response store; the stored response must have status `"completed"`, otherwise `400`. Stored history is prepended to the replayed `input`. The field is stripped from the upstream body (the backend echoes `null`) and restored into the response. Both restore paths are gated: the buffered restore (`eligible_previous_response_id`) needs a finite `200 OK` JSON body, and the streaming restore (`eligible_previous_response_id_stream`) passes through unrestored any response with `Content-Encoding`, `Content-Range`, or a validator or integrity digest. Previous usage is captured for auto-compaction. | Praxis-owned. Continuation comes from Praxis state, not the backend's own `previous_response_id`. |
| `conversation` | **Praxis (stateful)** | Used when no `previous_response_id` is present. Resolved to a conversation record in the Praxis store; its history is prepended and the field is stripped from the upstream body. A malformed value or a missing record → `400`; an unavailable store or failed lookup → `500` (`fetch_conversation`). No silent provider-delta fallback: the provider-owned delta path (`provider_owns_conversation`) applies only when the rehydrate filter is absent or bypassed. | Praxis-owned for local rehydration. |
| `store` | **Mixed** | Defaults to `true` (spec default). Controls whether the completed response, buffered or streamed, is persisted to the Praxis store so later `previous_response_id` and `GET` requests resolve. `store:false` disables Praxis persistence and so removes the replay source. The value is not stripped; the backend receives it and may honor it independently. | Praxis-owned for persistence into the Praxis store; the field value is backend-forwarded. |
| `background` | **Rejected on managed path** | `background:true` → `400 "background mode is not supported"` (`handle_unsupported_background`) from the gateway-managed request processor. The standalone classifier does not reject it, so provider-owned passthrough forwards the field. No asynchronous Responses lifecycle (no queued/polled `202` handoff). | Not supported on managed paths. No claim. |
| `prompt` (prompt template) | **Rejected on managed path** | A non-null `prompt` → `400 "prompt templates are supported only for OpenAI-owned upstreams; send prompt content via input …"` (`reject_prompt_template`). OpenAI has deprecated reusable prompts; only provider-owned OpenAI upstreams may carry it. | Not claimed for backend (vLLM) paths. |
| `previous_input_messages` (vLLM extension) | **Rejected on managed path** | Any value other than `null` or `[]` → `400 invalid_request_error`, `code: "unsupported_parameter"`, `param: "previous_input_messages"` (`reject_caller_supplied_history`), on first turns and continuations, with or without `store`; omitting it is accepted. Praxis owns the history sent to inference. On vLLM's Harmony rendering path (gpt-oss models), a caller-supplied array lands ahead of the rehydrated history, and vLLM's check against combining it with `previous_response_id` never fires because Praxis strips that selector first. Send earlier messages in `input`, or resume with `previous_response_id` or `conversation`. Provider-owned passthrough forwards it unchanged. | Not claimed for backend (vLLM) paths. |
| `stream` | **Mixed** | Selects the outbound transport (`select_terminal_response_mode`). Praxis composes the inference streams of all agentic rounds into one logical Responses SSE stream (`openai_stream_events`) and rewrites `previous_response_id` in lifecycle frames. | The backend owns event generation; Praxis owns logical-stream composition and the id restore. |
| `include` | **Passthrough (inspected)** | Forwarded unchanged. Read only so hosted-tool filters can shape the sections they emit locally (e.g. `file_search_call.results`, web-search sources). Does not drive continuation or persistence. | Backend owns the claim; Praxis fills only locally produced sections. |
| `context_management` | **Mixed (inspected)** | Forwarded unchanged. With the `openai_responses_compact` filter (feature `openai-compact`) configured, it drives compaction: `extract_compaction_config` reads it and `apply_compaction` rewrites `state.messages` into a summary that becomes the outbound `input`. Separate from the top-level `truncation` enum, which is pure passthrough. | Backend owns the field; Praxis adds local compaction. |
| `max_tool_calls` | **Mixed (enforced)** | Forwarded unchanged and also enforced locally as a dispatch budget for built-in/hosted tools across the agentic loop: over-budget web-search calls are marked failed, and file-search and other locally dispatched calls stay incomplete. Matters only when hosted or agentic tools run. | Praxis owns built-in/local tool budgeting; the backend owns native semantics. |
| `parallel_tool_calls` | **Passthrough (inspected)** | Forwarded unchanged (default `true`). Read only by local MCP/agentic concurrency scheduling, and never mutated across rounds. | Backend owns per-round generation semantics. |
| `tools` / `tool_choice` | **Agentic-loop owned / forwarded** | Rich client tools may be lowered to private `function` tools and restored on the way out. MCP, web-search, and file-search tools are dispatched locally in the loop. Plain backend tools are forwarded. | Praxis owns locally dispatched tools; the backend owns native tool generation. |

### Generation flags (forwarded, backend owns conformance)

Praxis forwards these unread, or reads them only for classification, and
makes no conformance claim for them:

`model`¹, `instructions`, `temperature`, `top_p`, `top_logprobs`,
`max_output_tokens`, `reasoning`, `text` / response_format, `truncation`,
`metadata`, `service_tier`, `logit_bias`, `seed`, `user` /
`safety_identifier`, `prompt_cache_key`, and any other sampling or output field.

¹ Praxis rewrites `model` only when the optional
`openai_responses_model_rewrite` filter is configured.

## Stateful endpoint surface

The store filter (`store/filter.rs`) serves the Responses lifecycle endpoints
locally from the Praxis response store; these requests do not reach the
backend:

| Endpoint | Handling | Owner |
| --- | --- | --- |
| `POST /v1/responses` | Create: rehydrate (if a selector is present) → forward to backend → persist. | Backend generates; Praxis owns state. |
| `GET /v1/responses/{id}` | Served from the store (`200`, or `404` before the response is stored). | Praxis-owned. |
| `GET /v1/responses/{id}?stream=true` | SSE replay of the stored event log, for completed and persisted responses only, with a `starting_after` cursor. Not live mid-generation resumption. | Praxis-owned. |
| `GET /v1/responses/{id}/input_items` | Served from the store. | Praxis-owned. |
| `DELETE /v1/responses/{id}` | Deleted from the store locally. | Praxis-owned. |
| `POST /v1/responses/{id}/cancel` | Route registered with `mode=Passthrough`; the store filter does not handle it, so it goes to the backend. Inert while `background` is unsupported. | Backend. |
| `POST /v1/responses/input_tokens` | Body is parsed and classified like a create, so a history selector gets the same rejection and rehydration handling. Token counting is the backend's. Not forwarded unread. | Backend (count); Praxis (selector handling). |
| `POST /v1/responses/compact` | OpenAI-spec endpoint (`/responses/compact`), handled locally (`handle_explicit_compact`) only when the optional `openai_responses_compact` filter (feature `openai-compact`) is configured; otherwise forwarded. Separate from reactive `context_management` compaction. Only `model` is required (`input`/`previous_response_id` are optional and nullable per the spec), so an empty conversation is compacted as a no-op: no summarization callout, `output: []`, zero usage, and an empty history persisted under the returned id (#1139). | Praxis-owned when the compact filter is configured. |

## Rules that govern the state flags

- Mutual exclusion: `previous_response_id` together with `conversation` →
  `400 mutually_exclusive_parameters` (`reject_conflicting_history_selectors`).
- Completion gate: continuing from a non-`completed` stored response →
  `400 "cannot continue from response with status '<status>'"`
  (`validate_response_status`).
- Store required for state: a request that needs the store (persistence or
  rehydration) when none is provisioned → `500`.
- Client body is not edited for rehydration: history lives in
  `ResponsesState.messages` (`state.rs`, `from_request_body`), and the upstream
  body is rebuilt only at serialization time.
- Continuation key: the store keys each response on the response body's
  own `id` (the backend's id), and the client sends that `id` back as the next
  `previous_response_id`. The request filter also mints a proxy-owned `resp_`
  id for conversation and metadata linkage. The buffered path rewrites only
  `previous_response_id`, not the response `id`. The streaming path does
  normalize the response `id` across agentic rounds:
  `normalize_logical_payload` pins the first round's id as the logical stream
  id and rewrites later backend ids to it (`stream_events/mod.rs`), so a
  multi-round streamed response shows one stable id.
