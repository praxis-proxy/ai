# OpenAI Responses — Stateful Agentic Loop Conformance

Scope: the **state-related** behavior of Praxis's OpenAI Responses pipeline
(`apis/src/openai/responses/`) when it fronts a `/v1/responses`-compatible
backend such as vLLM or llm-d.

> This document describes the **stateful agentic-loop pipeline** (the
> `rehydrate` + `response_store` + `agentic_loop` filter set, as wired in
> `examples/configs/`). A minimal passthrough chain without those filters
> forwards every field to the backend and claims none of the state behavior
> below.

## Conformance boundary (read this first)

On `POST /v1/responses` Praxis **forwards the request to the backend
essentially unchanged**. The outbound body is only rebuilt when state
genuinely requires it — rehydrated history must be replayed, the agentic loop
appended tool results, or a rewrite filter changed a provider-visible field
(`responses_proxy::request_needs_rebuild` in
`apis/src/openai/responses/responses_proxy/mod.rs`). Otherwise the
client bytes pass through to the backend verbatim
(`SelectedUpstreamBodyOutcome::Continue`).

Two consequences for conformance claims:

1. **Praxis claims conformance only for state it implements**:
   `previous_response_id` continuation, `conversation` linkage, response
   persistence (`store`), and the local lifecycle endpoints (`GET` / `DELETE` /
   `input_items`) served from its own response store.
2. **Praxis does not claim conformance for generation flags.** `temperature`,
   `reasoning`, `text`/response_format, `max_output_tokens`, `instructions`,
   `truncation`, `metadata`, `service_tier`, etc. are forwarded to the backend.
   **The backend's `/v1/responses` implementation (e.g. vLLM) owns those
   claims** — Praxis neither validates nor transforms them, so it must not be
   represented as conformant on them.

The only state mechanic Praxis layers on top of the passthrough is
**rehydration**: when a continuation selector is present it reconstructs prior
turns from its store and replays them through the `input` array, then strips the
selector from the upstream body. On the way out it restores **only
`previous_response_id`** — `conversation` is not restored — and the buffered
restore applies only to an eligible finite `200 OK` JSON response (streaming is
handled by a separate per-frame path; see the `stream` row). The backend itself
is always driven statelessly.

> **Out of scope — Chat-translation profile.** The
> `openai_responses_to_chat_completions` translation filter is a *separate* deployment
> for Chat-native backends. It rewrites/rejects many flags
> (`max_output_tokens`→`max_completion_tokens`, `reasoning.effort`→
> `reasoning_effort`, `instructions`→system message, `text.format`→
> `response_format`; rejects `truncation != "disabled"` and `background != false`
> with `400`; drops `metadata`/`store`/`include`/`max_tool_calls` from the
> upstream body). Those behaviors belong to that profile, **not** to the native
> `/v1/responses` → vLLM passthrough described here, and must not be attributed
> to it.

## State-related request flags

| Flag | Owner | Praxis behavior on `POST /v1/responses` | Conformance claim |
| --- | --- | --- | --- |
| `previous_response_id` | **Praxis (stateful)** | Validated against the Praxis response store; the stored response must have status `"completed"` or the request is rejected `400`. Stored history is prepended to the replayed `input`; the field is **stripped** from the upstream body (the backend never sees it and echoes `null`), then **restored** into the response. Restore is eligibility-gated on **both** paths: the buffered restore (`eligible_previous_response_id`) applies only to a finite `200 OK` JSON body, and the streaming restore (`eligible_previous_response_id_stream`) is declined for `Content-Encoding`/`Content-Range`/validator- or integrity-digest-bearing responses — those are passed through unrestored. Previous usage is captured for auto-compaction. | **Praxis-owned.** Continuation is served from Praxis state, not the backend's own `previous_response_id`. |
| `conversation` | **Praxis (stateful)** | When no `previous_response_id` is present, resolved to a conversation record in the Praxis store; its history is prepended and the field is stripped from the upstream body. A malformed value, or a **record not found**, → `400`; a store that is **unavailable or whose lookup fails** → `500` (`fetch_conversation`). There is no silent provider-delta fallback in this pipeline — the provider-owned delta path (`provider_owns_conversation`) applies only when the rehydrate filter is absent/bypassed. | **Praxis-owned** for local rehydration. |
| `store` | **Mixed** | Defaults to **`true`** (spec default). Governs whether the completed response is persisted to the **Praxis** store (buffered or streamed) so later `previous_response_id` / `GET` continuations resolve. `store:false` disables Praxis persistence (removing the rehydration/replay source). The raw value is **not** stripped — it is still forwarded to the backend, which may honor it independently. | **Praxis-owned** for persistence into the Praxis store; the field value is backend-forwarded. |
| `background` | **Rejected — unsupported** | `background:true` → `400 "background mode is not supported"` (`handle_unsupported_background`). Praxis implements no asynchronous Responses lifecycle (no queued/polled `202` handoff). | **Not supported by design** — no conformance claimed or implied. |
| `prompt` (prompt template) | **Rejected on managed path** | A non-null `prompt` → `400 "prompt templates are supported only for OpenAI-owned upstreams; send prompt content via input …"` (`reject_prompt_template`). OpenAI has deprecated reusable prompts; only provider-owned OpenAI upstreams may carry it. | Not claimed for backend (vLLM) paths. |
| `stream` | **Mixed** | Read to select the outbound transport (`select_terminal_response_mode`); Praxis composes the multiple agentic-round inference streams into **one logical Responses SSE stream** (`openai_stream_events`) and rewrites `previous_response_id` in lifecycle frames. | Event **generation** is the backend's claim; **logical-stream composition and the id restore** are Praxis's. |
| `include` | **Passthrough (inspected)** | Forwarded to the backend **unchanged**. Praxis reads it only so hosted-tool filters can shape their own locally-emitted sections (e.g. `file_search_call.results`, web-search sources). Not a continuation/persistence driver. | Backend owns the claim; Praxis populates only locally-produced sections. |
| `context_management` | **Mixed (inspected)** | Forwarded to the backend **unchanged**. When the `openai_responses_compact` filter (feature `openai-compact`) is configured it is the compaction driver: `extract_compaction_config` reads it and `apply_compaction` rewrites `state.messages` into a summary that becomes the outbound `input`. **Not** the top-level `truncation` enum (which is pure passthrough). | Backend owns the field; Praxis adds local compaction. |
| `max_tool_calls` | **Mixed (enforced)** | Forwarded **unchanged**, and enforced locally as a **built-in/hosted-tool** dispatch budget across the agentic loop (over-budget searches marked incomplete). Loop-relevant only when hosted/agentic tools run. | Praxis-owned for **built-in/local** tool budgeting; backend owns native semantics. |
| `parallel_tool_calls` | **Passthrough (inspected)** | Forwarded to the backend **unchanged** (default `true`). Read only by local MCP/agentic concurrency scheduling when those subsystems run; preserved, never mutated, across rounds. | Backend owns per-round generation semantics. |
| `tools` / `tool_choice` | **Agentic-loop owned / forwarded** | Rich client tools may be lowered to private `function` tools and restored on the way out; MCP / web-search / file-search tools are dispatched locally in the loop. Plain backend tools are forwarded. | Praxis owns **locally dispatched** tools; the backend owns native tool generation. |

### Generation flags — forwarded, backend owns conformance

These are **not** state flags. Praxis forwards them unread (or reads them only
for classification) and makes **no conformance claim** — the backend's
`/v1/responses` does:

`model`¹, `instructions`, `temperature`, `top_p`, `top_logprobs`,
`max_output_tokens`, `reasoning`, `text` / response_format, `truncation`,
`metadata`, `service_tier`, `logit_bias`, `seed`, `user` /
`safety_identifier`, `prompt_cache_key`, and any other sampling/output field.

¹ `model` is forwarded unless the optional `openai_responses_model_rewrite`
filter is configured, in which case Praxis rewrites it.

## Stateful endpoint surface

The Responses lifecycle endpoints are served **locally from the Praxis
response store** (`store/filter.rs`) — they do not reach the backend:

| Endpoint | Handling | Owner |
| --- | --- | --- |
| `POST /v1/responses` | Create: rehydrate (if a selector is present) → forward to backend → persist. | Backend generates; Praxis owns state. |
| `GET /v1/responses/{id}` | Served from the store (`200`, or `404` before the response is stored). | **Praxis-owned.** |
| `GET /v1/responses/{id}?stream=true` | SSE **replay** of the stored event log (completed/persisted responses only, via a `starting_after` cursor — **not** live mid-generation resumption). | **Praxis-owned.** |
| `GET /v1/responses/{id}/input_items` | Served from the store. | **Praxis-owned.** |
| `DELETE /v1/responses/{id}` | Deleted from the store locally. | **Praxis-owned.** |
| `POST /v1/responses/{id}/cancel` | Route registered but `mode=Passthrough`; the store filter does **not** handle it, so it is forwarded to the backend. Inert while `background` is unsupported. | Backend. |
| `POST /v1/responses/input_tokens` | Body is **parsed and classified** like a create (it is a body-bearing operation), so a history selector is subject to the same rejection/rehydration handling; token counting itself is the backend's. Not forwarded unread. | Backend (count); Praxis (selector handling). |
| `POST /v1/responses/compact` | An **OpenAI-spec** endpoint (`/responses/compact`), handled locally **only when the optional `openai_responses_compact` filter (feature `openai-compact`) is configured** (`handle_explicit_compact`); otherwise forwarded. Distinct from the reactive `context_management` compaction. | Praxis-owned when the compact filter is configured. |

## Rules that govern the state flags

- **Mutual exclusion**: supplying both `previous_response_id` and
  `conversation` → `400 mutually_exclusive_parameters`
  (`reject_conflicting_history_selectors`).
- **Completion gate**: continuation from a non-`completed` stored response →
  `400 "cannot continue from response with status '<status>'"`
  (`validate_response_status`).
- **Store required for state**: a request that needs the store (persistence or
  rehydration) but finds none provisioned → `500`.
- **Body is not mutated for rehydration**: history is carried in
  `ResponsesState.messages` (`state.rs`, `from_request_body`), not by editing the
  client body; the upstream body is rebuilt only at serialization time.
- **Continuation key**: the store persists keyed on the response body's own
  `id` (the backend's id). The client passes that `id` back as the next
  `previous_response_id`; the request filter also mints a proxy-owned `resp_` id
  for conversation/metadata linkage. On the **buffered** path only
  `previous_response_id` (not the response `id`) is rewritten. On the
  **streaming** path, however, the response `id` *is* normalized across agentic
  rounds: `normalize_logical_payload` pins the first round's id as the single
  logical stream id and rewrites later backend ids to it
  (`stream_events/mod.rs`), so a multi-round streamed response surfaces one
  stable id rather than each round's backend id.
