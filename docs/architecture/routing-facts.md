# Routing Facts

The request facts that routing filters may read, where each
one comes from, and the rules that keep them trustworthy.

## Why an allowlist

`intelligent_route` and `provider_route` decide where a request
goes from a small number of bounded strings derived from the
request before routing: the promoted model header, MCP tool
metadata, and a session key. Future candidate constraints
(entitlement, region, data classification, tool capability) will
read more of them. Every such input must be impossible for a
client to forge and must stay small enough to log and label.

A fact is therefore defined by three things:

1. **Producer** — the filter that derives it from the request
   head or body and is the only writer.
2. **Carrier** — a promoted request header, a `filter_metadata`
   key, or a typed extension.
3. **Hygiene** — the producer removes any client-supplied copy
   before writing (`remove-then-set`), and the value is bounded
   (256 bytes, no control characters, see `promotion::is_promotable_value`).

Routing filters read facts only from this table. A raw client
header is never a fact.

## Current facts

| Fact | Producer | Carrier | Remove-then-set | Read by routing |
|---|---|---|---|---|
| Model name | `model_to_header` | header `X-Model` (configurable) | yes (#1039) | `intelligent_route`, `provider_route`, `token_rate_limit` |
| MCP method and tool name | `mcp` | metadata `mcp.method`, `mcp.name`; headers `x-praxis-mcp-*` | reserved prefix stripped at ingress | `intelligent_route` (tool routing takes precedence over the model header) |
| A2A context/task owner | `a2a` | header `x-praxis-a2a-route-cluster` | reserved prefix stripped at ingress | core `router` |
| Operation and protocol | `ai_operation` | headers `x-praxis-ai-operation`, `x-praxis-ai-application-protocol`; metadata `ai_operation.*` | yes (removed on no match) | branch conditions only |
| Format, stream flag, tool presence, token limits | `openai_responses_request`, `anthropic_messages_request`, `openai_chat_completions_request` (metadata only) | headers `x-praxis-ai-format`, `x-praxis-ai-model`, `x-praxis-ai-stream`; metadata `<filter>.model`, `.stream`, `.has_tools`, `.max_tokens` | partial: default `x-praxis-ai-*` names are reserved; custom names are not stripped yet (praxis-proxy/ai#418) | `time_to_first_token` (model label); not yet by routing |
| Session key | request header or cookie named in `session_affinity` | request | not applicable (client-owned by design) | `intelligent_route` affinity |
| Tenant and subject | `identity_header_guard` (`identity.<header>` metadata), `state_owner` (`StateOwner` extension), authentication filters (`AuthenticatedIdentity` extension) | three carriers | partial: only `AuthenticatedIdentity` is verified in-process; `identity.*` and `StateOwner` are read from request headers and are trustworthy only behind an auth layer that strips client copies | `token_rate_limit` (subject), `external_metering` |
| Over-quota tier | `token_rate_limit` soft tiers | injected upstream header | written by the filter | core `router` |
| Guardrail verdict | `ai_guardrails`, `http_callout` | `filter_results` (ephemeral) | not applicable | branch chains only |

Facts that do not exist yet, and would be produced in the same
pre-read pass that `model_to_header` already performs: prompt
size, modality, and a tools-present flag for routing (tracked in
praxis-proxy/ai#106, #107, #108).

## Rules

- A routing filter reads a fact by its documented carrier and
  key. It never falls back to a raw request header with the same
  name.
- A producer that promotes a body field to a header pushes the
  header name to `request_headers_to_remove` before appending
  the derived value, whether or not the body carried the field.
  Core `json_body_field` only appends, so a client's own copy
  stays first and is the value a routing filter reads; use
  `model_to_header` for the model fact.
- Fact values are bounded and free of control characters. A
  value that fails the bound is treated as absent by the
  producer and never promoted.
- Facts may be logged. Prompt content, message bodies, cookies,
  and credentials are not facts and are never logged by routing
  filters. Session keys should be hashed before they appear in
  logs or traces.
- A new fact is added by extending this table and the producing
  filter, not by teaching a routing filter to read a new header.

## Open question

Three identity carriers exist (`identity.*` metadata, the
`StateOwner` extension, the `AuthenticatedIdentity` extension).
Candidate constraints that fence by tenant or claim need one
canonical source; that choice is tracked with the Intelligent
Router work (praxis-proxy/ai#74, praxis-proxy/ai#1282).

## Related

- [AI inference](ai-inference.md) — how format filters classify
  bodies and promote facts.
- [intelligent_route](../filters/intelligent_route.md) — the
  decision record and metrics the router writes from these facts.
