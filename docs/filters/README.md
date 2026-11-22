# Filters

Start with the problem you need to solve. The generated [filter reference](reference.md)
owns exact settings; this page explains composition. Unless marked optional,
these capabilities are registered in the default production binary. See
[build features](../features.md#cargo-features) before selecting an image.

## Choose a capability

| Operator task | Filters and prerequisites | Order and working material |
| --- | --- | --- |
| Identify the client operation | `ai_operation`; protocol body owners `openai_chat_completions_request`, `anthropic_messages_request`, or experimental `openai_responses_request` | Classify before operation-dependent policy. [Client paths](../client-provider-paths.md), [reference](ai_operation.md) |
| Forward a native API | Core `router`, `headers`, `load_balancer`; provider endpoint, Host, TLS/SNI, credentials | Select a cluster before credential injection. [Quickstart](../quickstart.md), [Anthropic recipe](../recipes/anthropic.md) |
| Translate protocols | Direction-specific request/response adapter plus SSE adapter where required | Request owner before translation; select the provider before selected-provider translation. [Compatibility matrix](../conformance/translation-matrix.md), [path table](../client-provider-paths.md) |
| Map public model aliases | `model_to_provider`, then `load_balancer` | Do not add a competing router. [Routing recipe](../recipes/routing.md), [reference](model_to_provider.md) |
| Select providers dynamically | `model_to_header`, `intelligent_route`, `credential_inject`, `load_balancer`; trusted candidate catalog and secrets | Selection publishes the profile used by adapters and credentials. [Routing recipe](../recipes/routing.md), [reference](intelligent_route.md) |
| Route within a provider | `peer_identity_trust`, `provider_route`, endpoint selection | Requires authenticated downstream mTLS and trusted routing assertions. [Example](../../examples/configs/provider-route.yaml), [reference](provider_route.md) |
| Change prompt content | `prompt_enrich`; protocol-compatible Chat body | Apply before final token ceilings and inference. [Example](../../examples/configs/prompt-enrichment.yaml), [reference](prompt_enrich.md) |
| Separate identities and secrets | Authentication boundary, `state_owner`, `identity_header_guard`, `project_state_owner_headers`, `callout_credentials` as appropriate | Header capture does not authenticate provenance. Authenticate or delete-then-set assertions at a trusted ingress. [Credential boundaries](../recipes/README.md#credentials), [state owner](state_owner.md) |
| Screen content | `ai_operation`, then `ai_guardrails`; NeMo 0.24.0+ and deployed rails | Configure outbound callout policy. Response screening has streaming restrictions. [Recipe](../recipes/guardrails-token-usage.md), [reference](ai_guardrails.md) |
| Extract and observe usage | `model_to_header`, `token_count`, optional `token_usage_metrics`, `time_to_first_token` | Body counts are available after headers; do not promise `token_usage_headers` on these paths. [Recipe](../recipes/guardrails-token-usage.md), [count reference](token_count.md) |
| Bound output or reserve token budgets | Optional experimental `token_ceiling`, `token_rate_limit`, `inflight_tracker` | Ceiling evaluates the final provider-bound body; rate limiting needs its configured budget backend and trusted key dimensions. [Ceiling example](../../examples/configs/token-ceiling.yaml), [rate-limit reference](token_rate_limit.md), [inflight reference](inflight_tracker.md) |
| Check balance and report usage | `external_metering`; reachable metering service, destination-bound callout authentication | Pre-request admission and post-response reporting are separate. [File-key example](../../examples/configs/external-metering-file-key.yaml), [reference](external_metering.md) |
| Broker MCP or route A2A | `mcp`, `a2a`, core JSON-RPC and routing; reachable services | Protocol discovery/routing does not execute model tools. [MCP example](../../examples/configs/mcp-stateless-broker.yaml), [A2A example](../../examples/configs/a2a-task-routing.yaml), [architecture](../architecture/agentic-protocols.md) |
| Manage Responses, tools, files, history | Experimental Responses filters, selected tool groups and store backend | Follow the whole pipeline, including IRR, streaming composition, owner scope, and callout limits. [Recipe](../recipes/managed-responses.md), [store architecture](../architecture/response-store.md) |

## Ordering and boundaries

Request hooks generally follow declared order; response hooks run in reverse.
Body and header phases are distinct: placing a usage reader later cannot make
body-derived data available before response headers are sent. Branches and IRR
steps have their own lifecycle; do not transplant a body filter into a branch
without checking its declared phase and pipeline validation.

Use the complete examples linked above and run `praxis-ai --validate -c FILE`
with the selected binary. A valid configuration proves registration and static
constraints, not backend compatibility, credentials, or successful tool execution.
Streaming transformations must retain their SSE adapters and bounds; a finite
JSON translation alone does not establish streaming support.

Core routing, load balancing, TLS, authentication, headers, rate limiting,
CORS, compression, and other base proxy behavior belong to the
[Praxis core reference][praxis-filters]. AI provider credentials have two distinct
owners: core `credential_injection` uses a selected cluster, while AI
`credential_inject` consumes the selected dynamic provider profile. Their
configuration formats are different.

## Organization

Rust crate layout and host integration are in the
[embedding guide](../developing/filter-hosts.md#organization).

## Registration

Custom Rust hosts use the [registration instructions](../developing/filter-hosts.md#registration),
including the response-store pipeline extension when needed.

## Related documentation

- [Generated filter reference](reference.md)
- [Example configurations](../../examples/README.md)
- [Build features](../features.md)
- [Embedding and registering filters](../developing/filter-hosts.md)
- [Writing extensions](extensions.md)

[praxis-filters]: https://github.com/praxis-proxy/praxis/tree/main/docs/filters
