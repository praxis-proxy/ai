# Filters

This page explains how filters compose; the generated
[filter reference](reference.md) has the exact settings. Capabilities not
marked optional are registered in the default production binary. Check
[build features](../features.md#cargo-features) before choosing an image.

## Choose a capability

| Operator task | Filters and prerequisites | Order and working material |
| --- | --- | --- |
| Identify the client operation | `ai_operation`; protocol body owners `openai_chat_completions_request`, `anthropic_messages_request`, or experimental `openai_responses_request` | Classify before operation-dependent policy. [Client paths](../getting-started/choose-client-and-provider.md), [reference](core/ai_operation.md) |
| Forward a native API | Core `router`, `headers`, `load_balancer`; provider endpoint, Host, TLS/SNI, credentials | Select a cluster before credential injection. [Quickstart](../getting-started/quickstart.md), [Anthropic recipe](../guides/anthropic.md) |
| Translate protocols | Direction-specific request/response adapter plus SSE adapter where required | Request owner before translation; select the provider before selected-provider translation. [Compatibility matrix](../conformance/translation-matrix.md), [path table](../getting-started/choose-client-and-provider.md) |
| Map public model aliases | `model_to_provider`, then `load_balancer` | Do not add a competing router. [Routing recipe](../guides/routing.md), [reference](routing/model_to_provider.md) |
| Select providers dynamically | `model_to_header`, `intelligent_route`, `credential_inject`, `load_balancer`; trusted candidate catalog and secrets | Selection publishes the profile used by adapters and credentials. [Routing recipe](../guides/routing.md), [reference](routing/intelligent_route.md) |
| Route within a provider | `peer_identity_trust`, `provider_route`, endpoint selection | Requires authenticated downstream mTLS and trusted routing assertions. [Example](../../examples/configs/provider-route.yaml), [reference](routing/provider_route.md) |
| Change prompt content | `prompt_enrich`; protocol-compatible Chat body | Apply before final token ceilings and inference. [Example](../../examples/configs/prompt-enrichment.yaml), [reference](core/prompt_enrich.md) |
| Separate identities and secrets | Authentication boundary, `state_owner`, `identity_header_guard`, `project_state_owner_headers`, `callout_credentials` as appropriate | Header capture does not authenticate provenance. Authenticate or delete-then-set assertions at a trusted ingress. [Credential boundaries](../guides/README.md#credentials), [state owner](core/state_owner.md) |
| Screen content | `ai_operation`, then `ai_guardrails`; NeMo 0.24.0+ and deployed rails | Configure outbound callout policy. Response screening has streaming restrictions. [Recipe](../guides/guardrails.md), [reference](control/ai_guardrails.md) |
| Extract and observe usage | `model_to_header`, `token_count`, optional `token_usage_metrics`, `time_to_first_token` | Body counts are available after headers; do not promise `token_usage_headers` on these paths. [Recipe](../guides/guardrails.md), [count reference](observability/token_count.md) |
| Bound output or reserve token budgets | Optional experimental `token_ceiling`, `token_rate_limit`, `inflight_tracker` | Ceiling evaluates the final provider-bound body; rate limiting needs its configured budget backend and trusted key dimensions. [Ceiling example](../../examples/configs/token-ceiling.yaml), [rate-limit reference](control/token_rate_limit.md), [inflight reference](control/inflight_tracker.md) |
| Check balance and report usage | `external_metering`; reachable metering service, destination-bound callout authentication | Pre-request admission and post-response reporting are separate. [File-key example](../../examples/configs/external-metering-file-key.yaml), [reference](control/external_metering.md) |
| Broker MCP or route A2A | `mcp`, `a2a`, core JSON-RPC and routing; reachable services | Protocol discovery/routing does not execute model tools. [MCP example](../../examples/configs/mcp-stateless-broker.yaml), [A2A example](../../examples/configs/a2a-task-routing.yaml), [architecture](../architecture/agentic-protocols.md) |
| Manage Responses, tools, files, history | Experimental Responses filters, selected tool groups and store backend | Follow the whole pipeline, including IRR, streaming composition, owner scope, and callout limits. [Recipe](../guides/managed-responses.md), [store architecture](../architecture/response-store.md) |

## Ordering and boundaries

Request hooks generally run in declared order and response hooks in reverse.
Body and header phases are separate: moving a usage reader later cannot make
body-derived data available before response headers are sent. Branches and IRR
steps have their own lifecycle, so check a body filter's declared phase and
pipeline validation before moving it into a branch.

Start from the complete examples linked above and run
`praxis-ai --validate -c FILE` with the binary you will deploy. Validation
proves registration and static constraints, not backend compatibility,
credentials, or tool execution. Streaming transformations need their SSE
adapters and bounds; a finite JSON translation does not imply streaming
support.

Core routing, load balancing, TLS, authentication, headers, rate limiting,
CORS, compression, and other base proxy behavior are in the
[Praxis core reference][praxis-filters]. Provider credentials have two owners
with different configuration formats: core `credential_injection` uses a
selected cluster, and AI `credential_inject` uses the selected dynamic
provider profile.

## Organization

Generated filter pages live in category folders under `docs/filters/`
(`apis`, `providers`, `routing`, `control`, `agents`, `observability`, and
`core`), grouped the same way in the [reference index](reference.md).
`cargo xtask generate-filter-docs` assigns each filter a category; the
category table and overrides are in `xtask/src/filter_docs.rs`.

Rust crate layout is in the [embedding guide](../developing/filter-hosts.md#organization).
Custom Rust hosts follow its [registration instructions](../developing/filter-hosts.md#registration),
including the response-store pipeline extension when needed.

## Related documentation

- [Generated filter reference](reference.md)
- [Example configurations](../../examples/README.md)
- [Build features](../features.md)
- [Embedding and registering filters](../developing/filter-hosts.md)
- [Writing extensions](extensions.md)

[praxis-filters]: https://github.com/praxis-proxy/praxis/tree/main/docs/filters
