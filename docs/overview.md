# AI Gateway

Praxis AI (`praxis-ai`) is an **AI-native proxy server** and
**AI Gateway** (also deployable as an **AI API Gateway**).
It extends the [Praxis][praxis] composable filter pipeline
with AI-specific filters that classify, enrich, and parse
inference and agentic traffic between workloads and model
or agent backends.

Router, load balancer, TLS, CORS, rate limiting, and other
core filters come from Praxis. AI filters register into the
same YAML filter pipeline.

[praxis]: https://github.com/praxis-proxy/praxis

## What is an AI Gateway?

An AI Gateway is a filter pipeline dedicated to inference and
agentic workloads. Callers may be external clients, in-cluster
applications, or outbound services; backends may be hosted
models, provider APIs, or agent runtimes.

The gateway:

- **Routes** traffic to the right provider or cluster
- **Manages** policy, credentials, rate limits, content
  safety, and resilience
- **Enriches** prompts and conversation state
- **Parses** request and response bodies (JSON and SSE)

Unlike routing on URL and headers alone, an AI Gateway
classifies provider API formats from request bodies and
applies policy in the data plane. Content safety combines
core [`guardrails`](https://github.com/praxis-proxy/praxis/blob/main/docs/filters/http/security/guardrails.md)
(rule and PII matching) with
[`ai_guardrails`](features.md#security-and-observability)
(external provider evaluation for pass, block, or redact).
Rate limiting and credential controls use the same core
filters as any Praxis deployment.

## AI-native proxy server

An **AI-native proxy server** understands AI wire formats
incrementally: OpenAI Responses and Chat Completions,
Anthropic Messages, and agentic JSON-RPC (MCP, A2A). It uses
[StreamBuffer][streambuffer] body access to inspect JSON
before upstream selection, handles streaming SSE responses,
and shares facts across filter phases via
[`filter_metadata`][filter-metadata].

[streambuffer]: https://github.com/praxis-proxy/praxis/blob/main/docs/architecture/payload-processing.md
[filter-metadata]: filters/extensions.md#filter_metadata

## Primary use-cases

- **Ingress**: API gateway for external clients and partners
- **Egress**: Application outbound calls to provider APIs
- **In-cluster**: Kubernetes `Service` (or similar) for
  pod-to-model traffic

```text
  [ Client / Pod / Service ]
            |
            v
     +--------------+
     |  praxis-ai   |  classify, guard, enrich, limit, parse
     +--------------+
            |
     +------+------+------+
     v      v      v      v
  OpenAI  Anthropic  MCP  A2A backends
```

YAML and filter chains are the same across placements; only
listener address, TLS, and upstream clusters change per
environment.

## Route, manage, enrich, parse

| Plane | What it does | Examples |
| ----- | ------------ | -------- |
| **Route** | Classify format; pick upstream cluster | `openai_responses_format`, `anthropic_messages_format`, `model_to_header`, `router`, branch chains |
| **Manage** | Policy, limits, credentials, resilience | `rate_limit`, `credential_injection`, `guardrails` (core), `ai_guardrails`, `ip_acl`, health checks, circuit breaker |
| **Enrich** | Prompts, stored state, multi-turn context | `prompt_enrich`, `openai_responses_rehydrate`, `openai_response_store`, `openai_conversations` |
| **Parse** | Bodies, SSE, agentic metadata, usage | StreamBuffer classifiers, `token_count`, `mcp`, `a2a`, `json_rpc` |

See [Features](features.md) for the full filter list.

## Unified AI API Gateway

A single listener can front multiple provider APIs. Classifier
filters promote format and model facts to headers; the router
selects the backend cluster. Example:
[unified-gateway.yaml](../examples/configs/anthropic/unified-gateway.yaml).

## Praxis core vs praxis-ai

| | `praxis` | `praxis-ai` |
| - | -------- | ----------- |
| **Role** | General proxy framework (ingress, egress, load balancing) | AI Gateway on the same runtime |
| **AI filters** | No | Yes (provider APIs, agentic, store, tokens) |
| **Config** | YAML filter chains | Same format |

Deploy `praxis` for non-AI or custom binaries. Deploy
`praxis-ai` when workloads need an AI Gateway or AI API
Gateway out of the box.

## Next steps

- [Quickstart](quickstart.md)
- [Features](features.md)
- [AI inference pipeline](architecture/ai-inference.md)
- [OpenAI Responses API](openai-responses.md)
- [Anthropic Messages API](anthropic-messages.md)
- [Agentic protocols](architecture/agentic-protocols.md)
- [Filter reference](filters/reference.md)
