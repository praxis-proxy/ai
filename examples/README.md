# Examples

Configuration examples organized by category.

## Running an Example

```console
cargo run -p praxis-ai-proxy -- -c examples/configs/traffic-management/basic-reverse-proxy.yaml
curl http://localhost:8080/
```

Configs use local ports (`3000`, `3001`, ...) for
upstreams. For quick experiments without a real backend,
use `static_response` (see
[static-response.yaml](configs/traffic-management/static-response.yaml))
or run Praxis with no config file for a built-in welcome
page.

## Configs

### Anthropic

| File | Description |
| ------ | ------------- |
| [messages-protocol.yaml](configs/anthropic/messages-protocol.yaml) | Routes Anthropic Messages API requests to a native `/v1/messages` backend |
| [messages-to-openai.yaml](configs/anthropic/messages-to-openai.yaml) | Transforms Anthropic Messages API requests and responses for Chat Completions-compatible inference backends |
| [request-validate.yaml](configs/anthropic/request-validate.yaml) | Rejects empty, malformed, or non-object JSON request bodies |
| [unified-gateway.yaml](configs/anthropic/unified-gateway.yaml) | Routes traffic by classifier-promoted headers so a single listener handles Anthropic Messages, OpenAI Chat Completions, and OpenAI Responses requests |

### OpenAI

| File | Description |
| ------ | ------------- |
| [conversations.yaml](configs/openai/conversations/conversations.yaml) | Local /v1/conversations endpoints for conversation lifecycle, backed by the ConversationItemStore |
| [format-routing.yaml](configs/openai/responses/format-routing.yaml) | Routes AI API traffic by detected body format |
| [full-flow.yaml](configs/openai/responses/full-flow.yaml) | Combines format classification, request validation, and backend routing into a single pipeline |
| [model-rewrite.yaml](configs/openai/responses/model-rewrite.yaml) | Rewrites or injects the top-level `model` field in Responses API request bodies before forwarding to the inference backend |
| [rehydrate.yaml](configs/openai/responses/rehydrate.yaml) | Validates `previous_response_id` by fetching the stored response, confirming its status is completed, and promoting the ID to filter metadata |
| [request-validate.yaml](configs/openai/responses/request-validate.yaml) | Validates Responses API requests and rejects invalid parameter combinations |
| [response-store.yaml](configs/openai/responses/response-store.yaml) | Persists non-streaming Responses API responses to a database and serves stored data via GET endpoints and handles DELETE /v1/responses/{id} locally |
| [responses-proxy.yaml](configs/openai/responses/responses-proxy.yaml) | Proxies OpenAI Responses API requests to a native /v1/responses backend |
| [responses-routing.yaml](configs/openai/responses/responses-routing.yaml) | Routes Responses API traffic by detected mode |

### Payload Processing

| File | Description |
| ------ | ------------- |
| [mcp-static-catalog.yaml](configs/payload-processing/mcp-static-catalog.yaml) | Provides a static MCP catalog and broker for initialize, tools/list, ping, and notifications/initialized requests |
