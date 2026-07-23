# Examples

Configuration examples organized by category.

## Running an Example

```console
cargo run -p praxis-ai-proxy -- -c examples/configs/openai/responses/full-flow.yaml
curl http://localhost:8080/
```

Configs use local ports (`3000`, `3001`, ...) for
upstreams — start a real backend or stub on those ports
before sending requests.

## Configs

### General

| File | Description |
| ------ | ------------- |
| [a2a-agent-card-routing.yaml](configs/a2a-agent-card-routing.yaml) | Routes agent card discovery requests to dedicated backends |
| [a2a-classifier-routing.yaml](configs/a2a-classifier-routing.yaml) | Routes A2A requests by body-derived method, family, context ID, task ID, and streaming detection |
| [a2a-task-routing.yaml](configs/a2a-task-routing.yaml) | Captures task and context ownership from SendMessage JSON responses and SendStreamingMessage / SubscribeToTask SSE responses, then routes follow-up requests back to the backend cluster that created the task or owns the context |
| [ai-inference-body-based-routing.yaml](configs/ai-inference-body-based-routing.yaml) | Routes LLM API requests to different backends based on the `model` field in the JSON request body |
| [credential-injection.yaml](configs/credential-injection.yaml) | Injects per-cluster API credentials into upstream requests and strips client-provided credentials to prevent forwarding |
| [json-rpc-routing.yaml](configs/json-rpc-routing.yaml) | Routes JSON-RPC 2.0 requests to different backends based on the "method" field in the JSON request body |
| [mcp-classifier-routing.yaml](configs/mcp-classifier-routing.yaml) | Routes MCP requests by body-derived method and tool name |
| [mcp-stateless-broker.yaml](configs/mcp-stateless-broker.yaml) | Configurable stateless MCP broker using the 2026-07-28 release candidate profile |
| [model-to-header-routing.yaml](configs/model-to-header-routing.yaml) | Routes LLM API requests to different backends based on the "model" field in the JSON request body |
| [prompt-enrichment.yaml](configs/prompt-enrichment.yaml) | Injects system messages into OpenAI-compatible chat completion requests before forwarding to the upstream provider |
| [token-counting.yaml](configs/token-counting.yaml) | Extracts token usage from AI inference responses (streaming and non-streaming) and makes counts available to downstream filters via filter metadata as token.input, token.output, and token.total |
| [token-usage-headers.yaml](configs/token-usage-headers.yaml) | Inject Praxis-Token-Input, Praxis-Token-Output, and Praxis-Token-Total headers into downstream responses when token counts are available in filter metadata |

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
| [embeddings-routing.yaml](configs/openai/embeddings/embeddings-routing.yaml) | Routes OpenAI Embeddings API requests to a dedicated Embeddings API backend |
| [prompts-routing.yaml](configs/openai/prompts/prompts-routing.yaml) | Routes OpenAI Prompts API requests to a dedicated Prompts API backend |
| [doc-extract.yaml](configs/openai/responses/doc-extract.yaml) | Converts `input_file` content parts to `input_text` for inference backends that do not natively support `input_file` (e.g. vLLM, llm-d) |
| [file-resolve.yaml](configs/openai/responses/file-resolve.yaml) | Resolves `file_id` and `file_url` references in Responses API input by fetching file metadata and content, then inlining base64 content as `file_data` or `image_url` before forwarding |
| [format-routing.yaml](configs/openai/responses/format-routing.yaml) | Routes AI API traffic by detected body format |
| [full-flow.yaml](configs/openai/responses/full-flow.yaml) | Combines conversations, format classification, request validation, file resolution, and backend routing into a single pipeline |
| [mcp-dispatch.yaml](configs/openai/responses/mcp-dispatch.yaml) | Demonstrates the `openai_mcp_dispatch` filter configuration |
| [mcp-tool-resolve.yaml](configs/openai/responses/mcp-tool-resolve.yaml) | Demonstrates the `openai_mcp_tool_resolve` filter, which resolves MCP tool entries in the Responses API `tools` array into concrete tool definitions by calling `tools/list` on each upstream MCP server |
| [model-rewrite.yaml](configs/openai/responses/model-rewrite.yaml) | Rewrites or injects the top-level `model` field in Responses API request bodies before forwarding to the inference backend |
| [rehydrate.yaml](configs/openai/responses/rehydrate.yaml) | Validates `previous_response_id` by fetching the stored response, confirming its status is completed, and promoting the ID to filter metadata |
| [request-validate.yaml](configs/openai/responses/request-validate.yaml) | Validates Responses API requests and rejects invalid parameter combinations |
| [response-store.yaml](configs/openai/responses/response-store.yaml) | Persists non-streaming Responses API responses to a database and serves stored data via GET endpoints and handles DELETE /v1/responses/{id} locally |
| [responses-proxy.yaml](configs/openai/responses/responses-proxy.yaml) | Proxies OpenAI Responses API requests to a native /v1/responses backend |
| [responses-routing.yaml](configs/openai/responses/responses-routing.yaml) | Routes Responses API traffic by detected mode |
| [stream-events.yaml](configs/openai/responses/stream-events.yaml) | Demonstrates the `openai_stream_events` filter, which observes SSE chunks from the backend without modification, accumulates state (response object, output items, tool calls, usage), and writes it to ResponsesState metadata |
| [tool-routing.yaml](configs/openai/responses/tool-routing.yaml) | Demonstrates using `openai_tool_parse` to route Responses API requests by their tool composition |
| [vector-stores-routing.yaml](configs/openai/responses/vector-stores-routing.yaml) | Routes /v1/vector_stores traffic and all its subresources to a dedicated backend (any server compatible with the OpenAI Files / Vector Stores API), while sending everything else to a default backend |
| [vllm-agentic-api.yaml](configs/openai/responses/vllm-agentic-api.yaml) | vLLM Agentic API: https://github.com/vllm-project/agentic-api |
| [web-search.yaml](configs/openai/responses/web-search.yaml) | Demonstrates the `openai_web_search` filter configuration |

### Payload Processing

| File | Description |
| ------ | ------------- |
| [mcp-static-catalog.yaml](configs/payload-processing/mcp-static-catalog.yaml) | Provides a static MCP catalog and broker for initialize, tools/list, ping, and notifications/initialized requests |
