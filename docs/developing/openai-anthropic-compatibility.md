# OpenAI Chat Completions to Anthropic Messages compatibility

This note defines the compatibility behavior of the OpenAI Chat Completions
request translator for Anthropic Messages. It does not make unsupported fields
generally permissive: the translator continues to reject fields and content
shapes outside the explicit contract below.

## Request contract

| OpenAI Chat Completions input | Anthropic Messages output | Behavior |
|---|---|---|
| `model` | selected Anthropic model | The client model is retained for the Chat Completions response; the configured target model is sent to Anthropic. |
| `max_completion_tokens` | `max_tokens` | Takes precedence when present and must be a positive integer. |
| `max_tokens` | `max_tokens` | Used when `max_completion_tokens` is absent; must be a positive integer. |
| Neither token field | `max_tokens: 4096` | Uses this translator's compatibility default. |
| Both token fields | `max_tokens` | `max_completion_tokens` wins, including when the values differ. This precedence is explicit and covered by tests. |
| `user` | omitted | Accepted only as a string, then discarded. User identity is not forwarded. |
| `n` | omitted | Accepted only as a positive integer, then discarded. This does not request multiple completions. |
| `seed` | omitted | Accepted only as an integer, then discarded. This does not make generation deterministic. |
| `parallel_tool_calls` | omitted | Accepted only as a boolean, then discarded. It does not constrain provider tool scheduling. |
| Text content | Anthropic text block | Preserved in order. |
| Base64 `image_url` data URL | Anthropic base64 image source | JPEG, PNG, GIF, and WebP are accepted when the base64 payload is non-empty, decodable, and has a matching format signature. |
| HTTPS `image_url` | Anthropic URL image source | The URL is passed through; the translator does not fetch it. |

`image_url.detail` accepts `auto`, `low`, or `high` and is discarded because
Anthropic Messages has no equivalent quality control. Unknown top-level or
nested fields, malformed values, unsupported content block types, non-HTTPS
remote image URLs, embedded URL credentials, and malformed or unsupported data
URLs remain rejected with HTTP 400. The translator does not accept arbitrary
OpenAI options and silently drop them.

Tool declarations, tool choices, assistant tool calls, and tool results retain
their existing validation and conversion behavior. The compatibility options
above do not alter response conversion or the incremental SSE translator.

## Body limits

Request and response buffering limits are independent:

| Setting | Default | Maximum | Purpose |
|---|---:|---:|---|
| `max_request_body_bytes` | 2 MiB | 16 MiB | Complete Chat Completions request needed for validation and translation. |
| `max_response_body_bytes` | 1 MiB | 16 MiB | Non-streaming Anthropic JSON response retained for conversion. |

For existing configurations, `max_body_bytes` remains supported as a shared
limit. A direction-specific setting takes precedence over the shared setting
for that direction. If neither direction-specific nor shared value is set,
the defaults in the table apply. Zero and values above the maximum are
rejected during configuration validation. Streaming responses continue to use
their independent event-size controls.

The request limit is checked before provider-format branching. Consequently,
an OpenAI-compatible candidate using this request filter has the same request
cap even though its body is passed through after the selected target model is
rewritten. The response limit applies only to buffered Anthropic JSON response
conversion; it does not change SSE processing.

## Validation

Run the focused converter tests with:

```sh
cargo test -p praxis-ai-apis anthropic::openai_chat_completions_to_anthropic
```

The tests cover token defaults and precedence, ignored compatibility fields,
strict rejection of invalid inputs, image source conversion and ordering,
selected target model identity, and independent request/response limits.
