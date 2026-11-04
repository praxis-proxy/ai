# OpenAI Chat Completions to Anthropic Messages compatibility

This note defines the compatibility behavior of the OpenAI Chat Completions
request translator for Anthropic Messages. It does not make unsupported fields
generally permissive: the translator continues to reject fields and content
shapes outside the explicit contract below.

## API version and sampling

The `openai_chat_completions_to_anthropic` filter selects the backend API
contract from trusted configuration, independently of caller headers:

```yaml
filter: openai_chat_completions_to_anthropic
api_translation_enabled: true
api_version: "2026-10-21"
```

`api_version` defaults to `2023-06-01`. Only `2023-06-01` and `2026-10-21`
are accepted; unknown versions fail configuration parsing. The selected value
overrides any caller-supplied `anthropic-version` and any co-configured
`anthropic_messages_protocol.default_version` on translated requests,
regardless of the order of those two filters. Native passthrough retains its
own version behavior.

Version selection is scoped to the translator's filter configuration, not to
individual provider entries in the routing overlay. Use separate configured
routes for Anthropic-compatible backends requiring different API versions.
OpenAI-compatible provider selections are unaffected by this setting.

Under `2023-06-01`, numeric `temperature` and `top_p` values are forwarded
unchanged. The existing rejection of `top_k` on this Chat Completions bridge
remains in place. Under `2026-10-21`, the presence of `temperature`, `top_p`,
or `top_k` causes HTTP 400 before dispatch, including default values and
`null`. This is deliberately strict: even clients that automatically send
`temperature: 1` or `top_p: 1` must omit those fields for the new contract.
Version-specific errors identify the field and API version; `top_k` instead
reports that the bridge cannot preserve it, under either version. No sampling
field is silently removed. Omit unsupported fields or ask the platform
administrator to configure a contract supported by the selected backend.

Native Messages passthrough has a different contract:
`anthropic_messages_protocol` preserves caller-supplied `anthropic-version`
and does not change the body or validate version-specific sampling rules.
Its `default_version` applies only when the caller supplies no version.
The backend can therefore reject a new-version request with sampling controls.
Messages-to-Chat Completions translation also retains sampling support,
including `top_k`, for compatible vLLM backends; it does not apply Anthropic
provider restrictions to those backends.

The controlled example
[`openai-selected-provider-version-fixture.yaml`](../../examples/configs/openai-selected-provider-version-fixture.yaml)
and inference replay cover new-version success and rejection before dispatch.
They use synthetic backend responses, not live Anthropic evidence.

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

Image conversion is supported in user messages only. Assistant image blocks
are rejected rather than silently dropped or moved to another role. System,
developer, and tool-result content remains text-only in this translator.

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

These are filter limits, not overrides of the listener's body ceilings. Praxis
listeners default to 10 MiB for both `body_limits.max_request_bytes` and
`body_limits.max_response_bytes`. To use a filter limit above 10 MiB, also raise
the corresponding listener ceiling to accommodate it. A lower listener ceiling
still takes precedence.

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
