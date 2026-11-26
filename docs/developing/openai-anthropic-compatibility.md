# OpenAI Chat Completions to Anthropic Messages compatibility

This is the contract of `openai_chat_completions_to_anthropic`, the request
translator from Chat Completions to Anthropic Messages. Fields and content
shapes outside it are rejected.

## API version and sampling

The translator takes the backend API version from trusted configuration,
not from caller headers:

```yaml
filter: openai_chat_completions_to_anthropic
api_version: "2026-10-21"
```

`api_version` defaults to `2023-06-01`. Only `2023-06-01` and `2026-10-21`
are accepted; any other value fails configuration parsing. On translated
requests the selected value overrides a caller's `anthropic-version` and any
co-configured `anthropic_messages_protocol.default_version`, in either filter
order.

The setting applies to the whole translator configuration, not to
individual provider entries in the routing overlay. Anthropic-compatible
backends that need different versions need separate configured routes.
OpenAI-compatible provider selections are unaffected.

Under either version, supported numeric `temperature` and `top_p` values,
including explicit defaults, are forwarded unchanged. The backend owns
version-specific sampling validation, and its rejection is returned through
the existing OpenAI-compatible error translation. The bridge still rejects
`top_k` and unsupported field types, whatever the version.

The other `2026-10-21` restrictions can't be triggered here. Translated
requests contain only text, image, and tool blocks, so they never include
assistant `document` blocks or user text blocks with citations.

For a selected Anthropic provider, the translator removes the caller's
`authorization` and `x-api-key` headers, and `credential_inject` installs the
provider's key. A Chat Completions client's bearer token never reaches
Anthropic.

Native passthrough and Messages-to-Chat Completions translation follow
different version rules; see
[API version compatibility](../anthropic-messages.md#api-version-compatibility).

The controlled example
[`openai-selected-provider-version-fixture.yaml`](../../examples/configs/openai-selected-provider-version-fixture.yaml)
and inference replay cover new-version sampling preservation and upstream
error propagation, using synthetic backend responses rather than live
Anthropic evidence.

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

Images are converted in user messages only; assistant image blocks are
rejected. System, developer, and tool-result content stays text-only.

`image_url.detail` accepts `auto`, `low`, or `high` and is discarded, since
Anthropic Messages has no equivalent. Unknown top-level or nested fields,
malformed values, unsupported content block types, non-HTTPS remote image
URLs, embedded URL credentials, and malformed or unsupported data URLs are
rejected with HTTP 400.

Tool declarations, tool choices, assistant tool calls, and tool results keep
their existing validation and conversion. None of the options above change
response conversion or the incremental SSE translator.

## Body limits

Request and response buffering limits are independent:

| Setting | Default | Maximum | Purpose |
|---|---:|---:|---|
| `max_request_body_bytes` | 2 MiB | 16 MiB | Complete Chat Completions request needed for validation and translation. |
| `max_response_body_bytes` | 1 MiB | 16 MiB | Non-streaming Anthropic JSON response retained for conversion. |

These do not override the listener's body ceilings, which default to 10 MiB
for both `body_limits.max_request_bytes` and `body_limits.max_response_bytes`.
To use a filter limit above 10 MiB, raise the listener ceiling too; a lower
listener ceiling always wins.

`max_body_bytes` is still supported as a shared limit, and a
direction-specific setting overrides it for that direction. With neither set,
the table defaults apply. Zero and values above the maximum fail
configuration validation. Streaming responses use their own event-size
controls.

The request limit is checked before provider-format branching, so an
OpenAI-compatible candidate using this filter has the same request cap even
though its body passes through after the target model is rewritten. The
response limit applies only to buffered Anthropic JSON conversion, not SSE.

## Validation

Run the focused converter tests with:

```sh
cargo test -p praxis-ai-apis anthropic::openai_chat_completions_to_anthropic
```

They cover token defaults and precedence, ignored compatibility fields,
strict rejection of invalid inputs, image source conversion and ordering,
selected target model identity, and independent request/response limits.
