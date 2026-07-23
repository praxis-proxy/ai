# Token Counting

Part of the **parse** plane on the [AI Gateway][overview]:
extract token usage from AI provider responses and write
unified counts to filter metadata.

[overview]: overview.md

## `token_count` filter

`token_count` inspects upstream response bodies (JSON or SSE),
parses provider-specific usage fields, and writes
`token.input`, `token.output`, and `token.total` to
`filter_metadata`. Response bodies pass through unchanged.

Set `provider` to match your upstream:

| Value | Formats |
| ----- | ------- |
| `openai` | OpenAI Chat Completions, Responses API |
| `anthropic` | Anthropic Messages (JSON and SSE) |
| `google` | Gemini-style usage blocks |
| `bedrock` | AWS Bedrock Converse API and Claude InvokeModel |
| `azure` | Azure OpenAI (same JSON as OpenAI) |

Example:

```yaml
filter_chains:
  - name: ai
    filters:
      - filter: router
        routes:
          - path_prefix: "/v1"
            cluster: backend
      - filter: load_balancer
        clusters:
          - name: backend
            endpoints: ["127.0.0.1:3000"]
      - filter: token_count
        provider: openai
```

Working config: [token-counting.yaml](../examples/configs/token-counting.yaml).

## `token_usage_headers` filter

`token_usage_headers` injects `Praxis-Token-Input`,
`Praxis-Token-Output`, and `Praxis-Token-Total` when those
metadata keys are already present. It runs in `on_response`,
before response body chunks are processed.

`token_count` writes usage in `on_response_body` after the
body is parsed. **A single-pass pipeline cannot yet turn
body-derived counts into client response headers.** The
[token usage headers proposal](proposals/00214_token-usage-response-headers.md)
documents the limitation and open options (trailers, buffering,
or a later pipeline hook).

Use `token_usage_headers` today when another filter (or an
earlier hook) populates token metadata before `on_response`.
Otherwise it is a no-op, as in
[token-usage-headers.yaml](../examples/configs/token-usage-headers.yaml).

## Related

- [Filter reference](filters/reference.md)
- [Features](features.md#security-and-observability)
- [Proposal 00214](proposals/00214_token-usage-response-headers.md)
