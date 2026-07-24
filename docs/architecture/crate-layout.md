# Crate Layout

Praxis AI extends the core proxy with three workspace crates.
All AI filters implement the same `HttpFilter` trait from
`praxis-filter` and register at startup in `server/src/lib.rs`.

## Dependency flow

```text
praxis-ai-proxy (server)
  -> praxis-ai-filters
    -> praxis-ai-apis
      -> praxis-filter / praxis-core / praxis-protocol (core)
```

| Crate | Role |
| ----- | ---- |
| `server` (`praxis-ai-proxy`) | Binary `praxis-ai`; registers AI + core filters |
| `filters` (`praxis-ai-filters`) | MCP, A2A, guardrails, model routing, token counting, token headers |
| `apis` (`praxis-ai-apis`) | OpenAI/Anthropic filters, classifier, store, token usage parsers |

## Key modules

```text
apis/src/
  classifier/       Request body format detection
  openai/           Responses API, conversations, SSE
  anthropic/        Messages API filters
  store/            ResponseStore trait, SQLite/Postgres
  token_usage/      Provider-specific usage extraction (library)

filters/src/
  agentic/          MCP and A2A filters
  guardrails/       AI content guardrails (ai_guardrails)
  inference/        Promotes model body field to header (model_to_header)
  prompt_enrich/    Chat completion message injection
  token_count/      Response token usage extraction
  token_usage_headers.rs  Injects Praxis-Token-* response headers
```

## Pipeline extensions

`ResponseStoreRegistry` implements `PipelineExtension` and is
injected when pipelines are built (`server/src/pipelines.rs`).
Filters such as `openai_response_store` and
`openai_responses_rehydrate` access stores through
`ctx.extensions`.

## Dynamic reload

Config hot-reload is inherited from Praxis core. Pipelines
swap atomically; `ResponseStoreRegistry` is recreated per
pipeline build.

## Related

- [AGENTS.md](../../AGENTS.md) - agent and contributor reference
- [Core crate layout](https://github.com/praxis-proxy/praxis/blob/main/docs/architecture/crate-layout.md)
