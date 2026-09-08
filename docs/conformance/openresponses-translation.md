# OpenResponses Translation Conformance

This suite runs the external [OpenResponses](https://github.com/openresponses/openresponses)
conformance oracle against the `responses_to_chat_completions` translation
filter — never against native Responses passthrough.

## What runs

A Praxis listener loads the shared
`examples/configs/openai/responses/responses-to-chat-completions.yaml`
example, which translates `POST /v1/responses` into `POST /v1/chat/completions`
and forwards to a Chat-Completions-only backend (vLLM CPU, `Qwen/Qwen3-0.6B`).
A translation-witness shim sits between Praxis and vLLM and asserts the
backend only ever receives `POST /v1/chat/completions` (criterion f).

## Pins

- Suite commit: `92c12d96d7b61d6d15e2214daa5e9c6000ab6e1c`
- Runner: bun `1.3.12`, zod `3.25.76`
- Backend: vLLM CPU, `Qwen/Qwen3-0.6B`

## Triage

The applicable/skip decision for every one of the suite's 17 templates lives
in [`tests/conformance/openresponses/manifest.yaml`](../../tests/conformance/openresponses/manifest.yaml).
The launcher enumerates the suite and fails if any template is untriaged.

- **Applicable (6):** basic-response, system-prompt, multi-turn,
  assistant-phase, tool-calling, streaming-response.
- **Skipped (11):** image-input (vision backend), 7 websocket-* (non-HTTP
  transport), compact-response / compact-missing-model (/responses/compact
  endpoint), response-output-phase-schema (no capability at this SHA).

## Running locally

```console
make test-responses-conformance
```

Requires `bun`, `uv`, and a reachable vLLM backend (`VLLM_BASE_URL`,
default `http://127.0.0.1:8000`).
