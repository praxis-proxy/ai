# LLM gateway metrics

Praxis AI emits `praxis_ai_*` Prometheus metrics from the `token_count` and
`time_to_first_token` filters. The metrics are served on the admin metrics
listener and are intended for gateway operations, capacity planning, and
comparison with `vllm bench serve`.

Every metric carries:

- `model`: the server-published effective model. It comes from the format
  request filter or the `model_to_header` metadata; a raw client `X-Model`
  header is never used. The distinct-value set is capped at 1024 per process,
  with later values labeled `other`; EnMaaS routing overlays bound the model
  set in normal deployments.
- `cluster`: the selected routing cluster, or `unknown`.

## Metric catalog

| Metric | Type | Definition |
|---|---|---|
| `praxis_ai_prompt_tokens_total` | counter | Backend-reported prompt tokens per request. |
| `praxis_ai_output_tokens_total` | counter | Backend-reported output tokens per request. |
| `praxis_ai_prompt_tokens` | histogram | Prompt-token distribution per request. |
| `praxis_ai_output_tokens` | histogram | Output-token distribution per request. |
| `praxis_ai_reasoning_tokens_total` | counter | Backend-reported reasoning/thinking tokens. |
| `praxis_ai_cache_read_tokens_total` | counter | Prompt-cache read tokens reported by the backend. |
| `praxis_ai_finish_reasons_total` | counter | One normalized finish reason per successful response. |
| `praxis_ai_empty_completions_total` | counter | Successful zero-output responses, excluding `tool_calls`, `refusal`, and `content_filter` terminations. |
| `praxis_ai_ttft_seconds` | histogram | Server-side time from proxy request receipt to the first generated output delta. |
| `praxis_ai_tpot_seconds` | histogram | vLLM-style per-request time per output token, excluding the first token. |
| `praxis_ai_itl_seconds` | histogram | Time between consecutive generated output deltas. |

`token_count` records token, finish-reason, TPOT, and ITL metrics.
`time_to_first_token` records TTFT. Both act on successful responses; all
timing metrics are server-side.

## vLLM parity

The timing definitions are designed to match `vllm bench serve`:

- **TTFT**: vLLM records client-side time from request send to the first
  generated token chunk. Praxis records server-side time from request receipt
  to the first output delta. The event definition matches; the clock origin
  does not.
- **TPOT**: vLLM computes one value per request:

  ```text
  TPOT = (last output chunk time - first output chunk time)
         / (output_tokens - 1)
  ```

  `output_tokens` is `usage.completion_tokens` (with a tokenizer fallback in
  vLLM). Praxis uses the same formula with backend-reported usage. If usage is
  missing or `output_tokens <= 1`, Praxis skips the TPOT sample.
- **ITL**: vLLM records the interval between consecutive generated token
  chunks and reports the flattened distribution across requests. Praxis does
  the same with `praxis_ai_itl_seconds`; the number of ITL samples is the
  number of output-chunk intervals, not the number of requests.

Both vLLM and Praxis use chunk timestamps. A chunk may contain more than one
token, so TPOT and ITL are chunk-level approximations of per-token timing.

### Output-delta rules

The shared detector follows vLLM's OpenAI rules:

| API | Events that count |
|---|---|
| OpenAI Chat Completions | non-empty content; reasoning/`reasoning_content`; tool calls; function calls; empty content chunks without a role |
| OpenAI Responses | `response.output_text.delta`; `response.reasoning_text.delta` only |
| Anthropic Messages | `content_block_delta` with text, thinking, or partial JSON |
| Google Gemini | non-empty candidate text |
| Bedrock Converse | `contentBlockDelta` text or tool input |

SSE comments and role, message-start, created/in-progress, usage, and finish
events never count as output deltas. Anthropic/Google/Bedrock have no direct
`vllm bench serve` counterpart; their rules are analogous, not byte-for-byte
vLLM behavior.

## Query examples

Mean TPOT over requests:

```promql
sum(rate(praxis_ai_tpot_seconds_sum[5m]))
  /
sum(rate(praxis_ai_tpot_seconds_count[5m]))
```

P50/P99 TPOT:

```promql
histogram_quantile(0.50, sum by (le, cluster) (rate(praxis_ai_tpot_seconds_bucket[5m])))
histogram_quantile(0.99, sum by (le, cluster) (rate(praxis_ai_tpot_seconds_bucket[5m])))
```

Mean ITL over output intervals:

```promql
sum(rate(praxis_ai_itl_seconds_sum[5m]))
  /
sum(rate(praxis_ai_itl_seconds_count[5m]))
```

Mean TTFT:

```promql
sum(rate(praxis_ai_ttft_seconds_sum[5m]))
  /
sum(rate(praxis_ai_ttft_seconds_count[5m]))
```

`praxis_ai_tpot_seconds_count` counts requests with TPOT samples.
`praxis_ai_itl_seconds_count` counts output intervals.

## Comparing with vLLM

Run the vLLM benchmark against the same gateway and model:

```bash
vllm bench serve \
  --backend openai-chat \
  --base-url https://gateway.example.com \
  --model <model> \
  --num-prompts 1000
```

Compare its reported `mean_tpot_ms` and `mean_itl_ms` with the PromQL means
above over the same window. Expect close relative values, not identical
numbers: vLLM measures from the client and includes client/network overhead,
while Praxis measures inside the proxy and uses backend-reported token counts.

## Performance design

The timing path is designed to stay off the critical streaming path:

- Only output deltas are inspected; comments and metadata events are skipped.
- The detector parses one small SSE payload and does not retain a delta list.
- TTFT scans only the stream prefix and deactivates after the first output delta.
- TPOT/ITL state is a typed per-request extension with `Instant` timestamps.
- Model/cluster labels and `metrics::Histogram` handles are created once per
  stream; later ITL samples call the cached handle directly.
- TPOT is recorded once at stream finalization.

Performance gates:

- Hard ceiling: 50 microseconds of added p99 CPU per output delta.
- Expected target: less than 5 microseconds p99 per output delta.
- No allocations per output delta after the first delta creates the cached
  handles.
- No meaningful canary regression in p99 streaming latency or CPU per request.

The release-mode local check on this change measured a p99 of under 1 microsecond for output-delta detection and well under
1 microsecond for the cached ITL record with a no-op recorder. The production recorder and canary
add their own overhead; the 50-microsecond ceiling remains the binding gate.

If a future change exceeds the budget, optimize the detector or sample ITL at a
documented rate rather than silently slowing every stream.

## Limitations

- TTFT is server-side, so it is not directly comparable to client-side vLLM
  TTFT.
- TPOT requires backend usage; streams without reported output tokens have no
  TPOT sample.
- ITL is chunk-level, not exactly per generated token.
- Provider-specific output-delta rules differ where vLLM has no equivalent
  benchmark backend.
- Events larger than 64 KiB are treated as non-deltas to keep the hot path
  bounded; normal content deltas are far smaller.
- The `model` label is intentionally bounded; `unknown` means no model
  metadata and `other` means the admission cap was reached.
