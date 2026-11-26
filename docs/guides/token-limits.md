# Limit token usage

Bound the size of provider-bound requests and reserve token budgets before
inference. [Usage extraction](token-usage.md) only observes what the provider
reports and enforces nothing.

## Choose a control

| Goal | Filter | Complete example |
| --- | --- | --- |
| Reject oversized input or unbounded output | [`token_ceiling`](../filters/control/token_ceiling.md) | [token-ceiling.yaml](../../examples/configs/token-ceiling.yaml) |
| Reserve and reconcile token budgets | [`token_rate_limit`](../filters/control/token_rate_limit.md) | [token-rate-limit.yaml](../../examples/configs/token-rate-limit.yaml) |
| Track live requests and reserved tokens per model | [`inflight_tracker`](../filters/control/inflight_tracker.md) | [inflight-tracking.yaml](../../examples/configs/inflight-tracking.yaml) |
| Weight cache and reasoning tokens in a quota | [PPE token quota plugin](../filters/control/token_quota.md) | [token-quota.yaml](../../examples/configs/token-quota.yaml) |

`token_ceiling` evaluates the final provider-bound body. Experimental
`token_rate_limit` reserves estimated budgets, reconciles them afterward, and
needs its configured budget backend and trusted key dimensions. Each filter
requires its own [build features](../features.md#cargo-features); check them
before choosing a binary, and run `praxis-ai --validate -c FILE` with that
binary.

## Related

- [Filter capability overview](../filters/README.md)
