# Limit token usage

Bound the size of provider-bound requests and reserve token budgets before
inference. These controls are separate from usage extraction, which only
observes what the provider reports.

## Choose a control

Usage extraction does not enforce token limits. Experimental
[token_rate_limit](../filters/control/token_rate_limit.md) reserves estimated budgets
and reconciles them; [token_ceiling](../filters/control/token_ceiling.md) constrains
the final provider-bound body. Both require their own features and policies.

| Goal | Filter | Complete example |
| --- | --- | --- |
| Reject oversized input or unbounded output | [`token_ceiling`](../filters/control/token_ceiling.md) | [token-ceiling.yaml](../../examples/configs/token-ceiling.yaml) |
| Reserve and reconcile token budgets | [`token_rate_limit`](../filters/control/token_rate_limit.md) | [token-rate-limit.yaml](../../examples/configs/token-rate-limit.yaml) |
| Track live requests and reserved tokens per model | [`inflight_tracker`](../filters/control/inflight_tracker.md) | [inflight-tracking.yaml](../../examples/configs/inflight-tracking.yaml) |
| Weight cache and reasoning tokens in a quota | [PPE token quota plugin](../filters/control/token_quota.md) | [token-quota.yaml](../../examples/configs/token-quota.yaml) |

The ceiling evaluates the final provider-bound body; rate limiting needs its
configured budget backend and trusted key dimensions. Check the
[build features](../features.md#cargo-features) each filter requires before
selecting a binary, and run `praxis-ai --validate -c FILE` with that binary.

## Related

- [Observe and meter token usage](token-usage.md)
- [Filter capability overview](../filters/README.md)
