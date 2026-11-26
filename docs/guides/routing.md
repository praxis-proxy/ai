# Route models and providers

Pick one filter to own routing: a static alias needs no provider registry, and
a dynamic provider catalog does. Routing selects an endpoint and may rewrite
the model string; neither translates the protocol.

## Start with a static alias

Follow the [guide prerequisites](README.md) and build the standard binary:

```console
cargo build -p praxis-ai-proxy
```

The [model-to-provider example](../../examples/configs/model-to-provider.yaml)
maps `claude-sonnet-4-5` on `/v1/messages` and `/v1/messages/count_tokens` to
cluster `vertex` and rewrites the body model to `claude-sonnet-4-5@20250929`.
Unmatched models and paths go to `fallback` unchanged. `vertex` is only a
cluster name: the example configures no Vertex authentication or endpoint
translation.

Start two backends on `127.0.0.1:9001` and `127.0.0.1:9002`. For real
inference they must speak the client protocol and serve the target models;
recording HTTP stubs are enough to show selection and rewriting. The example
uses no credentials. For real providers, add cluster-selected credential
injection after selection and configure Host, TLS/SNI, and the protocol adapter.

```console
./target/debug/praxis-ai --validate -c examples/configs/model-to-provider.yaml
./target/debug/praxis-ai -c examples/configs/model-to-provider.yaml
```

In another shell:

```console
curl --fail-with-body http://127.0.0.1:8080/v1/messages \
  -H 'content-type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"claude-sonnet-4-5","max_tokens":64,"messages":[{"role":"user","content":"Hello"}]}'
```

The first backend should receive the rewritten target model. Repeat with
`"model":"unmapped-model"` and the second backend receives that model
unchanged. Check backend recordings to tell a selection problem from a provider
error about an unavailable model. The response carries the target model unless
a provider response adapter restores the public ID.

Keep `model_to_provider` before `load_balancer`, with no competing router. The
load balancer owns endpoint health checks and retries. `fallback_upstream`
only handles unmatched requests; it does not retry a failed mapped inference on
another provider. See the [filter contract](../filters/routing/model_to_provider.md).

## Dynamic selection, failover, and affinity

For a changing provider catalog, start from the
[intelligent routing example](../../examples/configs/intelligent-route-inference.yaml)
or the [overlay configuration](../../examples/configs/intelligent-route-overlay.yaml).
The order is `model_to_header`, then `intelligent_route`, then selected-provider
adapters where needed, then `credential_inject` and endpoint selection. The
[projected-secret example](../../examples/configs/intelligent-route-projected-credentials.yaml)
supplies gateway-owned credentials without keys in YAML; `credential_inject` is
a different mechanism from core `credential_injection`.

Before deploying, read the [routing reference](../filters/routing/intelligent_route.md):
capability matching, admission, serving snapshots, and selection policy all
limit which candidates are eligible. Session affinity is off by default and,
when enabled, requires `store: local` plus a header or cookie source. Bindings
live in one filter instance, so they are lost on restart and not shared across
replicas. If a bound candidate becomes ineligible, another eligible one can be
selected; with none left the request gets 404. This failover happens at
selection time only and does not replay a started generation or recover a
partly emitted stream.

For provider-local routing, the [provider-route example](../../examples/configs/provider-route.yaml)
requires downstream mTLS and `peer_identity_trust` before `provider_route`.
Never trust routing assertions from an unauthenticated client.

## Limits and cleanup

Backends on a shared route must speak compatible protocols, or you install a
supported adapter. Streaming needs backend support, plus the matching SSE
adapter on translated routes. A successful route says nothing about tool
support, credential validity, or whether two providers behave the same.

Stop the proxy and both test backends with Ctrl+C, remove your local recordings
and temporary configurations, and unset any provider keys in the proxy shell.

## Filter reference

- [`model_to_provider`](../filters/routing/model_to_provider.md)
- [`model_to_header`](../filters/routing/model_to_header.md)
- [`intelligent_route`](../filters/routing/intelligent_route.md)
- [`credential_inject`](../filters/providers/credential_inject.md)
- [`provider_route`](../filters/routing/provider_route.md)

Core filters such as `router`, `headers`, `credential_injection`, and
`load_balancer` are documented in the [Praxis core filter reference](https://github.com/praxis-proxy/praxis/blob/main/docs/filters/reference.md).
