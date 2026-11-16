# Route models and providers

Choose one routing owner. A static alias does not require a provider registry;
a dynamic provider catalog does. Neither choosing an endpoint nor changing the
model string translates a protocol.

## Start with a static alias

Follow the [recipe prerequisites](README.md) and build the standard binary:

```console
cargo build -p praxis-ai-proxy
```

Use the [complete model-to-provider example](../../examples/configs/model-to-provider.yaml).
It maps `claude-sonnet-4-5` on `/v1/messages` and `/v1/messages/count_tokens`
to cluster `vertex`, rewriting the body model to
`claude-sonnet-4-5@20250929`. Unmatched models/paths go to `fallback` unchanged.
The cluster name `vertex` is just a name here: this example does **not**
configure Vertex authentication or endpoint translation.

Start two backends on `127.0.0.1:9001` and `127.0.0.1:9002`. For inference,
they must implement the client protocol and serve the appropriate target
models. Recording HTTP stubs can instead demonstrate selection and rewriting,
without claiming successful model inference. This local example uses no
credentials; for real providers add cluster-selected credential injection
after selection and configure their Host, TLS/SNI, and protocol adapter.

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

Expect the first backend to receive the rewritten target model. Repeat with
`"model":"unmapped-model"` to exercise the second backend; it receives that
model unchanged. Inspect backend recordings to distinguish selection from a
provider error about an unavailable model. The response retains the target
model unless a provider response adapter restores the public ID.

Keep `model_to_provider` before `load_balancer`; do not add a competing router.
Endpoint health checks and retries belong to the load balancer. An unmatched
request uses `fallback_upstream`; that setting does not mean automatic retry
of a failed mapped inference on another provider. See the
[exact contract](../filters/model_to_provider.md).

## Dynamic selection, failover, and affinity

For a changing provider catalog, start with the complete
[intelligent routing example](../../examples/configs/intelligent-route-inference.yaml)
or [overlay configuration](../../examples/configs/intelligent-route-overlay.yaml).
Use `model_to_header` before `intelligent_route`, selected-provider adapters
where needed, then `credential_inject` and endpoint selection. The
[projected-secret example](../../examples/configs/intelligent-route-projected-credentials.yaml)
shows gateway-owned credentials without embedding keys in YAML. This is a
different credential mechanism from core `credential_injection`.

Read the [routing reference](../filters/intelligent_route.md) before deploying:
capability matching, admission, serving snapshots, and selection policy all
constrain eligibility. Session affinity is disabled by default. When enabled,
it requires `store: local` plus a header or cookie source. Bindings live in
one filter instance and do not survive restart or coordinate replicas. If a
bound candidate becomes ineligible, a new eligible candidate can be selected;
if none is available, the documented result is 404. This is selection failover,
not a promise to replay an already-started generation or recover a partially
emitted stream.

For provider-local routing, the [provider-route example](../../examples/configs/provider-route.yaml)
requires downstream mTLS and `peer_identity_trust` before `provider_route`.
Never trust routing assertions from an unauthenticated client.

## Limits and cleanup

Use compatible backend protocols on a shared route, or install an explicitly
supported adapter. Streaming requires backend support and the appropriate SSE
adapter for translated routes. Route success does not establish tool support,
credential validity, or cross-provider semantic equivalence.

Stop the proxy and the two test backends with Ctrl+C. Remove only your local
recordings and temporary configurations. If you added provider keys, unset
them in the proxy shell. The committed example remains unchanged.
