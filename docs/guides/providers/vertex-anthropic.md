# Vertex AI Anthropic from Messages

Forward Anthropic Messages requests to Anthropic models on Vertex AI with the
experimental Vertex Anthropic adapter.

## Prerequisites

Complete the [shared prerequisites and workflow](README.md#prerequisites-and-shared-workflow)
for provider guides: account access, credentials, the validate/start commands
for `provider-local.yaml`, and how to read provider errors.

## Configure, run, and verify

This adapter is experimental:

```console
cargo build -p praxis-ai-proxy --features vertex-anthropic-filter
cp examples/configs/vertex/messages-to-anthropic.yaml provider-local.yaml
export GCP_ACCESS_TOKEN='<authorized-short-lived-access-token>'
```

Use the [complete source configuration](../../../examples/configs/vertex/messages-to-anthropic.yaml).
Edit the listener address to `127.0.0.1:8080`. Set `project`, choose the
adapter `location`, and replace both alias mappings' `target_model` values
with a Vertex Anthropic model snapshot available in that project/location.
For the default `global` location, replace the `vertex` cluster endpoint with
`aiplatform.googleapis.com:443` and add matching `tls.sni`. For a regional
location use its matching regional hostname. Retain the cluster's
`http.application_provider: vertex` metadata.

In the existing `credential_injection`, replace
`value: "ya29.example-static-token"` with `env_var: GCP_ACCESS_TOKEN`. Add
this header filter before credential injection:

```yaml
      - filter: headers
        request_remove: [authorization, x-api-key]
        request_set:
          - name: Host
            value: "aiplatform.googleapis.com"
```

Use your regional hostname there if applicable. Remove `fallback_upstream`
and the `fallback` cluster to avoid sending unknown aliases to the original
local test backend; test only the two configured aliases. Remove the public
endpoint's local `insecure_options`. Validate/start and request:

```console
curl --fail-with-body http://127.0.0.1:8080/v1/messages \
  -H 'content-type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"public-claude","max_tokens":128,"messages":[{"role":"user","content":"Hello"}]}'
```

Expect Messages output with the public model ID restored. The alias selector
owns cluster/target-model selection; the adapter moves the target model into
`:rawPredict` or `:streamRawPredict` and adapts the Vertex dialect. For SSE add
`"stream":true` and `-N`. Follow the [reference](../../filters/providers/anthropic_messages_to_vertexai_anthropic.md)
for beta-header and frame bounds. This is not Messages-to-Chat translation.
For automatic GCP token acquisition, additionally build `gcp-adc-filter` and
use the cluster-scoped helper described in [Vertex AI Gemini](vertex-gemini.md)
instead of static injection.

## Cleanup

Stop the proxy with Ctrl+C. In its shell, unset the credentials:

```console
unset GCP_ACCESS_TOKEN
```

Delete your disposable `provider-local.yaml`, and revoke credentials created
only for testing according to your account policy. See
[cleanup and qualification](README.md#cleanup-and-qualification) for what these
steps do and do not establish.

## Filter reference

- [`anthropic_messages_to_vertexai_anthropic`](../../filters/providers/anthropic_messages_to_vertexai_anthropic.md)
- [`gcp_adc`](../../filters/providers/gcp_adc.md)
- [Translation compatibility matrix](../../conformance/translation-matrix.md)
