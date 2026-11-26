# Vertex AI Anthropic from Messages

Forward Anthropic Messages requests to Anthropic models on Vertex AI through the
experimental Vertex Anthropic adapter.

## Prerequisites

Complete the [shared prerequisites and workflow](README.md#prerequisites-and-shared-workflow),
which also has the validate and start commands for `provider-local.yaml`.

## Configure, run, and verify

Build with the experimental adapter and copy the
[complete source configuration](../../../examples/configs/vertex/messages-to-anthropic.yaml):

```console
cargo build -p praxis-ai-proxy --features vertex-anthropic-filter
cp examples/configs/vertex/messages-to-anthropic.yaml provider-local.yaml
export GCP_ACCESS_TOKEN='<authorized-short-lived-access-token>'
```

In the copy, set the listener address to `127.0.0.1:8080`, set `project` and
the adapter `location`, and set both alias mappings' `target_model` to a Vertex
Anthropic model snapshot available in that project and location. For the
default `global` location, replace the `vertex` cluster endpoint with
`aiplatform.googleapis.com:443` and add a matching `tls.sni`; a regional
location uses its regional hostname. Keep the cluster's
`http.application_provider: vertex` metadata.

In the existing `credential_injection`, replace
`value: "ya29.example-static-token"` with `env_var: GCP_ACCESS_TOKEN`, and add
this header filter before it:

```yaml
      - filter: headers
        request_remove: [authorization, x-api-key]
        request_set:
          - name: Host
            value: "aiplatform.googleapis.com"
```

Use your regional hostname there if you chose one. Remove `fallback_upstream`
and the `fallback` cluster so unknown aliases do not go to the example's local
test backend, and test only the two configured aliases. Remove
`insecure_options` for the public endpoint. Validate, start, and send:

```console
curl --fail-with-body http://127.0.0.1:8080/v1/messages \
  -H 'content-type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"public-claude","max_tokens":128,"messages":[{"role":"user","content":"Hello"}]}'
```

Expect Messages output with the public model ID restored. The alias selector
picks the cluster and target model; the adapter moves the target model into
the `:rawPredict` or `:streamRawPredict` path and adapts the Vertex dialect
(it does not translate Messages to Chat). For SSE, add `"stream":true` and
`-N`. Beta-header handling and frame bounds are in the
[reference](../../filters/providers/anthropic_messages_to_vertexai_anthropic.md).
For automatic GCP token acquisition, also build `gcp-adc-filter` and replace
static injection with the cluster-scoped helper described in
[Vertex AI Gemini](vertex-gemini.md).

## Cleanup

Stop the proxy with Ctrl+C. In its shell, unset the credentials:

```console
unset GCP_ACCESS_TOKEN
```

Then finish the [shared cleanup](README.md#cleanup-and-qualification).

## Filter reference

- [`anthropic_messages_to_vertexai_anthropic`](../../filters/providers/anthropic_messages_to_vertexai_anthropic.md)
- [`gcp_adc`](../../filters/providers/gcp_adc.md)
- [Translation compatibility matrix](../../conformance/translation-matrix.md)
