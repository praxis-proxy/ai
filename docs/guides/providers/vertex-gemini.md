# Vertex AI Gemini from Chat Completions

Translate OpenAI Chat Completions requests to Vertex AI Gemini
`generateContent` and translate responses back.

## Prerequisites

Complete the [shared prerequisites and workflow](README.md#prerequisites-and-shared-workflow)
for provider guides: account access, credentials, the validate/start commands
for `provider-local.yaml`, and how to read provider errors.

## Configure, run, and verify

Build standard and copy the full [Gemini example](../../../examples/configs/vertex/chat-completions-to-gemini.yaml):

```console
cargo build -p praxis-ai-proxy
cp examples/configs/vertex/chat-completions-to-gemini.yaml provider-local.yaml
export GCP_ACCESS_TOKEN='<authorized-short-lived-access-token>'
```

Set the adapter's `project` and `region` to your project and selected region.
Replace the local load-balancer endpoint with
`<region>-aiplatform.googleapis.com:443` and add matching `tls.sni`. Between
`router` and `load_balancer`, insert:

```yaml
      - filter: headers
        request_remove: [authorization, x-api-key]
        request_set:
          - name: Host
            value: "<region>-aiplatform.googleapis.com"
      - filter: credential_injection
        clusters:
          - name: vertex-backend
            header: Authorization
            header_prefix: "Bearer "
            env_var: GCP_ACCESS_TOKEN
            strip_client_credential: true
```

Remove `insecure_options` for the public endpoint. Validate/start, then replace
the model placeholder and send:

```console
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"<available-gemini-model-id>","messages":[{"role":"user","content":"Hello"}]}'
```

Expect Chat Completions `choices`, mapped from Gemini `candidates`. The
adapter constructs the `generateContent` path; streaming uses
`streamGenerateContent` and returns Chat SSE. Add `"stream":true` and `-N`
to exercise that path. Review [schema, reasoning, and usage gaps](../../conformance/translation-matrix.md)
before enabling those features.

The environment bearer is for a short test and does not refresh itself.
For managed token acquisition, build with `--features gcp-adc-filter` and
replace core bearer injection with `gcp_adc`, scoped to `[vertex-backend]`,
after cluster selection. Use the complete [authentication example](../../../examples/configs/gcp-adc.yaml)
and [reference](../../filters/providers/gcp_adc.md) for metadata or service-account-key
sources. A `gcloud` user credential file is not a supported ADC source.

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

- [`openai_chat_completions_to_vertexai_gemini`](../../filters/providers/openai_chat_completions_to_vertexai_gemini.md)
- [`gcp_adc`](../../filters/providers/gcp_adc.md)
- [Translation compatibility matrix](../../conformance/translation-matrix.md)
