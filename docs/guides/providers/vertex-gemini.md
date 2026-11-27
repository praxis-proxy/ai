# Vertex AI Gemini from Chat Completions

Translate OpenAI Chat Completions requests to Vertex AI Gemini
`generateContent` and translate responses back.

## Prerequisites

Complete the [shared prerequisites and workflow](README.md#prerequisites-and-shared-workflow),
which also has the validate and start commands for `provider-local.yaml`.

## Configure, run, and verify

Build the standard binary and copy the full [Gemini example](../../../examples/configs/vertex/chat-completions-to-gemini.yaml):

```console
cargo build -p praxis-ai-proxy
cp examples/configs/vertex/chat-completions-to-gemini.yaml provider-local.yaml
export GCP_ACCESS_TOKEN='<authorized-short-lived-access-token>'
```

Set the adapter's `project` and `region`. Replace the local load-balancer
endpoint with `<region>-aiplatform.googleapis.com:443` and add a matching
`tls.sni`. Between `router` and `load_balancer`, insert:

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

Remove `insecure_options` for the public endpoint. Validate and start, then
replace the model placeholder and send:

```console
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"<available-gemini-model-id>","messages":[{"role":"user","content":"Hello"}]}'
```

Expect Chat Completions `choices` mapped from Gemini `candidates`. The adapter
builds the `generateContent` path, or `streamGenerateContent` for streaming,
which returns Chat SSE; add `"stream":true` and `-N` to try it. Check the
[schema, reasoning, and usage gaps](../../conformance/translation-matrix.md)
before relying on those features.

The environment bearer suits a short test and never refreshes. For managed
token acquisition, build with `--features gcp-adc-filter` and replace core
bearer injection with `gcp_adc` scoped to `[vertex-backend]`, after cluster
selection. The [authentication example](../../../examples/configs/gcp-adc.yaml)
and [reference](../../filters/providers/gcp_adc.md) cover metadata and
service-account-key sources. A `gcloud` user credential file is not a
supported ADC source.

## Cleanup

Stop the proxy with Ctrl+C. In its shell, unset the credentials:

```console
unset GCP_ACCESS_TOKEN
```

Then finish the [shared cleanup](README.md#cleanup-and-qualification).

## Filter reference

- [`openai_chat_completions_to_vertexai_gemini`](../../filters/providers/openai_chat_completions_to_vertexai_gemini.md)
- [`gcp_adc`](../../filters/providers/gcp_adc.md)
- [Translation compatibility matrix](../../conformance/translation-matrix.md)
