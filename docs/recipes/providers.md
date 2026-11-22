# Connect cloud provider adapters

Use this runbook with the complete committed configurations linked below.
Copy and adapt those files; do not replace them with an isolated adapter
fragment. The [path selector](../client-provider-paths.md) distinguishes native
forwarding, protocol translation, and experimental authentication helpers.
These instructions describe Development, not the published v0.10.0 snapshot.

## Prerequisites and shared workflow

Follow the [recipe prerequisites](README.md). You need an existing provider
account/project/resource, enabled inference service, permission to invoke the
chosen deployment/model, outbound DNS/HTTPS, and trusted CA certificates.
Account provisioning and model access approval are provider-side prerequisites;
Praxis authentication filters do not grant them.

The examples below use short-lived test credentials or environment-backed keys.
Replace every placeholder before starting. Run only one configuration at a
time on port 8080. The resulting listener is a local test gateway: configure
authenticated ingress before exposing gateway-owned credentials remotely.

For each copied configuration, after the edits in its section:

```console
./target/debug/praxis-ai --validate -c provider-local.yaml
./target/debug/praxis-ai -c provider-local.yaml
```

Run the selected section's request in another shell. A successful response
proves that request on that selected model/deployment, not every provider API.
Expect provider errors for invalid keys, permissions, quotas, models, or regions.
Review the response body before diagnosing a proxy fault.

## Azure OpenAI Chat Completions

Build standard and copy the full [Azure example](../../examples/configs/azure/chat-completions-to-openai.yaml):

```console
cargo build -p praxis-ai-proxy
cp examples/configs/azure/chat-completions-to-openai.yaml provider-local.yaml
export AZURE_OPENAI_KEY='<resource-api-key>'
```

Edit the copy:

1. Replace `gpt-4o` in the `path_rewrite` deployment path with your **deployment
   name**, and choose an API version supported by that deployment in the
   Azure adapter. Deployment names and model names are not interchangeable.
2. Replace the load-balancer cluster with the following, substituting your
   resource hostname consistently. Keep the `path_rewrite` before the adapter.

```yaml
      - filter: load_balancer
        clusters:
          - name: azure-backend
            endpoints: ["<resource>.openai.azure.com:443"]
            tls:
              sni: "<resource>.openai.azure.com"
```

3. Between `router` and `load_balancer`, insert:

```yaml
      - filter: headers
        request_remove: [authorization, x-api-key]
        request_set:
          - name: Host
            value: "<resource>.openai.azure.com"
      - filter: credential_injection
        clusters:
          - name: azure-backend
            header: api-key
            env_var: AZURE_OPENAI_KEY
            strip_client_credential: true
```

Remove the local example's `insecure_options` when using a public endpoint.
Validate/start using the shared commands, then test:

```console
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"deployment-is-selected-in-config","messages":[{"role":"user","content":"Hello"}]}'
```

Expect Chat Completions `choices` and usage. The adapter removes the body model
and uses the configured deployment path, adds `api-version`, and normalizes
responses. SSE is handled by the same adapter: add `"stream":true` and `-N`.
Azure safety annotations are discarded; do not rely on their preservation.
See the [adapter reference](../filters/openai_chat_completions_to_azureai_chat_completions.md)
and [matrix](../conformance/translation-matrix.md).

For Entra client-secret authentication instead of a resource key, build with
`--features azure-ad-filter` and replace the key injection with the complete
[Entra settings](../filters/azure_ad.md), after routing. Configure tenant,
client, scope, and `AZURE_CLIENT_SECRET` in the proxy shell. This helper is
experimental and does not implement every Azure identity flow.

## Vertex Gemini from Chat Completions

Build standard and copy the full [Gemini example](../../examples/configs/vertex/chat-completions-to-gemini.yaml):

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
to exercise that path. Review [schema, reasoning, and usage gaps](../conformance/translation-matrix.md)
before enabling those features.

The environment bearer is for a short test and does not refresh itself.
For managed token acquisition, build with `--features gcp-adc-filter` and
replace core bearer injection with `gcp_adc`, scoped to `[vertex-backend]`,
after cluster selection. Use the complete [authentication example](../../examples/configs/gcp-adc.yaml)
and [reference](../filters/gcp_adc.md) for metadata or service-account-key
sources. A `gcloud` user credential file is not a supported ADC source.

## Vertex Anthropic from Messages

This adapter is experimental:

```console
cargo build -p praxis-ai-proxy --features vertex-anthropic-filter
cp examples/configs/vertex/messages-to-anthropic.yaml provider-local.yaml
export GCP_ACCESS_TOKEN='<authorized-short-lived-access-token>'
```

Use the [complete source configuration](../../examples/configs/vertex/messages-to-anthropic.yaml).
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
`"stream":true` and `-N`. Follow the [reference](../filters/anthropic_messages_to_vertexai_anthropic.md)
for beta-header and frame bounds. This is not Messages-to-Chat translation.
For automatic GCP token acquisition, additionally build `gcp-adc-filter` and
use the cluster-scoped helper described above instead of static injection.

## Bedrock Converse from Chat Completions

Build standard and copy the full [Bedrock example](../../examples/configs/bedrock/chat-completions-to-converse.yaml):

```console
cargo build -p praxis-ai-proxy
cp examples/configs/bedrock/chat-completions-to-converse.yaml provider-local.yaml
export AWS_ACCESS_KEY_ID='<authorized-access-key-id>'
export AWS_SECRET_ACCESS_KEY='<secret-access-key>'
```

Choose a Bedrock Runtime region with access to your requested model. Edit the
listener to loopback, set the signer's `region`, and replace its `host` with
`bedrock-runtime.<region>.amazonaws.com`. Replace the local cluster endpoint
with that same hostname on port 443 and add matching `tls.sni`. Keep the
translator before routing and the signer after load balancing as in the full
example, so it signs the translated body/path. Remove local `insecure_options`
for a public endpoint.

The signer reads explicitly configured environment sources; it does not use
an ambient AWS credential chain. If your credentials include a session token,
set `session_token_env_var: AWS_SESSION_TOKEN` on `aws_sigv4_sign` and export
`AWS_SESSION_TOKEN='<session-token>'` in the proxy shell. Follow the
[signer reference](../filters/aws_sigv4_sign.md), keep credentials limited to
the intended service, and refresh them before expiry. Signing alone does not
implement Bedrock protocol translation.

Validate/start, then send:

```console
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"<available-bedrock-model-id>","messages":[{"role":"user","content":"Hello"}]}'
```

Expect Chat Completions JSON, mapped from Converse. With `"stream":true`
and `curl -N`, the adapter decodes AWS EventStream frames into Chat SSE and a
terminal `[DONE]`. Reasoning, cache accounting, and history gaps are documented
in the [matrix](../conformance/translation-matrix.md). The separate
[generated-reference omission](../documentation-audit.md#remaining-gaps-and-separate-blockers)
is a documentation tooling blocker, not proof of absent implementation.

## Cleanup and qualification

Stop the proxy with Ctrl+C. Unset the applicable variables in its shell:

```console
unset AZURE_OPENAI_KEY AZURE_CLIENT_SECRET GCP_ACCESS_TOKEN
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
```

Delete your disposable `provider-local.yaml`, and revoke credentials created
only for testing according to your account policy. A token copied into the
proxy environment is not refreshed by updating a different shell; refresh
or restart the pipeline with its credential source as documented.

These adaptations provide configuration/start/request instructions. They do
not provision cloud accounts, establish provider permissions, or certify a
provider/model. The [audit](../documentation-audit.md#verification-for-this-change)
records which configurations were validated and which runtime paths were
actually exercised. No live cloud request was made for this documentation change.
