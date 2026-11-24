# Azure OpenAI Chat Completions

Forward OpenAI Chat Completions requests to an Azure OpenAI deployment
through the Azure adapter, using a resource key or experimental Entra ID
authentication.

## Prerequisites

Complete the [shared prerequisites and workflow](README.md#prerequisites-and-shared-workflow)
for provider guides: account access, credentials, the validate/start commands
for `provider-local.yaml`, and how to read provider errors.

## Configure, run, and verify

Build standard and copy the full [Azure example](../../../examples/configs/azure/chat-completions-to-openai.yaml):

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
See the [adapter reference](../../filters/providers/openai_chat_completions_to_azureai_chat_completions.md)
and [matrix](../../conformance/translation-matrix.md).

For Entra client-secret authentication instead of a resource key, build with
`--features azure-ad-filter` and replace the key injection with the complete
[Entra settings](../../filters/providers/azure_ad.md), after routing. Configure tenant,
client, scope, and `AZURE_CLIENT_SECRET` in the proxy shell. This helper is
experimental and does not implement every Azure identity flow.

## Cleanup

Stop the proxy with Ctrl+C. In its shell, unset the credentials:

```console
unset AZURE_OPENAI_KEY AZURE_CLIENT_SECRET
```

Delete your disposable `provider-local.yaml`, and revoke credentials created
only for testing according to your account policy. See
[cleanup and qualification](README.md#cleanup-and-qualification) for what these
steps do and do not establish.

## Filter reference

- [`openai_chat_completions_to_azureai_chat_completions`](../../filters/providers/openai_chat_completions_to_azureai_chat_completions.md)
- [`azure_ad`](../../filters/providers/azure_ad.md)
- [Translation compatibility matrix](../../conformance/translation-matrix.md)
