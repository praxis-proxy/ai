# Azure OpenAI Chat Completions

Forward OpenAI Chat Completions requests to an Azure OpenAI deployment through
the Azure adapter, authenticating with a resource key or the experimental
Entra ID helper.

## Prerequisites

Complete the [shared prerequisites and workflow](README.md#prerequisites-and-shared-workflow),
which also has the validate and start commands for `provider-local.yaml`.

## Configure, run, and verify

Build the standard binary and copy the full [Azure example](../../../examples/configs/azure/chat-completions-to-openai.yaml):

```console
cargo build -p praxis-ai-proxy
cp examples/configs/azure/chat-completions-to-openai.yaml provider-local.yaml
export AZURE_OPENAI_KEY='<resource-api-key>'
```

Edit the copy:

1. Replace `gpt-4o` in the `path_rewrite` deployment path with your deployment
   name (not the model name), and set the Azure adapter's API version to one
   that deployment supports.
2. Replace the load-balancer cluster with the following, using your resource
   hostname throughout. Keep `path_rewrite` before the adapter.

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

Remove the example's `insecure_options` for the public endpoint. Validate and
start with the shared commands, then test:

```console
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"deployment-is-selected-in-config","messages":[{"role":"user","content":"Hello"}]}'
```

Expect Chat Completions `choices` and usage. The adapter drops the body model
in favor of the configured deployment path, adds `api-version`, and normalizes
responses. It also handles SSE: add `"stream":true` and `-N`. Azure safety
annotations are discarded. See the
[adapter reference](../../filters/providers/openai_chat_completions_to_azureai_chat_completions.md)
and [matrix](../../conformance/translation-matrix.md).

To use an Entra client secret instead of a resource key, build with
`--features azure-ad-filter` and replace the key injection, after routing, with
the complete [Entra settings](../../filters/providers/azure_ad.md). Set tenant,
client, scope, and `AZURE_CLIENT_SECRET` in the proxy shell. The helper is
experimental and does not cover every Azure identity flow.

## Cleanup

Stop the proxy with Ctrl+C. In its shell, unset the credentials:

```console
unset AZURE_OPENAI_KEY AZURE_CLIENT_SECRET
```

Then finish the [shared cleanup](README.md#cleanup-and-qualification).

## Filter reference

- [`openai_chat_completions_to_azureai_chat_completions`](../../filters/providers/openai_chat_completions_to_azureai_chat_completions.md)
- [`azure_ad`](../../filters/providers/azure_ad.md)
- [Translation compatibility matrix](../../conformance/translation-matrix.md)
