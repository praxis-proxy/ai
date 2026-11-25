# Connect cloud provider adapters

Use these guides with the complete committed configurations they link.
Copy and adapt those files; do not replace them with an isolated adapter
fragment. The [path selector](../../getting-started/choose-client-and-provider.md) distinguishes native
forwarding, protocol translation, and experimental authentication helpers.
These instructions describe Development, not the published v0.10.0 snapshot.

## Provider guides

| Provider path | Guide | Build |
| --- | --- | --- |
| Azure OpenAI Chat Completions | [Azure OpenAI](azure-openai.md) | Standard; `azure-ad-filter` for Entra ID |
| Vertex AI Gemini from Chat Completions | [Vertex AI Gemini](vertex-gemini.md) | Standard; `gcp-adc-filter` for ADC |
| Vertex AI Anthropic from Messages | [Vertex AI Anthropic](vertex-anthropic.md) | `vertex-anthropic-filter` (experimental) |
| Amazon Bedrock Converse from Chat Completions | [Amazon Bedrock](bedrock.md) | Standard |

## Prerequisites and shared workflow

Follow the [guide prerequisites](../README.md). You need an existing provider
account/project/resource, enabled inference service, permission to invoke the
chosen deployment/model, outbound DNS/HTTPS, and trusted CA certificates.
Account provisioning and model access approval are provider-side prerequisites;
Praxis authentication filters do not grant them.

The provider guides use short-lived test credentials or environment-backed keys.
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
provider/model. The [audit](../../developing/documentation-audit.md#verification-for-this-change)
records which configurations were validated and which runtime paths were
actually exercised. No live cloud request was made for this documentation change.
