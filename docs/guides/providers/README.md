# Connect cloud provider adapters

Each guide starts from a complete committed configuration. Copy and adapt that
file rather than building from an isolated adapter fragment. The
[path selector](../../getting-started/choose-client-and-provider.md) explains
native forwarding, protocol translation, and the experimental authentication
helpers. These guides describe Development, not the published v0.10.0 snapshot.

## Provider guides

| Provider path | Guide | Build |
| --- | --- | --- |
| Azure OpenAI Chat Completions | [Azure OpenAI](azure-openai.md) | Standard; `azure-ad-filter` for Entra ID |
| Vertex AI Gemini from Chat Completions | [Vertex AI Gemini](vertex-gemini.md) | Standard; `gcp-adc-filter` for ADC |
| Vertex AI Anthropic from Messages | [Vertex AI Anthropic](vertex-anthropic.md) | `vertex-anthropic-filter` (experimental) |
| Amazon Bedrock Converse from Chat Completions | [Amazon Bedrock](bedrock.md) | Standard |

## Prerequisites and shared workflow

Follow the [guide prerequisites](../README.md). You need a provider
account, project, or resource with the inference service enabled, permission to
invoke the chosen deployment or model, outbound DNS and HTTPS, and trusted CA
certificates. You arrange account provisioning and model access with the
provider; Praxis authentication filters do not grant them.

The guides use short-lived test credentials or keys from environment variables.
Replace every placeholder, and run one configuration at a time on port 8080.
The listener is a local test gateway: put authenticated ingress in front of it
before exposing gateway-owned credentials remotely.

After making a guide's edits to your copy, validate and start it:

```console
./target/debug/praxis-ai --validate -c provider-local.yaml
./target/debug/praxis-ai -c provider-local.yaml
```

Send the guide's request from another shell. Success shows that request works
on that model or deployment, not that every provider API does. Invalid keys,
permissions, quotas, models, or regions produce provider errors, so read the
response body before suspecting the proxy.

## Cleanup and qualification

Stop the proxy with Ctrl+C and unset the variables you set in its shell:

```console
unset AZURE_OPENAI_KEY AZURE_CLIENT_SECRET GCP_ACCESS_TOKEN
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
```

Delete `provider-local.yaml` and revoke any credentials created only for
testing, following your account policy. Updating a token in another shell does
not refresh the one in the proxy's environment; refresh or restart the pipeline
from its credential source.

These guides do not provision cloud accounts, grant provider permissions, or
certify a provider or model. The
[audit](../../developing/documentation-audit.md#verification-for-this-change)
records which configurations were validated and which runtime paths were
exercised; no live cloud request was made when these guides were written.
