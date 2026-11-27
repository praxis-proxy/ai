# Guides

Task guides for operators. Each one gives prerequisites, a complete
configuration, a test request with its expected result, cleanup, and the filter
pages it uses. If you have not run Praxis AI yet, start with the
[Quickstart](../getting-started/quickstart.md).

The guides track Development source. Run commands from the repository root with
Git, Make, curl, and the [native build prerequisites](../developing/getting-started.md#requirements).
Each guide names the build it needs; the default published image does not ship
every optional filter. Use a model your backend actually serves.

| Goal | Guide | External prerequisites |
| --- | --- | --- |
| OpenAI-compatible Chat Completions | [Build from source](../getting-started/build-from-source.md#chat-completions), using its full YAML | OpenAI account, key, model access, outbound HTTPS |
| Native Anthropic Messages | [Anthropic guide](anthropic.md#native-messages) | Native Messages backend and backend key |
| Anthropic to Chat Completions | [Anthropic guide](anthropic.md#translated-messages) | Chat backend; translation limitations apply |
| Stable model alias and provider selection | [Routing](routing.md) | Two compatible local backends |
| Azure OpenAI Chat Completions | [Azure OpenAI](providers/azure-openai.md) | Azure resource, deployment, key or Entra app, outbound HTTPS |
| Vertex AI Gemini from Chat Completions | [Vertex AI Gemini](providers/vertex-gemini.md) | GCP project, model access, access token or ADC, outbound HTTPS |
| Vertex AI Anthropic from Messages | [Vertex AI Anthropic](providers/vertex-anthropic.md) | GCP project, Anthropic model access on Vertex, outbound HTTPS |
| Amazon Bedrock Converse from Chat Completions | [Amazon Bedrock](providers/bedrock.md) | AWS account, model access, SigV4 credentials, outbound HTTPS |
| Request guardrails | [Screen requests with guardrails](guardrails.md) | NeMo service and OpenAI-compatible backend |
| Usage metrics and metering | [Observe and meter token usage](token-usage.md) | OpenAI-compatible backend; metering service for balance checks |
| Token budgets and ceilings | [Limit token usage](token-limits.md) | Feature-enabled build; budget backend for rate limiting |
| Managed Responses and tool execution | [Experimental managed path](managed-responses.md) | Inference and selected tool services; writable store |

Shared provider prerequisites are in [Connect cloud provider adapters](providers/README.md).

## Credentials

Keep three boundaries separate:

1. **Caller to gateway:** authenticate callers before they can use
   gateway-owned provider secrets. The local Anthropic recipes use optional
   Basic Auth; public deployments also need TLS and an ingress policy.
2. **Gateway to inference:** native forwarding can pass the caller's provider
   key on purpose ([quickstart](../getting-started/quickstart.md)). For a
   gateway-owned key, use cluster-selected core `credential_injection` as in the
   [Anthropic guide](../anthropic-messages.md#passthrough-to-anthropic-api), or
   selected-profile [credential_inject](../filters/providers/credential_inject.md)
   for dynamic provider routing. Remove both `x-api-key` and caller
   `authorization` where applicable, because SDK versions use different
   headers. An internal-looking header is not identity.
3. **Gateway to tools or policy services:** configure a destination-bound
   outbound chain. Never attach a blanket static secret to arbitrary
   client-selected MCP URLs. See [callout security](../architecture/outbound-callouts.md)
   and [scoped credentials](../filters/providers/callout_credentials.md).

Set placeholders such as `<backend-key>` in the proxy's environment; setting a
variable in the curl shell does not reach a proxy that is already running. Do
not commit keys or capture them in shared debug logs.

## Validation and cleanup

`--validate` must exit successfully before you start the proxy. Run one guide
at a time, since they all use port 8080. When you finish, stop the proxy with
Ctrl+C, stop any backend you started for it, unset its secret variables, and
delete generated local configuration or databases you no longer need. Leave
committed example YAML intact.
