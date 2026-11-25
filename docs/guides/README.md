# Guides

Task guides for operators. Each guide lists its prerequisites, a complete
configuration, validation and startup, a request with its expected result,
cleanup, and the filter reference pages it uses. Start with the
[Quickstart](../getting-started/quickstart.md) if you have not run Praxis AI yet.

These guides follow Development source. Run commands from the repository root
with Git, Make, curl, and the [native build prerequisites](../developing/getting-started.md#requirements).
Each guide selects its build explicitly. The default published image does not
supply every optional filter; build the requested binary rather than assuming
an image contains it. Choose a model that your backend actually serves.

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

1. **Caller to gateway:** authenticate callers before allowing them to use
   gateway-owned provider secrets. The local Anthropic recipes use optional
   Basic Auth. Public deployments also need TLS and a suitable ingress policy.
2. **Gateway to inference:** native forwarding can intentionally pass the
   caller's provider key ([quickstart](../getting-started/quickstart.md)). A gateway-owned key
   instead needs cluster-selected core `credential_injection` as in the
   [Anthropic guide](../anthropic-messages.md#passthrough-to-anthropic-api), or
   selected-profile [credential_inject](../filters/providers/credential_inject.md) for
   dynamic provider routing.
   Remove both `x-api-key` and caller `authorization` where applicable; SDK
   versions use different headers. Do not treat an internal-looking header as identity.
3. **Gateway to tools or policy services:** configure a destination-bound
   outbound chain. Never attach a blanket static secret to arbitrary
   client-selected MCP URLs. See [callout security](../architecture/outbound-callouts.md)
   and [scoped credentials](../filters/providers/callout_credentials.md).

Placeholders such as `<backend-key>` must be replaced in the proxy environment.
Setting a variable in the curl shell does not set it in an already-running proxy.
Do not commit keys or capture them in shared debug logs.

## Validation and cleanup

`--validate` must exit successfully before startup. Run one guide at a time:
they reuse port 8080. After testing, stop the proxy with Ctrl+C, stop any backend
you started for it, unset its secret variables, and delete only generated local
configuration or databases you no longer need. Keep committed example YAML intact.
The [audit](../developing/documentation-audit.md) distinguishes checks run for this change
from existing test and fixture coverage.
