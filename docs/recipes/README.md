# Operator recipes

These recipes follow Development source. Run commands from the repository root
with Git, Make, curl, and the [native build prerequisites](../developing/getting-started.md#requirements).
Each recipe selects its build explicitly. The default published image does not
supply every optional filter; build the requested binary rather than assuming
an image contains it. Choose a model that your backend actually serves.

| Goal | Complete starting point | External prerequisites |
| --- | --- | --- |
| OpenAI-compatible Chat Completions | [Quickstart](../quickstart.md#chat-completions), using its full YAML | OpenAI account, key, model access, outbound HTTPS |
| Native Anthropic Messages | [Anthropic recipe](anthropic.md#native-messages) | Native Messages backend and backend key |
| Anthropic to Chat Completions | [Anthropic recipe](anthropic.md#translated-messages) | Chat backend; translation limitations apply |
| Stable model alias and provider selection | [Routing](routing.md) | Two compatible local backends |
| Request guardrails and usage | [Guardrails and token usage](guardrails-token-usage.md) | NeMo service and OpenAI-compatible backend |
| Azure, Vertex Gemini/Anthropic, or Bedrock | [Provider adapters](providers.md) | Authorized account, model/deployment, credentials, outbound HTTPS |
| Managed Responses and tool execution | [Experimental managed path](managed-responses.md) | Inference and selected tool services; writable store |

## Credentials

Keep three boundaries separate:

1. **Caller to gateway:** authenticate callers before allowing them to use
   gateway-owned provider secrets. The local Anthropic recipes use optional
   Basic Auth. Public deployments also need TLS and a suitable ingress policy.
2. **Gateway to inference:** native forwarding can intentionally pass the
   caller's provider key ([quickstart](../quickstart.md)). A gateway-owned key
   instead needs cluster-selected core `credential_injection` as in the
   [Anthropic guide](../anthropic-messages.md#passthrough-to-anthropic-api), or
   selected-profile [credential_inject](../filters/credential_inject.md) for
   dynamic provider routing.
   Remove both `x-api-key` and caller `authorization` where applicable; SDK
   versions use different headers. Do not treat an internal-looking header as identity.
3. **Gateway to tools or policy services:** configure a destination-bound
   outbound chain. Never attach a blanket static secret to arbitrary
   client-selected MCP URLs. See [callout security](../architecture/outbound-callouts.md)
   and [scoped credentials](../filters/callout_credentials.md).

Placeholders such as `<backend-key>` must be replaced in the proxy environment.
Setting a variable in the curl shell does not set it in an already-running proxy.
Do not commit keys or capture them in shared debug logs.

## Validation and cleanup

`--validate` must exit successfully before startup. Run one recipe at a time:
they reuse port 8080. After testing, stop the proxy with Ctrl+C, stop any backend
you started for it, unset its secret variables, and delete only generated local
configuration or databases you no longer need. Keep committed example YAML intact.
The [audit](../documentation-audit.md) distinguishes checks run for this change
from existing test and fixture coverage.
